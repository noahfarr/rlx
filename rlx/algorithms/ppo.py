from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from rlx.environments.environment import Environment, EnvState
from rlx.buffers.rollout_buffer import RolloutBuffer
from rlx.utils import compute_generalized_advantage_estimate, flatten


@dataclass
class PPOConfig:
    num_envs: int = 4096
    num_steps: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 16
    update_epochs: int = 2
    normalize_advantages: bool = True
    clip_coefficient: float = 0.2
    clip_value_loss: bool = True
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class PPO:
    config: PPOConfig
    env: Environment
    network: nn.Module
    optimizer: optim.Optimizer
    buffer: RolloutBuffer
    key: mx.array
    step: int = 0

    def __post_init__(self):
        self._reset = mx.vmap(self.env.reset)
        self._step = mx.vmap(self.env.step)
        self._update_minibatch = self._build_update_minibatch()

    def reset(self) -> tuple[mx.array, EnvState]:
        self.key, reset_key = mx.random.split(self.key)
        keys = mx.random.split(reset_key, self.config.num_envs)
        observation, state = self._reset(keys)
        mx.eval(observation, *state.values())
        return observation, state

    def environment_step(self, state: EnvState, action: mx.array):
        self.key, step_key = mx.random.split(self.key)
        keys = mx.random.split(step_key, self.config.num_envs)
        observation, state, reward, terminated, truncated, info = self._step(
            keys, state, action
        )
        mx.eval(observation, reward, terminated, truncated, *state.values())
        return observation, state, reward, terminated, truncated, info

    def report(self, callback, episode_returns, episode_lengths, terminated, truncated):
        done = mx.logical_or(terminated, truncated)
        done_np = np.asarray(done)
        if callback is not None and done_np.any():
            callback(
                {
                    "episode": {
                        "r": np.asarray(episode_returns),
                        "l": np.asarray(episode_lengths),
                    },
                    "_episode": done_np,
                },
                self.step,
            )
        episode_returns = mx.where(done, 0.0, episode_returns)
        episode_lengths = mx.where(done, 0.0, episode_lengths)
        return episode_returns, episode_lengths

    def warmup(self, num_steps: int):
        pass

    def train(self, num_steps: int, callback: Optional[Callable] = None):
        observation, state = self.reset()
        episode_returns = mx.zeros((self.config.num_envs,))
        episode_lengths = mx.zeros((self.config.num_envs,))
        next_termination = mx.zeros(self.config.num_envs)
        while self.step < num_steps:
            self.buffer.reset()
            for _ in range(self.config.num_steps):
                distribution, value = self.network(observation)
                action = distribution.sample()
                log_probability = distribution.log_prob(action)

                next_observation, state, reward, terminated, truncated, info = (
                    self.environment_step(state, action)
                )

                episode_returns = episode_returns + reward
                episode_lengths = episode_lengths + 1
                episode_returns, episode_lengths = self.report(
                    callback, episode_returns, episode_lengths, terminated, truncated
                )

                self.buffer.add(
                    observation,
                    next_observation,
                    action,
                    reward,
                    terminated,
                    truncated,
                    value=value,
                    log_prob=log_probability,
                )

                observation = next_observation
                next_termination = terminated.astype(mx.float32)
                self.step += self.config.num_envs

            _, last_value = self.network(observation)
            advantages = compute_generalized_advantage_estimate(
                self.buffer.rewards,
                self.buffer.values,
                self.buffer.terminations,
                last_value.squeeze(-1),
                next_termination,
                self.config.gamma,
                self.config.gae_lambda,
            )
            returns = advantages + self.buffer.values

            self.update(advantages, returns)

    def _build_update_minibatch(self):
        def loss_fn(
            network,
            observations,
            actions,
            old_log_probabilities,
            old_values,
            advantages,
            returns,
        ):
            distribution, new_values = network(observations)
            new_log_probabilities = distribution.log_prob(actions)
            entropy = distribution.entropy()

            log_ratio = new_log_probabilities - old_log_probabilities
            ratio = mx.exp(log_ratio)

            if self.config.normalize_advantages:
                advantages = (advantages - mx.mean(advantages)) / (
                    mx.std(advantages) + 1e-8
                )

            policy_loss = mx.maximum(
                -advantages * ratio,
                -advantages
                * mx.clip(
                    ratio,
                    1 - self.config.clip_coefficient,
                    1 + self.config.clip_coefficient,
                ),
            )
            policy_loss = mx.mean(policy_loss)

            new_values = new_values.squeeze(-1)
            if self.config.clip_value_loss:
                value_loss_unclipped = mx.square(new_values - returns)
                clipped_values = old_values + mx.clip(
                    new_values - old_values,
                    -self.config.clip_coefficient,
                    self.config.clip_coefficient,
                )
                value_loss_clipped = mx.square(clipped_values - returns)
                value_loss = 0.5 * mx.mean(
                    mx.maximum(value_loss_unclipped, value_loss_clipped)
                )
            else:
                value_loss = 0.5 * mx.mean(mx.square(new_values - returns))

            entropy_loss = mx.mean(entropy)

            return (
                policy_loss
                - self.config.entropy_coefficient * entropy_loss
                + self.config.value_coefficient * value_loss
            )

        loss_and_grad = nn.value_and_grad(self.network, loss_fn)
        state = [self.network.state, self.optimizer.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def update_minibatch(
            observations,
            actions,
            old_log_probabilities,
            old_values,
            advantages,
            returns,
        ):
            loss, grads = loss_and_grad(
                self.network,
                observations,
                actions,
                old_log_probabilities,
                old_values,
                advantages,
                returns,
            )
            grads, _ = optim.clip_grad_norm(grads, self.config.max_grad_norm)
            self.optimizer.update(self.network, grads)
            return loss

        return update_minibatch

    def update(self, advantages: mx.array, returns: mx.array):
        observations = flatten(self.buffer.observations)
        actions = flatten(self.buffer.actions)
        log_probabilities = flatten(self.buffer.log_probs)
        values = flatten(self.buffer.values)
        advantages = flatten(advantages)
        returns = flatten(returns)

        batch_size = self.config.num_steps * self.config.num_envs
        minibatch_size = batch_size // self.config.num_minibatches

        for _ in range(self.config.update_epochs):
            permutation = np.random.permutation(batch_size)
            for start in range(0, batch_size, minibatch_size):
                minibatch_indices = mx.array(
                    permutation[start : start + minibatch_size]
                )
                self._update_minibatch(
                    observations[minibatch_indices],
                    actions[minibatch_indices],
                    log_probabilities[minibatch_indices],
                    values[minibatch_indices],
                    advantages[minibatch_indices],
                    returns[minibatch_indices],
                )
            mx.eval(self.network.state, self.optimizer.state)

    def evaluate(self, num_steps: int, callback: Optional[Callable] = None):
        observation, state = self.reset()
        episode_returns = mx.zeros((self.config.num_envs,))
        episode_lengths = mx.zeros((self.config.num_envs,))
        for _ in range(0, num_steps, self.config.num_envs):
            distribution, _ = self.network(observation)
            action = distribution.sample()

            observation, state, reward, terminated, truncated, info = (
                self.environment_step(state, action)
            )

            episode_returns = episode_returns + reward
            episode_lengths = episode_lengths + 1
            episode_returns, episode_lengths = self.report(
                callback, episode_returns, episode_lengths, terminated, truncated
            )
