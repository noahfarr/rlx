from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from rlx.environments.environment import Environment, EnvState
from rlx.buffers.rollout_buffer import RolloutBuffer
from rlx.utils import compute_generalized_advantage_estimate, flatten


@dataclass
class A2CConfig:
    num_envs: int = 4096
    num_steps: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantages: bool = True
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class A2C:
    config: A2CConfig
    env: Environment
    network: nn.Module
    optimizer: optim.Optimizer
    buffer: RolloutBuffer
    key: mx.array
    step: int = 0

    def __post_init__(self):
        self._reset = mx.vmap(self.env.reset)
        self._step = mx.vmap(self.env.step)

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

    def update(self, advantages: mx.array, returns: mx.array):
        observations = flatten(self.buffer.observations)
        actions = flatten(self.buffer.actions)
        advantages = flatten(advantages)
        returns = flatten(returns)

        def loss_fn(network, observations, actions, advantages, returns):
            distribution, values = network(observations)
            log_probabilities = distribution.log_prob(actions)
            entropy = distribution.entropy()

            if self.config.normalize_advantages:
                advantages = (advantages - mx.mean(advantages)) / (
                    mx.std(advantages) + 1e-8
                )

            policy_loss = -mx.mean(advantages * log_probabilities)
            value_loss = 0.5 * mx.mean(mx.square(values.squeeze(-1) - returns))
            entropy_loss = mx.mean(entropy)

            return (
                policy_loss
                - self.config.entropy_coefficient * entropy_loss
                + self.config.value_coefficient * value_loss
            )

        _, grads = nn.value_and_grad(self.network, loss_fn)(
            self.network,
            observations,
            actions,
            advantages,
            returns,
        )
        grads, _ = optim.clip_grad_norm(grads, self.config.max_grad_norm)
        self.optimizer.update(self.network, grads)
        mx.eval(self.network.parameters(), self.optimizer.state)

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
