from dataclasses import dataclass
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from rlx.environments.environment import Environment
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
        state = [self.network.state, self.optimizer.state]
        self.update_step = mx.compile(self.update_step, inputs=state, outputs=state)

    def warmup(self, num_steps: int):
        pass

    def train(self, num_steps: int, callback: Optional[Callable] = None):
        self.key, reset_key = mx.random.split(self.key)
        keys = mx.random.split(reset_key, self.config.num_envs)
        observation, state, _ = self.env.reset(keys)
        mx.eval(observation, *state.values())

        next_termination = mx.zeros(self.config.num_envs)
        while self.step < num_steps:
            self.buffer.reset()
            for _ in range(self.config.num_steps):
                distribution, value = self.network(observation)
                action = distribution.sample()

                self.key, step_key = mx.random.split(self.key)
                keys = mx.random.split(step_key, self.config.num_envs)
                next_observation, state, reward, terminated, truncated, info = (
                    self.env.step(keys, state, action)
                )
                mx.eval(
                    next_observation, reward, terminated, truncated, *state.values()
                )

                if callback is not None and "episode" in info:
                    callback(info, self.step)

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

    def loss_fn(self, network, observations, actions, advantages, returns):
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

    def update_step(self, observations, actions, advantages, returns):
        loss, grads = nn.value_and_grad(self.network, self.loss_fn)(
            self.network, observations, actions, advantages, returns
        )
        grads, _ = optim.clip_grad_norm(grads, self.config.max_grad_norm)
        self.optimizer.update(self.network, grads)
        return loss

    def update(self, advantages: mx.array, returns: mx.array):
        observations = flatten(self.buffer.observations)
        actions = flatten(self.buffer.actions)
        advantages = flatten(advantages)
        returns = flatten(returns)

        self.update_step(observations, actions, advantages, returns)
        mx.eval(self.network.state, self.optimizer.state)

    def evaluate(self, num_steps: int, callback: Optional[Callable] = None):
        self.key, reset_key = mx.random.split(self.key)
        keys = mx.random.split(reset_key, self.config.num_envs)
        observation, state, _ = self.env.reset(keys)
        mx.eval(observation, *state.values())

        for _ in range(0, num_steps, self.config.num_envs):
            distribution, _ = self.network(observation)
            action = distribution.sample()

            self.key, step_key = mx.random.split(self.key)
            keys = mx.random.split(step_key, self.config.num_envs)
            observation, state, reward, terminated, truncated, info = (
                self.env.step(keys, state, action)
            )
            mx.eval(observation, reward, terminated, truncated, *state.values())

            if callback is not None and "episode" in info:
                callback(info, self.step)
