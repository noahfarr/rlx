import random
from dataclasses import dataclass
from typing import Any, Callable, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

from rlx.environments.environment import Environment


@dataclass
class DQNConfig:
    num_envs: int = 1
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 500
    batch_size: int = 128
    train_frequency: int = 10


@dataclass
class DQN:
    config: DQNConfig
    env: Environment
    q_network: nn.Module
    target_network: nn.Module
    optimizer: optim.Optimizer
    buffer: Any
    epsilon_schedule: Callable
    key: mx.array
    step: int = 0

    def __post_init__(self):
        state = [self.q_network.state, self.optimizer.state]
        self.update_step = mx.compile(self.update_step, inputs=state, outputs=state)

    def loss_fn(self, q_network, observations, actions, td_target):
        q_values = q_network(observations)
        q_value = mx.take_along_axis(q_values, actions, axis=-1).squeeze()
        return nn.losses.mse_loss(td_target, q_value)

    def update_step(self, observations, actions, td_target):
        loss, grads = nn.value_and_grad(self.q_network, self.loss_fn)(
            self.q_network, observations, actions, td_target
        )
        self.optimizer.update(self.q_network, grads)
        return loss

    def random_action(self) -> mx.array:
        self.key, action_key = mx.random.split(self.key)
        return mx.random.randint(
            0, self.env.action_space.n, shape=(self.config.num_envs,), key=action_key
        )

    def warmup(self, num_steps: int):
        self.key, reset_key = mx.random.split(self.key)
        keys = mx.random.split(reset_key, self.config.num_envs)
        observation, state, _ = self.env.reset(keys)
        mx.eval(observation, *state.values())

        for _ in range(0, num_steps, self.config.num_envs):
            action = self.random_action()

            self.key, step_key = mx.random.split(self.key)
            keys = mx.random.split(step_key, self.config.num_envs)
            next_observation, state, reward, terminated, truncated, info = (
                self.env.step(keys, state, action)
            )
            mx.eval(next_observation, reward, terminated, truncated, *state.values())

            self.buffer.add(
                observation, next_observation, action, reward, terminated, truncated
            )

            observation = next_observation

    def train(self, num_steps: int, callback: Optional[Callable] = None):
        self.key, reset_key = mx.random.split(self.key)
        keys = mx.random.split(reset_key, self.config.num_envs)
        observation, state, _ = self.env.reset(keys)
        mx.eval(observation, *state.values())

        for _ in range(0, num_steps, self.config.num_envs):
            if random.random() < self.epsilon_schedule(t=self.step):
                action = self.random_action()
            else:
                q_values = self.q_network(observation)
                action = mx.argmax(q_values, axis=-1)

            self.key, step_key = mx.random.split(self.key)
            keys = mx.random.split(step_key, self.config.num_envs)
            next_observation, state, reward, terminated, truncated, info = (
                self.env.step(keys, state, action)
            )
            mx.eval(next_observation, reward, terminated, truncated, *state.values())

            if callback is not None and "episode" in info:
                callback(info, self.step)

            self.buffer.add(
                observation, next_observation, action, reward, terminated, truncated
            )

            observation = next_observation

            self.step += self.config.num_envs

            if self.step % self.config.train_frequency == 0:
                data = self.buffer.sample(self.config.batch_size)
                target_q_values = self.target_network(data.next_observations)
                target_q_value = mx.max(target_q_values, axis=1)
                td_target = (
                    data.rewards.flatten()
                    + self.config.gamma
                    * target_q_value
                    * (1 - data.terminations.flatten())
                )

                actions = data.actions.reshape(-1, 1)

                self.update_step(data.observations, actions, td_target)
                mx.eval(self.q_network.state, self.optimizer.state)

                if self.step % self.config.target_network_frequency == 0:
                    target_params = tree_map(
                        lambda q, t: self.config.tau * q
                        + (1 - self.config.tau) * t,
                        self.q_network.parameters(),
                        self.target_network.parameters(),
                    )
                    self.target_network.update(target_params)

    def evaluate(self, num_steps: int, callback: Optional[Callable] = None):
        self.key, reset_key = mx.random.split(self.key)
        keys = mx.random.split(reset_key, self.config.num_envs)
        observation, state, _ = self.env.reset(keys)
        mx.eval(observation, *state.values())

        for _ in range(0, num_steps, self.config.num_envs):
            q_values = self.q_network(observation)
            action = mx.argmax(q_values, axis=-1)

            self.key, step_key = mx.random.split(self.key)
            keys = mx.random.split(step_key, self.config.num_envs)
            observation, state, reward, terminated, truncated, info = (
                self.env.step(keys, state, action)
            )
            mx.eval(observation, reward, terminated, truncated, *state.values())

            if callback is not None and "episode" in info:
                callback(info, self.step)
