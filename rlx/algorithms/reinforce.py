from dataclasses import dataclass
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from rlx.environments.environment import Environment
from rlx.buffers.rollout_buffer import RolloutBuffer
from rlx.utils import (
    compute_completed_episode_mask,
    compute_discounted_returns,
    flatten,
)


@dataclass
class REINFORCEConfig:
    num_envs: int = 4096
    num_steps: int = 128
    gamma: float = 0.99


@dataclass
class REINFORCE:
    config: REINFORCEConfig
    env: Environment
    actor_network: nn.Module
    optimizer: optim.Optimizer
    buffer: RolloutBuffer
    key: mx.array
    step: int = 0

    def __post_init__(self):
        assert (
            self.buffer.buffer_size >= self.config.num_steps
        ), "buffer_size must be at least num_steps to store a full rollout"
        state = [self.actor_network.state, self.optimizer.state]
        self.update_step = mx.compile(self.update_step, inputs=state, outputs=state)

    def loss_fn(self, actor_network, observations, actions, returns, mask):
        logits = actor_network(observations)
        log_probs = nn.log_softmax(logits, axis=-1)
        log_probs = mx.take_along_axis(
            log_probs, actions[..., None], axis=-1
        ).squeeze(-1)
        return -mx.sum(mask * log_probs * returns) / mx.maximum(mx.sum(mask), 1.0)

    def update_step(self, observations, actions, returns, mask):
        loss, grads = nn.value_and_grad(self.actor_network, self.loss_fn)(
            self.actor_network, observations, actions, returns, mask
        )
        self.optimizer.update(self.actor_network, grads)
        return loss

    def warmup(self, num_steps: int):
        pass

    def train(self, num_steps: int, callback: Optional[Callable] = None):
        self.key, reset_key = mx.random.split(self.key)
        keys = mx.random.split(reset_key, self.config.num_envs)
        observation, state, _ = self.env.reset(keys)
        mx.eval(observation, *state.values())

        while self.step < num_steps:
            self.buffer.reset()
            for _ in range(self.config.num_steps):
                logits = self.actor_network(observation)
                action = mx.random.categorical(logits)

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
                    observation, next_observation, action, reward, terminated, truncated
                )

                observation = next_observation

                self.step += self.config.num_envs

            dones = mx.maximum(self.buffer.terminations, self.buffer.truncations)
            discounted_returns = compute_discounted_returns(
                self.buffer.rewards,
                dones,
                self.config.gamma,
            )
            mask = compute_completed_episode_mask(dones)

            observations = flatten(self.buffer.observations)
            actions = flatten(self.buffer.actions)
            returns = flatten(discounted_returns)
            mask = flatten(mask)

            self.update_step(observations, actions, returns, mask)
            mx.eval(self.actor_network.state, self.optimizer.state)

    def evaluate(self, num_steps: int, callback: Optional[Callable] = None):
        self.key, reset_key = mx.random.split(self.key)
        keys = mx.random.split(reset_key, self.config.num_envs)
        observation, state, _ = self.env.reset(keys)
        mx.eval(observation, *state.values())

        for _ in range(0, num_steps, self.config.num_envs):
            logits = self.actor_network(observation)
            action = mx.random.categorical(logits)

            self.key, step_key = mx.random.split(self.key)
            keys = mx.random.split(step_key, self.config.num_envs)
            observation, state, reward, terminated, truncated, info = (
                self.env.step(keys, state, action)
            )
            mx.eval(observation, reward, terminated, truncated, *state.values())

            if callback is not None and "episode" in info:
                callback(info, self.step)
