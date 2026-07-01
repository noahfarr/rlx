from dataclasses import dataclass
from typing import Any, Callable, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from rlx.environments.environment import Environment
from rlx.utils import soft_update


@dataclass
class TD3Config:
    num_envs: int = 1
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    policy_frequency: int = 2
    exploration_noise: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5


@dataclass
class TD3:
    config: TD3Config
    env: Environment
    actor_network: nn.Module
    target_actor_network: nn.Module
    critic_network: nn.Module
    target_critic_network: nn.Module
    actor_optimizer: optim.Optimizer
    critic_optimizer: optim.Optimizer
    key: mx.array
    buffer: Any = None
    step: int = 0

    def __post_init__(self):
        self.action_low = mx.array(self.env.action_space.low)
        self.action_high = mx.array(self.env.action_space.high)

        self.update_critic = mx.compile(
            self.update_critic,
            inputs=[
                self.critic_network.state,
                self.critic_optimizer.state,
                self.target_actor_network.state,
                self.target_critic_network.state,
                mx.random.state,
            ],
            outputs=[
                self.critic_network.state,
                self.critic_optimizer.state,
                mx.random.state,
            ],
        )
        self.update_actor = mx.compile(
            self.update_actor,
            inputs=[
                self.actor_network.state,
                self.actor_optimizer.state,
                self.critic_network.state,
            ],
            outputs=[self.actor_network.state, self.actor_optimizer.state],
        )

    def critic_loss_fn(self, critic_network, observations, actions, td_target):
        q = critic_network(observations, actions)
        return mx.sum(mx.mean(mx.square(q - td_target), axis=(1, 2)))

    def update_critic(
        self, observations, actions, next_observations, rewards, terminations
    ):
        clipped_noise = (
            mx.clip(
                mx.random.normal(actions.shape) * self.config.policy_noise,
                -self.config.noise_clip,
                self.config.noise_clip,
            )
            * self.target_actor_network.action_scale
        )
        next_action = mx.clip(
            self.target_actor_network(next_observations) + clipped_noise,
            self.action_low,
            self.action_high,
        )
        target_q = self.target_critic_network(next_observations, next_action)
        min_target_q = mx.min(target_q, axis=0)
        td_target = rewards + self.config.gamma * (1 - terminations) * min_target_q

        loss, grads = nn.value_and_grad(self.critic_network, self.critic_loss_fn)(
            self.critic_network, observations, actions, td_target
        )
        self.critic_optimizer.update(self.critic_network, grads)
        return loss

    def actor_loss_fn(self, actor_network, observations):
        action = actor_network(observations)
        q = self.critic_network(observations, action)
        return -mx.mean(q[0])

    def update_actor(self, observations):
        loss, grads = nn.value_and_grad(self.actor_network, self.actor_loss_fn)(
            self.actor_network, observations
        )
        self.actor_optimizer.update(self.actor_network, grads)
        return loss

    def random_action(self) -> mx.array:
        self.key, action_key = mx.random.split(self.key)
        shape = (self.config.num_envs, *self.env.action_space.shape)
        return (
            mx.random.uniform(shape=shape, key=action_key)
            * (self.action_high - self.action_low)
            + self.action_low
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
            action = self.actor_network(observation)
            noise = (
                mx.random.normal(action.shape)
                * self.actor_network.action_scale
                * self.config.exploration_noise
            )
            action = mx.clip(action + noise, self.action_low, self.action_high)

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

            self.learn()

    def learn(self):
        data = self.buffer.sample(self.config.batch_size)

        self.update_critic(
            data.observations,
            data.actions,
            data.next_observations,
            data.rewards,
            data.terminations,
        )
        mx.eval(self.critic_network.state, self.critic_optimizer.state)

        if self.step % self.config.policy_frequency == 0:
            self.update_actor(data.observations)
            mx.eval(self.actor_network.state, self.actor_optimizer.state)

            soft_update(
                self.target_actor_network, self.actor_network, self.config.tau
            )
            soft_update(
                self.target_critic_network, self.critic_network, self.config.tau
            )

    def evaluate(self, num_steps: int, callback: Optional[Callable] = None):
        self.key, reset_key = mx.random.split(self.key)
        keys = mx.random.split(reset_key, self.config.num_envs)
        observation, state, _ = self.env.reset(keys)
        mx.eval(observation, *state.values())

        for _ in range(0, num_steps, self.config.num_envs):
            action = self.actor_network(observation)

            self.key, step_key = mx.random.split(self.key)
            keys = mx.random.split(step_key, self.config.num_envs)
            observation, state, reward, terminated, truncated, info = (
                self.env.step(keys, state, action)
            )
            mx.eval(observation, reward, terminated, truncated, *state.values())

            if callback is not None and "episode" in info:
                callback(info, self.step)
