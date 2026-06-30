from dataclasses import dataclass
from typing import Any, Callable, Optional

import gymnasium as gym
from gymnasium.vector import AutoresetMode
import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from rlx.utils import soft_update


@dataclass
class SACConfig:
    num_envs: int = 1
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True


@dataclass
class SAC:
    config: SACConfig
    envs: gym.vector.VectorEnv
    actor_network: nn.Module
    critic_network: nn.Module
    target_critic_network: nn.Module
    actor_optimizer: optim.Optimizer
    critic_optimizer: optim.Optimizer
    alpha_optimizer: Optional[optim.Optimizer] = None
    buffer: Any = None
    step: int = 0

    def __post_init__(self):
        assert (
            self.envs.metadata.get("autoreset_mode") == AutoresetMode.NEXT_STEP
        ), "SAC assumes NEXT_STEP autoreset so the stored next_observation is the true final observation"

        action_dim = int(np.prod(self.envs.single_action_space.shape))
        self.target_entropy = -float(action_dim)
        self.log_alpha = mx.array(float(np.log(self.config.alpha)))

        self.update_critic = mx.compile(
            self.update_critic,
            inputs=[
                self.critic_network.state,
                self.critic_optimizer.state,
                self.actor_network.state,
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
                mx.random.state,
            ],
            outputs=[
                self.actor_network.state,
                self.actor_optimizer.state,
                mx.random.state,
            ],
        )

    def critic_loss_fn(self, critic_network, observations, actions, td_target):
        q = critic_network(observations, actions)
        return mx.sum(mx.mean(mx.square(q - td_target), axis=(1, 2)))

    def update_critic(
        self, observations, actions, next_observations, rewards, terminations, alpha
    ):
        next_distribution = self.actor_network(next_observations)
        next_action, next_log_probability = next_distribution.sample()
        target_q = self.target_critic_network(next_observations, next_action)
        min_target_q = mx.min(target_q, axis=0) - alpha * next_log_probability
        td_target = rewards + self.config.gamma * (1 - terminations) * min_target_q

        loss, grads = nn.value_and_grad(self.critic_network, self.critic_loss_fn)(
            self.critic_network, observations, actions, td_target
        )
        self.critic_optimizer.update(self.critic_network, grads)
        return loss

    def actor_loss_fn(self, actor_network, observations, alpha):
        distribution = actor_network(observations)
        action, log_probability = distribution.sample()
        q = self.critic_network(observations, action)
        min_q = mx.min(q, axis=0)
        return mx.mean(alpha * log_probability - min_q)

    def update_actor(self, observations, alpha):
        loss, grads = nn.value_and_grad(self.actor_network, self.actor_loss_fn)(
            self.actor_network, observations, alpha
        )
        self.actor_optimizer.update(self.actor_network, grads)
        return loss

    @property
    def alpha(self) -> mx.array:
        if self.config.autotune:
            return mx.exp(self.log_alpha)
        return mx.array(self.config.alpha)

    def warmup(self, num_steps: int):
        observation, info = self.envs.reset()
        for _ in range(0, num_steps, self.config.num_envs):
            action = self.envs.action_space.sample()
            next_observation, reward, terminated, truncated, info = self.envs.step(
                np.array(action)
            )

            self.buffer.add(observation, next_observation, action, reward, terminated, truncated)

            observation = next_observation

    def train(self, num_steps: int, callback: Optional[Callable] = None):
        observation, info = self.envs.reset()
        for _ in range(0, num_steps, self.config.num_envs):
            distribution = self.actor_network(mx.array(observation))
            action, _ = distribution.sample()

            next_observation, reward, terminated, truncated, info = self.envs.step(
                np.array(action)
            )

            if "episode" in info and callback:
                callback(info, self.step)

            self.buffer.add(observation, next_observation, action, reward, terminated, truncated)

            observation = next_observation

            self.step += self.config.num_envs

            self.learn()

    def learn(self):
        data = self.buffer.sample(self.config.batch_size)
        alpha = self.alpha

        self.update_critic(
            data.observations,
            data.actions,
            data.next_observations,
            data.rewards,
            data.terminations,
            alpha,
        )
        mx.eval(self.critic_network.state, self.critic_optimizer.state)

        if self.step % self.config.policy_frequency == 0:
            self.update_actor(data.observations, alpha)
            mx.eval(self.actor_network.state, self.actor_optimizer.state)

            if self.config.autotune:
                distribution = self.actor_network(data.observations)
                _, log_probability = distribution.sample()

                def alpha_loss_fn(log_alpha):
                    return mx.mean(
                        -mx.exp(log_alpha) * (log_probability + self.target_entropy)
                    )

                _, alpha_grad = mx.value_and_grad(alpha_loss_fn)(self.log_alpha)
                updated = self.alpha_optimizer.apply_gradients(
                    {"log_alpha": alpha_grad}, {"log_alpha": self.log_alpha}
                )
                self.log_alpha = updated["log_alpha"]
                mx.eval(self.log_alpha, self.alpha_optimizer.state)

        if self.step % self.config.target_network_frequency == 0:
            soft_update(
                self.target_critic_network, self.critic_network, self.config.tau
            )

    def evaluate(self, num_steps: int, callback: Optional[Callable] = None):
        observation, info = self.envs.reset()
        for _ in range(0, num_steps, self.config.num_envs):
            distribution = self.actor_network(mx.array(observation))
            action = distribution.mode()

            observation, *_, info = self.envs.step(np.array(action))

            if "episode" in info and callback:
                callback(info, self.step)
