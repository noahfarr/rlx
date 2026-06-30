from dataclasses import dataclass
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map


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
        action_dim = int(np.prod(self.envs.single_action_space.shape))
        self.target_entropy = -float(action_dim)
        self.log_alpha = mx.array(float(np.log(self.config.alpha)))

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

        next_distribution = self.actor_network(data.next_observations)
        next_action, next_log_probability = next_distribution.sample()
        target_q1, target_q2 = self.target_critic_network(
            data.next_observations, next_action
        )
        min_target_q = mx.minimum(target_q1, target_q2) - alpha * next_log_probability
        td_target = data.rewards + self.config.gamma * (1 - data.terminations) * min_target_q

        def critic_loss_fn(critic_network):
            q1, q2 = critic_network(data.observations, data.actions)
            return nn.losses.mse_loss(q1, td_target) + nn.losses.mse_loss(q2, td_target)

        _, critic_grads = nn.value_and_grad(self.critic_network, critic_loss_fn)(
            self.critic_network
        )
        self.critic_optimizer.update(self.critic_network, critic_grads)
        mx.eval(self.critic_network.parameters(), self.critic_optimizer.state)

        if self.step % self.config.policy_frequency == 0:

            def actor_loss_fn(actor_network):
                distribution = actor_network(data.observations)
                action, log_probability = distribution.sample()
                q1, q2 = self.critic_network(data.observations, action)
                min_q = mx.minimum(q1, q2)
                return mx.mean(alpha * log_probability - min_q)

            _, actor_grads = nn.value_and_grad(self.actor_network, actor_loss_fn)(
                self.actor_network
            )
            self.actor_optimizer.update(self.actor_network, actor_grads)
            mx.eval(self.actor_network.parameters(), self.actor_optimizer.state)

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
            target_params = tree_map(
                lambda online, target: self.config.tau * online
                + (1 - self.config.tau) * target,
                self.critic_network.parameters(),
                self.target_critic_network.parameters(),
            )
            self.target_critic_network.update(target_params)

    def evaluate(self, num_steps: int, callback: Optional[Callable] = None):
        observation, info = self.envs.reset()
        for _ in range(0, num_steps, self.config.num_envs):
            distribution = self.actor_network(mx.array(observation))
            action = distribution.mode()

            observation, *_, info = self.envs.step(np.array(action))

            if "episode" in info and callback:
                callback(info, self.step)
