import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


@dataclass
class DQNConfig:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "rlx"
    """the wandb's project name"""
    wandb_entity: str = ""
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""


@dataclass
class DQN:
    config: DQNConfig
    envs: gym.vector.VectorEnv
    q_network: nn.Module
    target_network: nn.Module
    optimizer: optim.Optimizer
    buffer: Any
    epsilon_schedule: Callable
    step: int = 0

    def warmup(self, num_steps: int):
        obs, info = self.envs.reset()
        for _ in range(0, num_steps, self.config.num_envs):
            action = self.envs.action_space.sample()
            next_obs, reward, terminated, truncated, info = self.envs.step(
                np.array(action)
            )

            self.buffer.add(obs, next_obs, action, reward, terminated, truncated)

            obs = next_obs

    def train(self, num_steps: int, callback: Optional[Callable] = None):
        obs, info = self.envs.reset()
        for _ in range(0, num_steps, self.config.num_envs):
            if random.random() < self.epsilon_schedule(t=self.step):
                action = self.envs.action_space.sample()
            else:
                q_values = self.q_network(mx.array(obs))
                action = mx.argmax(q_values, axis=-1)

            next_obs, reward, terminated, truncated, info = self.envs.step(
                np.array(action)
            )

            if "episode" in info and callback:
                callback(info, self.step)

            self.buffer.add(obs, next_obs, action, reward, terminated, truncated)

            obs = next_obs

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

                def loss_fn(q_network):
                    q_values = q_network(data.observations)
                    q_value = mx.take_along_axis(q_values, actions, axis=-1).squeeze()
                    loss = nn.losses.mse_loss(td_target, q_value)
                    return loss

                _, grads = nn.value_and_grad(self.q_network, loss_fn)(self.q_network)
                self.optimizer.update(self.q_network, grads)
                mx.eval(self.q_network.parameters(), self.optimizer.state)

                if self.step % self.config.target_network_frequency == 0:
                    self.target_network.update(self.q_network.parameters())

    def evaluate(self, num_steps: int, callback: Optional[Callable] = None):
        obs, info = self.envs.reset()
        for _ in range(0, num_steps, self.config.num_envs):
            q_values = self.q_network(mx.array(obs))
            action = mx.argmax(q_values, axis=-1)

            obs, *_, info = self.envs.step(np.array(action))

            if "episode" in info and callback:
                callback(info, self.step)
