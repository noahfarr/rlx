import os
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter



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

@dataclass(frozen=True)
class DQN:
    config: DQNConfig
    envs: gym.vector.VectorEnv
    q_network: nn.Module
    target_network: nn.Module
    optimizer: optim.Optimizer
    buffer: ReplayBuffer
    epsilon_schedule: Callable
    step: int = 0

    def _greedy_policy(self, obs):
        q_values = self.q_network(mx.array(obs))
        action = mx.argmax(q_values, axis=-1)
        return action

    def _random_policy(self, obs):
        return self.envs.action_space.sample()

    def _epsilon_greedy_policy(self, obs):
        if random.random() < self.epsilon_schedule(t=self.step):
            return self._random_policy(obs)
        else:
            return self._greedy_policy(obs)

    def _rollout(self, obs, *, policy: Callable, num_steps: Optional[int] = None, num_episodes: Optional[int] = None):
        if num_steps is not None:
            for _ in range(num_steps):
                action = policy(obs)
                action = np.array(action)
                next_obs, reward, terminated, truncated, info = self.envs.step(action)

                real_next_obs = next_obs.copy()
                for idx, trunc in enumerate(truncated):
                    if trunc:
                        real_next_obs[idx] = info["final_observation"][idx]

                self.buffer.add(obs, real_next_obs, action, reward, terminated, info)

                obs = next_obs
        elif num_episodes is not None:
            episode_idx = 0
            while episode_idx < num_episodes:
                action = policy(obs)
                action = np.array(action)
                next_obs, reward, terminated, truncated, info = self.envs.step(action)

                if terminated or truncated:
                    episode_idx += 1

                real_next_obs = next_obs.copy()
                for idx, trunc in enumerate(truncated):
                    if trunc:
                        real_next_obs[idx] = infos["final_observation"][idx]

                self.buffer.add(obs, real_next_obs, action, reward, terminated, infos)

                obs = next_obs
        return obs

    def _update(self, data):
        target_q_values = self.target_network(mx.array(data.next_observations.cpu().numpy()))
        target_q_value = mx.max(target_q_values, axis=1)
        td_target = mx.array(data.rewards.cpu().numpy()) + self.config.gamma * target_q_value * (1 - mx.array(data.dones.flatten().cpu().numpy()))

        observations = mx.array(data.observations.cpu().numpy())
        actions = mx.array(data.actions.cpu().numpy()).reshape(-1, 1)

        def loss_fn(q_network):
            q_values = q_network(observations)
            q_value = mx.take_along_axis(q_values, actions, axis=1).squeeze()
            loss = nn.losses.mse_loss(td_target, q_value)
            return loss

        _, grads = nn.value_and_grad(self.q_network, loss_fn)(self.q_network)
        self.optimizer.update(self.q_network, grads)
        mx.eval(self.q_network.parameters(), self.optimizer.state)

        if self.step % self.config.target_network_frequency == 0:
            self.target_network.update(self.q_network.parameters())

    def _learn(self, obs, num_steps):
        obs = self._rollout(obs, num_steps=num_steps, policy=self._epsilon_greedy_policy)
        data = self.buffer.sample(self.config.batch_size)
        self._update(data)
        return obs

    def warmup(self, num_steps: int):
        obs, info = self.envs.reset()
        num_rollout_steps = num_steps // self.config.num_envs
        self._rollout(obs, num_steps=num_rollout_steps, policy=self._random_policy)

    def train(self, num_steps: int):
        obs, info = self.envs.reset()
        num_learn_steps = num_steps // (self.config.train_frequency * self.config.num_envs)
        self._learn(obs, num_learn_steps)

    def evaluate(self, num_episodes: int):
        obs, info = self.envs.reset()
        num_rollout_episodes = num_episodes // self.config.num_envs
        self._rollout(obs, num_episodes=num_rollout_episodes, policy=self._greedy_policy)


