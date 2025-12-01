import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from rlx.buffers.rollout_buffer import RolloutBuffer
from rlx.utils import compute_discounted_returns


@dataclass
class REINFORCEConfig:
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
    gamma: float = 0.99
    """the discount factor gamma"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    num_steps: int = 128
    """the number of steps to take in each rollout"""


@dataclass
class REINFORCE:
    config: REINFORCEConfig
    envs: gym.vector.VectorEnv
    actor_network: nn.Module
    optimizer: optim.Optimizer
    buffer: RolloutBuffer
    step: int = 0

    def warmup(self, num_steps: int):
        pass

    def train(self, num_steps: int, callback: Optional[Callable] = None):
        obs, info = self.envs.reset()
        while self.step < num_steps:
            self.buffer.reset()
            terminated = truncated = False
            while not terminated and not truncated:
                logits = self.actor_network(mx.array(obs))
                action = mx.random.categorical(logits)

                next_obs, reward, terminated, truncated, info = self.envs.step(
                    np.array(action)
                )

                if "episode" in info and callback:
                    callback(info, self.step)

                self.buffer.add(obs, next_obs, action, reward, terminated, truncated)

                obs = next_obs

                self.step += self.config.num_envs

            def loss_fn(actor_network, observations, actions, rewards):
                logits = actor_network(observations)
                probs = nn.softmax(logits)[mx.arange(actions.shape[0]), actions]
                log_probs = mx.log(probs)
                loss = mx.sum(-log_probs * rewards)
                return loss

            discounted_returns = compute_discounted_returns(
                self.buffer.rewards, self.buffer.terminations, self.config.gamma
            )
            _, grads = nn.value_and_grad(self.actor_network, loss_fn)(
                self.actor_network,
                self.buffer.observations,
                self.buffer.actions,
                discounted_returns,
            )
            self.optimizer.update(self.actor_network, grads)
            mx.eval(self.actor_network.parameters(), self.optimizer.state)

    def evaluate(self, num_steps: int, callback: Optional[Callable] = None):
        obs, info = self.envs.reset()
        for _ in range(0, num_steps, self.config.num_envs):
            logits = self.actor_network(mx.array(obs))
            action = mx.random.categorical(logits)

            obs, *_, info = self.envs.step(np.array(action))

            if "episode" in info and callback:
                callback(info, self.step)
