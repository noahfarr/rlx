from functools import partial
import os
import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import gymnasium as gym
import numpy as np
import tyro
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.buffers import ReplayBuffer

from rlx.dqn import DQNConfig, DQN

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk

def linear_schedule(start_e: float, end_e: float, duration: float, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

class QNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, envs.single_action_space.n),
        )

    def __call__(self, x):
        return self.network(x)

if __name__ == "__main__":
    config = tyro.cli(DQNConfig)
    run_name = f"{config.env_id}__{config.exp_name}__{config.seed}__{int(time.time())}"
    if config.track:
        import wandb

        wandb.init(
            project=config.wandb_project_name,
            entity=config.wandb_entity,
            sync_tensorboard=True,
            config=vars(config),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    mx.random.seed(config.seed)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(config.env_id, config.seed + i, i, config.capture_video, run_name)
            for i in range(config.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    q_network = QNetwork(envs)
    mx.eval(q_network.parameters())
    optimizer = optim.Adam(learning_rate=config.learning_rate)
    target_network = QNetwork(envs).update(q_network.parameters())



    # TODO: Implement custom replay buffer to handle mlx arrays
    rb = ReplayBuffer(
        config.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        n_envs=config.num_envs,
        handle_timeout_termination=False,
    )

    epsilon_schedule = partial(linear_schedule, start_e=config.start_e, end_e=config.end_e, duration=config.exploration_fraction * config.total_timesteps)
    algorithm = DQN(config=config, envs=envs, q_network=q_network, target_network=target_network, optimizer=optimizer, buffer=rb, epsilon_schedule=epsilon_schedule)

    algorithm.warmup(10_000)
    algorithm.train(100_000)

    envs.close()
    writer.close()
