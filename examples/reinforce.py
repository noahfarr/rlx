from functools import partial
import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import gymnasium as gym
import numpy as np
import tyro
from torch.utils.tensorboard import SummaryWriter

from rlx.reinforce import REINFORCEConfig, REINFORCE
from rlx.buffers.rollout_buffer import RolloutBuffer


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


class ActorNetwork(nn.Module):
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
    config = tyro.cli(REINFORCEConfig)
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

    actor_network = ActorNetwork(envs)
    mx.eval(actor_network.parameters())
    optimizer = optim.Adam(learning_rate=config.learning_rate)

    buffer = RolloutBuffer(
        500,
        envs.single_observation_space,
        envs.single_action_space,
        n_envs=config.num_envs,
    )
    algorithm = REINFORCE(
        config=config,
        envs=envs,
        actor_network=actor_network,
        optimizer=optimizer,
        buffer=buffer,
    )

    def callback(info, step):
        if step % 100 == 0:
            print(info["episode"]["r"])

    algorithm.evaluate(10_000, callback=callback)
    algorithm.warmup(10_000)
    algorithm.train(500_000, callback=callback)
    algorithm.evaluate(10_000, callback=callback)

    envs.close()
    writer.close()
