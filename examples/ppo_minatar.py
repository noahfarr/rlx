import random
import time
from dataclasses import dataclass, field
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import gymnasium as gym
import numpy as np
import tyro
from torch.utils.tensorboard import SummaryWriter

from rlx.algorithms.ppo import PPOConfig, PPO
from rlx.buffers.rollout_buffer import RolloutBuffer
from rlx.environments import RecordEpisodeStatistics, Vectorize
from rlx.environments.minatar import (
    Breakout,
    Freeway,
    SpaceInvaders,
    Asterix,
    Seaquest,
)
from rlx.utils.logger import Logger
from rlx.utils.distributions import Categorical

ENVIRONMENTS = {
    "breakout": Breakout,
    "freeway": Freeway,
    "space_invaders": SpaceInvaders,
    "asterix": Asterix,
    "seaquest": Seaquest,
}


@dataclass
class Args:
    experiment_name: str = "ppo_minatar"
    env_name: Literal[
        "breakout", "freeway", "space_invaders", "asterix", "seaquest"
    ] = "breakout"
    seed: int = 1
    total_timesteps: int = 10_000_000
    learning_rate: float = 2.5e-4
    track: bool = False
    wandb_project_name: str = "rlx"
    wandb_entity: str = ""
    ppo: PPOConfig = field(default_factory=PPOConfig)


class ActorCritic(nn.Module):
    """CNN actor-critic matching the original MinAtar architecture.

    A single 3x3, stride-1 convolution with 16 filters feeds a 128-unit hidden
    layer that is shared by the policy and value heads. MLX convolutions expect
    channels-last input, which is exactly the (H, W, C) layout the MinAtar
    environments emit.
    """

    def __init__(self, env):
        super().__init__()
        height, width, channels = env.observation_space.shape

        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        num_linear_units = size_linear_unit(height) * size_linear_unit(width) * 16

        self.conv = nn.Conv2d(channels, 16, kernel_size=3, stride=1)
        self.fc_hidden = nn.Linear(num_linear_units, 128)
        self.actor = nn.Linear(128, env.action_space.n)
        self.critic = nn.Linear(128, 1)

    def __call__(self, x):
        x = nn.relu(self.conv(x))
        x = nn.relu(self.fc_hidden(x.reshape(x.shape[0], -1)))
        return Categorical(self.actor(x)), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    config = args.ppo
    run_name = (
        f"{args.experiment_name}__{args.env_name}__{args.seed}__{int(time.time())}"
    )
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    env = Vectorize(RecordEpisodeStatistics(ENVIRONMENTS[args.env_name]()))
    assert isinstance(
        env.action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    network = ActorCritic(env)
    mx.eval(network.parameters())
    optimizer = optim.Adam(learning_rate=args.learning_rate)

    buffer = RolloutBuffer(
        config.num_steps,
        env.observation_space,
        env.action_space,
        gamma=config.gamma,
        num_envs=config.num_envs,
    )
    algorithm = PPO(
        config=config,
        env=env,
        network=network,
        optimizer=optimizer,
        buffer=buffer,
        key=mx.random.key(args.seed),
    )

    logger = Logger()

    algorithm.train(args.total_timesteps, callback=logger)
    algorithm.evaluate(10_000, callback=logger)

    writer.close()
