from functools import partial
import random
import time
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import gymnasium as gym
import numpy as np
import tyro
from torch.utils.tensorboard import SummaryWriter

from rlx.algorithms.dqn import DQNConfig, DQN
from rlx.buffers.replay_buffer import ReplayBuffer
from rlx.environments import CartPole
from rlx.utils.logger import Logger


@dataclass
class Args:
    experiment_name: str = "dqn"
    seed: int = 1
    total_timesteps: int = 500000
    learning_rate: float = 2.5e-4
    buffer_size: int = 10000
    learning_starts: int = 10000
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    track: bool = False
    wandb_project_name: str = "rlx"
    wandb_entity: str = ""
    dqn: DQNConfig = field(default_factory=DQNConfig)


def linear_schedule(start_e: float, end_e: float, duration: float, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def __call__(self, x):
        return self.network(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    config = args.dqn
    run_name = f"{args.experiment_name}__{args.seed}__{int(time.time())}"
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

    env = CartPole()
    assert isinstance(
        env.action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    q_network = QNetwork(env)
    mx.eval(q_network.parameters())
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    target_network = QNetwork(env).update(q_network.parameters())

    buffer = ReplayBuffer(
        args.buffer_size,
        env.observation_space,
        env.action_space,
        num_envs=config.num_envs,
    )

    epsilon_schedule = partial(
        linear_schedule,
        start_e=args.start_e,
        end_e=args.end_e,
        duration=args.exploration_fraction * args.total_timesteps,
    )
    algorithm = DQN(
        config=config,
        env=env,
        q_network=q_network,
        target_network=target_network,
        optimizer=optimizer,
        buffer=buffer,
        epsilon_schedule=epsilon_schedule,
        key=mx.random.key(args.seed),
    )

    logger = Logger()

    algorithm.warmup(args.learning_starts)
    algorithm.train(args.total_timesteps, callback=logger)
    algorithm.evaluate(10_000, callback=logger)

    writer.close()
