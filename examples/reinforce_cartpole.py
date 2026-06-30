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

from rlx.algorithms.reinforce import REINFORCEConfig, REINFORCE
from rlx.buffers.rollout_buffer import RolloutBuffer
from rlx.environments import CartPole
from rlx.utils.logger import Logger


@dataclass
class Args:
    experiment_name: str = "reinforce"
    seed: int = 1
    total_timesteps: int = 100_000_000
    learning_rate: float = 2.5e-4
    track: bool = False
    wandb_project_name: str = "rlx"
    wandb_entity: str = ""
    reinforce: REINFORCEConfig = field(default_factory=REINFORCEConfig)


class ActorNetwork(nn.Module):
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
    config = args.reinforce
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

    actor_network = ActorNetwork(env)
    mx.eval(actor_network.parameters())
    optimizer = optim.Adam(learning_rate=args.learning_rate)

    buffer = RolloutBuffer(
        config.num_steps,
        env.observation_space,
        env.action_space,
        num_envs=config.num_envs,
    )
    algorithm = REINFORCE(
        config=config,
        env=env,
        actor_network=actor_network,
        optimizer=optimizer,
        buffer=buffer,
        key=mx.random.key(args.seed),
    )

    logger = Logger()

    algorithm.train(args.total_timesteps, callback=logger)
    algorithm.evaluate(10_000, callback=logger)

    writer.close()
