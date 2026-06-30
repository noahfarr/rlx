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

from rlx.algorithms.ppo import PPOConfig, PPO
from rlx.buffers.rollout_buffer import RolloutBuffer
from rlx.utils.logger import Logger
from rlx.utils.distributions import Gaussian
from rlx.environments.envpool import EnvPoolVectorEnv


@dataclass
class Args:
    env_id: str = "Pendulum-v1"
    experiment_name: str = "ppo_continuous"
    seed: int = 1
    total_timesteps: int = 5_000_000
    learning_rate: float = 3e-4
    track: bool = False
    wandb_project_name: str = "rlx"
    wandb_entity: str = ""
    ppo: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            num_envs=1024,
            num_steps=16,
            gamma=0.9,
            gae_lambda=0.95,
            num_minibatches=32,
            update_epochs=10,
            entropy_coefficient=0.0,
        )
    )


class ContinuousActorCritic(nn.Module):
    def __init__(self, envs):
        super().__init__()
        observation_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.array(envs.single_action_space.shape).prod()
        self.actor_mean = nn.Sequential(
            nn.Linear(observation_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )
        self.actor_log_std = mx.zeros((action_dim,))
        self.critic = nn.Sequential(
            nn.Linear(observation_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def __call__(self, x):
        mean = self.actor_mean(x)
        log_std = mx.broadcast_to(self.actor_log_std, mean.shape)
        return Gaussian(mean, log_std), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    config = args.ppo
    run_name = f"{args.env_id}__{args.experiment_name}__{args.seed}__{int(time.time())}"
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    # env setup
    envs = EnvPoolVectorEnv(
        args.env_id,
        config.num_envs,
        seed=args.seed,
        normalize_observations=True,
        normalize_rewards=True,
        gamma=config.gamma,
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    network = ContinuousActorCritic(envs)
    mx.eval(network.parameters())
    optimizer = optim.Adam(learning_rate=args.learning_rate)

    buffer = RolloutBuffer(
        config.num_steps,
        envs.single_observation_space,
        envs.single_action_space,
        gamma=config.gamma,
        num_envs=config.num_envs,
    )
    algorithm = PPO(
        config=config,
        envs=envs,
        network=network,
        optimizer=optimizer,
        buffer=buffer,
    )

    logger = Logger()

    algorithm.train(args.total_timesteps, callback=logger)
    algorithm.evaluate(10_000, callback=logger)

    envs.close()
    writer.close()
