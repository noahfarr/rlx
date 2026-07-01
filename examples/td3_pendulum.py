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

from rlx.algorithms.td3 import TD3Config, TD3
from rlx.buffers.replay_buffer import ReplayBuffer
from rlx.utils.logger import Logger


@dataclass
class Args:
    env_id: str = "Pendulum-v1"
    experiment_name: str = "td3"
    seed: int = 1
    total_timesteps: int = 1000000
    buffer_size: int = 1000000
    learning_starts: int = 25000
    policy_learning_rate: float = 3e-4
    q_learning_rate: float = 3e-4
    track: bool = False
    wandb_project_name: str = "rlx"
    wandb_entity: str = ""
    td3: TD3Config = field(default_factory=TD3Config)


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


class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        observation_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.array(envs.single_action_space.shape).prod()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

        high = mx.array(envs.single_action_space.high)
        low = mx.array(envs.single_action_space.low)
        self.action_scale = (high - low) / 2.0
        self.action_bias = (high + low) / 2.0

    def __call__(self, x):
        x = mx.tanh(self.network(x))
        return x * self.action_scale + self.action_bias


class TwinCritic(nn.Module):
    def __init__(self, envs, num_critics=2):
        super().__init__()
        observation_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.array(envs.single_action_space.shape).prod()
        input_dim = int(observation_dim + action_dim)
        self.num_critics = num_critics

        dims = [input_dim, 256, 256, 1]
        self.layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            critics = [nn.Linear(in_dim, out_dim) for _ in range(num_critics)]
            self.layers.append(
                {
                    "weight": mx.stack([critic.weight for critic in critics], axis=0),
                    "bias": mx.stack([critic.bias for critic in critics], axis=0),
                }
            )

    def __call__(self, observation, action):
        x = mx.concatenate([observation, action], axis=-1)

        def forward(layers, x):
            for index, layer in enumerate(layers):
                x = x @ layer["weight"].T + layer["bias"]
                if index < len(layers) - 1:
                    x = nn.relu(x)
            return x

        return mx.vmap(forward, in_axes=(0, None))(self.layers, x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    config = args.td3
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, args.seed + i, i, False, run_name)
            for i in range(config.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    actor_network = Actor(envs)
    critic_network = TwinCritic(envs)
    mx.eval(actor_network.parameters(), critic_network.parameters())

    target_actor_network = Actor(envs)
    target_actor_network.update(actor_network.parameters())
    target_critic_network = TwinCritic(envs)
    target_critic_network.update(critic_network.parameters())

    actor_optimizer = optim.Adam(learning_rate=args.policy_learning_rate)
    critic_optimizer = optim.Adam(learning_rate=args.q_learning_rate)

    buffer = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        num_envs=config.num_envs,
    )
    algorithm = TD3(
        config=config,
        envs=envs,
        actor_network=actor_network,
        target_actor_network=target_actor_network,
        critic_network=critic_network,
        target_critic_network=target_critic_network,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        buffer=buffer,
    )

    logger = Logger()

    algorithm.warmup(args.learning_starts)
    algorithm.train(args.total_timesteps, callback=logger)
    algorithm.evaluate(10_000, callback=logger)

    envs.close()
    writer.close()
