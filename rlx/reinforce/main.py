import argparse
from typing import Any

import numpy as np

import mlx.core as mx
import mlx.nn as nn

import gymnasium as gym

import rlx.reinforce.hyperparameters as h
from rlx.reinforce.reinforce import REINFORCE
from rlx.common.rollout_buffer import RolloutBuffer
from rlx.common.utils import get_rewards_to_go


def parse_args():
    """
    Input argument parser.
    Loads default hyperparameters from hyperparameters.py
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    # General Parameters
    parser.add_argument("--seed", type=int, default=h.seed)
    parser.add_argument("--env_id", type=str, default=h.env_id)
    parser.add_argument("--gamma", type=float, default=h.gamma)
    parser.add_argument("--total_timesteps", type=int, default=h.total_timesteps)
    parser.add_argument(
        "--render", action="store_const", const="human", default=h.render_mode
    )

    # Policy Network:
    parser.add_argument("--learning_rate", type=float, default=h.learning_rate)
    parser.add_argument("--num_layers", type=int, default=h.num_layers)
    parser.add_argument("--hidden_dim", nargs="+", type=int, default=h.hidden_dim)
    parser.add_argument("--activations", nargs="+", type=str, default=h.activations)
    parser.add_argument("--optimizer", type=str, default=h.optimizer)

    return parser.parse_args()


class Policy(nn.Module):
    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activations: str,
    ):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [
            nn.Linear(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.activations = activations
        assert (
            len(self.layers) == len(self.activations) + 1
        ), "Number of layers and activations must match"

    def __call__(self, x):
        for layer, activation in zip(self.layers[:-1], self.activations):
            x = activation(layer(x))
        x = self.layers[-1](x)
        return x


def main():
    args = parse_args()

    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    env = gym.make(
        id=args.env_id,
        render_mode=args.render,
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)

    policy = Policy(
        num_layers=args.num_layers,
        input_dim=env.observation_space.shape[0],
        hidden_dim=args.hidden_dim,
        output_dim=env.action_space.n,
        activations=args.activations,
    )
    mx.eval(policy.parameters())

    optimizer = args.optimizer(learning_rate=args.learning_rate)

    rollout_buffer = RolloutBuffer()

    agent = REINFORCE(policy, optimizer)

    timestep = 0

    while timestep < args.total_timesteps:
        obs, _ = env.reset()
        done = False
        rollout_buffer.clear()
        while not done:
            action = agent.get_action(mx.array(obs))
            next_obs, reward, terminated, truncated, info = env.step(action)
            rollout_buffer.add(
                obs=obs,
                action=action,
                reward=reward,
            )
            obs = next_obs
            done = terminated or truncated
            timestep += 1
            if "episode" in info:
                print(f"Timestep: {timestep}, Episodic Returns: {info['episode']['r']}")
        observations = rollout_buffer.get("obs")
        actions = rollout_buffer.get("action")
        rewards = rollout_buffer.get("reward")
        rewards_to_go = get_rewards_to_go(rewards, args.gamma)
        agent.update(observations, actions, rewards_to_go)


if __name__ == "__main__":
    main()
