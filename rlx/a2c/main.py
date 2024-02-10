import argparse

import numpy as np

import mlx.core as mx
import mlx.nn as nn

import gymnasium as gym

import rlx.a2c.hyperparameters as h
from rlx.a2c.a2c import A2C
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

    # Actor Network:
    parser.add_argument(
        "--actor_learning_rate", type=float, default=h.actor_learning_rate
    )
    parser.add_argument("--actor_num_layers", type=int, default=h.actor_num_layers)
    parser.add_argument(
        "--actor_hidden_dim", nargs="+", type=int, default=h.actor_hidden_dim
    )
    parser.add_argument("--actor_activations", nargs="+", default=h.actor_activations)
    parser.add_argument("--actor_optimizer", default=h.actor_optimizer)

    # Critic Network:
    parser.add_argument(
        "--critic_learning_rate", type=float, default=h.critic_learning_rate
    )
    parser.add_argument("--critic_num_layers", type=int, default=h.critic_num_layers)
    parser.add_argument(
        "--critic_hidden_dim", nargs="+", type=int, default=h.critic_hidden_dim
    )
    parser.add_argument("--critic_activations", nargs="+", default=h.critic_activations)
    parser.add_argument("--critic_optimizer", default=h.critic_optimizer)

    return parser.parse_args()


class Actor(nn.Module):
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


class Critic(nn.Module):
    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        activations: str,
    ):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [1]
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

    actor = Actor(
        num_layers=args.actor_num_layers,
        input_dim=env.observation_space.shape[0],
        hidden_dim=args.actor_hidden_dim,
        output_dim=env.action_space.n,
        activations=args.actor_activations,
    )
    mx.eval(actor.parameters())

    actor_optimizer = args.actor_optimizer(learning_rate=args.actor_learning_rate)

    critic = Critic(
        num_layers=args.critic_num_layers,
        input_dim=env.observation_space.shape[0],
        hidden_dim=args.critic_hidden_dim,
        activations=args.critic_activations,
    )
    mx.eval(critic.parameters())

    critic_optimizer = args.critic_optimizer(learning_rate=args.critic_learning_rate)

    rollout_buffer = RolloutBuffer()

    agent = A2C(
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
    )

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
                terminated=terminated,
            )
            obs = next_obs
            done = terminated or truncated

            if terminated:
                rollout_buffer.add(
                    obs=obs,
                )

            timestep += 1
            if "episode" in info:
                print(f"Timestep: {timestep}, Episodic Returns: {info['episode']['r']}")

        observations = rollout_buffer.get("obs")
        actions = rollout_buffer.get("action")
        rewards = rollout_buffer.get("reward")
        terminations = rollout_buffer.get("terminated")
        rewards_to_go = get_rewards_to_go(rewards, args.gamma)
        agent.update(observations, actions, rewards_to_go, terminations)


if __name__ == "__main__":
    main()
