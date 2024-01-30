import argparse

import numpy as np

import mlx.core as mx

import gymnasium as gym

import rlx.reinforce.hyperparameters as h
from rlx.reinforce.reinforce import REINFORCE
from rlx.common.mlp import MLP
from rlx.common.rollout_buffer import RolloutBuffer
from rlx.common.utils import get_discounted_sum_of_rewards


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


def main():
    args = parse_args()
    env = gym.make(
        id=args.env_id,
        render_mode=args.render,
    )
    policy = MLP(
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
        obs, info = env.reset()
        terminated = truncated = False
        rollout_buffer.clear()
        while not terminated and not truncated:
            action = agent.get_action(mx.array(obs))
            new_obs, reward, terminated, truncated, info = env.step(action)
            rollout_buffer.append(
                obs,
                new_obs,
                action,
                reward,
                terminated,
                truncated,
            )
            obs = new_obs
            timestep += 1
            if timestep % 5000 == 0:
                print("Mean return: ", np.mean(np.array(rollout_buffer.returns)))
        observations = mx.array(rollout_buffer.observations)
        actions = mx.array(rollout_buffer.actions)
        rewards = np.array(rollout_buffer.rewards)
        rollout_buffer.returns.append(np.sum(rewards))
        discounted_rewards = get_discounted_sum_of_rewards(rewards, gamma=args.gamma)
        agent.update(observations, actions, discounted_rewards)


if __name__ == "__main__":
    main()
