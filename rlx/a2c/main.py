import argparse

import numpy as np

import mlx.core as mx

import gymnasium as gym

import rlx.a2c.hyperparameters as h
from rlx.a2c.a2c import A2C
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


def main():
    args = parse_args()
    env = gym.make(
        id=args.env_id,
        render_mode=args.render,
    )
    actor = MLP(
        num_layers=args.actor_num_layers,
        input_dim=env.observation_space.shape[0],
        hidden_dim=args.actor_hidden_dim,
        output_dim=env.action_space.n,
        activations=args.actor_activations,
    )
    mx.eval(actor.parameters())

    actor_optimizer = args.actor_optimizer(learning_rate=args.actor_learning_rate)

    critic = MLP(
        num_layers=args.critic_num_layers,
        input_dim=env.observation_space.shape[0],
        hidden_dim=args.critic_hidden_dim,
        output_dim=1,
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

            if terminated:
                rollout_buffer.observations.append(mx.array(obs.tolist()))
                break
            if truncated:
                break
        observations = mx.array(rollout_buffer.observations)
        actions = mx.array(rollout_buffer.actions)
        rewards = np.array(rollout_buffer.rewards)
        terminations = mx.array(rollout_buffer.terminations)
        rollout_buffer.returns.append(np.sum(rewards))
        discounted_rewards = get_discounted_sum_of_rewards(rewards, gamma=args.gamma)
        agent.update(observations, actions, discounted_rewards, terminations)


if __name__ == "__main__":
    main()
