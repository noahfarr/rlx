# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import random
import time

import gymnasium as gym
import numpy as np
import tyro

import mlx.core as mx
import mlx.optimizers as optim
import mlx.nn as nn

from rlx.common.mlp import MLP
from rlx.ppo.ppo import PPO
import rlx.ppo.hyperparameters as h


def parse_args():
    """
    Input argument parser.
    Loads default hyperparameters from hyperparameters.py
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    # General Parameters
    parser.add_argument("--exp_name", type=str, default=h.exp_name)
    parser.add_argument("--seed", type=int, default=h.seed)

    # Algorithm specific arguments
    parser.add_argument("--env_id", type=str, default=h.env_id)
    parser.add_argument("--total_timesteps", type=int, default=h.total_timesteps)
    parser.add_argument("--learning_rate", type=float, default=h.learning_rate)
    parser.add_argument("--num_envs", type=int, default=h.num_envs)
    parser.add_argument("--num_steps", type=int, default=h.num_steps)
    parser.add_argument("--anneal_lr", type=bool, default=h.anneal_lr)
    parser.add_argument("--gamma", type=float, default=h.gamma)
    parser.add_argument("--gae_lambda", type=float, default=h.gae_lambda)
    parser.add_argument("--num_minibatches", type=int, default=h.num_minibatches)
    parser.add_argument("--update_epochs", type=int, default=h.update_epochs)
    parser.add_argument("--norm_adv", type=bool, default=h.norm_adv)
    parser.add_argument("--clip_coef", type=float, default=h.clip_coef)
    parser.add_argument("--clip_vloss", type=bool, default=h.clip_vloss)
    parser.add_argument("--ent_coef", type=float, default=h.ent_coef)
    parser.add_argument("--vf_coef", type=float, default=h.vf_coef)
    parser.add_argument("--max_grad_norm", type=float, default=h.max_grad_norm)
    parser.add_argument("--target_kl", type=float, default=h.target_kl)

    parser.add_argument("--batch_size", type=int, default=h.batch_size)
    parser.add_argument("--minibatch_size", type=int, default=h.minibatch_size)
    parser.add_argument("--num_iterations", type=int, default=h.num_iterations)

    args = parser.parse_args()
    return args


def make_env(env_id):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


if __name__ == "__main__":
    args = parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id) for i in range(args.num_envs)],
    )

    actor = MLP(
        num_layers=2,
        input_dim=np.array(envs.single_observation_space.shape).prod(),
        hidden_dim=64,
        output_dim=envs.single_action_space.n,
        activations=[nn.Tanh(), nn.Tanh()],
    )
    mx.eval(actor.parameters())

    actor_optimizer = optim.Adam(learning_rate=args.learning_rate, eps=1e-5)

    critic = MLP(
        num_layers=2,
        input_dim=np.array(envs.single_observation_space.shape).prod(),
        hidden_dim=64,
        output_dim=1,
        activations=[nn.Tanh(), nn.Tanh()],
    )
    mx.eval(critic.parameters())

    critic_optimizer = optim.Adam(learning_rate=args.learning_rate, eps=1e-5)

    agent = PPO(
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
    )

    # ALGO Logic: Storage setup
    obs = mx.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    )
    actions = mx.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape)
    log_probs = mx.zeros((args.num_steps, args.num_envs))
    rewards = mx.zeros((args.num_steps, args.num_envs))
    dones = mx.zeros((args.num_steps, args.num_envs))
    values = mx.zeros((args.num_steps, args.num_envs))

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = mx.array(next_obs)
    next_done = mx.zeros(args.num_envs)

    for iteration in range(1, args.num_iterations + 1):
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            action = agent.get_action(next_obs)
            log_prob = agent.get_log_prob(next_obs, action)
            value = agent.get_value(next_obs)
            values[step] = value.flatten()
            actions[step] = action
            log_probs[step] = log_prob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                np.array(action)
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = mx.array(reward)
            next_obs, next_done = mx.array(next_obs), mx.array(next_done)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}"
                        )

        # bootstrap value if not done
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = mx.zeros_like(rewards)
        last_gae_lambda = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                next_non_terminal = 1.0 - next_done
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]
            delta = (
                rewards[t] + args.gamma * next_values * next_non_terminal - values[t]
            )
            advantages[t] = last_gae_lambda = (
                delta
                + args.gamma * args.gae_lambda * next_non_terminal * last_gae_lambda
            )
        returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_log_probs = log_probs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        agent.update(
            args, b_obs, b_log_probs, b_actions, b_advantages, b_returns, b_values
        )

    envs.close()
