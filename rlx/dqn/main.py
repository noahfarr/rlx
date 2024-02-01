import argparse
import random
import time

import gymnasium as gym
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as utils

# from rlx.dqn.dqn import DQN
import rlx.dqn.hyperparameters as h
from rlx.dqn.dqn import DQN
from rlx.common.mlp import MLP


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
    parser.add_argument("--buffer_size", type=int, default=h.buffer_size)
    parser.add_argument("--gamma", type=float, default=h.gamma)
    parser.add_argument("--tau", type=float, default=h.tau)
    parser.add_argument(
        "--target_network_frequency", type=int, default=h.target_network_frequency
    )
    parser.add_argument("--batch_size", type=int, default=h.batch_size)
    parser.add_argument("--start_e", type=float, default=h.start_e)
    parser.add_argument("--end_e", type=float, default=h.end_e)
    parser.add_argument(
        "--exploration_fraction", type=float, default=h.exploration_fraction
    )
    parser.add_argument("--learning_starts", type=int, default=h.learning_starts)
    parser.add_argument("--train_frequency", type=int, default=h.train_frequency)

    args = parser.parse_args()
    return args


def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def copy_weights(source, target, tau):
    # layers = source.parameters()["layers"]
    # weights = []
    # for i, layer in enumerate(layers):
    #     weights.append((f"layers.{i}.weight", layer["weight"]))
    #     weights.append((f"layers.{i}.bias", layer["bias"]))
    # target.load_weights(weights)

    weights = []
    for i, (target_network_param, q_network_param) in enumerate(
        zip(
            target.parameters()["layers"],
            source.parameters()["layers"],
        )
    ):
        target_weight = target_network_param["weight"]
        target_bias = target_network_param["bias"]
        q_weight = q_network_param["weight"]
        q_bias = q_network_param["bias"]

        weight = tau * q_weight + (1.0 - tau) * target_weight
        bias = tau * q_bias + (1.0 - tau) * target_bias

        weights.append((f"layers.{i}.weight", weight))
        weights.append((f"layers.{i}.bias", bias))
    target.load_weights(weights)


def linear_schedule(start_e, end_e, duration, t):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, seed + 1) for seed in range(args.num_envs)]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "Only discrete action spaces are supported"

    q_network = MLP(
        num_layers=2,
        input_dim=(np.array(envs.single_observation_space.shape).prod()),
        hidden_dim=128,
        output_dim=envs.single_action_space.n,
        activations=[nn.relu, nn.relu],
    )
    mx.eval(q_network.parameters())

    optimizer = optim.Adam(learning_rate=args.learning_rate)

    target_network = MLP(
        num_layers=2,
        input_dim=(np.array(envs.single_observation_space.shape).prod()),
        hidden_dim=128,
        output_dim=envs.single_action_space.n,
        activations=[nn.relu, nn.relu],
    )
    copy_weights(q_network, target_network, tau=1.0)

    agent = DQN(
        q_network=q_network,
        target_network=target_network,
        optimizer=optimizer,
    )

    replay_buffer = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        handle_timeout_termination=False,
    )

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)

    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )
        if random.random() < epsilon:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            q_values = agent.q_network(mx.array(obs))
            actions = np.array(mx.argmax(q_values, axis=1))

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(
                        f"global_step={global_step}, episodic_return={info['episode']['r']}"
                    )

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        replay_buffer.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = replay_buffer.sample(args.batch_size)
                agent.update(data, args)

                if global_step % 100 == 0:
                    print("SPS:", int(global_step / (time.time() - start_time)))

            # UPdate model

            # update target network
            if global_step % args.target_network_frequency == 0:
                copy_weights(agent.q_network, agent.target_network, tau=args.tau)

    envs.close()


if __name__ == "__main__":
    main()
