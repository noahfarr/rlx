# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import random

import gymnasium as gym
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import rlx.sac.hyperparameters as h
from rlx.sac.sac import SAC


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
    parser.add_argument("--buffer_size", type=int, default=h.buffer_size)
    parser.add_argument("--q_lr", type=float, default=h.q_lr)
    parser.add_argument("--policy_lr", type=float, default=h.policy_lr)
    parser.add_argument("--policy_frequency", type=int, default=h.policy_frequency)
    parser.add_argument(
        "--target_network_frequency", type=int, default=h.target_network_frequency
    )
    parser.add_argument("--tau", type=float, default=h.tau)
    parser.add_argument("--batch_size", type=int, default=h.batch_size)
    parser.add_argument("--learning_starts", type=int, default=h.learning_starts)
    parser.add_argument("--gamma", type=float, default=h.gamma)
    parser.add_argument("--alpha", type=float, default=h.alpha)
    parser.add_argument("--autotune", type=bool, default=h.autotune)

    args = parser.parse_args()
    return args


def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


class SoftQNetwork(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        activations,
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
        ), "Number of layers and activations should match"

    def __call__(self, x, a):
        x = mx.concatenate([x, a], axis=1)
        for layer, activation in zip(self.layers[:-1], self.activations):
            x = activation(layer(x))
        x = self.layers[-1](x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))

        self.action_scale = mx.array(
            (env.single_action_space.high - env.single_action_space.low) / 2.0,
            dtype=mx.float32,
        )
        self.action_bias = mx.array(
            (env.single_action_space.high + env.single_action_space.low) / 2.0,
            dtype=mx.float32,
        )

        self.freeze(recurse=False, keys=["action_scale", "action_bias"])

    def __call__(self, x):
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = nn.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std


def copy_weights(source, target, tau):
    weights = []
    for i, ((target_params), (q_params)) in enumerate(
        zip(
            target.parameters()["layers"],
            source.parameters()["layers"],
        )
    ):
        target_weight = target_params["weight"]
        target_bias = target_params["bias"]
        q_weight = q_params["weight"]
        q_bias = q_params["bias"]

        weight = tau * q_weight + (1.0 - tau) * target_weight
        bias = tau * q_bias + (1.0 - tau) * target_bias

        weights.append((f"layers.{i}.weight", weight))
        weights.append((f"layers.{i}.bias", bias))
    target.load_weights(weights)


if __name__ == "__main__":
    args = parse_args()

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed)])
    obs, info = envs.reset()
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs)
    mx.eval(actor.parameters())
    qf1 = SoftQNetwork(
        2,
        np.array(envs.single_observation_space.shape).prod()
        + np.prod(envs.single_action_space.shape),
        256,
        [nn.relu, nn.relu],
    )
    mx.eval(qf1.parameters())
    qf2 = SoftQNetwork(
        2,
        np.array(envs.single_observation_space.shape).prod()
        + np.prod(envs.single_action_space.shape),
        256,
        [nn.relu, nn.relu],
    )
    mx.eval(qf2.parameters())
    qf1_target = SoftQNetwork(
        2,
        np.array(envs.single_observation_space.shape).prod()
        + np.prod(envs.single_action_space.shape),
        256,
        [nn.relu, nn.relu],
    )
    qf2_target = SoftQNetwork(
        2,
        np.array(envs.single_observation_space.shape).prod()
        + np.prod(envs.single_action_space.shape),
        256,
        [nn.relu, nn.relu],
    )
    copy_weights(qf1, qf1_target, 1.0)
    copy_weights(qf2, qf2_target, 1.0)

    actor_optimizer = optim.Adam(learning_rate=args.policy_lr)
    qf1_optimizer = optim.Adam(learning_rate=args.q_lr)
    qf2_optimizer = optim.Adam(learning_rate=args.q_lr)

    agent = SAC(
        actor=actor,
        qf1=qf1,
        qf2=qf2,
        qf1_target=qf1_target,
        qf2_target=qf2_target,
        actor_optimizer=actor_optimizer,
        qf1_optimizer=qf1_optimizer,
        qf2_optimizer=qf2_optimizer,
    )

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -mx.prod(mx.array(envs.single_action_space.shape)).item()
        log_alpha = mx.zeros(1)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam(learning_rate=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        handle_timeout_termination=False,
    )

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions, _, _ = agent.get_action(mx.array(obs))
            actions = np.array(actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}"
                )
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            observations = mx.array(data.observations.numpy())
            actions = mx.array(data.actions.numpy())
            rewards = mx.array(data.rewards.numpy())
            dones = mx.array(data.dones.numpy())
            next_observations = mx.array(data.next_observations.numpy())

            next_state_actions, next_state_log_pi, _ = agent.get_action(
                next_observations
            )
            qf1_next_target = qf1_target(
                next_observations,
                next_state_actions,
            )
            qf2_next_target = qf2_target(
                next_observations,
                next_state_actions,
            )
            min_qf_next_target = (
                mx.minimum(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
            )

            next_q_value = rewards.flatten() + (
                1 - dones.flatten()
            ) * args.gamma * min_qf_next_target.reshape(-1)

            agent.update_q_networks(next_q_value, observations, actions)

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    agent.update_actor(observations, alpha)
                    if args.autotune:
                        _, log_prob, _ = actor.get_action(observations)
                        agent.update_entropy(log_alpha, log_prob, target_entropy)
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                copy_weights(qf1, qf1_target, args.tau)
                copy_weights(qf2, qf2_target, args.tau)

    envs.close()
