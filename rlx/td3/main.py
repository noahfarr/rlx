import argparse

import gymnasium as gym

import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer


import rlx.td3.hyperparameters as h
from rlx.td3.td3 import TD3


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
    parser.add_argument("--buffer_size", type=int, default=h.buffer_size)
    parser.add_argument("--gamma", type=float, default=h.gamma)
    parser.add_argument("--tau", type=float, default=h.tau)
    parser.add_argument("--batch_size", type=int, default=h.batch_size)
    parser.add_argument("--policy_noise", type=float, default=h.policy_noise)
    parser.add_argument("--exploration_noise", type=float, default=h.exploration_noise)
    parser.add_argument("--learning_starts", type=int, default=h.learning_starts)
    parser.add_argument("--policy_frequency", type=int, default=h.policy_frequency)
    parser.add_argument("--noise_clip", type=float, default=h.noise_clip)

    args = parser.parse_args()
    return args


class QNetwork(nn.Module):
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
        return self.layers[-1](x)


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        layer_sizes = (
            [np.array(env.single_observation_space.shape).prod()]
            + [256]
            + [256]
            + [np.prod(env.single_action_space.shape)]
        )
        self.layers = [
            nn.Linear(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

        self.activations = [nn.relu, nn.relu, nn.tanh]

        # action rescaling
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
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        return x * self.action_scale + self.action_bias


def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


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

    if hasattr(source, "action_scale"):
        weights.append(("action_scale", source.action_scale))
    if hasattr(source, "action_bias"):
        weights.append(("action_bias", source.action_bias))
    target.load_weights(weights)


def sample_normal(mean, std, shape=None):
    if shape is None:
        shape = []
    normal = mx.random.normal(shape=shape)
    return mean + std * normal


def main():
    args = parse_args()
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed)])
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    actor = Actor(envs)
    qf1 = QNetwork(
        num_layers=2,
        input_dim=np.array(envs.single_observation_space.shape).prod()
        + np.prod(envs.single_action_space.shape),
        hidden_dim=256,
        activations=[nn.relu, nn.relu],
    )
    qf2 = QNetwork(
        num_layers=2,
        input_dim=np.array(envs.single_observation_space.shape).prod()
        + np.prod(envs.single_action_space.shape),
        hidden_dim=256,
        activations=[nn.relu, nn.relu],
    )

    qf1_target = QNetwork(
        num_layers=2,
        input_dim=np.array(envs.single_observation_space.shape).prod()
        + np.prod(envs.single_action_space.shape),
        hidden_dim=256,
        activations=[nn.relu, nn.relu],
    )
    qf2_target = QNetwork(
        num_layers=2,
        input_dim=np.array(envs.single_observation_space.shape).prod()
        + np.prod(envs.single_action_space.shape),
        hidden_dim=256,
        activations=[nn.relu, nn.relu],
    )

    target_actor = Actor(envs)

    copy_weights(actor, target_actor, 1.0)
    copy_weights(qf1, qf1_target, 1.0)
    copy_weights(qf2, qf2_target, 1.0)

    qf1_optimizer = optim.Adam(learning_rate=args.learning_rate)
    qf2_optimizer = optim.Adam(learning_rate=args.learning_rate)
    actor_optimizer = optim.Adam(learning_rate=args.learning_rate)

    agent = TD3(
        actor=actor,
        qf1=qf1,
        qf2=qf2,
        qf1_target=qf1_target,
        qf2_target=qf2_target,
        actor_optimizer=actor_optimizer,
        qf1_optimizer=qf1_optimizer,
        qf2_optimizer=qf2_optimizer,
    )

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
            actions = actor(mx.array(obs))
            actions += sample_normal(0, actor.action_scale * args.exploration_noise)
            actions = np.array(actions).clip(
                envs.single_action_space.low, envs.single_action_space.high
            )

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
            rewards = mx.array(data.rewards.numpy()).flatten()
            dones = mx.array(data.dones.numpy()).flatten()
            next_observations = mx.array(data.next_observations.numpy())

            clipped_noise = (
                mx.clip(
                    sample_normal(0, 1, actions.shape) * args.policy_noise,
                    -args.noise_clip,
                    args.noise_clip,
                )
                * target_actor.action_scale
            )

            next_state_actions = mx.clip(
                target_actor(next_observations) + clipped_noise,
                mx.array(envs.single_action_space.low[0]),
                mx.array(envs.single_action_space.high[0]),
            )
            qf1_next_target = qf1_target(next_observations, next_state_actions)
            qf2_next_target = qf2_target(next_observations, next_state_actions)
            min_qf_next_target = mx.minimum(qf1_next_target, qf2_next_target)
            next_q_value = rewards.flatten() + (1 - dones.flatten()) * args.gamma * (
                min_qf_next_target
            ).reshape(-1)

            agent.update_q_networks(next_q_value, observations, actions)

            if global_step % args.policy_frequency == 0:
                agent.update_actor(observations)

                # update the target network
                copy_weights(actor, target_actor, args.tau)
                copy_weights(qf1, qf1_target, args.tau)
                copy_weights(qf2, qf2_target, args.tau)

    envs.close()


if __name__ == "__main__":
    main()
