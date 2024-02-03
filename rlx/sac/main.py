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


def q_loss_fn(qf1, qf2, next_q_value, observations, actions):
    qf1_a_values = qf1(observations, actions).reshape(-1)
    qf2_a_values = qf2(observations, actions).reshape(-1)
    qf1_loss = nn.losses.mse_loss(qf1_a_values, next_q_value)
    qf2_loss = nn.losses.mse_loss(qf2_a_values, next_q_value)
    qf_loss = qf1_loss + qf2_loss
    return qf_loss


def actor_loss_fn(actor, qf1, qf2, observations, alpha):
    pi, log_pi, _ = actor.get_action(observations)
    qf1_pi = qf1(observations, pi)
    qf2_pi = qf2(observations, pi)
    min_qf_pi = mx.minimum(qf1_pi, qf2_pi)
    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()
    return actor_loss


def entropy_loss_fn(log_alpha, log_prob, target_entropy):
    alpha_loss = (-log_alpha.exp() * (log_prob + target_entropy)).mean()
    return alpha_loss


entropy_loss_and_grad_fn = mx.value_and_grad(entropy_loss_fn)


class SoftQNetwork(nn.Module):
    def __init__(
        self,
        env,
    ):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod()
            + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        self.loss_and_grad_fn = nn.value_and_grad(self, q_loss_fn)

    def __call__(self, x, a):
        x = mx.concatenate([x, a], axis=1)
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        x = self.fc3(x)
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

        self.loss_and_grad_fn = nn.value_and_grad(self, actor_loss_fn)

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

    def sample_normal(self, mean, std):
        normal = mx.random.normal()
        return mean + std * normal

    def get_log_prob(self, sample, mean, std):
        variance = std.square()
        log_variance = variance.log()
        return -0.5 * (
            log_variance
            + mx.log(mx.array(2 * mx.pi))
            + ((sample - mean).square() / variance)
        )

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        x_t = self.sample_normal(
            mean, std
        )  # for reparameterization trick (mean + std * N(0,1))
        y_t = nn.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = self.get_log_prob(x_t, mean, std)
        # Enforcing Action Bound
        log_prob -= mx.log(self.action_scale * (1 - y_t.square()) + 1e-6)
        log_prob = log_prob.sum(1, keepdims=True)
        mean = nn.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


def copy_weights(source, target, tau):
    weights = []
    for i, ((_, target_params), (_, q_params)) in enumerate(
        zip(
            target.parameters().items(),
            source.parameters().items(),
        )
    ):
        target_weight = target_params["weight"]
        target_bias = target_params["bias"]
        q_weight = q_params["weight"]
        q_bias = q_params["bias"]

        weight = tau * q_weight + (1.0 - tau) * target_weight
        bias = tau * q_bias + (1.0 - tau) * target_bias

        weights.append((f"fc{i+1}.weight", weight))
        weights.append((f"fc{i+1}.bias", bias))
    target.load_weights(weights)


if __name__ == "__main__":
    args = parse_args()

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed)])
    obs, info = envs.reset()
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs)
    mx.eval(actor.parameters())
    qf1 = SoftQNetwork(envs)
    mx.eval(qf1.parameters())
    qf2 = SoftQNetwork(envs)
    mx.eval(qf2.parameters())
    qf1_target = SoftQNetwork(envs)
    qf2_target = SoftQNetwork(envs)
    copy_weights(qf1, qf1_target, 1.0)
    copy_weights(qf2, qf2_target, 1.0)

    qf1_optimizer = optim.Adam(learning_rate=args.q_lr)
    qf2_optimizer = optim.Adam(learning_rate=args.q_lr)

    actor_optimizer = optim.Adam(learning_rate=args.policy_lr)

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
            actions, _, _ = actor.get_action(mx.array(obs))
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

            next_state_actions, next_state_log_pi, _ = actor.get_action(
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

            qf1_loss, qf1_grads = qf1.loss_and_grad_fn(
                qf1,
                qf2,
                next_q_value,
                observations,
                actions,
            )
            qf2_loss, qf2_grads = qf2.loss_and_grad_fn(
                qf1,
                qf2,
                next_q_value,
                observations,
                actions,
            )
            qf1_optimizer.update(qf1, qf1_grads)
            qf2_optimizer.update(qf2, qf2_grads)

            mx.eval(qf1.parameters(), qf1_optimizer.state)
            mx.eval(qf2.parameters(), qf2_optimizer.state)

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    actor_loss, actor_grads = actor.loss_and_grad_fn(
                        actor,
                        qf1,
                        qf2,
                        observations,
                        alpha,
                    )
                    actor_optimizer.update(actor, actor_grads)
                    mx.eval(actor.parameters(), actor_optimizer.state)

                    if args.autotune:
                        _, log_prob, _ = actor.get_action(observations)
                        entropy_loss, entropy_grad = entropy_loss_and_grad_fn(
                            log_alpha,
                            log_prob,
                            target_entropy,
                        )
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                copy_weights(qf1, qf1_target, args.tau)
                copy_weights(qf2, qf2_target, args.tau)

    envs.close()
