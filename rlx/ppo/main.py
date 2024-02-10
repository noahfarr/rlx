# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import random
import time

import gymnasium as gym
import numpy as np

import mlx.core as mx
import mlx.optimizers as optim
import mlx.nn as nn

from rlx.ppo.ppo import PPO
import rlx.ppo.hyperparameters as h
from rlx.common.rollout_buffer import RolloutBuffer


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
        ), "Number of layers and activations should match"

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
        ), "Number of layers and activations should match"

    def __call__(self, x):
        for layer, activation in zip(self.layers[:-1], self.activations):
            x = activation(layer(x))
        x = self.layers[-1](x)
        return x


def anneal_lr(actor_optim, critic_optim, iteration, num_iterations, learning_rate):
    frac = 1.0 - (iteration - 1.0) / num_iterations
    new_lr = frac * learning_rate

    actor_optim_state = actor_optim.state
    critic_optim_state = critic_optim.state

    actor_optim = optim.Adam(learning_rate=new_lr, eps=1e-5)
    critic_optim = optim.Adam(learning_rate=new_lr, eps=1e-5)

    actor_optim.state = actor_optim_state
    critic_optim.state = critic_optim_state


if __name__ == "__main__":
    args = parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id) for i in range(args.num_envs)],
    )
    assert isinstance(
        envs.single_observation_space,
        gym.spaces.Discrete,
        "Only discrete action spaces are supported",
    )

    actor = Actor(
        num_layers=2,
        input_dim=np.array(envs.single_observation_space.shape).prod(),
        hidden_dim=64,
        output_dim=envs.single_action_space.n,
        activations=[nn.Tanh(), nn.Tanh()],
    )
    mx.eval(actor.parameters())

    actor_optimizer = optim.Adam(learning_rate=args.learning_rate, eps=1e-5)

    critic = Critic(
        num_layers=2,
        input_dim=np.array(envs.single_observation_space.shape).prod(),
        hidden_dim=64,
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

    rollout_buffer = RolloutBuffer()

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = mx.array(next_obs)
    next_done = mx.zeros(args.num_envs)

    for iteration in range(1, args.num_iterations + 1):
        rollout_buffer.clear()

        if args.anneal_lr:
            anneal_lr(
                actor_optimizer,
                critic_optimizer,
                iteration,
                args.num_iterations,
                args.learning_rate,
            )

        for step in range(0, args.num_steps):
            global_step += args.num_envs

            # ALGO LOGIC: action logic
            action = agent.get_action(next_obs)
            log_prob = agent.get_log_prob(next_obs, action)
            value = agent.get_value(next_obs)
            rollout_buffer.add(
                obs=next_obs,
                action=action,
                log_prob=log_prob,
                value=value.flatten(),
                done=next_done,
            )

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                np.array(action)
            )
            rollout_buffer.add(reward=reward)
            next_obs = mx.array(next_obs)
            next_done = np.logical_or(terminations, truncations)
            next_done = mx.array(next_done)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}"
                        )

        # bootstrap value if not done
        next_value = agent.get_value(next_obs).reshape(1, -1)

        observations = rollout_buffer.get("obs")
        actions = rollout_buffer.get("action")
        log_probs = rollout_buffer.get("log_prob")
        rewards = rollout_buffer.get("reward")
        values = rollout_buffer.get("value")
        dones = rollout_buffer.get("done")
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
        b_obs = observations.reshape((-1,) + envs.single_observation_space.shape)
        b_log_probs = log_probs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        agent.update(
            args, b_obs, b_log_probs, b_actions, b_advantages, b_returns, b_values
        )

    envs.close()
