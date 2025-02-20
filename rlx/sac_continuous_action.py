import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from rlx.utils.copy_weights import copy_weights


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked wit Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = "noahfarr"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: float = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")  # type: ignore
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)  # type: ignore
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(
                np.array(env.single_observation_space.shape).prod()
                + np.prod(env.single_action_space.shape),
                256,
            ),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def __call__(self, x, a):
        x = mx.concatenate([x, a], 1)
        x = self.critic(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.action_dim = np.prod(env.single_action_space.shape)
        self.actor = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))

        # action rescaling
        self.action_scale = mx.array(
            (env.single_action_space.high - env.single_action_space.low) / 2.0,
            dtype=mx.float32,
        )
        self.action_bias = mx.array(
            (env.single_action_space.high + env.single_action_space.low) / 2.0,
            dtype=mx.float32,
        )

    def __call__(self, x):
        x = self.actor(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = mx.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        x_t = mean + std * mx.random.normal((1, self.action_dim))
        y_t = mx.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = -mx.log(self.action_scale * (1 - y_t.square()))
        # Enforcing Action Bound
        log_prob = log_prob.sum(axis=-1)
        mean = mx.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


def critic_loss_fn(qf1, qf2, next_q_value, data):
    qf1_a_values = qf1(data["obs"], data["actions"]).reshape(-1)
    qf2_a_values = qf2(data["obs"], data["actions"]).reshape(-1)
    qf1_loss = nn.losses.mse_loss(qf1_a_values, next_q_value)
    qf2_loss = nn.losses.mse_loss(qf2_a_values, next_q_value)
    qf_loss = qf1_loss + qf2_loss

    writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
    # writer.add_scalar("losses/qf2_values", qf2_a_values.mean(), global_step)
    # writer.add_scalar("losses/qf1_loss", qf1_loss, global_step)
    # writer.add_scalar("losses/qf2_loss", qf2_loss, global_step)
    # writer.add_scalar("losses/qf_loss", qf_loss / 2.0, global_step)
    return qf_loss


def actor_loss_fn(actor, data, alpha):
    pi, log_pi, _ = actor.get_action(data["obs"])
    qf1_pi = qf1(data["obs"], pi)
    qf2_pi = qf2(data["obs"], pi)
    min_qf_pi = mx.minimum(qf1_pi, qf2_pi)  # type: ignore
    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
    return actor_loss


def alpha_loss_fn(log_alpha, log_prob, target_entropy):
    alpha_loss = (-log_alpha.exp() * (log_prob + target_entropy)).mean()
    writer.add_scalar("losses/alpha", alpha, global_step)
    return alpha_loss


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs)
    mx.eval(actor.parameters())

    qf1 = SoftQNetwork(envs)
    mx.eval(qf1.parameters())
    qf1_target = SoftQNetwork(envs)
    copy_weights(qf1, qf1_target, 1.0)

    qf2 = SoftQNetwork(envs)
    qf2_target = SoftQNetwork(envs)
    copy_weights(qf2, qf2_target, 1.0)

    qf1_optimizer = optim.Adam(learning_rate=args.q_lr)
    qf2_optimizer = optim.Adam(learning_rate=args.q_lr)

    actor_optimizer = optim.Adam(learning_rate=args.policy_lr)

    qf1_loss_and_grad_fn = nn.value_and_grad(qf1, critic_loss_fn)
    qf2_loss_and_grad_fn = nn.value_and_grad(qf2, critic_loss_fn)
    actor_loss_and_grad_fn = nn.value_and_grad(actor, actor_loss_fn)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -mx.prod(mx.array(envs.single_action_space.shape))
        log_alpha = mx.zeros(1)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam(learning_rate=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32  # type: ignore
    rb = ReplayBuffer(
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
                writer.add_scalar(
                    "charts/episodic_return", info["episode"]["r"], global_step
                )
                writer.add_scalar(
                    "charts/episodic_length", info["episode"]["l"], global_step
                )
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)  # type: ignore

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            data = {
                "obs": mx.array(data.observations),
                "actions": mx.array(data.actions),
                "rewards": mx.array(data.rewards),
                "next_obs": mx.array(data.next_observations),
                "dones": mx.array(data.dones),
            }

            next_state_actions, next_state_log_pi, _ = actor.get_action(
                data["next_obs"]
            )
            qf1_next_target = qf1_target(data["next_obs"], next_state_actions).reshape(
                -1
            )
            qf2_next_target = qf2_target(data["next_obs"], next_state_actions).reshape(
                -1
            )
            min_qf_next_target = (
                mx.minimum(qf1_next_target, qf2_next_target)  # type: ignore
                - alpha * next_state_log_pi
            )

            next_q_value = data["rewards"].flatten() + (
                1 - data["dones"].flatten()
            ) * args.gamma * (min_qf_next_target).reshape(-1)

            # optimize the model
            _, grads = qf1_loss_and_grad_fn(qf1, qf2, next_q_value, data)
            qf1_optimizer.update(qf1, grads)
            mx.eval(qf1.parameters(), qf1_optimizer.state)

            _, grads = qf2_loss_and_grad_fn(qf1, qf2, next_q_value, data)
            qf2_optimizer.update(qf2, grads)
            mx.eval(qf2.parameters(), qf2_optimizer.state)

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    _, grads = actor_loss_and_grad_fn(actor, data, alpha)
                    actor_optimizer.update(actor, grads)
                    mx.eval(actor.parameters(), actor_optimizer.state)

                    if args.autotune:
                        _, log_pi, _ = actor.get_action(data["obs"])
                        grads = mx.grad(alpha_loss_fn)(
                            log_alpha, log_pi, target_entropy
                        )
                        log_alpha = a_optimizer.apply_gradients(
                            {"log_alpha": grads}, {"log_alpha": log_alpha}
                        )["log_alpha"]
                        # mx.eval(log_alpha, a_optimizer.state)  # type: ignore
                        alpha = log_alpha.exp().item()  # type: ignore

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                copy_weights(qf1, qf1_target, args.tau)
                copy_weights(qf2, qf2_target, args.tau)

            if global_step % 100 == 0:
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

    envs.close()
    writer.close()
