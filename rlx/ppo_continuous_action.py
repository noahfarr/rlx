import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

import mlx.core as mx
import mlx.optimizers as optim
import mlx.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tyro


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(
            env
        )  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


# TODO: Properly initialize the layers
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, np.prod(envs.single_action_space.shape)),
        )
        self.actor_logstd = mx.zeros((1, np.prod(envs.single_action_space.shape)))
        self.critic = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = mx.broadcast_to(self.actor_logstd, action_mean.shape)
        action_std = mx.exp(action_logstd)

        if action is None:
            action = action_mean + action_std * mx.random.normal(
                shape=action_mean.shape
            )

        # TODO: Proberly compute probs
        probs = self.get_probs(action_mean, action_std)
        log_probs = probs
        entropy = self.get_entropy(probs, log_probs)

        return (
            action,
            self.get_log_prob(log_probs, action),
            entropy,
            self.critic(x),
        )

    def get_probs(self, action_mean, action_std):
        pass

    def get_log_prob(self, log_probs, action):
        action = mx.expand_dims(action, axis=-1).astype(mx.int64)
        log_prob = mx.take_along_axis(log_probs, action, axis=-1).squeeze()
        return log_prob

    def get_entropy(self, probs, log_probs):
        entropy = -mx.sum(probs * log_probs, axis=-1)
        return entropy


def loss_fn(
    agent,
    mb_obs,
    mb_action,
    mb_log_probs,
    mb_advantages,
    mb_values,
    mb_returns,
    ent_coef,
    vf_coef,
    clip_vloss,
    clip_coef,
):
    # Actor loss
    _, new_log_probs, entropy, new_value = agent.get_action_and_value(mb_obs, mb_action)
    log_ratio = new_log_probs - mb_log_probs
    ratio = mx.exp(log_ratio)
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * mx.clip(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = mx.maximum(pg_loss1, pg_loss2).mean()

    # Critic loss
    if clip_vloss:
        v_loss_unclipped = (new_value - mb_returns).square()
        v_clipped = mb_values + mx.clip(new_value - mb_values, -clip_coef, clip_coef)
        v_loss_clipped = (v_clipped - mb_returns).square()
        v_loss_max = mx.maximum(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((new_value - mb_returns).square()).mean()

    # Entropy loss
    entropy_loss = entropy.mean()

    loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)  # type: ignore
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)  # type: ignore
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)  # type: ignore

    return loss


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
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
        [
            make_env(args.env_id, i, args.capture_video, run_name, args.gamma)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    agent = Agent(envs)
    mx.eval(agent.parameters())
    optimizer = optim.Adam(learning_rate=args.learning_rate, eps=1e-5)
    loss_and_grad_fn = nn.value_and_grad(agent, loss_fn)

    # ALGO Logic: Storage setup
    obs = mx.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape  # type: ignore
    )
    actions = mx.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape)
    logprobs = mx.zeros((args.num_steps, args.num_envs))
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
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.learning_rate = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                np.array(action)
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = mx.array(reward).reshape(-1)
            next_obs, next_done = mx.array(next_obs), mx.array(next_done)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        # bootstrap value if not done
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = mx.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = (
                delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            )
        returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_log_probs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                mb_inds = mx.array(mb_inds)

                _, new_log_prob, entropy, new_value = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.astype(mx.int32)[mb_inds]
                )
                log_ratio = new_log_prob - b_log_probs[mb_inds]
                ratio = mx.exp(log_ratio)

                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-log_ratio).mean()
                approx_kl = ((ratio - 1) - log_ratio).mean()
                clipfracs += [
                    ((ratio - 1.0).abs() > args.clip_coef)
                    .astype(mx.float32)
                    .mean()
                    .item()
                ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                _, grads = loss_and_grad_fn(
                    agent,
                    b_obs[mb_inds],
                    b_actions[mb_inds],
                    b_log_probs[mb_inds],
                    mb_advantages,
                    b_advantages[mb_inds],
                    b_returns[mb_inds],
                    args.ent_coef,
                    args.vf_coef,
                    args.clip_vloss,
                    args.clip_coef,
                )
                optimizer.update(agent, grads)
                # TODO: Clip grad norm
                mx.eval(agent.parameters(), optimizer.state)

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = np.array(b_values), np.array(b_returns)
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.learning_rate.item(), global_step
        )
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)  # type: ignore
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)  # type: ignore
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

    envs.close()
    writer.close()
