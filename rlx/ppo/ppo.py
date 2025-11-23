import mlx.core as mx
import mlx.nn as nn
import numpy as np


class PPO:
    def __init__(self, actor, critic, actor_optimizer, critic_optimizer):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.actor_loss_and_grad_fn = nn.value_and_grad(actor, self.actor_loss_fn)
        self.critic_loss_and_grad_fn = nn.value_and_grad(critic, self.critic_loss_fn)

    def get_action(self, observations):
        logits = self.actor(observations)
        actions = mx.random.categorical(logits)
        return actions

    def get_value(self, observations):
        return self.critic(observations)

    def get_prob(self, observations, actions):
        logits = self.actor(observations)
        probs = nn.softmax(logits)[
            mx.arange(actions.shape[0]), actions.astype(mx.int32)
        ]
        return probs

    def get_log_prob(self, observations, actions):
        probs = self.get_prob(observations, actions)
        log_probs = mx.log(probs)
        return log_probs

    def get_entropy(self, observations):
        logits = self.actor(observations)
        probs = nn.softmax(logits)
        log_probs = nn.log_softmax(logits)
        entropy = -mx.sum(probs * log_probs, axis=1)
        return entropy

    def get_approx_kl(self, observations, actions, old_log_probs):
        log_ratio = self.get_log_prob(observations, actions) - old_log_probs
        ratio = mx.exp(log_ratio)
        approx_kl = mx.mean((ratio - 1) - log_ratio)
        return approx_kl

    def actor_loss_fn(
        self,
        observations,
        actions,
        old_log_probs,
        advantages,
        entropy_coef,
    ):
        probs = self.get_prob(observations, actions)
        log_probs = mx.log(probs)
        ratios = mx.exp(log_probs - old_log_probs)
        entropy = self.get_entropy(observations)
        loss = mx.minimum(
            ratios * advantages,
            mx.clip(ratios, 1 - 0.2, 1 + 0.2) * advantages,
        )
        entropy_loss = entropy_coef * entropy.mean()
        return -mx.mean(loss) - entropy_loss

    def critic_loss_fn(self, values, observations, returns, clip_vloss, clip_coef):
        new_values = self.get_value(observations).squeeze()

        if clip_vloss:
            v_loss_unclipped = (new_values - returns).square()
            v_clipped = values + mx.clip(
                new_values - values,
                -clip_coef,
                clip_coef,
            )
            v_loss_clipped = (v_clipped - returns).square()
            v_loss_max = mx.maximum(v_loss_unclipped, v_loss_clipped)
            loss = 0.5 * v_loss_max.mean()
        else:
            loss = nn.losses.mse_loss(values, returns)
        return loss

    def update(
        self,
        args,
        batch_observations,
        batch_log_probs,
        batch_actions,
        batch_advantages,
        batch_returns,
        batch_values,
    ):
        batch_indices = np.arange(args.batch_size)
        for _ in range(args.update_epochs):
            np.random.shuffle(batch_indices)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                minibatch_indices = batch_indices[start:end]
                minibatch_indices = mx.array(minibatch_indices)

                approx_kl = self.get_approx_kl(
                    batch_observations[minibatch_indices],
                    batch_actions[minibatch_indices],
                    batch_log_probs[minibatch_indices],
                )

                minibatch_advantages = batch_advantages[minibatch_indices]

                if args.norm_adv:
                    minibatch_advantages = (
                        minibatch_advantages - minibatch_advantages.mean()
                    ) / mx.sqrt((minibatch_advantages.var()) + 1e-8)

                _, actor_grads = self.actor_loss_and_grad_fn(
                    batch_observations[minibatch_indices],
                    batch_actions[minibatch_indices],
                    batch_log_probs[minibatch_indices],
                    minibatch_advantages,
                    args.ent_coef,
                )

                _, critic_grads = self.critic_loss_and_grad_fn(
                    batch_values[minibatch_indices],
                    batch_observations[minibatch_indices],
                    batch_returns[minibatch_indices],
                    args.clip_vloss,
                    args.clip_coef,
                )
                self.actor_optimizer.update(self.actor, actor_grads)
                mx.eval(self.actor.parameters(), self.actor_optimizer.state)
                self.critic_optimizer.update(self.critic, critic_grads)
                mx.eval(self.critic.parameters(), self.critic_optimizer.state)

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
