import mlx.core as mx
import mlx.nn as nn
import numpy as np

from rlx.common.mlp import MLP


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

    def get_log_prob(self, observations, actions):
        logits = self.actor(observations)
        probs = nn.softmax(logits)[
            mx.arange(actions.shape[0]), actions.astype(mx.int32)
        ]
        log_probs = mx.log(probs)
        return log_probs

    def actor_loss_fn(
        self,
        observations,
        actions,
        old_log_probs,
        advantages,
    ):
        log_probs = self.get_log_prob(observations, actions)
        ratios = mx.exp(log_probs - old_log_probs)
        loss = mx.minimum(
            ratios * advantages,
            mx.clip(ratios, 1 - 0.2, 1 + 0.2) * advantages,
        )
        return -mx.mean(loss)

    def critic_loss_fn(self, observations, rewards_to_go):
        values = self.get_value(observations).squeeze()
        loss = nn.losses.mse_loss(values, rewards_to_go)
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
        for epoch in range(args.update_epochs):
            np.random.shuffle(batch_indices)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                minibatch_indices = batch_indices[start:end]
                minibatch_indices = mx.array(minibatch_indices)

                actor_loss, actor_grads = self.actor_loss_and_grad_fn(
                    batch_observations[minibatch_indices],
                    batch_actions[minibatch_indices],
                    batch_log_probs[minibatch_indices],
                    batch_advantages[minibatch_indices],
                )

                critic_loss, critic_grads = self.critic_loss_and_grad_fn(
                    batch_observations[minibatch_indices],
                    batch_returns[minibatch_indices],
                )
                self.actor_optimizer.update(self.actor, actor_grads)
                mx.eval(self.actor.parameters(), self.actor_optimizer.state)
                self.critic_optimizer.update(self.critic, critic_grads)
                mx.eval(self.critic.parameters(), self.critic_optimizer.state)
