import mlx.core as mx
import mlx.nn as nn


class A2C:
    def __init__(self, actor, critic, actor_optimizer, critic_optimizer):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.actor_loss_and_grad_fn = nn.value_and_grad(actor, self.actor_loss_fn)
        self.critic_loss_and_grad_fn = nn.value_and_grad(critic, self.critic_loss_fn)

    def get_action(self, obs):
        logits = self.actor(obs)
        action = mx.random.categorical(logits)
        return action.item()

    def get_values(self, observations):
        return self.critic(observations)

    def get_advantages(self, observations, rewards, terminations):
        values = self.get_values(observations[:-1])
        next_values = self.get_values(observations[1:])
        advantages = (
            mx.array(rewards)
            + (mx.ones_like(terminations) - terminations) * next_values
            - values
        )
        return advantages

    def get_log_probs(self, observations, actions):
        logits = self.actor(observations)
        probs = nn.softmax(logits)[mx.arange(len(actions)), actions]
        log_probs = mx.log(probs)
        return log_probs

    def actor_loss_fn(self, observations, actions, advantages):
        log_probs = self.get_log_probs(observations, actions)
        loss = mx.sum(-log_probs * advantages)
        return loss

    def critic_loss_fn(self, observations, rewards, terminations):
        advantages = self.get_advantages(observations, rewards, terminations)
        loss = advantages.square().mean()
        return loss

    def update(self, observations, actions, rewards, terminations):
        advantages = self.get_advantages(observations, rewards, terminations)
        actor_loss, actor_grads = self.actor_loss_and_grad_fn(
            observations, actions, advantages
        )
        critic_loss, critic_grads = self.critic_loss_and_grad_fn(
            observations, rewards, terminations
        )
        self.actor_optimizer.update(self.actor, actor_grads)
        mx.eval(self.actor.parameters(), self.actor_optimizer.state)
        self.critic_optimizer.update(self.critic, critic_grads)
        mx.eval(self.critic.parameters(), self.critic_optimizer.state)
