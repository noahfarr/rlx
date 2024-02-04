import mlx.core as mx
import mlx.nn as nn
import numpy as np


class SAC:
    def __init__(
        self,
        actor,
        qf1,
        qf2,
        qf1_target,
        qf2_target,
        actor_optimizer,
        qf1_optimizer,
        qf2_optimizer,
    ):
        self.actor = actor
        self.qf1 = qf1
        self.qf2 = qf2
        self.qf1_target = qf1_target
        self.qf2_target = qf2_target

        self.actor_optimizer = actor_optimizer
        self.qf1_optimizer = qf1_optimizer
        self.qf2_optimizer = qf2_optimizer

        self.actor_loss_and_grad_fn = nn.value_and_grad(actor, self.actor_loss_fn)
        self.qf1_loss_and_grad_fn = nn.value_and_grad(qf1, self.q_loss_fn)
        self.qf2_loss_and_grad_fn = nn.value_and_grad(qf2, self.q_loss_fn)
        self.entropy_loss_and_grad_fn = mx.value_and_grad(self.entropy_loss_fn)

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
        mean, log_std = self.actor(x)
        std = log_std.exp()
        x_t = self.sample_normal(
            mean, std
        )  # for reparameterization trick (mean + std * N(0,1))
        y_t = nn.tanh(x_t)
        action = y_t * self.actor.action_scale + self.actor.action_bias
        log_prob = self.get_log_prob(x_t, mean, std)
        # Enforcing Action Bound
        log_prob -= mx.log(self.actor.action_scale * (1 - y_t.square()) + 1e-6)
        log_prob = log_prob.sum(1, keepdims=True)
        mean = nn.tanh(mean) * self.actor.action_scale + self.actor.action_bias
        return action, log_prob, mean

    def q_loss_fn(self, next_q_value, observations, actions):
        qf1_a_values = self.qf1(observations, actions).reshape(-1)
        qf2_a_values = self.qf2(observations, actions).reshape(-1)
        qf1_loss = nn.losses.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = nn.losses.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        return qf_loss

    def actor_loss_fn(self, observations, alpha):
        pi, log_pi, _ = self.get_action(observations)
        qf1_pi = self.qf1(observations, pi)
        qf2_pi = self.qf2(observations, pi)
        min_qf_pi = mx.minimum(qf1_pi, qf2_pi)
        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()
        return actor_loss

    def entropy_loss_fn(log_alpha, log_prob, target_entropy):
        alpha_loss = (-log_alpha.exp() * (log_prob + target_entropy)).mean()
        return alpha_loss

    def update_q_networks(
        self,
        next_q_value,
        observations,
        actions,
    ):
        _, qf1_grad = self.qf1_loss_and_grad_fn(next_q_value, observations, actions)
        _, qf2_grad = self.qf2_loss_and_grad_fn(next_q_value, observations, actions)
        self.qf1_optimizer.update(self.qf1, qf1_grad)
        self.qf2_optimizer.update(self.qf2, qf2_grad)

        mx.eval(self.qf1.parameters(), self.qf1_optimizer.state)
        mx.eval(self.qf2.parameters(), self.qf2_optimizer.state)

    def update_actor(
        self,
        observations,
        alpha,
    ):
        _, actor_grads = self.actor_loss_and_grad_fn(observations, alpha)
        self.actor_optimizer.update(self.actor, actor_grads)
        mx.eval(self.actor.parameters(), self.actor_optimizer.state)

    def update_entropy(self, log_alpha, log_prob, target_entropy):
        entropy_loss, entropy_grad = self.entropy_loss_and_grad_fn(
            log_alpha, log_prob, target_entropy
        )
        self.alpha_optimizer.update(log_alpha, entropy_grad)
        mx.eval(log_alpha, self.alpha_optimizer.state)
