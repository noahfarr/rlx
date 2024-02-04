import mlx.core as mx
import mlx.nn as nn


class REINFORCE:
    def __init__(self, policy, optimizer):
        self.policy = policy
        self.optimizer = optimizer
        self.loss_and_grad_fn = nn.value_and_grad(self.policy, self.loss_fn)

    def get_action(self, obs):
        logits = self.policy(obs)
        action = mx.random.categorical(logits)
        return action.item()

    def loss_fn(self, observations, actions, rewards):
        log_probs = self.get_log_probs(observations, actions)
        loss = mx.sum(-log_probs * rewards)
        return loss

    def get_log_probs(self, observations, actions):
        logits = self.policy(observations)
        probs = nn.softmax(logits)[mx.arange(actions.shape[0]), actions]
        log_probs = mx.log(probs)
        return log_probs

    def update(self, observations, actions, rewards):
        loss, grads = self.loss_and_grad_fn(observations, actions, rewards)
        self.optimizer.update(self.policy, grads)
        mx.eval(self.policy.parameters(), self.optimizer.state)
