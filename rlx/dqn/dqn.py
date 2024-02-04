import mlx.core as mx
import mlx.nn as nn


class DQN:
    def __init__(self, q_network, target_network, optimizer):
        self.q_network = q_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.loss_and_grad_fn = nn.value_and_grad(q_network, self.loss_fn)

    def loss_fn(self, td_target, observations, actions):
        old_value = mx.take_along_axis(
            self.q_network(observations),
            actions,
            1,
        ).squeeze()
        loss = nn.losses.mse_loss(td_target, old_value)
        return loss

    def update(self, data, args):
        target_max = self.target_network(mx.array(data.next_observations.numpy())).max(
            axis=1
        )
        td_target = mx.array(
            data.rewards.flatten().numpy()
        ) + args.gamma * target_max * (1 - mx.array(data.dones.flatten().numpy()))

        observations = mx.array(data.observations.numpy())
        actions = mx.array(data.actions.numpy())

        _, grads = self.loss_and_grad_fn(
            td_target,
            observations,
            actions,
        )
        self.optimizer.update(self.q_network, grads)
        mx.eval(self.q_network.parameters(), self.optimizer.state)
