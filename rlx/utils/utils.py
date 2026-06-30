import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map


def flatten(array: mx.array) -> mx.array:
    return array.reshape(-1, *array.shape[2:])


def soft_update(target_network: nn.Module, online_network: nn.Module, tau: float):
    target_params = tree_map(
        lambda online, target: tau * online + (1 - tau) * target,
        online_network.parameters(),
        target_network.parameters(),
    )
    target_network.update(target_params)


def compute_completed_episode_mask(dones: mx.array) -> mx.array:
    masks = []
    seen_done = mx.zeros(dones.shape[1:])

    for done in reversed(list(dones)):
        seen_done = mx.maximum(seen_done, done)
        masks.append(seen_done)

    masks = mx.stack(masks[::-1])
    return masks


def compute_discounted_returns(
    rewards: mx.array,
    dones: mx.array,
    gamma: float = 0.99,
) -> mx.array:
    returns = []
    next_return = mx.array(0.0)

    for reward, done in reversed(list(zip(rewards, dones))):
        next_return = mx.where(done, mx.array(0.0), next_return)
        current_return = reward + gamma * next_return

        returns.append(current_return)
        next_return = current_return

    returns = mx.stack(returns[::-1])
    return returns


def compute_generalized_advantage_estimate(
    rewards: mx.array,
    values: mx.array,
    terminations: mx.array,
    last_value: mx.array,
    last_termination: mx.array,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> mx.array:
    advantages = []
    next_advantage = mx.array(0.0)
    next_value = last_value
    next_non_terminal = 1.0 - last_termination

    for reward, value, termination in reversed(
        list(zip(rewards, values, terminations))
    ):
        delta = reward + gamma * next_value * next_non_terminal - value
        next_advantage = (
            delta + gamma * gae_lambda * next_non_terminal * next_advantage
        )

        advantages.append(next_advantage)
        next_value = value
        next_non_terminal = 1.0 - termination

    advantages = mx.stack(advantages[::-1])
    return advantages
