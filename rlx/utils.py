import mlx.core as mx


def compute_discounted_returns(
    rewards: mx.array,
    dones: mx.array,
    gamma: float = 0.99,
) -> mx.array:
    returns = []
    next_return = mx.array(0.0)

    for t in reversed(range(rewards.shape[0])):
        r = rewards[t]
        d = dones[t]

        next_return = mx.where(d, mx.array(0.0), next_return)
        current_return = r + gamma * next_return

        returns.append(current_return)
        next_return = current_return

    returns = mx.stack(returns[::-1])
    return returns
