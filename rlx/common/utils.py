import numpy as np
import mlx.core as mx


def get_discounted_sum_of_rewards(rewards, gamma):
    return mx.array(
        [
            np.sum(rewards[i:] * (gamma ** np.array(range(i, len(rewards)))))
            for i in range(len(rewards))
        ]
    )
