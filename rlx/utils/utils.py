import mlx.core as mx
import mlx.nn as nn

import numpy as np


def get_log_prob(logits, action):
    # Normalize the logits
    logits = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    log_prob: mx.array = nn.log_softmax(logits, axis=-1)

    action = mx.broadcast_to(action, log_prob.shape)
    action = action[..., :1]
    return mx.take_along_axis(log_prob, action, axis=-1).squeeze(-1)


# Define test logits and actions
logits = mx.array(
    [
        [2.0, 1.0, 0.1],
        [1.2, 2.3, 1.5],
        [0.1, 0.3, 0.6],
        [2.0, 1.5, 1.0],
        [1.0, 1.2, 0.8],
    ]
)
action = mx.array([[0], [1], [2], [0], [1]])

# Case 1: Test with action provided
log_prob_with_action = get_log_prob(logits, action)
print("Log probability with action provided:", log_prob_with_action)


import torch
from torch.distributions import Categorical

logits = torch.tensor(
    [
        [2.0, 1.0, 0.1],
        [1.2, 2.3, 1.5],
        [0.1, 0.3, 0.6],
        [2.0, 1.5, 1.0],
        [1.0, 1.2, 0.8],
    ]
)
action = torch.tensor([[0], [1], [2], [0], [1]])
probs = Categorical(logits=logits)
action = probs.sample()
log_prob = probs.log_prob(action)
print("Log probability with torch:", log_prob)
