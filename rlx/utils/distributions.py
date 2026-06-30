import math

import mlx.core as mx
import mlx.nn as nn


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class Categorical:

    def __init__(self, logits: mx.array):
        self.logits = logits
        self.log_probabilities = nn.log_softmax(logits, axis=-1)

    def sample(self) -> mx.array:
        return mx.random.categorical(self.logits)

    def log_prob(self, actions: mx.array) -> mx.array:
        return mx.take_along_axis(
            self.log_probabilities, actions[..., None], axis=-1
        ).squeeze(-1)

    def entropy(self) -> mx.array:
        probabilities = mx.exp(self.log_probabilities)
        return -mx.sum(probabilities * self.log_probabilities, axis=-1)


class Gaussian:

    def __init__(self, mean: mx.array, log_std: mx.array):
        self.mean = mean
        self.log_std = log_std
        self.std = mx.exp(log_std)

    def sample(self) -> mx.array:
        return self.mean + self.std * mx.random.normal(self.mean.shape)

    def log_prob(self, actions: mx.array) -> mx.array:
        log_prob = (
            -0.5 * mx.square((actions - self.mean) / self.std)
            - self.log_std
            - 0.5 * math.log(2 * math.pi)
        )
        return mx.sum(log_prob, axis=-1)

    def entropy(self) -> mx.array:
        return mx.sum(0.5 + 0.5 * math.log(2 * math.pi) + self.log_std, axis=-1)


class SquashedNormal:

    def __init__(
        self,
        mean: mx.array,
        log_std: mx.array,
        action_scale: mx.array,
        action_bias: mx.array,
    ):
        log_std = mx.tanh(log_std)
        self.log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        self.mean = mean
        self.std = mx.exp(self.log_std)
        self.action_scale = action_scale
        self.action_bias = action_bias

    def sample(self):
        noise = mx.random.normal(self.mean.shape)
        pre_tanh = self.mean + self.std * noise
        squashed = mx.tanh(pre_tanh)
        action = squashed * self.action_scale + self.action_bias

        log_prob = -0.5 * mx.square(noise) - self.log_std - 0.5 * math.log(2 * math.pi)
        log_prob = log_prob - mx.log(
            self.action_scale * (1 - mx.square(squashed)) + 1e-6
        )
        log_prob = mx.sum(log_prob, axis=-1, keepdims=True)

        return action, log_prob

    def mode(self) -> mx.array:
        return mx.tanh(self.mean) * self.action_scale + self.action_bias
