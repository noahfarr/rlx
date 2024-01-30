from dataclasses import dataclass


@dataclass
class RolloutBuffer:
    observations: list[list[float]]
    actions: list[list[int]]
    rewards: list[int]
    terminations: list[int]
    truncations: list[int]

    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.terminations = []
        self.truncations = []
        self.returns = []

    def append(self, obs, next_obs, action, reward, terminated, truncated):
        self.observations.append(obs.tolist())
        if terminated:
            self.observations.append(next_obs.tolist())
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminations.append(int(terminated))
        self.truncations.append(int(truncated))

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.terminations = []
        self.truncations = []
