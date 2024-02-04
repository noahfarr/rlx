import os

exp_name: str = os.path.basename(__file__)[: -len(".py")]
seed: int = 1

# Algorithm specific arguments
env_id: str = "Pendulum-v1"
total_timesteps: int = 1000000
learning_rate: float = 3e-4
buffer_size: int = int(1e6)
gamma: float = 0.99
tau: float = 0.005
batch_size: int = 256
policy_noise: float = 0.2
exploration_noise: float = 0.1
learning_starts: int = 25e3
policy_frequency: int = 2
noise_clip: float = 0.5
