import os

exp_name: str = os.path.basename(__file__)[: -len(".py")]
seed: int = 1

# Algorithm specific arguments
env_id: str = "CartPole-v1"
total_timesteps: int = 500000
learning_rate: float = 2.5e-4
num_envs: int = 1
buffer_size: int = 10000
gamma: float = 0.99
tau: float = 1.0
target_network_frequency: int = 500
batch_size: int = 128
start_e: float = 1
end_e: float = 0.05
exploration_fraction: float = 0.5
learning_starts: int = 10000
train_frequency: int = 10
