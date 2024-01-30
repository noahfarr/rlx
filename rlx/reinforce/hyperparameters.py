import mlx.nn as nn
import mlx.optimizers as optim

# General Parameters:
env_id = "CartPole-v1"
gamma = 0.99
total_timesteps = 500_000
render_mode = None
seed = 0

# Policy Network:
num_layers = 1
hidden_dim = 128
activations = [nn.relu]
learning_rate = 1e-3
optimizer = optim.Adam
