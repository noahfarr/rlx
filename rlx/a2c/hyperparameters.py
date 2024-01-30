import mlx.nn as nn
import mlx.optimizers as optim

# General Parameters:
env_id = "CartPole-v1"
gamma = 0.99
total_timesteps = 500_000
render_mode = None
seed = 0

# Actor Network:
actor_num_layers = 1
actor_hidden_dim = 128
actor_activations = [nn.relu]
actor_learning_rate = 1e-3
actor_optimizer = optim.Adam

# Critic Network:
critic_num_layers = 1
critic_hidden_dim = 128
critic_activations = [nn.relu]
critic_learning_rate = 1e-3
critic_optimizer = optim.Adam
