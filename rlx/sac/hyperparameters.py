# Create a variable for each of the fields of the class
# @dataclass
# class Args:
#     exp_name: str = os.path.basename(__file__)[: -len(".py")]
#     """the name of this experiment"""
#     seed: int = 1
#     """seed of the experiment"""
#     torch_deterministic: bool = True
#     """if toggled, `torch.backends.cudnn.deterministic=False`"""
#     cuda: bool = True
#     """if toggled, cuda will be enabled by default"""
#     track: bool = False
#     """if toggled, this experiment will be tracked with Weights and Biases"""
#     wandb_project_name: str = "cleanRL"
#     """the wandb's project name"""
#     wandb_entity: str = None
#     """the entity (team) of wandb's project"""
#     capture_video: bool = False
#     """whether to capture videos of the agent performances (check out `videos` folder)"""

#     # Algorithm specific arguments
#     env_id: str = "Hopper-v4"
#     """the environment id of the task"""
#     total_timesteps: int = 1000000
#     """total timesteps of the experiments"""
#     buffer_size: int = int(1e6)
#     """the replay memory buffer size"""
#     gamma: float = 0.99
#     """the discount factor gamma"""
#     tau: float = 0.005
#     """target smoothing coefficient (default: 0.005)"""
#     batch_size: int = 256
#     """the batch size of sample from the reply memory"""
#     learning_starts: int = 5e3
#     """timestep to start learning"""
#     policy_lr: float = 3e-4
#     """the learning rate of the policy network optimizer"""
#     q_lr: float = 1e-3
#     """the learning rate of the Q network network optimizer"""
#     policy_frequency: int = 2
#     """the frequency of training policy (delayed)"""
#     target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
#     """the frequency of updates for the target nerworks"""
#     noise_clip: float = 0.5
#     """noise clip parameter of the Target Policy Smoothing Regularization"""
#     alpha: float = 0.2
#     """Entropy regularization coefficient."""
#     autotune: bool = True
#     """automatic tuning of the entropy coefficient"""

import os


exp_name: str = os.path.basename(__file__)[: -len(".py")]
seed: int = 1

# Algorithm specific arguments
env_id: str = "Hopper-v4"
total_timesteps: int = 1000000
buffer_size: int = int(1e6)
gamma: float = 0.99
tau: float = 0.005
batch_size: int = 256
learning_starts: int = 5e3
policy_lr: float = 3e-4
q_lr: float = 1e-3
policy_frequency: int = 2
target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
noise_clip: float = 0.5
alpha: float = 0.2
autotune: bool = False
