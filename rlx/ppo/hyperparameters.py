import os

exp_name: str = os.path.basename(__file__)[: -len(".py")]
seed: int = 1

# Algorithm specific arguments
env_id: str = "CartPole-v1"
total_timesteps: int = 500000
learning_rate: float = 2.5e-4
num_envs: int = 4
num_steps: int = 128
anneal_lr: bool = True
gamma: float = 0.99
gae_lambda: float = 0.95
num_minibatches: int = 4
update_epochs: int = 4
norm_adv: bool = True
clip_coef: float = 0.2
clip_vloss: bool = True
ent_coef: float = 0.01
vf_coef: float = 0.5
max_grad_norm: float = 0.5
target_kl: float = None

# to be filled in runtime
batch_size: int = 0
minibatch_size: int = 0
num_iterations: int = 0
