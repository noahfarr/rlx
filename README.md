# RLX, Reinforcement Learning with MLX

RLX is a collection of Reinforcement Learning algorithms implemented in [MLX](https://github.com/ml-explore/mlx), Apple's array framework. Algorithms follow the single-file, CleanRL-style philosophy, run their environments and learners entirely on device, and use `mx.compile` on the update step to fuse the training graph. On Apple silicon the bundled MLX environments reach well over a million environment steps per second.

## Algorithms

| Algorithm | File | Action space |
| --- | --- | --- |
| DQN | `rlx/algorithms/dqn.py` | discrete |
| REINFORCE | `rlx/algorithms/reinforce.py` | discrete |
| A2C | `rlx/algorithms/a2c.py` | discrete |
| PPO | `rlx/algorithms/ppo.py` | discrete and continuous |
| SAC | `rlx/algorithms/sac.py` | continuous |
| TD3 | `rlx/algorithms/td3.py` | continuous |

## Prerequisites

- Python 3.11 or later
- [uv](https://github.com/astral-sh/uv) for dependency management
- macOS on Apple silicon (Metal) or Linux (CUDA), and the correct MLX backend is selected automatically

## Installation

```bash
git clone https://github.com/noahfarr/rlx.git
cd rlx
uv sync
```

## Project structure

- `rlx/algorithms/` holds the algorithm implementations (`DQN`, `REINFORCE`, `A2C`, `PPO`, `SAC`, `TD3`) and their configs
- `rlx/environments/` holds the vectorized MLX-native `CartPole`, the `Environment` interface, and the `EnvPool` adapter
- `rlx/buffers/` holds `RolloutBuffer` for on-policy and `ReplayBuffer` for off-policy
- `rlx/utils/` holds action distributions, the `Logger`, and shared helpers like GAE, returns, and `soft_update`
- `examples/` holds a runnable training script per algorithm

## Usage

Each algorithm has a runnable example wired with a [tyro](https://github.com/brentyi/tyro) CLI.

```bash
uv run examples/ppo.py
uv run examples/sac.py --env-id Pendulum-v1
uv run examples/ppo.py --ppo.num-envs 8192 --ppo.num-steps 16
```

Experiment level flags such as `--env-id`, `--seed`, `--total-timesteps`, and `--learning-rate` live on the example. Algorithm hyperparameters are nested under the algorithm name, for example `--ppo.gamma` or `--sac.tau`. Add `--help` to any example to see all options.

## Importing as a library

```python
from rlx import PPO, PPOConfig
from rlx.environments import CartPole
```

## Contributing

Contributions are welcome. Fork the repository, create a branch, commit your changes, and open a pull request.

## License

MIT, and the full text is in the LICENSE file.

## Acknowledgments

Thanks to the MLX team for the framework and to CleanRL for the reference implementations this project draws on.
