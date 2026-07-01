<div align="center">

# 🦾 RLX

**Reinforcement learning that runs end-to-end on [MLX](https://github.com/ml-explore/mlx), Apple's array framework.**

Single-file, CleanRL-style algorithms and vectorized environments that live entirely on device: over a million environment steps per second on Apple silicon.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![MLX](https://img.shields.io/badge/backend-MLX-black.svg)](https://github.com/ml-explore/mlx)
[![Status: Production/Stable](https://img.shields.io/badge/status-stable-brightgreen.svg)](https://github.com/noahfarr/rlx)

</div>

---

## Why RLX

- **On-device, end-to-end.** Environments, buffers, and the learner all run on the GPU, with no host round-trips between steps.
- **Fused updates.** Every update step is wrapped in `mx.compile`, fusing the training graph for maximum throughput.
- **Fast.** The bundled MLX-native environments reach well over a million environment steps per second on Apple silicon.
- **Readable.** Each algorithm is a single, self-contained, CleanRL-style file you can read top to bottom.
- **Portable.** The correct MLX backend (Metal on Apple silicon, CUDA on Linux) is selected automatically.

## Algorithms

| Algorithm | File | Action space |
| --- | --- | --- |
| DQN | `rlx/algorithms/dqn.py` | discrete |
| REINFORCE | `rlx/algorithms/reinforce.py` | discrete |
| A2C | `rlx/algorithms/a2c.py` | discrete |
| PPO | `rlx/algorithms/ppo.py` | discrete & continuous |
| SAC | `rlx/algorithms/sac.py` | continuous |
| TD3 | `rlx/algorithms/td3.py` | continuous |

## Environments

Every environment is MLX-native and vectorized, so resets, steps, and the learner all run on device. Classic-control
environments import from `rlx.environments`; the MinAtar suite lives under `rlx.environments.minatar`. The MinAtar ports
are branchless and validated against the reference [`minatar`](https://github.com/kenjyoung/MinAtar) package.

| Environment | Import | Action space |
| --- | --- | --- |
| CartPole | `rlx.environments.CartPole` | discrete |
| MountainCar | `rlx.environments.MountainCar` | discrete |
| Acrobot | `rlx.environments.Acrobot` | discrete |
| Pendulum | `rlx.environments.Pendulum` | continuous |
| MinAtar Breakout | `rlx.environments.minatar.Breakout` | discrete |
| MinAtar Freeway | `rlx.environments.minatar.Freeway` | discrete |
| MinAtar SpaceInvaders | `rlx.environments.minatar.SpaceInvaders` | discrete |
| MinAtar Asterix | `rlx.environments.minatar.Asterix` | discrete |
| MinAtar Seaquest | `rlx.environments.minatar.Seaquest` | discrete |

Need something outside these suites? The `EnvPool` adapter wraps a pre-vectorized
[EnvPool](https://github.com/sail-sg/envpool) environment behind the same `Environment` interface.

## Quickstart

**Requirements:** Python 3.11+, [uv](https://github.com/astral-sh/uv), and macOS on Apple silicon (Metal) or Linux (CUDA).

```bash
git clone https://github.com/noahfarr/rlx.git
cd rlx
uv sync
```

Each algorithm ships with a runnable example wired to a [tyro](https://github.com/brentyi/tyro) CLI:

```bash
uv run examples/ppo_cartpole.py
uv run examples/sac_pendulum.py

# Scale the vectorized rollout right from the CLI
uv run examples/ppo_cartpole.py --ppo.num-envs 8192 --ppo.num-steps 16
```

Experiment-level flags (`--seed`, `--total-timesteps`, `--learning-rate`, `--track`) live on the example.
Algorithm hyperparameters are nested under the algorithm name, e.g. `--ppo.gamma` or `--sac.tau`. Append `--help` to
any example to see every option.

## Use it as a library

```python
from rlx.algorithms import PPO, PPOConfig
from rlx.environments import CartPole
from rlx.environments.minatar import Breakout
```

The top-level `rlx` package also re-exports the core algorithms and their configs (`from rlx import PPO, PPOConfig`).

## Project layout

| Path | What's inside |
| --- | --- |
| `rlx/algorithms/` | Algorithm implementations (DQN, REINFORCE, A2C, PPO, SAC, TD3) and their configs |
| `rlx/environments/` | MLX-native environments (`classic_control/` and `minatar/`) plus the `Environment` interface and `EnvPool` adapter |
| `rlx/buffers/` | `RolloutBuffer` (on-policy) and `ReplayBuffer` (off-policy) |
| `rlx/utils/` | Action distributions, the `Logger`, and shared helpers (GAE, returns, `soft_update`) |
| `examples/` | One runnable training script per algorithm |

## Contributing

Contributions are welcome! Fork the repo, create a branch, commit your changes, and open a pull request.

## License

Released under the MIT License. See [LICENSE](LICENSE) for the full text.

## Acknowledgments

Thanks to the [MLX](https://github.com/ml-explore/mlx) team for the framework and to
[CleanRL](https://github.com/vwxyzjn/cleanrl) for the reference implementations this project draws on.
