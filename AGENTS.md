# Repository Guidelines

## Project Structure & Module Organization
All framework code lives under `rlx/`. Each subdirectory (for example `rlx/dqn`, `rlx/ppo`, `rlx/sac`) contains an algorithm-specific `main.py`, its model definitions, and a `hyperparameters.py` module that stores default experiment settings. Shared utilities are centralized in `rlx/common/` (buffers, rollout helpers) and `rlx/utils/` for auxiliary tooling. Keep new assets, notebooks, or evaluation scripts inside dedicated subfolders; avoid placing executable code at the repository root.

## Build, Test, and Development Commands
- `uv sync` — create the virtual environment and install MLX, Gymnasium, and internal packages.
- `uv run python rlx/<algo>/main.py --help` — inspect runnable entry points before launching a training job.
- `uv run python rlx/dqn/main.py` — example command to run DQN with default hyperparameters; swap the algorithm path as needed.
- `uv run pyright` — run static type analysis (configure locally if Pyright is not already installed in the environment).

## Coding Style & Naming Conventions
Use 4-space indentation and keep modules formatted with `black` compatibility in mind (line length ≤ 88). Favor type hints on function signatures, mirroring existing modules. Name files and directories with `snake_case`, classes in `CamelCase`, and constants (such as hyperparameters) in `lower_snake_case`. Hyperparameter modules should expose simple module-level variables so they can be imported seamlessly by `main.py`.

## Testing Guidelines
Add new tests under `tests/` mirrored to each algorithm (e.g., `tests/dqn/test_main.py`). Install `pytest` with `uv add --group dev pytest` and use fixtures to stub Gym environments while seeding via `hyperparameters.py`. For regression checks, record expected episode rewards and keep seeds configurable through CLI flags. Run `uv run pytest` before submitting changes; add coverage gates when experiments run long.

## Commit & Pull Request Guidelines
Commit messages should be short, imperative, and capitalized (examples in history: “Add TD3”, “Update README.md”). Group related changes into a single commit rather than bundling multiple experiments. For pull requests, supply a concise summary of behavior changes, list any new commands or configs, and link tracking issues. Attach logs, tensorboard screenshots, or reward plots when training behavior changes, and call out hyperparameter adjustments in the description.

## Experiment & Configuration Tips
Keep reproducibility in mind: expose every new tunable via `argparse` and default it through the module’s `hyperparameters.py`. When introducing checkpointing or logging, write to an experiment-specific directory under `runs/<exp_name>` to avoid overwriting past results. Document non-default MLX or Gym versions directly in the PR if a newer feature is required.
