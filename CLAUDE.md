# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PyTorch refactor of the Deep Symbolic Optimization (DSO) framework. DSO uses deep reinforcement learning to discover symbolic mathematical expressions and control policies. All neural network components use PyTorch; TensorFlow is only needed for control task anchor models (via stable-baselines).

## Development Commands

### Setup
```bash
python3 -m venv venv3 && source venv3/bin/activate
pip install --upgrade "setuptools<81" pip
export CFLAGS="-I $(python -c "import numpy; print(numpy.get_include())") $CFLAGS"  # Mac only
pip install -e ./dso              # Core + regression task
pip install -e ./dso[control]     # Add control task dependencies
pip install -e ./dso[all]         # All dependencies
```

### Formatting and Linting
```bash
python -m black .        # Format (run from repo root)
python -m flake8 dso     # Lint
```

### Testing
```bash
cd dso && python -m pytest                        # All tests
python -m pytest dso/test/test_core.py            # Single test file
python -m pytest dso/test/test_skeleton.py -k "test_parse"  # Single test
```
Control task tests require gym/pybullet dependencies.

### Running DSO
```bash
python -m dso.run dso/dso/config/config_regression.json --b Nguyen-7     # Single benchmark
python -m dso.run --b=Nguyen-1 --runs=100 --n_cores_task=12 --seed=500  # Batch runs
```

### LLM Config Planner (experimental)
```bash
# Dry-run mode (deterministic, no API calls)
PYTHONPATH=dso python -m dso.scripts.llm_suggest_config \
  --task regression --dataset Nguyen-2 \
  --goal "Fast baseline config with robust priors" \
  --model gpt-4o-mini --output ./config_output.json --dry_run

# LLM mode (requires OPENAI_API_KEY env var)
PYTHONPATH=dso python -m dso.scripts.llm_suggest_config \
  --task regression --dataset Nguyen-2 \
  --goal "Accurate config under moderate compute budget" \
  --model gpt-4o-mini --output ./config_output.json
```

## Architecture

### Execution Flow

1. `DeepSymbolicOptimizer.setup()` initializes all components
2. Policy (LSTM) generates batch of token sequences -> Programs
3. Programs executed on data, rewards computed (optionally parallelized via `n_cores_batch`)
4. PolicyOptimizer updates policy parameters based on rewards
5. StatsLogger records statistics (hall of fame, Pareto front)
6. Repeat until `n_samples` reached or early stopping

### Core Components (all under `dso/dso/`)

| Component | Files | Role |
|-----------|-------|------|
| **Orchestrator** | `core.py` | `DeepSymbolicOptimizer` - top-level class, config, device management |
| **Program** | `program.py` | Executable symbolic expression as pre-order token traversal. Memoized by token sequence (`Program.cache`). Properties (reward, complexity) are lazy via `@cached_property` |
| **Library** | `library.py`, `functions.py` | `Library` of available `Token`s (operators, variables, constants). Built from config's `function_set` |
| **Task** | `task/task.py` | Abstract base. Implementations: `task/regression/regression.py`, `task/control/control.py` |
| **Policy** | `policy/policy.py`, `policy/rnn_policy.py` | `RNNPolicy` uses LSTM to generate token sequences. `StateManager` handles input features (action, parent, sibling, dangling) |
| **PolicyOptimizer** | `policy_optimizer/` | `pg_policy_optimizer.py` (risk-seeking PG, default), `pqt_policy_optimizer.py` (priority queue), `ppo_policy_optimizer.py` (PPO) |
| **Trainer** | `train.py` | Training loop: sampling, evaluation, optimization, memory replay, early stopping |
| **Prior** | `prior.py` | `JointPrior` combines multiple priors to constrain token probabilities during sampling |
| **Skeleton** | `skeleton.py` | Template expressions with optimizable coefficients (see below) |
| **Cython** | `cyfunc.pyx` | Fast Program execution; falls back to Python. Recompile with `pip install -e ./dso` |

### Skeleton Expressions

Skeleton expressions allow defining mathematical templates with optimizable coefficients:
```json
{
  "task": {
    "skeleton_expressions": [
      {"name": "fourier1", "expression": "a*cos(x1) + b*sin(x1) + c"},
      {"name": "exp_decay", "expression": "a*exp(-b*x1) + c"}
    ]
  }
}
```

Key classes in `skeleton.py`:
- `parse_skeleton()` - parses expression string into structured metadata
- `SkeletonExpression` - token class integrating with Library/Program system
- `SkeletonOptimizer` - fits coefficients (linear: `numpy.linalg.lstsq`, nonlinear: `scipy.optimize.minimize`)

Example configs: `dso/dso/config/examples/regression/skeleton_*.json`

### Configuration System

JSON configs in `dso/dso/config/`. Custom configs inherit from defaults:
- `config_common.json` - shared defaults
- `config_regression.json` - regression defaults
- `config_control.json` - control defaults

Key sections: `task`, `training`, `policy`, `policy_optimizer`, `prior`, `gp_meld`, `experiment`, `logging`.

### Special Token Types

| Token | Description | Performance Impact |
|-------|-------------|-------------------|
| `const` | Placeholder constant optimized via scipy inner loop | Slow (~hours) |
| `poly` | LINEAR polynomial token, least-squares fit | Fast (recommended) |
| Skeleton tokens | Template expressions with coefficient fitting | Varies by template |
| `StateChecker` | Decision tree nodes for control task (arity 2) | - |
| Hard-coded (e.g. `1.0`) | Fixed constants | None |

### GP-Meld

Optional genetic programming inner loop for local search. Enable with `"gp_meld": {"run_gp_meld": true}` in config.

## Important Details

- Set `logging.save_summary=false` if TensorBoard not installed
- `experiment.device` in config: `"cpu"`, `"cuda"`, `"cuda:0"` (auto-falls back to CPU)
- Custom tasks/priors: use `"module.source:function"` format in config (e.g., `"task_type": "custom_mod.my_source:make_task"`)
- LLM Config Planner: `dso/llm/config_planner.py` + `dso/scripts/llm_suggest_config.py`. Dry-run mode uses deterministic planning; LLM mode needs OpenAI-compatible endpoint
- Output files in logdir: `*_hof.csv` (hall of fame), `*_pf.csv` (Pareto front), `*.csv` (batch stats), `*_summary.csv`, `config.json`
