# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a PyTorch refactor of the Deep Symbolic Optimization (DSO) framework. DSO uses deep reinforcement learning to discover symbolic mathematical expressions and control policies. The core neural network components run on PyTorch (TensorFlow is not required for basic usage).

**Branch**: `llm_support` - Adds LLM-based config planning features (experimental)

## Development Commands

### Setup
```bash
# Create and activate virtual environment
python3 -m venv venv3
source venv3/bin/activate

# Install DSO package in editable mode
pip install --upgrade setuptools pip
export CFLAGS="-I $(python -c "import numpy; print(numpy.get_include())") $CFLAGS"  # Mac only
pip install -e ./dso  # Core + regression task
pip install -e ./dso[control]  # Add control task dependencies
pip install -e ./dso[all]  # All dependencies
```

### Formatting and Linting
```bash
# Format code (run from repo root)
python -m black .

# Lint code
python -m flake8 dso
```

### Running DSO

**Command-line interface:**
```bash
# Run with config file
python -m dso.run path/to/config.json

# Run a benchmark
python -m dso.run dso/dso/config/config_regression.json --b Nguyen-7

# Batch runs with multiple seeds
python -m dso.run --b=Nguyen-1 --runs=100 --n_cores_task=12 --seed=500
```

**Python interface:**
```python
from dso import DeepSymbolicOptimizer

model = DeepSymbolicOptimizer("path/to/config.json")
model.train()
```

**Sklearn-like interface (regression only):**
```python
from dso import DeepSymbolicRegressor

model = DeepSymbolicRegressor()
model.fit(X, y)
print(model.program_.pretty())
```

**LLM Config Planner (experimental):**
```bash
# Dry-run mode (deterministic, no API calls)
PYTHONPATH=dso python -m dso.scripts.llm_suggest_config \
  --task regression \
  --dataset Nguyen-2 \
  --goal "Fast baseline config with robust priors" \
  --model gpt-4o-mini \
  --output ./config_output.json \
  --dry_run

# LLM mode (requires OpenAI-compatible API)
export OPENAI_API_KEY="<key>"
export OPENAI_API_BASE="https://api.openai.com/v1"  # optional

PYTHONPATH=dso python -m dso.scripts.llm_suggest_config \
  --task regression \
  --dataset Nguyen-2 \
  --goal "Accurate config under moderate compute budget" \
  --model gpt-4o-mini \
  --output ./config_output.json
```

### Testing
```bash
# Run tests from the dso/ subdirectory
cd dso
python -m pytest

# Run specific test
python -m pytest dso/test/test_core.py

# Note: Control task tests require gym/pybullet dependencies
```

## Architecture

### Core Components

**Program (dso/program.py)**
- `Program`: Executable symbolic expression represented as a pre-order traversal of tokens
- Supports constant optimization, polynomial tokens, and memoization via `Program.cache`
- Key class methods: `set_task()`, `set_const_optimizer()`, `set_complexity()`
- Execution can use Cython (fast) or Python fallback

**Library (dso/library.py)**
- `Library`: Collection of available tokens (operators, variables, constants)
- `Token`: Base class for expression building blocks (functions, variables, constants)
- Special tokens: `PlaceholderConstant` (optimized), `Polynomial` (LINEAR/poly token), `StateChecker` (decision trees), `HardCodedConstant`

**Task (dso/task/task.py)**
- Abstract base class defining the optimization problem
- Two main implementations:
  - `dso/task/regression/regression.py`: Symbolic regression on datasets
  - `dso/task/control/control.py`: Symbolic policies for RL environments
- Tasks define reward functions and evaluation metrics

**Policy (dso/policy/policy.py)**
- Abstract `Policy` class: Parametrized distribution over Programs
- Main implementation: `RNNPolicy` (dso/policy/rnn_policy.py) - uses LSTM to generate token sequences
- Sampling generates actions, observations, and prior probabilities
- Managed by `StateManager` for input features

**Policy Optimizer (dso/policy_optimizer/)**
- Optimizes policy parameters to maximize expected reward
- Implementations:
  - `pg_policy_optimizer.py`: Policy gradient (vanilla and risk-seeking)
  - `pqt_policy_optimizer.py`: Priority queue training
  - `ppo_policy_optimizer.py`: Proximal policy optimization
- Risk-seeking PG uses epsilon quantile for baseline (default epsilon=0.05)

**Trainer (dso/train.py)**
- `Trainer`: Main training loop coordinating sampling, evaluation, and optimization
- Handles reward computation (optionally parallelized via multiprocessing.Pool)
- Manages memory replay, early stopping, and GP-meld integration

**Prior (dso/prior.py)**
- `JointPrior`: Combines multiple priors to constrain/guide search
- Built-in priors: length constraints, uniform arity, relational, soft/hard constraints
- Priors adjust token probabilities during sampling

**Core DSO (dso/core.py)**
- `DeepSymbolicOptimizer`: Top-level class orchestrating all components
- Configures and initializes: Task, Library, Prior, Policy, PolicyOptimizer, Trainer
- Handles device management (CPU/CUDA), checkpointing, logging

### Configuration System

All runs are configured via JSON files in `dso/dso/config/`:
- `config_common.json`: Shared defaults
- `config_regression.json`: Regression-specific defaults
- `config_control.json`: Control-specific defaults

Key config sections:
- `task`: Dataset/environment, function set, task-specific params
- `training`: n_samples, batch_size, epsilon, complexity function
- `policy`: max_length, RNN architecture (num_layers, num_units)
- `policy_optimizer`: learning_rate, optimizer type, entropy coefficient
- `prior`: Constraints and priors on search space
- `gp_meld`: Neural-guided genetic programming settings

Custom configs inherit from these defaults and override as needed.

### Execution Flow

1. **Setup**: `DeepSymbolicOptimizer.setup()` initializes all components
2. **Sampling**: Policy generates batch of token sequences → Programs
3. **Evaluation**: Programs executed on data, rewards computed (parallelized)
4. **Optimization**: PolicyOptimizer updates policy parameters based on rewards
5. **Logging**: StatsLogger records statistics (hall of fame, Pareto front, etc.)
6. **Repeat**: Until n_samples reached or early stopping triggered

### Key Design Patterns

**Program Caching**: Programs are memoized by token sequence to avoid redundant computation (for deterministic tasks).

**Lazy Evaluation**: Program properties (reward, complexity, SymPy expression) computed on first access via `@cached_property`.

**Token Optimization**:
- `const` token: Placeholder optimized via scipy/custom optimizer
- `poly` token: Fits polynomial via least squares (fast, recommended)

**Multi-core Reward**: Use `n_cores_batch` to parallelize reward computation across CPU cores.

**State Manager**: Handles observation features for RNN policy (action, parent, sibling, dangling nodes).

**GP-Meld**: Optional genetic programming inner loop for local search (`gp_meld.run_gp_meld=true`).

## Important Implementation Details

### PyTorch vs TensorFlow
- All neural network code uses PyTorch (torch.nn, torch.optim)
- TensorFlow only required for `control` task with anchor models (via stable-baselines)
- Set `logging.save_summary=false` if TensorBoard not installed

### Cython Acceleration
- `cyfunc.pyx` provides fast Program execution
- Falls back to Python if Cython import fails
- Recompile after changes: `pip install -e ./dso`

### Device Management
- Set `experiment.device` in config: "cpu", "cuda", "cuda:0", etc.
- Automatically falls back to CPU if CUDA unavailable
- Device applied to Policy (RNN) and training

### Token Types
- Input variables: `x1`, `x2`, etc. (arity 0, no function)
- Operators: `add`, `sub`, `mul`, `div`, `sin`, `cos`, etc. (arity 1-2)
- `const`: Placeholder constant (optimized)
- `poly`: Polynomial token (LINEAR from NeurIPS 2022 paper)
- Hard-coded constants: `1.0`, `5.0`, etc.
- Decision tree tokens: `StateChecker` (arity 2, for control)

### Function Set Configuration
```json
{
  "task": {
    "function_set": ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "poly"]
  }
}
```
- Including `"const"` enables constant optimization (slower runtime)
- Including `"poly"` enables LINEAR token (recommended, fast)
- See `dso/functions.py` for all supported tokens

### Output Files
After training, `config["experiment"]["logdir"]` contains:
- `dso_<taskname>_<seed>.csv`: Batch-wise statistics
- `dso_<taskname>_<seed>_summary.csv`: Overall summary
- `dso_<taskname>_<seed>_hof.csv`: Hall of fame (best expressions)
- `dso_<taskname>_<seed>_pf.csv`: Pareto front (reward-complexity)
- `config.json`: Complete config used for run

### Custom Tasks and Priors
Specify custom modules in config:
```json
{
  "task": {
    "task_type": "custom_mod.my_source:make_task"
  },
  "prior": {
    "custom_mod.my_source:CustomPrior": {
      "on": true,
      "param": "value"
    }
  }
}
```

### LLM Config Planner
- Located in `dso/llm/config_planner.py` and `dso/scripts/llm_suggest_config.py`
- Generates runnable configs from natural language goals
- Dry-run mode uses deterministic planning (no API)
- LLM mode requires OpenAI-compatible Chat Completions endpoint
- Outputs: planned config JSON + planner report JSON

## Common Patterns

### Adding New Tokens
1. Define function in `dso/functions.py`
2. Add to task's `function_set` in config
3. Library automatically built from function_set

### Debugging Expression Issues
- Check `Program.invalid` flag for numerical errors
- Use `program.pretty()` for readable SymPy expression
- Examine `program.traversal` for token sequence
- `program.print_stats()` shows full diagnostics

### Implementing Custom Reward
Subclass `Task` and override `reward_function(self, program)`:
```python
def reward_function(self, program):
    y_pred = program.execute(self.X)
    # Compute and return reward (higher is better)
    return -np.mean((y_pred - self.y)**2)  # Negative MSE
```

### Working with Checkpoints
```python
model = DeepSymbolicOptimizer(config)
model.train()
model.save("checkpoint.pkl")

# Later
model2 = DeepSymbolicOptimizer(config)
model2.load("checkpoint.pkl")
model2.train()  # Continue training
```

### Batch Runs for Benchmarking
```bash
# Run 100 seeds in parallel on 12 cores
python -m dso.run --b=Nguyen-1 --runs=100 --n_cores_task=12 --seed=500
```
Each run gets a different seed (500, 501, ..., 599).
