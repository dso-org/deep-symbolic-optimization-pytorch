# PyTorch Refactor

The following repository is designed for generative AI for symbolic regression, i.e., equation discovery from data. The code uses `tensorflow` to deal with neural networks (defining, training, and inference).
In particular, the file `deep-symbolic-optimization/dso/dso/policy/rnn_policy.py` uses an RNN to generate mathematical expressions. TensorFlow is used all over the place in the codebase.

I need to refactor the entire codebase to use `pytorch` instead of `tensorflow`. This includes rewriting the neural network definitions, training loops, and any other relevant parts of the code that rely on TensorFlow functionalities. It means that, at the end, I need to be able to run the code without having `tensorflow` installed, only `pytorch`. I need to create `tests` to ensure that the refactored code works as expected.


Here’s a significantly improved, **Codex-ready prompt** that is clearer, more actionable, and much more likely to produce a high-quality refactor. I’ve focused on: explicit scope, concrete expectations, constraints, and stepwise execution—things Codex responds to very well.

# Create a new branch

```bash
git checkout -b refactor/tensorflow-to-pytorch
```

---

## Improved Prompt for CODEX

# Full TensorFlow → PyTorch Refactor

You are an expert Python ML engineer with deep knowledge of PyTorch, TensorFlow, and reinforcement-learning–based generative models.

## Context

This repository implements **Deep Symbolic Optimization (DSO)** for symbolic regression (equation discovery from data).  
Neural networks are currently implemented using **TensorFlow**, including:
- Model definitions
- Training loops
- Inference logic
- TensorFlow-specific utilities (sessions, placeholders, optimizers, etc.)

In particular, the file:

```
deep-symbolic-optimization/dso/dso/policy/rnn_policy.py
```

implements an RNN-based policy that generates mathematical expressions and is tightly coupled to TensorFlow.

## Objective

Refactor the **entire codebase** to use **PyTorch exclusively**, such that:

- TensorFlow is **completely removed** as a dependency
- The project runs with **only PyTorch installed**
- All neural-network functionality is faithfully preserved

## Scope of Work

1. **Replace TensorFlow with PyTorch**
   - Rewrite all neural network modules using `torch.nn`
   - Replace TensorFlow training logic with idiomatic PyTorch training loops
   - Convert TensorFlow tensors, variables, and operations to PyTorch equivalents
   - Remove sessions, placeholders, graphs, and TensorFlow-specific abstractions

2. **Model Architecture Parity**
   - Preserve architecture details (RNN type, hidden sizes, activations, outputs)
   - Maintain identical input/output semantics where possible
   - Ensure stochastic behavior (e.g., sampling, logits, softmax) matches original intent

3. **Training & Optimization**
   - Use PyTorch optimizers (`torch.optim`)
   - Implement loss computation, backpropagation, and gradient updates explicitly
   - Ensure device support (CPU by default; CUDA optional but cleanly supported)

4. **Inference & Sampling**
   - Ensure expression generation behaves the same as the TensorFlow version
   - Maintain reproducibility controls (random seeds)

5. **Testing**
   - Add unit and/or integration tests using `pytest`
   - Tests should verify:
     - Models initialize and run forward passes correctly
     - Training loops execute without error
     - Output shapes and basic behaviors are correct
   - Tests must run without TensorFlow installed

6. **Code Quality**
   - Follow PyTorch best practices and idiomatic style
   - Keep changes minimal outside TensorFlow-related code
   - Clearly document any behavioral differences or unavoidable deviations

## Deliverables

- Fully refactored PyTorch-based codebase
- Removal of TensorFlow from imports, requirements, and setup files
- New or updated test suite validating correctness
- Brief inline comments explaining major refactoring decisions

## Execution Strategy

Proceed file by file, starting with:
1. Core neural network modules (especially `rnn_policy.py`)
2. Training and optimization logic
3. Inference/sampling utilities
4. Supporting utilities and helpers
5. Tests

If any TensorFlow functionality has no direct PyTorch equivalent, implement a clean, well-documented alternative.
