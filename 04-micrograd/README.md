# 04 — Micrograd

A scalar-valued autograd engine and neural network library built from scratch. Follows Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) and his Neural Networks: Zero to Hero series.

## What's here

`micrograd.py` — complete implementation including:

- **Value class** — wraps scalars to track a computational DAG. Each arithmetic operation (`+`, `*`, `**`, `tanh`, `relu`) creates a new node and attaches a `_backward` closure encoding the local chain rule
- **backward()** — topological sort via recursive DFS, then reverse-order gradient accumulation. Each node's `_backward` closure pushes gradients to its children
- **Neuron / Layer / MLP** — simple neural network built on top of Value. Forward pass computes `tanh(w*x + b)`, parameters collected for gradient updates
- **Training loop** — forward pass, MSE loss, zero gradients, backward pass, parameter update. Trains a 3-input, 2-hidden-layer MLP to convergence

## Key concepts

- Autograd via closures: each operation defines its gradient rule at creation time and stores it as a function attribute on the output node
- Topological ordering ensures gradients propagate in the correct (reverse dependency) order
- `+=` for gradient accumulation handles nodes that feed into multiple downstream operations

## Running it

```bash
python micrograd.py
```

Trains for 100 steps and prints loss + final predictions vs targets.
