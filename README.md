# ML From Scratch

Implementations of core machine learning algorithms built from first principles. No frameworks, no autograd -- just NumPy and an understanding of the maths.

Built during self-study alongside an Economics, Finance & Data Science degree at Imperial College London.

## Implementations

| #  | Topic              | Status    | Description                                       |
|----|--------------------|-----------| --------------------------------------------------|
| 01 | NumPy fluency      | Complete  | Exercises covering arrays, broadcasting, vectorised operations, and ML-specific patterns |
| 02 | Gradient descent   | Complete  | Vanilla GD on quadratic and Rosenbrock surfaces with analytical gradients and visualisation |
| 03 | Linear regression  | Complete  | MSE loss, hand-derived gradients, GD training loop, closed-form comparison |

## Structure

```
01-numpy-fluency/       NumPy exercises -- arrays, broadcasting, vectorised ops,
                        loss functions, forward passes, and a mini GD implementation.

02-gradient-descent/    Gradient descent on two test functions (quadratic bowl and
                        Rosenbrock). Demonstrates the core update rule, learning
                        rate sensitivity, and trajectory visualisation.

03-linear-regression/   Linear regression trained with gradient descent on synthetic
                        data. Derives MSE gradients analytically, compares against
                        the closed-form normal equation, and plots loss curves,
                        fit quality, and residual distributions.
```

## Coming soon

- **micrograd** -- a minimal autograd engine following Karpathy's micrograd. Scalar-valued reverse-mode automatic differentiation, backpropagation through arbitrary computation graphs, and a small neural network trained on it.
- **makemore** -- character-level language models. Bigram models, MLPs, and eventually RNNs/transformers, all built incrementally from scratch.

## Requirements

Python 3.8+ with NumPy and Matplotlib. No other dependencies.

```bash
pip install numpy matplotlib
```
