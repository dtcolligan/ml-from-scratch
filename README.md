# ML From Scratch

Implementations of core machine learning algorithms, progressing from NumPy fundamentals through a from-scratch autograd engine to PyTorch neural networks.

Built during self-study alongside an Economics, Finance & Data Science degree at Imperial College London.

## Implementations

| #  | Topic              | Status    | Description                                       |
|----|--------------------|-----------| --------------------------------------------------|
| 01 | NumPy fluency      | Complete  | Exercises covering arrays, broadcasting, vectorised operations, and ML-specific patterns |
| 02 | Gradient descent   | Complete  | Vanilla GD on quadratic and Rosenbrock surfaces with analytical gradients and visualisation |
| 03 | Linear regression  | Complete  | MSE loss, hand-derived gradients, GD training loop, closed-form comparison |
| 04 | Micrograd          | Complete  | Scalar-valued autograd engine with backpropagation through computation graphs, plus a small neural network trained on it |
| 05 | MNIST MLP          | Complete  | Multi-layer perceptron in PyTorch classifying handwritten digits (96% accuracy) |

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

04-micrograd/           A minimal autograd engine. Scalar-valued reverse-mode
                        automatic differentiation, topological sort for gradient
                        propagation, and a 3-layer MLP trained to convergence.

05-mnist-mlp/           MLP in PyTorch on MNIST. 784 -> 100 -> 100 -> 10 with tanh
                        activations, SGD with momentum, 96% test accuracy.
```

## Coming soon

- **makemore** -- character-level language models. Bigram models, MLPs, and eventually RNNs/transformers, all built incrementally from scratch.
- **transformer** -- attention mechanism and transformer architecture built from scratch in PyTorch.

## Requirements

Python 3.8+ with NumPy and Matplotlib. 05-mnist-mlp also requires PyTorch and torchvision.

```bash
pip install numpy matplotlib torch torchvision
```
