# 05 -- MNIST MLP

A multi-layer perceptron trained on MNIST handwritten digits using PyTorch.

## Architecture

```
Input (28x28 = 784) --> Linear(784, 100) --> tanh --> Linear(100, 100) --> tanh --> Linear(100, 10) --> logits
```

- 3 layers: 784 --> 100 --> 100 --> 10
- Activation: tanh (no activation on output layer -- CrossEntropyLoss applies softmax internally)
- Optimizer: SGD with momentum (lr=0.001, momentum=0.9)
- Loss: cross-entropy

## Results

- **96.07% test accuracy** after 10 epochs
- 5 epochs reached 94% (below target), doubling to 10 crossed the 95% threshold

## What I learned

- `nn.Linear(nin, nout)` is exactly micrograd's `Layer(nin, nout)` -- it creates and stores the weight matrix and bias internally
- `__init__` defines layers, `forward` wires them -- same pattern as micrograd but with PyTorch handling autograd
- Images arrive as (batch, 1, 28, 28) and must be flattened to (batch, 784) with `x.view(x.size(0), -1)`
- No activation on the output layer when using `nn.CrossEntropyLoss` -- it expects raw logits
- The training loop structure (zero_grad, forward, loss, backward, step) transferred directly from the micrograd training loop

## Running it

Requires PyTorch and torchvision. MNIST downloads automatically on first run.

```bash
pip install torch torchvision matplotlib
jupyter notebook mnist_mlp.ipynb
```
