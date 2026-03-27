# ML From Scratch

A structured machine learning project built from first principles, progressing from NumPy fundamentals through gradient descent, linear regression, autograd, and neural networks.

Built during self-study alongside an Economics, Finance & Data Science degree at Imperial College London.

## Highlights

- Implemented core machine learning building blocks from scratch instead of relying only on high-level libraries
- Built a **micrograd-style autograd engine** with reverse-mode backpropagation
- Implemented and trained an **MNIST multilayer perceptron** achieving ~96% test accuracy
- Used the project to build intuition for optimisation, gradients, computation graphs, and neural network training
- Created a clear progression from mathematical foundations to practical deep learning implementation

## What this project demonstrates

This repo is designed to show:
- fluency with **NumPy-based numerical computing**
- understanding of **gradient descent and optimisation dynamics**
- ability to derive and implement **linear regression from first principles**
- understanding of **automatic differentiation and backpropagation**
- ability to move from low-level intuition to higher-level neural network training in PyTorch

## Implementations

| #  | Topic | Status | What it shows |
|----|-------|--------|---------------|
| 01 | NumPy fluency | Complete | Arrays, broadcasting, vectorised operations, and ML-oriented numerical patterns |
| 02 | Gradient descent | Complete | Optimisation behaviour, learning-rate sensitivity, and trajectory visualisation |
| 03 | Linear regression | Complete | MSE loss, analytical gradients, GD training loop, and closed-form comparison |
| 04 | Micrograd | Complete | Reverse-mode autograd, computation graphs, backpropagation, and a small neural net |
| 05 | MNIST MLP | Complete | End-to-end neural network training in PyTorch with ~96% accuracy |
| 06 | Char-level LM | Complete | Next-character prediction, autoregressive generation, and the bridge into language modelling |

## Project structure

```text
01-numpy-fluency/     NumPy exercises and ML-oriented numerical patterns
02-gradient-descent/  Gradient descent on test functions with visualisation
03-linear-regression/ Linear regression with analytical gradients and diagnostics
04-micrograd/         Minimal autograd engine + small neural network
05-mnist-mlp/         PyTorch MLP on MNIST
06-char-level-lm/     Character-level language modelling and autoregressive text generation
```

## Quickstart

### Requirements

- Python 3.8+
- NumPy
- Matplotlib
- PyTorch
- torchvision

### Install

```bash
pip install numpy matplotlib torch torchvision
```

### Run

Open a project folder and run the relevant notebook or script for that section.

## Why this repo matters

This is not just a collection of notebooks. It is a learning-and-implementation track designed to build real intuition for how machine learning systems work under the hood.

For portfolio purposes, the strongest signal is that the work moves from:
1. numerical foundations,
2. to optimisation,
3. to analytical modelling,
4. to autograd,
5. to neural network training.

## Current status

### Shipped
- NumPy foundations
- Gradient descent
- Linear regression
- Micrograd-style autograd engine
- MNIST MLP
- Character-level language model

### Next likely extensions
- MLP-based next-character model
- makemore-style progression beyond bigrams
- transformer fundamentals

## Notes

The goal of this repo is depth, not breadth. Each section is intended to make the mechanics of machine learning more legible by implementing them directly.

The latest addition extends the repo into sequence modelling, so the progression now reaches beyond classification and into the basic mechanics of language models.
