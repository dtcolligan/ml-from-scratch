# ml-from-scratch

Core machine learning pieces implemented from first principles, in NumPy first and PyTorch after.

- 01 NumPy fluency. Arrays, broadcasting, vectorised loss and gradient exercises, a line fitted by gradient descent. Notebook.
- 02 Gradient descent. Hand-derived gradients on a quadratic bowl and the Rosenbrock function, a learning-rate sweep, contour plots.
- 03 Linear regression. MSE, analytical gradients, a training loop, checked against the normal equation.
- 04 Micrograd. A scalar autograd engine with reverse-mode backpropagation and an MLP trained on it, in pure Python, after Karpathy.
- 05 MNIST MLP. A three-layer network in PyTorch, 96% test accuracy. Notebook.
- 06 Character-level language model. A bigram model in PyTorch trained on a short list of names and sampled autoregressively.

Install with `pip install -r requirements.txt`. Run a script with `python <folder>/<file>.py`; notebooks run in Jupyter. MNIST downloads on first run.
