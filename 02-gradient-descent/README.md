# Gradient Descent from Scratch

Vanilla gradient descent implemented in NumPy with no autograd or frameworks.

## What this covers

The script optimises two classical test functions using hand-derived analytical gradients:

1. **Quadratic bowl** -- `f(x, y) = x^2 + 4y^2`. A convex elliptical surface with a known minimum at the origin. The asymmetric curvature causes the characteristic zig-zag behaviour that motivates momentum-based optimisers.

2. **Rosenbrock function** -- `f(x, y) = (1 - x)^2 + 100(y - x^2)^2`. A non-convex function with a narrow curved valley. Finding the valley is easy; converging along it to the global minimum at (1, 1) is the hard part.

The third visualisation shows how the learning rate affects convergence: too small and the optimiser crawls, too large and it oscillates or diverges.

## Running

```bash
python gradient_descent.py
```

Requires `numpy` and `matplotlib`.

## Key ideas

- The gradient of a scalar function points in the direction of steepest ascent. We step in the opposite direction.
- The learning rate controls step size. It is the single most important hyperparameter.
- Vanilla GD struggles on ill-conditioned surfaces (different curvature along different axes). This is why methods like SGD with momentum, RMSProp, and Adam exist.
