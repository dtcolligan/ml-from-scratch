"""
Gradient Descent from Scratch
=============================

Vanilla gradient descent on two functions:
  1. A simple quadratic bowl: f(x, y) = x^2 + 4y^2
  2. The Rosenbrock function:  f(x, y) = (a - x)^2 + b(y - x^2)^2

We compute the analytical gradients by hand, step through the parameter
space, and visualise the optimisation trajectory on a contour plot.

No autograd, no frameworks -- just NumPy.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Define the objective functions and their gradients
# ---------------------------------------------------------------------------

def quadratic(x, y):
    """
    f(x, y) = x^2 + 4y^2

    A simple elliptical bowl. The minimum is at (0, 0).
    The factor of 4 on y^2 makes the curvature steeper along y,
    so gradient descent zig-zags if the learning rate is too large.
    """
    return x ** 2 + 4 * y ** 2


def quadratic_grad(x, y):
    """
    Partial derivatives of f(x, y) = x^2 + 4y^2:
        df/dx = 2x
        df/dy = 8y
    """
    return np.array([2 * x, 8 * y])


def rosenbrock(x, y, a=1.0, b=100.0):
    """
    f(x, y) = (a - x)^2 + b * (y - x^2)^2

    Classic non-convex test function. Global minimum at (a, a^2) = (1, 1).
    The minimum sits in a narrow, curved valley -- easy to find the valley,
    hard to converge along it.
    """
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


def rosenbrock_grad(x, y, a=1.0, b=100.0):
    """
    Partial derivatives:
        df/dx = -2(a - x) + b * 2(y - x^2) * (-2x)
              = -2(a - x) - 4bx(y - x^2)
        df/dy = b * 2(y - x^2)
              = 2b(y - x^2)
    """
    dfdx = -2 * (a - x) - 4 * b * x * (y - x ** 2)
    dfdy = 2 * b * (y - x ** 2)
    return np.array([dfdx, dfdy])


# ---------------------------------------------------------------------------
# 2. Gradient descent implementation
# ---------------------------------------------------------------------------

def gradient_descent(grad_fn, x0, lr, n_steps, fn=None):
    """
    Run vanilla gradient descent.

    Parameters
    ----------
    grad_fn : callable
        Takes (x, y) and returns the gradient as a 2-element array.
    x0 : array-like
        Starting point [x, y].
    lr : float
        Learning rate (step size).
    n_steps : int
        Number of gradient descent iterations.
    fn : callable, optional
        Objective function for recording the loss at each step.

    Returns
    -------
    path : ndarray of shape (n_steps + 1, 2)
        The sequence of (x, y) positions visited.
    losses : ndarray of shape (n_steps + 1,)
        The function value at each position (empty if fn is None).
    """
    params = np.array(x0, dtype=float)
    path = [params.copy()]
    losses = []

    if fn is not None:
        losses.append(fn(params[0], params[1]))

    for _ in range(n_steps):
        grad = grad_fn(params[0], params[1])

        # The core update: move opposite to the gradient
        params = params - lr * grad

        path.append(params.copy())
        if fn is not None:
            losses.append(fn(params[0], params[1]))

    return np.array(path), np.array(losses)


# ---------------------------------------------------------------------------
# 3. Visualisation
# ---------------------------------------------------------------------------

def plot_contour_and_path(fn, path, losses, title, xlim, ylim, levels=50):
    """
    Two-panel figure:
        Left  -- contour plot of the surface with the GD trajectory overlaid
        Right -- loss curve over iterations
    """
    fig, (ax_contour, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Contour plot ---
    xs = np.linspace(xlim[0], xlim[1], 400)
    ys = np.linspace(ylim[0], ylim[1], 400)
    X, Y = np.meshgrid(xs, ys)
    Z = fn(X, Y)

    # Use log-spaced levels for Rosenbrock (huge dynamic range)
    if Z.max() / (Z.min() + 1e-12) > 1000:
        level_values = np.logspace(np.log10(Z.min() + 1e-6),
                                   np.log10(Z.max()), levels)
    else:
        level_values = levels

    ax_contour.contour(X, Y, Z, levels=level_values, cmap="viridis", alpha=0.7)
    ax_contour.plot(path[:, 0], path[:, 1],
                    "o-", color="red", markersize=2, linewidth=0.8, alpha=0.85)
    ax_contour.plot(path[0, 0], path[0, 1], "s", color="white",
                    markersize=7, zorder=5, label="start")
    ax_contour.plot(path[-1, 0], path[-1, 1], "*", color="yellow",
                    markersize=12, zorder=5, label="end")
    ax_contour.set_xlabel("x")
    ax_contour.set_ylabel("y")
    ax_contour.set_title(f"{title} -- trajectory")
    ax_contour.legend(loc="upper right")

    # --- Loss curve ---
    ax_loss.plot(losses, color="steelblue", linewidth=1.5)
    ax_loss.set_xlabel("iteration")
    ax_loss.set_ylabel("f(x, y)")
    ax_loss.set_title(f"{title} -- loss")
    ax_loss.set_yscale("log")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Run the experiments
# ---------------------------------------------------------------------------

def main():
    # Experiment 1: Quadratic bowl
    # The elliptical shape means the gradient points away from the minimum
    # direction -- a well-known issue that motivates momentum-based methods.
    path_q, losses_q = gradient_descent(
        grad_fn=quadratic_grad,
        x0=[4.0, 2.0],
        lr=0.08,
        n_steps=60,
        fn=quadratic,
    )

    print("=== Quadratic: f(x,y) = x^2 + 4y^2 ===")
    print(f"Start:  ({path_q[0, 0]:.4f}, {path_q[0, 1]:.4f})  "
          f"f = {losses_q[0]:.4f}")
    print(f"End:    ({path_q[-1, 0]:.4f}, {path_q[-1, 1]:.4f})  "
          f"f = {losses_q[-1]:.6f}")
    print(f"Minimum is at (0, 0), f = 0\n")

    fig1 = plot_contour_and_path(
        quadratic, path_q, losses_q,
        title="Quadratic bowl",
        xlim=(-5, 5), ylim=(-3, 3),
    )

    # Experiment 2: Rosenbrock function
    # Much harder -- the narrow valley makes large learning rates unstable,
    # and small learning rates converge painfully slowly.
    path_r, losses_r = gradient_descent(
        grad_fn=rosenbrock_grad,
        x0=[-1.0, 1.0],
        lr=0.001,
        n_steps=8000,
        fn=rosenbrock,
    )

    print("=== Rosenbrock: f(x,y) = (1-x)^2 + 100(y-x^2)^2 ===")
    print(f"Start:  ({path_r[0, 0]:.4f}, {path_r[0, 1]:.4f})  "
          f"f = {losses_r[0]:.4f}")
    print(f"End:    ({path_r[-1, 0]:.4f}, {path_r[-1, 1]:.4f})  "
          f"f = {losses_r[-1]:.6f}")
    print(f"Minimum is at (1, 1), f = 0\n")

    fig2 = plot_contour_and_path(
        rosenbrock, path_r, losses_r,
        title="Rosenbrock function",
        xlim=(-2, 2), ylim=(-1, 3),
    )

    # Experiment 3: Learning rate sensitivity on the quadratic
    # Show how too-large lr causes divergence, too-small crawls.
    fig3, axes = plt.subplots(1, 3, figsize=(15, 4))
    lrs = [0.01, 0.08, 0.26]
    labels = ["too small (0.01)", "good (0.08)", "too large (0.26)"]

    for ax, lr, label in zip(axes, lrs, labels):
        path, losses = gradient_descent(
            quadratic_grad, x0=[4.0, 2.0], lr=lr, n_steps=60, fn=quadratic
        )

        # Contour background
        xs = np.linspace(-5, 5, 300)
        ys = np.linspace(-3, 3, 300)
        X, Y = np.meshgrid(xs, ys)
        ax.contour(X, Y, quadratic(X, Y), levels=30, cmap="viridis", alpha=0.5)
        ax.plot(path[:, 0], path[:, 1], "o-", color="red",
                markersize=2, linewidth=0.8)
        ax.set_title(f"lr = {label}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig3.suptitle("Effect of learning rate on convergence", fontsize=13)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
