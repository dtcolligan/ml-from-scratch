"""
Linear Regression from Scratch
===============================

Fit y = Xw + b to synthetic data using gradient descent on the MSE loss.
Everything is computed by hand in NumPy -- no sklearn, no autograd.

The script:
  1. Generates a synthetic dataset with known ground-truth weights
  2. Derives and implements the MSE gradients analytically
  3. Runs gradient descent to recover the weights
  4. Plots the loss curve and the final fit

Supports arbitrary input dimensionality, but the visualisation is designed
for a single-feature case so you can see the regression line.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Generate synthetic data
# ---------------------------------------------------------------------------

def generate_data(n_samples=150, noise_std=0.8, seed=42):
    """
    y = w_true * x + b_true + noise

    We use a single feature for easy visualisation, but the maths
    generalises to any number of features.
    """
    rng = np.random.default_rng(seed)

    w_true = 2.5
    b_true = -1.0

    x = rng.uniform(-3, 3, size=(n_samples, 1))          # (n, 1)
    noise = rng.normal(0, noise_std, size=(n_samples, 1)) # (n, 1)
    y = x * w_true + b_true + noise                       # (n, 1)

    return x, y, w_true, b_true


# ---------------------------------------------------------------------------
# 2. Loss function
# ---------------------------------------------------------------------------

def mse_loss(y_pred, y_true):
    """
    Mean Squared Error: L = (1/n) * sum_i (y_pred_i - y_true_i)^2

    This is the standard regression loss. Squaring penalises large errors
    more than small ones, and the mean makes the loss independent of
    batch size.
    """
    n = y_true.shape[0]
    residuals = y_pred - y_true
    return np.sum(residuals ** 2) / n


# ---------------------------------------------------------------------------
# 3. Gradients (derived analytically)
# ---------------------------------------------------------------------------
#
# Model:  y_pred = X @ w + b       where w is (d, 1) and b is scalar
# Loss:   L = (1/n) ||y_pred - y||^2
#
# Expand:
#   L = (1/n) (Xw + b - y)^T (Xw + b - y)
#
# Gradient w.r.t. w:
#   dL/dw = (2/n) X^T (Xw + b - y)
#         = (2/n) X^T @ residuals
#
# Gradient w.r.t. b:
#   dL/db = (2/n) sum(Xw + b - y)
#         = (2/n) * 1^T @ residuals
#
# These are exact -- no finite differences or autograd needed.

def compute_gradients(X, y, w, b):
    """
    Compute analytical gradients of MSE loss w.r.t. w and b.

    Parameters
    ----------
    X : ndarray (n, d) -- input features
    y : ndarray (n, 1) -- targets
    w : ndarray (d, 1) -- weight vector
    b : float          -- bias

    Returns
    -------
    dw : ndarray (d, 1) -- gradient w.r.t. w
    db : float          -- gradient w.r.t. b
    """
    n = X.shape[0]
    y_pred = X @ w + b         # (n, 1)
    residuals = y_pred - y     # (n, 1)

    dw = (2 / n) * (X.T @ residuals)     # (d, 1)
    db = (2 / n) * np.sum(residuals)      # scalar

    return dw, db


# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------

def train(X, y, lr=0.1, n_epochs=200):
    """
    Train a linear regression model with gradient descent.

    Initialise weights to zero (common for linear models -- the loss
    surface is convex so any starting point reaches the same minimum).
    """
    n_features = X.shape[1]
    w = np.zeros((n_features, 1))
    b = 0.0

    loss_history = []

    for epoch in range(n_epochs):
        # Forward pass
        y_pred = X @ w + b
        loss = mse_loss(y_pred, y)
        loss_history.append(loss)

        # Backward pass (analytical gradients)
        dw, db = compute_gradients(X, y, w, b)

        # Parameter update -- step opposite to the gradient
        w = w - lr * dw
        b = b - lr * db

        # Log progress at milestones
        if epoch % 50 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch:4d}  |  loss = {loss:.6f}  |  "
                  f"w = {w.flatten()}  b = {b:.4f}")

    return w, b, loss_history


# ---------------------------------------------------------------------------
# 5. Closed-form solution (for comparison)
# ---------------------------------------------------------------------------

def normal_equation(X, y):
    """
    w* = (X^T X)^{-1} X^T y   (with bias absorbed into X via a column of 1s)

    The closed-form OLS solution. We compute this to verify that gradient
    descent converges to the right answer.
    """
    n = X.shape[0]
    X_aug = np.hstack([X, np.ones((n, 1))])  # add bias column
    theta = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y
    w_closed = theta[:-1]
    b_closed = theta[-1, 0]
    return w_closed, b_closed


# ---------------------------------------------------------------------------
# 6. Visualisation
# ---------------------------------------------------------------------------

def plot_results(X, y, w, b, loss_history, w_true, b_true, w_closed, b_closed):
    """
    Three-panel figure:
        1. Loss curve over training
        2. Data with the learned regression line vs ground truth
        3. Residual distribution
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # --- Panel 1: Loss curve ---
    ax = axes[0]
    ax.plot(loss_history, color="steelblue", linewidth=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Training loss")
    ax.set_yscale("log")

    # --- Panel 2: Data + fit ---
    ax = axes[1]
    x_flat = X.flatten()
    y_flat = y.flatten()

    sort_idx = np.argsort(x_flat)
    x_sorted = x_flat[sort_idx]

    ax.scatter(x_flat, y_flat, alpha=0.4, s=15, color="grey", label="data")

    # Ground truth line
    ax.plot(x_sorted, w_true * x_sorted + b_true,
            color="green", linewidth=1.5, linestyle="--",
            label=f"true: y = {w_true:.2f}x + ({b_true:.2f})")

    # Learned line
    w_val = w.flatten()[0]
    ax.plot(x_sorted, w_val * x_sorted + b,
            color="red", linewidth=2,
            label=f"learned: y = {w_val:.2f}x + ({b:.2f})")

    # Closed-form line
    w_cf = w_closed.flatten()[0]
    ax.plot(x_sorted, w_cf * x_sorted + b_closed,
            color="blue", linewidth=1.2, linestyle=":",
            label=f"closed-form: y = {w_cf:.2f}x + ({b_closed:.2f})")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Regression fit")
    ax.legend(fontsize=8)

    # --- Panel 3: Residual distribution ---
    ax = axes[2]
    y_pred = X @ w + b
    residuals = (y_pred - y).flatten()
    ax.hist(residuals, bins=25, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("residual (predicted - actual)")
    ax.set_ylabel("count")
    ax.set_title(f"Residuals (mean = {residuals.mean():.4f})")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
    # Generate data
    X, y, w_true, b_true = generate_data(n_samples=150, noise_std=0.8)
    print(f"Ground truth:  w = {w_true},  b = {b_true}")
    print(f"Data shape:    X {X.shape},  y {y.shape}\n")

    # Train with gradient descent
    print("--- Gradient Descent ---")
    w, b, loss_history = train(X, y, lr=0.1, n_epochs=200)
    print(f"\nFinal parameters:  w = {w.flatten()},  b = {b:.4f}")

    # Closed-form solution for comparison
    w_closed, b_closed = normal_equation(X, y)
    print(f"\n--- Closed-Form (Normal Equation) ---")
    print(f"Parameters:        w = {w_closed.flatten()},  b = {b_closed:.4f}")

    # Check they match
    w_diff = np.abs(w.flatten() - w_closed.flatten())
    b_diff = abs(b - b_closed)
    print(f"\nDifference from closed-form:  |dw| = {w_diff},  |db| = {b_diff:.6f}")

    if np.all(w_diff < 0.01) and b_diff < 0.01:
        print("Gradient descent converged to the optimal solution.")
    else:
        print("Note: GD has not fully converged. Try more epochs or adjust lr.")

    # Visualise
    fig = plot_results(X, y, w, b, loss_history, w_true, b_true,
                       w_closed, b_closed)
    plt.show()


if __name__ == "__main__":
    main()
