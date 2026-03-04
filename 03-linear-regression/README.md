# Linear Regression from Scratch

Linear regression trained with gradient descent on MSE loss, implemented entirely in NumPy.

## What this covers

- **Synthetic data generation** with known ground-truth parameters, so we can verify the model recovers the correct weights.
- **MSE loss** computed explicitly -- no library calls.
- **Analytical gradients** derived by hand from the loss function and implemented directly. The key results:
  - `dL/dw = (2/n) * X^T @ (Xw + b - y)`
  - `dL/db = (2/n) * sum(Xw + b - y)`
- **Gradient descent loop** that iteratively updates `w` and `b`.
- **Closed-form comparison** via the normal equation `w* = (X^T X)^{-1} X^T y` to verify that GD converges to the analytically optimal solution.

## Running

```bash
python linear_regression.py
```

Requires `numpy` and `matplotlib`.

## Output

The script prints the learned parameters at regular intervals and produces a three-panel figure:

1. **Loss curve** -- MSE over training epochs (log scale). Should decrease monotonically for a convex problem like this.
2. **Regression fit** -- the learned line plotted against the data, the ground truth, and the closed-form solution.
3. **Residual distribution** -- histogram of prediction errors. Should be roughly centred at zero with spread matching the noise standard deviation.

## Key ideas

- Linear regression with MSE is a convex optimisation problem. There is exactly one minimum, and gradient descent will find it given a reasonable learning rate.
- The closed-form solution is elegant but requires inverting `X^T X`, which is `O(d^3)` in the number of features. Gradient descent scales better to high dimensions and large datasets.
- Inspecting residuals is a basic but important sanity check -- systematic patterns indicate model misspecification.
