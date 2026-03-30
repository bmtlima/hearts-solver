"""Train a linear regression model to approximate the paranoid solver.

Usage:
    python regression/train_eval.py [--data PATH] [--plots-dir DIR]

Reads CSV from regression/data/training_data.csv, trains linear + Ridge models,
reports metrics, generates scatter plot, and outputs Rust weights constant.
"""

import argparse
import os

import numpy as np


def load_data(path):
    """Load CSV with header row."""
    with open(path) as f:
        header = f.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    feature_names = header[:-1]  # all columns except last
    X = data[:, :-1]
    y = data[:, -1]
    return X, y, feature_names


def train_test_split(X, y, test_size=0.2, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y)
    indices = rng.permutation(n)
    split = int(n * (1 - test_size))
    train_idx = indices[:split]
    test_idx = indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def fit_linear(X_train, y_train):
    """Ordinary least squares via normal equations."""
    # Add bias column
    ones = np.ones((X_train.shape[0], 1))
    X_aug = np.hstack([ones, X_train])
    # Solve (X^T X) w = X^T y
    w, _, _, _ = np.linalg.lstsq(X_aug, y_train, rcond=None)
    return w  # w[0] = bias, w[1:] = feature weights


def fit_ridge(X_train, y_train, alpha):
    """Ridge regression via closed-form solution."""
    ones = np.ones((X_train.shape[0], 1))
    X_aug = np.hstack([ones, X_train])
    n_features = X_aug.shape[1]
    # Don't regularize the bias term
    reg = alpha * np.eye(n_features)
    reg[0, 0] = 0
    w = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y_train)
    return w


def predict(X, weights):
    ones = np.ones((X.shape[0], 1))
    X_aug = np.hstack([ones, X])
    return X_aug @ weights


def metrics(y_true, y_pred):
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    mae = np.mean(np.abs(residuals))
    max_err = np.max(np.abs(residuals))
    return r2, mae, max_err


def print_weights_rust(weights, feature_names):
    """Print weights as a Rust constant."""
    n = len(weights)
    print(f"\npub const EVAL_WEIGHTS: [f64; {n}] = [")
    print(f"    // bias")
    print(f"    {weights[0]:.10},")
    for i, name in enumerate(feature_names):
        print(f"    // {name}")
        print(f"    {weights[i + 1]:.10},")
    print("];")


def make_scatter_plot(y_test, y_pred, path):
    """Generate predicted vs actual scatter plot."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\nWARNING: matplotlib not installed, skipping scatter plot")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.5, s=20)
    lo = min(y_test.min(), y_pred.min()) - 1
    hi = max(y_test.max(), y_pred.max()) + 1
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_xlabel("Actual Score (paranoid solver)")
    ax.set_ylabel("Predicted Score (regression)")
    ax.set_title("Predicted vs Actual")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nScatter plot saved to: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train regression eval model")
    parser.add_argument(
        "--data",
        default="regression/data/training_data.csv",
        help="Path to training CSV",
    )
    parser.add_argument(
        "--plots-dir", default="regression/plots", help="Directory for plots"
    )
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    X, y, feature_names = load_data(args.data)
    print(f"  {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Score range: [{y.min():.0f}, {y.max():.0f}], mean={y.mean():.2f}")

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(f"  Train: {len(y_train)}, Test: {len(y_test)}")

    # ── Linear Regression ────────────────────────────────────────────────
    print("\n=== Linear Regression ===")
    w_linear = fit_linear(X_train, y_train)
    y_pred = predict(X_test, w_linear)
    r2, mae, max_err = metrics(y_test, y_pred)
    print(f"  R²:        {r2:.4f}")
    print(f"  MAE:       {mae:.3f}")
    print(f"  Max Error: {max_err:.3f}")

    # ── Ridge Regression ─────────────────────────────────────────────────
    print("\n=== Ridge Regression ===")
    best_ridge_w = None
    best_ridge_r2 = -1
    best_alpha = None
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        w_ridge = fit_ridge(X_train, y_train, alpha)
        y_pred_r = predict(X_test, w_ridge)
        r2_r, mae_r, max_r = metrics(y_test, y_pred_r)
        print(f"  alpha={alpha:>5.2f}: R²={r2_r:.4f}, MAE={mae_r:.3f}, Max={max_r:.3f}")
        if r2_r > best_ridge_r2:
            best_ridge_r2 = r2_r
            best_ridge_w = w_ridge
            best_alpha = alpha

    # Choose the best model
    if best_ridge_r2 > r2 + 0.001:
        print(f"\n>> Using Ridge (alpha={best_alpha}) — R² {best_ridge_r2:.4f} vs {r2:.4f}")
        best_w = best_ridge_w
    else:
        print(f"\n>> Using Linear Regression — R² {r2:.4f}")
        best_w = w_linear

    # ── Feature importance ───────────────────────────────────────────────
    print("\n=== Feature Importance (by |weight|) ===")
    importance = [(abs(best_w[i + 1]), feature_names[i], best_w[i + 1]) for i in range(len(feature_names))]
    importance.sort(reverse=True)
    for abs_w, name, w in importance:
        print(f"  {name:>40s}: {w:>+10.4f}")

    # ── Scatter plot ─────────────────────────────────────────────────────
    y_pred_best = predict(X_test, best_w)
    plot_path = os.path.join(args.plots_dir, "predicted_vs_actual.png")
    make_scatter_plot(y_test, y_pred_best, plot_path)

    # ── Error analysis ───────────────────────────────────────────────────
    residuals = y_test - y_pred_best
    abs_residuals = np.abs(residuals)
    large_error_mask = abs_residuals > 5.0
    n_large = np.sum(large_error_mask)
    if n_large > 0:
        print(f"\n=== Large Errors (|error| > 5) ===")
        print(f"  Count: {n_large} / {len(y_test)} ({100*n_large/len(y_test):.1f}%)")
        print(f"  Actual scores: {y_test[large_error_mask]}")
        # Check if they cluster near moon scores (0 or 26)
        near_moon = np.sum((y_test[large_error_mask] <= 2) | (y_test[large_error_mask] >= 24))
        print(f"  Near moon (score <= 2 or >= 24): {near_moon}")

    # ── Rust weights output ──────────────────────────────────────────────
    print_weights_rust(best_w, feature_names)


if __name__ == "__main__":
    main()
