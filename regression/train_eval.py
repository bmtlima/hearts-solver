"""Train evaluation models for the Hearts solver.

Usage:
    python regression/train_eval.py [--mode linear|combinatorial|nn] [--input PATH] [--filter-moon]

Modes:
    linear        — Linear/Ridge regression on 35 hand-crafted features (default)
    combinatorial — Ridge with pairwise feature products
    nn            — PyTorch MLP on 214 raw card inputs
"""

import argparse
import os

import numpy as np


# ── Data loading ─────────────────────────────────────────────────────────

def load_data(path):
    """Load CSV with header row."""
    with open(path) as f:
        header = f.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    all_names = header[:-1]  # all columns except score
    X = data[:, :-1]
    y = data[:, -1]
    return X, y, all_names


def split_columns(X, all_names):
    """Split into raw card columns (214) and feature columns (35)."""
    # Detect if raw card columns are present by checking column count
    if len(all_names) > 40:
        # Raw card columns come first, then 35 features
        n_raw = len(all_names) - 35
        X_raw = X[:, :n_raw]
        X_feat = X[:, n_raw:]
        raw_names = all_names[:n_raw]
        feat_names = all_names[n_raw:]
        return X_raw, raw_names, X_feat, feat_names
    else:
        return None, [], X, all_names


def train_test_split(X, y, test_size=0.2, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y)
    indices = rng.permutation(n)
    split = int(n * (1 - test_size))
    train_idx = indices[:split]
    test_idx = indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y)
    indices = rng.permutation(n)
    test_end = int(n * test_size)
    val_end = test_end + int(n * val_size)
    test_idx = indices[:test_end]
    val_idx = indices[test_end:val_end]
    train_idx = indices[val_end:]
    return (
        X[train_idx], X[val_idx], X[test_idx],
        y[train_idx], y[val_idx], y[test_idx],
    )


# ── Models ───────────────────────────────────────────────────────────────

def fit_linear(X_train, y_train):
    ones = np.ones((X_train.shape[0], 1))
    X_aug = np.hstack([ones, X_train])
    w, _, _, _ = np.linalg.lstsq(X_aug, y_train, rcond=None)
    return w


def fit_ridge(X_train, y_train, alpha):
    ones = np.ones((X_train.shape[0], 1))
    X_aug = np.hstack([ones, X_train])
    n_features = X_aug.shape[1]
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


# ── Output ───────────────────────────────────────────────────────────────

def print_weights_rust(weights, feature_names):
    n = len(weights)
    print(f"\npub const EVAL_WEIGHTS: [f64; {n}] = [")
    print(f"    // bias")
    print(f"    {weights[0]:.10},")
    for i, name in enumerate(feature_names):
        print(f"    // {name}")
        print(f"    {weights[i + 1]:.10},")
    print("];")


def make_scatter_plot(y_test, y_pred, path, title=""):
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
    ax.set_xlabel("Actual Score")
    ax.set_ylabel("Predicted Score")
    ax.set_title(title or "Predicted vs Actual")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nScatter plot saved to: {path}")
    plt.close(fig)


# ── Mode: linear ─────────────────────────────────────────────────────────

def mode_linear(X_feat, y, feat_names, args):
    print(f"\n{'='*60}")
    print("MODE: linear (35 hand-crafted features)")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test = train_test_split(X_feat, y)
    print(f"  Train: {len(y_train)}, Test: {len(y_test)}")

    # Linear
    print("\n=== Linear Regression ===")
    w_linear = fit_linear(X_train, y_train)
    y_pred = predict(X_test, w_linear)
    r2, mae, max_err = metrics(y_test, y_pred)
    print(f"  R²:        {r2:.4f}")
    print(f"  MAE:       {mae:.3f}")
    print(f"  Max Error: {max_err:.3f}")

    # Ridge
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

    if best_ridge_r2 > r2 + 0.001:
        print(f"\n>> Using Ridge (alpha={best_alpha}) — R² {best_ridge_r2:.4f} vs {r2:.4f}")
        best_w = best_ridge_w
    else:
        print(f"\n>> Using Linear Regression — R² {r2:.4f}")
        best_w = w_linear

    # Feature importance
    print("\n=== Feature Importance (by |weight|) ===")
    importance = [(abs(best_w[i + 1]), feat_names[i], best_w[i + 1]) for i in range(len(feat_names))]
    importance.sort(reverse=True)
    for abs_w, name, w in importance:
        print(f"  {name:>40s}: {w:>+10.4f}")

    # Scatter plot
    y_pred_best = predict(X_test, best_w)
    plot_path = os.path.join(args.plots_dir, "predicted_vs_actual.png")
    make_scatter_plot(y_test, y_pred_best, plot_path, f"Linear — R²={metrics(y_test, y_pred_best)[0]:.4f}")

    # Error analysis
    residuals = y_test - y_pred_best
    abs_residuals = np.abs(residuals)
    large_error_mask = abs_residuals > 5.0
    n_large = np.sum(large_error_mask)
    if n_large > 0:
        print(f"\n=== Large Errors (|error| > 5) ===")
        print(f"  Count: {n_large} / {len(y_test)} ({100*n_large/len(y_test):.1f}%)")
        near_moon = np.sum((y_test[large_error_mask] <= 2) | (y_test[large_error_mask] >= 24))
        print(f"  Near moon (score <= 2 or >= 24): {near_moon}")

    print_weights_rust(best_w, feat_names)


# ── Mode: combinatorial ──────────────────────────────────────────────────

def mode_combinatorial(X_feat, y, feat_names, args):
    print(f"\n{'='*60}")
    print("MODE: combinatorial (35 features + pairwise products)")
    print(f"{'='*60}")

    from sklearn.linear_model import Ridge

    X_train, X_test, y_train, y_test = train_test_split(X_feat, y)

    # Baseline: linear on original 35
    w_base = fit_linear(X_train, y_train)
    y_pred_base = predict(X_test, w_base)
    r2_base, mae_base, _ = metrics(y_test, y_pred_base)
    print(f"\n  Baseline (35 features): R²={r2_base:.4f}, MAE={mae_base:.3f}")

    # Generate pairwise products
    n_feat = X_feat.shape[1]
    pairs = []
    pair_names = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            product = X_feat[:, i] * X_feat[:, j]
            # Keep only if both features are nonzero in >= 1% of samples
            nonzero_rate = np.mean(product != 0)
            if nonzero_rate >= 0.01:
                pairs.append(product)
                pair_names.append(f"{feat_names[i]}_x_{feat_names[j]}")

    print(f"  Pairwise products: {len(pairs)} (after filtering <1% nonzero)")

    X_combo = np.column_stack([X_feat] + pairs)
    combo_names = list(feat_names) + pair_names
    X_combo_train = X_combo[: len(y_train)]
    X_combo_test = X_combo[len(y_train) :]

    # Need to re-split since we built combo from full X
    X_combo_train, X_combo_test, _, _ = train_test_split(X_combo, y)

    # Ridge with CV
    print("\n=== Ridge on 35 + pairwise features ===")
    best_r2 = -1
    best_alpha = None
    best_model = None
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = Ridge(alpha=alpha).fit(X_combo_train, y_train)
        y_pred = model.predict(X_combo_test)
        r2, mae, max_err = metrics(y_test, y_pred)
        print(f"  alpha={alpha:>6.2f}: R²={r2:.4f}, MAE={mae:.3f}")
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha
            best_model = model

    improvement = best_r2 - r2_base
    print(f"\n>> Best Ridge: alpha={best_alpha}, R²={best_r2:.4f} (Δ={improvement:+.4f} vs baseline)")

    # Feature importance (top 30)
    coefs = best_model.coef_
    importance = sorted(
        [(abs(coefs[i]), combo_names[i], coefs[i]) for i in range(len(combo_names))],
        reverse=True,
    )
    print("\n=== Top 30 Features by |weight| ===")
    for abs_w, name, w in importance[:30]:
        print(f"  {name:>60s}: {w:>+10.4f}")

    # Scatter plot
    y_pred_best = best_model.predict(X_combo_test)
    plot_path = os.path.join(args.plots_dir, "predicted_vs_actual_combo.png")
    make_scatter_plot(y_test, y_pred_best, plot_path, f"Combinatorial — R²={best_r2:.4f}")

    # Triple-wise if improvement > 0.03
    if improvement > 0.03:
        print("\n=== Trying triple-wise combinations of top 20 features ===")
        top_idx = sorted(range(len(coefs)), key=lambda i: abs(coefs[i]), reverse=True)[:20]
        triples = []
        triple_names = []
        for a in range(len(top_idx)):
            for b in range(a + 1, len(top_idx)):
                for c in range(b + 1, len(top_idx)):
                    i, j, k = top_idx[a], top_idx[b], top_idx[c]
                    product = X_combo[:, i] * X_combo[:, j] * X_combo[:, k]
                    if np.mean(product != 0) >= 0.01:
                        triples.append(product)
                        triple_names.append(f"{combo_names[i]}_x_{combo_names[j]}_x_{combo_names[k]}")

        print(f"  Triple products: {len(triples)}")
        if triples:
            X_triple = np.column_stack([X_combo] + triples)
            X_triple_train, X_triple_test, _, _ = train_test_split(X_triple, y)
            model_t = Ridge(alpha=best_alpha).fit(X_triple_train, y_train)
            y_pred_t = model_t.predict(X_triple_test)
            r2_t, mae_t, _ = metrics(y_test, y_pred_t)
            print(f"  Triple model: R²={r2_t:.4f}, MAE={mae_t:.3f}")

    # Note about Rust integration
    print("\n  NOTE: To use combinatorial features in Rust, update eval.rs to compute")
    print(f"  the {len(pair_names)} pairwise products. Output weights above for reference.")


# ── Mode: nn ─────────────────────────────────────────────────────────────

def mode_nn(X, y, all_names, args):
    print(f"\n{'='*60}")
    print("MODE: nn (PyTorch MLP on raw card positions)")
    print(f"{'='*60}")

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("ERROR: PyTorch required. pip install torch")
        return

    # Split raw card columns from features
    X_raw, raw_names, X_feat, feat_names = split_columns(X, all_names)
    if X_raw is None:
        print("ERROR: No raw card columns found. Run with --raw-cards flag when generating data.")
        return

    # Use 214 raw inputs (208 card + 6 scalar)
    print(f"  Raw inputs: {X_raw.shape[1]} columns")
    print(f"  Feature inputs: {X_feat.shape[1]} columns (not used for NN)")

    X_nn = X_raw.astype(np.float32)

    # Normalize scalar columns (last 6 of raw section)
    # Card columns (0:208) are already binary 0/1
    for i in range(208, X_nn.shape[1]):
        col = X_nn[:, i]
        mu, std = col.mean(), col.std()
        if std > 0:
            X_nn[:, i] = (col - mu) / std

    # 70/15/15 split
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X_nn, y)
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
    y_test = y_test.astype(np.float32)

    print(f"  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # Datasets
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)

    n_inputs = X_nn.shape[1]

    def train_model(hidden_sizes, name):
        """Train an MLP with given hidden layer sizes."""
        layers = []
        prev = n_inputs
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        model = nn.Sequential(*layers)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        patience = 10
        best_state = None

        print(f"\n  Training {name}...")
        for epoch in range(100):
            model.train()
            train_loss = 0
            for xb, yb in train_loader:
                pred = model(xb).squeeze(-1)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(y_train)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    pred = model(xb).squeeze()
                    val_loss += criterion(pred, yb).item() * len(xb)
            val_loss /= len(y_val)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or patience_counter == patience:
                print(
                    f"    Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, best={best_val_loss:.4f}"
                )

            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

        model.load_state_dict(best_state)
        return model

    def eval_model(model, name):
        """Evaluate model on test set."""
        model.eval()
        with torch.no_grad():
            y_pred = model(torch.from_numpy(X_test)).squeeze(-1).numpy()
        r2, mae, max_err = metrics(y_test, y_pred)
        print(f"\n  {name} — Test: R²={r2:.4f}, MAE={mae:.3f}, Max Error={max_err:.3f}")
        return y_pred, r2

    # Train both architectures
    model_large = train_model([64, 32], f"Large MLP ({n_inputs}→64→32→1)")
    y_pred_large, r2_large = eval_model(model_large, "Large MLP")

    model_small = train_model([32], f"Small MLP ({n_inputs}→32→1)")
    y_pred_small, r2_small = eval_model(model_small, "Small MLP")

    # Pick best
    if r2_large >= r2_small:
        best_model, best_name, best_pred = model_large, "Large MLP", y_pred_large
    else:
        best_model, best_name, best_pred = model_small, "Small MLP", y_pred_small

    print(f"\n>> Best: {best_name}")

    # Scatter plot
    plot_path = os.path.join(args.plots_dir, "predicted_vs_actual_nn.png")
    r2_best = metrics(y_test, best_pred)[0]
    make_scatter_plot(y_test, best_pred, plot_path, f"NN ({best_name}) — R²={r2_best:.4f}")

    # Save model
    os.makedirs("regression/models", exist_ok=True)
    torch.save(best_model.state_dict(), "regression/models/eval_nn.pt")
    print(f"  Model saved to regression/models/eval_nn.pt")

    # Export weights as Rust constants
    print(f"\n=== Rust Weight Export ({best_name}) ===")
    state = best_model.state_dict()
    for name_key, tensor in state.items():
        arr = tensor.numpy()
        if arr.ndim == 2:
            rows, cols = arr.shape
            print(f"\npub const {name_key.upper().replace('.', '_')}: [[f64; {cols}]; {rows}] = [")
            for row in arr:
                vals = ", ".join(f"{v:.8f}" for v in row)
                print(f"    [{vals}],")
            print("];")
        elif arr.ndim == 1:
            vals = ", ".join(f"{v:.8f}" for v in arr)
            print(f"\npub const {name_key.upper().replace('.', '_')}: [f64; {len(arr)}] = [{vals}];")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train evaluation models")
    parser.add_argument(
        "--input", "--data",
        default="regression/data/training_data.csv",
        dest="data",
        help="Path to training CSV",
    )
    parser.add_argument(
        "--plots-dir", default="regression/plots", help="Directory for plots"
    )
    parser.add_argument(
        "--filter-moon",
        action="store_true",
        help="Remove moon rows (score == 0 or score == 26) before training",
    )
    parser.add_argument(
        "--mode",
        choices=["linear", "combinatorial", "nn"],
        default="linear",
        help="Training mode (default: linear)",
    )
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    X, y, all_names = load_data(args.data)
    print(f"  {X.shape[0]} samples, {X.shape[1]} columns")
    print(f"  Score range: [{y.min():.0f}, {y.max():.0f}], mean={y.mean():.2f}")

    if args.filter_moon:
        moon_mask = (y == 0) | (y == 26)
        n_moon = np.sum(moon_mask)
        X = X[~moon_mask]
        y = y[~moon_mask]
        print(f"  Filtered {n_moon} moon rows, {len(y)} remaining")

    # Split raw vs feature columns
    X_raw, raw_names, X_feat, feat_names = split_columns(X, all_names)

    if args.mode == "linear":
        mode_linear(X_feat, y, feat_names, args)
    elif args.mode == "combinatorial":
        mode_combinatorial(X_feat, y, feat_names, args)
    elif args.mode == "nn":
        mode_nn(X, y, all_names, args)


if __name__ == "__main__":
    main()
