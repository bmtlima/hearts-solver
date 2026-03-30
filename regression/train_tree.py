"""Train tree-based models on the regression data.

Usage:
    python regression/train_tree.py [--data PATH]
"""

import argparse
import numpy as np


def load_data(path):
    with open(path) as f:
        header = f.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    feature_names = header[:-1]
    X = data[:, :-1]
    y = data[:, -1]
    return X, y, feature_names


def split_data(X, y, seed=42):
    """70/20/10 train/val/test split."""
    rng = np.random.RandomState(seed)
    n = len(y)
    indices = rng.permutation(n)
    train_end = int(n * 0.7)
    val_end = int(n * 0.9)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    return (
        X[train_idx], X[val_idx], X[test_idx],
        y[train_idx], y[val_idx], y[test_idx],
    )


def metrics(y_true, y_pred):
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    mae = np.mean(np.abs(residuals))
    max_err = np.max(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    return {"R²": r2, "MAE": mae, "RMSE": rmse, "Max Error": max_err}


def print_metrics(name, m):
    print(f"  R²:        {m['R²']:.4f}")
    print(f"  MAE:       {m['MAE']:.3f}")
    print(f"  RMSE:      {m['RMSE']:.3f}")
    print(f"  Max Error: {m['Max Error']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="regression/data/training_data.csv")
    args = parser.parse_args()

    try:
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import (
            RandomForestRegressor,
            GradientBoostingRegressor,
        )
        from sklearn.linear_model import LinearRegression
    except ImportError:
        print("ERROR: scikit-learn required. pip install scikit-learn")
        return

    print(f"Loading {args.data}...")
    X, y, feature_names = load_data(args.data)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    print(f"  Score range: [{y.min():.0f}, {y.max():.0f}], mean={y.mean():.2f}")

    results = {}

    # ── Linear baseline ──────────────────────────────────────────────────
    print("\n=== Linear Regression (baseline) ===")
    lr = LinearRegression().fit(X_train, y_train)
    m = metrics(y_val, lr.predict(X_val))
    print_metrics("Linear", m)
    results["Linear"] = m

    # ── Decision Tree ────────────────────────────────────────────────────
    print("\n=== Decision Tree (tuning max_depth on val) ===")
    best_dt = None
    best_dt_r2 = -1
    for depth in [5, 8, 10, 15, 20, None]:
        dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        m = metrics(y_val, dt.predict(X_val))
        label = f"depth={depth}"
        print(f"  {label:>12s}: R²={m['R²']:.4f}, MAE={m['MAE']:.3f}, RMSE={m['RMSE']:.3f}")
        if m["R²"] > best_dt_r2:
            best_dt_r2 = m["R²"]
            best_dt = dt
            best_dt_depth = depth
    print(f"  >> Best: depth={best_dt_depth}")
    results["Decision Tree"] = metrics(y_val, best_dt.predict(X_val))

    # ── Random Forest ────────────────────────────────────────────────────
    print("\n=== Random Forest (tuning n_estimators + max_depth on val) ===")
    best_rf = None
    best_rf_r2 = -1
    best_rf_label = ""
    for n_est in [50, 100, 200]:
        for depth in [10, 15, 20, None]:
            rf = RandomForestRegressor(
                n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1
            )
            rf.fit(X_train, y_train)
            m = metrics(y_val, rf.predict(X_val))
            label = f"n={n_est}, depth={depth}"
            print(f"  {label:>22s}: R²={m['R²']:.4f}, MAE={m['MAE']:.3f}")
            if m["R²"] > best_rf_r2:
                best_rf_r2 = m["R²"]
                best_rf = rf
                best_rf_label = label
    print(f"  >> Best: {best_rf_label}")
    results["Random Forest"] = metrics(y_val, best_rf.predict(X_val))

    # ── Gradient Boosting ────────────────────────────────────────────────
    print("\n=== Gradient Boosting (tuning on val) ===")
    best_gb = None
    best_gb_r2 = -1
    best_gb_label = ""
    for n_est in [100, 200, 500]:
        for depth in [3, 5, 7]:
            for lr_val in [0.05, 0.1]:
                gb = GradientBoostingRegressor(
                    n_estimators=n_est, max_depth=depth, learning_rate=lr_val,
                    random_state=42,
                )
                gb.fit(X_train, y_train)
                m = metrics(y_val, gb.predict(X_val))
                label = f"n={n_est}, d={depth}, lr={lr_val}"
                print(f"  {label:>28s}: R²={m['R²']:.4f}, MAE={m['MAE']:.3f}")
                if m["R²"] > best_gb_r2:
                    best_gb_r2 = m["R²"]
                    best_gb = gb
                    best_gb_label = label
    print(f"  >> Best: {best_gb_label}")
    results["Gradient Boosting"] = metrics(y_val, best_gb.predict(X_val))

    # ── Final comparison on TEST set ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL TEST SET RESULTS (held-out 10%)")
    print("=" * 60)
    models = [
        ("Linear", lr),
        ("Decision Tree", best_dt),
        ("Random Forest", best_rf),
        ("Gradient Boosting", best_gb),
    ]
    for name, model in models:
        m = metrics(y_test, model.predict(X_test))
        print(f"\n  {name}:")
        print_metrics(name, m)

    # ── Feature importance from best tree model ──────────────────────────
    best_name, best_model = max(
        [(n, m) for n, m in models[1:]],  # skip linear
        key=lambda x: metrics(y_val, x[1].predict(X_val))["R²"],
    )
    print(f"\n=== Feature Importance ({best_name}) ===")
    importances = best_model.feature_importances_
    order = np.argsort(importances)[::-1]
    for i in order:
        if importances[i] > 0.001:
            print(f"  {feature_names[i]:>40s}: {importances[i]:.4f}")


if __name__ == "__main__":
    main()
