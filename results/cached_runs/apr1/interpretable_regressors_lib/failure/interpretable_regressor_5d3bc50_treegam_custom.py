"""
Interpretable regressor autoresearch script.
Defines a scikit-learn compatible interpretable regressor and evaluates it
on interpretability tests and TabArena regression datasets (same suite used
for baselines in run_baselines.py).

Usage: uv run model.py
"""

import csv
import os
import subprocess
import sys
import time

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class TreeGAMRegressor(BaseEstimator, RegressorMixin):
    """
    Generalized Additive Model using small decision trees as components.

    y = intercept + tree_0(x0) + tree_1(x1) + ... (each tree uses ONE feature)

    Fitted via cyclic backfitting: repeatedly fit each tree to the partial
    residual (y minus all other components). Each tree is shallow (max_depth=3)
    to stay readable.

    Display shows per-feature partial effect tables derived from the trees,
    formatted for easy LLM tracing.
    """

    def __init__(self, max_depth=3, n_iters=10, learning_rate=0.8):
        self.max_depth = max_depth
        self.n_iters = n_iters
        self.learning_rate = learning_rate

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        n_samples, n_features = X.shape

        self.intercept_ = float(np.mean(y))
        residual = y - self.intercept_

        # Initialize per-feature predictions
        self.trees_ = [None] * n_features
        feature_preds = np.zeros((n_features, n_samples))

        for iteration in range(self.n_iters):
            for j in range(n_features):
                # Partial residual: what's left after removing all other components
                partial_residual = residual + feature_preds[j]

                # Fit tree on feature j only
                X_j = X[:, j:j+1]
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_leaf=max(5, n_samples // 50),
                    random_state=42,
                )
                tree.fit(X_j, partial_residual)

                # Update with learning rate
                new_pred = tree.predict(X_j)
                feature_preds[j] = (1 - self.learning_rate) * feature_preds[j] + self.learning_rate * new_pred
                self.trees_[j] = tree

                # Recompute residual
                residual = y - self.intercept_ - feature_preds.sum(axis=0)

        # Build per-feature effect tables for display
        self._build_effect_tables(X)
        return self

    def _build_effect_tables(self, X):
        """Extract effect tables from fitted trees for display."""
        self.effect_tables_ = []
        for j in range(self.n_features_in_):
            tree = self.trees_[j]
            # Sample points spanning the feature range
            x_min, x_max = X[:, j].min(), X[:, j].max()
            x_grid = np.linspace(x_min, x_max, 200)
            preds = tree.predict(x_grid.reshape(-1, 1))

            # Find unique prediction values and their boundaries
            segments = []
            prev_pred = preds[0]
            seg_start = x_grid[0]
            for k in range(1, len(x_grid)):
                if abs(preds[k] - prev_pred) > 1e-8:
                    segments.append((seg_start, x_grid[k], prev_pred))
                    seg_start = x_grid[k]
                    prev_pred = preds[k]
            segments.append((seg_start, x_grid[-1], prev_pred))
            self.effect_tables_.append(segments)

    def predict(self, X):
        check_is_fitted(self, "trees_")
        X = np.asarray(X, dtype=np.float64)
        pred = np.full(X.shape[0], self.intercept_)
        for j in range(self.n_features_in_):
            pred += self.trees_[j].predict(X[:, j:j+1])
        return pred

    def __str__(self):
        check_is_fitted(self, "trees_")
        names = [f"x{i}" for i in range(self.n_features_in_)]

        lines = [
            "TreeGAM Regressor (additive: one small tree per feature):",
            "  y = intercept + f(x0) + f(x1) + ...  (each feature's effect is INDEPENDENT)",
            f"  intercept: {self.intercept_:.4f}",
            "",
            "Per-feature partial effects:",
        ]

        for j, (name, segments) in enumerate(zip(names, self.effect_tables_)):
            lines.append(f"\n  {name}:")
            for seg_idx, (lo, hi, val) in enumerate(segments):
                if seg_idx == 0 and seg_idx == len(segments) - 1:
                    label = f"any value"
                elif seg_idx == 0:
                    label = f"{name} < {hi:.2f}"
                elif seg_idx == len(segments) - 1:
                    label = f"{name} >= {lo:.2f}"
                else:
                    label = f"{lo:.2f} <= {name} < {hi:.2f}"
                lines.append(f"    {label}  →  effect = {val:+.4f}")

            # Summarize overall shape
            effects = [s[2] for s in segments]
            if len(effects) > 1:
                if all(effects[i] <= effects[i+1] + 0.01 for i in range(len(effects)-1)):
                    shape = "increasing"
                elif all(effects[i] >= effects[i+1] - 0.01 for i in range(len(effects)-1)):
                    shape = "decreasing"
                elif max(effects) - min(effects) < 0.1:
                    shape = "flat/negligible"
                else:
                    shape = "non-monotone"
                lines.append(f"    (shape: {shape})")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
TreeGAMRegressor.__module__ = "interpretable_regressor"


# ---------------------------------------------------------------------------
# Evaluation (do not edit any of the evaluation functions, only the names and model descriptions below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    model_defs = [("TreeGAM_custom", TreeGAMRegressor())]

    # Interpretability tests
    interp_results = run_all_interp_tests(model_defs)
    n_passed = sum(r["passed"] for r in interp_results)
    total = len(interp_results)

    # prediction performance (RMSE)
    dataset_rmses = evaluate_all_regressors(model_defs)

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_hash = ""

    # --- Upsert interpretability_results.csv ---
    model_name = "TreeGAM_custom"
    interp_csv = os.path.join(RESULTS_DIR, "interpretability_results.csv")
    interp_fields = ["model", "test", "suite", "passed", "ground_truth", "response"]

    def _suite(test_name):
        if test_name.startswith("insight_"): return "insight"
        if test_name.startswith("hard_"):    return "hard"
        return "standard"

    # Load existing rows, dropping old rows for this model
    existing_interp = []
    if os.path.exists(interp_csv):
        with open(interp_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_interp.append(row)

    new_interp = [{
        "model": r["model"],
        "test": r["test"],
        "suite": _suite(r["test"]),
        "passed": r["passed"],
        "ground_truth": r.get("ground_truth", ""),
        "response": r.get("response", ""),
    } for r in interp_results]

    with open(interp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=interp_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_interp + new_interp)
    print(f"Interpretability results saved → {interp_csv}")

    # --- Upsert performance_results.csv and recompute ranks ---
    perf_csv = os.path.join(RESULTS_DIR, "performance_results.csv")
    perf_fields = ["dataset", "model", "rmse", "rank"]

    # Load existing rows, dropping old rows for this model
    existing_perf = []
    if os.path.exists(perf_csv):
        with open(perf_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_perf.append(row)

    # Add new rows (without rank for now)
    for ds_name, model_rmses in dataset_rmses.items():
        rmse_val = model_rmses.get(model_name, float("nan"))
        existing_perf.append({
            "dataset": ds_name,
            "model": model_name,
            "rmse": "" if np.isnan(rmse_val) else f"{rmse_val:.6f}",
            "rank": "",
        })

    # Recompute ranks per dataset
    from collections import defaultdict
    by_dataset = defaultdict(list)
    for row in existing_perf:
        by_dataset[row["dataset"]].append(row)

    for ds_name, rows in by_dataset.items():
        valid = [(r, float(r["rmse"])) for r in rows if r["rmse"] not in ("", None)]
        valid.sort(key=lambda x: x[1])
        for rank_idx, (r, _) in enumerate(valid, 1):
            r["rank"] = rank_idx
        # Leave rank empty for rows with no RMSE
        for r in rows:
            if r["rmse"] in ("", None):
                r["rank"] = ""

    with open(perf_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=perf_fields)
        writer.writeheader()
        for ds_name in by_dataset:
            for row in by_dataset[ds_name]:
                writer.writerow(row)
    print(f"Performance results saved → {perf_csv}")

    # --- Compute mean_rank from the updated performance_results.csv ---
    # Build dataset_rmses dict with all models from the CSV for ranking
    all_dataset_rmses = defaultdict(dict)
    for row in existing_perf:
        rmse_str = row.get("rmse", "")
        if rmse_str not in ("", None):
            all_dataset_rmses[row["dataset"]][row["model"]] = float(rmse_str)
        else:
            all_dataset_rmses[row["dataset"]][row["model"]] = float("nan")
    avg_rank, _ = compute_rank_scores(dict(all_dataset_rmses))
    mean_rank = avg_rank.get(model_name, float("nan"))

    upsert_overall_results([{
        "commit":                             git_hash,
        "mean_rank":                          f"{mean_rank:.2f}" if not np.isnan(mean_rank) else "nan",
        "frac_interpretability_tests_passed": f"{n_passed / total:.4f}" if total > 0 else "nan",
        "status":                             "",
        "model_name":                         "TreeGAM_custom",
        "description":                        "Custom TreeGAM: one shallow tree per feature fitted via backfitting",
    }], RESULTS_DIR)

    # --- Plot ---
    overall_csv = os.path.join(RESULTS_DIR, "overall_results.csv")
    plot_interp_vs_performance(
        overall_csv,
        os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"),
    )

    print()
    print("---")
    print(f"tests_passed:  {n_passed}/{total}" + (f" ({n_passed/total:.2%})" if total > 0 else ""))
    print(f"mean_rank:     {mean_rank:.2f}" if not np.isnan(mean_rank) else "mean_rank:     nan")
    print(f"total_seconds: {time.time() - t0:.1f}s")

