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
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class CyclicBoostGAMRegressor(BaseEstimator, RegressorMixin):
    """
    EBM-style cyclic boosting GAM: builds per-feature shape functions by
    cycling through features and fitting small trees on residuals.

    For each cycle, for each feature, fit a depth-1 stump on the current
    residuals using only that feature, then update the shape function for
    that feature. This produces an additive model where each feature has
    an independent, potentially nonlinear effect.

    Adaptive display: features with high linear R² are shown as linear
    coefficients; features with nonlinear effects are shown as piecewise
    effect tables (like GAM).
    """

    def __init__(self, n_cycles=10, learning_rate=0.3, min_samples_leaf=10, n_grid=7):
        self.n_cycles = n_cycles
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.n_grid = n_grid

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        self.intercept_ = float(np.mean(y))
        residuals = y - self.intercept_

        # Per-feature: list of (threshold, left_val, right_val) stumps
        feature_stumps = defaultdict(list)

        for cycle in range(self.n_cycles):
            for j in range(n_features):
                # Remove this feature's current contribution from residuals
                # (backfitting step - not needed for first cycle)
                stump = DecisionTreeRegressor(
                    max_depth=1,
                    min_samples_leaf=max(self.min_samples_leaf, n_samples // 50),
                    random_state=42
                )
                stump.fit(X[:, j:j+1], residuals)
                tree = stump.tree_

                if tree.children_left[0] != -1:
                    thresh = float(tree.threshold[0])
                    left_val = float(tree.value[tree.children_left[0], 0, 0]) * self.learning_rate
                    right_val = float(tree.value[tree.children_right[0], 0, 0]) * self.learning_rate
                else:
                    continue  # no split found

                feature_stumps[j].append((thresh, left_val, right_val))

                # Update residuals
                mask = X[:, j] <= thresh
                residuals[mask] -= left_val
                residuals[~mask] -= right_val

        self.feature_stumps_ = dict(feature_stumps)

        # Build shape function grids for display
        self._build_display(X)
        return self

    def _eval_feature(self, j, x_vals):
        """Evaluate feature j's shape function at given x values."""
        effects = np.zeros(len(x_vals))
        for thresh, left_val, right_val in self.feature_stumps_.get(j, []):
            mask = x_vals <= thresh
            effects[mask] += left_val
            effects[~mask] += right_val
        return effects

    def _build_display(self, X):
        """Build per-feature display info: grid points, effects, and linear fit R²."""
        self.display_ = {}
        for j in range(self.n_features_):
            if j not in self.feature_stumps_ or not self.feature_stumps_[j]:
                continue
            x_col = X[:, j]
            grid = np.linspace(float(x_col.min()), float(x_col.max()), self.n_grid)
            effects = self._eval_feature(j, grid)
            effect_range = float(np.max(effects) - np.min(effects))

            # Check linear fit quality
            if effect_range > 1e-6:
                lr = LinearRegression()
                lr.fit(grid.reshape(-1, 1), effects)
                pred = lr.predict(grid.reshape(-1, 1))
                r2 = float(r2_score(effects, pred))
                slope = float(lr.coef_[0])
            else:
                r2 = 1.0
                slope = 0.0

            self.display_[j] = {
                'grid': grid,
                'effects': effects.tolist(),
                'effect_range': effect_range,
                'r2_linear': r2,
                'slope': slope,
            }

    def predict(self, X):
        check_is_fitted(self, "feature_stumps_")
        X = np.asarray(X, dtype=np.float64)
        preds = np.full(X.shape[0], self.intercept_)
        for j, stumps in self.feature_stumps_.items():
            for thresh, left_val, right_val in stumps:
                mask = X[:, j] <= thresh
                preds[mask] += left_val
                preds[~mask] += right_val
        return preds

    def __str__(self):
        check_is_fitted(self, "feature_stumps_")
        lines = [
            "Generalized Additive Model (Cyclic Boosting):",
            "  y = intercept + f0(x0) + f1(x1) + ...  (each feature's effect is INDEPENDENT)",
            f"  intercept: {self.intercept_:.4f}",
            "",
        ]

        # Separate into linear and nonlinear features
        linear_feats = []
        nonlinear_feats = []

        for j in sorted(self.display_.keys()):
            info = self.display_[j]
            if info['effect_range'] < 1e-4:
                continue
            if info['r2_linear'] > 0.85:
                linear_feats.append((j, info))
            else:
                nonlinear_feats.append((j, info))

        # Display linear features as coefficients
        if linear_feats:
            lines.append("Linear effects (coefficient × feature):")
            # Sort by importance
            linear_feats.sort(key=lambda x: abs(x[1]['slope']), reverse=True)
            for j, info in linear_feats:
                lines.append(f"  x{j}: {info['slope']:.4f}")

        # Display nonlinear features as tables
        if nonlinear_feats:
            lines.append("")
            lines.append("Non-linear feature effects:")
            # Sort by importance
            nonlinear_feats.sort(key=lambda x: x[1]['effect_range'], reverse=True)
            for j, info in nonlinear_feats:
                grid = info['grid']
                effects = info['effects']
                # Determine shape
                if effects[-1] > effects[0] + 0.3:
                    shape = "increasing"
                elif effects[-1] < effects[0] - 0.3:
                    shape = "decreasing"
                else:
                    shape = "non-monotone"
                lines.append(f"\n  x{j}:")
                for xv, ev in zip(grid, effects):
                    lines.append(f"    x{j}={xv:+.2f}  →  effect={ev:+.4f}")
                lines.append(f"    (shape: {shape})")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
CyclicBoostGAMRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "CyclicBoostGAM_v1"
model_description = "EBM-style cyclic boosting GAM with adaptive linear/table display per feature"
model_defs = [(model_shorthand_name, CyclicBoostGAMRegressor())]


# ---------------------------------------------------------------------------
# Evaluation (do not edit anything below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

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
    model_name = model_defs[0][0]
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
    mean_rank = avg_rank.get(model_shorthand_name, float("nan"))

    upsert_overall_results([{
        "commit":                             git_hash,
        "mean_rank":                          f"{mean_rank:.2f}" if not np.isnan(mean_rank) else "nan",
        "frac_interpretability_tests_passed": f"{n_passed / total:.4f}" if total > 0 else "nan",
        "status":                             "",
        "model_name":                         model_shorthand_name,
        "description":                        model_description,
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

