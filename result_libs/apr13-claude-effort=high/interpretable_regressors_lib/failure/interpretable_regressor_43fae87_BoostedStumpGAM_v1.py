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
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class BoostedStumpGAMRegressor(BaseEstimator, RegressorMixin):
    """
    Additive model built from boosted single-feature decision stumps.

    Each boosting round fits a depth-1 tree (stump) on residuals, selecting
    the best single feature. After all rounds, stumps are aggregated into
    per-feature piecewise-constant shape functions. The model is displayed
    as a GAM: y = intercept + f0(x0) + f1(x1) + ...

    Each f_i is shown as a table of (x_i value → effect) evaluated at
    representative points, making the model highly interpretable.
    """

    def __init__(self, n_rounds=200, learning_rate=0.1, min_samples_leaf=10):
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        # Store stumps grouped by feature
        self.intercept_ = float(np.mean(y))
        residuals = y - self.intercept_

        # For each feature, we'll accumulate (threshold, left_val, right_val) tuples
        self.stumps_ = defaultdict(list)  # feature_idx -> [(threshold, left_delta, right_delta)]

        for round_idx in range(self.n_rounds):
            best_feat = None
            best_thresh = None
            best_left_val = None
            best_right_val = None
            best_mse = float('inf')

            for j in range(n_features):
                stump = DecisionTreeRegressor(max_depth=1,
                                               min_samples_leaf=self.min_samples_leaf,
                                               random_state=42 + round_idx)
                stump.fit(X[:, j:j+1], residuals)
                preds = stump.predict(X[:, j:j+1])
                mse = float(np.mean((residuals - preds) ** 2))
                if mse < best_mse:
                    best_mse = mse
                    best_feat = j
                    tree = stump.tree_
                    if tree.children_left[0] != -1:
                        best_thresh = float(tree.threshold[0])
                        left_node = tree.children_left[0]
                        right_node = tree.children_right[0]
                        best_left_val = float(tree.value[left_node, 0, 0])
                        best_right_val = float(tree.value[right_node, 0, 0])
                    else:
                        best_thresh = 0.0
                        best_left_val = float(tree.value[0, 0, 0])
                        best_right_val = best_left_val

            if best_feat is None:
                break

            lr = self.learning_rate
            self.stumps_[best_feat].append((best_thresh, best_left_val * lr, best_right_val * lr))

            # Update residuals
            mask = X[:, best_feat] <= best_thresh
            residuals[mask] -= best_left_val * lr
            residuals[~mask] -= best_right_val * lr

        # Pre-compute shape functions for each feature at evaluation grid points
        self._build_shape_functions(X)
        return self

    def _build_shape_functions(self, X):
        """Build per-feature shape functions as sorted (threshold, cumulative_effect) arrays."""
        self.shape_funcs_ = {}
        for j in range(self.n_features_):
            stumps = self.stumps_.get(j, [])
            if not stumps:
                continue
            # Collect all thresholds for this feature
            thresholds = sorted(set(t for t, _, _ in stumps))
            # Evaluate shape function at representative grid points
            x_min = float(np.min(X[:, j]))
            x_max = float(np.max(X[:, j]))
            grid = np.linspace(x_min, x_max, 9)
            effects = []
            for x_val in grid:
                effect = 0.0
                for thresh, left_val, right_val in stumps:
                    if x_val <= thresh:
                        effect += left_val
                    else:
                        effect += right_val
                effects.append(effect)
            self.shape_funcs_[j] = (grid, effects)

    def predict(self, X):
        check_is_fitted(self, "stumps_")
        X = np.asarray(X, dtype=np.float64)
        preds = np.full(X.shape[0], self.intercept_)
        for j, stumps in self.stumps_.items():
            for thresh, left_val, right_val in stumps:
                mask = X[:, j] <= thresh
                preds[mask] += left_val
                preds[~mask] += right_val
        return preds

    def __str__(self):
        check_is_fitted(self, "stumps_")
        lines = [
            "Generalized Additive Model (Boosted Stumps):",
            "  y = intercept + f0(x0) + f1(x1) + ...  (each feature's effect is INDEPENDENT)",
            f"  intercept: {self.intercept_:.4f}",
            "",
            "Feature partial effects (each feature's independent contribution):",
        ]

        # Sort features by importance (range of effect)
        feat_importance = {}
        for j in range(self.n_features_):
            if j in self.shape_funcs_:
                grid, effects = self.shape_funcs_[j]
                feat_importance[j] = max(effects) - min(effects)
            else:
                feat_importance[j] = 0.0

        sorted_feats = sorted(feat_importance.keys(), key=lambda j: feat_importance[j], reverse=True)

        for j in sorted_feats:
            if j not in self.shape_funcs_:
                continue
            grid, effects = self.shape_funcs_[j]
            effect_range = max(effects) - min(effects)
            if effect_range < 1e-4:
                continue  # skip negligible features

            # Determine shape
            if effects[-1] > effects[0] + 0.3:
                shape = "increasing"
            elif effects[-1] < effects[0] - 0.3:
                shape = "decreasing"
            elif effect_range < 0.3:
                shape = "flat/negligible"
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
BoostedStumpGAMRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "BoostedStumpGAM_v1"
model_description = "Additive model from boosted stumps with per-feature shape function display as GAM"
model_defs = [(model_shorthand_name, BoostedStumpGAMRegressor())]


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

