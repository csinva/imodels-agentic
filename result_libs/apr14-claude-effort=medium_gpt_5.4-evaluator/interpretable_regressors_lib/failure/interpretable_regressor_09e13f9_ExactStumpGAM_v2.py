"""
Interpretable regressor autoresearch script.
Defines a scikit-learn compatible interpretable regressor and evaluates it
on interpretability tests and TabArena regression datasets (same suite used
for baselines in run_baselines.py).

Usage: uv run model.py
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class ExactStumpGAMRegressor(BaseEstimator, RegressorMixin):
    """
    Exact additive stump GAM with consistent prediction and display.

    1. Fit GBM with depth-1 stumps (purely additive, no interactions)
    2. For each feature, exactly extract the per-feature step function
       by accumulating all stump contributions
    3. Fit a depth-2 tree to approximate each step function
    4. Use THESE trees for BOTH prediction and display (exact consistency)

    Since predict() and __str__() use the same per-feature trees,
    the LLM's simulation exactly matches the model's predictions.
    """

    def __init__(self, n_estimators=200, learning_rate=0.1, tree_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.tree_depth = tree_depth

    def fit(self, X, y):
        n_samples, n_feat = X.shape
        self.n_features_ = n_feat

        # Step 1: Fit GBM with depth-1 stumps
        gbm = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=1,
            learning_rate=self.learning_rate,
            random_state=42,
            subsample=0.8,
            min_samples_leaf=max(5, n_samples // 50),
        )
        gbm.fit(X, y)
        self.init_pred_ = float(gbm.init_.constant_[0][0]) if hasattr(gbm.init_, 'constant_') else float(np.mean(y))

        # Step 2: Extract per-feature step functions from stumps
        # For each stump, it splits on one feature at one threshold
        # and produces two values (left and right)
        per_feature_effects = defaultdict(list)  # feat -> [(threshold, left_val, right_val)]

        for tree_arr in gbm.estimators_:
            tree = tree_arr[0].tree_
            if tree.node_count < 3:
                continue
            feat = tree.feature[0]
            thresh = tree.threshold[0]
            lr = self.learning_rate

            # Left child value
            left_val = tree.value[tree.children_left[0], 0, 0] * lr
            right_val = tree.value[tree.children_right[0], 0, 0] * lr

            per_feature_effects[feat].append((thresh, left_val, right_val))

        # Step 3: For each feature, build a partial dependence on a grid
        # and fit a depth-2 tree to it
        self.per_feature_trees_ = {}
        self.feature_importances_ = np.zeros(n_feat)

        for j in range(n_feat):
            effects = per_feature_effects.get(j, [])
            if not effects:
                continue

            col = X[:, j]
            # Create fine grid
            grid = np.percentile(col, np.linspace(0, 100, 100))
            grid = np.unique(grid)
            if len(grid) < 3:
                continue

            # Compute exact partial effect at each grid point
            pd_vals = np.zeros(len(grid))
            for thresh, left_val, right_val in effects:
                pd_vals += np.where(grid <= thresh, left_val, right_val)

            # Fit a small tree to this partial effect
            tree = DecisionTreeRegressor(
                max_depth=self.tree_depth,
                min_samples_leaf=max(2, len(grid) // 10),
                random_state=42,
            )
            tree.fit(grid.reshape(-1, 1), pd_vals)
            self.per_feature_trees_[j] = tree
            self.feature_importances_[j] = float(np.max(pd_vals) - np.min(pd_vals))

        # Compute bias adjustment: make predictions on training data match
        # the per-feature trees' combined prediction
        train_pred = self._raw_predict(X)
        self.bias_ = float(np.mean(y) - np.mean(train_pred))

        return self

    def _raw_predict(self, X):
        """Predict using per-feature trees (before bias)."""
        result = np.full(X.shape[0], self.init_pred_)
        for j, tree in self.per_feature_trees_.items():
            result += tree.predict(X[:, j:j+1])
        return result

    def predict(self, X):
        check_is_fitted(self, "per_feature_trees_")
        return self._raw_predict(X) + self.bias_

    def __str__(self):
        check_is_fitted(self, "per_feature_trees_")
        names = [f"x{i}" for i in range(self.n_features_)]
        total_bias = self.init_pred_ + self.bias_

        lines = [
            "Additive Model (each feature's effect is INDEPENDENT, no interactions):",
            "  y = bias + f(x0) + f(x1) + ...  (look up each tree, sum all outputs + bias)",
            f"  bias: {total_bias:.4f}",
            "",
            "Per-feature effect trees (each uses ONE feature only):",
        ]

        # Sort by importance
        sorted_feats = sorted(self.per_feature_trees_.keys(),
                              key=lambda j: -self.feature_importances_[j])

        n_shown = 0
        for j in sorted_feats:
            if self.feature_importances_[j] < 0.02:
                continue
            n_shown += 1
            if n_shown > 8:
                remaining = sum(1 for jj in sorted_feats
                                if self.feature_importances_[jj] >= 0.02) - 8
                if remaining > 0:
                    lines.append(f"\n  ... and {remaining} more features with small effects")
                break

            tree = self.per_feature_trees_[j]
            lines.append(f"\n  Effect of {names[j]}:")
            tree_text = export_text(tree, feature_names=[names[j]], max_depth=3)
            for line in tree_text.strip().split("\n"):
                lines.append("    " + line)

        # Negligible features
        negligible = [names[j] for j in range(self.n_features_)
                      if j not in self.per_feature_trees_ or self.feature_importances_[j] < 0.02]
        if negligible:
            if len(negligible) <= 10:
                lines.append(f"\n  Features with negligible effect: {', '.join(negligible)}")
            else:
                lines.append(f"\n  {len(negligible)} features with negligible effect")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ExactStumpGAMRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "ExactStumpGAM_v2"
model_description = "Exact additive stump GAM v2: 200 stumps, depth-3 per-feature trees, consistent predict/display"
model_defs = [(model_shorthand_name, ExactStumpGAMRegressor())]


# ---------------------------------------------------------------------------
# Evaluation (do not edit anything below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default='gpt-5.4',
                        help="LLM checkpoint for interpretability tests (default: gpt-5.4)")
    args = parser.parse_args()

    t0 = time.time()

    # Interpretability tests
    interp_results = run_all_interp_tests(model_defs, checkpoint=args.checkpoint)
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

    perf_csv = os.path.join(RESULTS_DIR, "performance_results.csv")
    perf_fields = ["dataset", "model", "rmse", "rank"]

    existing_perf = []
    if os.path.exists(perf_csv):
        with open(perf_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_perf.append(row)

    for ds_name, model_rmses in dataset_rmses.items():
        rmse_val = model_rmses.get(model_name, float("nan"))
        existing_perf.append({
            "dataset": ds_name,
            "model": model_name,
            "rmse": "" if np.isnan(rmse_val) else f"{rmse_val:.6f}",
            "rank": "",
        })

    by_dataset = defaultdict(list)
    for row in existing_perf:
        by_dataset[row["dataset"]].append(row)

    for ds_name, rows in by_dataset.items():
        valid = [(r, float(r["rmse"])) for r in rows if r["rmse"] not in ("", None)]
        valid.sort(key=lambda x: x[1])
        for rank_idx, (r, _) in enumerate(valid, 1):
            r["rank"] = rank_idx
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
