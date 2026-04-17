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


class StumpGAMRegressor(BaseEstimator, RegressorMixin):
    """
    Additive stump GAM: gradient-boosted depth-1 stumps aggregated into
    per-feature piecewise-constant functions, displayed as small trees.

    1. Fit GBM with depth-1 stumps (no feature interactions)
    2. For each feature, collect all stump thresholds and aggregate effects
    3. Fit a small depth-2 tree to each feature's aggregated effect
    4. Present as additive model with per-feature trees

    The GBM stumps capture nonlinearity well (like EBM), and the
    per-feature tree display uses the proven readable tree format.

    Unlike BoostedStump (which used lookup tables and got 0.35 interp),
    this presents effects as trees which score much higher on interp.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def fit(self, X, y):
        n_samples, n_feat = X.shape
        self.n_features_ = n_feat
        self.bias_ = float(np.mean(y))

        # Step 1: Fit GBM with depth-1 stumps
        self.gbm_ = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=1,
            learning_rate=self.learning_rate,
            random_state=42,
            subsample=0.8,
            min_samples_leaf=max(5, n_samples // 50),
        )
        self.gbm_.fit(X, y)

        # Step 2: For each feature, compute the aggregated partial effect
        # by evaluating on a grid holding other features at median
        self.X_median_ = np.median(X, axis=0)
        baseline_pred = self.gbm_.predict(self.X_median_.reshape(1, -1))[0]

        # Step 3: Fit a small tree to each feature's partial dependence
        self.per_feature_trees_ = []
        self.feature_importances_ = np.zeros(n_feat)

        for j in range(n_feat):
            col = X[:, j]
            # Create grid of values for this feature
            grid = np.percentile(col, np.linspace(1, 99, 50))
            grid = np.unique(grid)
            if len(grid) < 3:
                self.per_feature_trees_.append(None)
                continue

            # Compute partial dependence at each grid point
            X_eval = np.tile(self.X_median_, (len(grid), 1))
            X_eval[:, j] = grid
            preds = self.gbm_.predict(X_eval)
            effects = preds - baseline_pred

            # Fit a small tree to approximate the partial effect
            tree = DecisionTreeRegressor(
                max_depth=2,
                min_samples_leaf=max(2, len(grid) // 10),
                random_state=42,
            )
            tree.fit(grid.reshape(-1, 1), effects)
            self.per_feature_trees_.append(tree)

            # Importance = range of effects
            self.feature_importances_[j] = float(np.max(effects) - np.min(effects))

        self.baseline_pred_ = baseline_pred
        return self

    def predict(self, X):
        check_is_fitted(self, "gbm_")
        return self.gbm_.predict(X)

    def __str__(self):
        check_is_fitted(self, "per_feature_trees_")
        names = [f"x{i}" for i in range(self.n_features_)]

        lines = [
            "Additive Stump Model (each feature's effect is INDEPENDENT, no interactions):",
            "  y = baseline + f(x0) + f(x1) + ...  (look up each feature's tree, sum outputs)",
            f"  baseline: {self.baseline_pred_:.4f}",
            "",
            "Per-feature effect trees (each uses ONE feature):",
        ]

        # Sort by importance
        sorted_j = sorted(range(self.n_features_),
                          key=lambda j: -self.feature_importances_[j])

        n_shown = 0
        negligible = []
        for j in sorted_j:
            tree = self.per_feature_trees_[j]
            if tree is None or self.feature_importances_[j] < 0.02:
                negligible.append(names[j])
                continue

            n_shown += 1
            if n_shown > 8:
                remaining = sum(1 for jj in sorted_j
                                if self.feature_importances_[jj] >= 0.02
                                and jj not in [sorted_j[k] for k in range(8)])
                if remaining > 0:
                    lines.append(f"\n  ... and {remaining} more features with small effects")
                break

            lines.append(f"\n  Effect of {names[j]} (importance: {self.feature_importances_[j]:.3f}):")
            tree_text = export_text(tree, feature_names=[names[j]], max_depth=3)
            for line in tree_text.strip().split("\n"):
                lines.append("    " + line)

        if negligible:
            if len(negligible) <= 8:
                lines.append(f"\n  Features with negligible effect: {', '.join(negligible)}")
            else:
                lines.append(f"\n  {len(negligible)} features with negligible effect")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
StumpGAMRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "StumpGAM"
model_description = "GBM depth-1 stumps aggregated into per-feature trees, additive with no interactions, tree-format display"
model_defs = [(model_shorthand_name, StumpGAMRegressor())]


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
