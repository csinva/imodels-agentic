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
from sklearn.tree import export_text
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class BoostedStumpRegressor(BaseEstimator, RegressorMixin):
    """
    Gradient-boosted depth-1 stumps with interpretable additive output.

    Since each stump splits on exactly one feature at one threshold,
    the ensemble is purely additive: y = bias + f_0(x0) + f_1(x1) + ...
    with no feature interactions.

    After fitting, we compute the aggregate partial effect for each feature
    by evaluating the ensemble's prediction across a grid of values for that
    feature, holding others at their median. This produces a step-function
    per feature that we display as a lookup table.

    Combines the predictive power of gradient boosting with the readability
    of an additive model.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, min_samples_leaf=10):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        n_samples, n_feat = X.shape
        self.n_features_ = n_feat

        # Fit GBM with depth-1 stumps (no interactions)
        self.gbm_ = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=1,
            learning_rate=self.learning_rate,
            min_samples_leaf=max(self.min_samples_leaf, n_samples // 50),
            random_state=42,
            subsample=0.8,
        )
        self.gbm_.fit(X, y)

        # Store training data statistics for partial effect computation
        self.X_median_ = np.median(X, axis=0)
        self.X_train_ = X  # needed for partial effects

        # Precompute partial effects for each feature
        self._compute_partial_effects(X)

        return self

    def _compute_partial_effects(self, X):
        """Compute piecewise-constant partial effects per feature."""
        self.partial_effects_ = []

        for j in range(self.n_features_):
            # Collect all split thresholds used for this feature
            thresholds = set()
            for tree_arr in self.gbm_.estimators_:
                tree = tree_arr[0].tree_
                for node_id in range(tree.node_count):
                    if (tree.children_left[node_id] != -1 and
                            tree.feature[node_id] == j):
                        thresholds.add(float(tree.threshold[node_id]))

            if not thresholds:
                self.partial_effects_.append(None)  # feature not used
                continue

            # Create evaluation grid: points between thresholds
            thresholds = sorted(thresholds)

            # Build evaluation points (midpoints of threshold regions)
            eval_points = []
            eval_points.append(thresholds[0] - 1.0)  # below lowest threshold
            for i in range(len(thresholds) - 1):
                eval_points.append((thresholds[i] + thresholds[i + 1]) / 2)
            eval_points.append(thresholds[-1] + 1.0)  # above highest threshold

            # Evaluate partial effect at each point
            # Use the median of other features
            X_eval = np.tile(self.X_median_, (len(eval_points), 1))
            for i, val in enumerate(eval_points):
                X_eval[i, j] = val

            preds = self.gbm_.predict(X_eval)
            baseline = self.gbm_.predict(self.X_median_.reshape(1, -1))[0]
            effects = preds - baseline

            self.partial_effects_.append({
                'thresholds': thresholds,
                'eval_points': eval_points,
                'effects': effects.tolist(),
            })

    def predict(self, X):
        check_is_fitted(self, "gbm_")
        return self.gbm_.predict(X)

    def __str__(self):
        check_is_fitted(self, "gbm_")
        names = [f"x{i}" for i in range(self.n_features_)]

        bias = float(self.gbm_.predict(self.X_median_.reshape(1, -1))[0])

        lines = [
            "Additive Boosted Stump Model (no feature interactions):",
            "  y = bias + f(x0) + f(x1) + ...  (each feature's effect is INDEPENDENT)",
            f"  bias (at median inputs): {bias:.4f}",
            "",
            "Per-feature partial effects (look up the interval, add the effect to bias):",
        ]

        # Sort features by importance
        importances = self.gbm_.feature_importances_
        feat_order = np.argsort(importances)[::-1]

        n_shown = 0
        for j in feat_order:
            pe = self.partial_effects_[j]
            if pe is None:
                continue

            thresholds = pe['thresholds']
            effects = pe['effects']
            effect_range = max(effects) - min(effects)

            if effect_range < 0.01:
                continue  # skip negligible features

            n_shown += 1
            lines.append(f"\n  {names[j]} (importance: {importances[j]:.3f}):")

            # Show effect for each region
            for i, effect in enumerate(effects):
                if i == 0:
                    lines.append(f"    {names[j]} <= {thresholds[0]:.2f}  →  effect = {effect:+.4f}")
                elif i == len(effects) - 1:
                    lines.append(f"    {names[j]} > {thresholds[-1]:.2f}  →  effect = {effect:+.4f}")
                else:
                    lines.append(f"    {thresholds[i-1]:.2f} < {names[j]} <= {thresholds[i]:.2f}  →  effect = {effect:+.4f}")

        # List unused features
        unused = [names[j] for j in range(self.n_features_) if self.partial_effects_[j] is None]
        if unused:
            lines.append(f"\n  Unused features (no effect): {', '.join(unused)}")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
BoostedStumpRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "BoostedStump"
model_description = "Gradient-boosted depth-1 stumps (additive, no interactions) with per-feature partial effect lookup table"
model_defs = [(model_shorthand_name, BoostedStumpRegressor())]


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
