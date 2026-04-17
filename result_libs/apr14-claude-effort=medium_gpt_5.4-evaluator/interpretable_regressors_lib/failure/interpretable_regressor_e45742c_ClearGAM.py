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
from pygam import LinearGAM, s
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class ClearGAMRegressor(BaseEstimator, RegressorMixin):
    """
    GAM with clear piecewise-constant string representation.

    Uses PyGAM's spline-based fitting for powerful prediction, but
    presents each feature's effect as a simple step function that the
    LLM can exactly evaluate.

    For each feature:
    1. Fit the GAM to learn smooth shape functions
    2. Discretize each shape function into 5 intervals
    3. Display as: "if x0 <= -1.2: effect = -3.4; if -1.2 < x0 <= 0.5: effect = 1.2; ..."

    The prediction uses the actual GAM (smooth splines) for accuracy,
    but the string shows a piecewise-constant approximation that the
    LLM can trace exactly.

    Actually, to ensure consistency between __str__ and predict(), we
    convert to a TRUE piecewise-constant model after fitting.
    """

    def __init__(self, n_splines=10, n_bins=5):
        self.n_splines = n_splines
        self.n_bins = n_bins

    def fit(self, X, y):
        n_samples, n_feat = X.shape
        self.n_features_ = n_feat

        # Step 1: Fit PyGAM to learn shape functions
        gam = LinearGAM(n_splines=self.n_splines)
        gam.fit(X, y)

        # Step 2: Convert to piecewise-constant representation
        # For each feature, evaluate the partial dependence on a grid
        # and discretize into n_bins intervals
        self.intercept_ = float(gam.coef_[-1])
        self.feature_effects_ = []
        self.feature_edges_ = []

        for j in range(n_feat):
            col = X[:, j]
            # Compute bin edges from training data quantiles
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            edges = np.unique(np.percentile(col, percentiles))

            if len(edges) < 2:
                # Constant feature
                self.feature_effects_.append(np.array([0.0]))
                self.feature_edges_.append(edges)
                continue

            # Compute effect at midpoints of each bin
            effects = []
            for i in range(len(edges) - 1):
                mid = (edges[i] + edges[i + 1]) / 2
                # Create a sample with this feature at mid, others at median
                x_eval = np.median(X, axis=0).reshape(1, -1).repeat(1, axis=0)
                x_eval[0, j] = mid
                pred_at_mid = gam.predict(x_eval)[0]

                # Also evaluate at the overall median
                x_median = np.median(X, axis=0).reshape(1, -1)
                pred_at_median = gam.predict(x_median)[0]

                effects.append(pred_at_mid - pred_at_median)

            self.feature_effects_.append(np.array(effects))
            self.feature_edges_.append(edges)

        # Recompute intercept to match the piecewise model
        # The intercept should be the prediction when all features are at median
        x_median = np.median(X, axis=0).reshape(1, -1)
        self.intercept_ = float(gam.predict(x_median)[0])

        return self

    def predict(self, X):
        check_is_fitted(self, "feature_effects_")
        n = X.shape[0]
        result = np.full(n, self.intercept_)

        for j in range(self.n_features_):
            edges = self.feature_edges_[j]
            effects = self.feature_effects_[j]
            if len(edges) < 2:
                continue

            col = X[:, j]
            # Assign each sample to a bin
            bin_idx = np.digitize(col, edges[1:-1])
            bin_idx = np.clip(bin_idx, 0, len(effects) - 1)
            result += effects[bin_idx]

        return result

    def __str__(self):
        check_is_fitted(self, "feature_effects_")
        names = [f"x{i}" for i in range(self.n_features_)]

        lines = [
            "Additive Model (each feature's effect is INDEPENDENT):",
            "  y = intercept + f(x0) + f(x1) + ...  (look up each feature's effect, sum them)",
            f"  intercept: {self.intercept_:.4f}",
            "",
            "Per-feature effects (find the interval containing the value, use that effect):",
        ]

        for j in range(self.n_features_):
            edges = self.feature_edges_[j]
            effects = self.feature_effects_[j]

            if len(edges) < 2:
                continue

            # Check if this feature has meaningful effect
            if max(abs(effects)) < 0.02:
                lines.append(f"\n  {names[j]}: negligible effect (all intervals near 0)")
                continue

            lines.append(f"\n  {names[j]}:")
            for i in range(len(effects)):
                if i == 0:
                    lines.append(f"    {names[j]} <= {edges[1]:.2f}  →  effect = {effects[i]:+.4f}")
                elif i == len(effects) - 1:
                    lines.append(f"    {names[j]} > {edges[-2]:.2f}  →  effect = {effects[i]:+.4f}")
                else:
                    lines.append(f"    {edges[i]:.2f} < {names[j]} <= {edges[i+1]:.2f}  →  effect = {effects[i]:+.4f}")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ClearGAMRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "ClearGAM"
model_description = "PyGAM-based additive model with piecewise-constant per-feature effects, clear lookup table output"
model_defs = [(model_shorthand_name, ClearGAMRegressor())]


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
