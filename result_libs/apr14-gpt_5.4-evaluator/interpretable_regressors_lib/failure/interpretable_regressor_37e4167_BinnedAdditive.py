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
from sklearn.linear_model import RidgeCV
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class BinnedAdditiveRegressor(BaseEstimator, RegressorMixin):
    """
    Additive piecewise-constant regressor using quantile binning.

    For each feature:
      1. Compute quantile bin edges (n_bins bins)
      2. One-hot encode which bin the feature falls in
    Then fit a RidgeCV on the one-hot features.

    The result is an additive model: y = intercept + sum of per-feature bin effects.
    String representation shows a lookup table for each feature.
    """

    def __init__(self, n_bins=5):
        self.n_bins = n_bins

    def fit(self, X, y):
        n_samples, n_feat = X.shape
        self.n_features_ = n_feat

        # Compute quantile bin edges for each feature
        self.bin_edges_ = []
        for j in range(n_feat):
            col = X[:, j]
            # Compute quantile edges, handle duplicates
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            edges = np.percentile(col, percentiles)
            # Remove duplicate edges
            edges = np.unique(edges)
            self.bin_edges_.append(edges)

        # Create binned features
        X_binned = self._transform(X)

        # Fit ridge regression on binned features
        self.ridge_ = RidgeCV(cv=3)
        self.ridge_.fit(X_binned, y)

        return self

    def _transform(self, X):
        """Transform X into one-hot binned features."""
        all_bins = []
        for j in range(self.n_features_):
            col = X[:, j]
            edges = self.bin_edges_[j]
            # Digitize: assign bin index (0 to n_bins-1)
            bin_idx = np.digitize(col, edges[1:-1])  # n_bins-1 thresholds
            n_actual_bins = len(edges) - 1
            # One-hot encode
            one_hot = np.zeros((len(col), n_actual_bins))
            for i in range(len(col)):
                b = min(bin_idx[i], n_actual_bins - 1)
                one_hot[i, b] = 1.0
            all_bins.append(one_hot)
        return np.hstack(all_bins)

    def predict(self, X):
        check_is_fitted(self, "ridge_")
        X_binned = self._transform(X)
        return self.ridge_.predict(X_binned)

    def __str__(self):
        check_is_fitted(self, "ridge_")
        coefs = self.ridge_.coef_
        intercept = self.ridge_.intercept_
        names = [f"x{i}" for i in range(self.n_features_)]

        lines = [
            "Binned Additive Model (piecewise-constant per feature):",
            "  y = intercept + f(x0) + f(x1) + ...  (each feature's effect is INDEPENDENT)",
            f"  intercept: {intercept:.4f}",
            "",
            "Per-feature effects (look up the bin containing the feature value, sum all effects):",
        ]

        idx = 0
        for j in range(self.n_features_):
            edges = self.bin_edges_[j]
            n_actual_bins = len(edges) - 1
            feature_coefs = coefs[idx:idx + n_actual_bins]
            idx += n_actual_bins

            # Check if this feature has meaningful effect
            effect_range = max(feature_coefs) - min(feature_coefs) if len(feature_coefs) > 1 else 0
            if effect_range < 0.01 and all(abs(c) < 0.05 for c in feature_coefs):
                lines.append(f"\n  {names[j]}: negligible effect (all bins ≈ 0)")
                continue

            lines.append(f"\n  {names[j]}:")
            for b in range(n_actual_bins):
                lo = edges[b]
                hi = edges[b + 1]
                if b == 0:
                    lines.append(f"    {names[j]} ≤ {hi:+.2f}  →  effect = {feature_coefs[b]:+.4f}")
                elif b == n_actual_bins - 1:
                    lines.append(f"    {names[j]} > {lo:+.2f}  →  effect = {feature_coefs[b]:+.4f}")
                else:
                    lines.append(f"    {lo:+.2f} < {names[j]} ≤ {hi:+.2f}  →  effect = {feature_coefs[b]:+.4f}")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
BinnedAdditiveRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "BinnedAdditive"
model_description = "Additive piecewise-constant model using quantile bins per feature with ridge regression, presented as lookup table"
model_defs = [(model_shorthand_name, BinnedAdditiveRegressor())]


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
