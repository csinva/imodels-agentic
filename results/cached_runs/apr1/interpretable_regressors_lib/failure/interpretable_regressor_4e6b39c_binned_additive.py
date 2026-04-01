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
    Additive model with binned features.

    For each feature, bins values into n_bins quantile-based bins and learns
    a per-bin effect. The prediction is:

        y = intercept + effect_0(x0) + effect_1(x1) + ...

    where effect_i(xi) is looked up from the bin that xi falls into.
    This captures nonlinear effects while being trivially traceable:
    the LLM just looks up each feature's bin and sums the effects.

    Fitting uses one-hot encoded bins with Ridge regularization.
    """

    def __init__(self, n_bins=5, alpha=1.0):
        self.n_bins = n_bins
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        # Compute bin edges per feature using quantiles
        self.bin_edges_ = []
        for j in range(X.shape[1]):
            col = X[:, j]
            quantiles = np.linspace(0, 100, self.n_bins + 1)[1:-1]
            edges = np.unique(np.percentile(col, quantiles))
            self.bin_edges_.append(edges)

        # Build one-hot bin matrix
        Z = self._transform(X)

        # Ridge regression on the bin features
        from sklearn.linear_model import Ridge
        self.ridge_ = Ridge(alpha=self.alpha, fit_intercept=True)
        self.ridge_.fit(Z, y)

        # Store per-feature bin effects for display and prediction
        self.intercept_ = self.ridge_.intercept_
        self.bin_effects_ = []
        col_idx = 0
        for j in range(X.shape[1]):
            n_bins_j = len(self.bin_edges_[j]) + 1
            effects = self.ridge_.coef_[col_idx:col_idx + n_bins_j]
            self.bin_effects_.append(effects)
            col_idx += n_bins_j

        return self

    def _transform(self, X):
        """Convert X to one-hot bin matrix."""
        parts = []
        for j in range(X.shape[1]):
            edges = self.bin_edges_[j]
            bins = np.digitize(X[:, j], edges)  # 0..len(edges)
            n_bins_j = len(edges) + 1
            onehot = np.zeros((X.shape[0], n_bins_j))
            for i in range(X.shape[0]):
                onehot[i, bins[i]] = 1.0
            parts.append(onehot)
        return np.hstack(parts)

    def predict(self, X):
        check_is_fitted(self, "ridge_")
        X = np.asarray(X, dtype=np.float64)
        Z = self._transform(X)
        return self.ridge_.predict(Z)

    def __str__(self):
        check_is_fitted(self, "ridge_")
        names = [f"x{i}" for i in range(self.n_features_in_)]

        lines = [
            "Binned Additive Model (each feature's effect is INDEPENDENT):",
            f"  y = intercept + effect(x0) + effect(x1) + ...",
            f"  intercept: {self.intercept_:.4f}",
            "",
            "Per-feature effects (look up the bin each feature value falls into):",
        ]

        for j, (name, edges, effects) in enumerate(
            zip(names, self.bin_edges_, self.bin_effects_)
        ):
            lines.append(f"\n  {name}:")
            n_bins_j = len(edges) + 1
            for b in range(n_bins_j):
                if b == 0:
                    label = f"{name} < {edges[0]:.2f}" if len(edges) > 0 else "any"
                elif b == n_bins_j - 1:
                    label = f"{name} >= {edges[-1]:.2f}"
                else:
                    label = f"{edges[b-1]:.2f} <= {name} < {edges[b]:.2f}"
                lines.append(f"    {label}  →  effect = {effects[b]:+.4f}")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
BinnedAdditiveRegressor.__module__ = "interpretable_regressor"


# ---------------------------------------------------------------------------
# Evaluation (do not edit any of the evaluation functions, only the names and model descriptions below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    model_defs = [("BinnedAdditive", BinnedAdditiveRegressor())]

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
    model_name = "BinnedAdditive"
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
        "model_name":                         "BinnedAdditive",
        "description":                        "Additive model with 5 quantile bins per feature, Ridge-regularized",
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

