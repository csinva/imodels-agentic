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
from sklearn.linear_model import RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class HingeBasisRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Hinge Basis Ridge: discovers piecewise-linear basis functions from data
    using shallow decision tree stumps, then fits Ridge regression on them.

    For each feature, we find optimal split thresholds using a depth-1 tree,
    creating hinge basis functions: max(0, x_i - t) and max(0, t - x_i).
    The original features are also included. Ridge regression is fit on the
    expanded basis. The result is displayed as a linear model with named
    piecewise-linear components.
    """

    def __init__(self, n_knots=3, alpha_range=(0.01, 0.1, 1.0, 10.0, 100.0)):
        self.n_knots = n_knots
        self.alpha_range = alpha_range

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.knots_ = {}  # {feature_idx: [thresholds]}

        # Discover knots per feature using small decision trees
        for j in range(n_features):
            tree = DecisionTreeRegressor(max_depth=2, min_samples_leaf=max(5, n_samples // 20), random_state=42)
            tree.fit(X[:, j:j+1], y)
            thresholds = []
            for node_idx in range(tree.tree_.node_count):
                if tree.tree_.children_left[node_idx] != -1:  # internal node
                    thresholds.append(float(tree.tree_.threshold[node_idx]))
            # Also add quantile-based knots
            for q in np.linspace(0.2, 0.8, self.n_knots):
                thresholds.append(float(np.quantile(X[:, j], q)))
            # Deduplicate and sort
            thresholds = sorted(set(round(t, 6) for t in thresholds))
            self.knots_[j] = thresholds

        # Build expanded basis
        X_basis = self._build_basis(X)
        self.basis_names_ = self._get_basis_names()

        # Fit Ridge on expanded basis
        self.ridge_ = RidgeCV(alphas=list(self.alpha_range), cv=5)
        self.ridge_.fit(X_basis, y)
        self.coef_ = self.ridge_.coef_
        self.intercept_ = self.ridge_.intercept_
        return self

    def _build_basis(self, X):
        X = np.asarray(X, dtype=np.float64)
        cols = [X]  # original features
        for j in range(self.n_features_):
            for t in self.knots_.get(j, []):
                cols.append(np.maximum(0, X[:, j:j+1] - t))  # hinge+
                cols.append(np.maximum(0, t - X[:, j:j+1]))  # hinge-
        return np.hstack(cols)

    def _get_basis_names(self):
        names = [f"x{j}" for j in range(self.n_features_)]
        for j in range(self.n_features_):
            for t in self.knots_.get(j, []):
                names.append(f"max(0, x{j} - {t:.3f})")
                names.append(f"max(0, {t:.3f} - x{j})")
        return names

    def predict(self, X):
        check_is_fitted(self, "ridge_")
        X_basis = self._build_basis(np.asarray(X, dtype=np.float64))
        return self.ridge_.predict(X_basis)

    def __str__(self):
        check_is_fitted(self, "ridge_")
        lines = [
            f"Hinge Basis Ridge Regression (alpha={self.ridge_.alpha_:.4g}):",
            f"  y = intercept + sum of (coefficient * basis_function)",
            f"  intercept: {self.intercept_:.4f}",
            "",
            "Coefficients (sorted by |coefficient|, showing top active terms):",
        ]
        # Sort by absolute coefficient value
        pairs = list(zip(self.basis_names_, self.coef_))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        # Show top terms (skip near-zero)
        shown = 0
        for name, coef in pairs:
            if abs(coef) < 1e-4:
                continue
            lines.append(f"  {coef:+.4f} * {name}")
            shown += 1
            if shown >= 25:
                remaining = sum(1 for _, c in pairs[shown:] if abs(c) >= 1e-4)
                if remaining > 0:
                    lines.append(f"  ... ({remaining} more small terms)")
                break
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
HingeBasisRidgeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "HingeBasisRidge_v1"
model_description = "Ridge regression on hinge basis functions with data-driven knots from stumps + quantiles"
model_defs = [(model_shorthand_name, HingeBasisRidgeRegressor())]


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

