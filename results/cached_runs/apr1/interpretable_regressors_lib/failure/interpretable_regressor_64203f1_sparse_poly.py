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
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class SparsePolyRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse polynomial regressor: degree-2 polynomial features + LassoCV.

    For datasets with <= 15 features, creates x_i, x_i^2, and x_i*x_j terms.
    LassoCV automatically zeroes out irrelevant terms, yielding a sparse equation.
    For datasets with > 15 features, uses plain RidgeCV (too many poly terms).

    Display shows only non-zero terms as a clean equation, like:
      y = 5.0*x0 + 2.1*x0^2 - 1.3*x0*x1 + 3.2*x1 + intercept
    """

    def __init__(self, poly_threshold=15):
        self.poly_threshold = poly_threshold

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        n = X.shape[1]

        self.use_poly_ = n <= self.poly_threshold
        self.feature_names_ = [f"x{i}" for i in range(n)]

        if self.use_poly_:
            # Build polynomial feature matrix and names
            X_aug, aug_names = self._poly_features(X)
            self.aug_names_ = aug_names
            self.model_ = LassoCV(cv=3, max_iter=10000)
            self.model_.fit(X_aug, y)
            self.coefs_ = self.model_.coef_
            self.intercept_ = self.model_.intercept_
        else:
            # Too many features for polynomial expansion — use Ridge
            self.aug_names_ = self.feature_names_
            self.model_ = RidgeCV(cv=3)
            self.model_.fit(X, y)
            self.coefs_ = self.model_.coef_
            self.intercept_ = self.model_.intercept_

        return self

    def _poly_features(self, X):
        """Create degree-2 polynomial features: linear, quadratic, interactions."""
        n = X.shape[1]
        names = self.feature_names_
        parts = [X]          # linear terms
        aug_names = list(names)

        # Quadratic terms: x_i^2
        for i in range(n):
            parts.append((X[:, i:i+1] ** 2))
            aug_names.append(f"{names[i]}^2")

        # Interaction terms: x_i * x_j
        for i in range(n):
            for j in range(i + 1, n):
                parts.append((X[:, i:i+1] * X[:, j:j+1]))
                aug_names.append(f"{names[i]}*{names[j]}")

        return np.hstack(parts), aug_names

    def predict(self, X):
        check_is_fitted(self, "model_")
        X = np.asarray(X, dtype=np.float64)
        if self.use_poly_:
            X_aug, _ = self._poly_features(X)
            return self.model_.predict(X_aug)
        else:
            return self.model_.predict(X)

    def __str__(self):
        check_is_fitted(self, "model_")

        # Build equation with only non-zero terms
        active_terms = []
        for name, coef in zip(self.aug_names_, self.coefs_):
            if abs(coef) > 1e-6:
                active_terms.append((name, coef))

        equation_parts = [f"{c:.4f}*{n}" for n, c in active_terms]
        equation_parts.append(f"{self.intercept_:.4f}")
        equation = " + ".join(equation_parts)

        if self.use_poly_:
            alpha_str = f"α={self.model_.alpha_:.4g}"
            model_type = "Sparse Polynomial Regression (Lasso, degree 2)"
        else:
            alpha_str = f"α={self.model_.alpha_:.4g}" if hasattr(self.model_, 'alpha_') else ""
            model_type = "Ridge Regression"

        lines = [
            f"{model_type} ({alpha_str} chosen by CV):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]
        for name, coef in active_terms:
            lines.append(f"  {name}: {coef:.4f}")

        # Show zeroed-out features
        zeroed = [n for n, c in zip(self.aug_names_, self.coefs_) if abs(c) <= 1e-6]
        if zeroed and len(zeroed) <= 20:
            lines.append(f"  Zero coefficients (excluded): {', '.join(zeroed)}")

        lines.append(f"  intercept: {self.intercept_:.4f}")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparsePolyRegressor.__module__ = "interpretable_regressor"


# ---------------------------------------------------------------------------
# Evaluation (do not edit any of the evaluation functions, only the names and model descriptions below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    model_defs = [("SparsePoly", SparsePolyRegressor())]

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
    model_name = "SparsePoly"
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
        "model_name":                         "SparsePoly",
        "description":                        "Degree-2 polynomial features + LassoCV sparsity (Ridge fallback for >15 features)",
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

