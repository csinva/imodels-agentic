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
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class NormalEqRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Ridge Regression from scratch via normal equations with LOO-CV alpha.

    Solves: coef = (X^T X + alpha * I)^{-1} X^T y
    Alpha is selected from a grid using leave-one-out cross-validation.

    Novel implementation: direct normal equation solve with eigendecomposition
    for efficient LOO-CV (uses the shortcut: LOO residual_i = e_i / (1 - h_ii)
    where h_ii are diagonal elements of the hat matrix).
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        # Center data
        self.x_mean_ = np.mean(X, axis=0)
        self.y_mean_ = float(np.mean(y))
        Xc = X - self.x_mean_
        yc = y - self.y_mean_

        # Eigendecomposition of X^T X for efficient multi-alpha solve
        XtX = Xc.T @ Xc
        eigvals, eigvecs = np.linalg.eigh(XtX)
        Xty = Xc.T @ yc

        # Transform to eigen-space
        Xty_eigen = eigvecs.T @ Xty
        X_eigen = Xc @ eigvecs  # n x p matrix in eigenspace

        # LOO-CV for alpha selection
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        best_alpha = 1.0
        best_loo_mse = float('inf')

        for alpha in alphas:
            # Coefficients in eigenspace: coef_eigen_j = Xty_eigen_j / (eigval_j + alpha)
            d = eigvals + alpha  # eigenvalues + alpha
            coef_eigen = Xty_eigen / d

            # Hat matrix diagonal: h_ii = sum_j (X_eigen_ij^2 / d_j)
            H_diag = np.sum(X_eigen ** 2 / d[np.newaxis, :], axis=1)

            # Predictions and residuals
            preds = X_eigen @ coef_eigen
            residuals = yc - preds

            # LOO residuals: r_i / (1 - h_ii)
            loo_residuals = residuals / np.maximum(1 - H_diag, 1e-10)
            loo_mse = float(np.mean(loo_residuals ** 2))

            if loo_mse < best_loo_mse:
                best_loo_mse = loo_mse
                best_alpha = alpha

        # Final fit with best alpha
        d = eigvals + best_alpha
        coef_eigen = Xty_eigen / d
        self.coef_ = eigvecs @ coef_eigen
        self.intercept_ = float(self.y_mean_ - self.x_mean_ @ self.coef_)
        self.alpha_ = best_alpha
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_")
        names = [f"x{i}" for i in range(self.n_features_)]
        equation = " + ".join(f"{c:.4f}*{n}" for c, n in zip(self.coef_, names))
        equation += f" + {self.intercept_:.4f}"

        lines = [
            f"Ridge Regression (L2 regularization, α={self.alpha_:.4g} chosen by CV):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]
        for n, c in zip(names, self.coef_):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {self.intercept_:.4f}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
NormalEqRidgeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "NormalEqRidge_v1"
model_description = "Ridge via eigendecomposition with efficient LOO-CV alpha selection, RidgeCV display"
model_defs = [(model_shorthand_name, NormalEqRidgeRegressor())]


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

