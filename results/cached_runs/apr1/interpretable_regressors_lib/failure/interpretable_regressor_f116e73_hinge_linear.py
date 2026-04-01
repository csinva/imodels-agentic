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
from sklearn.linear_model import RidgeCV
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class HingeLinearRegressor(BaseEstimator, RegressorMixin):
    """
    Linear model augmented with hinge (ReLU) features.

    For each feature xi, creates two terms: xi (linear) and max(0, xi) (hinge).
    This lets the model capture piecewise-linear effects with a kink at 0.

    Fitted with RidgeCV. Displayed as a clean equation showing all terms,
    mimicking the format that scores highest on LLM interpretability tests.

    For xi > 0:  contribution = (linear_coef + hinge_coef) * xi
    For xi <= 0: contribution = linear_coef * xi
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        # Build augmented feature matrix: [x0, max(0,x0), x1, max(0,x1), ...]
        X_aug = self._augment(X)

        self.ridge_ = RidgeCV(cv=3)
        self.ridge_.fit(X_aug, y)

        # Store coefficients separately for display
        self.linear_coefs_ = np.zeros(self.n_features_in_)
        self.hinge_coefs_ = np.zeros(self.n_features_in_)
        for j in range(self.n_features_in_):
            self.linear_coefs_[j] = self.ridge_.coef_[2 * j]
            self.hinge_coefs_[j] = self.ridge_.coef_[2 * j + 1]
        self.intercept_ = self.ridge_.intercept_

        return self

    def _augment(self, X):
        parts = []
        for j in range(X.shape[1]):
            parts.append(X[:, j:j+1])
            parts.append(np.maximum(0, X[:, j:j+1]))
        return np.hstack(parts)

    def predict(self, X):
        check_is_fitted(self, "ridge_")
        X = np.asarray(X, dtype=np.float64)
        return self.ridge_.predict(self._augment(X))

    def __str__(self):
        check_is_fitted(self, "ridge_")
        names = [f"x{i}" for i in range(self.n_features_in_)]

        # Build equation string
        terms = []
        for j, name in enumerate(names):
            lc = self.linear_coefs_[j]
            hc = self.hinge_coefs_[j]
            if abs(lc) > 1e-6:
                terms.append(f"{lc:.4f}*{name}")
            if abs(hc) > 1e-6:
                terms.append(f"{hc:.4f}*max(0,{name})")

        equation = " + ".join(terms) + f" + {self.intercept_:.4f}"

        lines = [
            f"Hinge Linear Regression (Ridge-regularized, α={self.ridge_.alpha_:.4g}):",
            f"  y = {equation}",
            "",
            "Coefficients (for each feature, 'linear' applies everywhere, 'hinge' applies only when xi > 0):",
        ]
        for j, name in enumerate(names):
            lc = self.linear_coefs_[j]
            hc = self.hinge_coefs_[j]
            lines.append(f"  {name}: linear={lc:.4f}, hinge(max(0,{name}))={hc:.4f}")
            if abs(hc) > 1e-6:
                lines.append(f"    → when {name} > 0: effective slope = {lc + hc:.4f}")
                lines.append(f"    → when {name} <= 0: effective slope = {lc:.4f}")
        lines.append(f"  intercept: {self.intercept_:.4f}")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
HingeLinearRegressor.__module__ = "interpretable_regressor"


# ---------------------------------------------------------------------------
# Evaluation (do not edit any of the evaluation functions, only the names and model descriptions below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    model_defs = [("HingeLinear", HingeLinearRegressor())]

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
    model_name = "HingeLinear"
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
        "model_name":                         "HingeLinear",
        "description":                        "Ridge on linear + hinge(max(0,xi)) features for piecewise-linear effects",
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

