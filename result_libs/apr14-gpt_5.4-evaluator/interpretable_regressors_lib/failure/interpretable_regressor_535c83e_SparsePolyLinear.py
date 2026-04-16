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
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class SparsePolyLinearRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse polynomial linear regressor.

    Generates degree-2 polynomial features (including interactions),
    then fits a LassoCV to select the most important terms.
    Presents the result as a clean sparse linear equation.
    """

    def __init__(self, max_features=50):
        self.max_features = max_features

    def fit(self, X, y):
        n_samples, n_feat = X.shape
        self.n_original_features_ = n_feat

        # Generate polynomial features (degree 2: x_i, x_i^2, x_i*x_j)
        self.poly_ = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        X_poly = self.poly_.fit_transform(X)

        # Cap features if too many
        if X_poly.shape[1] > self.max_features:
            # Pre-select by correlation with y
            corrs = np.array([abs(np.corrcoef(X_poly[:, j], y)[0, 1])
                              if np.std(X_poly[:, j]) > 1e-10 else 0.0
                              for j in range(X_poly.shape[1])])
            top_idx = np.argsort(corrs)[-self.max_features:]
            self.selected_poly_idx_ = np.sort(top_idx)
            X_poly = X_poly[:, self.selected_poly_idx_]
        else:
            self.selected_poly_idx_ = None

        # Fit LassoCV for automatic regularization + sparsity
        self.lasso_ = LassoCV(cv=3, max_iter=5000, random_state=42)
        self.lasso_.fit(X_poly, y)

        # Store feature names for string representation
        raw_names = [f"x{i}" for i in range(n_feat)]
        all_poly_names = self.poly_.get_feature_names_out(raw_names)
        if self.selected_poly_idx_ is not None:
            self.feature_names_ = all_poly_names[self.selected_poly_idx_]
        else:
            self.feature_names_ = all_poly_names

        return self

    def predict(self, X):
        check_is_fitted(self, "lasso_")
        X_poly = self.poly_.transform(X)
        if self.selected_poly_idx_ is not None:
            X_poly = X_poly[:, self.selected_poly_idx_]
        return self.lasso_.predict(X_poly)

    def __str__(self):
        check_is_fitted(self, "lasso_")
        coefs = self.lasso_.coef_
        intercept = self.lasso_.intercept_
        alpha = self.lasso_.alpha_

        # Collect active (non-zero) terms
        active = [(self.feature_names_[i], coefs[i])
                  for i in range(len(coefs)) if abs(coefs[i]) > 1e-8]

        lines = [f"Sparse Polynomial Linear Regression (L1 regularization, α={alpha:.4g}):"]

        if active:
            # Build equation string
            terms = []
            for name, c in sorted(active, key=lambda x: -abs(x[1])):
                terms.append(f"{c:+.4f}*{name}")
            equation = " ".join(terms) + f" + {intercept:.4f}"
            lines.append(f"  y = {equation}")
            lines.append("")
            lines.append(f"  Active terms ({len(active)} non-zero coefficients, sorted by magnitude):")
            for name, c in sorted(active, key=lambda x: -abs(x[1])):
                lines.append(f"    {name}: {c:.4f}")
            zeroed = [self.feature_names_[i] for i in range(len(coefs)) if abs(coefs[i]) <= 1e-8]
            if zeroed:
                lines.append(f"  Zeroed terms ({len(zeroed)} excluded by L1): {', '.join(zeroed[:10])}" +
                             (f" ... and {len(zeroed)-10} more" if len(zeroed) > 10 else ""))
        else:
            lines.append(f"  y = {intercept:.4f} (all coefficients zeroed)")

        lines.append(f"  intercept: {intercept:.4f}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparsePolyLinearRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparsePolyLinear"
model_description = "Lasso with degree-2 polynomial features (interactions + squares) for nonlinear prediction with sparse linear equation output"
model_defs = [(model_shorthand_name, SparsePolyLinearRegressor())]


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
