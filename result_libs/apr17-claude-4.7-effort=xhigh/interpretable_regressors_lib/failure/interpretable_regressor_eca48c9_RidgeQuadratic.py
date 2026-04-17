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
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class RidgeQuadratic(BaseEstimator, RegressorMixin):
    """
    Ridge regression on linear + quadratic features.

    Features: x_i (standardized) and x_i^2 (standardized), for each i.
    No interaction terms (k*(k-1)/2 would blow up for large datasets).

    Display: OLS-style single equation with active linear + quadratic terms.
    Predict matches display exactly (RidgeCV keeps all coefs; we round in display).
    """

    def __init__(self, cv=3):
        self.cv = cv

    def _build_basis(self, X):
        # No feature scaling here — done inside fit via StandardScaler.
        return np.hstack([X, X * X])

    def fit(self, X, y):
        from sklearn.linear_model import RidgeCV
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.n_features_in_ = X.shape[1]
        self.feature_names_ = [f"x{i}" for i in range(self.n_features_in_)]

        basis = self._build_basis(X)
        self.scaler_ = StandardScaler()
        Bs = self.scaler_.fit_transform(basis)
        self.ridge_ = RidgeCV(cv=self.cv)
        self.ridge_.fit(Bs, y)

        scale = self.scaler_.scale_
        mean = self.scaler_.mean_
        coef_orig = self.ridge_.coef_ / np.where(scale > 0, scale, 1.0)
        intercept_orig = float(self.ridge_.intercept_) - float(np.dot(coef_orig, mean))
        self.coef_ = coef_orig
        self.intercept_ = intercept_orig
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        X = np.asarray(X, dtype=np.float64)
        basis = self._build_basis(X)
        return basis @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_")
        n = self.n_features_in_
        # Term list: linear_i then quadratic_i
        all_terms = []
        for i in range(n):
            all_terms.append(("linear", i, self.coef_[i]))
        for i in range(n):
            all_terms.append(("quadratic", i, self.coef_[n + i]))
        # Sort by |coef|
        all_terms.sort(key=lambda x: -abs(x[2]))

        parts = []
        active_feats = set()
        for kind, i, c in all_terms:
            if abs(c) < 1e-10:
                continue
            nm = self.feature_names_[i]
            active_feats.add(nm)
            if kind == "linear":
                parts.append((f"{c:.4f}*{nm}", nm, c))
            else:
                parts.append((f"{c:.4f}*{nm}^2", f"{nm}^2", c))

        if parts:
            equation = " + ".join(p[0] for p in parts) + f" + {self.intercept_:.4f}"
        else:
            equation = f"{self.intercept_:.4f}"

        alpha = getattr(self.ridge_, "alpha_", 0.0)
        lines = [
            f"Ridge Regression on linear + quadratic features (α={alpha:.4g} via CV):",
            f"  y = {equation}",
            "",
            "Coefficients (sorted by |coef|):",
        ]
        for _, label, c in parts:
            lines.append(f"  {label}: {c:.4f}")
        lines.append(f"  intercept: {self.intercept_:.4f}")

        inactive = [nm for nm in self.feature_names_ if nm not in active_feats]
        if inactive:
            lines.append(f"  Features with zero contribution (neither linear nor quadratic): {', '.join(inactive)}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
RidgeQuadratic.__module__ = "interpretable_regressor"

model_shorthand_name = "RidgeQuadratic"
model_description = "RidgeCV on [x_i, x_i^2] standardized. Quadratic features capture curvature; display is OLS-style clean equation."
model_defs = [(model_shorthand_name, RidgeQuadratic())]


# ---------------------------------------------------------------------------
# Evaluation (do not edit anything below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default='gpt-4o',
                        help="LLM checkpoint for interpretability tests (default: gpt-4o)")
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
