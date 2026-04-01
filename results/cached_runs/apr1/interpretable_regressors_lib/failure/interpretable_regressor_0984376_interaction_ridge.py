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
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class InteractionRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Standardized Ridge regression with top-k pairwise interaction terms.

    Steps:
    1. Standardize all features
    2. Identify top interaction pairs by correlation of x_i*x_j with residual
    3. Fit RidgeCV on [original features + interaction features]
    4. Display as a clean equation in original feature scale

    This extends Ridge with interaction capture while keeping the simple
    linear equation format that LLMs can easily trace.
    """

    def __init__(self, max_interactions=5):
        self.max_interactions = max_interactions

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        n_features = X.shape[1]

        # Standardize
        self.scaler_ = StandardScaler()
        X_std = self.scaler_.fit_transform(X)

        # Initial Ridge fit to get residuals for interaction selection
        ridge_init = RidgeCV(cv=3)
        ridge_init.fit(X_std, y)
        residuals = y - ridge_init.predict(X_std)

        # Find top interaction pairs by correlation with residual
        self.interaction_pairs_ = []
        if n_features >= 2 and self.max_interactions > 0:
            candidates = []
            for i in range(min(n_features, 20)):
                for j in range(i + 1, min(n_features, 20)):
                    interaction = X_std[:, i] * X_std[:, j]
                    corr = abs(np.corrcoef(interaction, residuals)[0, 1])
                    if not np.isnan(corr):
                        candidates.append((corr, i, j))
            candidates.sort(reverse=True)
            self.interaction_pairs_ = [(i, j) for _, i, j in candidates[:self.max_interactions]]

        # Build final feature matrix
        X_final = self._build_features(X_std)

        # Fit final Ridge
        self.ridge_ = RidgeCV(cv=3)
        self.ridge_.fit(X_final, y)

        # Convert coefficients to original scale for display
        self._compute_original_coefs()
        return self

    def _build_features(self, X_std):
        parts = [X_std]
        for i, j in self.interaction_pairs_:
            parts.append((X_std[:, i] * X_std[:, j]).reshape(-1, 1))
        return np.hstack(parts)

    def _compute_original_coefs(self):
        """Convert standardized coefficients to original feature scale."""
        n = self.n_features_in_
        std = self.scaler_.scale_
        mean = self.scaler_.mean_

        # Coefficients for main features (in standardized space)
        w_std = self.ridge_.coef_[:n]

        # Original-scale coefficients: w_orig_i = w_std_i / std_i
        self.coefs_orig_ = w_std / std

        # Intercept adjustment
        self.intercept_orig_ = self.ridge_.intercept_ - np.sum(w_std * mean / std)

        # Interaction coefficients (in standardized space → original scale)
        self.interaction_coefs_orig_ = []
        for k, (i, j) in enumerate(self.interaction_pairs_):
            w_int = self.ridge_.coef_[n + k]
            # x_i_std * x_j_std = (x_i - mu_i)/s_i * (x_j - mu_j)/s_j
            # = x_i*x_j/(s_i*s_j) - x_i*mu_j/(s_i*s_j) - x_j*mu_i/(s_i*s_j) + mu_i*mu_j/(s_i*s_j)
            coef_ij = w_int / (std[i] * std[j])
            # Adjustments to linear terms and intercept
            self.coefs_orig_[i] -= w_int * mean[j] / (std[i] * std[j])
            self.coefs_orig_[j] -= w_int * mean[i] / (std[i] * std[j])
            self.intercept_orig_ += w_int * mean[i] * mean[j] / (std[i] * std[j])
            self.interaction_coefs_orig_.append(coef_ij)

    def predict(self, X):
        check_is_fitted(self, "ridge_")
        X = np.asarray(X, dtype=np.float64)
        X_std = self.scaler_.transform(X)
        X_final = self._build_features(X_std)
        return self.ridge_.predict(X_final)

    def __str__(self):
        check_is_fitted(self, "ridge_")
        names = [f"x{i}" for i in range(self.n_features_in_)]

        # Build equation
        terms = []
        for j, name in enumerate(names):
            c = self.coefs_orig_[j]
            if abs(c) > 1e-8:
                terms.append(f"{c:.4f}*{name}")
        for k, (i, j) in enumerate(self.interaction_pairs_):
            c = self.interaction_coefs_orig_[k]
            if abs(c) > 1e-8:
                terms.append(f"{c:.4f}*{names[i]}*{names[j]}")
        terms.append(f"{self.intercept_orig_:.4f}")
        equation = " + ".join(terms)

        lines = [
            f"Ridge Regression with Interactions (α={self.ridge_.alpha_:.4g} chosen by CV):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]
        for j, name in enumerate(names):
            lines.append(f"  {name}: {self.coefs_orig_[j]:.4f}")
        for k, (i, j) in enumerate(self.interaction_pairs_):
            c = self.interaction_coefs_orig_[k]
            if abs(c) > 1e-8:
                lines.append(f"  {names[i]}*{names[j]}: {c:.4f}")
        lines.append(f"  intercept: {self.intercept_orig_:.4f}")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
InteractionRidgeRegressor.__module__ = "interpretable_regressor"


# ---------------------------------------------------------------------------
# Evaluation (do not edit any of the evaluation functions, only the names and model descriptions below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    model_defs = [("InteractionRidge", InteractionRidgeRegressor())]

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
    model_name = "InteractionRidge"
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
        "model_name":                         "InteractionRidge",
        "description":                        "Standardized Ridge with top-5 residual-correlated pairwise interactions",
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

