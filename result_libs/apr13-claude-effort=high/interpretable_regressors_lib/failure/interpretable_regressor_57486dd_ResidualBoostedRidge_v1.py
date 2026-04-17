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
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class ResidualBoostedRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Residual-Boosted Ridge: fits a Ridge linear model first, then adds
    a small set of threshold correction rules on the residuals.

    Stage 1: Fit RidgeCV linear model on all features.
    Stage 2: Greedily add correction rules by fitting depth-1 stumps on
    residuals. Each rule: "IF x_j > threshold THEN add correction".
    Only add rules that meaningfully reduce residual variance.

    Display: linear equation + correction rules (clean, interpretable).
    """

    def __init__(self, n_rules=5, min_improvement=0.02):
        self.n_rules = n_rules
        self.min_improvement = min_improvement

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        # Stage 1: Fit Ridge linear model
        self.ridge_ = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
        self.ridge_.fit(X, y)
        residuals = y - self.ridge_.predict(X)
        base_var = float(np.var(residuals))

        # Stage 2: Greedily add correction rules
        self.rules_ = []
        for rule_idx in range(self.n_rules):
            best_feat = None
            best_thresh = None
            best_left = None
            best_right = None
            best_var = base_var

            for j in range(n_features):
                stump = DecisionTreeRegressor(
                    max_depth=1,
                    min_samples_leaf=max(10, n_samples // 20),
                    random_state=42
                )
                stump.fit(X[:, j:j+1], residuals)
                preds = stump.predict(X[:, j:j+1])
                new_residuals = residuals - preds
                new_var = float(np.var(new_residuals))

                if new_var < best_var:
                    best_var = new_var
                    best_feat = j
                    tree = stump.tree_
                    if tree.children_left[0] != -1:
                        best_thresh = float(tree.threshold[0])
                        best_left = float(tree.value[tree.children_left[0], 0, 0])
                        best_right = float(tree.value[tree.children_right[0], 0, 0])

            # Check if improvement is significant
            if best_feat is not None and (base_var - best_var) / max(base_var, 1e-10) > self.min_improvement:
                self.rules_.append((best_feat, best_thresh, best_left, best_right))
                mask = X[:, best_feat] <= best_thresh
                residuals[mask] -= best_left
                residuals[~mask] -= best_right
                base_var = best_var
            else:
                break

        return self

    def predict(self, X):
        check_is_fitted(self, "ridge_")
        X = np.asarray(X, dtype=np.float64)
        preds = self.ridge_.predict(X)
        for feat, thresh, left_val, right_val in self.rules_:
            mask = X[:, feat] <= thresh
            preds[mask] += left_val
            preds[~mask] += right_val
        return preds

    def __str__(self):
        check_is_fitted(self, "ridge_")
        coef = self.ridge_.coef_
        intercept = self.ridge_.intercept_

        # Build equation like OLS format
        equation = " + ".join(f"{c:.4f}*x{i}" for i, c in enumerate(coef))
        equation += f" + {intercept:.4f}"

        lines = [
            f"Linear Regression with Correction Rules (Ridge α={self.ridge_.alpha_:.4g}):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]
        for i, c in enumerate(coef):
            lines.append(f"  x{i}: {c:.4f}")
        lines.append(f"  intercept: {intercept:.4f}")

        if self.rules_:
            lines.append("")
            lines.append("Correction rules (applied to residuals after linear prediction):")
            for feat, thresh, left_val, right_val in self.rules_:
                lines.append(f"  IF x{feat} <= {thresh:.4f}: add {left_val:+.4f}")
                lines.append(f"  IF x{feat} >  {thresh:.4f}: add {right_val:+.4f}")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ResidualBoostedRidgeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "ResidualBoostedRidge_v1"
model_description = "Ridge linear model + greedy threshold correction rules on residuals"
model_defs = [(model_shorthand_name, ResidualBoostedRidgeRegressor())]


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

