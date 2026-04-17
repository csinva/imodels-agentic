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
from sklearn.linear_model import ElasticNetCV, HuberRegressor
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


class EnsembleLinearRegressor(BaseEstimator, RegressorMixin):
    """
    Ensemble of two linear models: ElasticNet + Huber regression.

    Averages predictions from:
    1. ElasticNetCV (L1+L2, good for feature selection)
    2. HuberRegressor (robust to outliers, epsilon-insensitive loss)

    The average is more robust than either alone because:
    - ElasticNet handles correlated features and high dimensions well
    - Huber handles outlier contamination well
    - Averaging reduces variance

    The displayed equation uses the ElasticNet's coefficients (since
    they're sparse and more interpretable), but prediction uses the
    average of both models.

    Actually, to ensure display/predict consistency, we fit the final
    display coefficients by averaging the two models' coefficients.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        n_samples, n_feat = X.shape
        self.n_features_ = n_feat

        # Standardize
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Model 1: ElasticNetCV
        self.enet_ = ElasticNetCV(
            l1_ratio=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
            cv=5, max_iter=5000, random_state=42,
        )
        self.enet_.fit(X_scaled, y)

        # Model 2: HuberRegressor with CV for alpha
        best_huber = None
        best_huber_mse = float('inf')
        kf = KFold(n_splits=3, shuffle=True, random_state=42)

        for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]:
            fold_mses = []
            for train_idx, val_idx in kf.split(X_scaled):
                huber = HuberRegressor(alpha=alpha, max_iter=200)
                try:
                    huber.fit(X_scaled[train_idx], y[train_idx])
                    preds = huber.predict(X_scaled[val_idx])
                    fold_mses.append(np.mean((preds - y[val_idx]) ** 2))
                except Exception:
                    fold_mses.append(float('inf'))
            avg_mse = np.mean(fold_mses)
            if avg_mse < best_huber_mse:
                best_huber_mse = avg_mse
                best_huber = alpha

        self.huber_ = HuberRegressor(alpha=best_huber or 1.0, max_iter=200)
        try:
            self.huber_.fit(X_scaled, y)
            self.has_huber_ = True
        except Exception:
            self.has_huber_ = False

        # Average coefficients for display (on scaled features)
        if self.has_huber_:
            avg_coef_scaled = 0.5 * self.enet_.coef_ + 0.5 * self.huber_.coef_
            avg_intercept = 0.5 * self.enet_.intercept_ + 0.5 * self.huber_.intercept_
        else:
            avg_coef_scaled = self.enet_.coef_
            avg_intercept = self.enet_.intercept_

        # Convert to original scale
        scale = self.scaler_.scale_
        mean = self.scaler_.mean_
        self.coef_original_ = avg_coef_scaled / scale
        self.intercept_original_ = avg_intercept - np.sum(avg_coef_scaled * mean / scale)

        # Store averaged scaled coefficients for consistent prediction
        self.avg_coef_ = avg_coef_scaled
        self.avg_intercept_ = avg_intercept

        return self

    def predict(self, X):
        check_is_fitted(self, "avg_coef_")
        X_scaled = self.scaler_.transform(X)
        return X_scaled @ self.avg_coef_ + self.avg_intercept_

    def __str__(self):
        check_is_fitted(self, "coef_original_")
        coefs = self.coef_original_
        intercept = self.intercept_original_
        names = [f"x{i}" for i in range(self.n_features_)]

        active = [(names[i], coefs[i]) for i in range(len(coefs)) if abs(coefs[i]) > 1e-8]
        zeroed = [names[i] for i in range(len(coefs)) if abs(coefs[i]) <= 1e-8]

        lines = [
            "Sparse Linear Regression (ElasticNet + Huber ensemble):",
            "  To predict: multiply each feature value by its coefficient, sum all products, add intercept.",
        ]

        if active:
            equation = " + ".join(f"{c:.4f}*{n}" for n, c in sorted(active, key=lambda x: -abs(x[1])))
            equation += f" + {intercept:.4f}"
            lines.append(f"  y = {equation}")
            lines.append("")
            lines.append(f"  Coefficients ({len(active)} active features, sorted by magnitude):")
            for n, c in sorted(active, key=lambda x: -abs(x[1])):
                lines.append(f"    {n}: {c:.4f}")
            if zeroed:
                lines.append(f"  Features with zero effect (excluded): {', '.join(zeroed)}")
        else:
            lines.append(f"  y = {intercept:.4f} (all features zeroed)")

        lines.append(f"  intercept: {intercept:.4f}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
EnsembleLinearRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "EnsembleLinear"
model_description = "Ensemble of ElasticNetCV + HuberRegressor (outlier-robust), averaged coefficients, clean equation"
model_defs = [(model_shorthand_name, EnsembleLinearRegressor())]


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

    perf_csv = os.path.join(RESULTS_DIR, "performance_results.csv")
    perf_fields = ["dataset", "model", "rmse", "rank"]

    existing_perf = []
    if os.path.exists(perf_csv):
        with open(perf_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_perf.append(row)

    for ds_name, model_rmses in dataset_rmses.items():
        rmse_val = model_rmses.get(model_name, float("nan"))
        existing_perf.append({
            "dataset": ds_name,
            "model": model_name,
            "rmse": "" if np.isnan(rmse_val) else f"{rmse_val:.6f}",
            "rank": "",
        })

    by_dataset = defaultdict(list)
    for row in existing_perf:
        by_dataset[row["dataset"]].append(row)

    for ds_name, rows in by_dataset.items():
        valid = [(r, float(r["rmse"])) for r in rows if r["rmse"] not in ("", None)]
        valid.sort(key=lambda x: x[1])
        for rank_idx, (r, _) in enumerate(valid, 1):
            r["rank"] = rank_idx
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
