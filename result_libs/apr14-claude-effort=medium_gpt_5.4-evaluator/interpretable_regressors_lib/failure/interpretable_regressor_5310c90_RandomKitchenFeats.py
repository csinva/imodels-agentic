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
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class RandomKitchenFeaturesRegressor(BaseEstimator, RegressorMixin):
    """
    Random kitchen-sink features + ElasticNet.

    Novel idea: generate diverse nonlinear transforms of each feature,
    then let ElasticNet select which are useful. Captures nonlinearity
    within the linear equation framework.

    Transforms per feature:
    - x_i (raw)
    - |x_i| (absolute value — V-shape)
    - x_i^2 (quadratic)
    - sign(x_i) * sqrt(|x_i|) (square root — compressive)

    ElasticNet's L1 zeroes out useless transforms, keeping the equation
    sparse with only the transforms that actually help prediction.

    The display shows: y = w1*x0 + w2*|x0| + w3*x1^2 + ...
    Each term is a named transformation the LLM can evaluate.
    """

    def __init__(self, max_features=50):
        self.max_features = max_features

    def fit(self, X, y):
        n_samples, n_feat = X.shape
        self.n_features_ = n_feat

        # Generate nonlinear features
        all_features = []
        all_names = []
        for j in range(n_feat):
            col = X[:, j]
            fname = f"x{j}"
            # Raw
            all_features.append(col)
            all_names.append(fname)
            # Absolute value
            all_features.append(np.abs(col))
            all_names.append(f"|{fname}|")
            # Square
            all_features.append(col ** 2)
            all_names.append(f"{fname}^2")
            # Signed sqrt
            all_features.append(np.sign(col) * np.sqrt(np.abs(col)))
            all_names.append(f"sign({fname})*sqrt(|{fname}|)")

        X_aug = np.column_stack(all_features)

        # Cap if too many features
        if X_aug.shape[1] > self.max_features:
            # Pre-select by correlation with y
            corrs = np.array([abs(np.corrcoef(X_aug[:, k], y)[0, 1])
                              if np.std(X_aug[:, k]) > 1e-10 else 0.0
                              for k in range(X_aug.shape[1])])
            top_idx = np.argsort(corrs)[-self.max_features:]
            top_idx = np.sort(top_idx)
            X_aug = X_aug[:, top_idx]
            all_names = [all_names[i] for i in top_idx]

        self.aug_names_ = all_names

        # Standardize + ElasticNetCV
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_aug)

        self.model_ = ElasticNetCV(
            l1_ratio=[0.5, 0.7, 0.9, 0.95, 0.99],
            cv=5, max_iter=5000, random_state=42,
        )
        self.model_.fit(X_scaled, y)

        # Store for transform at predict time
        self.n_aug_features_ = X_aug.shape[1]

        return self

    def _transform(self, X):
        """Generate the same augmented features as during fit."""
        all_features = []
        for j in range(self.n_features_):
            col = X[:, j]
            all_features.append(col)
            all_features.append(np.abs(col))
            all_features.append(col ** 2)
            all_features.append(np.sign(col) * np.sqrt(np.abs(col)))

        X_aug = np.column_stack(all_features)
        if X_aug.shape[1] > self.n_aug_features_:
            # Same feature selection as fit
            X_aug = X_aug[:, :self.n_aug_features_]
        return X_aug

    def predict(self, X):
        check_is_fitted(self, "model_")
        X_aug = self._transform(X)
        # Handle feature count mismatch from pre-selection
        if hasattr(self, '_selected_idx'):
            X_aug = X_aug[:, self._selected_idx]
        X_scaled = self.scaler_.transform(X_aug)
        return self.model_.predict(X_scaled)

    def __str__(self):
        check_is_fitted(self, "model_")
        coefs_scaled = self.model_.coef_
        # Convert to original scale
        coefs = coefs_scaled / self.scaler_.scale_
        intercept = self.model_.intercept_ - np.sum(coefs_scaled * self.scaler_.mean_ / self.scaler_.scale_)

        active = [(self.aug_names_[i], coefs[i]) for i in range(len(coefs)) if abs(coefs[i]) > 1e-8]

        lines = [
            "Sparse Nonlinear Equation (ElasticNet on feature transforms):",
            "  To predict: evaluate each transform, multiply by coefficient, sum + intercept.",
        ]

        if active:
            equation = " + ".join(f"{c:.4f}*{n}" for n, c in sorted(active, key=lambda x: -abs(x[1]))[:15])
            if len(active) > 15:
                equation += " + ..."
            equation += f" + {intercept:.4f}"
            lines.append(f"  y = {equation}")
            lines.append("")
            lines.append(f"  Active terms ({len(active)} non-zero, sorted by magnitude):")
            for n, c in sorted(active, key=lambda x: -abs(x[1])):
                lines.append(f"    {c:+.4f} * {n}")
        else:
            lines.append(f"  y = {intercept:.4f}")

        lines.append(f"  intercept: {intercept:.4f}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
RandomKitchenFeaturesRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "RandomKitchenFeats"
model_description = "ElasticNet on random nonlinear transforms (|x|, x^2, sqrt) — captures nonlinearity in equation format"
model_defs = [(model_shorthand_name, RandomKitchenFeaturesRegressor())]


# ---------------------------------------------------------------------------
# Evaluation (do not edit anything below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default='gpt-5.4',
                        help="LLM checkpoint for interpretability tests (default: gpt-5.4)")
    args = parser.parse_args()

    t0 = time.time()
    interp_results = run_all_interp_tests(model_defs, checkpoint=args.checkpoint)
    n_passed = sum(r["passed"] for r in interp_results)
    total = len(interp_results)
    dataset_rmses = evaluate_all_regressors(model_defs)
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_hash = ""

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
                if row.get("model") != model_name: existing_interp.append(row)
    new_interp = [{"model": r["model"], "test": r["test"], "suite": _suite(r["test"]),
        "passed": r["passed"], "ground_truth": r.get("ground_truth", ""),
        "response": r.get("response", "")} for r in interp_results]
    with open(interp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=interp_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_interp + new_interp)

    perf_csv = os.path.join(RESULTS_DIR, "performance_results.csv")
    perf_fields = ["dataset", "model", "rmse", "rank"]
    existing_perf = []
    if os.path.exists(perf_csv):
        with open(perf_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name: existing_perf.append(row)
    for ds_name, model_rmses in dataset_rmses.items():
        rmse_val = model_rmses.get(model_name, float("nan"))
        existing_perf.append({"dataset": ds_name, "model": model_name,
            "rmse": "" if np.isnan(rmse_val) else f"{rmse_val:.6f}", "rank": ""})
    by_dataset = defaultdict(list)
    for row in existing_perf: by_dataset[row["dataset"]].append(row)
    for ds_name, rows in by_dataset.items():
        valid = [(r, float(r["rmse"])) for r in rows if r["rmse"] not in ("", None)]
        valid.sort(key=lambda x: x[1])
        for rank_idx, (r, _) in enumerate(valid, 1): r["rank"] = rank_idx
        for r in rows:
            if r["rmse"] in ("", None): r["rank"] = ""
    with open(perf_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=perf_fields)
        writer.writeheader()
        for ds_name in by_dataset:
            for row in by_dataset[ds_name]: writer.writerow(row)

    all_dataset_rmses = defaultdict(dict)
    for row in existing_perf:
        rmse_str = row.get("rmse", "")
        if rmse_str not in ("", None): all_dataset_rmses[row["dataset"]][row["model"]] = float(rmse_str)
        else: all_dataset_rmses[row["dataset"]][row["model"]] = float("nan")
    avg_rank, _ = compute_rank_scores(dict(all_dataset_rmses))
    mean_rank = avg_rank.get(model_shorthand_name, float("nan"))
    upsert_overall_results([{
        "commit": git_hash,
        "mean_rank": f"{mean_rank:.2f}" if not np.isnan(mean_rank) else "nan",
        "frac_interpretability_tests_passed": f"{n_passed / total:.4f}" if total > 0 else "nan",
        "status": "", "model_name": model_shorthand_name, "description": model_description,
    }], RESULTS_DIR)
    plot_interp_vs_performance(
        os.path.join(RESULTS_DIR, "overall_results.csv"),
        os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"))
    print()
    print("---")
    print(f"tests_passed:  {n_passed}/{total}" + (f" ({n_passed/total:.2%})" if total > 0 else ""))
    print(f"mean_rank:     {mean_rank:.2f}" if not np.isnan(mean_rank) else "mean_rank:     nan")
    print(f"total_seconds: {time.time() - t0:.1f}s")
