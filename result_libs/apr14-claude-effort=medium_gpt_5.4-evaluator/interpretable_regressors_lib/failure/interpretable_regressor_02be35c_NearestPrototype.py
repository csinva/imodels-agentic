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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class NearestPrototypeRegressor(BaseEstimator, RegressorMixin):
    """
    Nearest-prototype regressor: K-means + nearest centroid prediction.

    Genuinely novel approach:
    1. Standardize features
    2. K-means cluster the training data into k prototypes
    3. For each prototype, compute the average target value
    4. To predict: find the nearest prototype, return its value

    This is like a Voronoi-based piecewise-constant model.
    The LLM can see each prototype's coordinates and prediction,
    then determine which is closest to a given input.

    Display format:
      Prototype 1 (x0=1.2, x1=-0.5, ...): predict 5.4
      Prototype 2 (x0=-0.8, x1=2.1, ...): predict -2.1
      ...
      Predict the value of the nearest prototype.
    """

    def __init__(self, n_prototypes=6):
        self.n_prototypes = n_prototypes

    def fit(self, X, y):
        n_samples, n_feat = X.shape
        self.n_features_ = n_feat

        # Standardize for distance computation
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # K-means clustering
        k = min(self.n_prototypes, n_samples // 5)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)

        # Compute prototype values (mean y per cluster)
        self.centroids_scaled_ = km.cluster_centers_
        # Convert centroids back to original scale
        self.centroids_ = self.scaler_.inverse_transform(self.centroids_scaled_)
        self.prototype_values_ = np.array([
            np.mean(y[labels == i]) for i in range(k)
        ])
        self.n_prototypes_actual_ = k

        return self

    def predict(self, X):
        check_is_fitted(self, "centroids_scaled_")
        X_scaled = self.scaler_.transform(X)
        # Find nearest centroid for each sample
        dists = np.linalg.norm(X_scaled[:, None, :] - self.centroids_scaled_[None, :, :], axis=2)
        nearest = np.argmin(dists, axis=1)
        return self.prototype_values_[nearest]

    def __str__(self):
        check_is_fitted(self, "centroids_")
        names = [f"x{i}" for i in range(self.n_features_)]

        lines = [
            f"Nearest Prototype Model ({self.n_prototypes_actual_} prototypes):",
            "  To predict: find the prototype closest to the input, use its prediction value.",
            "  Distance is measured after standardization (each feature scaled to mean=0, std=1).",
            "",
        ]

        for i in range(self.n_prototypes_actual_):
            coords = ", ".join(f"{names[j]}={self.centroids_[i, j]:.2f}" for j in range(self.n_features_))
            lines.append(f"  Prototype {i+1} ({coords})")
            lines.append(f"    → prediction: {self.prototype_values_[i]:.4f}")
            lines.append("")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
NearestPrototypeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "NearestPrototype"
model_description = "K-means nearest-prototype regressor: 6 prototypes with coordinates and prediction values"
model_defs = [(model_shorthand_name, NearestPrototypeRegressor())]


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
