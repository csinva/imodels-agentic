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
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class BinnedRidgeGAMRegressor(BaseEstimator, RegressorMixin):
    """
    Binned Ridge GAM: quantile-bin each feature, one-hot encode the bins,
    fit RidgeCV on the resulting indicator matrix. This produces a
    step-function GAM where each feature has a piecewise-constant effect.

    Display format mimics PyGAM: per-feature tables showing
    (bin_center → effect), making it highly readable by LLMs.
    """

    def __init__(self, n_bins=6, alphas=(0.01, 0.1, 1.0, 10.0, 100.0)):
        self.n_bins = n_bins
        self.alphas = alphas

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        # Compute quantile bin edges for each feature
        self.bin_edges_ = {}
        for j in range(n_features):
            quantiles = np.linspace(0, 100, self.n_bins + 1)
            edges = np.percentile(X[:, j], quantiles)
            # Deduplicate edges
            edges = np.unique(edges)
            self.bin_edges_[j] = edges

        # Build one-hot encoded bin matrix
        X_binned = self._build_bins(X)

        # Fit Ridge on binned features
        self.ridge_ = RidgeCV(alphas=list(self.alphas), cv=5)
        self.ridge_.fit(X_binned, y)

        # Extract per-feature effects for display
        self._extract_effects(X)
        return self

    def _build_bins(self, X):
        """One-hot encode each feature into its quantile bins."""
        cols = []
        self.bin_info_ = {}  # (feature, bin_idx) -> column index in X_binned
        col_idx = 0
        for j in range(self.n_features_):
            edges = self.bin_edges_[j]
            n_bins_j = len(edges) - 1
            if n_bins_j < 1:
                n_bins_j = 1
                edges = np.array([X[:, j].min() - 1, X[:, j].max() + 1])
                self.bin_edges_[j] = edges

            bin_indices = np.digitize(X[:, j], edges[1:-1])  # 0..n_bins_j-1
            one_hot = np.zeros((X.shape[0], n_bins_j))
            for b in range(n_bins_j):
                one_hot[:, b] = (bin_indices == b).astype(float)
            cols.append(one_hot)
            for b in range(n_bins_j):
                self.bin_info_[(j, b)] = col_idx
                col_idx += 1
        return np.hstack(cols)

    def _extract_effects(self, X):
        """Compute per-feature effect tables for display."""
        self.effects_ = {}
        coef = self.ridge_.coef_

        for j in range(self.n_features_):
            edges = self.bin_edges_[j]
            n_bins_j = len(edges) - 1
            if n_bins_j < 1:
                continue

            bin_centers = []
            bin_effects = []
            for b in range(n_bins_j):
                col = self.bin_info_[(j, b)]
                center = (edges[b] + edges[min(b + 1, len(edges) - 1)]) / 2.0
                bin_centers.append(center)
                bin_effects.append(float(coef[col]))

            # Center effects (subtract mean) for cleaner display
            mean_effect = np.mean(bin_effects)
            bin_effects = [e - mean_effect for e in bin_effects]
            self.effects_[j] = (bin_centers, bin_effects, mean_effect)

    def predict(self, X):
        check_is_fitted(self, "ridge_")
        X_binned = self._build_bins(np.asarray(X, dtype=np.float64))
        return self.ridge_.predict(X_binned)

    def __str__(self):
        check_is_fitted(self, "ridge_")
        lines = [
            "Generalized Additive Model (Binned Ridge):",
            "  y = intercept + f0(x0) + f1(x1) + ...  (each feature's effect is INDEPENDENT)",
            f"  intercept: {self.ridge_.intercept_:.4f}",
            "",
            "Feature partial effects (each feature's independent contribution):",
        ]

        # Sort features by importance (range of effect)
        feat_importance = {}
        for j in range(self.n_features_):
            if j in self.effects_:
                _, effects, _ = self.effects_[j]
                feat_importance[j] = max(effects) - min(effects)
            else:
                feat_importance[j] = 0.0

        sorted_feats = sorted(feat_importance.keys(), key=lambda j: feat_importance[j], reverse=True)

        for j in sorted_feats:
            if j not in self.effects_:
                continue
            centers, effects, _ = self.effects_[j]
            effect_range = max(effects) - min(effects)
            if effect_range < 1e-4:
                continue

            # Determine shape
            if effects[-1] > effects[0] + 0.3:
                shape = "increasing"
            elif effects[-1] < effects[0] - 0.3:
                shape = "decreasing"
            elif effect_range < 0.3:
                shape = "flat/negligible"
            else:
                shape = "non-monotone"

            lines.append(f"\n  x{j}:")
            for xv, ev in zip(centers, effects):
                lines.append(f"    x{j}={xv:+.2f}  →  effect={ev:+.4f}")
            lines.append(f"    (shape: {shape})")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
BinnedRidgeGAMRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "BinnedRidgeGAM_v1"
model_description = "Quantile-binned Ridge GAM: step-function GAM via one-hot bin encoding + Ridge"
model_defs = [(model_shorthand_name, BinnedRidgeGAMRegressor())]


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

