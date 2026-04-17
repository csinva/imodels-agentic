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
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class AdditiveTreeRegressor(BaseEstimator, RegressorMixin):
    """
    Additive model using one small decision tree per feature.

    y = bias + tree_0(x0) + tree_1(x1) + ...

    Each tree uses only ONE feature and has limited depth, making each
    component independently interpretable. The additive structure means
    features don't interact — the total prediction is just the sum.

    Uses cyclic boosting: iterates over features multiple times, fitting
    each tree to the current residuals for that feature.
    """

    def __init__(self, max_depth=3, n_cycles=3, learning_rate=0.5):
        self.max_depth = max_depth
        self.n_cycles = n_cycles
        self.learning_rate = learning_rate

    def fit(self, X, y):
        n_samples, n_feat = X.shape
        self.n_features_ = n_feat
        self.bias_ = float(np.mean(y))

        # Initialize per-feature trees
        self.trees_ = [None] * n_feat
        self.contributions_ = np.zeros((n_samples, n_feat))

        residual = y - self.bias_

        # Cyclic boosting: iterate over features multiple times
        for cycle in range(self.n_cycles):
            for j in range(n_feat):
                # Remove this feature's current contribution from residual
                target = residual + self.contributions_[:, j]

                # Fit a small tree on this single feature
                X_single = X[:, j:j+1]
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_leaf=max(5, n_samples // 20),
                    random_state=42,
                )
                tree.fit(X_single, target)

                # Update contribution with learning rate
                new_contrib = tree.predict(X_single) * self.learning_rate
                if cycle > 0:
                    # Blend with previous contribution
                    new_contrib = new_contrib + self.contributions_[:, j] * (1 - self.learning_rate)

                self.contributions_[:, j] = new_contrib
                self.trees_[j] = tree

                # Update residual
                residual = y - self.bias_ - np.sum(self.contributions_, axis=1)

        return self

    def predict(self, X):
        check_is_fitted(self, "trees_")
        result = np.full(X.shape[0], self.bias_)
        for j in range(self.n_features_):
            if self.trees_[j] is not None:
                result += self.trees_[j].predict(X[:, j:j+1]) * self.learning_rate
        return result

    def __str__(self):
        check_is_fitted(self, "trees_")
        names = [f"x{i}" for i in range(self.n_features_)]

        lines = [
            "Additive Tree Model (one tree per feature, effects are INDEPENDENT):",
            "  y = bias + f(x0) + f(x1) + ...  (sum each feature's tree output)",
            f"  bias (intercept): {self.bias_:.4f}",
            "",
            "Per-feature trees (each uses ONE feature only):",
        ]

        for j in range(self.n_features_):
            tree = self.trees_[j]
            if tree is None:
                continue

            # Check if tree is trivial (single leaf)
            if tree.tree_.node_count <= 1:
                val = float(tree.tree_.value[0, 0, 0]) * self.learning_rate
                if abs(val) < 0.01:
                    lines.append(f"\n  {names[j]}: negligible effect")
                else:
                    lines.append(f"\n  {names[j]}: constant effect = {val:+.4f}")
                continue

            lines.append(f"\n  Tree for {names[j]} (scaled by {self.learning_rate}):")
            tree_text = export_text(tree, feature_names=[names[j]], max_depth=4)
            for line in tree_text.strip().split("\n"):
                lines.append("    " + line)

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
AdditiveTreeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "AdditiveTree"
model_description = "Additive model with one depth-3 tree per feature, cyclic boosting, clear per-feature tree output"
model_defs = [(model_shorthand_name, AdditiveTreeRegressor())]


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
