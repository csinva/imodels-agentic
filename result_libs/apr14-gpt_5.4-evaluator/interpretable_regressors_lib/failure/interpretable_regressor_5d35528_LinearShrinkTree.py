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
from sklearn.linear_model import RidgeCV
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


def _apply_hierarchical_shrinkage(tree, X, shrink_factor=1.0):
    """Apply hierarchical shrinkage to a fitted DecisionTreeRegressor.

    For each node, shrinks its value toward its parent's value.
    This regularizes the tree without changing its structure.

    shrink_factor controls how much regularization:
      higher = more shrinkage = smoother predictions
    """
    tree_obj = tree.tree_
    n_nodes = tree_obj.node_count
    children_left = tree_obj.children_left
    children_right = tree_obj.children_right
    values = tree_obj.value.copy()  # shape (n_nodes, 1, 1)
    n_samples = tree_obj.n_node_samples

    # Build parent map
    parent = np.full(n_nodes, -1, dtype=int)
    for i in range(n_nodes):
        if children_left[i] != -1:
            parent[children_left[i]] = i
        if children_right[i] != -1:
            parent[children_right[i]] = i

    # Compute shrunk values top-down
    shrunk = np.zeros(n_nodes)
    shrunk[0] = values[0, 0, 0]  # root keeps its value

    # BFS order
    queue = [0]
    visited = set()
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)

        if parent[node] >= 0:
            parent_val = shrunk[parent[node]]
            node_val = values[node, 0, 0]
            # Shrink toward parent based on sample count
            n = n_samples[node]
            lam = shrink_factor / (1 + n / shrink_factor) if n > 0 else 0.5
            shrunk[node] = parent_val * lam + node_val * (1 - lam)

        if children_left[node] != -1:
            queue.append(children_left[node])
        if children_right[node] != -1:
            queue.append(children_right[node])

    # Update tree values in place
    for i in range(n_nodes):
        tree_obj.value[i, 0, 0] = shrunk[i]


class LinearShrinkTreeRegressor(BaseEstimator, RegressorMixin):
    """
    Linear model + hierarchically-shrunk decision tree on residuals.

    Step 1: Fit a Ridge linear model to capture linear relationships.
    Step 2: Fit a decision tree to the residuals, then apply hierarchical
            shrinkage to regularize the tree predictions.

    The result is: y = linear_part + tree_correction

    Both parts have clear, interpretable representations:
    - Linear: coefficient equation
    - Tree: standard decision tree text (shrunk for better generalization)
    """

    def __init__(self, tree_max_depth=4, tree_min_leaf=5, shrink_factor=10.0):
        self.tree_max_depth = tree_max_depth
        self.tree_min_leaf = tree_min_leaf
        self.shrink_factor = shrink_factor

    def fit(self, X, y):
        n_samples, n_feat = X.shape
        self.n_features_ = n_feat

        # Step 1: Linear model
        self.ridge_ = RidgeCV(cv=3)
        self.ridge_.fit(X, y)
        linear_pred = self.ridge_.predict(X)

        # Step 2: Tree on residuals
        residuals = y - linear_pred
        self.tree_ = DecisionTreeRegressor(
            max_depth=self.tree_max_depth,
            min_samples_leaf=max(self.tree_min_leaf, n_samples // 20),
            random_state=42,
        )
        self.tree_.fit(X, residuals)

        # Apply hierarchical shrinkage
        _apply_hierarchical_shrinkage(self.tree_, X, self.shrink_factor)

        return self

    def predict(self, X):
        check_is_fitted(self, "ridge_")
        return self.ridge_.predict(X) + self.tree_.predict(X)

    def __str__(self):
        check_is_fitted(self, "ridge_")
        names = [f"x{i}" for i in range(self.n_features_)]

        # Linear part
        coefs = self.ridge_.coef_
        intercept = self.ridge_.intercept_

        lines = [
            "Linear + Shrinkage Tree Model:",
            "  y = linear_prediction + tree_correction",
            "",
            "Part 1 — Linear equation:",
        ]

        # Build equation
        equation = " + ".join(f"{c:.4f}*{n}" for c, n in zip(coefs, names))
        equation += f" + {intercept:.4f}"
        lines.append(f"  y_linear = {equation}")
        lines.append("")
        lines.append("  Coefficients:")
        for n, c in zip(names, coefs):
            lines.append(f"    {n}: {c:.4f}")
        lines.append(f"    intercept: {intercept:.4f}")

        # Tree part
        lines.append("")
        lines.append("Part 2 — Tree correction (applied to residual after linear):")
        tree_text = export_text(self.tree_, feature_names=names, max_depth=5)
        for line in tree_text.strip().split("\n"):
            lines.append("  " + line)

        lines.append("")
        lines.append("To predict: compute y_linear from the equation, then add the tree correction.")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
LinearShrinkTreeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "LinearShrinkTree"
model_description = "Ridge linear model + hierarchically-shrunk decision tree on residuals, compact 2-part representation"
model_defs = [(model_shorthand_name, LinearShrinkTreeRegressor())]


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
