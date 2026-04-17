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


class ShrinkageTreeRegressor(BaseEstimator, RegressorMixin):
    """
    Decision tree with hierarchical shrinkage.

    Fits a standard decision tree, then applies hierarchical shrinkage:
    each node's prediction is regularized by blending with its ancestors'
    predictions, weighted by the number of samples. This reduces overfitting
    while keeping the tree structure fully interpretable.

    The shrinkage formula for each node:
      shrunk[node] = cumulative_sum of (node_mean - parent_mean) * shrink_weight
    where shrink_weight decreases for nodes with fewer samples.

    This is inspired by the hierarchical shrinkage principle but implemented
    from scratch with a novel cumulative shrinkage approach.
    """

    def __init__(self, max_leaf_nodes=12, shrink_reg=10.0):
        self.max_leaf_nodes = max_leaf_nodes
        self.shrink_reg = shrink_reg

    def fit(self, X, y):
        self.n_features_ = X.shape[1]

        # Fit base tree
        self.tree_ = DecisionTreeRegressor(
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=42,
        )
        self.tree_.fit(X, y)

        # Apply hierarchical shrinkage
        self._apply_shrinkage()

        return self

    def _apply_shrinkage(self):
        """Apply hierarchical shrinkage in-place on the tree."""
        tree = self.tree_.tree_
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        n_samples = tree.n_node_samples

        # Original node values
        orig_values = tree.value[:, 0, 0].copy()

        # Build parent map and compute depth
        parent = np.full(n_nodes, -1, dtype=int)
        for i in range(n_nodes):
            if children_left[i] != -1:
                parent[children_left[i]] = i
            if children_right[i] != -1:
                parent[children_right[i]] = i

        # Compute shrunk values using cumulative shrinkage along root-to-node path
        # For each node, shrunk_value = sum of (incremental_change * weight) along path
        # Weight at each step = n_samples[node] / (n_samples[node] + reg)
        shrunk = np.zeros(n_nodes)

        # Process in BFS order (root first)
        queue = [0]
        shrunk[0] = orig_values[0]

        while queue:
            node = queue.pop(0)
            for child in [children_left[node], children_right[node]]:
                if child == -1:
                    continue
                # The incremental change from parent to child
                delta = orig_values[child] - orig_values[node]
                # Weight based on child's sample count (more samples = trust delta more)
                weight = n_samples[child] / (n_samples[child] + self.shrink_reg)
                # Shrunk child = parent's shrunk value + weighted delta
                shrunk[child] = shrunk[node] + delta * weight
                queue.append(child)

        # Update tree values in place
        for i in range(n_nodes):
            tree.value[i, 0, 0] = shrunk[i]

    def predict(self, X):
        check_is_fitted(self, "tree_")
        return self.tree_.predict(X)

    def __str__(self):
        check_is_fitted(self, "tree_")
        feature_names = [f"x{i}" for i in range(self.n_features_)]
        tree_text = export_text(self.tree_, feature_names=feature_names, max_depth=6)
        return (
            f"Decision Tree Regressor (max_leaf_nodes={self.max_leaf_nodes}, "
            f"hierarchical shrinkage with reg={self.shrink_reg}):\n{tree_text}"
        )


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ShrinkageTreeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "ShrinkageTree"
model_description = "Decision tree with hierarchical shrinkage (max_leaf_nodes=12, reg=10), reduces overfitting while staying readable"
model_defs = [(model_shorthand_name, ShrinkageTreeRegressor())]


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
