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
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class ModelTreeRegressor(BaseEstimator, RegressorMixin):
    """
    Model Tree: decision tree with linear equations in each leaf.

    Instead of predicting a constant at each leaf, fits a sparse linear
    model within each leaf region. This captures both:
    - Non-linear splits (from tree structure)
    - Linear trends within regions (from leaf models)

    More expressive than constant-leaf trees while still presentable
    as a readable tree with equations at leaves.

    The LLM can:
    1. Trace through the tree to find the right leaf
    2. Evaluate the linear equation at that leaf

    Uses a shallow tree (max 8 leaves) with Ridge regression at each leaf.
    Only features used in the tree splits are included in the leaf models
    to keep the equations sparse.
    """

    def __init__(self, max_leaf_nodes=15, min_samples_leaf=10):
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        n_samples, n_feat = X.shape
        self.n_features_ = n_feat

        # Step 1: Fit the tree structure
        self.tree_ = DecisionTreeRegressor(
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=max(self.min_samples_leaf, n_samples // 20),
            random_state=42,
        )
        self.tree_.fit(X, y)

        # Step 2: Identify features used in tree splits
        tree_obj = self.tree_.tree_
        used_features = set()
        for node_id in range(tree_obj.node_count):
            if tree_obj.children_left[node_id] != -1:
                used_features.add(int(tree_obj.feature[node_id]))
        self.used_features_ = sorted(used_features)

        # Step 3: Fit linear model in each leaf
        leaf_ids = self.tree_.apply(X)
        unique_leaves = np.unique(leaf_ids)
        self.leaf_models_ = {}

        for leaf_id in unique_leaves:
            mask = leaf_ids == leaf_id
            n_leaf = np.sum(mask)

            if n_leaf < 5 or len(self.used_features_) == 0:
                # Too few samples or no features — use constant
                self.leaf_models_[leaf_id] = {
                    'type': 'constant',
                    'value': float(np.mean(y[mask]))
                }
            else:
                # Fit Ridge on the features used by the tree
                X_leaf = X[mask][:, self.used_features_]
                y_leaf = y[mask]
                ridge = Ridge(alpha=1.0)
                ridge.fit(X_leaf, y_leaf)

                # Check if linear model is actually better than constant
                const_mse = np.mean((y_leaf - np.mean(y_leaf)) ** 2)
                ridge_mse = np.mean((y_leaf - ridge.predict(X_leaf)) ** 2)

                if ridge_mse < const_mse * 0.9:  # at least 10% improvement
                    self.leaf_models_[leaf_id] = {
                        'type': 'linear',
                        'coefs': ridge.coef_,
                        'intercept': float(ridge.intercept_),
                        'features': self.used_features_,
                    }
                else:
                    self.leaf_models_[leaf_id] = {
                        'type': 'constant',
                        'value': float(np.mean(y_leaf))
                    }

        return self

    def predict(self, X):
        check_is_fitted(self, "tree_")
        leaf_ids = self.tree_.apply(X)
        preds = np.zeros(X.shape[0])

        for i, leaf_id in enumerate(leaf_ids):
            model = self.leaf_models_[leaf_id]
            if model['type'] == 'constant':
                preds[i] = model['value']
            else:
                x_feat = X[i, model['features']]
                preds[i] = np.dot(model['coefs'], x_feat) + model['intercept']

        return preds

    def __str__(self):
        check_is_fitted(self, "tree_")
        names = [f"x{i}" for i in range(self.n_features_)]

        # Build custom tree text with linear equations at leaves
        tree_obj = self.tree_.tree_
        lines = [
            "> ------------------------------",
            "> Model Tree (tree with linear equations at leaves)",
            "> \tTrace the tree to find your leaf, then evaluate the equation",
            "> ------------------------------",
        ]

        def _format_node(node_id, prefix="", is_left=True):
            """Recursively format the tree with linear leaf values."""
            if tree_obj.children_left[node_id] == -1:
                # Leaf node
                model = self.leaf_models_.get(node_id)
                if model is None or model['type'] == 'constant':
                    val = tree_obj.value[node_id, 0, 0] if model is None else model['value']
                    lines.append(f"{prefix}|--- value: [{val:.4f}]")
                else:
                    # Linear equation
                    coefs = model['coefs']
                    intercept = model['intercept']
                    feat_names = [names[f] for f in model['features']]
                    eq_parts = [f"{c:.3f}*{n}" for c, n in zip(coefs, feat_names) if abs(c) > 0.01]
                    if eq_parts:
                        eq = f"{intercept:.3f} + " + " + ".join(eq_parts)
                    else:
                        eq = f"{intercept:.4f}"
                    lines.append(f"{prefix}|--- value: [{eq}]")
                return

            feat = tree_obj.feature[node_id]
            thresh = tree_obj.threshold[node_id]
            fname = names[feat]

            # Left child
            lines.append(f"{prefix}|--- {fname} <= {thresh:.2f}")
            _format_node(tree_obj.children_left[node_id], prefix + "|   ")

            # Right child
            lines.append(f"{prefix}|--- {fname} >  {thresh:.2f}")
            _format_node(tree_obj.children_right[node_id], prefix + "|   ")

        _format_node(0)
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ModelTreeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "ModelTree_v2"
model_description = "Model tree v2: 15-leaf tree with Ridge linear equations at each leaf, captures local trends"
model_defs = [(model_shorthand_name, ModelTreeRegressor())]


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
