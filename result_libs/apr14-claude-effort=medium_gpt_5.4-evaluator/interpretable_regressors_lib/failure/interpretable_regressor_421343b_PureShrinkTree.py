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
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


def _hierarchical_shrinkage_inplace(tree_model, reg=1.0):
    """Apply hierarchical shrinkage in-place on a fitted DecisionTreeRegressor.

    For each node, shrinks its prediction toward its parent.
    Weight = n_samples[child] / (n_samples[child] + reg).
    More samples = less shrinkage. Higher reg = more shrinkage overall.
    """
    tree = tree_model.tree_
    n_nodes = tree.node_count
    cl = tree.children_left
    cr = tree.children_right
    n_samples = tree.n_node_samples
    orig = tree.value[:, 0, 0].copy()

    shrunk = np.zeros(n_nodes)
    shrunk[0] = orig[0]

    queue = [0]
    while queue:
        node = queue.pop(0)
        for child in [cl[node], cr[node]]:
            if child == -1:
                continue
            delta = orig[child] - orig[node]
            w = n_samples[child] / (n_samples[child] + reg)
            shrunk[child] = shrunk[node] + delta * w
            queue.append(child)

    for i in range(n_nodes):
        tree.value[i, 0, 0] = shrunk[i]


class PureShrinkTreeRegressor(BaseEstimator, RegressorMixin):
    """
    Decision tree with cross-validated hierarchical shrinkage.

    1. Try multiple max_leaf_nodes and shrinkage reg values
    2. Cross-validate to find the best combination
    3. Fit final tree with best parameters and apply shrinkage

    Uses the same readable tree output format proven to achieve
    high LLM interpretability scores (~0.79-0.84).
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        n_samples, n_feat = X.shape
        self.n_features_ = n_feat

        # Grid of hyperparameters to cross-validate
        leaf_options = [8, 12, 16, 20]
        reg_options = [0.1, 1.0, 5.0, 10.0, 25.0, 50.0]

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        best_mse = float('inf')
        best_leaves = 12
        best_reg = 10.0

        for max_leaves in leaf_options:
            for reg in reg_options:
                fold_mses = []
                for train_idx, val_idx in kf.split(X):
                    X_tr, X_val = X[train_idx], X[val_idx]
                    y_tr, y_val = y[train_idx], y[val_idx]

                    tree = DecisionTreeRegressor(
                        max_leaf_nodes=max_leaves,
                        random_state=42,
                    )
                    tree.fit(X_tr, y_tr)
                    _hierarchical_shrinkage_inplace(tree, reg=reg)

                    preds = tree.predict(X_val)
                    fold_mses.append(np.mean((preds - y_val) ** 2))

                avg_mse = np.mean(fold_mses)
                if avg_mse < best_mse:
                    best_mse = avg_mse
                    best_leaves = max_leaves
                    best_reg = reg

        # Fit final tree with best parameters
        self.tree_ = DecisionTreeRegressor(
            max_leaf_nodes=best_leaves,
            random_state=42,
        )
        self.tree_.fit(X, y)
        _hierarchical_shrinkage_inplace(self.tree_, reg=best_reg)

        self.best_leaves_ = best_leaves
        self.best_reg_ = best_reg

        return self

    def predict(self, X):
        check_is_fitted(self, "tree_")
        return self.tree_.predict(X)

    def __str__(self):
        check_is_fitted(self, "tree_")
        feature_names = [f"x{i}" for i in range(self.n_features_)]
        tree_text = export_text(self.tree_, feature_names=feature_names, max_depth=6)
        header = (
            "> ------------------------------\n"
            "> Decision Tree with Hierarchical Shrinkage\n"
            "> \tPrediction is made by looking at the value in the appropriate leaf of the tree\n"
            "> ------------------------------\n"
        )
        return header + tree_text


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
PureShrinkTreeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "PureShrinkTree"
model_description = "Decision tree with CV-tuned max_leaf_nodes and hierarchical shrinkage regularization, clean tree output"
model_defs = [(model_shorthand_name, PureShrinkTreeRegressor())]


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
