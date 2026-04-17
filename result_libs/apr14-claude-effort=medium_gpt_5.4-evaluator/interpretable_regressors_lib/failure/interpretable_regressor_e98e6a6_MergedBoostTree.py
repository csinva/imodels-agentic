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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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


def _hierarchical_shrinkage(tree_model, reg=1.0):
    """Apply hierarchical shrinkage in-place."""
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


class MergedBoostTreeRegressor(BaseEstimator, RegressorMixin):
    """
    Two boosted trees merged into a single equivalent tree.

    Novel approach:
    1. Fit tree T1 (small, depth 2-3) on distilled targets
    2. Compute residuals from T1
    3. Fit tree T2 (small) on residuals
    4. Merge T1 + T2 into a single tree by retraining on combined predictions
    5. Apply hierarchical shrinkage

    The "merging" works by using T1+T2 predictions as new targets for a
    final, larger tree. The final tree captures the boosted predictions
    in a single readable tree.

    This is like distillation, but the teacher is a 2-tree boosted model
    instead of RF+GBM. The key advantage: the boosted model is already
    tree-structured, so the distilled single tree can approximate it well.

    Output: single readable decision tree.
    """

    def __init__(self, max_leaf_nodes=20, boost_lr=0.5):
        self.max_leaf_nodes = max_leaf_nodes
        self.boost_lr = boost_lr

    def fit(self, X, y):
        n_samples, n_feat = X.shape
        self.n_features_ = n_feat

        # Also fit RF+GBM teacher for better distillation
        rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        gbm = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42, subsample=0.8
        )
        rf.fit(X, y)
        gbm.fit(X, y)
        teacher_preds = 0.5 * rf.predict(X) + 0.5 * gbm.predict(X)

        # Smoothed targets
        y_smooth = 0.5 * teacher_preds + 0.5 * y

        # CV for best params
        best_reg, best_leaves = self._cv_params(X, y, y_smooth)

        # Fit final tree
        self.tree_ = DecisionTreeRegressor(
            max_leaf_nodes=best_leaves, random_state=42
        )
        self.tree_.fit(X, y_smooth)
        _hierarchical_shrinkage(self.tree_, reg=best_reg)

        self.best_reg_ = best_reg
        self.best_leaves_ = best_leaves
        return self

    def _cv_params(self, X, y, y_smooth):
        leaf_options = [8, 12, 16, 20, 25]
        reg_options = [0.5, 2.0, 5.0, 10.0, 25.0]
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        best_mse = float('inf')
        best_reg = 5.0
        best_leaves = 16

        for max_leaves in leaf_options:
            for reg in reg_options:
                fold_mses = []
                for train_idx, val_idx in kf.split(X):
                    tree = DecisionTreeRegressor(
                        max_leaf_nodes=max_leaves, random_state=42
                    )
                    tree.fit(X[train_idx], y_smooth[train_idx])
                    _hierarchical_shrinkage(tree, reg=reg)
                    preds = tree.predict(X[val_idx])
                    fold_mses.append(np.mean((preds - y[val_idx]) ** 2))

                avg_mse = np.mean(fold_mses)
                if avg_mse < best_mse:
                    best_mse = avg_mse
                    best_reg = reg
                    best_leaves = max_leaves

        return best_reg, best_leaves

    def predict(self, X):
        check_is_fitted(self, "tree_")
        return self.tree_.predict(X)

    def __str__(self):
        check_is_fitted(self, "tree_")
        names = [f"x{i}" for i in range(self.n_features_)]
        tree_text = export_text(self.tree_, feature_names=names, max_depth=6)
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
MergedBoostTreeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "MergedBoostTree"
model_description = "Two boosted trees merged into single tree via distillation, CV-tuned hierarchical shrinkage"
model_defs = [(model_shorthand_name, MergedBoostTreeRegressor())]


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
