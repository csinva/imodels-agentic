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
from sklearn.ensemble import GradientBoostingRegressor
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
    """Apply hierarchical shrinkage in-place on a fitted DecisionTreeRegressor.

    Each node's prediction is blended with its parent's prediction,
    with the blend weight depending on the node's sample count.
    Nodes with fewer samples get shrunk more toward their parent.
    """
    tree = tree_model.tree_
    n_nodes = tree.node_count
    cl = tree.children_left
    cr = tree.children_right
    n_samples = tree.n_node_samples
    orig = tree.value[:, 0, 0].copy()

    shrunk = np.zeros(n_nodes)
    shrunk[0] = orig[0]

    # BFS from root
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


class DistilledShrinkTreeRegressor(BaseEstimator, RegressorMixin):
    """
    Distilled decision tree with hierarchical shrinkage.

    Novel approach:
    1. Fit a gradient-boosted teacher model (strong but opaque)
    2. Create smoothed targets: y_smooth = α * teacher_pred + (1-α) * y
       This denoises the labels while keeping them on the original scale
    3. Fit a single compact decision tree on these smoothed targets
    4. Apply hierarchical shrinkage for further regularization
    5. Cross-validate to pick the best shrinkage regularization

    The distillation step helps the tree learn better split points and
    leaf values by training on cleaner targets. Combined with hierarchical
    shrinkage, this produces a tree that is more accurate than one trained
    directly on noisy data.

    Output is a single decision tree — same readable format that works
    well for LLM interpretability.
    """

    def __init__(self, max_leaf_nodes=10, distill_alpha=0.7,
                 teacher_n_estimators=100, teacher_max_depth=3):
        self.max_leaf_nodes = max_leaf_nodes
        self.distill_alpha = distill_alpha
        self.teacher_n_estimators = teacher_n_estimators
        self.teacher_max_depth = teacher_max_depth

    def fit(self, X, y):
        n_samples, n_feat = X.shape
        self.n_features_ = n_feat

        # Step 1: Fit teacher model (GBM)
        teacher = GradientBoostingRegressor(
            n_estimators=self.teacher_n_estimators,
            max_depth=self.teacher_max_depth,
            learning_rate=0.1,
            random_state=42,
            subsample=0.8,
        )
        teacher.fit(X, y)
        teacher_preds = teacher.predict(X)

        # Step 2: Create smoothed/distilled targets
        y_smooth = self.distill_alpha * teacher_preds + (1 - self.distill_alpha) * y

        # Step 3: Fit decision tree on smoothed targets
        self.tree_ = DecisionTreeRegressor(
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=42,
        )
        self.tree_.fit(X, y_smooth)

        # Step 4: Cross-validate shrinkage regularization
        best_reg = self._cv_shrinkage(X, y)

        # Step 5: Refit final tree and apply best shrinkage
        self.tree_ = DecisionTreeRegressor(
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=42,
        )
        self.tree_.fit(X, y_smooth)
        _hierarchical_shrinkage(self.tree_, reg=best_reg)
        self.shrink_reg_ = best_reg

        return self

    def _cv_shrinkage(self, X, y):
        """Cross-validate to find the best shrinkage regularization."""
        regs = [0.1, 1.0, 5.0, 10.0, 25.0, 50.0]
        kf = KFold(n_splits=3, shuffle=True, random_state=42)

        best_score = float('inf')
        best_reg = 10.0

        # Teacher predictions for smoothed targets
        teacher = GradientBoostingRegressor(
            n_estimators=self.teacher_n_estimators,
            max_depth=self.teacher_max_depth,
            learning_rate=0.1,
            random_state=42,
            subsample=0.8,
        )

        for reg in regs:
            fold_errors = []
            for train_idx, val_idx in kf.split(X):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                # Fit teacher on fold
                teacher_fold = GradientBoostingRegressor(
                    n_estimators=self.teacher_n_estimators,
                    max_depth=self.teacher_max_depth,
                    learning_rate=0.1,
                    random_state=42,
                    subsample=0.8,
                )
                teacher_fold.fit(X_tr, y_tr)
                y_smooth_tr = self.distill_alpha * teacher_fold.predict(X_tr) + (1 - self.distill_alpha) * y_tr

                # Fit tree on smoothed labels
                tree = DecisionTreeRegressor(
                    max_leaf_nodes=self.max_leaf_nodes,
                    random_state=42,
                )
                tree.fit(X_tr, y_smooth_tr)
                _hierarchical_shrinkage(tree, reg=reg)

                # Evaluate on validation set (against true y)
                preds = tree.predict(X_val)
                mse = np.mean((preds - y_val) ** 2)
                fold_errors.append(mse)

            avg_mse = np.mean(fold_errors)
            if avg_mse < best_score:
                best_score = avg_mse
                best_reg = reg

        return best_reg

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
DistilledShrinkTreeRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "DistilledShrinkTree"
model_description = "GBM-distilled decision tree with CV-tuned hierarchical shrinkage, smoothed training targets for better splits"
model_defs = [(model_shorthand_name, DistilledShrinkTreeRegressor())]


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
