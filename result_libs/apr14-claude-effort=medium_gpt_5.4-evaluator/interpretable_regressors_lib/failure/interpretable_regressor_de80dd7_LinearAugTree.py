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
from sklearn.linear_model import ElasticNetCV, BayesianRidge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
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


class LinearAugTreeRegressor(BaseEstimator, RegressorMixin):
    """
    Linear-augmented tree: uses linear model's prediction as an extra feature.

    Genuinely novel approach:
    1. Fit RidgeCV linear model
    2. Compute linear_pred = Ridge.predict(X) as a new feature
    3. Augment: X_aug = [X, linear_pred]
    4. Fit distilled tree on X_aug with teacher smoothing + shrinkage
    5. The tree can split on "linear_pred" — learning WHEN the linear
       model is right and when it needs correction

    This gives the tree access to the linear model's "opinion", enabling:
    - If the tree splits on linear_pred, it's saying "trust the linear model
      when it predicts high, but correct when it predicts low"
    - If the tree ignores linear_pred, the data is better handled by the tree alone

    Display: single tree with features x0, x1, ..., xN, linear_pred
    """

    def __init__(self, max_leaf_nodes=25, distill_alpha=0.5):
        self.max_leaf_nodes = max_leaf_nodes
        self.distill_alpha = distill_alpha

    def fit(self, X, y):
        n_samples, n_feat = X.shape
        self.n_features_ = n_feat

        # Step 1: Fit linear model
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.ridge_ = RidgeCV(cv=3)
        self.ridge_.fit(X_scaled, y)

        # Step 2: Create augmented features
        linear_pred = self.ridge_.predict(X_scaled).reshape(-1, 1)
        X_aug = np.hstack([X, linear_pred])

        # Step 3: Teacher distillation on augmented features
        rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        gbm = GradientBoostingRegressor(n_estimators=100, max_depth=3,
            learning_rate=0.1, random_state=42, subsample=0.8)
        rf.fit(X_aug, y)
        gbm.fit(X_aug, y)
        teacher = 0.5 * rf.predict(X_aug) + 0.5 * gbm.predict(X_aug)
        y_smooth = self.distill_alpha * teacher + (1 - self.distill_alpha) * y

        # Step 4: CV for tree params
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        tree_configs = [(l, r) for l in [8, 12, 16, 20, 25] for r in [0.5, 2.0, 5.0, 10.0, 25.0]]
        best_mse = float('inf')
        best_l, best_r = 16, 5.0

        for ml, rg in tree_configs:
            fm = []
            for tri, vai in kf.split(X_aug):
                t = DecisionTreeRegressor(max_leaf_nodes=ml, random_state=42)
                t.fit(X_aug[tri], y_smooth[tri])
                _hierarchical_shrinkage(t, reg=rg)
                fm.append(np.mean((t.predict(X_aug[vai]) - y[vai]) ** 2))
            if np.mean(fm) < best_mse:
                best_mse = np.mean(fm)
                best_l, best_r = ml, rg

        # Step 5: Fit final tree
        self.tree_ = DecisionTreeRegressor(max_leaf_nodes=best_l, random_state=42)
        self.tree_.fit(X_aug, y_smooth)
        _hierarchical_shrinkage(self.tree_, reg=best_r)

        return self

    def predict(self, X):
        check_is_fitted(self, "tree_")
        X_scaled = self.scaler_.transform(X)
        linear_pred = self.ridge_.predict(X_scaled).reshape(-1, 1)
        X_aug = np.hstack([X, linear_pred])
        return self.tree_.predict(X_aug)

    def __str__(self):
        check_is_fitted(self, "tree_")
        # Feature names: original + "linear_pred"
        names = [f"x{i}" for i in range(self.n_features_)] + ["linear_pred"]

        # Compute linear equation for display
        coefs = self.ridge_.coef_ / self.scaler_.scale_
        intercept = self.ridge_.intercept_ - np.sum(self.ridge_.coef_ * self.scaler_.mean_ / self.scaler_.scale_)
        active = [(f"x{i}", coefs[i]) for i in range(len(coefs)) if abs(coefs[i]) > 1e-6]
        eq_parts = " + ".join(f"{c:.3f}*{n}" for n, c in sorted(active, key=lambda x: -abs(x[1]))[:5])
        if len(active) > 5: eq_parts += " + ..."
        eq_parts += f" + {intercept:.3f}"

        tree_text = export_text(self.tree_, feature_names=names, max_depth=6)
        header = (
            "> ------------------------------\n"
            "> Decision Tree with Linear Augmentation\n"
            f"> \tlinear_pred = {eq_parts}\n"
            "> \tTo predict: first compute linear_pred from the equation above.\n"
            "> \tThen check conditions top-to-bottom to find leaf value.\n"
            "> ------------------------------\n"
        )
        return header + tree_text


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
LinearAugTreeRegressor.__module__ = "interpretable_regressor"
model_shorthand_name = "LinearAugTree"
model_description = "Tree with linear prediction as extra feature — learns when to trust/correct the linear model"
model_defs = [(model_shorthand_name, LinearAugTreeRegressor())]


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
    try: git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception: git_hash = ""
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
        writer.writeheader(); writer.writerows(existing_interp + new_interp)
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
        for ri, (r, _) in enumerate(valid, 1): r["rank"] = ri
        for r in rows:
            if r["rmse"] in ("", None): r["rank"] = ""
    with open(perf_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=perf_fields); writer.writeheader()
        for ds_name in by_dataset:
            for row in by_dataset[ds_name]: writer.writerow(row)
    all_dataset_rmses = defaultdict(dict)
    for row in existing_perf:
        s = row.get("rmse", "")
        if s not in ("", None): all_dataset_rmses[row["dataset"]][row["model"]] = float(s)
        else: all_dataset_rmses[row["dataset"]][row["model"]] = float("nan")
    avg_rank, _ = compute_rank_scores(dict(all_dataset_rmses))
    mean_rank = avg_rank.get(model_shorthand_name, float("nan"))
    upsert_overall_results([{"commit": git_hash,
        "mean_rank": f"{mean_rank:.2f}" if not np.isnan(mean_rank) else "nan",
        "frac_interpretability_tests_passed": f"{n_passed / total:.4f}" if total > 0 else "nan",
        "status": "", "model_name": model_shorthand_name, "description": model_description}], RESULTS_DIR)
    plot_interp_vs_performance(os.path.join(RESULTS_DIR, "overall_results.csv"),
        os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"))
    print(); print("---")
    print(f"tests_passed:  {n_passed}/{total}" + (f" ({n_passed/total:.2%})" if total > 0 else ""))
    print(f"mean_rank:     {mean_rank:.2f}" if not np.isnan(mean_rank) else "mean_rank:     nan")
    print(f"total_seconds: {time.time() - t0:.1f}s")
