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
from sklearn.linear_model import ElasticNetCV, BayesianRidge
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


class AdaptiveElasticModel(BaseEstimator, RegressorMixin):
    """
    Adaptive model that auto-selects between linear and tree per dataset.

    Fits two models:
    1. ElasticBayesRidge (sparse linear, good for linear data)
    2. Distilled shrinkage tree (good for nonlinear data)

    Uses 3-fold CV to select which is better for each dataset.
    Displays whichever model was selected.

    This adapts to the data: uses linear when the relationship is
    linear, tree when it's nonlinear. Gets the best of both formats.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        n_samples, n_feat = X.shape
        self.n_features_ = n_feat

        # Standardize for linear model
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # --- Candidate 1: ElasticBayesRidge ---
        enet = ElasticNetCV(
            l1_ratio=[0.01, 0.1, 0.5, 0.9, 0.99],
            cv=3, max_iter=5000, random_state=42,
        )
        enet.fit(X_scaled, y)
        important = np.where(np.abs(enet.coef_) > 1e-8)[0]
        if len(important) < 1:
            important = np.arange(n_feat)
        self.selected_features_ = important

        bayes = BayesianRidge(max_iter=300)
        bayes.fit(X_scaled[:, important], y)

        # --- Candidate 2: Distilled shrinkage tree ---
        rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        gbm = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42, subsample=0.8
        )
        rf.fit(X, y)
        gbm.fit(X, y)
        teacher_preds = 0.5 * rf.predict(X) + 0.5 * gbm.predict(X)
        y_smooth = 0.5 * teacher_preds + 0.5 * y

        # CV to select best model
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        linear_mses = []
        tree_mses = []
        tree_configs = [(leaves, reg) for leaves in [8, 16, 25] for reg in [2.0, 10.0]]

        for train_idx, val_idx in kf.split(X):
            # Linear
            enet_f = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=3000, random_state=42)
            enet_f.fit(X_scaled[train_idx], y[train_idx])
            imp_f = np.where(np.abs(enet_f.coef_) > 1e-8)[0]
            if len(imp_f) < 1:
                imp_f = np.arange(n_feat)
            br_f = BayesianRidge(max_iter=200)
            br_f.fit(X_scaled[train_idx][:, imp_f], y[train_idx])
            lin_pred = br_f.predict(X_scaled[val_idx][:, imp_f])
            linear_mses.append(np.mean((lin_pred - y[val_idx]) ** 2))

            # Best tree
            best_tree_mse = float('inf')
            ys_tr = y_smooth[train_idx]
            for max_l, reg in tree_configs:
                t = DecisionTreeRegressor(max_leaf_nodes=max_l, random_state=42)
                t.fit(X[train_idx], ys_tr)
                _hierarchical_shrinkage(t, reg=reg)
                tp = t.predict(X[val_idx])
                mse = np.mean((tp - y[val_idx]) ** 2)
                if mse < best_tree_mse:
                    best_tree_mse = mse
            tree_mses.append(best_tree_mse)

        self.use_linear_ = np.mean(linear_mses) <= np.mean(tree_mses)

        if self.use_linear_:
            # Final linear fit
            self.bayes_ = bayes
            scale = self.scaler_.scale_[important]
            mean = self.scaler_.mean_[important]
            self.coef_original_ = np.zeros(n_feat)
            self.coef_original_[important] = self.bayes_.coef_ / scale
            self.intercept_original_ = self.bayes_.intercept_ - np.sum(self.bayes_.coef_ * mean / scale)
        else:
            # Final tree fit — find best config via CV
            best_mse = float('inf')
            best_l, best_r = 16, 5.0
            for max_l, reg in tree_configs:
                fold_m = []
                for train_idx, val_idx in kf.split(X):
                    t = DecisionTreeRegressor(max_leaf_nodes=max_l, random_state=42)
                    t.fit(X[train_idx], y_smooth[train_idx])
                    _hierarchical_shrinkage(t, reg=reg)
                    fold_m.append(np.mean((t.predict(X[val_idx]) - y[val_idx]) ** 2))
                if np.mean(fold_m) < best_mse:
                    best_mse = np.mean(fold_m)
                    best_l, best_r = max_l, reg

            self.tree_ = DecisionTreeRegressor(max_leaf_nodes=best_l, random_state=42)
            self.tree_.fit(X, y_smooth)
            _hierarchical_shrinkage(self.tree_, reg=best_r)

        return self

    def predict(self, X):
        check_is_fitted(self)
        if self.use_linear_:
            X_scaled = self.scaler_.transform(X)
            return self.bayes_.predict(X_scaled[:, self.selected_features_])
        else:
            return self.tree_.predict(X)

    def __str__(self):
        check_is_fitted(self)
        names = [f"x{i}" for i in range(self.n_features_)]

        if self.use_linear_:
            coefs = self.coef_original_
            intercept = self.intercept_original_
            active = [(names[i], coefs[i]) for i in range(len(coefs)) if abs(coefs[i]) > 1e-8]
            zeroed = [names[i] for i in range(len(coefs)) if abs(coefs[i]) <= 1e-8]

            lines = [
                "Sparse Linear Regression (ElasticNet-selected + Bayesian Ridge):",
                "  To predict: multiply each feature value by its coefficient, sum all products, add intercept.",
            ]
            if active:
                equation = " + ".join(f"{c:.4f}*{n}" for n, c in sorted(active, key=lambda x: -abs(x[1])))
                equation += f" + {intercept:.4f}"
                lines.append(f"  y = {equation}")
                lines.append("")
                lines.append(f"  Coefficients ({len(active)} active features, sorted by magnitude):")
                for n, c in sorted(active, key=lambda x: -abs(x[1])):
                    lines.append(f"    {n}: {c:.4f}")
                if zeroed:
                    lines.append(f"  Features with zero effect (excluded): {', '.join(zeroed)}")
            else:
                lines.append(f"  y = {intercept:.4f}")
            lines.append(f"  intercept: {intercept:.4f}")
            return "\n".join(lines)
        else:
            tree_text = export_text(self.tree_, feature_names=names, max_depth=6)
            header = (
                "> ------------------------------\n"
                "> Decision Tree with Hierarchical Shrinkage\n"
                "> \tTo predict: start at the top. At each |--- line, check the condition.\n"
                "> \tIf TRUE, follow that branch. Continue until you reach a 'value: [N]' line.\n"
                "> \tThat number N is the prediction.\n"
                "> ------------------------------\n"
            )
            return header + tree_text


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
AdaptiveElasticModel.__module__ = "interpretable_regressor"

model_shorthand_name = "AdaptiveElastic"
model_description = "Auto-selects ElasticBayesRidge (linear) or distilled tree per dataset via CV, adaptive format"
model_defs = [(model_shorthand_name, AdaptiveElasticModel())]


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
                if row.get("model") != model_name:
                    existing_interp.append(row)

    new_interp = [{
        "model": r["model"], "test": r["test"], "suite": _suite(r["test"]),
        "passed": r["passed"], "ground_truth": r.get("ground_truth", ""),
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
        existing_perf.append({"dataset": ds_name, "model": model_name,
            "rmse": "" if np.isnan(rmse_val) else f"{rmse_val:.6f}", "rank": ""})
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
        "commit": git_hash,
        "mean_rank": f"{mean_rank:.2f}" if not np.isnan(mean_rank) else "nan",
        "frac_interpretability_tests_passed": f"{n_passed / total:.4f}" if total > 0 else "nan",
        "status": "", "model_name": model_shorthand_name, "description": model_description,
    }], RESULTS_DIR)

    plot_interp_vs_performance(
        os.path.join(RESULTS_DIR, "overall_results.csv"),
        os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"),
    )

    print()
    print("---")
    print(f"tests_passed:  {n_passed}/{total}" + (f" ({n_passed/total:.2%})" if total > 0 else ""))
    print(f"mean_rank:     {mean_rank:.2f}" if not np.isnan(mean_rank) else "mean_rank:     nan")
    print(f"total_seconds: {time.time() - t0:.1f}s")
