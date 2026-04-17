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
from sklearn.linear_model import ElasticNetCV, BayesianRidge, LassoCV, RidgeCV
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


def _recalibrate_leaves(tree_model, X, y):
    tree = tree_model.tree_
    leaf_ids = tree_model.apply(X)
    for leaf_id in np.unique(leaf_ids):
        mask = leaf_ids == leaf_id
        if np.sum(mask) > 0:
            tree.value[leaf_id, 0, 0] = np.mean(y[mask])


class KitchenSinkAdaptive(BaseEstimator, RegressorMixin):
    """
    Kitchen-sink adaptive model: picks the best from 4 diverse candidates.

    Candidates (all with proven readable formats):
    1. ElasticBayesRidge — ElasticNet feature selection + BayesianRidge
    2. Distilled shrinkage tree — RF+GBM teacher + hierarchical shrinkage
    3. Recalibrated distilled tree — tree with leaf values from raw data
    4. Pure RidgeCV — sometimes the simplest model wins

    3-fold CV selects the best candidate per dataset. Each candidate
    uses a proven display format (equation or tree).

    The diversity of candidates means the model adapts well:
    - Linear candidates for linear data
    - Tree candidates for nonlinear data
    - Distilled vs recalibrated trees for different noise levels
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        n_samples, n_feat = X.shape
        self.n_features_ = n_feat

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # ElasticNet feature selection (shared)
        enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=5000, random_state=42)
        enet.fit(X_scaled, y)
        important = np.where(np.abs(enet.coef_) > 1e-8)[0]
        if len(important) < 1:
            important = np.arange(n_feat)
        self.selected_features_ = important

        # Teacher for distilled trees
        rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        gbm = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42, subsample=0.8
        )
        rf.fit(X, y)
        gbm.fit(X, y)
        teacher_preds = 0.5 * rf.predict(X) + 0.5 * gbm.predict(X)
        y_smooth = 0.5 * teacher_preds + 0.5 * y

        # CV to select best candidate
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        candidate_mses = {name: [] for name in ['elastic_bayes', 'distilled_tree', 'recal_tree', 'ridge']}

        tree_configs = [(l, r) for l in [8, 16, 25] for r in [2.0, 10.0]]

        for train_idx, val_idx in kf.split(X):
            Xtr_s, Xval_s = X_scaled[train_idx], X_scaled[val_idx]
            Xtr, Xval = X[train_idx], X[val_idx]
            ytr, yval = y[train_idx], y[val_idx]
            ytr_smooth = y_smooth[train_idx]

            # Candidate 1: ElasticBayesRidge
            enet_f = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=3000, random_state=42)
            enet_f.fit(Xtr_s, ytr)
            imp_f = np.where(np.abs(enet_f.coef_) > 1e-8)[0]
            if len(imp_f) < 1:
                imp_f = np.arange(n_feat)
            br = BayesianRidge(max_iter=200)
            br.fit(Xtr_s[:, imp_f], ytr)
            candidate_mses['elastic_bayes'].append(
                np.mean((br.predict(Xval_s[:, imp_f]) - yval) ** 2))

            # Candidate 2: Distilled tree (best config)
            best_dt_mse = float('inf')
            for ml, rg in tree_configs:
                t = DecisionTreeRegressor(max_leaf_nodes=ml, random_state=42)
                t.fit(Xtr, ytr_smooth)
                _hierarchical_shrinkage(t, reg=rg)
                mse = np.mean((t.predict(Xval) - yval) ** 2)
                best_dt_mse = min(best_dt_mse, mse)
            candidate_mses['distilled_tree'].append(best_dt_mse)

            # Candidate 3: Recalibrated tree
            best_rt_mse = float('inf')
            for ml, rg in tree_configs:
                t = DecisionTreeRegressor(max_leaf_nodes=ml, random_state=42)
                t.fit(Xtr, ytr_smooth)
                _recalibrate_leaves(t, Xtr, ytr)
                _hierarchical_shrinkage(t, reg=rg)
                mse = np.mean((t.predict(Xval) - yval) ** 2)
                best_rt_mse = min(best_rt_mse, mse)
            candidate_mses['recal_tree'].append(best_rt_mse)

            # Candidate 4: Pure RidgeCV
            ridge = RidgeCV(cv=3)
            ridge.fit(Xtr_s, ytr)
            candidate_mses['ridge'].append(
                np.mean((ridge.predict(Xval_s) - yval) ** 2))

        # Select best candidate
        avg_mses = {k: np.mean(v) for k, v in candidate_mses.items()}
        self.selected_model_ = min(avg_mses, key=avg_mses.get)

        # Fit final model
        if self.selected_model_ == 'elastic_bayes':
            self.bayes_ = BayesianRidge(max_iter=300)
            self.bayes_.fit(X_scaled[:, important], y)
            scale = self.scaler_.scale_[important]
            mean = self.scaler_.mean_[important]
            self.coef_orig_ = np.zeros(n_feat)
            self.coef_orig_[important] = self.bayes_.coef_ / scale
            self.intercept_orig_ = self.bayes_.intercept_ - np.sum(self.bayes_.coef_ * mean / scale)

        elif self.selected_model_ == 'ridge':
            self.ridge_ = RidgeCV(cv=5)
            self.ridge_.fit(X_scaled, y)
            scale = self.scaler_.scale_
            mean = self.scaler_.mean_
            self.coef_orig_ = self.ridge_.coef_ / scale
            self.intercept_orig_ = self.ridge_.intercept_ - np.sum(self.ridge_.coef_ * mean / scale)

        elif self.selected_model_ in ('distilled_tree', 'recal_tree'):
            # Find best tree config
            best_mse = float('inf')
            best_l, best_r = 16, 5.0
            for ml, rg in tree_configs:
                fold_m = []
                for train_idx, val_idx in kf.split(X):
                    t = DecisionTreeRegressor(max_leaf_nodes=ml, random_state=42)
                    t.fit(X[train_idx], y_smooth[train_idx])
                    if self.selected_model_ == 'recal_tree':
                        _recalibrate_leaves(t, X[train_idx], y[train_idx])
                    _hierarchical_shrinkage(t, reg=rg)
                    fold_m.append(np.mean((t.predict(X[val_idx]) - y[val_idx]) ** 2))
                if np.mean(fold_m) < best_mse:
                    best_mse = np.mean(fold_m)
                    best_l, best_r = ml, rg

            self.tree_ = DecisionTreeRegressor(max_leaf_nodes=best_l, random_state=42)
            self.tree_.fit(X, y_smooth)
            if self.selected_model_ == 'recal_tree':
                _recalibrate_leaves(self.tree_, X, y)
            _hierarchical_shrinkage(self.tree_, reg=best_r)

        return self

    def predict(self, X):
        check_is_fitted(self)
        if self.selected_model_ == 'elastic_bayes':
            Xs = self.scaler_.transform(X)
            return self.bayes_.predict(Xs[:, self.selected_features_])
        elif self.selected_model_ == 'ridge':
            return self.ridge_.predict(self.scaler_.transform(X))
        else:
            return self.tree_.predict(X)

    def __str__(self):
        check_is_fitted(self)
        names = [f"x{i}" for i in range(self.n_features_)]

        if self.selected_model_ in ('elastic_bayes', 'ridge'):
            coefs = self.coef_orig_
            intercept = self.intercept_orig_
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
KitchenSinkAdaptive.__module__ = "interpretable_regressor"

model_shorthand_name = "KitchenSinkAdaptive"
model_description = "4-candidate adaptive: ElasticBayesRidge, distilled tree, recal tree, RidgeCV — CV picks best per dataset"
model_defs = [(model_shorthand_name, KitchenSinkAdaptive())]


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
    new_interp = [{"model": r["model"], "test": r["test"], "suite": _suite(r["test"]),
        "passed": r["passed"], "ground_truth": r.get("ground_truth", ""),
        "response": r.get("response", "")} for r in interp_results]
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
            if r["rmse"] in ("", None): r["rank"] = ""
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
        os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"))
    print()
    print("---")
    print(f"tests_passed:  {n_passed}/{total}" + (f" ({n_passed/total:.2%})" if total > 0 else ""))
    print(f"mean_rank:     {mean_rank:.2f}" if not np.isnan(mean_rank) else "mean_rank:     nan")
    print(f"total_seconds: {time.time() - t0:.1f}s")
