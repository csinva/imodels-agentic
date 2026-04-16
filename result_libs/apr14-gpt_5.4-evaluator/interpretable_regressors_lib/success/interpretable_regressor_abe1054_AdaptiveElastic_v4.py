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


def _fit_elastic_bayes(X_scaled, y, n_feat):
    """Fit ElasticNet selection + BayesianRidge. Returns (model, features, coef_orig, intercept_orig)."""
    enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=3000, random_state=42)
    enet.fit(X_scaled, y)
    imp = np.where(np.abs(enet.coef_) > 1e-8)[0]
    if len(imp) < 1:
        imp = np.arange(n_feat)
    br = BayesianRidge(max_iter=200)
    br.fit(X_scaled[:, imp], y)
    return br, imp


class AdaptiveElasticV4(BaseEstimator, RegressorMixin):
    """
    Adaptive model v4: 3 candidates including piecewise-linear.

    Candidates:
    1. ElasticBayesRidge — sparse linear equation
    2. Distilled shrinkage tree — readable tree
    3. Piecewise-linear — ONE split + linear model in each half:
       "IF x_best <= threshold THEN y=eq1 ELSE y=eq2"

    The piecewise-linear candidate captures regime changes
    (one threshold + different linear relationships on each side).
    It's more expressive than pure linear but simpler than a tree.

    CV picks the best candidate per dataset.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        n_samples, n_feat = X.shape
        self.n_features_ = n_feat

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Teacher for trees
        rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        gbm = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42, subsample=0.8
        )
        rf.fit(X, y)
        gbm.fit(X, y)
        teacher_preds = 0.5 * rf.predict(X) + 0.5 * gbm.predict(X)
        y_smooth = 0.5 * teacher_preds + 0.5 * y

        # CV
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        cand_mses = {'linear': [], 'tree': [], 'piecewise': []}
        tree_configs = [(l, r) for l in [6, 10, 16, 25] for r in [1.0, 5.0, 15.0]]

        for train_idx, val_idx in kf.split(X):
            Xtr_s, Xval_s = X_scaled[train_idx], X_scaled[val_idx]
            Xtr, Xval = X[train_idx], X[val_idx]
            ytr, yval = y[train_idx], y[val_idx]

            # Linear
            br, imp = _fit_elastic_bayes(Xtr_s, ytr, n_feat)
            cand_mses['linear'].append(np.mean((br.predict(Xval_s[:, imp]) - yval) ** 2))

            # Tree
            best_t = float('inf')
            for ml, rg in tree_configs:
                t = DecisionTreeRegressor(max_leaf_nodes=ml, random_state=42)
                t.fit(Xtr, y_smooth[train_idx])
                _hierarchical_shrinkage(t, reg=rg)
                best_t = min(best_t, np.mean((t.predict(Xval) - yval) ** 2))
            cand_mses['tree'].append(best_t)

            # Piecewise-linear: find best split feature + threshold
            best_pw = float('inf')
            for j in range(min(n_feat, 20)):  # top 20 features
                thresh = np.median(Xtr[:, j])
                left = Xtr[:, j] <= thresh
                right = ~left
                if np.sum(left) < 10 or np.sum(right) < 10:
                    continue
                try:
                    r_l = RidgeCV(cv=3).fit(Xtr_s[left], ytr[left])
                    r_r = RidgeCV(cv=3).fit(Xtr_s[right], ytr[right])
                    # Predict
                    val_left = Xval[:, j] <= thresh
                    preds = np.zeros(len(Xval))
                    if np.any(val_left):
                        preds[val_left] = r_l.predict(Xval_s[val_left])
                    if np.any(~val_left):
                        preds[~val_left] = r_r.predict(Xval_s[~val_left])
                    mse = np.mean((preds - yval) ** 2)
                    best_pw = min(best_pw, mse)
                except Exception:
                    pass
            cand_mses['piecewise'].append(best_pw)

        avg = {k: np.mean(v) for k, v in cand_mses.items()}
        self.selected_ = min(avg, key=avg.get)

        # Final fit
        if self.selected_ == 'linear':
            enet = ElasticNetCV(l1_ratio=[0.01, 0.1, 0.5, 0.9, 0.99],
                                cv=3, max_iter=5000, random_state=42)
            enet.fit(X_scaled, y)
            imp = np.where(np.abs(enet.coef_) > 1e-8)[0]
            if len(imp) < 1: imp = np.arange(n_feat)
            self.selected_features_ = imp
            self.bayes_ = BayesianRidge(max_iter=300)
            self.bayes_.fit(X_scaled[:, imp], y)
            scale = self.scaler_.scale_[imp]
            mean = self.scaler_.mean_[imp]
            self.coef_orig_ = np.zeros(n_feat)
            self.coef_orig_[imp] = self.bayes_.coef_ / scale
            self.intercept_orig_ = self.bayes_.intercept_ - np.sum(self.bayes_.coef_ * mean / scale)

        elif self.selected_ == 'tree':
            best_mse = float('inf')
            best_l, best_r = 16, 5.0
            for ml, rg in tree_configs:
                fm = []
                for tri, vai in kf.split(X):
                    t = DecisionTreeRegressor(max_leaf_nodes=ml, random_state=42)
                    t.fit(X[tri], y_smooth[tri])
                    _hierarchical_shrinkage(t, reg=rg)
                    fm.append(np.mean((t.predict(X[vai]) - y[vai]) ** 2))
                if np.mean(fm) < best_mse:
                    best_mse = np.mean(fm)
                    best_l, best_r = ml, rg
            self.tree_ = DecisionTreeRegressor(max_leaf_nodes=best_l, random_state=42)
            self.tree_.fit(X, y_smooth)
            _hierarchical_shrinkage(self.tree_, reg=best_r)

        else:  # piecewise
            # Find best split
            best_mse = float('inf')
            best_j, best_t = 0, 0.0
            for j in range(min(n_feat, 20)):
                thresh = np.median(X[:, j])
                left = X[:, j] <= thresh
                if np.sum(left) < 10 or np.sum(~left) < 10:
                    continue
                fm = []
                for tri, vai in kf.split(X):
                    ltr = X[tri, j] <= thresh
                    rtr = ~ltr
                    if np.sum(ltr) < 5 or np.sum(rtr) < 5: continue
                    rl = RidgeCV(cv=3).fit(X_scaled[tri][ltr], y[tri][ltr])
                    rr = RidgeCV(cv=3).fit(X_scaled[tri][rtr], y[tri][rtr])
                    p = np.zeros(len(vai))
                    lv = X[vai, j] <= thresh
                    if np.any(lv): p[lv] = rl.predict(X_scaled[vai][lv])
                    if np.any(~lv): p[~lv] = rr.predict(X_scaled[vai][~lv])
                    fm.append(np.mean((p - y[vai]) ** 2))
                if fm and np.mean(fm) < best_mse:
                    best_mse = np.mean(fm)
                    best_j, best_t = j, thresh

            self.split_feat_ = best_j
            self.split_thresh_ = best_t
            left = X[:, best_j] <= best_t
            self.ridge_left_ = RidgeCV(cv=3).fit(X_scaled[left], y[left])
            self.ridge_right_ = RidgeCV(cv=3).fit(X_scaled[~left], y[~left])
            # Convert coeffs to original scale
            sc = self.scaler_.scale_
            mn = self.scaler_.mean_
            self.coef_left_ = self.ridge_left_.coef_ / sc
            self.int_left_ = self.ridge_left_.intercept_ - np.sum(self.ridge_left_.coef_ * mn / sc)
            self.coef_right_ = self.ridge_right_.coef_ / sc
            self.int_right_ = self.ridge_right_.intercept_ - np.sum(self.ridge_right_.coef_ * mn / sc)

        return self

    def predict(self, X):
        check_is_fitted(self)
        if self.selected_ == 'linear':
            return self.bayes_.predict(self.scaler_.transform(X)[:, self.selected_features_])
        elif self.selected_ == 'tree':
            return self.tree_.predict(X)
        else:
            Xs = self.scaler_.transform(X)
            left = X[:, self.split_feat_] <= self.split_thresh_
            preds = np.zeros(X.shape[0])
            if np.any(left): preds[left] = self.ridge_left_.predict(Xs[left])
            if np.any(~left): preds[~left] = self.ridge_right_.predict(Xs[~left])
            return preds

    def __str__(self):
        check_is_fitted(self)
        names = [f"x{i}" for i in range(self.n_features_)]

        if self.selected_ == 'linear':
            coefs = self.coef_orig_
            intercept = self.intercept_orig_
            active = sorted([(names[i], coefs[i]) for i in range(len(coefs)) if abs(coefs[i]) > 1e-8],
                           key=lambda x: -abs(x[1]))
            zeroed = [names[i] for i in range(len(coefs)) if abs(coefs[i]) <= 1e-8]
            lines = ["Sparse Linear Regression (ElasticNet-selected + Bayesian Ridge):",
                     "  To predict: multiply each feature value by its coefficient, sum all products, add intercept."]
            if active:
                eq = " + ".join(f"{c:.4f}*{n}" for n, c in active) + f" + {intercept:.4f}"
                lines.append(f"  y = {eq}")
                lines.append("")
                lines.append(f"  Coefficients ({len(active)} active features, sorted by magnitude):")
                for n, c in active: lines.append(f"    {n}: {c:.4f}")
                if zeroed: lines.append(f"  Features with zero effect: {', '.join(zeroed)}")
            lines.append(f"  intercept: {intercept:.4f}")
            return "\n".join(lines)

        elif self.selected_ == 'tree':
            tree_text = export_text(self.tree_, feature_names=names, max_depth=6)
            return (">\n> Decision Tree with Hierarchical Shrinkage\n"
                    "> \tTo predict: check conditions top-to-bottom, follow matching branch to leaf value.\n>\n" + tree_text)

        else:  # piecewise
            fname = names[self.split_feat_]
            def _eq(coefs, intercept):
                active = [(names[i], coefs[i]) for i in range(len(coefs)) if abs(coefs[i]) > 1e-6]
                if not active: return f"{intercept:.4f}"
                return " + ".join(f"{c:.4f}*{n}" for n, c in sorted(active, key=lambda x: -abs(x[1]))) + f" + {intercept:.4f}"

            lines = [
                f"Piecewise Linear Model (one split on {fname}):",
                f"  To predict: check if {fname} <= {self.split_thresh_:.2f}, then use the corresponding equation.",
                "",
                f"  IF {fname} <= {self.split_thresh_:.2f}:",
                f"    y = {_eq(self.coef_left_, self.int_left_)}",
                "",
                f"  ELSE ({fname} > {self.split_thresh_:.2f}):",
                f"    y = {_eq(self.coef_right_, self.int_right_)}",
            ]
            return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
AdaptiveElasticV4.__module__ = "interpretable_regressor"
model_shorthand_name = "AdaptiveElastic_v4"
model_description = "3-candidate adaptive: linear + tree + piecewise-linear (1 split + 2 equations)"
model_defs = [(model_shorthand_name, AdaptiveElasticV4())]


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
