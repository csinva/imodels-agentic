"""SparseAdditiveHinges: per-feature linear + hinge basis (quantile knots),
fit by LassoCV. Highly interpretable additive model.

Usage: uv run interpretable_regressor.py
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
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores, recompute_all_mean_ranks
from visualize import plot_interp_vs_performance


class SparseAdditiveHingesRegressor(BaseEstimator, RegressorMixin):
    """Additive model: f(x) = b + sum_j [ a_j * x_j + sum_k c_{j,k} * relu(x_j - knot_{j,k}) ].
    Knots placed at within-feature quantiles. LassoCV chooses sparsity.
    """

    def __init__(self, n_knots=3, alphas=20, cv=3, max_keep=8):
        self.n_knots = n_knots
        self.alphas = alphas
        self.cv = cv
        self.max_keep = max_keep

    def _expand(self, X):
        n, d = X.shape
        cols = [X]  # linear part
        for j in range(d):
            knots = self.knots_[j]
            if len(knots) == 0:
                continue
            xj = X[:, j:j+1]
            cols.append(np.maximum(xj - knots[None, :], 0.0))
        return np.hstack(cols)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        # quantile knots per feature
        qs = np.linspace(0.1, 0.9, self.n_knots)
        self.knots_ = []
        for j in range(d):
            vals = np.unique(np.quantile(Xs[:, j], qs))
            self.knots_.append(vals)
        Z = self._expand(Xs)
        self.lasso_ = LassoCV(n_alphas=self.alphas, cv=self.cv, max_iter=5000).fit(Z, y)
        coef = self.lasso_.coef_.copy()
        # Hard-cap nonzeros to top-|max_keep| for compactness
        if self.max_keep is not None and np.sum(coef != 0) > self.max_keep:
            order = np.argsort(-np.abs(coef))
            mask = np.zeros_like(coef, dtype=bool)
            mask[order[: self.max_keep]] = True
            coef[~mask] = 0.0
        self.coef_ = coef
        self.intercept_ = float(self.lasso_.intercept_)
        self.d_ = d
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        Z = self._expand(Xs)
        return Z @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_")
        d = self.d_
        mu = self.scaler_.mean_
        sd = self.scaler_.scale_
        coef = self.coef_
        # Convert standardized-space coefficients to original feature space:
        #   a*z_j = (a/sd)*x_j - a*mu/sd
        #   c*relu(z_j - k) = (c/sd)*relu(x_j - (mu + k*sd))
        intercept_orig = self.intercept_
        per_feature = []  # list of (j, lin_coef, [(orig_knot, orig_hinge_coef), ...])
        idx = 0
        for j in range(d):
            a = coef[idx]; idx += 1
            knots = self.knots_[j]
            hinge_coefs = coef[idx: idx + len(knots)]
            idx += len(knots)
            lin_orig = a / sd[j] if sd[j] > 0 else 0.0
            intercept_orig -= a * mu[j] / sd[j] if sd[j] > 0 else 0.0
            hinges = []
            for k_val, c in zip(knots, hinge_coefs):
                if abs(c) < 1e-8:
                    continue
                orig_k = mu[j] + k_val * sd[j] if sd[j] > 0 else mu[j]
                orig_c = c / sd[j] if sd[j] > 0 else 0.0
                hinges.append((orig_k, orig_c))
            per_feature.append((j, lin_orig if abs(a) > 1e-8 else 0.0, hinges))
        lines = ["SparseAdditiveHinges: y = intercept + sum_j f_j(x_j)",
                 f"intercept = {intercept_orig:+.4f}"]
        n_active = 0
        for j, lin, hinges in per_feature:
            terms = []
            if abs(lin) > 1e-8:
                terms.append(f"{lin:+.4g}*x{j}")
                n_active += 1
            for k, c in hinges:
                terms.append(f"{c:+.4g}*max(0, x{j} - {k:.4g})")
                n_active += 1
            if terms:
                lines.append(f"  f_{j}(x{j}) = " + " ".join(terms))
        lines.append(f"# active terms: {n_active}")
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseAdditiveHingesRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "SparseAdditiveHinges_v2"
model_description = "Additive model: per-feature linear + 3 hinges, LassoCV, hard-cap 8 terms (more compact for interp)"
model_defs = [(model_shorthand_name, SparseAdditiveHingesRegressor())]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default='gpt-4o')
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
        writer.writeheader(); writer.writerows(existing_interp + new_interp)

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
    recompute_all_mean_ranks(RESULTS_DIR)

    overall_csv = os.path.join(RESULTS_DIR, "overall_results.csv")
    plot_interp_vs_performance(overall_csv, os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"))

    print()
    print("---")
    print(f"tests_passed:  {n_passed}/{total}" + (f" ({n_passed/total:.2%})" if total > 0 else ""))
    print(f"mean_rank:     {mean_rank:.2f}" if not np.isnan(mean_rank) else "mean_rank:     nan")
    print(f"total_seconds: {time.time() - t0:.1f}s")
