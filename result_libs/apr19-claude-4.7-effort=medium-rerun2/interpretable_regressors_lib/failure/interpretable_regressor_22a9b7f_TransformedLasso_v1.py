"""TransformedLasso: linear model on small set of monotone transforms per feature.
Output is a clean linear formula in original feature space.

Usage: uv run interpretable_regressor.py
"""

import argparse, csv, os, subprocess, sys, time
from collections import defaultdict
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores, recompute_all_mean_ranks
from visualize import plot_interp_vs_performance


def _safe_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


class TransformedLassoRegressor(BaseEstimator, RegressorMixin):
    """Per-feature transforms: identity, square, signed-log. LassoCV selects.
    Then refit OLS on selected support for unbiased coefficients. Cap K terms.
    """

    TRANSFORMS = [
        ("x",       lambda x: x),
        ("x^2",     lambda x: x * x),
        ("slog(x)", _safe_log),
    ]

    def __init__(self, max_terms=8, cv=3):
        self.max_terms = max_terms
        self.cv = cv

    def _expand(self, X):
        cols = []
        names = []
        n, d = X.shape
        for j in range(d):
            for tname, tfn in self.TRANSFORMS:
                cols.append(tfn(X[:, j]))
                names.append((j, tname))
        Z = np.column_stack(cols)
        return Z, names

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        Z, names = self._expand(X)
        self.scaler_ = StandardScaler().fit(Z)
        Zs = self.scaler_.transform(Z)
        lasso = LassoCV(cv=self.cv, n_alphas=20, max_iter=5000).fit(Zs, y)
        coef = lasso.coef_
        # pick top-K nonzero
        order = np.argsort(-np.abs(coef))
        keep = [i for i in order if coef[i] != 0][: self.max_terms]
        if len(keep) == 0:
            keep = [int(np.argmax(np.abs(coef)))] if np.any(coef) else [0]
        self.support_ = keep
        # refit OLS in original (unscaled) space on selected columns
        Z_sel = Z[:, keep]
        ols = LinearRegression().fit(Z_sel, y)
        self.coef_orig_ = ols.coef_
        self.intercept_ = float(ols.intercept_)
        self.names_ = names
        self.d_ = d
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_orig_")
        Z, _ = self._expand(np.asarray(X, dtype=float))
        Z_sel = Z[:, self.support_]
        return Z_sel @ self.coef_orig_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_orig_")
        lines = ["TransformedLasso: y = intercept + sum of selected terms",
                 f"intercept = {self.intercept_:+.4f}"]
        for c, idx in zip(self.coef_orig_, self.support_):
            j, tname = self.names_[idx]
            expr = tname.replace("x", f"x{j}")
            lines.append(f"  {c:+.4g} * {expr}")
        lines.append(f"# active terms: {len(self.support_)}  (features used: "
                     f"{sorted(set(self.names_[i][0] for i in self.support_))})")
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
TransformedLassoRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "TransformedLasso_v1"
model_description = "LassoCV over per-feature [x, x^2, slog(x)] transforms; refit OLS on top-8 support; clean linear-in-transforms formula"
model_defs = [(model_shorthand_name, TransformedLassoRegressor())]


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

    def _suite(t):
        if t.startswith("insight_"): return "insight"
        if t.startswith("hard_"):    return "hard"
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
        w = csv.DictWriter(f, fieldnames=interp_fields, extrasaction="ignore")
        w.writeheader(); w.writerows(existing_interp + new_interp)

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
        w = csv.DictWriter(f, fieldnames=perf_fields)
        w.writeheader()
        for ds_name in by_dataset:
            for row in by_dataset[ds_name]:
                w.writerow(row)

    all_dataset_rmses = defaultdict(dict)
    for row in existing_perf:
        rmse_str = row.get("rmse", "")
        if rmse_str not in ("", None):
            all_dataset_rmses[row["dataset"]][row["model"]] = float(rmse_str)
        else:
            all_dataset_rmses[row["dataset"]][row["model"]] = float("nan")
    avg_rank, _ = compute_rank_scores(dict(all_dataset_rmses))
    mean_rank = avg_rank.get(model_shorthand_name, float("nan"))
    upsert_overall_results([{"commit": git_hash,
        "mean_rank": f"{mean_rank:.2f}" if not np.isnan(mean_rank) else "nan",
        "frac_interpretability_tests_passed": f"{n_passed/total:.4f}" if total > 0 else "nan",
        "status": "", "model_name": model_shorthand_name, "description": model_description}], RESULTS_DIR)
    recompute_all_mean_ranks(RESULTS_DIR)
    overall_csv = os.path.join(RESULTS_DIR, "overall_results.csv")
    plot_interp_vs_performance(overall_csv, os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"))

    print()
    print("---")
    print(f"tests_passed:  {n_passed}/{total}" + (f" ({n_passed/total:.2%})" if total > 0 else ""))
    print(f"mean_rank:     {mean_rank:.2f}" if not np.isnan(mean_rank) else "mean_rank:     nan")
    print(f"total_seconds: {time.time() - t0:.1f}s")
