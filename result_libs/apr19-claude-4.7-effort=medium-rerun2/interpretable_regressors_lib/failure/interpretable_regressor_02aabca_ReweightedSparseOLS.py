"""ReweightedSparseOLS: SparseLinearOLS but iteratively reweight by 1/(1+|resid|/MAD) for robustness.

Usage: uv run interpretable_regressor.py
"""

import argparse, csv, os, subprocess, sys, time
from collections import defaultdict
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores, recompute_all_mean_ranks
from visualize import plot_interp_vs_performance


class ReweightedSparseOLSRegressor(BaseEstimator, RegressorMixin):
    """LassoCV select top-k, then 2 IRLS refits with weights = 1/(1+|resid|/MAD)."""
    def __init__(self, max_features=8, cv=3, n_irls=2):
        self.max_features = max_features
        self.cv = cv
        self.n_irls = n_irls

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        sc = StandardScaler().fit(X); Xs = sc.transform(X)
        lasso = LassoCV(cv=self.cv, n_alphas=20, max_iter=5000).fit(Xs, y)
        coef = lasso.coef_
        order = np.argsort(-np.abs(coef))
        kept = [int(i) for i in order if coef[i] != 0][: self.max_features]
        if not kept:
            kept = [int(np.argmax(np.abs(coef)))] if np.any(coef) else [0]
        self.support_ = sorted(kept)
        Xs2 = X[:, self.support_]
        ols = LinearRegression().fit(Xs2, y)
        for _ in range(self.n_irls):
            resid = y - ols.predict(Xs2)
            mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
            w = 1.0 / (1.0 + np.abs(resid) / (3*mad))
            ols = LinearRegression().fit(Xs2, y, sample_weight=w)
        self.ols_coef_ = ols.coef_
        self.intercept_ = float(ols.intercept_)
        self.d_ = d
        return self

    def predict(self, X):
        check_is_fitted(self, "ols_coef_")
        X = np.asarray(X, dtype=float)
        return X[:, self.support_] @ self.ols_coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "ols_coef_")
        names = [f"x{j}" for j in self.support_]
        equation = " + ".join(f"{c:.4f}*{n}" for c, n in zip(self.ols_coef_, names))
        equation += f" + {self.intercept_:.4f}"
        excluded = [f"x{j}" for j in range(self.d_) if j not in self.support_]
        lines = [f"OLS Linear Regression (sparse subset of {len(self.support_)}/{self.d_} features):",
                 f"  y = {equation}", "", "Coefficients:"]
        for c, n in zip(self.ols_coef_, names):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {self.intercept_:.4f}")
        if excluded:
            lines.append(f"  (Excluded features with zero effect: {', '.join(excluded)})")
        return "\n".join(lines)
    """Per top-K feature: linear term + one stump indicator (best median-area threshold). Joint OLS fit."""
    def __init__(self, max_features=8, cv=3):
        self.max_features = max_features
        self.cv = cv

    def _select(self, X, y):
        sc = StandardScaler().fit(X); Xs = sc.transform(X)
        lasso = LassoCV(cv=self.cv, n_alphas=20, max_iter=5000).fit(Xs, y)
        coef = lasso.coef_
        order = np.argsort(-np.abs(coef))
        kept = [int(i) for i in order if coef[i] != 0][: self.max_features]
        if not kept:
            kept = [int(np.argmax(np.abs(coef)))] if np.any(coef) else [0]
        return sorted(kept)

    def _best_threshold(self, x, r):
        # try 7 quantile thresholds; choose the one minimizing residual SSE w/ a step
        qs = np.quantile(x, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        best_t, best_sse = None, np.inf
        for t in np.unique(qs):
            mask = x > t
            if mask.sum() < 3 or (~mask).sum() < 3: continue
            mu_hi = r[mask].mean(); mu_lo = r[~mask].mean()
            sse = ((r[mask]-mu_hi)**2).sum() + ((r[~mask]-mu_lo)**2).sum()
            if sse < best_sse:
                best_sse = sse; best_t = float(t)
        return best_t

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.support_ = self._select(X, y)
        # build per-feature linear+stump basis
        cols = []
        thresholds = []
        for j in self.support_:
            cols.append(X[:, j])
            t = self._best_threshold(X[:, j], y - y.mean())
            thresholds.append(t)
            if t is not None:
                cols.append((X[:, j] > t).astype(float))
            else:
                cols.append(np.zeros(n))
        Xb = np.column_stack(cols) if cols else np.zeros((n, 0))
        ols = LinearRegression().fit(Xb, y)
        self.thresholds_ = thresholds
        self.ols_coef_ = ols.coef_
        self.intercept_ = float(ols.intercept_)
        self.d_ = d
        return self

    def _basis(self, X):
        cols = []
        for j, t in zip(self.support_, self.thresholds_):
            cols.append(X[:, j])
            if t is not None:
                cols.append((X[:, j] > t).astype(float))
            else:
                cols.append(np.zeros(X.shape[0]))
        return np.column_stack(cols) if cols else np.zeros((X.shape[0], 0))

    def predict(self, X):
        check_is_fitted(self, "ols_coef_")
        X = np.asarray(X, dtype=float)
        return self._basis(X) @ self.ols_coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "ols_coef_")
        lines = [f"Additive Linear+Step model on {len(self.support_)}/{self.d_} features:",
                 "  y = sum of per-feature contributions + intercept", "", "Per-feature contributions:"]
        idx = 0
        terms = []
        for j, t in zip(self.support_, self.thresholds_):
            a = self.ols_coef_[idx]; idx += 1
            b = self.ols_coef_[idx]; idx += 1
            if t is not None:
                lines.append(f"  x{j}: {a:.4f}*x{j} + {b:.4f} * (1 if x{j} > {t:.4f} else 0)")
                terms.append(f"f{j}(x{j})")
            else:
                lines.append(f"  x{j}: {a:.4f}*x{j}")
                terms.append(f"f{j}(x{j})")
        lines.append(f"  intercept: {self.intercept_:.4f}")
        excluded = [f"x{j}" for j in range(self.d_) if j not in self.support_]
        if excluded:
            lines.append(f"  (Excluded features with zero effect: {', '.join(excluded)})")
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
ReweightedSparseOLSRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "ReweightedSparseOLS"
model_description = "LassoCV top-8 + 2 IRLS refits with weights 1/(1+|r|/3MAD); clean linear formula"
model_defs = [(model_shorthand_name, ReweightedSparseOLSRegressor())]


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
