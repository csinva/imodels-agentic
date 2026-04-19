"""MicroEBM: per-top-feature piecewise-constant additive functions (4 bins).
Boosted residual fitting; output as per-feature step-function tables.

Usage: uv run interpretable_regressor.py
"""

import argparse, csv, os, subprocess, sys, time
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


class MicroEBMRegressor(BaseEstimator, RegressorMixin):
    """Additive model: f(x) = intercept + sum_j g_j(x_j) where g_j is a piecewise-constant
    function defined on `n_bins` quantile bins. Fit by cyclic coordinate descent on residuals.
    """
    def __init__(self, n_bins=4, n_rounds=20, max_features=8):
        self.n_bins = n_bins
        self.n_rounds = n_rounds
        self.max_features = max_features

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        # pick top features by lasso magnitude
        sc = StandardScaler().fit(X); Xs = sc.transform(X)
        lasso = LassoCV(cv=3, n_alphas=10, max_iter=3000).fit(Xs, y)
        order = np.argsort(-np.abs(lasso.coef_))
        kept = [int(i) for i in order if lasso.coef_[i] != 0][: self.max_features]
        if len(kept) == 0:
            kept = [int(np.argmax(np.abs(lasso.coef_)))] if np.any(lasso.coef_) else [0]
        self.support_ = sorted(kept)
        # bin edges per feature (quantiles)
        edges = {}; bin_idx = {}
        for j in self.support_:
            qs = np.unique(np.quantile(X[:, j], np.linspace(0, 1, self.n_bins + 1)))
            if len(qs) < 2:
                qs = np.array([X[:, j].min(), X[:, j].max() + 1e-9])
            edges[j] = qs
            # assign bin index in [0, len(qs)-2]
            idx = np.clip(np.searchsorted(qs, X[:, j], side='right') - 1, 0, len(qs) - 2)
            bin_idx[j] = idx
        # initial intercept
        self.intercept_ = float(y.mean())
        # bin values per feature
        bin_vals = {j: np.zeros(len(edges[j]) - 1) for j in self.support_}
        # boosted backfitting
        pred = np.full(n, self.intercept_)
        for _ in range(self.n_rounds):
            for j in self.support_:
                # compute residual partial: r minus current g_j
                idx = bin_idx[j]
                cur = bin_vals[j][idx]
                resid = y - pred + cur
                # new bin means
                new_vals = np.zeros_like(bin_vals[j])
                for b in range(len(new_vals)):
                    mask = idx == b
                    if mask.any():
                        new_vals[b] = float(resid[mask].mean())
                # update prediction
                pred = pred - cur + new_vals[idx]
                bin_vals[j] = new_vals
        # center bin values per feature (move mean to intercept) for cleaner attribution
        for j in self.support_:
            counts = np.array([float((bin_idx[j] == b).sum()) for b in range(len(bin_vals[j]))])
            mean_eff = float((bin_vals[j] * counts).sum() / max(counts.sum(), 1))
            bin_vals[j] -= mean_eff
            self.intercept_ += mean_eff
        self.edges_ = edges; self.bin_vals_ = bin_vals
        self.d_ = d
        return self

    def _g(self, j, x):
        ed = self.edges_[j]
        idx = np.clip(np.searchsorted(ed, x, side='right') - 1, 0, len(ed) - 2)
        return self.bin_vals_[j][idx]

    def predict(self, X):
        check_is_fitted(self, "edges_")
        X = np.asarray(X, dtype=float)
        out = np.full(X.shape[0], self.intercept_)
        for j in self.support_:
            out = out + self._g(j, X[:, j])
        return out

    def __str__(self):
        check_is_fitted(self, "edges_")
        lines = ["MicroEBM additive model: y = intercept + sum_j g_j(x_j)",
                 "  Each g_j is a piecewise-constant step function over quantile bins.",
                 f"  intercept = {self.intercept_:+.4f}", ""]
        for j in self.support_:
            lines.append(f"  g_{j}(x{j}):")
            ed = self.edges_[j]
            for b, v in enumerate(self.bin_vals_[j]):
                lo, hi = ed[b], ed[b + 1]
                lines.append(f"    if {lo:.4g} <= x{j} <= {hi:.4g}:  effect = {v:+.4f}")
        excluded = [f"x{j}" for j in range(self.d_) if j not in self.support_]
        if excluded:
            lines.append("")
            lines.append(f"(Inactive features with zero effect: {', '.join(excluded)})")
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
MicroEBMRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "MicroEBM_v1"
model_description = "Per-top-8 features, 4-bin quantile step functions, backfitting; clean per-feature step tables"
model_defs = [(model_shorthand_name, MicroEBMRegressor())]


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
