"""LinearPlusStumps: sparse OLS trunk + greedy decision stumps fit on residuals.

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


def _best_stump(X, r, n_quantiles=20):
    """Find best (feature, threshold, left_mean, right_mean) reducing SSE of r."""
    n, d = X.shape
    best = None
    best_gain = 0.0
    base_sse = float(np.sum(r * r))
    for j in range(d):
        col = X[:, j]
        qs = np.unique(np.quantile(col, np.linspace(0.1, 0.9, n_quantiles)))
        for thr in qs:
            left = col <= thr
            nl = int(left.sum()); nr = n - nl
            if nl < 5 or nr < 5:
                continue
            ml = float(r[left].mean()); mr = float(r[~left].mean())
            sse = float(np.sum((r[left] - ml) ** 2) + np.sum((r[~left] - mr) ** 2))
            gain = base_sse - sse
            if gain > best_gain:
                best_gain = gain
                best = (int(j), float(thr), ml, mr)
    return best


class LinearPlusStumpsRegressor(BaseEstimator, RegressorMixin):
    """Sparse linear trunk (LassoCV select, OLS refit) + greedy stumps on residuals.
    Output is a clean linear formula plus a short list of if-then adjustments.
    """

    def __init__(self, max_features=8, n_stumps=4, cv=3):
        self.max_features = max_features
        self.n_stumps = n_stumps
        self.cv = cv

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        lasso = LassoCV(cv=self.cv, n_alphas=20, max_iter=5000).fit(Xs, y)
        coef = lasso.coef_
        order = np.argsort(-np.abs(coef))
        kept = [int(i) for i in order if coef[i] != 0][: self.max_features]
        if len(kept) == 0:
            kept = [int(np.argmax(np.abs(coef)))] if np.any(coef) else [0]
        self.support_ = sorted(kept)
        ols = LinearRegression().fit(X[:, self.support_], y)
        self.ols_coef_ = ols.coef_
        self.intercept_ = float(ols.intercept_)
        # residual stumps
        pred = X[:, self.support_] @ self.ols_coef_ + self.intercept_
        r = y - pred
        self.stumps_ = []
        for _ in range(self.n_stumps):
            stump = _best_stump(X, r)
            if stump is None: break
            j, thr, ml, mr = stump
            # Re-center: predict ml on left, mr on right, but to be additive add (ml on left, mr on right) - mean of stump? Keep raw — model intercept covers shift.
            self.stumps_.append(stump)
            col = X[:, j]
            r = r - np.where(col <= thr, ml, mr)
        self.d_ = d
        return self

    def predict(self, X):
        check_is_fitted(self, "ols_coef_")
        X = np.asarray(X, dtype=float)
        out = X[:, self.support_] @ self.ols_coef_ + self.intercept_
        for j, thr, ml, mr in self.stumps_:
            out = out + np.where(X[:, j] <= thr, ml, mr)
        return out

    def __str__(self):
        check_is_fitted(self, "ols_coef_")
        names = [f"x{j}" for j in self.support_]
        equation = " + ".join(f"{c:.4f}*{n}" for c, n in zip(self.ols_coef_, names))
        equation += f" + {self.intercept_:.4f}"
        excluded = [f"x{j}" for j in range(self.d_) if j not in self.support_]
        lines = [
            "LinearPlusStumps: y = linear_part + sum of stump adjustments",
            "",
            f"Linear part:  y_lin = {equation}",
            "Coefficients:",
        ]
        for c, n in zip(self.ols_coef_, names):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {self.intercept_:.4f}")
        if excluded:
            lines.append(f"  (Excluded features with zero effect: {', '.join(excluded)})")
        if self.stumps_:
            lines.append("")
            lines.append(f"Stump adjustments (added to y_lin):")
            for k, (j, thr, ml, mr) in enumerate(self.stumps_, 1):
                lines.append(f"  stump {k}: if x{j} <= {thr:.4g}: add {ml:+.4f}, else: add {mr:+.4f}")
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
LinearPlusStumpsRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "LinearPlusStumps_v1"
model_description = "Sparse OLS (k=8) + 4 greedy decision stumps on residuals; linear formula + if-then adjustments"
model_defs = [(model_shorthand_name, LinearPlusStumpsRegressor())]


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
