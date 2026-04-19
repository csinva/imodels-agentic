"""BoostedHinges: gradient-boosted hinge basis additive model. Each round picks the
best (feature, knot) hinge that reduces the residual.

Usage: uv run interpretable_regressor.py
"""

import argparse, csv, os, subprocess, sys, time
from collections import defaultdict
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores, recompute_all_mean_ranks
from visualize import plot_interp_vs_performance


def _greedy_forward(X, y, max_features=8):
    n, d = X.shape
    selected = []
    remaining = list(range(d))
    best_resid = y - y.mean()
    base_sse = float(np.sum(best_resid ** 2)) + 1e-12
    for _ in range(min(max_features, d)):
        best_j, best_sse = None, np.inf
        for j in remaining:
            cols = selected + [j]
            Xs = X[:, cols]
            try:
                ols = LinearRegression().fit(Xs, y)
                sse = float(np.sum((y - ols.predict(Xs)) ** 2))
            except Exception:
                continue
            if sse < best_sse:
                best_sse = sse; best_j = j
        if best_j is None: break
        selected.append(best_j); remaining.remove(best_j)
        if best_sse / base_sse < 1e-6: break
    return sorted(selected)


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


def _best_hinge(x, r, n_quantiles=15):
    """Find best (knot, post_slope) such that fitting r ~ a*x + c*relu(x-knot) reduces SSE.
    Returns (knot, c) where c is the additional slope after the knot.
    If no improvement, returns None.
    """
    qs = np.unique(np.quantile(x, np.linspace(0.15, 0.85, n_quantiles)))
    base_sse = float(((r - r.mean()) ** 2).sum())
    best = None; best_sse = base_sse
    for thr in qs:
        h = np.maximum(x - thr, 0.0)
        Z = np.column_stack([x, h, np.ones_like(x)])
        try:
            coef, *_ = np.linalg.lstsq(Z, r, rcond=None)
        except Exception:
            continue
        resid = r - Z @ coef
        sse = float((resid ** 2).sum())
        if sse < best_sse:
            best_sse = sse; best = (float(thr), float(coef[0]), float(coef[1]), float(coef[2]))
    return best  # (knot, lin, hinge, intercept)


def _best_hinge_pair(X, r, n_quantiles=12):
    """Find best (j, knot, slope) where adding c*max(0, x_j - knot) most reduces SSE of r.
    Uses simple linear regression of r on a single hinge feature.
    """
    n, d = X.shape
    base_sse = float((r ** 2).sum())
    best = None; best_sse = base_sse
    for j in range(d):
        col = X[:, j]
        qs = np.unique(np.quantile(col, np.linspace(0.1, 0.9, n_quantiles)))
        for thr in qs:
            h = np.maximum(col - thr, 0.0)
            denom = float((h * h).sum())
            if denom < 1e-12:
                continue
            c = float((h * r).sum() / denom)
            new_r = r - c * h
            sse = float((new_r ** 2).sum())
            if sse < best_sse:
                best_sse = sse
                best = (int(j), float(thr), c)
    return best


class BoostedHingesRegressor(BaseEstimator, RegressorMixin):
    """Forward greedy boosted hinges: repeat T times, find best hinge feature and add it.
    Then jointly refit all hinge coefficients by OLS for unbiasedness.
    """

    def __init__(self, n_rounds=10, n_quantiles=12):
        self.n_rounds = n_rounds
        self.n_quantiles = n_quantiles

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        # start with intercept
        intercept = float(y.mean())
        r = y - intercept
        hinges = []  # list of (j, knot)
        for _ in range(self.n_rounds):
            best = _best_hinge_pair(X, r, n_quantiles=self.n_quantiles)
            if best is None: break
            j, thr, c = best
            hinges.append((j, thr))
            r = r - c * np.maximum(X[:, j] - thr, 0.0)
        # also fit a linear trunk per feature jointly
        # build basis: linear features + hinges
        cols = [X]
        meta = [("lin", j, None) for j in range(d)]
        for (j, thr) in hinges:
            cols.append(np.maximum(X[:, j] - thr, 0.0).reshape(-1, 1))
            meta.append(("hinge", j, thr))
        Z = np.hstack(cols)
        # ridge-style: use OLS with small reg via lstsq
        ols = LinearRegression().fit(Z, y)
        self.coef_ = ols.coef_
        self.intercept_ = float(ols.intercept_)
        self.meta_ = meta
        self.d_ = d
        return self

    def _expand(self, X):
        cols = [X]
        for (kind, j, thr) in self.meta_:
            if kind == "hinge":
                cols.append(np.maximum(X[:, j] - thr, 0.0).reshape(-1, 1))
        return np.hstack(cols)

    def predict(self, X):
        check_is_fitted(self, "coef_")
        X = np.asarray(X, dtype=float)
        return self._expand(X) @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_")
        per_j = defaultdict(list)
        for c, (kind, j, thr) in zip(self.coef_, self.meta_):
            if abs(c) < 1e-8:
                continue
            per_j[j].append((kind, thr, c))
        lines = [
            "BoostedHinges: additive piecewise-linear model.",
            "  y = intercept + sum_j f_j(x_j),  each f_j is linear with optional extra slopes.",
            f"  intercept = {self.intercept_:+.4f}",
            "",
            "Per-feature contributions:",
        ]
        for j in sorted(per_j):
            terms = []
            for kind, thr, c in per_j[j]:
                if kind == "lin":
                    terms.append(f"{c:+.4g}*x{j}")
                else:
                    terms.append(f"{c:+.4g}*max(0, x{j} - {thr:.4g})")
            lines.append(f"  f_{j}(x{j}) = " + " ".join(terms))
        excluded = [f"x{j}" for j in range(self.d_) if j not in per_j]
        if excluded:
            lines.append(f"(Inactive features: {', '.join(excluded)})")
        return "\n".join(lines)


import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
BoostedHingesRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "BoostedHinges_v1"
model_description = "Greedy boosted hinges (10 rounds) + linear trunk, jointly refit OLS; per-feature piecewise-linear formula"
model_defs = [(model_shorthand_name, BoostedHingesRegressor())]


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
