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
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class SmartKnotLinear(BaseEstimator, RegressorMixin):
    """
    Linear regressor with data-driven piecewise-linear extensions.

    For each feature we:
      1. Fit linear OLS (single feature) against y
      2. Find the ONE knot t* that, when added as max(0, x - t*), maximizes the
         improvement in fit over linear alone
      3. Only add that hinge if it reduces residual variance by >= `min_improvement`
    This keeps the equation lean and the hinges interpretable.

    After basis selection, we fit LassoCV + OLS debias over the selected basis.
    """

    def __init__(self, cv=3, max_iter=5000, min_effect_frac=0.03,
                 min_improvement=0.01, n_candidate_knots=19):
        self.cv = cv
        self.max_iter = max_iter
        self.min_effect_frac = min_effect_frac
        self.min_improvement = min_improvement
        self.n_candidate_knots = n_candidate_knots

    def _pick_knot(self, x, r):
        """Find best single knot t minimizing MSE of fit with [x, max(0,x-t)] vs r."""
        n = len(x)
        # Candidate knots: interior quantiles
        qs = np.linspace(0.1, 0.9, self.n_candidate_knots)
        candidates = np.quantile(x, qs)
        # Unique (after rounding)
        candidates = np.unique(np.round(candidates, 6))

        # Baseline: linear-only fit
        X_lin = np.column_stack([x, np.ones_like(x)])
        beta0, *_ = np.linalg.lstsq(X_lin, r, rcond=None)
        pred0 = X_lin @ beta0
        sse0 = np.sum((r - pred0) ** 2)

        best_t, best_sse = None, sse0
        for t in candidates:
            h = np.maximum(0.0, x - t)
            # Require at least 20 samples on each side for stability
            n_above = int((x > t).sum())
            if n_above < 20 or (n - n_above) < 20:
                continue
            X_pw = np.column_stack([x, h, np.ones_like(x)])
            beta, *_ = np.linalg.lstsq(X_pw, r, rcond=None)
            pred = X_pw @ beta
            sse = np.sum((r - pred) ** 2)
            if sse < best_sse:
                best_sse, best_t = sse, t
        # improvement fraction
        improvement = (sse0 - best_sse) / max(sse0, 1e-9)
        if best_t is None or improvement < self.min_improvement:
            return None
        return float(best_t)

    def _build_basis(self, X):
        n = self.n_features_in_
        parts = [X]
        for f in range(n):
            if self.knots_[f] is not None:
                parts.append(np.maximum(0.0, X[:, f] - self.knots_[f]).reshape(-1, 1))
        return np.hstack(parts)

    def _term_descriptions(self):
        out = []
        n = self.n_features_in_
        for i in range(n):
            out.append(("linear", i, None))
        for i in range(n):
            if self.knots_[i] is not None:
                out.append(("pos_hinge", i, float(self.knots_[i])))
        return out

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.n_features_in_ = X.shape[1]
        self.feature_names_ = [f"x{i}" for i in range(self.n_features_in_)]

        # Pick best knot per feature from residual after fitting all-linear model first.
        # Initial: one pass — compute residual using OLS on linear-only.
        X_with_ones = np.column_stack([X, np.ones(len(X))])
        beta_lin, *_ = np.linalg.lstsq(X_with_ones, y, rcond=None)
        residual = y - X_with_ones @ beta_lin

        self.knots_ = []
        for f in range(self.n_features_in_):
            t = self._pick_knot(X[:, f], residual)
            self.knots_.append(t)

        basis = self._build_basis(X)

        # Stage 1: LassoCV on standardized basis
        self.scaler_ = StandardScaler(with_mean=True, with_std=True)
        Bs = self.scaler_.fit_transform(basis)
        lasso = LassoCV(cv=self.cv, max_iter=self.max_iter, random_state=42)
        lasso.fit(Bs, y)

        # Stage 2: rank by effective contribution = |coef_std| * std(col)
        # In standardized space, |coef_std| IS the effect magnitude (std=1).
        effects = np.abs(lasso.coef_)
        if effects.max() > 0:
            keep_mask = effects >= self.min_effect_frac * effects.max()
        else:
            keep_mask = np.zeros_like(effects, dtype=bool)

        # Refit OLS on the kept columns (in ORIGINAL unscaled basis to keep
        # knot thresholds interpretable).
        if keep_mask.any():
            B_kept = basis[:, keep_mask]
            # Closed-form OLS with intercept
            B_aug = np.hstack([B_kept, np.ones((B_kept.shape[0], 1))])
            beta, *_ = np.linalg.lstsq(B_aug, y, rcond=None)
            kept_coefs = beta[:-1]
            intercept = float(beta[-1])
        else:
            kept_coefs = np.array([])
            intercept = float(y.mean())

        # Expand back to full coefficient vector so predict is simple
        full_coefs = np.zeros_like(effects)
        idx = 0
        for k, keep in enumerate(keep_mask):
            if keep:
                full_coefs[k] = kept_coefs[idx]
                idx += 1

        self.coef_ = full_coefs
        self.intercept_ = intercept
        self.keep_mask_ = keep_mask
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        X = np.asarray(X, dtype=np.float64)
        basis = self._build_basis(X)
        return basis @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_")
        descs = self._term_descriptions()
        terms = []
        active_feats = set()
        for k, (kind, feat_i, thresh) in enumerate(descs):
            c = self.coef_[k]
            if abs(c) < 1e-10:
                continue
            name = self.feature_names_[feat_i]
            if kind == "linear":
                s = f"{c:+.4f} * {name}"
            elif kind == "pos_hinge":
                s = f"{c:+.4f} * max(0, {name} - {thresh:.3f})"
            else:
                s = f"{c:+.4f} * max(0, {thresh:.3f} - {name})"
            terms.append((abs(c), s))
            active_feats.add(name)

        terms.sort(key=lambda x: -x[0])

        header = (
            "Sparse hinge-linear regressor. y is a sum of the terms below. "
            "Each line contributes its coefficient times its (possibly hinge-activated) feature. "
            "Terms are sorted by |coefficient|; features not listed contribute 0."
        )
        lines = [header, "", f"y = {self.intercept_:+.4f}"]
        for _, s in terms:
            lines.append(f"    {s}")

        inactive = [nm for nm in self.feature_names_ if nm not in active_feats]
        if inactive:
            lines.append("")
            lines.append(f"Features with zero contribution: {', '.join(inactive)}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SmartKnotLinear.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SmartKnotLinear"
model_description = "Linear base + at most one data-driven hinge per feature (added only if it cuts residual SSE >=1%), LassoCV select, OLS debias."
model_defs = [(model_shorthand_name, SmartKnotLinear())]


# ---------------------------------------------------------------------------
# Evaluation (do not edit anything below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default='gpt-4o',
                        help="LLM checkpoint for interpretability tests (default: gpt-4o)")
    args = parser.parse_args()

    t0 = time.time()

    # Interpretability tests
    interp_results = run_all_interp_tests(model_defs, checkpoint=args.checkpoint)
    n_passed = sum(r["passed"] for r in interp_results)
    total = len(interp_results)

    # prediction performance (RMSE)
    dataset_rmses = evaluate_all_regressors(model_defs)

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_hash = ""

    # --- Upsert interpretability_results.csv ---
    model_name = model_defs[0][0]
    interp_csv = os.path.join(RESULTS_DIR, "interpretability_results.csv")
    interp_fields = ["model", "test", "suite", "passed", "ground_truth", "response"]

    def _suite(test_name):
        if test_name.startswith("insight_"): return "insight"
        if test_name.startswith("hard_"):    return "hard"
        return "standard"

    # Load existing rows, dropping old rows for this model
    existing_interp = []
    if os.path.exists(interp_csv):
        with open(interp_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_interp.append(row)

    new_interp = [{
        "model": r["model"],
        "test": r["test"],
        "suite": _suite(r["test"]),
        "passed": r["passed"],
        "ground_truth": r.get("ground_truth", ""),
        "response": r.get("response", ""),
    } for r in interp_results]

    with open(interp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=interp_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_interp + new_interp)
    print(f"Interpretability results saved → {interp_csv}")

    # --- Upsert performance_results.csv and recompute ranks ---
    perf_csv = os.path.join(RESULTS_DIR, "performance_results.csv")
    perf_fields = ["dataset", "model", "rmse", "rank"]

    # Load existing rows, dropping old rows for this model
    existing_perf = []
    if os.path.exists(perf_csv):
        with open(perf_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_perf.append(row)

    # Add new rows (without rank for now)
    for ds_name, model_rmses in dataset_rmses.items():
        rmse_val = model_rmses.get(model_name, float("nan"))
        existing_perf.append({
            "dataset": ds_name,
            "model": model_name,
            "rmse": "" if np.isnan(rmse_val) else f"{rmse_val:.6f}",
            "rank": "",
        })

    # Recompute ranks per dataset
    by_dataset = defaultdict(list)
    for row in existing_perf:
        by_dataset[row["dataset"]].append(row)

    for ds_name, rows in by_dataset.items():
        valid = [(r, float(r["rmse"])) for r in rows if r["rmse"] not in ("", None)]
        valid.sort(key=lambda x: x[1])
        for rank_idx, (r, _) in enumerate(valid, 1):
            r["rank"] = rank_idx
        # Leave rank empty for rows with no RMSE
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

    # --- Compute mean_rank from the updated performance_results.csv ---
    # Build dataset_rmses dict with all models from the CSV for ranking
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
        "commit":                             git_hash,
        "mean_rank":                          f"{mean_rank:.2f}" if not np.isnan(mean_rank) else "nan",
        "frac_interpretability_tests_passed": f"{n_passed / total:.4f}" if total > 0 else "nan",
        "status":                             "",
        "model_name":                         model_shorthand_name,
        "description":                        model_description,
    }], RESULTS_DIR)

    # --- Plot ---
    overall_csv = os.path.join(RESULTS_DIR, "overall_results.csv")
    plot_interp_vs_performance(
        overall_csv,
        os.path.join(RESULTS_DIR, "interpretability_vs_performance.png"),
    )

    print()
    print("---")
    print(f"tests_passed:  {n_passed}/{total}" + (f" ({n_passed/total:.2%})" if total > 0 else ""))
    print(f"mean_rank:     {mean_rank:.2f}" if not np.isnan(mean_rank) else "mean_rank:     nan")
    print(f"total_seconds: {time.time() - t0:.1f}s")
