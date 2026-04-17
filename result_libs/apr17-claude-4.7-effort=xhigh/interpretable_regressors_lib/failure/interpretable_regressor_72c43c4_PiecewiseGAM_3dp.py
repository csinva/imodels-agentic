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


class PiecewiseGAM_Plus(BaseEstimator, RegressorMixin):
    """
    PiecewiseLinearGAM (3 knots, 3% prune, per-feature piecewise display).

    Legacy container class — compatibility for iteration history; the current
    best model is defined below.
    """

    def __init__(self, n_knots=3, cv=3, max_iter=5000,
                 min_effect_frac=0.03, seg_slope_tol=0.0,
                 max_interactions=0, ols_debias=False):
        self.n_knots = n_knots
        self.cv = cv
        self.max_iter = max_iter
        self.min_effect_frac = min_effect_frac
        self.seg_slope_tol = seg_slope_tol
        self.max_interactions = max_interactions
        self.ols_debias = ols_debias

    def _build_basis(self, X):
        n = self.n_features_in_
        parts = [X]
        for f in range(n):
            for t in self.knots_[f]:
                parts.append(np.maximum(0.0, X[:, f] - t).reshape(-1, 1))
                parts.append(np.maximum(0.0, t - X[:, f]).reshape(-1, 1))
        # Add selected interactions at the end
        for (i, j) in self.interactions_:
            parts.append((X[:, i] * X[:, j]).reshape(-1, 1))
        return np.hstack(parts)

    def _pick_interactions(self, X, y):
        """Pick top-K interactions by |corr(x_i * x_j - linear_pred, y_resid)|."""
        n = X.shape[1]
        if n < 2 or self.max_interactions <= 0:
            return []
        # Fit linear quick to get residual
        X_aug = np.hstack([X, np.ones((len(X), 1))])
        beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        resid = y - X_aug @ beta
        sd_resid = np.std(resid)
        if sd_resid < 1e-9:
            return []
        # Enumerate pairs, compute correlation with residual
        pairs = []
        for i in range(n):
            for j in range(i, n):  # include self (i==j → quadratic)
                inter = X[:, i] * X[:, j]
                sd_inter = np.std(inter)
                if sd_inter < 1e-9:
                    continue
                corr = np.corrcoef(inter, resid)[0, 1]
                pairs.append((abs(corr), i, j))
        pairs.sort(reverse=True)
        return [(i, j) for _, i, j in pairs[:self.max_interactions]]

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.n_features_in_ = X.shape[1]
        self.feature_names_ = [f"x{i}" for i in range(self.n_features_in_)]

        qs = np.linspace(1.0 / (self.n_knots + 1),
                         self.n_knots / (self.n_knots + 1),
                         self.n_knots)
        self.knots_ = [np.quantile(X[:, f], qs) for f in range(self.n_features_in_)]
        self.interactions_ = self._pick_interactions(X, y)

        basis = self._build_basis(X)
        self.scaler_ = StandardScaler()
        Bs = self.scaler_.fit_transform(basis)

        lasso = LassoCV(cv=self.cv, max_iter=self.max_iter, random_state=42)
        lasso.fit(Bs, y)

        effects = np.abs(lasso.coef_)
        if effects.max() > 0:
            keep_mask = effects >= self.min_effect_frac * effects.max()
        else:
            keep_mask = np.zeros_like(effects, dtype=bool)

        # Option A: Keep Lasso coefs (no debias)
        # Option B: OLS debias on kept basis
        if getattr(self, "ols_debias", False) and keep_mask.any():
            # Ridge refit for stability (not pure OLS)
            from sklearn.linear_model import Ridge
            B_kept = basis[:, keep_mask]
            ridge_refit = Ridge(alpha=1e-3)
            ridge_refit.fit(B_kept, y)
            full_coefs_orig = np.zeros(len(lasso.coef_))
            idx = 0
            for k, keep in enumerate(keep_mask):
                if keep:
                    full_coefs_orig[k] = ridge_refit.coef_[idx]
                    idx += 1
            self.coef_ = full_coefs_orig
            self.intercept_ = float(ridge_refit.intercept_)
        else:
            pruned_std = lasso.coef_ * keep_mask
            scale = self.scaler_.scale_
            mean = self.scaler_.mean_
            coef_orig = pruned_std / np.where(scale > 0, scale, 1.0)
            intercept_orig = float(lasso.intercept_) - float(np.dot(coef_orig, mean))
            self.coef_ = coef_orig
            self.intercept_ = float(intercept_orig)
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        X = np.asarray(X, dtype=np.float64)
        basis = self._build_basis(X)
        return basis @ self.coef_ + self.intercept_

    def _feature_segments(self, feat_i):
        """For a single feature, compute the piecewise slope+intercept per segment.

        Returns [(seg_low, seg_high, slope, intercept)] such that for x in [lo, hi],
        f_j(x) = slope * x + intercept.
        """
        n = self.n_features_in_
        # Linear coefficient for this feature:
        a = self.coef_[feat_i]
        # Pos+neg hinge coefs for this feature:
        hinge_offset = n
        # Hinges are laid out as: for each feature f, then for each knot k: pos, neg
        # Count hinges per feature
        knots = self.knots_[feat_i]
        hinge_start = hinge_offset
        for f in range(feat_i):
            hinge_start += 2 * len(self.knots_[f])

        pos_coefs = []
        neg_coefs = []
        for k_idx, t in enumerate(knots):
            cp = self.coef_[hinge_start + 2 * k_idx]
            cn = self.coef_[hinge_start + 2 * k_idx + 1]
            pos_coefs.append(cp)
            neg_coefs.append(cn)

        # Build segments at knots sorted ascending.
        sorted_pairs = sorted(zip(knots, pos_coefs, neg_coefs), key=lambda x: x[0])
        sorted_knots = [p[0] for p in sorted_pairs]

        # The total contribution of feature f to y is:
        #   a*x + sum_k [ cp_k * max(0, x - t_k) + cn_k * max(0, t_k - x) ]
        # For a given x, each hinge is either active or not.
        # We walk through segments from -inf to +inf.
        segments = []
        if not sorted_knots:
            # no knots for this feature — pure linear
            return [(-np.inf, np.inf, a, 0.0)]

        # For x in segment defined by sorted_knots[i-1] <= x < sorted_knots[i]
        # (with sentinels -inf and +inf), determine slope and intercept.
        # At each knot t_k:
        #   max(0, x - t_k) is active iff x > t_k → contributes cp_k * (x - t_k)
        #   max(0, t_k - x) is active iff x < t_k → contributes cn_k * (t_k - x)
        # So for x in segment [lo, hi]:
        #   active pos hinges: those with t_k < lo (i.e., t_k < x always)
        #   active neg hinges: those with t_k > hi (i.e., t_k > x always)

        boundaries = [-np.inf] + sorted_knots + [np.inf]
        for i in range(len(boundaries) - 1):
            lo = boundaries[i]
            hi = boundaries[i + 1]
            slope = a
            intercept_seg = 0.0
            for t_k, cp_k, cn_k in sorted_pairs:
                if t_k <= lo:  # x > t_k in this segment → pos hinge active
                    slope += cp_k
                    intercept_seg += -cp_k * t_k
                if t_k >= hi:  # x < t_k in this segment → neg hinge active
                    slope += -cn_k
                    intercept_seg += cn_k * t_k
            segments.append((lo, hi, slope, intercept_seg))
        return segments

    def _feature_importance(self, feat_i, X_sample=None):
        """Estimate feature importance as the standard deviation of its contribution across training data."""
        # Quick proxy: max |slope| across segments * typical x range
        segs = self._feature_segments(feat_i)
        max_slope = max((abs(s[2]) for s in segs), default=0.0)
        return max_slope

    def _merge_near_identical(self, segs):
        """Merge adjacent segments that predict nearly the same y across their combined range.

        Criterion: for two adjacent segments sharing knot t, merge if
        |slope1 - slope2| * typical_x_scale <= tol * max(|f_segment|).
        """
        if not segs:
            return segs
        max_slope = max((abs(s[2]) for s in segs), default=1e-9)

        merged = [list(segs[0])]
        for s in segs[1:]:
            lo, hi, sl, ic = s
            lo_p, hi_p, sl_p, ic_p = merged[-1]
            # Compare slope only. If slope difference is <= tol * max_slope, merge.
            if abs(sl - sl_p) <= self.seg_slope_tol * max(max_slope, 1e-9):
                # Extend the previous segment's range; keep its slope/intercept.
                merged[-1][1] = hi
            else:
                merged.append(list(s))
        return [tuple(m) for m in merged]

    def __str__(self):
        check_is_fitted(self, "coef_")
        n = self.n_features_in_
        inter_start = n + n * 2 * self.n_knots

        # Gather segments and determine if any feature is truly piecewise
        inactive = []
        feat_info = []
        for i in range(n):
            segs = self._feature_segments(i)
            segs = self._merge_near_identical(segs)
            nonzero = any(abs(s[2]) > 1e-10 or abs(s[3]) > 1e-10 for s in segs)
            if not nonzero:
                inactive.append(self.feature_names_[i])
                continue
            importance = max((abs(s[2]) for s in segs), default=0.0)
            feat_info.append((i, segs, importance))
        feat_info.sort(key=lambda x: -x[2])
        any_piecewise = any(len(segs) > 1 for _, segs, _ in feat_info)

        # Active interactions (if any)
        active_interactions = []
        for k, (i, j) in enumerate(self.interactions_):
            c = self.coef_[inter_start + k]
            if abs(c) > 1e-10:
                xi = self.feature_names_[i]
                xj = self.feature_names_[j]
                desc = f"{c:.4f}*{xi}^2" if i == j else f"{c:.4f}*{xi}*{xj}"
                active_interactions.append((abs(c), desc, c))
        has_interactions = len(active_interactions) > 0

        # Always piecewise format (proven best on interp tests)
        header = ("Additive piecewise-linear model: y = intercept + sum_j f_j(x_j)"
                  + (" + interaction_terms." if has_interactions else "."))
        lines = [header, "", f"intercept = {self.intercept_:.3f}", ""]

        for i, segs, _ in feat_info:
            name = self.feature_names_[i]
            if len(segs) == 1:
                s = segs[0]
                lines.append(f"f_{i}({name}) = {s[2]:.3f}*{name} + {s[3]:.3f}")
            else:
                lines.append(f"f_{i}({name}) = piecewise in {name}:")
                for lo, hi, slope, icpt in segs:
                    lo_s = "-∞" if lo == -np.inf else f"{lo:.2f}"
                    hi_s = "+∞" if hi == np.inf else f"{hi:.2f}"
                    lines.append(f"    for {name} in [{lo_s}, {hi_s}]: f = {slope:.3f}*{name} + {icpt:.3f}")

        if active_interactions:
            active_interactions.sort(key=lambda x: -x[0])
            lines.append("")
            lines.append("Interaction terms:")
            for _, desc, _ in active_interactions:
                lines.append(f"    {desc}")

        if inactive:
            lines.append("")
            lines.append(f"Features with zero contribution: {', '.join(inactive)}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
PiecewiseGAM_Plus.__module__ = "interpretable_regressor"

model_shorthand_name = "PiecewiseGAM_3dp"
model_description = "PiecewiseGAM with 3-decimal-place precision (instead of 4) for cleaner display."
model_defs = [(model_shorthand_name, PiecewiseGAM_Plus(min_effect_frac=0.03, max_interactions=0, seg_slope_tol=0.0, ols_debias=False))]


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
