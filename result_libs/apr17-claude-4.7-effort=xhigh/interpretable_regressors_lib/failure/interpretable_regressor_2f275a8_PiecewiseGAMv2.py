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


class PiecewiseLinearGAM(BaseEstimator, RegressorMixin):
    """
    Generalized additive model: y = intercept + sum_j f_j(x_j),
    where each f_j is a continuous piecewise-linear function with a small
    number of knots.

    Fit: same hinge basis as SparseHingeLinear, LassoCV to pick support.
    Display: per-feature piecewise description — LLM can simulate by
    evaluating just ONE feature's f_j, then summing across features.

    For each feature, the piecewise function is canonicalized into a
    sequence of linear segments between knots (easier to reason about
    than a sum of hinge terms).
    """

    def __init__(self, n_knots=3, cv=3, max_iter=5000, min_effect_frac=0.03):
        self.n_knots = n_knots
        self.cv = cv
        self.max_iter = max_iter
        self.min_effect_frac = min_effect_frac

    def _build_basis(self, X):
        n = self.n_features_in_
        parts = [X]
        for f in range(n):
            for t in self.knots_[f]:
                parts.append(np.maximum(0.0, X[:, f] - t).reshape(-1, 1))
                parts.append(np.maximum(0.0, t - X[:, f]).reshape(-1, 1))
        return np.hstack(parts)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.n_features_in_ = X.shape[1]
        self.feature_names_ = [f"x{i}" for i in range(self.n_features_in_)]
        qs = np.linspace(1.0 / (self.n_knots + 1),
                         self.n_knots / (self.n_knots + 1),
                         self.n_knots)
        self.knots_ = [np.quantile(X[:, f], qs) for f in range(self.n_features_in_)]

        basis = self._build_basis(X)
        self.scaler_ = StandardScaler()
        Bs = self.scaler_.fit_transform(basis)
        lasso = LassoCV(cv=self.cv, max_iter=self.max_iter, random_state=42)
        lasso.fit(Bs, y)

        # Prune small standardized effects
        effects = np.abs(lasso.coef_)
        if effects.max() > 0:
            keep_mask = effects >= self.min_effect_frac * effects.max()
        else:
            keep_mask = np.zeros_like(effects, dtype=bool)
        pruned_std = lasso.coef_ * keep_mask

        scale = self.scaler_.scale_
        mean = self.scaler_.mean_
        coef_orig = pruned_std / np.where(scale > 0, scale, 1.0)
        intercept_orig = lasso.intercept_ - np.dot(coef_orig, mean)
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

    def _merge_segments(self, segs):
        """Merge consecutive segments that have the same (slope, intercept)."""
        if not segs:
            return segs
        merged = [segs[0]]
        for s in segs[1:]:
            lo_prev, hi_prev, sl_prev, ic_prev = merged[-1]
            lo, hi, sl, ic = s
            if abs(sl - sl_prev) < 1e-8 and abs(ic - ic_prev) < 1e-8:
                merged[-1] = (lo_prev, hi, sl, ic)
            else:
                merged.append(s)
        return merged

    def _fmt_linear(self, slope, icpt, name):
        # Avoid "-0.0000*..." noise.
        s_round = 0.0 if abs(slope) < 1e-5 else slope
        i_round = 0.0 if abs(icpt) < 1e-5 else icpt
        if abs(s_round) < 1e-10 and abs(i_round) < 1e-10:
            return "0"
        if abs(s_round) < 1e-10:
            return f"{i_round:.4f}"
        if abs(i_round) < 1e-10:
            return f"{s_round:.4f}*{name}"
        sign = "+" if i_round >= 0 else "-"
        return f"{s_round:.4f}*{name} {sign} {abs(i_round):.4f}"

    def __str__(self):
        check_is_fitted(self, "coef_")
        lines = [
            "Additive piecewise-linear model: y = intercept + sum_j f_j(x_j).",
            "To predict: for each feature, pick the segment containing x_j, compute f_j, sum all.",
            "",
            f"intercept = {self.intercept_:.4f}",
            "",
        ]
        inactive = []
        for i in range(self.n_features_in_):
            segs = self._feature_segments(i)
            segs = self._merge_segments(segs)
            nonzero = any(abs(s[2]) > 1e-10 or abs(s[3]) > 1e-10 for s in segs)
            if not nonzero:
                inactive.append(self.feature_names_[i])
                continue
            name = self.feature_names_[i]
            if len(segs) == 1:
                s = segs[0]
                lines.append(f"f_{i}({name}) = {self._fmt_linear(s[2], s[3], name)}")
            else:
                lines.append(f"f_{i}({name}) = piecewise:")
                for lo, hi, slope, icpt in segs:
                    lo_s = "-inf" if lo == -np.inf else f"{lo:.3f}"
                    hi_s = "+inf" if hi == np.inf else f"{hi:.3f}"
                    lines.append(f"    if {lo_s} <= {name} < {hi_s}: f = {self._fmt_linear(slope, icpt, name)}")

        if inactive:
            lines.append("")
            lines.append(f"Features with zero contribution: {', '.join(inactive)}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
PiecewiseLinearGAM.__module__ = "interpretable_regressor"

model_shorthand_name = "PiecewiseGAMv2"
model_description = "Hinge basis LassoCV, per-feature piecewise display with segment merging and cleaner formatting."
model_defs = [(model_shorthand_name, PiecewiseLinearGAM())]


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
