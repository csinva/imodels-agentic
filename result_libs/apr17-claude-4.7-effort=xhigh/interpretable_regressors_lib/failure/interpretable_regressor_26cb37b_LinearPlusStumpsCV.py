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


class LinearModelTree(BaseEstimator, RegressorMixin):
    """
    Depth-1 decision tree with linear leaves.

    Splits once on the best feature/threshold (by MSE), then fits a separate
    Ridge regression in each leaf. Both leaves are displayed as equations.

    Display: "if x_j > t: y = eq_right else: y = eq_left"
    LLM can simulate by selecting branch then evaluating linear equation.
    """

    def __init__(self, cv=3, min_leaf_samples=50, n_candidate_thresholds=15):
        self.cv = cv
        self.min_leaf_samples = min_leaf_samples
        self.n_candidate_thresholds = n_candidate_thresholds

    def _fit_leaf(self, X, y):
        from sklearn.linear_model import RidgeCV
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = RidgeCV(cv=min(self.cv, max(2, len(y) // 50)))
        model.fit(Xs, y)
        # Convert to original scale
        scale = scaler.scale_
        mean = scaler.mean_
        coef = model.coef_ / np.where(scale > 0, scale, 1.0)
        intercept = float(model.intercept_) - float(np.dot(coef, mean))
        return coef, intercept

    def _find_best_split(self, X, y):
        """Find (feature, threshold) that best reduces residual MSE
        by fitting linear models in each side."""
        n_samples, n_features = X.shape
        best = None
        best_mse = float("inf")
        qs = np.linspace(0.2, 0.8, self.n_candidate_thresholds)
        for i in range(n_features):
            xf = X[:, i]
            if np.std(xf) < 1e-10:
                continue
            candidates = np.unique(np.round(np.quantile(xf, qs), 6))
            for t in candidates:
                mask = xf > t
                n_right = int(mask.sum())
                n_left = n_samples - n_right
                if n_right < self.min_leaf_samples or n_left < self.min_leaf_samples:
                    continue
                # Fit linear on each side; use Ridge for stability
                try:
                    coef_l, intc_l = self._fit_leaf(X[~mask], y[~mask])
                    coef_r, intc_r = self._fit_leaf(X[mask], y[mask])
                except Exception:
                    continue
                # Compute MSE
                pred_l = X[~mask] @ coef_l + intc_l
                pred_r = X[mask] @ coef_r + intc_r
                mse = (np.sum((y[~mask] - pred_l) ** 2) + np.sum((y[mask] - pred_r) ** 2)) / n_samples
                if mse < best_mse:
                    best_mse = mse
                    best = (i, float(t), coef_l, intc_l, coef_r, intc_r)
        return best

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.n_features_in_ = X.shape[1]
        self.feature_names_ = [f"x{i}" for i in range(self.n_features_in_)]

        # Fit baseline linear
        self.coef_baseline_, self.intercept_baseline_ = self._fit_leaf(X, y)
        # Try to find a split
        split = self._find_best_split(X, y)
        # Only keep split if it improves over baseline
        if split is not None:
            pred_baseline = X @ self.coef_baseline_ + self.intercept_baseline_
            mse_base = float(np.mean((y - pred_baseline) ** 2))
            # Evaluate split MSE
            i, t, coef_l, intc_l, coef_r, intc_r = split
            mask = X[:, i] > t
            pred_split = np.where(mask, X @ coef_r + intc_r, X @ coef_l + intc_l)
            mse_split = float(np.mean((y - pred_split) ** 2))
            if mse_split < mse_base * 0.97:  # require 3% improvement
                self.split_ = split
            else:
                self.split_ = None
        else:
            self.split_ = None
        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_baseline_"])
        X = np.asarray(X, dtype=np.float64)
        if self.split_ is None:
            return X @ self.coef_baseline_ + self.intercept_baseline_
        i, t, coef_l, intc_l, coef_r, intc_r = self.split_
        mask = X[:, i] > t
        pred = np.where(mask, X @ coef_r + intc_r, X @ coef_l + intc_l)
        return pred

    def _fmt_eq(self, coef, intercept):
        names = self.feature_names_
        parts = [(n, c) for n, c in zip(names, coef) if abs(c) > 1e-10]
        parts.sort(key=lambda x: -abs(x[1]))
        if not parts:
            return f"{intercept:.4f}"
        eq = " + ".join(f"{c:.4f}*{n}" for n, c in parts)
        eq += f" + {intercept:.4f}"
        return eq

    def __str__(self):
        check_is_fitted(self, ["coef_baseline_"])
        if self.split_ is None:
            return f"Linear regression (no split found):\n  y = {self._fmt_eq(self.coef_baseline_, self.intercept_baseline_)}"

        i, t, coef_l, intc_l, coef_r, intc_r = self.split_
        name = self.feature_names_[i]
        lines = [
            f"Linear Model Tree: split on {name} at {t:.4f}",
            "",
            f"if {name} <= {t:.4f}:",
            f"  y = {self._fmt_eq(coef_l, intc_l)}",
            "",
            f"if {name} > {t:.4f}:",
            f"  y = {self._fmt_eq(coef_r, intc_r)}",
        ]
        return "\n".join(lines)


class LinearPlusStumps(BaseEstimator, RegressorMixin):
    """
    Linear base + a small budget of additive stumps on residuals.

    Architecture:
      1. Fit RidgeCV on standardized features (linear base)
      2. Compute residuals
      3. Greedy boosting: at each step, find single feature & threshold that
         best reduces residual MSE with a stump ("+delta if x_i > t"),
         up to `max_stumps`.
      4. Display: linear equation + list of stumps
      5. Predict uses linear + stumps (matches display)
    """

    def __init__(self, cv=3, max_stumps=6, min_stump_frac=0.005,
                 n_candidate_thresholds=15, unique_features=False):
        self.cv = cv
        self.max_stumps = max_stumps
        self.min_stump_frac = min_stump_frac
        self.n_candidate_thresholds = n_candidate_thresholds
        self.unique_features = unique_features

    def _best_stump(self, X, residual, exclude_features=None):
        """Find (feat_i, threshold, delta) that best reduces MSE with
           stump f(x) = delta * I(x_i > threshold)."""
        n = X.shape[1]
        best = None
        best_gain = 0.0
        sse_base = float(np.sum(residual ** 2))
        qs = np.linspace(0.1, 0.9, self.n_candidate_thresholds)
        exclude = exclude_features or set()
        for i in range(n):
            if i in exclude:
                continue
            xf = X[:, i]
            candidates = np.unique(np.round(np.quantile(xf, qs), 6))
            for t in candidates:
                mask = xf > t
                n_right = int(mask.sum())
                if n_right < 20 or (len(xf) - n_right) < 20:
                    continue
                mean_right = float(residual[mask].mean())
                # If stump is mean-based, optimal delta = mean_right - mean_left,
                # but we're only adding ON RIGHT → shift.
                # For mean-based predictions, optimal delta for "+delta * I(x>t)" is mean_right - overall_mean.
                # Actually for additive residual on residual (which has mean 0 after RidgeCV), it's just mean_right.
                # But more precisely: minimize sum((r - delta*I)^2) → delta = mean_right
                delta = mean_right
                # Change in SSE: from sum(r^2) to sum((r - delta*I)^2)
                # = sum(r^2) - 2*delta*sum(r*I) + delta^2 * n_right
                # = sum(r^2) - 2*delta*(n_right*mean_right) + delta^2 * n_right
                # Gain = n_right * mean_right^2
                gain = n_right * mean_right ** 2
                if gain > best_gain:
                    best_gain = gain
                    best = (i, float(t), delta)
        return best, best_gain, sse_base

    def fit(self, X, y):
        from sklearn.linear_model import RidgeCV
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.n_features_in_ = X.shape[1]
        self.feature_names_ = [f"x{i}" for i in range(self.n_features_in_)]

        # Linear base (RidgeCV, no feature scaling here — RidgeCV handles it)
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X)
        self.ridge_ = RidgeCV(cv=self.cv)
        self.ridge_.fit(Xs, y)

        # Convert coefs to original scale
        scale = self.scaler_.scale_
        mean = self.scaler_.mean_
        self.coef_ = self.ridge_.coef_ / np.where(scale > 0, scale, 1.0)
        self.intercept_ = float(self.ridge_.intercept_) - float(np.dot(self.coef_, mean))

        # Compute residuals and fit stumps greedily
        residual = y - (X @ self.coef_ + self.intercept_)
        total_sse_init = float(np.sum(residual ** 2))

        self.stumps_ = []  # list of (feat_i, threshold, delta)
        used_features = set()
        for _ in range(self.max_stumps):
            exclude = used_features if self.unique_features else None
            stump, gain, sse = self._best_stump(X, residual, exclude_features=exclude)
            if stump is None:
                break
            if gain < self.min_stump_frac * total_sse_init:
                break
            i, t, delta = stump
            used_features.add(i)
            self.stumps_.append((i, t, delta))
            # Apply stump to residuals
            residual = residual - delta * (X[:, i] > t).astype(np.float64)
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        X = np.asarray(X, dtype=np.float64)
        y = X @ self.coef_ + self.intercept_
        for i, t, delta in self.stumps_:
            y = y + delta * (X[:, i] > t).astype(np.float64)
        return y

    def __str__(self):
        check_is_fitted(self, "coef_")
        names = self.feature_names_
        # Sort linear coefs by |coef|
        order = np.argsort(-np.abs(self.coef_))
        sorted_pairs = [(names[i], float(self.coef_[i])) for i in order]
        # Build linear equation (OLS-style)
        active = [(n, c) for n, c in sorted_pairs if abs(c) > 1e-10]
        zeroed = [n for n, c in sorted_pairs if abs(c) <= 1e-10]
        equation = " + ".join(f"{c:.4f}*{n}" for n, c in active) if active else ""
        if equation:
            equation += f" + {self.intercept_:.4f}"
        else:
            equation = f"{self.intercept_:.4f}"

        lines = [
            "Linear base + stump corrections. Prediction recipe:",
            f"  y = {equation}",
            "",
            "Then apply each stump (add delta if condition is true):",
        ]
        for i, t, delta in self.stumps_:
            nm = names[i]
            lines.append(f"  if {nm} > {t:.4f}: add {delta:.4f}")
        if not self.stumps_:
            lines.append("  (none — pure linear)")

        lines.append("")
        lines.append("Linear coefficients (sorted by |coef|):")
        for n, c in active:
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {self.intercept_:.4f}")
        if zeroed:
            lines.append(f"  Features with zero coefficients (excluded): {', '.join(zeroed)}")
        return "\n".join(lines)


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

    def _feature_importance(self, feat_i):
        """Importance = max |slope| across segments * feature std (scale-free)."""
        segs = self._feature_segments(feat_i)
        max_slope = max((abs(s[2]) for s in segs), default=0.0)
        # Multiply by feature std to get scale-free importance
        scale = getattr(self, "scaler_", None)
        x_std = 1.0
        if scale is not None:
            x_std = float(scale.scale_[feat_i]) if feat_i < len(scale.scale_) else 1.0
        return max_slope * x_std

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
            importance = self._feature_importance(i)
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
        lines = [header, "", f"intercept = {self.intercept_:.4f}", ""]

        for i, segs, _ in feat_info:
            name = self.feature_names_[i]
            if len(segs) == 1:
                s = segs[0]
                lines.append(f"f_{i}({name}) = {s[2]:.4f}*{name} + {s[3]:.4f}")
            else:
                lines.append(f"f_{i}({name}) = piecewise in {name}:")
                for lo, hi, slope, icpt in segs:
                    lo_s = "-∞" if lo == -np.inf else f"{lo:.3f}"
                    hi_s = "+∞" if hi == np.inf else f"{hi:.3f}"
                    lines.append(f"    for {name} in [{lo_s}, {hi_s}]: f = {slope:.4f}*{name} + {icpt:.4f}")

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

class DecisionListLinear(BaseEstimator, RegressorMixin):
    """
    Sequential decision list with linear catch-all.

    Algorithm:
      1. Fit a depth-3 decision tree to capture structure
      2. Extract leaves as "rule → constant" pairs
      3. For "catch-all" leaf (the largest leaf OR random fallback), replace with
         linear model on just the remaining data
      4. Display: sequential if-elif-else

    Display format:
      if x0 > 1.5 and x1 > 0.3: y = 5.2
      elif x0 > 1.5: y = 3.1
      elif x1 < -0.5: y = -1.2
      else: y = 0.5*x0 + 0.3*x1 + 0.1

    LLM evaluates sequentially - the first matching rule wins.
    """

    def __init__(self, max_leaves=6, cv=3):
        self.max_leaves = max_leaves
        self.cv = cv

    def _fit_linear(self, X, y):
        from sklearn.linear_model import RidgeCV
        if len(y) < 10:
            return np.zeros(X.shape[1]), float(np.mean(y) if len(y) else 0.0)
        scaler = StandardScaler()
        try:
            Xs = scaler.fit_transform(X)
            cv = min(self.cv, max(2, len(y) // 50))
            model = RidgeCV(cv=cv)
            model.fit(Xs, y)
            scale = scaler.scale_
            mean = scaler.mean_
            coef = model.coef_ / np.where(scale > 0, scale, 1.0)
            intercept = float(model.intercept_) - float(np.dot(coef, mean))
            return coef, intercept
        except Exception:
            return np.zeros(X.shape[1]), float(np.mean(y))

    def fit(self, X, y):
        from sklearn.tree import DecisionTreeRegressor
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.n_features_in_ = X.shape[1]
        self.feature_names_ = [f"x{i}" for i in range(self.n_features_in_)]

        # Fit tree to find rules
        tree = DecisionTreeRegressor(max_leaf_nodes=self.max_leaves, random_state=42)
        tree.fit(X, y)

        # Extract rules for each leaf as sequence of (feature, threshold, direction)
        tree_ = tree.tree_
        leaves = []  # list of (path_conditions, leaf_prediction, n_samples, leaf_mask_indices)

        def collect(node_id, conditions):
            if tree_.children_left[node_id] == -1:
                # Leaf
                leaves.append((conditions, float(tree_.value[node_id].ravel()[0]),
                               int(tree_.n_node_samples[node_id]), node_id))
            else:
                feat = tree_.feature[node_id]
                thresh = tree_.threshold[node_id]
                left_cond = conditions + [(feat, thresh, "<=")]
                right_cond = conditions + [(feat, thresh, ">")]
                collect(tree_.children_left[node_id], left_cond)
                collect(tree_.children_right[node_id], right_cond)
        collect(0, [])

        # Sort leaves by sample count desc (biggest catch-all last)
        leaves.sort(key=lambda x: -x[2])
        # The LARGEST leaf becomes catch-all (last rule)
        catch_all_idx = 0  # pick biggest
        catch_all_leaf = leaves.pop(catch_all_idx)

        # Now leaves[] are the rules to check first (they match fewer points, more specific)
        # For the catch-all, fit linear on those samples
        leaves_order = leaves  # order of check: specific → general
        # Compute mask for catch-all: all points NOT matched by any preceding rule
        # Since the tree leaves PARTITION the data, catch-all samples are exactly those not in leaves_order

        # Determine sample assignment via tree's apply
        leaf_ids = tree.apply(X)
        catch_all_node = catch_all_leaf[3]
        catch_all_mask = leaf_ids == catch_all_node

        # Fit linear on catch-all samples
        if catch_all_mask.sum() >= 10:
            self.catch_all_coef_, self.catch_all_intercept_ = self._fit_linear(
                X[catch_all_mask], y[catch_all_mask]
            )
        else:
            self.catch_all_coef_, self.catch_all_intercept_ = self._fit_linear(X, y)

        # Store rules: list of (conditions, prediction_constant)
        self.rules_ = [(cond, pred) for cond, pred, _, _ in leaves_order]
        return self

    def predict(self, X):
        check_is_fitted(self, ["rules_"])
        X = np.asarray(X, dtype=np.float64)
        y = np.empty(len(X))
        matched = np.zeros(len(X), dtype=bool)
        for cond, pred in self.rules_:
            mask = ~matched
            for feat, thresh, direction in cond:
                if direction == "<=":
                    mask = mask & (X[:, feat] <= thresh)
                else:
                    mask = mask & (X[:, feat] > thresh)
            y[mask] = pred
            matched = matched | mask
        # Catch-all for remaining
        if (~matched).any():
            y[~matched] = X[~matched] @ self.catch_all_coef_ + self.catch_all_intercept_
        return y

    def __str__(self):
        check_is_fitted(self, ["rules_"])
        names = self.feature_names_
        lines = ["Decision list (check rules top-to-bottom; first match wins):", ""]
        for cond, pred in self.rules_:
            parts = []
            for feat, thresh, direction in cond:
                parts.append(f"{names[feat]} {direction} {thresh:.4f}")
            cond_str = " AND ".join(parts) if parts else "True"
            lines.append(f"  if {cond_str}: y = {pred:.4f}")
        # Catch-all linear
        active = [(names[i], c) for i, c in enumerate(self.catch_all_coef_) if abs(c) > 1e-10]
        active.sort(key=lambda x: -abs(x[1]))
        if active:
            eq = " + ".join(f"{c:.4f}*{n}" for n, c in active)
            eq += f" + {self.catch_all_intercept_:.4f}"
        else:
            eq = f"{self.catch_all_intercept_:.4f}"
        lines.append(f"  else: y = {eq}")
        return "\n".join(lines)


class PolynomialPlusStumps(BaseEstimator, RegressorMixin):
    """
    Polynomial basis (x, x^2, x^3) with LassoCV for sparsity + greedy stumps.

    Rationale: polynomials are easy for LLM to compute (x=0.5 → x^2=0.25 → x^3=0.125)
    and capture smooth nonlinearities. LassoCV picks which powers matter per feature.
    """

    def __init__(self, cv=3, max_stumps=8, min_stump_frac=0.005,
                 n_candidate_thresholds=15, min_effect_frac=0.03):
        self.cv = cv
        self.max_stumps = max_stumps
        self.min_stump_frac = min_stump_frac
        self.n_candidate_thresholds = n_candidate_thresholds
        self.min_effect_frac = min_effect_frac

    def _build_poly_basis(self, X):
        """Build [x_0, x_0^2, x_0^3, x_1, x_1^2, x_1^3, ...]."""
        parts = []
        for i in range(self.n_features_in_):
            xf = X[:, i]
            parts.append(xf.reshape(-1, 1))
            parts.append((xf * xf).reshape(-1, 1))
            parts.append((xf * xf * xf).reshape(-1, 1))
        return np.hstack(parts)

    def _best_stump(self, X, residual):
        n = X.shape[1]
        best = None
        best_gain = 0.0
        sse_base = float(np.sum(residual ** 2))
        qs = np.linspace(0.1, 0.9, self.n_candidate_thresholds)
        for i in range(n):
            xf = X[:, i]
            candidates = np.unique(np.round(np.quantile(xf, qs), 6))
            for t in candidates:
                mask = xf > t
                n_right = int(mask.sum())
                if n_right < 20 or (len(xf) - n_right) < 20:
                    continue
                mean_right = float(residual[mask].mean())
                gain = n_right * mean_right ** 2
                if gain > best_gain:
                    best_gain = gain
                    best = (i, float(t), mean_right)
        return best, best_gain, sse_base

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.n_features_in_ = X.shape[1]
        self.feature_names_ = [f"x{i}" for i in range(self.n_features_in_)]

        basis = self._build_poly_basis(X)
        self.scaler_ = StandardScaler()
        Bs = self.scaler_.fit_transform(basis)
        lasso = LassoCV(cv=self.cv, max_iter=5000, random_state=42)
        lasso.fit(Bs, y)

        # Prune tiny effects
        effects = np.abs(lasso.coef_)
        if effects.max() > 0:
            keep_mask = effects >= self.min_effect_frac * effects.max()
        else:
            keep_mask = np.zeros_like(effects, dtype=bool)
        pruned_std = lasso.coef_ * keep_mask

        scale = self.scaler_.scale_
        mean = self.scaler_.mean_
        self.coef_ = pruned_std / np.where(scale > 0, scale, 1.0)
        self.intercept_ = float(lasso.intercept_) - float(np.dot(self.coef_, mean))

        # Residuals + stumps
        residual = y - (basis @ self.coef_ + self.intercept_)
        total_sse_init = float(np.sum(residual ** 2))
        self.stumps_ = []
        for _ in range(self.max_stumps):
            stump, gain, sse = self._best_stump(X, residual)
            if stump is None or gain < self.min_stump_frac * total_sse_init:
                break
            i, t, delta = stump
            self.stumps_.append((i, t, delta))
            residual = residual - delta * (X[:, i] > t).astype(np.float64)
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        X = np.asarray(X, dtype=np.float64)
        basis = self._build_poly_basis(X)
        y = basis @ self.coef_ + self.intercept_
        for i, t, delta in self.stumps_:
            y = y + delta * (X[:, i] > t).astype(np.float64)
        return y

    def __str__(self):
        check_is_fitted(self, "coef_")
        terms = []
        active_feats = set()
        for i in range(self.n_features_in_):
            nm = self.feature_names_[i]
            c1 = self.coef_[3 * i]
            c2 = self.coef_[3 * i + 1]
            c3 = self.coef_[3 * i + 2]
            if abs(c1) > 1e-10:
                terms.append((abs(c1), f"{c1:.4f}*{nm}"))
                active_feats.add(nm)
            if abs(c2) > 1e-10:
                terms.append((abs(c2), f"{c2:.4f}*{nm}^2"))
                active_feats.add(nm)
            if abs(c3) > 1e-10:
                terms.append((abs(c3), f"{c3:.4f}*{nm}^3"))
                active_feats.add(nm)
        terms.sort(key=lambda x: -x[0])
        if terms:
            equation = " + ".join(t[1] for t in terms) + f" + {self.intercept_:.4f}"
        else:
            equation = f"{self.intercept_:.4f}"

        lines = [
            "Polynomial (up to cubic) linear base + stump corrections.",
            f"  y = {equation}",
            "",
            "Then apply each stump (add delta if condition is true):",
        ]
        for i, t, delta in self.stumps_:
            nm = self.feature_names_[i]
            lines.append(f"  if {nm} > {t:.4f}: add {delta:.4f}")
        if not self.stumps_:
            lines.append("  (none)")
        inactive = [n for n in self.feature_names_ if n not in active_feats]
        if inactive:
            lines.append("")
            lines.append(f"Features with zero polynomial contribution: {', '.join(inactive)}")
        return "\n".join(lines)


class RichLinearPlusStumps(BaseEstimator, RegressorMixin):
    """
    Enriched linear base (x, sqrt(|x|), log(|x|+1)) + greedy stumps on residuals.

    For each feature, auto-select the BEST transform by correlation with y,
    use that as the feature in RidgeCV. Then add stumps on residuals.

    Display shows the transform selected per feature, making it clear why
    (e.g., "y = 3.2*sqrt(|x0|) + ... — monotone but decelerating").
    """

    def __init__(self, cv=3, max_stumps=12, min_stump_frac=0.005,
                 n_candidate_thresholds=15):
        self.cv = cv
        self.max_stumps = max_stumps
        self.min_stump_frac = min_stump_frac
        self.n_candidate_thresholds = n_candidate_thresholds

    def _candidate_transforms(self, xf):
        """Return list of (name, transformed_xf)."""
        out = [("x", xf)]
        # sqrt of |x|, signed
        out.append(("sqrt|x|_signed", np.sign(xf) * np.sqrt(np.abs(xf))))
        # log(|x|+1), signed
        out.append(("log(|x|+1)_signed", np.sign(xf) * np.log1p(np.abs(xf))))
        return out

    def _pick_transform(self, xf, y):
        """Pick transform with highest |correlation| with y."""
        best_corr = -1
        best = None
        sd_y = float(np.std(y))
        if sd_y < 1e-9:
            return ("x", xf)
        for name, x_t in self._candidate_transforms(xf):
            sd_t = float(np.std(x_t))
            if sd_t < 1e-9:
                continue
            corr = abs(np.corrcoef(x_t, y)[0, 1])
            if corr > best_corr:
                best_corr = corr
                best = (name, x_t)
        return best if best else ("x", xf)

    def _best_stump(self, X, residual):
        n = X.shape[1]
        best = None
        best_gain = 0.0
        sse_base = float(np.sum(residual ** 2))
        qs = np.linspace(0.1, 0.9, self.n_candidate_thresholds)
        for i in range(n):
            xf = X[:, i]
            candidates = np.unique(np.round(np.quantile(xf, qs), 6))
            for t in candidates:
                mask = xf > t
                n_right = int(mask.sum())
                if n_right < 20 or (len(xf) - n_right) < 20:
                    continue
                mean_right = float(residual[mask].mean())
                gain = n_right * mean_right ** 2
                if gain > best_gain:
                    best_gain = gain
                    best = (i, float(t), mean_right)
        return best, best_gain, sse_base

    def fit(self, X, y):
        from sklearn.linear_model import RidgeCV
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.n_features_in_ = X.shape[1]
        self.feature_names_ = [f"x{i}" for i in range(self.n_features_in_)]

        # Pick best transform per feature
        self.transforms_ = []
        X_transformed = np.zeros_like(X)
        for i in range(self.n_features_in_):
            name, x_t = self._pick_transform(X[:, i], y)
            self.transforms_.append(name)
            X_transformed[:, i] = x_t

        # Fit RidgeCV on transformed features
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X_transformed)
        self.ridge_ = RidgeCV(cv=self.cv)
        self.ridge_.fit(Xs, y)

        scale = self.scaler_.scale_
        mean = self.scaler_.mean_
        self.coef_ = self.ridge_.coef_ / np.where(scale > 0, scale, 1.0)
        self.intercept_ = float(self.ridge_.intercept_) - float(np.dot(self.coef_, mean))

        # Residuals and stumps on ORIGINAL features (simpler display)
        residual = y - self._predict_linear(X)
        total_sse_init = float(np.sum(residual ** 2))

        self.stumps_ = []
        for _ in range(self.max_stumps):
            stump, gain, sse = self._best_stump(X, residual)
            if stump is None or gain < self.min_stump_frac * total_sse_init:
                break
            i, t, delta = stump
            self.stumps_.append((i, t, delta))
            residual = residual - delta * (X[:, i] > t).astype(np.float64)
        return self

    def _transform_X(self, X):
        Xt = np.zeros_like(X)
        for i, name in enumerate(self.transforms_):
            xf = X[:, i]
            if name == "x":
                Xt[:, i] = xf
            elif name == "sqrt|x|_signed":
                Xt[:, i] = np.sign(xf) * np.sqrt(np.abs(xf))
            elif name == "log(|x|+1)_signed":
                Xt[:, i] = np.sign(xf) * np.log1p(np.abs(xf))
        return Xt

    def _predict_linear(self, X):
        return self._transform_X(X) @ self.coef_ + self.intercept_

    def predict(self, X):
        check_is_fitted(self, "coef_")
        X = np.asarray(X, dtype=np.float64)
        y = self._predict_linear(X)
        for i, t, delta in self.stumps_:
            y = y + delta * (X[:, i] > t).astype(np.float64)
        return y

    def _fmt_term(self, coef, name, transform):
        if transform == "x":
            return f"{coef:.4f}*{name}"
        elif transform == "sqrt|x|_signed":
            return f"{coef:.4f}*sign({name})*sqrt(|{name}|)"
        elif transform == "log(|x|+1)_signed":
            return f"{coef:.4f}*sign({name})*log(|{name}|+1)"
        return f"{coef:.4f}*{name}"

    def __str__(self):
        check_is_fitted(self, "coef_")
        names = self.feature_names_
        order = np.argsort(-np.abs(self.coef_))
        active = [(i, self.coef_[i]) for i in order if abs(self.coef_[i]) > 1e-10]
        zeroed = [names[i] for i, c in zip(range(len(self.coef_)), self.coef_) if abs(c) <= 1e-10]
        terms = [self._fmt_term(c, names[i], self.transforms_[i]) for i, c in active]
        equation = " + ".join(terms) + f" + {self.intercept_:.4f}" if terms else f"{self.intercept_:.4f}"

        lines = [
            "Linear base with per-feature transforms + stump corrections.",
            f"  y = {equation}",
            "",
            "Feature transforms used:",
        ]
        for i in range(self.n_features_in_):
            lines.append(f"  {names[i]}: transform={self.transforms_[i]}")
        lines.append("")
        lines.append("Then apply each stump (add delta if condition is true):")
        for i, t, delta in self.stumps_:
            lines.append(f"  if {names[i]} > {t:.4f}: add {delta:.4f}")
        if not self.stumps_:
            lines.append("  (none)")
        return "\n".join(lines)


class LinearPlusStumpsCV(BaseEstimator, RegressorMixin):
    """
    LinearPlusStumps with 20% held-out validation for stump count selection.

    Fit linear + try up to max_stumps stumps on TRAIN residuals. At each step,
    evaluate on held-out validation. Stop when val MSE no longer improves.
    Then refit on FULL data using the best number of stumps.
    """

    def __init__(self, cv=3, max_stumps=20, min_stump_frac=0.0,
                 n_candidate_thresholds=15, val_fraction=0.2, patience=3):
        self.cv = cv
        self.max_stumps = max_stumps
        self.min_stump_frac = min_stump_frac
        self.n_candidate_thresholds = n_candidate_thresholds
        self.val_fraction = val_fraction
        self.patience = patience

    def _fit_ridge(self, X, y):
        from sklearn.linear_model import RidgeCV
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        cv = min(self.cv, max(2, len(y) // 50))
        model = RidgeCV(cv=cv)
        model.fit(Xs, y)
        scale = scaler.scale_
        mean = scaler.mean_
        coef = model.coef_ / np.where(scale > 0, scale, 1.0)
        intercept = float(model.intercept_) - float(np.dot(coef, mean))
        return coef, intercept

    def _best_stump(self, X, residual):
        n = X.shape[1]
        best = None
        best_gain = 0.0
        qs = np.linspace(0.1, 0.9, self.n_candidate_thresholds)
        for i in range(n):
            xf = X[:, i]
            candidates = np.unique(np.round(np.quantile(xf, qs), 6))
            for t in candidates:
                mask = xf > t
                n_right = int(mask.sum())
                if n_right < 20 or (len(xf) - n_right) < 20:
                    continue
                mean_right = float(residual[mask].mean())
                gain = n_right * mean_right ** 2
                if gain > best_gain:
                    best_gain = gain
                    best = (i, float(t), mean_right)
        return best, best_gain

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.n_features_in_ = X.shape[1]
        self.feature_names_ = [f"x{i}" for i in range(self.n_features_in_)]

        # Split train/val
        n = len(X)
        rng = np.random.RandomState(42)
        idx = rng.permutation(n)
        n_val = max(50, int(n * self.val_fraction))
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Fit base linear on train
        coef, intercept = self._fit_ridge(X_tr, y_tr)

        # Greedy stumps with validation-based early stopping
        train_residual = y_tr - (X_tr @ coef + intercept)
        val_pred = X_val @ coef + intercept
        best_val_mse = float(np.mean((y_val - val_pred) ** 2))
        best_stumps = []
        patience_counter = 0

        tentative_stumps = []
        for _ in range(self.max_stumps):
            stump, gain = self._best_stump(X_tr, train_residual)
            if stump is None:
                break
            if gain < self.min_stump_frac * float(np.sum(train_residual ** 2)):
                break
            i, t, delta = stump
            tentative_stumps.append((i, t, delta))
            train_residual = train_residual - delta * (X_tr[:, i] > t).astype(np.float64)
            val_pred = val_pred + delta * (X_val[:, i] > t).astype(np.float64)
            cur_val_mse = float(np.mean((y_val - val_pred) ** 2))
            if cur_val_mse < best_val_mse:
                best_val_mse = cur_val_mse
                best_stumps = list(tentative_stumps)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        # Refit on FULL data using best number of stumps
        n_final_stumps = len(best_stumps)
        coef, intercept = self._fit_ridge(X, y)
        self.coef_ = coef
        self.intercept_ = intercept
        residual = y - (X @ coef + intercept)
        self.stumps_ = []
        for _ in range(n_final_stumps):
            stump, gain = self._best_stump(X, residual)
            if stump is None:
                break
            i, t, delta = stump
            self.stumps_.append((i, t, delta))
            residual = residual - delta * (X[:, i] > t).astype(np.float64)
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        X = np.asarray(X, dtype=np.float64)
        y = X @ self.coef_ + self.intercept_
        for i, t, delta in self.stumps_:
            y = y + delta * (X[:, i] > t).astype(np.float64)
        return y

    def __str__(self):
        check_is_fitted(self, "coef_")
        names = self.feature_names_
        order = np.argsort(-np.abs(self.coef_))
        active = [(names[i], float(self.coef_[i])) for i in order if abs(self.coef_[i]) > 1e-10]
        zeroed = [n for n, c in zip(names, self.coef_) if abs(c) <= 1e-10]
        equation = " + ".join(f"{c:.4f}*{n}" for n, c in active) if active else ""
        equation = equation + f" + {self.intercept_:.4f}" if equation else f"{self.intercept_:.4f}"

        lines = [
            "Ridge linear base + CV-selected stump corrections. Prediction recipe:",
            f"  y = {equation}",
            "",
            "Then apply each stump (add delta if condition is true):",
        ]
        for i, t, delta in self.stumps_:
            lines.append(f"  if {names[i]} > {t:.4f}: add {delta:.4f}")
        if not self.stumps_:
            lines.append("  (none)")
        lines.append("")
        lines.append("Linear coefficients (sorted by |coef|):")
        for n, c in active:
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {self.intercept_:.4f}")
        if zeroed:
            lines.append(f"  Features with zero coefficients (excluded): {', '.join(zeroed)}")
        return "\n".join(lines)


LinearPlusStumps.__module__ = "interpretable_regressor"
LinearModelTree.__module__ = "interpretable_regressor"
RichLinearPlusStumps.__module__ = "interpretable_regressor"
PolynomialPlusStumps.__module__ = "interpretable_regressor"
DecisionListLinear.__module__ = "interpretable_regressor"
LinearPlusStumpsCV.__module__ = "interpretable_regressor"
model_shorthand_name = "LinearPlusStumpsCV"
model_description = "RidgeCV linear + stumps with validation-based early stopping (20% holdout, patience 3). Refit on full data with best stump count."
model_defs = [(model_shorthand_name, LinearPlusStumpsCV(max_stumps=20, patience=3))]


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
