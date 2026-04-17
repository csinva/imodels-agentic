"""Interpretable regression models for data science analysis.

This module provides scikit-learn compatible interpretable regressors that
produce human-readable model descriptions. After fitting, call str(model)
to get a clear textual explanation of the learned relationship, including
which features matter, their directions, and their magnitudes.

Available models
----------------
SmartAdditiveRegressor
    Greedy additive boosted stumps. Learns a separate shape function for each
    feature: y = intercept + f(col_A) + f(col_B) + ...
    Features with approximately linear effects are shown as coefficients;
    nonlinear features are shown as piecewise-constant lookup tables.
    Best for interpretability and understanding nonlinear effects.

HingeEBMRegressor
    Two-stage model: LassoCV on piecewise-linear hinge features (stage 1)
    plus a hidden EBM on residuals (stage 2). Display shows a sparse linear
    equation with only the active terms.
    Best for predictive performance while maintaining interpretable display.

Key features
------------
- **Column-name aware**: Pass `feature_names=df.columns.tolist()` to `fit()`
  and `str(model)` will use actual column names instead of x0, x1, ...
- **DataFrame support**: You can pass a pandas DataFrame directly to `fit()`
  and `predict()`; column names are extracted automatically.
- **feature_effects()**: Returns a dict mapping each feature name to its
  effect direction (positive/negative/nonlinear) and importance score,
  making it easy to summarize findings for a research question.

Recommended workflow for answering a research question
------------------------------------------------------
1. Identify the dependent variable (DV) and independent variable (IV) from
   the research question.
2. Include ALL other available numeric columns as control variables.
3. Fit the model on all features, then examine `str(model)` and
   `model.feature_effects()` to see:
   - Whether the IV has a significant effect after controlling for other vars
   - The direction and magnitude of the IV's effect
   - Which control variables also matter
4. Compare with OLS regression results for consistency.
5. Base your Likert score on whether the IV's effect survives controlling
   for confounders — if a bivariate relationship disappears after adding
   controls, the answer should lean toward "No".

Example
-------
>>> import pandas as pd
>>> from interp_models import SmartAdditiveRegressor
>>>
>>> df = pd.read_csv("hurricane.csv")
>>> y = df["deaths"]
>>> X = df.drop(columns=["deaths", "name"])  # keep all numeric controls
>>>
>>> model = SmartAdditiveRegressor()
>>> model.fit(X, y)  # column names extracted automatically from DataFrame
>>> print(model)      # shows equation with actual column names
>>>
>>> effects = model.feature_effects()
>>> print(effects)
>>> # {'femininity': {'direction': 'positive', 'importance': 0.42, 'rank': 1},
>>> #  'pressure':   {'direction': 'negative', 'importance': 0.31, 'rank': 2},
>>> #  ...}
>>>
>>> # Check: does femininity still matter after controlling for pressure, wind, etc.?
>>> if effects.get('femininity', {}).get('importance', 0) > 0.05:
>>>     print("Femininity has a meaningful effect even after controlling for other vars")
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.utils.validation import check_is_fitted


def _to_array_and_names(X, feature_names=None):
    """Convert X to numpy array and extract feature names if available."""
    if isinstance(X, pd.DataFrame):
        names = X.columns.tolist()
        arr = X.values.astype(np.float64)
    else:
        arr = np.asarray(X, dtype=np.float64)
        names = feature_names
    return arr, names


class SmartAdditiveRegressor(BaseEstimator, RegressorMixin):
    """Greedy additive boosted stumps with adaptive display.

    Builds an additive model: y = intercept + f(col_A) + f(col_B) + ...
    where each f is a univariate shape function learned by greedily boosting
    depth-1 decision stumps.

    Display (str(model)) classifies each feature's shape:
    - Approximately linear (R^2 > 0.90): shown as a coefficient
    - Nonlinear: shown as a piecewise-constant lookup table with thresholds

    Parameters
    ----------
    n_rounds : int, default=200
        Number of boosting rounds.
    learning_rate : float, default=0.1
        Shrinkage per stump.
    min_samples_leaf : int, default=5
        Minimum samples in each side of a stump split.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv("data.csv")
    >>> model = SmartAdditiveRegressor()
    >>> model.fit(df.drop(columns=["target"]), df["target"])
    >>> print(model)  # uses actual column names
    >>> effects = model.feature_effects()
    >>> # {'age': {'direction': 'positive', 'importance': 0.45, 'rank': 1}, ...}
    """

    def __init__(self, n_rounds=200, learning_rate=0.1, min_samples_leaf=5):
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y, feature_names=None):
        """Fit the model.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        feature_names : list of str, optional
            Column names. Extracted automatically if X is a DataFrame.
        """
        X, names = _to_array_and_names(X, feature_names)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.feature_names_ = names or [f"x{i}" for i in range(n_features)]
        self.intercept_ = float(np.mean(y))

        feature_stumps = defaultdict(list)
        residuals = y - self.intercept_

        for _ in range(self.n_rounds):
            best_stump = None
            best_reduction = -np.inf

            for j in range(n_features):
                xj = X[:, j]
                order = np.argsort(xj)
                xj_sorted = xj[order]
                r_sorted = residuals[order]

                cum_sum = np.cumsum(r_sorted)
                total_sum = cum_sum[-1]

                min_leaf = self.min_samples_leaf
                if n_samples < 2 * min_leaf:
                    continue
                lo = min_leaf - 1
                hi = n_samples - min_leaf - 1
                if hi < lo:
                    continue

                valid = np.where(xj_sorted[lo:hi + 1] != xj_sorted[lo + 1:hi + 2])[0] + lo
                if len(valid) == 0:
                    continue

                left_sum = cum_sum[valid]
                left_count = valid + 1
                right_sum = total_sum - left_sum
                right_count = n_samples - left_count

                reduction = left_sum ** 2 / left_count + right_sum ** 2 / right_count
                best_idx = np.argmax(reduction)

                if reduction[best_idx] > best_reduction:
                    best_reduction = reduction[best_idx]
                    split_pos = valid[best_idx]
                    threshold = (xj_sorted[split_pos] + xj_sorted[split_pos + 1]) / 2
                    left_mean = left_sum[best_idx] / left_count[best_idx]
                    right_mean = right_sum[best_idx] / right_count[best_idx]
                    best_stump = (j, threshold,
                                  left_mean * self.learning_rate,
                                  right_mean * self.learning_rate)

            if best_stump is None:
                break

            j, threshold, left_val, right_val = best_stump
            feature_stumps[j].append((threshold, left_val, right_val))

            mask = X[:, j] <= threshold
            residuals[mask] -= left_val
            residuals[~mask] -= right_val

        # Collapse into shape functions
        self.shape_functions_ = {}

        for j in range(n_features):
            stumps = feature_stumps.get(j, [])
            if not stumps:
                continue

            thresholds = sorted(set(t for t, _, _ in stumps))
            intervals = []
            for i in range(len(thresholds) + 1):
                if i == 0:
                    test_x = thresholds[0] - 1
                elif i == len(thresholds):
                    test_x = thresholds[-1] + 1
                else:
                    test_x = (thresholds[i - 1] + thresholds[i]) / 2
                val = sum(lv if test_x <= t else rv for t, lv, rv in stumps)
                intervals.append(val)

            # Laplacian smoothing
            if len(intervals) > 2:
                smooth_intervals = list(intervals)
                for _ in range(3):
                    new_intervals = [smooth_intervals[0]]
                    for k in range(1, len(smooth_intervals) - 1):
                        new_intervals.append(
                            0.6 * smooth_intervals[k] +
                            0.2 * smooth_intervals[k - 1] +
                            0.2 * smooth_intervals[k + 1]
                        )
                    new_intervals.append(smooth_intervals[-1])
                    smooth_intervals = new_intervals
                intervals = smooth_intervals

            self.shape_functions_[j] = (thresholds, intervals)

        # Feature importance
        self.feature_importances_ = np.zeros(n_features)
        for j, (thresholds, intervals) in self.shape_functions_.items():
            self.feature_importances_[j] = max(intervals) - min(intervals)

        # Linear approximation for display
        self.linear_approx_ = {}
        for j, (thresholds, intervals) in self.shape_functions_.items():
            xj = X[:, j]
            bins = np.digitize(xj, thresholds)
            fx = np.array([intervals[b] for b in bins])

            if np.std(xj) > 1e-10:
                slope = np.cov(xj, fx)[0, 1] / np.var(xj)
                offset = np.mean(fx) - slope * np.mean(xj)
                fx_linear = slope * xj + offset
                ss_res = np.sum((fx - fx_linear) ** 2)
                ss_tot = np.sum((fx - np.mean(fx)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 1.0
                self.linear_approx_[j] = (slope, offset, r2)
            else:
                self.linear_approx_[j] = (0.0, np.mean(fx), 1.0)

        return self

    def predict(self, X):
        check_is_fitted(self, "shape_functions_")
        X, _ = _to_array_and_names(X)
        n = X.shape[0]
        pred = np.full(n, self.intercept_)

        for j, (thresholds, intervals) in self.shape_functions_.items():
            xj = X[:, j]
            bins = np.digitize(xj, thresholds)
            pred += np.array([intervals[b] for b in bins])

        return pred

    def feature_effects(self):
        """Return a dict summarizing each feature's learned effect.

        Returns
        -------
        dict : {feature_name: {"direction": str, "importance": float, "rank": int}}
            - direction: "positive", "negative", "nonlinear", or "zero"
            - importance: relative importance (0-1 scale, sums to 1)
            - rank: importance rank (1 = most important)

        Use this to quickly check which features matter and their directions,
        especially useful for answering research questions about whether a
        specific variable has an effect after controlling for others.
        """
        check_is_fitted(self, "shape_functions_")
        total_imp = sum(self.feature_importances_)
        if total_imp < 1e-10:
            return {}

        effects = {}
        for j in range(self.n_features_in_):
            name = self.feature_names_[j]
            imp = self.feature_importances_[j]
            rel_imp = imp / total_imp

            if rel_imp < 0.01:
                effects[name] = {"direction": "zero", "importance": 0.0, "rank": 0}
                continue

            slope, offset, r2 = self.linear_approx_.get(j, (0, 0, 1.0))
            if r2 > 0.90:
                direction = "positive" if slope > 0 else "negative"
            else:
                # Check overall trend for nonlinear features
                thresholds, intervals = self.shape_functions_[j]
                if intervals[-1] > intervals[0] + 0.01 * total_imp:
                    direction = "nonlinear (increasing trend)"
                elif intervals[-1] < intervals[0] - 0.01 * total_imp:
                    direction = "nonlinear (decreasing trend)"
                else:
                    direction = "nonlinear (non-monotonic)"

            effects[name] = {"direction": direction, "importance": round(rel_imp, 4)}

        # Add ranks
        ranked = sorted(
            [(n, e) for n, e in effects.items() if e["importance"] > 0],
            key=lambda x: -x[1]["importance"]
        )
        for rank, (name, _) in enumerate(ranked, 1):
            effects[name]["rank"] = rank
        for name, e in effects.items():
            if e.get("rank") is None:
                e["rank"] = 0

        return effects

    def __str__(self):
        check_is_fitted(self, "shape_functions_")
        feature_names = self.feature_names_

        linear_features = {}
        nonlinear_features = {}

        total_importance = sum(self.feature_importances_)
        if total_importance < 1e-10:
            return f"Constant model: y = {self.intercept_:.4f}"

        for j in self.shape_functions_:
            if self.feature_importances_[j] / total_importance < 0.01:
                continue

            slope, offset, r2 = self.linear_approx_[j]
            if r2 > 0.90:
                linear_features[j] = (slope, offset)
            else:
                nonlinear_features[j] = self.shape_functions_[j]

        combined_intercept = self.intercept_ + sum(off for _, off in linear_features.values())

        lines = ["Additive Model (interpretable, per-feature effects):"]

        # Linear features as equation
        terms = []
        for j in sorted(linear_features.keys()):
            slope, _ = linear_features[j]
            name = feature_names[j]
            terms.append(f"{slope:.4f}*{name}")

        eq = " + ".join(terms) + f" + {combined_intercept:.4f}" if terms else f"{combined_intercept:.4f}"
        lines.append(f"  y = {eq}")
        lines.append("")
        lines.append("Feature effects (sorted by importance):")

        # All features sorted by importance
        active_features = list(linear_features.keys()) + list(nonlinear_features.keys())
        active_features.sort(key=lambda j: self.feature_importances_[j], reverse=True)

        for j in active_features:
            name = feature_names[j]
            imp = self.feature_importances_[j] / total_importance
            if j in linear_features:
                slope, _ = linear_features[j]
                direction = "+" if slope > 0 else "-"
                lines.append(f"  {name}: {slope:+.4f} (linear, importance={imp:.1%}) [{direction}]")
            else:
                thresholds, intervals = nonlinear_features[j]
                lines.append(f"  {name}: nonlinear effect (importance={imp:.1%})")
                for i, val in enumerate(intervals):
                    if i == 0:
                        lines.append(f"    {name} <= {thresholds[0]:.4f}: {val:+.4f}")
                    elif i == len(thresholds):
                        lines.append(f"    {name} >  {thresholds[-1]:.4f}: {val:+.4f}")
                    else:
                        lines.append(f"    {thresholds[i-1]:.4f} < {name} <= {thresholds[i]:.4f}: {val:+.4f}")

        lines.append(f"  intercept: {combined_intercept:.4f}")

        inactive = [feature_names[j] for j in range(self.n_features_in_) if j not in set(active_features)]
        if inactive:
            lines.append(f"\n  Features with no effect (excluded): {', '.join(inactive)}")

        return "\n".join(lines)


class HingeEBMRegressor(BaseEstimator, RegressorMixin):
    """Two-stage interpretable regressor with sparse linear display.

    Stage 1: Creates piecewise-linear hinge features at quantile knots,
    then uses LassoCV to select a sparse set. This gives a clean equation.

    Stage 2: Fits an EBM on residuals (hidden from display, only helps
    predictions on real data).

    Display (str(model)) shows the sparse linear equation with actual
    column names and importance rankings.

    Parameters
    ----------
    n_knots : int, default=2
        Number of quantile knots per feature.
    max_input_features : int, default=15
        Maximum features to use (selects by correlation with y).
    ebm_outer_bags : int, default=3
        Outer bags for residual EBM.
    ebm_max_rounds : int, default=1000
        Max boosting rounds for residual EBM.

    Note: Requires `interpret` package (pip install interpret).
    """

    def __init__(self, n_knots=2, max_input_features=15,
                 ebm_outer_bags=3, ebm_max_rounds=1000):
        self.n_knots = n_knots
        self.max_input_features = max_input_features
        self.ebm_outer_bags = ebm_outer_bags
        self.ebm_max_rounds = ebm_max_rounds

    def fit(self, X, y, feature_names=None):
        """Fit the model.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        feature_names : list of str, optional
            Column names. Extracted automatically if X is a DataFrame.
        """
        from interpret.glassbox import ExplainableBoostingRegressor

        X, names = _to_array_and_names(X, feature_names)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        self.feature_names_ = names or [f"x{i}" for i in range(X.shape[1])]
        n_samples, n_orig = X.shape

        # Feature selection if too many
        if n_orig > self.max_input_features:
            corrs = np.array([abs(np.corrcoef(X[:, j], y)[0, 1])
                              if np.std(X[:, j]) > 1e-10 else 0
                              for j in range(n_orig)])
            self.selected_ = np.sort(np.argsort(corrs)[-self.max_input_features:])
        else:
            self.selected_ = np.arange(n_orig)

        X_sel = X[:, self.selected_]
        n_feat = X_sel.shape[1]

        # Build hinge basis
        quantiles = np.linspace(0.25, 0.75, self.n_knots)
        self.hinge_info_ = []
        hinge_cols = [X_sel]
        self.hinge_names_ = [self.feature_names_[j] for j in self.selected_]

        for i in range(n_feat):
            xj = X_sel[:, i]
            if np.std(xj) < 1e-10:
                continue
            knots = np.unique(np.quantile(xj, quantiles))
            for t in knots:
                h_pos = np.maximum(0, xj - t)
                hinge_cols.append(h_pos.reshape(-1, 1))
                self.hinge_info_.append((i, t, 'pos'))
                self.hinge_names_.append(f"max(0,{self.feature_names_[self.selected_[i]]}-{t:.2f})")

                h_neg = np.maximum(0, t - xj)
                hinge_cols.append(h_neg.reshape(-1, 1))
                self.hinge_info_.append((i, t, 'neg'))
                self.hinge_names_.append(f"max(0,{t:.2f}-{self.feature_names_[self.selected_[i]]})")

        X_hinge = np.hstack(hinge_cols)

        # Stage 1: Lasso for sparsity
        self.lasso_ = LassoCV(cv=3, max_iter=5000, random_state=42)
        self.lasso_.fit(X_hinge, y)

        # Stage 2: EBM on residuals
        residuals = y - self.lasso_.predict(X_hinge)
        residual_frac = np.var(residuals) / np.var(y) if np.var(y) > 1e-10 else 0
        if residual_frac > 0.10:
            self.ebm_ = ExplainableBoostingRegressor(
                random_state=42,
                outer_bags=self.ebm_outer_bags,
                max_rounds=self.ebm_max_rounds,
            )
            self.ebm_.fit(X, residuals)
        else:
            self.ebm_ = None

        return self

    def _build_hinge_features(self, X):
        X_sel = X[:, self.selected_]
        cols = [X_sel]
        for feat_idx, knot, direction in self.hinge_info_:
            xj = X_sel[:, feat_idx]
            if direction == 'pos':
                cols.append(np.maximum(0, xj - knot).reshape(-1, 1))
            else:
                cols.append(np.maximum(0, knot - xj).reshape(-1, 1))
        return np.hstack(cols)

    def predict(self, X):
        check_is_fitted(self, "lasso_")
        X, _ = _to_array_and_names(X)
        X_hinge = self._build_hinge_features(X)
        pred = self.lasso_.predict(X_hinge)
        if self.ebm_ is not None:
            pred += self.ebm_.predict(X)
        return pred

    def _compute_effective_coefs(self):
        """Compute effective linear coefficient per original feature."""
        coefs = self.lasso_.coef_
        intercept = self.lasso_.intercept_
        n_sel = len(self.selected_)

        effective_coefs = {}
        effective_intercept = intercept

        for i in range(n_sel):
            j_orig = self.selected_[i]
            effective_coefs[j_orig] = coefs[i]

        for idx, (feat_idx, knot, direction) in enumerate(self.hinge_info_):
            j_orig = self.selected_[feat_idx]
            c = coefs[n_sel + idx]
            if abs(c) < 1e-6:
                continue
            if direction == 'pos':
                effective_coefs[j_orig] = effective_coefs.get(j_orig, 0) + c
                effective_intercept -= c * knot
            else:
                effective_coefs[j_orig] = effective_coefs.get(j_orig, 0) - c
                effective_intercept += c * knot

        active = {j: c for j, c in effective_coefs.items() if abs(c) > 1e-6}
        return active, effective_intercept

    def feature_effects(self):
        """Return a dict summarizing each feature's learned effect.

        Returns
        -------
        dict : {feature_name: {"direction": str, "importance": float, "rank": int}}
        """
        check_is_fitted(self, "lasso_")
        active, _ = self._compute_effective_coefs()

        total_abs = sum(abs(c) for c in active.values())
        if total_abs < 1e-10:
            return {}

        effects = {}
        for j in range(self.n_features_in_):
            name = self.feature_names_[j]
            c = active.get(j, 0.0)
            rel_imp = abs(c) / total_abs if total_abs > 0 else 0

            if rel_imp < 0.01:
                effects[name] = {"direction": "zero", "importance": 0.0, "rank": 0}
            else:
                direction = "positive" if c > 0 else "negative"
                effects[name] = {"direction": direction, "importance": round(rel_imp, 4)}

        ranked = sorted(
            [(n, e) for n, e in effects.items() if e["importance"] > 0],
            key=lambda x: -x[1]["importance"]
        )
        for rank, (name, _) in enumerate(ranked, 1):
            effects[name]["rank"] = rank
        for name, e in effects.items():
            if e.get("rank") is None:
                e["rank"] = 0

        return effects

    def __str__(self):
        check_is_fitted(self, "lasso_")
        feature_names = self.feature_names_
        active, effective_intercept = self._compute_effective_coefs()

        lines = [f"Sparse Linear Model (Lasso, α={self.lasso_.alpha_:.4g}):"]

        terms = []
        for j in sorted(active.keys()):
            terms.append(f"{active[j]:.4f}*{feature_names[j]}")

        eq = " + ".join(terms) + f" + {effective_intercept:.4f}" if terms else f"{effective_intercept:.4f}"
        lines.append(f"  y = {eq}")
        lines.append("")
        lines.append("Feature effects (sorted by importance):")

        sorted_active = sorted(active.items(), key=lambda x: abs(x[1]), reverse=True)
        total_abs = sum(abs(c) for _, c in sorted_active)
        for j, c in sorted_active:
            direction = "+" if c > 0 else "-"
            imp = abs(c) / total_abs if total_abs > 0 else 0
            lines.append(f"  {feature_names[j]}: {c:+.4f} (importance={imp:.1%}) [{direction}]")
        lines.append(f"  intercept: {effective_intercept:.4f}")

        inactive = [feature_names[j] for j in range(self.n_features_in_) if j not in active]
        if inactive:
            lines.append(f"\n  Features with no effect (excluded): {', '.join(inactive)}")

        return "\n".join(lines)
