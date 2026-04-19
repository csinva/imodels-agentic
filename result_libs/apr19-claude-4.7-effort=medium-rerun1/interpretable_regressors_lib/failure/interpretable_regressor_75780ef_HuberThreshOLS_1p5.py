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
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression, ElasticNetCV, HuberRegressor, BayesianRidge, LassoLarsIC, Lars
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class IterativeThresholdOLSRegressor(BaseEstimator, RegressorMixin):
    """Iteratively fit OLS, prune features with |coef*std(x)| below rel_thresh
    of the current max, refit. Converges to a stable subset. Idea: once weak
    features are removed, the remaining coefficients re-estimate more cleanly,
    which can move the threshold boundary and reveal features that were only
    weak due to collinearity with now-removed features."""

    def __init__(self, rel_thresh=0.01, max_rounds=5, min_features=3):
        self.rel_thresh = rel_thresh
        self.max_rounds = max_rounds
        self.min_features = min_features

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.n_features_in_ = d
        active = np.arange(d)
        for _ in range(self.max_rounds):
            ols = LinearRegression().fit(X[:, active], y)
            stds = X[:, active].std(axis=0) + 1e-12
            contrib = np.abs(ols.coef_) * stds
            if contrib.size == 0: break
            cmax = contrib.max()
            if cmax <= 0: break
            keep_local = np.where(contrib >= self.rel_thresh * cmax)[0]
            if keep_local.size < self.min_features:
                keep_local = np.argsort(-contrib)[: min(self.min_features, active.size)]
            if keep_local.size == active.size:
                break  # converged
            active = active[keep_local]
        # final sort by contribution
        ols = LinearRegression().fit(X[:, active], y)
        stds = X[:, active].std(axis=0) + 1e-12
        order = np.argsort(-(np.abs(ols.coef_) * stds))
        self.selected_ = active[order]
        self.ols_ = LinearRegression().fit(X[:, self.selected_], y)
        return self

    def predict(self, X):
        check_is_fitted(self, "ols_")
        return self.ols_.predict(np.asarray(X, dtype=float)[:, self.selected_])

    def __str__(self):
        check_is_fitted(self, "ols_")
        coefs = self.ols_.coef_
        names = [f"x{j}" for j in self.selected_]
        equation = " ".join(f"{c:+.4f}*{n}" for c, n in zip(coefs, names)) + f" {self.ols_.intercept_:+.4f}"
        lines = [
            f"IterativeThresholdOLS ({len(self.selected_)} features kept after iterative "
            f"pruning at {self.rel_thresh:.0%} relative contribution threshold, OLS refit):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]
        for n, c in zip(names, coefs):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {self.ols_.intercept_:.4f}")
        dropped = [f"x{j}" for j in range(self.n_features_in_) if j not in set(self.selected_)]
        if dropped:
            lines.append(f"\nExcluded features (coefficient = 0): {', '.join(dropped)}")
        return "\n".join(lines)


class ThresholdOLSRegressor(BaseEstimator, RegressorMixin):
    """Fit OLS on all features; rank features by standardized contribution
    |coef * std(x)|; drop any with contribution below `rel_thresh` x the max;
    refit (OLS or RidgeCV) on the retained subset."""

    def __init__(self, rel_thresh=0.01, min_features=3, refit="ols"):
        self.rel_thresh = rel_thresh
        self.min_features = min_features
        self.refit = refit

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.n_features_in_ = d
        ols_full = LinearRegression().fit(X, y)
        stds = X.std(axis=0) + 1e-12
        contrib = np.abs(ols_full.coef_) * stds
        cmax = contrib.max() if contrib.size else 0.0
        if cmax <= 0:
            keep = np.arange(min(d, self.min_features))
        else:
            keep = np.where(contrib >= self.rel_thresh * cmax)[0]
            if keep.size < self.min_features:
                keep = np.argsort(-contrib)[: min(self.min_features, d)]
        self.selected_ = keep[np.argsort(-contrib[keep])]
        if self.refit == "ridge":
            self.ols_ = RidgeCV(alphas=(0.001, 0.01, 0.1, 1.0, 10.0)).fit(X[:, self.selected_], y)
        else:
            self.ols_ = LinearRegression().fit(X[:, self.selected_], y)
        return self

    def predict(self, X):
        check_is_fitted(self, "ols_")
        return self.ols_.predict(np.asarray(X, dtype=float)[:, self.selected_])

    def __str__(self):
        check_is_fitted(self, "ols_")
        coefs = self.ols_.coef_
        names = [f"x{j}" for j in self.selected_]
        equation = " ".join(f"{c:+.4f}*{n}" for c, n in zip(coefs, names)) + f" {self.ols_.intercept_:+.4f}"
        lines = [
            f"ThresholdOLS ({len(self.selected_)} features kept: those whose "
            f"standardized contribution |coef*std(x)| is at least {self.rel_thresh:.0%} of the top feature's, refit via OLS):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]
        for n, c in zip(names, coefs):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {self.ols_.intercept_:.4f}")
        dropped = [f"x{j}" for j in range(self.n_features_in_) if j not in set(self.selected_)]
        if dropped:
            lines.append(f"\nExcluded features (coefficient = 0): {', '.join(dropped)}")
        return "\n".join(lines)


class BayesianRidgeRegressor(BaseEstimator, RegressorMixin):
    """Bayesian Ridge on LassoCV-selected features: Lasso picks an adaptively
    sparse set, then BayesianRidge provides principled shrinkage with a
    per-model alpha/lambda learned via evidence maximization."""

    def __init__(self, cv=3, max_iter=5000):
        self.cv = cv
        self.max_iter = max_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.n_features_in_ = d
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.lasso_ = LassoCV(cv=self.cv, max_iter=self.max_iter, n_alphas=50).fit(Xs, y)
        mags = np.abs(self.lasso_.coef_)
        nz = np.where(mags > 1e-10)[0]
        if nz.size == 0:
            corrs = np.abs(Xs.T @ (y - y.mean())) / (n + 1e-9)
            nz = np.argsort(-corrs)[: min(3, d)]
        self.selected_ = nz[np.argsort(-mags[nz])]
        self.model_ = BayesianRidge(max_iter=300, compute_score=False).fit(X[:, self.selected_], y)
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict(np.asarray(X, dtype=float)[:, self.selected_])

    def __str__(self):
        check_is_fitted(self, "model_")
        coefs = self.model_.coef_
        names = [f"x{j}" for j in self.selected_]
        equation = " ".join(f"{c:+.4f}*{n}" for c, n in zip(coefs, names)) + f" {self.model_.intercept_:+.4f}"
        lines = [
            f"Bayesian Ridge Regression ({len(self.selected_)} non-zero features picked by LassoCV, "
            f"coefficients estimated by Bayesian evidence maximization):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]
        for n, c in zip(names, coefs):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {self.model_.intercept_:.4f}")
        dropped = [f"x{j}" for j in range(self.n_features_in_) if j not in set(self.selected_)]
        if dropped:
            lines.append(f"\nExcluded features (coefficient = 0): {', '.join(dropped)}")
        return "\n".join(lines)


class LassoHuberRegressor(BaseEstimator, RegressorMixin):
    """LassoCV picks the non-zero set; Huber refit on that subset for robustness
    to outliers. Falls back to OLS when Huber fails to converge."""

    def __init__(self, cv=3, max_iter=5000, huber_epsilon=1.35):
        self.cv = cv
        self.max_iter = max_iter
        self.huber_epsilon = huber_epsilon

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.n_features_in_ = d
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.lasso_ = LassoCV(cv=self.cv, max_iter=self.max_iter, n_alphas=50).fit(Xs, y)
        mags = np.abs(self.lasso_.coef_)
        nz = np.where(mags > 1e-10)[0]
        if nz.size == 0:
            corrs = np.abs(Xs.T @ (y - y.mean())) / (n + 1e-9)
            nz = np.argsort(-corrs)[: min(3, d)]
        self.selected_ = nz[np.argsort(-mags[nz])] if mags[nz].size else nz
        X_sel = X[:, self.selected_]
        try:
            self.model_ = HuberRegressor(epsilon=self.huber_epsilon,
                                         max_iter=500, alpha=0.0).fit(X_sel, y)
            self._used_ = "Huber"
        except Exception:
            self.model_ = LinearRegression().fit(X_sel, y)
            self._used_ = "OLS"
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict(np.asarray(X, dtype=float)[:, self.selected_])

    def __str__(self):
        check_is_fitted(self, "model_")
        coefs = self.model_.coef_
        names = [f"x{j}" for j in self.selected_]
        equation = " ".join(f"{c:+.4f}*{n}" for c, n in zip(coefs, names)) + f" {self.model_.intercept_:+.4f}"
        lines = [
            f"Lasso→{self._used_} ({len(self.selected_)} non-zero features picked by LassoCV, refit via {self._used_} regression):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]
        for n, c in zip(names, coefs):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {self.model_.intercept_:.4f}")
        dropped = [f"x{j}" for j in range(self.n_features_in_) if j not in set(self.selected_)]
        if dropped:
            lines.append(f"\nExcluded features (coefficient = 0): {', '.join(dropped)}")
        return "\n".join(lines)


class MutualInfoOLSRegressor(BaseEstimator, RegressorMixin):
    """Mutual-info ranks features (captures non-linear dependencies); keep those
    with MI above a fraction of the max; OLS refit for a compact linear formula."""

    def __init__(self, mi_threshold=0.05, min_features=3, max_features=20, random_state=42):
        self.mi_threshold = mi_threshold
        self.min_features = min_features
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.n_features_in_ = d
        mi = mutual_info_regression(X, y, random_state=self.random_state)
        self.mi_ = mi
        mi_max = mi.max() if mi.size else 0.0
        if mi_max <= 0:
            self.selected_ = np.arange(min(d, self.min_features))
        else:
            keep = np.where(mi >= self.mi_threshold * mi_max)[0]
            order = keep[np.argsort(-mi[keep])]
            if order.size < self.min_features:
                order = np.argsort(-mi)[: min(self.min_features, d)]
            self.selected_ = order[: self.max_features]
        self.ols_ = LinearRegression().fit(X[:, self.selected_], y)
        return self

    def predict(self, X):
        check_is_fitted(self, "ols_")
        return self.ols_.predict(np.asarray(X, dtype=float)[:, self.selected_])

    def __str__(self):
        check_is_fitted(self, "ols_")
        coefs = self.ols_.coef_
        names = [f"x{j}" for j in self.selected_]
        equation = " ".join(f"{c:+.4f}*{n}" for c, n in zip(coefs, names)) + f" {self.ols_.intercept_:+.4f}"
        lines = [
            f"MutualInfo→OLS ({len(self.selected_)} features ranked by mutual information with y, refit via OLS):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]
        for n, c in zip(names, coefs):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {self.ols_.intercept_:.4f}")
        dropped = [f"x{j}" for j in range(self.n_features_in_) if j not in set(self.selected_)]
        if dropped:
            lines.append(f"\nExcluded features (coefficient = 0): {', '.join(dropped)}")
        return "\n".join(lines)


class AllRidgePlusTopQuadRegressor(BaseEstimator, RegressorMixin):
    """
    RidgeCV over all features, with an optional single quadratic term for the
    most important feature (identified by Ridge's |coef * std|).

    Goal: match RidgeCV's interpretability-test score (linear formula, easy for
    an LLM to read) while picking up a bit of curvature that pure linear models
    miss, nudging performance rank below RidgeCV's 8.94 baseline.
    """

    def __init__(self, add_quad=True):
        self.add_quad = add_quad

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.n_features_in_ = d

        # Identify top feature via Ridge on standardized X (pure ranking).
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        rank_ridge = RidgeCV(alphas=(0.001, 0.01, 0.1, 1.0, 10.0)).fit(Xs, y)
        mag = np.abs(rank_ridge.coef_)
        self.top_idx_ = int(np.argmax(mag)) if d else 0

        # Fit Ridge on original X; if add_quad, append x_top^2.
        if self.add_quad and d > 0:
            Xq = np.column_stack([X, X[:, self.top_idx_] ** 2])
        else:
            Xq = X
        self.model_ = RidgeCV(alphas=(0.001, 0.01, 0.1, 1.0, 10.0)).fit(Xq, y)
        return self

    def _design(self, X):
        if self.add_quad:
            return np.column_stack([X, X[:, self.top_idx_] ** 2])
        return X

    def predict(self, X):
        check_is_fitted(self, "model_")
        X = np.asarray(X, dtype=float)
        return self.model_.predict(self._design(X))

    def __str__(self):
        check_is_fitted(self, "model_")
        d = self.n_features_in_
        coefs = self.model_.coef_
        intercept = float(self.model_.intercept_)
        lin_coefs = coefs[:d]
        quad_coef = coefs[d] if self.add_quad and d > 0 else 0.0

        # Single-line equation matching the RidgeCV baseline format
        terms = [f"{c:+.4f}*x{i}" for i, c in enumerate(lin_coefs)]
        if self.add_quad:
            terms.append(f"{quad_coef:+.4f}*x{self.top_idx_}^2")
        equation = " ".join(terms) + f" {intercept:+.4f}"

        lines = [
            f"Ridge Regression (L2 regularization, α={self.model_.alpha_:.4g} chosen by CV) "
            f"with one extra quadratic term on x{self.top_idx_} (the most important feature):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]
        for i, c in enumerate(lin_coefs):
            lines.append(f"  x{i}: {c:.4f}")
        if self.add_quad:
            lines.append(f"  x{self.top_idx_}^2: {quad_coef:.4f}  (quadratic correction)")
        lines.append(f"  intercept: {intercept:.4f}")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
AllRidgePlusTopQuadRegressor.__module__ = "interpretable_regressor"
MutualInfoOLSRegressor.__module__ = "interpretable_regressor"
LassoHuberRegressor.__module__ = "interpretable_regressor"
BayesianRidgeRegressor.__module__ = "interpretable_regressor"
ThresholdOLSRegressor.__module__ = "interpretable_regressor"
IterativeThresholdOLSRegressor.__module__ = "interpretable_regressor"

class LassoLarsAICRoundedOLSRegressor(BaseEstimator, RegressorMixin):
    """LassoLarsIC(AIC) picks features, OLS refit, then round coefficients to
    `decimals` places. Rounding keeps prediction arithmetic simpler for the LLM
    interpretability tests and produces a cleaner printed formula."""

    def __init__(self, criterion="aic", decimals=3):
        self.criterion = criterion
        self.decimals = decimals

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.n_features_in_ = d
        Xs = StandardScaler().fit_transform(X)
        lars = LassoLarsIC(criterion=self.criterion, max_iter=2000).fit(Xs, y)
        mags = np.abs(lars.coef_)
        nz = np.where(mags > 1e-10)[0]
        if nz.size == 0:
            corrs = np.abs(Xs.T @ (y - y.mean())) / (n + 1e-9)
            nz = np.argsort(-corrs)[: min(3, d)]
        self.selected_ = nz[np.argsort(-mags[nz])] if mags[nz].size else nz
        ols = LinearRegression().fit(X[:, self.selected_], y)
        self.coef_ = np.round(ols.coef_, self.decimals)
        self.intercept_ = float(np.round(ols.intercept_, self.decimals))
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        X = np.asarray(X, dtype=float)
        return X[:, self.selected_] @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_")
        names = [f"x{j}" for j in self.selected_]
        equation = " ".join(f"{c:+.{self.decimals}f}*{n}" for c, n in zip(self.coef_, names)) + f" {self.intercept_:+.{self.decimals}f}"
        lines = [
            f"LassoLarsAIC→OLS with coefficients rounded to {self.decimals} decimals ({len(self.selected_)} features):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]
        for n, c in zip(names, self.coef_):
            lines.append(f"  {n}: {c:.{self.decimals}f}")
        lines.append(f"  intercept: {self.intercept_:.{self.decimals}f}")
        dropped = [f"x{j}" for j in range(self.n_features_in_) if j not in set(self.selected_)]
        if dropped:
            lines.append(f"\nExcluded features (coefficient = 0): {', '.join(dropped)}")
        return "\n".join(lines)


class QuantileBinOLSRegressor(BaseEstimator, RegressorMixin):
    """Per-feature quantile binning (3 bins) + OLS. Each feature becomes one
    ordinal variable (low/mid/high → -1/0/+1). Yields a compact linear
    formula in terms of quantile-bin indicators."""

    def __init__(self, n_bins=3, min_features=3, rel_thresh=0.01):
        self.n_bins = n_bins
        self.min_features = min_features
        self.rel_thresh = rel_thresh

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.n_features_in_ = d
        self.quantiles_ = []
        X_bin = np.zeros_like(X)
        for j in range(d):
            qs = np.quantile(X[:, j], np.linspace(0, 1, self.n_bins + 1)[1:-1])
            self.quantiles_.append(qs)
            binned = np.searchsorted(qs, X[:, j]).astype(float)
            # center: 0 → -1, 1 → 0, 2 → +1 for n_bins=3
            X_bin[:, j] = binned - (self.n_bins - 1) / 2.0
        ols_full = LinearRegression().fit(X_bin, y)
        stds = X_bin.std(axis=0) + 1e-12
        contrib = np.abs(ols_full.coef_) * stds
        cmax = contrib.max() if contrib.size else 0.0
        if cmax <= 0:
            keep = np.arange(min(d, self.min_features))
        else:
            keep = np.where(contrib >= self.rel_thresh * cmax)[0]
            if keep.size < self.min_features:
                keep = np.argsort(-contrib)[: min(self.min_features, d)]
        self.selected_ = keep[np.argsort(-contrib[keep])]
        self.ols_ = LinearRegression().fit(X_bin[:, self.selected_], y)
        return self

    def _bin(self, X):
        X = np.asarray(X, dtype=float)
        X_bin = np.zeros_like(X)
        for j in range(self.n_features_in_):
            qs = self.quantiles_[j]
            binned = np.searchsorted(qs, X[:, j]).astype(float)
            X_bin[:, j] = binned - (self.n_bins - 1) / 2.0
        return X_bin

    def predict(self, X):
        check_is_fitted(self, "ols_")
        return self.ols_.predict(self._bin(X)[:, self.selected_])

    def __str__(self):
        check_is_fitted(self, "ols_")
        coefs = self.ols_.coef_
        names = [f"bin(x{j})" for j in self.selected_]
        equation = " ".join(f"{c:+.4f}*{n}" for c, n in zip(coefs, names)) + f" {self.ols_.intercept_:+.4f}"
        lines = [
            f"QuantileBin→OLS ({self.n_bins} bins per feature; bin(x) returns -1/0/+1 based on which tercile; OLS on top features):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]
        for n, c in zip(names, coefs):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {self.ols_.intercept_:.4f}")
        return "\n".join(lines)


class LassoLarsAICThreshOLSRegressor(BaseEstimator, RegressorMixin):
    """LassoLarsIC(AIC) → on the resulting non-zero set, refit OLS, then prune
    any feature whose standardized contribution drops below rel_thresh x top,
    and OLS-refit again. Aims to keep AIC's strong rank while getting ThresholdOLS's
    brevity-driven interp boost."""

    def __init__(self, criterion="aic", rel_thresh=0.01, min_features=3):
        self.criterion = criterion
        self.rel_thresh = rel_thresh
        self.min_features = min_features

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.n_features_in_ = d
        Xs = StandardScaler().fit_transform(X)
        lars = LassoLarsIC(criterion=self.criterion, max_iter=2000).fit(Xs, y)
        mags = np.abs(lars.coef_)
        nz = np.where(mags > 1e-10)[0]
        if nz.size == 0:
            corrs = np.abs(Xs.T @ (y - y.mean())) / (n + 1e-9)
            nz = np.argsort(-corrs)[: min(3, d)]
        # OLS on nz, then prune by relative contribution
        ols1 = LinearRegression().fit(X[:, nz], y)
        stds = X[:, nz].std(axis=0) + 1e-12
        contrib = np.abs(ols1.coef_) * stds
        cmax = contrib.max() if contrib.size else 0.0
        if cmax <= 0:
            keep_local = np.arange(min(nz.size, self.min_features))
        else:
            keep_local = np.where(contrib >= self.rel_thresh * cmax)[0]
            if keep_local.size < self.min_features:
                keep_local = np.argsort(-contrib)[: min(self.min_features, nz.size)]
        order = keep_local[np.argsort(-contrib[keep_local])]
        self.selected_ = nz[order]
        self.ols_ = LinearRegression().fit(X[:, self.selected_], y)
        return self

    def predict(self, X):
        check_is_fitted(self, "ols_")
        return self.ols_.predict(np.asarray(X, dtype=float)[:, self.selected_])

    def __str__(self):
        check_is_fitted(self, "ols_")
        coefs = self.ols_.coef_
        names = [f"x{j}" for j in self.selected_]
        equation = " ".join(f"{c:+.4f}*{n}" for c, n in zip(coefs, names)) + f" {self.ols_.intercept_:+.4f}"
        lines = [
            f"LassoLarsAIC+Threshold→OLS ({len(self.selected_)} features: AIC-selected and then pruned at {self.rel_thresh:.0%} relative contribution, OLS refit):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]
        for n, c in zip(names, coefs):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {self.ols_.intercept_:.4f}")
        dropped = [f"x{j}" for j in range(self.n_features_in_) if j not in set(self.selected_)]
        if dropped:
            lines.append(f"\nExcluded features (coefficient = 0): {', '.join(dropped)}")
        return "\n".join(lines)


class LassoLarsICOLSRegressor(BaseEstimator, RegressorMixin):
    """LassoLarsIC (IC-selected alpha, no CV needed) picks the non-zero set;
    OLS or Ridge refit on selected features. `max_features` caps the subset size."""

    def __init__(self, criterion="bic", max_features=10**6, refit="ols"):
        self.criterion = criterion
        self.max_features = max_features
        self.refit = refit

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.n_features_in_ = d
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.lars_ = LassoLarsIC(criterion=self.criterion, max_iter=2000).fit(Xs, y)
        mags = np.abs(self.lars_.coef_)
        nz = np.where(mags > 1e-10)[0]
        if nz.size == 0:
            corrs = np.abs(Xs.T @ (y - y.mean())) / (n + 1e-9)
            nz = np.argsort(-corrs)[: min(3, d)]
        self.selected_ = (nz[np.argsort(-mags[nz])] if mags[nz].size else nz)[: self.max_features]
        if self.refit == "ridge":
            self.ols_ = RidgeCV(alphas=(0.001, 0.01, 0.1, 1.0, 10.0)).fit(X[:, self.selected_], y)
        else:
            self.ols_ = LinearRegression().fit(X[:, self.selected_], y)
        return self

    def predict(self, X):
        check_is_fitted(self, "ols_")
        return self.ols_.predict(np.asarray(X, dtype=float)[:, self.selected_])

    def __str__(self):
        check_is_fitted(self, "ols_")
        coefs = self.ols_.coef_
        names = [f"x{j}" for j in self.selected_]
        equation = " ".join(f"{c:+.4f}*{n}" for c, n in zip(coefs, names)) + f" {self.ols_.intercept_:+.4f}"
        lines = [
            f"LassoLarsIC({self.criterion.upper()})→OLS ({len(self.selected_)} features selected by {self.criterion.upper()} criterion, OLS refit):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]
        for n, c in zip(names, coefs):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {self.ols_.intercept_:.4f}")
        dropped = [f"x{j}" for j in range(self.n_features_in_) if j not in set(self.selected_)]
        if dropped:
            lines.append(f"\nExcluded features (coefficient = 0): {', '.join(dropped)}")
        return "\n".join(lines)


class ForwardStepwiseOLSRegressor(BaseEstimator, RegressorMixin):
    """Forward stepwise OLS. At each step add the feature with the largest
    marginal R^2 gain. Stop when gain < min_gain or max_features reached."""

    def __init__(self, max_features=15, min_gain=0.001):
        self.max_features = max_features
        self.min_gain = min_gain

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.n_features_in_ = d
        remaining = list(range(d))
        selected = []
        base_ss = ((y - y.mean()) ** 2).sum() + 1e-12
        prev_rss = base_ss
        while remaining and len(selected) < self.max_features:
            best_j, best_rss = None, prev_rss
            for j in remaining:
                cols = selected + [j]
                ols = LinearRegression().fit(X[:, cols], y)
                rss = ((y - ols.predict(X[:, cols])) ** 2).sum()
                if rss < best_rss:
                    best_rss = rss; best_j = j
            if best_j is None: break
            gain = (prev_rss - best_rss) / base_ss
            if gain < self.min_gain: break
            selected.append(best_j); remaining.remove(best_j); prev_rss = best_rss
        if not selected:
            selected = [int(np.argmax(np.abs(X.T @ (y - y.mean()))))]
        self.selected_ = np.array(selected)
        self.ols_ = LinearRegression().fit(X[:, self.selected_], y)
        return self

    def predict(self, X):
        check_is_fitted(self, "ols_")
        return self.ols_.predict(np.asarray(X, dtype=float)[:, self.selected_])

    def __str__(self):
        check_is_fitted(self, "ols_")
        coefs = self.ols_.coef_
        names = [f"x{j}" for j in self.selected_]
        equation = " ".join(f"{c:+.4f}*{n}" for c, n in zip(coefs, names)) + f" {self.ols_.intercept_:+.4f}"
        lines = [
            f"ForwardStepwise→OLS ({len(self.selected_)} features picked greedily by R^2 gain; refit via OLS):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]
        for n, c in zip(names, coefs):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {self.ols_.intercept_:.4f}")
        dropped = [f"x{j}" for j in range(self.n_features_in_) if j not in set(self.selected_)]
        if dropped:
            lines.append(f"\nExcluded features (coefficient = 0): {', '.join(dropped)}")
        return "\n".join(lines)


ForwardStepwiseOLSRegressor.__module__ = "interpretable_regressor"
LassoLarsICOLSRegressor.__module__ = "interpretable_regressor"
LassoLarsAICThreshOLSRegressor.__module__ = "interpretable_regressor"
QuantileBinOLSRegressor.__module__ = "interpretable_regressor"
LassoLarsAICRoundedOLSRegressor.__module__ = "interpretable_regressor"
class LarsOLSRegressor(BaseEstimator, RegressorMixin):
    """Least Angle Regression (LARS) selects n_nonzero_coefs features in order
    of correlation with residuals; OLS refit on the selected set."""

    def __init__(self, n_nonzero=10):
        self.n_nonzero = n_nonzero

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.n_features_in_ = d
        Xs = StandardScaler().fit_transform(X)
        lars = Lars(n_nonzero_coefs=min(self.n_nonzero, d), fit_path=False).fit(Xs, y)
        mags = np.abs(lars.coef_)
        nz = np.where(mags > 1e-10)[0]
        if nz.size == 0:
            nz = np.arange(min(3, d))
        self.selected_ = nz[np.argsort(-mags[nz])]
        self.ols_ = LinearRegression().fit(X[:, self.selected_], y)
        return self

    def predict(self, X):
        check_is_fitted(self, "ols_")
        return self.ols_.predict(np.asarray(X, dtype=float)[:, self.selected_])

    def __str__(self):
        check_is_fitted(self, "ols_")
        coefs = self.ols_.coef_
        names = [f"x{j}" for j in self.selected_]
        equation = " ".join(f"{c:+.4f}*{n}" for c, n in zip(coefs, names)) + f" {self.ols_.intercept_:+.4f}"
        lines = [
            f"LARS→OLS ({len(self.selected_)} features picked via Least Angle Regression correlation path, OLS refit):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]
        for n, c in zip(names, coefs):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {self.ols_.intercept_:.4f}")
        dropped = [f"x{j}" for j in range(self.n_features_in_) if j not in set(self.selected_)]
        if dropped:
            lines.append(f"\nExcluded features (coefficient = 0): {', '.join(dropped)}")
        return "\n".join(lines)


LarsOLSRegressor.__module__ = "interpretable_regressor"


class SignificanceOLSRegressor(BaseEstimator, RegressorMixin):
    """Fit OLS on all features, compute per-coefficient t-statistics (coef/stderr),
    drop features with |t| below t_thresh, OLS refit. Uses statistical significance
    rather than standardized magnitude to decide which features matter."""

    def __init__(self, t_thresh=2.0, min_features=3):
        self.t_thresh = t_thresh
        self.min_features = min_features

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.n_features_in_ = d
        ols = LinearRegression().fit(X, y)
        resid = y - ols.predict(X)
        dof = max(n - d - 1, 1)
        sigma2 = (resid ** 2).sum() / dof
        Xc = np.column_stack([np.ones(n), X])
        try:
            cov = np.linalg.pinv(Xc.T @ Xc) * sigma2
            se = np.sqrt(np.clip(np.diag(cov)[1:], 1e-18, None))
        except Exception:
            se = np.ones(d)
        t = np.abs(ols.coef_) / (se + 1e-18)
        self.tstats_ = t
        keep = np.where(t >= self.t_thresh)[0]
        if keep.size < self.min_features:
            keep = np.argsort(-t)[: min(self.min_features, d)]
        self.selected_ = keep[np.argsort(-t[keep])]
        self.ols_ = LinearRegression().fit(X[:, self.selected_], y)
        return self

    def predict(self, X):
        check_is_fitted(self, "ols_")
        return self.ols_.predict(np.asarray(X, dtype=float)[:, self.selected_])

    def __str__(self):
        check_is_fitted(self, "ols_")
        coefs = self.ols_.coef_
        names = [f"x{j}" for j in self.selected_]
        equation = " ".join(f"{c:+.4f}*{n}" for c, n in zip(coefs, names)) + f" {self.ols_.intercept_:+.4f}"
        lines = [
            f"SignificanceOLS ({len(self.selected_)} features kept by t-statistic >= {self.t_thresh}, OLS refit):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]
        for n, c in zip(names, coefs):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {self.ols_.intercept_:.4f}")
        dropped = [f"x{j}" for j in range(self.n_features_in_) if j not in set(self.selected_)]
        if dropped:
            lines.append(f"\nExcluded features (coefficient = 0): {', '.join(dropped)}")
        return "\n".join(lines)


SignificanceOLSRegressor.__module__ = "interpretable_regressor"


class AdaptiveLassoOLSRegressor(BaseEstimator, RegressorMixin):
    """Two-stage adaptive Lasso: first RidgeCV provides initial weights
    w_j = 1/|beta_j^ridge|; rescale features by 1/w_j so Lasso penalizes
    weak-coefficient features more; second-stage LassoCV gives oracle
    property of consistent variable selection; OLS refit on selected set."""

    def __init__(self, cv=3, max_iter=5000, eps=1e-6):
        self.cv = cv
        self.max_iter = max_iter
        self.eps = eps

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.n_features_in_ = d
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        ridge = RidgeCV(alphas=(0.001, 0.01, 0.1, 1.0, 10.0)).fit(Xs, y)
        w = 1.0 / (np.abs(ridge.coef_) + self.eps)
        Xw = Xs / w[np.newaxis, :]
        lasso = LassoCV(cv=self.cv, max_iter=self.max_iter, n_alphas=50).fit(Xw, y)
        # map back: effective coef on Xs = lasso.coef_ / w
        adaptive_coef = lasso.coef_ / w
        nz = np.where(np.abs(adaptive_coef) > 1e-10)[0]
        if nz.size == 0:
            corrs = np.abs(Xs.T @ (y - y.mean())) / (n + 1e-9)
            nz = np.argsort(-corrs)[: min(3, d)]
        self.selected_ = nz[np.argsort(-np.abs(adaptive_coef[nz]))]
        self.ols_ = LinearRegression().fit(X[:, self.selected_], y)
        return self

    def predict(self, X):
        check_is_fitted(self, "ols_")
        return self.ols_.predict(np.asarray(X, dtype=float)[:, self.selected_])

    def __str__(self):
        check_is_fitted(self, "ols_")
        coefs = self.ols_.coef_
        names = [f"x{j}" for j in self.selected_]
        equation = " ".join(f"{c:+.4f}*{n}" for c, n in zip(coefs, names)) + f" {self.ols_.intercept_:+.4f}"
        lines = [
            f"AdaptiveLasso→OLS ({len(self.selected_)} features: Ridge-weighted adaptive Lasso selection, OLS refit):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]
        for n, c in zip(names, coefs):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {self.ols_.intercept_:.4f}")
        dropped = [f"x{j}" for j in range(self.n_features_in_) if j not in set(self.selected_)]
        if dropped:
            lines.append(f"\nExcluded features (coefficient = 0): {', '.join(dropped)}")
        return "\n".join(lines)


AdaptiveLassoOLSRegressor.__module__ = "interpretable_regressor"


class BaggedOLSRegressor(BaseEstimator, RegressorMixin):
    """Average OLS coefficients from bootstrap resamples (bagging) for lower-variance
    linear model. Final equation is the mean coefficient vector, refit intercept."""

    def __init__(self, n_bootstraps=30, random_state=42):
        self.n_bootstraps = n_bootstraps
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.n_features_in_ = d
        rng = np.random.default_rng(self.random_state)
        coefs = np.zeros(d)
        intercepts = 0.0
        k = 0
        for _ in range(self.n_bootstraps):
            idx = rng.integers(0, n, size=n)
            try:
                ols = LinearRegression().fit(X[idx], y[idx])
                coefs += ols.coef_; intercepts += float(ols.intercept_); k += 1
            except Exception:
                continue
        if k == 0:
            ols = LinearRegression().fit(X, y)
            self.coef_ = ols.coef_; self.intercept_ = float(ols.intercept_)
        else:
            self.coef_ = coefs / k
            # refit intercept to training mean residual (improves calibration)
            pred0 = X @ self.coef_
            self.intercept_ = float(y.mean() - pred0.mean())
        self.selected_ = np.arange(d)
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coef_")
        names = [f"x{j}" for j in range(self.n_features_in_)]
        equation = " ".join(f"{c:+.4f}*{n}" for c, n in zip(self.coef_, names)) + f" {self.intercept_:+.4f}"
        lines = [
            f"BaggedOLS (average of {self.n_bootstraps} bootstrap OLS fits on all features; intercept recalibrated on full data):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]
        for n, c in zip(names, self.coef_):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {self.intercept_:.4f}")
        return "\n".join(lines)


BaggedOLSRegressor.__module__ = "interpretable_regressor"


class OMPOLSRegressor(BaseEstimator, RegressorMixin):
    """Orthogonal Matching Pursuit selects `n_nonzero` features greedily by
    maximum correlation with residual; OLS refit on the chosen subset.
    Fast, deterministic, closely related to forward-stepwise but cheaper."""

    def __init__(self, n_nonzero=10):
        self.n_nonzero = n_nonzero

    def fit(self, X, y):
        from sklearn.linear_model import OrthogonalMatchingPursuit
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.n_features_in_ = d
        Xs = StandardScaler().fit_transform(X)
        k = min(self.n_nonzero, d)
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k).fit(Xs, y)
        mags = np.abs(omp.coef_)
        nz = np.where(mags > 1e-10)[0]
        if nz.size == 0:
            nz = np.argsort(-mags)[: min(3, d)]
        self.selected_ = nz[np.argsort(-mags[nz])]
        self.ols_ = LinearRegression().fit(X[:, self.selected_], y)
        return self

    def predict(self, X):
        check_is_fitted(self, "ols_")
        return self.ols_.predict(np.asarray(X, dtype=float)[:, self.selected_])

    def __str__(self):
        check_is_fitted(self, "ols_")
        coefs = self.ols_.coef_
        names = [f"x{j}" for j in self.selected_]
        equation = " ".join(f"{c:+.4f}*{n}" for c, n in zip(coefs, names)) + f" {self.ols_.intercept_:+.4f}"
        lines = [
            f"OMP→OLS ({len(self.selected_)} features picked via Orthogonal Matching Pursuit, OLS refit):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]
        for n, c in zip(names, coefs):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {self.ols_.intercept_:.4f}")
        dropped = [f"x{j}" for j in range(self.n_features_in_) if j not in set(self.selected_)]
        if dropped:
            lines.append(f"\nExcluded features (coefficient = 0): {', '.join(dropped)}")
        return "\n".join(lines)


OMPOLSRegressor.__module__ = "interpretable_regressor"
class PearsonCorrOLSRegressor(BaseEstimator, RegressorMixin):
    """Rank features by |Pearson correlation(x_j, y)|; keep top_k; OLS refit.
    Correlation-based selection is pre-model and robust to feature scaling."""

    def __init__(self, top_k=10):
        self.top_k = top_k

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.n_features_in_ = d
        yc = y - y.mean()
        Xc = X - X.mean(axis=0)
        sx = X.std(axis=0) + 1e-12
        sy = y.std() + 1e-12
        corrs = np.abs((Xc * yc[:, None]).sum(axis=0) / (n * sx * sy))
        k = min(self.top_k, d)
        sel = np.argsort(-corrs)[:k]
        self.corrs_ = corrs
        self.selected_ = sel
        self.ols_ = LinearRegression().fit(X[:, self.selected_], y)
        return self

    def predict(self, X):
        check_is_fitted(self, "ols_")
        return self.ols_.predict(np.asarray(X, dtype=float)[:, self.selected_])

    def __str__(self):
        check_is_fitted(self, "ols_")
        coefs = self.ols_.coef_
        names = [f"x{j}" for j in self.selected_]
        equation = " ".join(f"{c:+.4f}*{n}" for c, n in zip(coefs, names)) + f" {self.ols_.intercept_:+.4f}"
        lines = [
            f"PearsonCorr→OLS (top-{self.top_k} features by |Pearson corr with y|, OLS refit):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]
        for n, c in zip(names, coefs):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {self.ols_.intercept_:.4f}")
        dropped = [f"x{j}" for j in range(self.n_features_in_) if j not in set(self.selected_)]
        if dropped:
            lines.append(f"\nExcluded features (coefficient = 0): {', '.join(dropped)}")
        return "\n".join(lines)


PearsonCorrOLSRegressor.__module__ = "interpretable_regressor"
class WinsorizedOLSRegressor(BaseEstimator, RegressorMixin):
    """Clip X to [p, 100-p] percentile per feature to limit leverage from outliers;
    fit OLS on clipped X; threshold-prune to retain interpretability.
    Uses LinearRegression on the clipped training data; at predict time applies
    the same clipping."""

    def __init__(self, p=1.0, rel_thresh=0.015):
        self.p = p
        self.rel_thresh = rel_thresh

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.n_features_in_ = d
        self.lo_ = np.percentile(X, self.p, axis=0)
        self.hi_ = np.percentile(X, 100 - self.p, axis=0)
        Xc = np.clip(X, self.lo_, self.hi_)
        ols_full = LinearRegression().fit(Xc, y)
        stds = Xc.std(axis=0) + 1e-12
        contrib = np.abs(ols_full.coef_) * stds
        cmax = contrib.max() if contrib.size else 0.0
        if cmax <= 0:
            keep = np.arange(min(d, 3))
        else:
            keep = np.where(contrib >= self.rel_thresh * cmax)[0]
            if keep.size < 3:
                keep = np.argsort(-contrib)[: min(3, d)]
        self.selected_ = keep[np.argsort(-contrib[keep])]
        self.ols_ = LinearRegression().fit(Xc[:, self.selected_], y)
        return self

    def predict(self, X):
        check_is_fitted(self, "ols_")
        X = np.asarray(X, dtype=float)
        Xc = np.clip(X, self.lo_, self.hi_)
        return self.ols_.predict(Xc[:, self.selected_])

    def __str__(self):
        check_is_fitted(self, "ols_")
        coefs = self.ols_.coef_
        names = [f"x{j}" for j in self.selected_]
        equation = " ".join(f"{c:+.4f}*{n}" for c, n in zip(coefs, names)) + f" {self.ols_.intercept_:+.4f}"
        lines = [
            f"Winsorized→ThresholdOLS ({len(self.selected_)} features kept at {self.rel_thresh:.1%} contribution threshold; X clipped to [{self.p}%, {100-self.p}%] percentiles):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]
        for n, c in zip(names, coefs):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {self.ols_.intercept_:.4f}")
        dropped = [f"x{j}" for j in range(self.n_features_in_) if j not in set(self.selected_)]
        if dropped:
            lines.append(f"\nExcluded features (coefficient = 0): {', '.join(dropped)}")
        return "\n".join(lines)


WinsorizedOLSRegressor.__module__ = "interpretable_regressor"
class StabilitySelOLSRegressor(BaseEstimator, RegressorMixin):
    """Stability selection: LassoCV on B bootstrap subsamples; count per-feature
    selection frequency; keep features with frequency >= tau; OLS refit. More
    robust feature set than a single Lasso run."""

    def __init__(self, n_bootstraps=20, tau=0.6, cv=3, random_state=42):
        self.n_bootstraps = n_bootstraps
        self.tau = tau
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.n_features_in_ = d
        Xs = StandardScaler().fit_transform(X)
        rng = np.random.default_rng(self.random_state)
        counts = np.zeros(d)
        k = 0
        for _ in range(self.n_bootstraps):
            idx = rng.choice(n, size=max(n // 2, 10), replace=False)
            try:
                l = LassoCV(cv=min(self.cv, 3), max_iter=2000, n_alphas=20).fit(Xs[idx], y[idx])
                counts += (np.abs(l.coef_) > 1e-10).astype(float); k += 1
            except Exception:
                continue
        if k == 0:
            l = LassoCV(cv=self.cv, max_iter=2000).fit(Xs, y)
            nz = np.where(np.abs(l.coef_) > 1e-10)[0]
        else:
            freq = counts / k
            self.freq_ = freq
            nz = np.where(freq >= self.tau)[0]
            if nz.size == 0:
                nz = np.argsort(-freq)[: min(3, d)]
        self.selected_ = nz
        self.ols_ = LinearRegression().fit(X[:, self.selected_], y)
        return self

    def predict(self, X):
        check_is_fitted(self, "ols_")
        return self.ols_.predict(np.asarray(X, dtype=float)[:, self.selected_])

    def __str__(self):
        check_is_fitted(self, "ols_")
        coefs = self.ols_.coef_
        names = [f"x{j}" for j in self.selected_]
        equation = " ".join(f"{c:+.4f}*{n}" for c, n in zip(coefs, names)) + f" {self.ols_.intercept_:+.4f}"
        lines = [
            f"Stability-Selection→OLS ({len(self.selected_)} features selected in >={self.tau:.0%} of {self.n_bootstraps} Lasso bootstrap fits, OLS refit):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]
        for n, c in zip(names, coefs):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {self.ols_.intercept_:.4f}")
        dropped = [f"x{j}" for j in range(self.n_features_in_) if j not in set(self.selected_)]
        if dropped:
            lines.append(f"\nExcluded features (coefficient = 0): {', '.join(dropped)}")
        return "\n".join(lines)


StabilitySelOLSRegressor.__module__ = "interpretable_regressor"
class HuberThresholdOLSRegressor(BaseEstimator, RegressorMixin):
    """Huber regression on all features (robust to y-outliers) to rank by
    standardized contribution |coef*std|; drop below rel_thresh*top; OLS refit
    on retained subset for interpretable equation."""

    def __init__(self, rel_thresh=0.015, min_features=3, epsilon=1.35):
        self.rel_thresh = rel_thresh
        self.min_features = min_features
        self.epsilon = epsilon

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.n_features_in_ = d
        try:
            huber = HuberRegressor(epsilon=self.epsilon, max_iter=300, alpha=0.0).fit(X, y)
            coef = huber.coef_
        except Exception:
            coef = LinearRegression().fit(X, y).coef_
        stds = X.std(axis=0) + 1e-12
        contrib = np.abs(coef) * stds
        cmax = contrib.max() if contrib.size else 0.0
        if cmax <= 0:
            keep = np.arange(min(d, self.min_features))
        else:
            keep = np.where(contrib >= self.rel_thresh * cmax)[0]
            if keep.size < self.min_features:
                keep = np.argsort(-contrib)[: min(self.min_features, d)]
        self.selected_ = keep[np.argsort(-contrib[keep])]
        self.ols_ = LinearRegression().fit(X[:, self.selected_], y)
        return self

    def predict(self, X):
        check_is_fitted(self, "ols_")
        return self.ols_.predict(np.asarray(X, dtype=float)[:, self.selected_])

    def __str__(self):
        check_is_fitted(self, "ols_")
        coefs = self.ols_.coef_
        names = [f"x{j}" for j in self.selected_]
        equation = " ".join(f"{c:+.4f}*{n}" for c, n in zip(coefs, names)) + f" {self.ols_.intercept_:+.4f}"
        lines = [
            f"Huber-ranked ThresholdOLS ({len(self.selected_)} features kept: Huber regression ranks by |coef*std|, retained at {self.rel_thresh:.1%} of top; OLS refit):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]
        for n, c in zip(names, coefs):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {self.ols_.intercept_:.4f}")
        dropped = [f"x{j}" for j in range(self.n_features_in_) if j not in set(self.selected_)]
        if dropped:
            lines.append(f"\nExcluded features (coefficient = 0): {', '.join(dropped)}")
        return "\n".join(lines)


HuberThresholdOLSRegressor.__module__ = "interpretable_regressor"
model_shorthand_name = "HuberThreshOLS_1p5"
model_description = "Huber regression (robust to y-outliers) ranks features by |coef*std|, keep those >=1.5% of top, OLS refit for final equation"
model_defs = [(model_shorthand_name, HuberThresholdOLSRegressor(rel_thresh=0.015))]


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
