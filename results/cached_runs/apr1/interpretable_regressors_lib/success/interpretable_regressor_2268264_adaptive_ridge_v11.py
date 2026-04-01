"""
Interpretable regressor autoresearch script.
Defines a scikit-learn compatible interpretable regressor and evaluates it
on interpretability tests and TabArena regression datasets (same suite used
for baselines in run_baselines.py).

Usage: uv run model.py
"""

import csv
import os
import subprocess
import sys
import time

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RidgeCV
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class RidgePlusStumpsRegressor(BaseEstimator, RegressorMixin):
    """
    Ridge regression augmented with a few gradient-boosted stump corrections.

    Stage 1: RidgeCV captures linear effects (displayed as clean equation)
    Stage 2: A small number of depth-1 stumps on the residual capture
             the most important nonlinear patterns

    Prediction: y = ridge_prediction + sum(stump_corrections)

    Display: the linear equation followed by a few simple if/else rules.
    The LLM evaluates the equation, then applies each stump correction.
    """

    def __init__(self, n_stumps=3, stump_lr=1.0):
        self.n_stumps = n_stumps
        self.stump_lr = stump_lr

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        # Stage 1: Ridge
        self.ridge_ = RidgeCV(cv=3)
        self.ridge_.fit(X, y)

        # Stage 2: Greedy stumps on residual with optional learning rate
        residual = y - self.ridge_.predict(X)
        self.stumps_ = []
        for k in range(self.n_stumps):
            stump = DecisionTreeRegressor(
                max_depth=1,
                min_samples_leaf=max(5, len(y) // 20),
                random_state=42 + k,
            )
            stump.fit(X, residual)
            self.stumps_.append(stump)
            residual = residual - self.stump_lr * stump.predict(X)

        return self

    def predict(self, X):
        check_is_fitted(self, "ridge_")
        X = np.asarray(X, dtype=np.float64)
        pred = self.ridge_.predict(X)
        for stump in self.stumps_:
            pred = pred + self.stump_lr * stump.predict(X)
        return pred

    def __str__(self):
        check_is_fitted(self, "ridge_")
        names = [f"x{i}" for i in range(self.n_features_in_)]
        coefs = self.ridge_.coef_
        intercept = self.ridge_.intercept_

        # Ridge equation
        equation = " + ".join(f"{c:.4f}*{n}" for c, n in zip(coefs, names))
        equation += f" + {intercept:.4f}"

        lines = [
            f"Ridge Regression with Stump Corrections (α={self.ridge_.alpha_:.4g}):",
            f"  y = ({equation}) + stump_corrections",
            "",
            "Step 1 — Linear equation:",
            f"  y_linear = {equation}",
            "",
            "Coefficients:",
        ]
        for n, c in zip(names, coefs):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {intercept:.4f}")

        # Stump corrections
        lines.append("")
        lines.append(f"Step 2 — Add each stump correction to y_linear:")
        for i, stump in enumerate(self.stumps_):
            tree = stump.tree_
            if tree.node_count <= 1:
                lines.append(f"  Stump {i+1}: constant correction {float(tree.value[0].flatten()[0]):.4f}")
                continue
            feat = int(tree.feature[0])
            thresh = float(tree.threshold[0])
            left_val = float(tree.value[tree.children_left[0]].flatten()[0]) * self.stump_lr
            right_val = float(tree.value[tree.children_right[0]].flatten()[0]) * self.stump_lr
            lines.append(
                f"  Stump {i+1}: if {names[feat]} <= {thresh:.2f} then add {left_val:+.4f}, "
                f"else add {right_val:+.4f}"
            )

        return "\n".join(lines)


class RidgePlusTreeRegressor(BaseEstimator, RegressorMixin):
    """
    Ridge regression + shallow decision tree on residuals.

    Stage 1: RidgeCV captures linear effects
    Stage 2: A depth-2 decision tree (max 4 leaves) on the residual
             captures nonlinear interactions

    Display: linear equation + small tree text.
    """

    def __init__(self, tree_max_depth=2, tree_max_leaf_nodes=4):
        self.tree_max_depth = tree_max_depth
        self.tree_max_leaf_nodes = tree_max_leaf_nodes

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        self.ridge_ = RidgeCV(cv=3)
        self.ridge_.fit(X, y)

        residual = y - self.ridge_.predict(X)
        self.tree_ = DecisionTreeRegressor(
            max_depth=self.tree_max_depth,
            max_leaf_nodes=self.tree_max_leaf_nodes,
            min_samples_leaf=max(5, len(y) // 20),
            random_state=42,
        )
        self.tree_.fit(X, residual)
        return self

    def predict(self, X):
        check_is_fitted(self, "ridge_")
        X = np.asarray(X, dtype=np.float64)
        return self.ridge_.predict(X) + self.tree_.predict(X)

    def __str__(self):
        check_is_fitted(self, "ridge_")
        names = [f"x{i}" for i in range(self.n_features_in_)]
        coefs = self.ridge_.coef_
        intercept = self.ridge_.intercept_

        equation = " + ".join(f"{c:.4f}*{n}" for c, n in zip(coefs, names))
        equation += f" + {intercept:.4f}"

        from sklearn.tree import export_text
        tree_text = export_text(self.tree_, feature_names=names, max_depth=4)

        lines = [
            f"Ridge Regression with Tree Correction (α={self.ridge_.alpha_:.4g}):",
            f"  y = ({equation}) + tree_correction",
            "",
            "Step 1 — Compute the linear part:",
            f"  y_linear = {equation}",
            "",
            "Coefficients:",
        ]
        for n, c in zip(names, coefs):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {intercept:.4f}")
        lines.append("")
        lines.append("Step 2 — Follow the tree below to find the correction, then add it to y_linear:")
        lines.append(tree_text)

        return "\n".join(lines)


class ThresholdRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Ridge regression with automatic threshold indicator features.

    1. Fit initial Ridge
    2. For each feature, find the best split threshold on residuals
    3. Add top-k binary indicators I(xi > threshold_i) as extra features
    4. Refit Ridge on [original features + indicators]
    5. Display as a single clean equation with indicator terms

    For high-dim datasets (>15 features): selects top features by
    univariate correlation first, keeping the equation short.
    """

    def __init__(self, max_indicators=2, max_linear_features=15):
        self.max_indicators = max_indicators
        self.max_linear_features = max_linear_features

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        n = X.shape[1]

        # Feature selection for high-dim data
        if n > self.max_linear_features:
            correlations = np.array([
                abs(np.corrcoef(X[:, j], y)[0, 1]) if np.std(X[:, j]) > 1e-10 else 0
                for j in range(n)
            ])
            self.selected_ = np.sort(np.argsort(correlations)[::-1][:self.max_linear_features])
        else:
            self.selected_ = np.arange(n)
        X_sel = X[:, self.selected_]

        # Initial Ridge
        ridge_init = RidgeCV(cv=3)
        ridge_init.fit(X_sel, y)
        residual = y - ridge_init.predict(X_sel)

        # Find best threshold per selected feature
        candidates = []
        for idx, j in enumerate(self.selected_):
            col = X[:, j]
            best_score = -np.inf
            best_thresh = None
            for q in np.linspace(10, 90, 17):
                thresh = np.percentile(col, q)
                left = residual[col <= thresh]
                right = residual[col > thresh]
                if len(left) < 5 or len(right) < 5:
                    continue
                score = np.var(residual) - (len(left) * np.var(left) + len(right) * np.var(right)) / len(residual)
                if score > best_score:
                    best_score = score
                    best_thresh = thresh
            if best_thresh is not None:
                candidates.append((best_score, j, best_thresh))

        candidates.sort(reverse=True)
        self.indicators_ = [(j, t) for _, j, t in candidates[:self.max_indicators]]

        # Build augmented feature matrix
        X_aug = self._augment(X)

        # Fit final Ridge
        self.ridge_ = RidgeCV(cv=3)
        self.ridge_.fit(X_aug, y)
        return self

    def _augment(self, X):
        parts = [X[:, self.selected_]]
        for j, thresh in self.indicators_:
            parts.append((X[:, j] > thresh).astype(np.float64).reshape(-1, 1))
        return np.hstack(parts)

    def predict(self, X):
        check_is_fitted(self, "ridge_")
        X = np.asarray(X, dtype=np.float64)
        return self.ridge_.predict(self._augment(X))

    def __str__(self):
        check_is_fitted(self, "ridge_")
        all_names = [f"x{i}" for i in range(self.n_features_in_)]
        sel_names = [all_names[i] for i in self.selected_]
        coefs = self.ridge_.coef_
        intercept = self.ridge_.intercept_
        n_sel = len(self.selected_)

        # Build equation with linear + indicator terms
        terms = []
        for idx in range(n_sel):
            c = coefs[idx]
            terms.append(f"{c:.4f}*{sel_names[idx]}")
        for k, (j, thresh) in enumerate(self.indicators_):
            c = coefs[n_sel + k]
            terms.append(f"{c:.4f}*I({all_names[j]}>{thresh:.2f})")
        terms.append(f"{intercept:.4f}")
        equation = " + ".join(terms)

        lines = [
            f"Ridge Regression with Threshold Indicators (α={self.ridge_.alpha_:.4g}):",
            f"  y = {equation}",
            f"  where I(condition) = 1 if condition is true, 0 otherwise",
            "",
            "Coefficients:",
        ]
        for idx in range(n_sel):
            lines.append(f"  {sel_names[idx]}: {coefs[idx]:.4f}")
        for k, (j, thresh) in enumerate(self.indicators_):
            c = coefs[n_sel + k]
            lines.append(f"  I({all_names[j]}>{thresh:.2f}): {c:.4f}  (adds {c:.4f} when {all_names[j]} > {thresh:.2f})")
        lines.append(f"  intercept: {intercept:.4f}")

        # Show excluded features
        excluded = [all_names[i] for i in range(self.n_features_in_) if i not in self.selected_]
        if excluded:
            lines.append(f"  Features excluded (zero effect): {', '.join(excluded)}")

        return "\n".join(lines)


class AdaptiveRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Ridge with adaptive per-dataset nonlinearity selection.

    For each dataset, automatically selects the single best augmentation
    from: x^2 (quadratic), I(x>t) (threshold), or max(0,x-t) (hinge).
    Uses CV score to pick the winner. Always exactly 1 extra term.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        from sklearn.model_selection import cross_val_score
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        n = X.shape[1]
        cv = min(3, len(y) // 5) if len(y) >= 15 else 2

        # Search all features (no pre-filter for best results)
        base_score = np.mean(cross_val_score(RidgeCV(cv=cv), X, y, cv=cv, scoring='neg_mean_squared_error'))
        top_feats = list(range(n))
        best_score = base_score
        best_type = None
        best_feat = None
        best_thresh = None

        # Try quadratic for top features
        for j in top_feats:
            X_aug = np.hstack([X, (X[:, j:j+1] ** 2)])
            try:
                score = np.mean(cross_val_score(RidgeCV(cv=cv), X_aug, y, cv=cv, scoring='neg_mean_squared_error'))
                if score > best_score:
                    best_score = score
                    best_type = 'quad'
                    best_feat = j
            except Exception:
                pass

        # Try threshold indicator for all features
        for j in top_feats:
            col = X[:, j]
            for q in [25, 50, 75]:
                t = np.percentile(col, q)
                ind = (col > t).astype(np.float64).reshape(-1, 1)
                if np.std(ind) < 1e-10:
                    continue
                X_aug = np.hstack([X, ind])
                try:
                    score = np.mean(cross_val_score(RidgeCV(cv=cv), X_aug, y, cv=cv, scoring='neg_mean_squared_error'))
                    if score > best_score:
                        best_score = score
                        best_type = 'thresh'
                        best_feat = j
                        best_thresh = t
                except Exception:
                    pass

        # Try hinge for all features
        for j in top_feats:
            col = X[:, j]
            for q in [25, 50, 75]:
                t = np.percentile(col, q)
                hinge = np.maximum(0, col - t).reshape(-1, 1)
                if np.std(hinge) < 1e-10:
                    continue
                X_aug = np.hstack([X, hinge])
                try:
                    score = np.mean(cross_val_score(RidgeCV(cv=cv), X_aug, y, cv=cv, scoring='neg_mean_squared_error'))
                    if score > best_score:
                        best_score = score
                        best_type = 'hinge'
                        best_feat = j
                        best_thresh = t
                except Exception:
                    pass

        # Try abs for all features
        for j in top_feats:
            ax = np.abs(X[:, j:j+1])
            if np.std(ax) < 1e-10:
                continue
            X_aug = np.hstack([X, ax])
            try:
                score = np.mean(cross_val_score(RidgeCV(cv=cv), X_aug, y, cv=cv, scoring='neg_mean_squared_error'))
                if score > best_score:
                    best_score = score
                    best_type = 'abs'
                    best_feat = j
            except Exception:
                pass

        # Try log(1+|x|) for all features — captures diminishing returns
        for j in top_feats:
            lx = np.log1p(np.abs(X[:, j:j+1]))
            if np.std(lx) < 1e-10:
                continue
            X_aug = np.hstack([X, lx])
            try:
                score = np.mean(cross_val_score(RidgeCV(cv=cv), X_aug, y, cv=cv, scoring='neg_mean_squared_error'))
                if score > best_score:
                    best_score = score
                    best_type = 'log'
                    best_feat = j
            except Exception:
                pass

        # Try sqrt(|x|) for all features
        for j in top_feats:
            sx = np.sqrt(np.abs(X[:, j:j+1]))
            if np.std(sx) < 1e-10:
                continue
            X_aug = np.hstack([X, sx])
            try:
                score = np.mean(cross_val_score(RidgeCV(cv=cv), X_aug, y, cv=cv, scoring='neg_mean_squared_error'))
                if score > best_score:
                    best_score = score
                    best_type = 'sqrt'
                    best_feat = j
            except Exception:
                pass

        # Try exp(-|x|) for all features
        for j in top_feats:
            ex = np.exp(-np.abs(X[:, j:j+1]))
            if np.std(ex) < 1e-10:
                continue
            X_aug = np.hstack([X, ex])
            try:
                score = np.mean(cross_val_score(RidgeCV(cv=cv), X_aug, y, cv=cv, scoring='neg_mean_squared_error'))
                if score > best_score:
                    best_score = score
                    best_type = 'exp_decay'
                    best_feat = j
            except Exception:
                pass

        # Try sign(x)*x^2 (signed square) for all features
        for j in top_feats:
            sx2 = np.sign(X[:, j:j+1]) * X[:, j:j+1]**2
            if np.std(sx2) < 1e-10:
                continue
            X_aug = np.hstack([X, sx2])
            try:
                score = np.mean(cross_val_score(RidgeCV(cv=cv), X_aug, y, cv=cv, scoring='neg_mean_squared_error'))
                if score > best_score:
                    best_score = score
                    best_type = 'signed_sq'
                    best_feat = j
            except Exception:
                pass

        # Try 1/(1+|x|) for all features
        for j in top_feats:
            inv = 1.0 / (1.0 + np.abs(X[:, j:j+1]))
            if np.std(inv) < 1e-10:
                continue
            X_aug = np.hstack([X, inv])
            try:
                score = np.mean(cross_val_score(RidgeCV(cv=cv), X_aug, y, cv=cv, scoring='neg_mean_squared_error'))
                if score > best_score:
                    best_score = score
                    best_type = 'inv'
                    best_feat = j
            except Exception:
                pass

        # Try top pairwise interaction x_i * x_j
        for i_idx in range(min(len(top_feats), 10)):
            for j_idx in range(i_idx + 1, min(len(top_feats), 10)):
                i, j = top_feats[i_idx], top_feats[j_idx]
                ij = (X[:, i] * X[:, j]).reshape(-1, 1)
                if np.std(ij) < 1e-10:
                    continue
                X_aug = np.hstack([X, ij])
                try:
                    score = np.mean(cross_val_score(RidgeCV(cv=cv), X_aug, y, cv=cv, scoring='neg_mean_squared_error'))
                    if score > best_score:
                        best_score = score
                        best_type = 'interact'
                        best_feat = i
                        best_thresh = j  # store second feature index in thresh
                except Exception:
                    pass

        self.aug_type_ = best_type
        self.aug_feat_ = best_feat
        self.aug_thresh_ = best_thresh

        # Fit final model
        X_aug = self._augment(X)
        self.ridge_ = RidgeCV(cv=cv)
        self.ridge_.fit(X_aug, y)
        return self

    def _augment(self, X):
        if self.aug_type_ is None:
            return X
        elif self.aug_type_ == 'quad':
            return np.hstack([X, X[:, self.aug_feat_:self.aug_feat_+1] ** 2])
        elif self.aug_type_ == 'thresh':
            return np.hstack([X, (X[:, self.aug_feat_] > self.aug_thresh_).astype(np.float64).reshape(-1, 1)])
        elif self.aug_type_ == 'hinge':
            return np.hstack([X, np.maximum(0, X[:, self.aug_feat_] - self.aug_thresh_).reshape(-1, 1)])
        elif self.aug_type_ == 'abs':
            return np.hstack([X, np.abs(X[:, self.aug_feat_:self.aug_feat_+1])])
        elif self.aug_type_ == 'log':
            return np.hstack([X, np.log1p(np.abs(X[:, self.aug_feat_:self.aug_feat_+1]))])
        elif self.aug_type_ == 'sqrt':
            return np.hstack([X, np.sqrt(np.abs(X[:, self.aug_feat_:self.aug_feat_+1]))])
        elif self.aug_type_ == 'exp_decay':
            return np.hstack([X, np.exp(-np.abs(X[:, self.aug_feat_:self.aug_feat_+1]))])
        elif self.aug_type_ == 'signed_sq':
            x = X[:, self.aug_feat_:self.aug_feat_+1]
            return np.hstack([X, np.sign(x) * x**2])
        elif self.aug_type_ == 'inv':
            return np.hstack([X, 1.0 / (1.0 + np.abs(X[:, self.aug_feat_:self.aug_feat_+1]))])
        elif self.aug_type_ == 'interact':
            j2 = int(self.aug_thresh_)
            return np.hstack([X, (X[:, self.aug_feat_] * X[:, j2]).reshape(-1, 1)])

    def predict(self, X):
        check_is_fitted(self, "ridge_")
        X = np.asarray(X, dtype=np.float64)
        return self.ridge_.predict(self._augment(X))

    def __str__(self):
        check_is_fitted(self, "ridge_")
        names = [f"x{i}" for i in range(self.n_features_in_)]
        coefs = self.ridge_.coef_
        intercept = self.ridge_.intercept_
        n = self.n_features_in_

        terms = [f"{coefs[j]:.4f}*{names[j]}" for j in range(n)]
        if self.aug_type_ == 'quad':
            terms.append(f"{coefs[n]:.4f}*{names[self.aug_feat_]}^2")
        elif self.aug_type_ == 'thresh':
            terms.append(f"{coefs[n]:.4f}*I({names[self.aug_feat_]}>{self.aug_thresh_:.2f})")
        elif self.aug_type_ == 'hinge':
            terms.append(f"{coefs[n]:.4f}*max(0,{names[self.aug_feat_]}-{self.aug_thresh_:.2f})")
        elif self.aug_type_ == 'abs':
            terms.append(f"{coefs[n]:.4f}*|{names[self.aug_feat_]}|")
        elif self.aug_type_ == 'log':
            terms.append(f"{coefs[n]:.4f}*log(1+|{names[self.aug_feat_]}|)")
        elif self.aug_type_ == 'sqrt':
            terms.append(f"{coefs[n]:.4f}*sqrt(|{names[self.aug_feat_]}|)")
        elif self.aug_type_ == 'exp_decay':
            terms.append(f"{coefs[n]:.4f}*exp(-|{names[self.aug_feat_]}|)")
        elif self.aug_type_ == 'signed_sq':
            terms.append(f"{coefs[n]:.4f}*sign({names[self.aug_feat_]})*{names[self.aug_feat_]}^2")
        elif self.aug_type_ == 'inv':
            terms.append(f"{coefs[n]:.4f}/(1+|{names[self.aug_feat_]}|)")
        elif self.aug_type_ == 'interact':
            j2 = int(self.aug_thresh_)
            terms.append(f"{coefs[n]:.4f}*{names[self.aug_feat_]}*{names[j2]}")
        terms.append(f"{intercept:.4f}")
        equation = " + ".join(terms)

        lines = [
            f"Ridge Regression (α={self.ridge_.alpha_:.4g}):",
            f"  y = {equation}",
        ]
        if self.aug_type_ == 'thresh':
            lines.append(f"  where I(condition) = 1 if true, 0 otherwise")
        elif self.aug_type_ == 'hinge':
            lines.append(f"  where max(0, x-t) = 0 when x<=t, x-t when x>t")
        lines.append("")
        lines.append("Coefficients:")
        for j in range(n):
            lines.append(f"  {names[j]}: {coefs[j]:.4f}")
        if self.aug_type_ == 'quad':
            lines.append(f"  {names[self.aug_feat_]}^2: {coefs[n]:.4f}")
        elif self.aug_type_ == 'thresh':
            lines.append(f"  I({names[self.aug_feat_]}>{self.aug_thresh_:.2f}): {coefs[n]:.4f}")
        elif self.aug_type_ == 'hinge':
            lines.append(f"  max(0,{names[self.aug_feat_]}-{self.aug_thresh_:.2f}): {coefs[n]:.4f}")
        elif self.aug_type_ == 'abs':
            lines.append(f"  |{names[self.aug_feat_]}|: {coefs[n]:.4f}")
        elif self.aug_type_ == 'log':
            lines.append(f"  log(1+|{names[self.aug_feat_]}|): {coefs[n]:.4f}")
        elif self.aug_type_ == 'sqrt':
            lines.append(f"  sqrt(|{names[self.aug_feat_]}|): {coefs[n]:.4f}")
        elif self.aug_type_ == 'exp_decay':
            lines.append(f"  exp(-|{names[self.aug_feat_]}|): {coefs[n]:.4f}")
        elif self.aug_type_ == 'signed_sq':
            lines.append(f"  sign({names[self.aug_feat_]})*{names[self.aug_feat_]}^2: {coefs[n]:.4f}")
        elif self.aug_type_ == 'inv':
            lines.append(f"  1/(1+|{names[self.aug_feat_]}|): {coefs[n]:.4f}")
        elif self.aug_type_ == 'interact':
            j2 = int(self.aug_thresh_)
            lines.append(f"  {names[self.aug_feat_]}*{names[j2]}: {coefs[n]:.4f}")
        lines.append(f"  intercept: {intercept:.4f}")

        return "\n".join(lines)


class FullQuadRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    RidgeCV on [x, x^2] features (no interactions). All features get
    both linear and quadratic terms. Ridge regularization prevents overfitting.
    Display shows all terms.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        n = X.shape[1]

        X_aug = np.hstack([X, X ** 2])
        names = [f"x{i}" for i in range(n)]
        self.aug_names_ = names + [f"{nm}^2" for nm in names]

        self.ridge_ = RidgeCV(cv=3)
        self.ridge_.fit(X_aug, y)
        return self

    def predict(self, X):
        check_is_fitted(self, "ridge_")
        X = np.asarray(X, dtype=np.float64)
        return self.ridge_.predict(np.hstack([X, X ** 2]))

    def __str__(self):
        check_is_fitted(self, "ridge_")
        coefs = self.ridge_.coef_
        intercept = self.ridge_.intercept_

        terms = [f"{c:.4f}*{n}" for n, c in zip(self.aug_names_, coefs)]
        terms.append(f"{intercept:.4f}")
        equation = " + ".join(terms)

        lines = [
            f"Ridge Regression with Quadratic Terms (α={self.ridge_.alpha_:.4g}):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]
        for n, c in zip(self.aug_names_, coefs):
            lines.append(f"  {n}: {c:.4f}")
        lines.append(f"  intercept: {intercept:.4f}")

        return "\n".join(lines)


class HingeRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Ridge + learned-threshold hinge features: max(0, x_j - t).

    For each feature, finds the best threshold t on Ridge residuals,
    then adds max(0, x_j - t) as a feature. This captures ramp-shaped
    nonlinearities (linear above threshold, zero below).

    Display: y = a*x0 + ... + c*max(0, x_j - t) + intercept
    """

    def __init__(self, max_hinges=1):
        self.max_hinges = max_hinges

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        n = X.shape[1]

        ridge_init = RidgeCV(cv=3)
        ridge_init.fit(X, y)
        residual = y - ridge_init.predict(X)

        # Find best hinge threshold per feature
        candidates = []
        for j in range(n):
            col = X[:, j]
            best_score = -np.inf
            best_thresh = None
            for q in np.linspace(10, 90, 17):
                thresh = np.percentile(col, q)
                hinge = np.maximum(0, col - thresh)
                if np.std(hinge) < 1e-10:
                    continue
                corr = abs(np.corrcoef(hinge, residual)[0, 1])
                if not np.isnan(corr) and corr > best_score:
                    best_score = corr
                    best_thresh = thresh
            if best_thresh is not None:
                candidates.append((best_score, j, best_thresh))

        candidates.sort(reverse=True)
        self.hinges_ = [(j, t) for _, j, t in candidates[:self.max_hinges]]

        X_aug = self._augment(X)
        self.ridge_ = RidgeCV(cv=3)
        self.ridge_.fit(X_aug, y)
        return self

    def _augment(self, X):
        parts = [X]
        for j, thresh in self.hinges_:
            parts.append(np.maximum(0, X[:, j] - thresh).reshape(-1, 1))
        return np.hstack(parts)

    def predict(self, X):
        check_is_fitted(self, "ridge_")
        X = np.asarray(X, dtype=np.float64)
        return self.ridge_.predict(self._augment(X))

    def __str__(self):
        check_is_fitted(self, "ridge_")
        names = [f"x{i}" for i in range(self.n_features_in_)]
        coefs = self.ridge_.coef_
        intercept = self.ridge_.intercept_
        n = self.n_features_in_

        terms = []
        for j in range(n):
            terms.append(f"{coefs[j]:.4f}*{names[j]}")
        for k, (j, t) in enumerate(self.hinges_):
            c = coefs[n + k]
            terms.append(f"{c:.4f}*max(0,{names[j]}-{t:.2f})")
        terms.append(f"{intercept:.4f}")
        equation = " + ".join(terms)

        lines = [
            f"Ridge Regression with Hinge Terms (α={self.ridge_.alpha_:.4g}):",
            f"  y = {equation}",
            f"  (max(0, x-t) = 0 when x <= t, and x-t when x > t)",
            "",
            "Coefficients:",
        ]
        for j in range(n):
            lines.append(f"  {names[j]}: {coefs[j]:.4f}")
        for k, (j, t) in enumerate(self.hinges_):
            c = coefs[n + k]
            lines.append(f"  max(0,{names[j]}-{t:.2f}): {c:.4f}")
        lines.append(f"  intercept: {intercept:.4f}")

        return "\n".join(lines)


class RoundedQuadRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    QuadRidge1 with coefficients rounded to 2 significant figures.
    Rounding makes arithmetic easier for LLM while predictions stay close.
    """

    def __init__(self, sig_figs=2):
        self.sig_figs = sig_figs

    def _round_sig(self, x, sig=2):
        if abs(x) < 1e-10:
            return 0.0
        return round(x, sig - 1 - int(np.floor(np.log10(abs(x)))))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        # Fit QuadRidge1 internally
        self._inner = QuadraticRidgeRegressor(max_quad_features=1)
        self._inner.fit(X, y)

        # Round coefficients
        coefs = self._inner.ridge_.coef_.copy()
        intercept = self._inner.ridge_.intercept_
        self.coefs_ = np.array([self._round_sig(c, self.sig_figs) for c in coefs])
        self.intercept_ = self._round_sig(intercept, self.sig_figs)
        self.quad_features_ = self._inner.quad_features_
        return self

    def predict(self, X):
        check_is_fitted(self, "coefs_")
        X = np.asarray(X, dtype=np.float64)
        X_aug = self._inner._augment(X)
        return X_aug @ self.coefs_ + self.intercept_

    def __str__(self):
        check_is_fitted(self, "coefs_")
        names = [f"x{i}" for i in range(self.n_features_in_)]
        n = self.n_features_in_

        terms = []
        for j in range(n):
            c = self.coefs_[j]
            if abs(c) > 1e-10:
                terms.append(f"{c}*{names[j]}")
        for k, j in enumerate(self.quad_features_):
            c = self.coefs_[n + k]
            if abs(c) > 1e-10:
                terms.append(f"{c}*{names[j]}^2")
        terms.append(f"{self.intercept_}")
        equation = " + ".join(terms)

        lines = [
            "Ridge Regression with Quadratic Term (coefficients rounded):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]
        for j in range(n):
            if abs(self.coefs_[j]) > 1e-10:
                lines.append(f"  {names[j]}: {self.coefs_[j]}")
        for k, j in enumerate(self.quad_features_):
            if abs(self.coefs_[n + k]) > 1e-10:
                lines.append(f"  {names[j]}^2: {self.coefs_[n + k]}")
        lines.append(f"  intercept: {self.intercept_}")

        # Show zeroed features
        zeroed = [names[j] for j in range(n) if abs(self.coefs_[j]) <= 1e-10]
        if zeroed:
            lines.append(f"  Features with zero coefficient: {', '.join(zeroed)}")

        return "\n".join(lines)


class QuadraticRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Ridge + targeted quadratic features for top nonlinear features.

    1. Fit initial Ridge
    2. For each feature, compute correlation of x_i^2 with residual
    3. Add x_i^2 for top-k features
    4. Refit Ridge on [original + quadratic features]
    5. Display as equation: y = a*x0 + b*x0^2 + c*x1 + ... + intercept
    """

    def __init__(self, max_quad_features=2):
        self.max_quad_features = max_quad_features

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        n = X.shape[1]

        # Initial Ridge
        ridge_init = RidgeCV(cv=3)
        ridge_init.fit(X, y)
        residual = y - ridge_init.predict(X)

        # Find features where x^2 best correlates with residual
        candidates = []
        for j in range(n):
            x2 = X[:, j] ** 2
            if np.std(x2) < 1e-10:
                continue
            corr = abs(np.corrcoef(x2, residual)[0, 1])
            if not np.isnan(corr):
                candidates.append((corr, j))
        candidates.sort(reverse=True)
        self.quad_features_ = [j for _, j in candidates[:self.max_quad_features]]

        # Build augmented features
        X_aug = self._augment(X)

        # Fit final Ridge
        self.ridge_ = RidgeCV(cv=3)
        self.ridge_.fit(X_aug, y)
        return self

    def _augment(self, X):
        parts = [X]
        for j in self.quad_features_:
            parts.append((X[:, j:j+1] ** 2))
        return np.hstack(parts)

    def predict(self, X):
        check_is_fitted(self, "ridge_")
        X = np.asarray(X, dtype=np.float64)
        return self.ridge_.predict(self._augment(X))

    def __str__(self):
        check_is_fitted(self, "ridge_")
        names = [f"x{i}" for i in range(self.n_features_in_)]
        coefs = self.ridge_.coef_
        intercept = self.ridge_.intercept_
        n = self.n_features_in_

        terms = []
        for j in range(n):
            terms.append(f"{coefs[j]:.4f}*{names[j]}")
        for k, j in enumerate(self.quad_features_):
            c = coefs[n + k]
            terms.append(f"{c:.4f}*{names[j]}^2")
        terms.append(f"{intercept:.4f}")
        equation = " + ".join(terms)

        lines = [
            f"Ridge Regression with Quadratic Terms (α={self.ridge_.alpha_:.4g}):",
            f"  y = {equation}",
            f"  (Note: {names[j]}^2 means {names[j]}*{names[j]})" if self.quad_features_ else "",
            "",
            "Coefficients:",
        ]
        lines = [l for l in lines if l is not None]
        for j in range(n):
            lines.append(f"  {names[j]}: {coefs[j]:.4f}")
        for k, j in enumerate(self.quad_features_):
            lines.append(f"  {names[j]}^2: {coefs[n + k]:.4f}")
        lines.append(f"  intercept: {intercept:.4f}")

        return "\n".join(lines)


class AbsRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Ridge + |x_i| absolute value terms for top nonlinear features.
    |x| captures V-shaped effects and is simpler than x^2 for LLM.
    """

    def __init__(self, max_abs_features=1):
        self.max_abs_features = max_abs_features

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        n = X.shape[1]

        ridge_init = RidgeCV(cv=3)
        ridge_init.fit(X, y)
        residual = y - ridge_init.predict(X)

        candidates = []
        for j in range(n):
            ax = np.abs(X[:, j])
            if np.std(ax) < 1e-10:
                continue
            corr = abs(np.corrcoef(ax, residual)[0, 1])
            if not np.isnan(corr):
                candidates.append((corr, j))
        candidates.sort(reverse=True)
        self.abs_features_ = [j for _, j in candidates[:self.max_abs_features]]

        X_aug = self._augment(X)
        self.ridge_ = RidgeCV(cv=3)
        self.ridge_.fit(X_aug, y)
        return self

    def _augment(self, X):
        parts = [X]
        for j in self.abs_features_:
            parts.append(np.abs(X[:, j:j+1]))
        return np.hstack(parts)

    def predict(self, X):
        check_is_fitted(self, "ridge_")
        X = np.asarray(X, dtype=np.float64)
        return self.ridge_.predict(self._augment(X))

    def __str__(self):
        check_is_fitted(self, "ridge_")
        names = [f"x{i}" for i in range(self.n_features_in_)]
        coefs = self.ridge_.coef_
        intercept = self.ridge_.intercept_
        n = self.n_features_in_

        terms = []
        for j in range(n):
            terms.append(f"{coefs[j]:.4f}*{names[j]}")
        for k, j in enumerate(self.abs_features_):
            c = coefs[n + k]
            terms.append(f"{c:.4f}*|{names[j]}|")
        terms.append(f"{intercept:.4f}")
        equation = " + ".join(terms)

        lines = [
            f"Ridge Regression with Absolute Value Terms (α={self.ridge_.alpha_:.4g}):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]
        for j in range(n):
            lines.append(f"  {names[j]}: {coefs[j]:.4f}")
        for k, j in enumerate(self.abs_features_):
            lines.append(f"  |{names[j]}|: {coefs[n + k]:.4f}")
        lines.append(f"  intercept: {intercept:.4f}")

        return "\n".join(lines)


class QuadRidgePlusStumpsRegressor(BaseEstimator, RegressorMixin):
    """
    QuadraticRidge + greedy stumps on residual.
    Stage 1: Ridge + best quad feature
    Stage 2: Greedy stumps on remaining residual
    """

    def __init__(self, max_quad=1, n_stumps=1):
        self.max_quad = max_quad
        self.n_stumps = n_stumps

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        # Stage 1: QuadRidge
        self.quad_ridge_ = QuadraticRidgeRegressor(max_quad_features=self.max_quad)
        self.quad_ridge_.fit(X, y)

        # Stage 2: Stumps on residual
        residual = y - self.quad_ridge_.predict(X)
        self.stumps_ = []
        for k in range(self.n_stumps):
            stump = DecisionTreeRegressor(
                max_depth=1,
                min_samples_leaf=max(5, len(y) // 20),
                random_state=42 + k,
            )
            stump.fit(X, residual)
            self.stumps_.append(stump)
            residual = residual - stump.predict(X)
        return self

    def predict(self, X):
        check_is_fitted(self, "quad_ridge_")
        X = np.asarray(X, dtype=np.float64)
        pred = self.quad_ridge_.predict(X)
        for stump in self.stumps_:
            pred = pred + stump.predict(X)
        return pred

    def __str__(self):
        check_is_fitted(self, "quad_ridge_")
        names = [f"x{i}" for i in range(self.n_features_in_)]
        base_str = str(self.quad_ridge_)

        lines = base_str.split("\n")
        if self.stumps_:
            lines.append("")
            lines.append("Then add each stump correction:")
            for i, stump in enumerate(self.stumps_):
                tree = stump.tree_
                if tree.node_count <= 1:
                    lines.append(f"  Stump {i+1}: add {float(tree.value[0].flatten()[0]):.4f}")
                    continue
                feat = int(tree.feature[0])
                thresh = float(tree.threshold[0])
                left_val = float(tree.value[tree.children_left[0]].flatten()[0])
                right_val = float(tree.value[tree.children_right[0]].flatten()[0])
                lines.append(
                    f"  Stump {i+1}: if {names[feat]} <= {thresh:.2f} then add {left_val:+.4f}, "
                    f"else add {right_val:+.4f}"
                )
        return "\n".join(lines)


class InteractRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Ridge + top-k interaction terms (x_i * x_j).

    1. Fit initial Ridge
    2. Find pairwise interactions most correlated with residual
    3. Add x_i*x_j features
    4. Refit Ridge
    5. Display as equation with interaction terms
    """

    def __init__(self, max_interactions=1):
        self.max_interactions = max_interactions

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        n = X.shape[1]

        ridge_init = RidgeCV(cv=3)
        ridge_init.fit(X, y)
        residual = y - ridge_init.predict(X)

        candidates = []
        for i in range(min(n, 15)):
            for j in range(i + 1, min(n, 15)):
                ij = X[:, i] * X[:, j]
                if np.std(ij) < 1e-10:
                    continue
                corr = abs(np.corrcoef(ij, residual)[0, 1])
                if not np.isnan(corr):
                    candidates.append((corr, i, j))
        candidates.sort(reverse=True)
        self.interactions_ = [(i, j) for _, i, j in candidates[:self.max_interactions]]

        X_aug = self._augment(X)
        self.ridge_ = RidgeCV(cv=3)
        self.ridge_.fit(X_aug, y)
        return self

    def _augment(self, X):
        parts = [X]
        for i, j in self.interactions_:
            parts.append((X[:, i] * X[:, j]).reshape(-1, 1))
        return np.hstack(parts)

    def predict(self, X):
        check_is_fitted(self, "ridge_")
        X = np.asarray(X, dtype=np.float64)
        return self.ridge_.predict(self._augment(X))

    def __str__(self):
        check_is_fitted(self, "ridge_")
        names = [f"x{i}" for i in range(self.n_features_in_)]
        coefs = self.ridge_.coef_
        intercept = self.ridge_.intercept_
        n = self.n_features_in_

        terms = []
        for j in range(n):
            terms.append(f"{coefs[j]:.4f}*{names[j]}")
        for k, (i, j) in enumerate(self.interactions_):
            c = coefs[n + k]
            terms.append(f"{c:.4f}*{names[i]}*{names[j]}")
        terms.append(f"{intercept:.4f}")
        equation = " + ".join(terms)

        lines = [
            f"Ridge Regression with Interaction Terms (α={self.ridge_.alpha_:.4g}):",
            f"  y = {equation}",
            "",
            "Coefficients:",
        ]
        for j in range(n):
            lines.append(f"  {names[j]}: {coefs[j]:.4f}")
        for k, (i, j) in enumerate(self.interactions_):
            lines.append(f"  {names[i]}*{names[j]}: {coefs[n + k]:.4f}")
        lines.append(f"  intercept: {intercept:.4f}")

        return "\n".join(lines)


class QuadThreshRidgeRegressor(BaseEstimator, RegressorMixin):
    """
    Ridge + quadratic term for top feature + threshold indicators.
    Combines both nonlinear augmentation strategies in one equation.
    """

    def __init__(self, max_quad=1, max_indicators=2):
        self.max_quad = max_quad
        self.max_indicators = max_indicators

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        n = X.shape[1]

        # Initial Ridge to get residuals
        ridge_init = RidgeCV(cv=3)
        ridge_init.fit(X, y)
        residual = y - ridge_init.predict(X)

        # Find best quadratic features
        quad_candidates = []
        for j in range(n):
            x2 = X[:, j] ** 2
            if np.std(x2) < 1e-10:
                continue
            corr = abs(np.corrcoef(x2, residual)[0, 1])
            if not np.isnan(corr):
                quad_candidates.append((corr, j))
        quad_candidates.sort(reverse=True)
        self.quad_features_ = [j for _, j in quad_candidates[:self.max_quad]]

        # Find best threshold indicators (on residual after accounting for quad)
        X_with_quad = np.hstack([X] + [X[:, j:j+1]**2 for j in self.quad_features_])
        ridge_q = RidgeCV(cv=3)
        ridge_q.fit(X_with_quad, y)
        residual2 = y - ridge_q.predict(X_with_quad)

        thresh_candidates = []
        for j in range(n):
            col = X[:, j]
            best_score = -np.inf
            best_thresh = None
            for q in np.linspace(10, 90, 17):
                thresh = np.percentile(col, q)
                left = residual2[col <= thresh]
                right = residual2[col > thresh]
                if len(left) < 5 or len(right) < 5:
                    continue
                score = np.var(residual2) - (len(left)*np.var(left) + len(right)*np.var(right)) / len(residual2)
                if score > best_score:
                    best_score = score
                    best_thresh = thresh
            if best_thresh is not None:
                thresh_candidates.append((best_score, j, best_thresh))
        thresh_candidates.sort(reverse=True)
        self.indicators_ = [(j, t) for _, j, t in thresh_candidates[:self.max_indicators]]

        # Build final augmented features and fit
        X_aug = self._augment(X)
        self.ridge_ = RidgeCV(cv=3)
        self.ridge_.fit(X_aug, y)
        return self

    def _augment(self, X):
        parts = [X]
        for j in self.quad_features_:
            parts.append(X[:, j:j+1] ** 2)
        for j, thresh in self.indicators_:
            parts.append((X[:, j] > thresh).astype(np.float64).reshape(-1, 1))
        return np.hstack(parts)

    def predict(self, X):
        check_is_fitted(self, "ridge_")
        X = np.asarray(X, dtype=np.float64)
        return self.ridge_.predict(self._augment(X))

    def __str__(self):
        check_is_fitted(self, "ridge_")
        names = [f"x{i}" for i in range(self.n_features_in_)]
        coefs = self.ridge_.coef_
        intercept = self.ridge_.intercept_
        n = self.n_features_in_

        terms = []
        for j in range(n):
            terms.append(f"{coefs[j]:.4f}*{names[j]}")
        idx = n
        for j in self.quad_features_:
            terms.append(f"{coefs[idx]:.4f}*{names[j]}^2")
            idx += 1
        for j, thresh in self.indicators_:
            terms.append(f"{coefs[idx]:.4f}*I({names[j]}>{thresh:.2f})")
            idx += 1
        terms.append(f"{intercept:.4f}")
        equation = " + ".join(terms)

        lines = [
            f"Ridge with Quadratic and Threshold Terms (α={self.ridge_.alpha_:.4g}):",
            f"  y = {equation}",
            f"  where I(condition) = 1 if true, 0 if false",
            "",
            "Coefficients:",
        ]
        for j in range(n):
            lines.append(f"  {names[j]}: {coefs[j]:.4f}")
        idx = n
        for j in self.quad_features_:
            lines.append(f"  {names[j]}^2: {coefs[idx]:.4f}")
            idx += 1
        for j, thresh in self.indicators_:
            lines.append(f"  I({names[j]}>{thresh:.2f}): {coefs[idx]:.4f}")
            idx += 1
        lines.append(f"  intercept: {intercept:.4f}")

        return "\n".join(lines)


class ThreshRidgePlusStumpsRegressor(BaseEstimator, RegressorMixin):
    """
    Hybrid: ThreshRidge (2 indicators) + 1 greedy stump on residual.

    Stage 1: Ridge + 2 threshold indicators (all in one equation)
    Stage 2: 1 stump on the remaining residual

    Combines the best of both approaches.
    """

    def __init__(self, max_indicators=2, n_stumps=1):
        self.max_indicators = max_indicators
        self.n_stumps = n_stumps

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        # Stage 1: ThreshRidge
        self.thresh_ridge_ = ThresholdRidgeRegressor(max_indicators=self.max_indicators)
        self.thresh_ridge_.fit(X, y)

        # Stage 2: Stump on residual
        residual = y - self.thresh_ridge_.predict(X)
        self.stumps_ = []
        for k in range(self.n_stumps):
            stump = DecisionTreeRegressor(
                max_depth=1,
                min_samples_leaf=max(5, len(y) // 20),
                random_state=42 + k,
            )
            stump.fit(X, residual)
            self.stumps_.append(stump)
            residual = residual - stump.predict(X)

        return self

    def predict(self, X):
        check_is_fitted(self, "thresh_ridge_")
        X = np.asarray(X, dtype=np.float64)
        pred = self.thresh_ridge_.predict(X)
        for stump in self.stumps_:
            pred = pred + stump.predict(X)
        return pred

    def __str__(self):
        check_is_fitted(self, "thresh_ridge_")
        names = [f"x{i}" for i in range(self.n_features_in_)]

        # Get the ThreshRidge string
        base_str = str(self.thresh_ridge_)

        # Add stump corrections
        lines = base_str.split("\n")
        lines.append("")
        lines.append("Then add each stump correction:")
        for i, stump in enumerate(self.stumps_):
            tree = stump.tree_
            if tree.node_count <= 1:
                lines.append(f"  Stump {i+1}: add {float(tree.value[0].flatten()[0]):.4f}")
                continue
            feat = int(tree.feature[0])
            thresh = float(tree.threshold[0])
            left_val = float(tree.value[tree.children_left[0]].flatten()[0])
            right_val = float(tree.value[tree.children_right[0]].flatten()[0])
            lines.append(
                f"  Stump {i+1}: if {names[feat]} <= {thresh:.2f} then add {left_val:+.4f}, "
                f"else add {right_val:+.4f}"
            )

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
RidgePlusStumpsRegressor.__module__ = "interpretable_regressor"
RidgePlusTreeRegressor.__module__ = "interpretable_regressor"
ThresholdRidgeRegressor.__module__ = "interpretable_regressor"
ThreshRidgePlusStumpsRegressor.__module__ = "interpretable_regressor"
QuadraticRidgeRegressor.__module__ = "interpretable_regressor"
QuadThreshRidgeRegressor.__module__ = "interpretable_regressor"
InteractRidgeRegressor.__module__ = "interpretable_regressor"
QuadRidgePlusStumpsRegressor.__module__ = "interpretable_regressor"
AbsRidgeRegressor.__module__ = "interpretable_regressor"
RoundedQuadRidgeRegressor.__module__ = "interpretable_regressor"
HingeRidgeRegressor.__module__ = "interpretable_regressor"
FullQuadRidgeRegressor.__module__ = "interpretable_regressor"
AdaptiveRidgeRegressor.__module__ = "interpretable_regressor"


# ---------------------------------------------------------------------------
# Evaluation (do not edit any of the evaluation functions, only the names and model descriptions below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    model_defs = [("AdaptiveRidge_v11", AdaptiveRidgeRegressor())]

    # Interpretability tests
    interp_results = run_all_interp_tests(model_defs)
    n_passed = sum(r["passed"] for r in interp_results)
    total = len(interp_results)

    # prediction performance (RMSE)
    dataset_rmses = evaluate_all_regressors(model_defs)

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_hash = ""

    # --- Upsert interpretability_results.csv ---
    model_name = "AdaptiveRidge_v11"
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
    from collections import defaultdict
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
    mean_rank = avg_rank.get(model_name, float("nan"))

    upsert_overall_results([{
        "commit":                             git_hash,
        "mean_rank":                          f"{mean_rank:.2f}" if not np.isnan(mean_rank) else "nan",
        "frac_interpretability_tests_passed": f"{n_passed / total:.4f}" if total > 0 else "nan",
        "status":                             "",
        "model_name":                         "AbsRidge1",
        "description":                        "AdaptiveRidge v11: 10 types + pairwise interaction search",
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

