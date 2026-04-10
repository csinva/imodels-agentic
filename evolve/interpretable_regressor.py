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
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class SparseScreenedHingeRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse robust linear model with a small screened hinge expansion.

    Steps:
    1) Robust-scale features using median/IQR and optional winsorization.
    2) Fit ridge linear model, keep only strongest coefficients.
    3) Screen candidate one-feature hinge terms against residual correlation.
    4) Refit a tiny ridge model on kept linear + hinge terms.

    Final form:
      y = b + sum_j w_j * z_j + sum_k v_k * max(0, s_k * (z_{f_k} - t_k))
    where s_k is +1 or -1.
    """

    def __init__(
        self,
        alpha_linear=0.8,
        alpha_hinge=2.5,
        max_linear_features=8,
        max_hinges=6,
        coef_tol=0.015,
        clip_quantile=0.01,
        hinge_quantiles=(0.15, 0.35, 0.5, 0.65, 0.85),
        screen_multiplier=2,
    ):
        self.alpha_linear = alpha_linear
        self.alpha_hinge = alpha_hinge
        self.max_linear_features = max_linear_features
        self.max_hinges = max_hinges
        self.coef_tol = coef_tol
        self.clip_quantile = clip_quantile
        self.hinge_quantiles = hinge_quantiles
        self.screen_multiplier = screen_multiplier

    def _solve_diag_ridge(self, Z, y, penalties):
        p = Z.shape[1]
        lhs = Z.T @ Z + np.diag(penalties)
        rhs = Z.T @ y
        try:
            return np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    def _robust_scale(self, X):
        med = np.median(X, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        iqr = q75 - q25
        iqr[iqr < 1e-8] = 1.0
        z = (X - med) / iqr
        lo = None
        hi = None
        if self.clip_quantile > 0:
            lo = np.quantile(z, self.clip_quantile, axis=0)
            hi = np.quantile(z, 1.0 - self.clip_quantile, axis=0)
            z = np.minimum(np.maximum(z, lo), hi)
        return z, med, iqr, lo, hi

    @staticmethod
    def _hinge(z_col, threshold, sign):
        # sign=+1 => max(0, z-threshold), sign=-1 => max(0, threshold-z)
        return np.maximum(0.0, sign * (z_col - threshold))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, p = X.shape
        self.n_features_in_ = p

        z, med, iqr, z_lo, z_hi = self._robust_scale(X)
        self.x_median_ = med
        self.x_iqr_ = iqr
        self.z_lo_ = z_lo
        self.z_hi_ = z_hi

        self.y_mean_ = float(y.mean())
        yc = y - self.y_mean_

        # Initial full linear fit.
        penalties_full = self.alpha_linear * np.ones(p, dtype=float)
        w_full = self._solve_diag_ridge(z, yc, penalties_full)

        # Keep strongest linear terms for sparsity and clearer explanations.
        keep = np.zeros(p, dtype=bool)
        if p > 0:
            top = np.argsort(np.abs(w_full))[::-1][: min(self.max_linear_features, p)]
            keep[top] = True
        keep &= np.abs(w_full) >= self.coef_tol
        self.active_linear_features_ = np.where(keep)[0].tolist()

        if self.active_linear_features_:
            Z_lin = z[:, self.active_linear_features_]
            w_lin = self._solve_diag_ridge(
                Z_lin,
                yc,
                self.alpha_linear * np.ones(Z_lin.shape[1], dtype=float),
            )
            resid = yc - Z_lin @ w_lin
        else:
            Z_lin = np.zeros((n, 0), dtype=float)
            w_lin = np.zeros(0, dtype=float)
            resid = yc.copy()

        # Screen hinge candidates with residual correlation, then refit jointly.
        hinge_candidates = []
        if self.max_hinges > 0 and p > 0:
            screened_feature_budget = min(p, max(1, self.screen_multiplier * max(len(self.active_linear_features_), 1)))
            feature_scores = np.abs(w_full)
            screened_features = np.argsort(feature_scores)[::-1][:screened_feature_budget]

            for j in screened_features:
                z_j = z[:, j]
                thresholds = np.quantile(z_j, self.hinge_quantiles)
                for t in np.unique(thresholds):
                    for sgn in (1.0, -1.0):
                        h = self._hinge(z_j, t, sgn)
                        norm = np.sqrt(np.dot(h, h)) + 1e-12
                        score = np.abs(np.dot(h, resid)) / norm
                        hinge_candidates.append((score, j, float(t), float(sgn), h))

        hinge_candidates.sort(key=lambda x: x[0], reverse=True)

        self.hinge_terms_ = []
        hinge_cols = []
        used_sign_threshold = set()
        for _, j, t, sgn, h in hinge_candidates:
            key = (j, round(t, 6), int(sgn))
            if key in used_sign_threshold:
                continue
            used_sign_threshold.add(key)
            self.hinge_terms_.append((j, t, sgn))
            hinge_cols.append(h.reshape(-1, 1))
            if len(self.hinge_terms_) >= self.max_hinges:
                break

        Z_hinge = np.hstack(hinge_cols) if hinge_cols else np.zeros((n, 0), dtype=float)

        Z_all = np.hstack([Z_lin, Z_hinge])
        if Z_all.shape[1] > 0:
            penalties = np.concatenate(
                [
                    self.alpha_linear * np.ones(Z_lin.shape[1], dtype=float),
                    self.alpha_hinge * np.ones(Z_hinge.shape[1], dtype=float),
                ]
            )
            coef = self._solve_diag_ridge(Z_all, yc, penalties)
            self.linear_coefs_ = coef[: Z_lin.shape[1]]
            self.hinge_coefs_ = coef[Z_lin.shape[1] :]
        else:
            self.linear_coefs_ = np.zeros(0, dtype=float)
            self.hinge_coefs_ = np.zeros(0, dtype=float)

        return self

    def _transform(self, X):
        z = (X - self.x_median_) / self.x_iqr_
        if self.clip_quantile > 0 and self.z_lo_ is not None and self.z_hi_ is not None:
            z = np.minimum(np.maximum(z, self.z_lo_), self.z_hi_)

        if self.active_linear_features_:
            Z_lin = z[:, self.active_linear_features_]
        else:
            Z_lin = np.zeros((X.shape[0], 0), dtype=float)

        hinge_cols = []
        for j, t, sgn in self.hinge_terms_:
            hinge_cols.append(self._hinge(z[:, j], t, sgn).reshape(-1, 1))
        Z_hinge = np.hstack(hinge_cols) if hinge_cols else np.zeros((X.shape[0], 0), dtype=float)
        return Z_lin, Z_hinge

    def predict(self, X):
        check_is_fitted(
            self,
            [
                "x_median_",
                "x_iqr_",
                "active_linear_features_",
                "hinge_terms_",
                "linear_coefs_",
                "hinge_coefs_",
                "y_mean_",
            ],
        )
        X = np.asarray(X, dtype=float)
        Z_lin, Z_hinge = self._transform(X)
        pred = np.full(X.shape[0], self.y_mean_, dtype=float)
        if Z_lin.shape[1] > 0:
            pred += Z_lin @ self.linear_coefs_
        if Z_hinge.shape[1] > 0:
            pred += Z_hinge @ self.hinge_coefs_
        return pred

    def __str__(self):
        check_is_fitted(
            self,
            [
                "active_linear_features_",
                "hinge_terms_",
                "linear_coefs_",
                "hinge_coefs_",
                "y_mean_",
                "n_features_in_",
            ],
        )
        lines = [
            "Sparse Screened Hinge Regressor",
            "z_j = clip((x_j - median_j) / iqr_j)",
            "y = intercept + sum_j w_j*z_j + sum_k v_k*max(0, s_k*(z_f - t_k))",
            "",
            f"intercept: {self.y_mean_:+.4f}",
            f"active_linear_features: {len(self.active_linear_features_)} / {self.n_features_in_}",
            f"hinge_terms: {len(self.hinge_terms_)}",
            "",
            "Linear terms:",
        ]
        if self.active_linear_features_:
            for j, c in zip(self.active_linear_features_, self.linear_coefs_):
                lines.append(f"  {c:+.4f} * z{xstr(j)}")
        else:
            lines.append("  none")

        lines.append("")
        lines.append("Hinge terms:")
        if self.hinge_terms_:
            for (j, t, sgn), c in zip(self.hinge_terms_, self.hinge_coefs_):
                if sgn > 0:
                    expr = f"max(0, z{xstr(j)} - {t:+.3f})"
                else:
                    expr = f"max(0, {t:+.3f} - z{xstr(j)})"
                lines.append(f"  {c:+.4f} * {expr}")
        else:
            lines.append("  none")
        return "\n".join(lines)


def xstr(i):
    return str(int(i))


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseScreenedHingeRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseScreenHinge_v1"
model_description = "Robust sparse linear core plus residual-screened one-feature hinge basis with stronger hinge shrinkage"
model_defs = [(model_shorthand_name, SparseScreenedHingeRegressor())]


# ---------------------------------------------------------------------------
# Evaluation (do not edit anything below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

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
