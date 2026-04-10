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


class SparseDualKnotInteractionRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse dual-knot additive model with a tiny interaction budget.

    For each robustly scaled feature z_j we use five piecewise-linear pieces:
      z_j, max(0, z_j-k1), max(0, k1-z_j), max(0, z_j-k2), max(0, k2-z_j)
    then add up to `max_interactions` products z_i*z_j selected by residual
    correlation and strongly shrunk.
    """

    def __init__(
        self,
        alpha_add=1.0,
        alpha_inter=6.0,
        max_active_features=8,
        max_interactions=4,
        coef_tol=0.012,
        clip_quantile=0.02,
        knot_quantiles=(0.3, 0.7),
    ):
        self.alpha_add = alpha_add
        self.alpha_inter = alpha_inter
        self.max_active_features = max_active_features
        self.max_interactions = max_interactions
        self.coef_tol = coef_tol
        self.clip_quantile = clip_quantile
        self.knot_quantiles = knot_quantiles

    def _solve_ridge(self, Z, y, alpha):
        p = Z.shape[1]
        lhs = Z.T @ Z + alpha * np.eye(p)
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

    def _feature_basis(self, z, k1, k2):
        return np.column_stack(
            [
                z,
                np.maximum(0.0, z - k1),
                np.maximum(0.0, k1 - z),
                np.maximum(0.0, z - k2),
                np.maximum(0.0, k2 - z),
            ]
        )

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

        kq1, kq2 = self.knot_quantiles
        k1 = np.quantile(z, kq1, axis=0)
        k2 = np.quantile(z, kq2, axis=0)
        self.k1_ = k1
        self.k2_ = k2

        B = np.zeros((n, 5 * p), dtype=float)
        for j in range(p):
            B[:, 5 * j : 5 * (j + 1)] = self._feature_basis(z[:, j], k1[j], k2[j])

        self.y_mean_ = float(y.mean())
        yc = y - self.y_mean_
        coef_add_full = self._solve_ridge(B, yc, self.alpha_add)

        feat_strength = np.zeros(p, dtype=float)
        for j in range(p):
            c = coef_add_full[5 * j : 5 * (j + 1)]
            feat_strength[j] = np.sum(np.abs(c))

        keep = np.zeros(p, dtype=bool)
        if p > 0:
            top = np.argsort(feat_strength)[::-1][: min(self.max_active_features, p)]
            keep[top] = True
        keep &= feat_strength >= self.coef_tol
        self.active_features_ = np.where(keep)[0].tolist()

        kept_cols = []
        for j in self.active_features_:
            kept_cols.append(B[:, 5 * j : 5 * (j + 1)])
        Z_add = np.hstack(kept_cols) if kept_cols else np.zeros((n, 0), dtype=float)

        if Z_add.shape[1] > 0:
            coef_add = self._solve_ridge(Z_add, yc, self.alpha_add)
            residual = yc - Z_add @ coef_add
        else:
            residual = yc.copy()

        inter_pairs = []
        inter_cols = []
        if len(self.active_features_) >= 2 and self.max_interactions > 0:
            scores = []
            af = self.active_features_
            for a in range(len(af)):
                i = af[a]
                zi = z[:, i]
                for b in range(a + 1, len(af)):
                    j = af[b]
                    term = zi * z[:, j]
                    denom = np.sqrt(np.dot(term, term)) + 1e-12
                    score = np.abs(np.dot(term, residual)) / denom
                    scores.append((score, i, j, term))
            scores.sort(key=lambda t: t[0], reverse=True)
            for _, i, j, term in scores[: self.max_interactions]:
                inter_pairs.append((i, j))
                inter_cols.append(term.reshape(-1, 1))

        Z_inter = np.hstack(inter_cols) if inter_cols else np.zeros((n, 0), dtype=float)
        self.interaction_pairs_ = inter_pairs

        Z_all = np.hstack([Z_add, Z_inter])
        if Z_all.shape[1] > 0:
            penalties = np.concatenate(
                [
                    self.alpha_add * np.ones(Z_add.shape[1], dtype=float),
                    self.alpha_inter * np.ones(Z_inter.shape[1], dtype=float),
                ]
            )
            lhs = Z_all.T @ Z_all + np.diag(penalties)
            rhs = Z_all.T @ yc
            try:
                coef_all = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                coef_all = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            self.additive_coefs_ = coef_all[: Z_add.shape[1]]
            self.interaction_coefs_ = coef_all[Z_add.shape[1] :]
        else:
            self.additive_coefs_ = np.zeros(0, dtype=float)
            self.interaction_coefs_ = np.zeros(0, dtype=float)
        return self

    def _transform_active_additive(self, z):
        cols = []
        for j in self.active_features_:
            cols.append(self._feature_basis(z[:, j], self.k1_[j], self.k2_[j]))
        return np.hstack(cols) if cols else np.zeros((z.shape[0], 0), dtype=float)

    def _transform_interactions(self, z):
        cols = []
        for i, j in self.interaction_pairs_:
            cols.append((z[:, i] * z[:, j]).reshape(-1, 1))
        return np.hstack(cols) if cols else np.zeros((z.shape[0], 0), dtype=float)

    def predict(self, X):
        check_is_fitted(
            self,
            [
                "x_median_",
                "x_iqr_",
                "k1_",
                "k2_",
                "active_features_",
                "interaction_pairs_",
                "additive_coefs_",
                "interaction_coefs_",
                "y_mean_",
            ],
        )
        X = np.asarray(X, dtype=float)
        z = (X - self.x_median_) / self.x_iqr_
        if self.clip_quantile > 0 and self.z_lo_ is not None and self.z_hi_ is not None:
            z = np.minimum(np.maximum(z, self.z_lo_), self.z_hi_)

        Z_add = self._transform_active_additive(z)
        Z_inter = self._transform_interactions(z)
        pred = np.full(X.shape[0], self.y_mean_, dtype=float)
        if Z_add.shape[1] > 0:
            pred += Z_add @ self.additive_coefs_
        if Z_inter.shape[1] > 0:
            pred += Z_inter @ self.interaction_coefs_
        return pred

    def __str__(self):
        check_is_fitted(
            self,
            [
                "active_features_",
                "interaction_pairs_",
                "additive_coefs_",
                "interaction_coefs_",
                "y_mean_",
                "k1_",
                "k2_",
            ],
        )
        lines = [
            "Sparse Dual-Knot Additive + Limited Interactions Regressor",
            "z_j = clip((x_j - median_j) / iqr_j)",
            "f_j(z_j) = a*z_j + b*max(0,z_j-k1) + c*max(0,k1-z_j) + d*max(0,z_j-k2) + e*max(0,k2-z_j)",
            "y = intercept + sum_j f_j(z_j) + sum_(i,j) g_ij * (z_i*z_j)",
            "",
            f"intercept: {self.y_mean_:+.4f}",
            f"active_features: {len(self.active_features_)} / {self.n_features_in_}",
            f"interaction_terms: {len(self.interaction_pairs_)}",
            "",
            "Active feature pieces:",
        ]
        for idx, j in enumerate(self.active_features_):
            c = self.additive_coefs_[5 * idx : 5 * (idx + 1)]
            lines.append(
                f"  x{j}: k1={self.k1_[j]:+.3f}, k2={self.k2_[j]:+.3f}, "
                f"[{c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f}, {c[3]:+.4f}, {c[4]:+.4f}]"
            )
        if not self.active_features_:
            lines.append("  none")

        lines.append("")
        lines.append("Interaction terms:")
        if self.interaction_pairs_:
            for (i, j), c in zip(self.interaction_pairs_, self.interaction_coefs_):
                lines.append(f"  x{i}*x{j}: {c:+.4f}")
        else:
            lines.append("  none")
        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseDualKnotInteractionRegressor.__module__ = "interpretable_regressor"

# Update the model shorthand name and description below to reflect the class above and any changes you make to it.
# The shorthand name should be unique across all experiments (it is used to identify rows in the results CSV files)
# The description should briefly summarize what this experiment tried.
model_shorthand_name = "SparseDualKnotInt_v1"
model_description = "Sparse dual-knot additive model with residual-screened pairwise interactions and stronger interaction shrinkage"
model_defs = [(model_shorthand_name, SparseDualKnotInteractionRegressor())]


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
