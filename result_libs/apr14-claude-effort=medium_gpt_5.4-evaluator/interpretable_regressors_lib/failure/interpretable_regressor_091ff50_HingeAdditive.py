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
from sklearn.linear_model import RidgeCV
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class HingeAdditiveRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse additive hinge regressor (simplified MARS).

    Builds a model of the form:
      y = intercept + w1*max(0, x_j1 - t1) + w2*max(0, t2 - x_j2) + w3*x_j3 + ...

    Uses forward stepwise selection to greedily add the best hinge or linear
    basis function at each step, then refits all coefficients with ridge regression.

    Hinge functions max(0, x-t) and max(0, t-x) capture thresholds and
    nonlinearity while remaining easy for an LLM to trace step by step.
    """

    def __init__(self, max_terms=15, n_knot_candidates=10):
        self.max_terms = max_terms
        self.n_knot_candidates = n_knot_candidates

    def fit(self, X, y):
        n_samples, n_feat = X.shape
        self.n_features_ = n_feat

        # Generate candidate basis functions
        # For each feature: linear term, plus hinge terms at quantile knots
        candidates = []
        for j in range(n_feat):
            col = X[:, j]
            # Linear term
            candidates.append(('linear', j, None, col.copy()))

            # Hinge terms at quantile knots
            knots = np.percentile(col, np.linspace(10, 90, self.n_knot_candidates))
            knots = np.unique(knots)
            for t in knots:
                # max(0, x - t): activates above threshold
                h_plus = np.maximum(0, col - t)
                if np.std(h_plus) > 1e-10:
                    candidates.append(('hinge+', j, t, h_plus))
                # max(0, t - x): activates below threshold
                h_minus = np.maximum(0, t - col)
                if np.std(h_minus) > 1e-10:
                    candidates.append(('hinge-', j, t, h_minus))

        # Forward stepwise selection: greedily pick the best term
        selected_indices = []
        residual = y - np.mean(y)
        used = set()

        for step in range(min(self.max_terms, len(candidates))):
            best_reduction = -1
            best_idx = -1
            ss_residual = np.sum(residual ** 2)

            for i, (kind, feat, knot, basis) in enumerate(candidates):
                if i in used:
                    continue
                # Simple correlation-based selection
                corr = np.dot(basis, residual)
                var_basis = np.dot(basis, basis)
                if var_basis < 1e-10:
                    continue
                # Reduction in RSS from adding this term
                reduction = corr ** 2 / var_basis
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_idx = i

            if best_idx < 0 or best_reduction < 1e-8 * ss_residual:
                break

            selected_indices.append(best_idx)
            used.add(best_idx)

            # Update residual using simple projection
            basis = candidates[best_idx][3]
            coef = np.dot(basis, residual) / np.dot(basis, basis)
            residual = residual - coef * basis

        # Build the selected basis matrix and refit with ridge
        self.terms_ = []
        if not selected_indices:
            # Fallback: just use intercept
            self.intercept_ = float(np.mean(y))
            self.coefs_ = np.array([])
            return self

        B = np.column_stack([candidates[i][3] for i in selected_indices])
        for i in selected_indices:
            kind, feat, knot, _ = candidates[i]
            self.terms_.append((kind, feat, knot))

        ridge = RidgeCV(cv=3)
        ridge.fit(B, y)
        self.coefs_ = ridge.coef_
        self.intercept_ = ridge.intercept_

        # Store knot info for transform
        return self

    def predict(self, X):
        check_is_fitted(self, "coefs_")
        if len(self.coefs_) == 0:
            return np.full(X.shape[0], self.intercept_)

        B = self._build_basis(X)
        return B @ self.coefs_ + self.intercept_

    def _build_basis(self, X):
        """Build the basis matrix for the selected terms."""
        cols = []
        for kind, feat, knot in self.terms_:
            col = X[:, feat]
            if kind == 'linear':
                cols.append(col)
            elif kind == 'hinge+':
                cols.append(np.maximum(0, col - knot))
            elif kind == 'hinge-':
                cols.append(np.maximum(0, knot - col))
        return np.column_stack(cols)

    def __str__(self):
        check_is_fitted(self, "coefs_")
        names = [f"x{i}" for i in range(self.n_features_)]

        lines = ["Sparse Hinge Additive Model:"]

        if len(self.coefs_) == 0:
            lines.append(f"  y = {self.intercept_:.4f}")
            return "\n".join(lines)

        # Build equation
        eq_parts = []
        term_details = []
        for (kind, feat, knot), coef in zip(self.terms_, self.coefs_):
            if abs(coef) < 1e-8:
                continue
            fname = names[feat]
            if kind == 'linear':
                eq_parts.append(f"{coef:+.4f}*{fname}")
                term_details.append((fname, coef, f"{fname}"))
            elif kind == 'hinge+':
                eq_parts.append(f"{coef:+.4f}*max(0, {fname}-{knot:.2f})")
                term_details.append((fname, coef, f"max(0, {fname}-{knot:.2f})"))
            elif kind == 'hinge-':
                eq_parts.append(f"{coef:+.4f}*max(0, {knot:.2f}-{fname})")
                term_details.append((fname, coef, f"max(0, {knot:.2f}-{fname})"))

        equation = f"{self.intercept_:.4f} " + " ".join(eq_parts)
        lines.append(f"  y = {equation}")
        lines.append("")
        lines.append(f"  Terms ({len(term_details)} active):")
        for fname, coef, term_str in sorted(term_details, key=lambda x: -abs(x[1])):
            lines.append(f"    {coef:+.4f} * {term_str}")
        lines.append(f"  intercept: {self.intercept_:.4f}")

        # Show which features are used
        used_features = sorted(set(fname for fname, _, _ in term_details))
        unused = [f"x{i}" for i in range(self.n_features_) if f"x{i}" not in used_features]
        if unused:
            lines.append(f"  Unused features (no effect): {', '.join(unused)}")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
HingeAdditiveRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "HingeAdditive"
model_description = "Sparse additive hinge model (simplified MARS) with forward stepwise selection and ridge refit"
model_defs = [(model_shorthand_name, HingeAdditiveRegressor())]


# ---------------------------------------------------------------------------
# Evaluation (do not edit anything below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default='gpt-5.4',
                        help="LLM checkpoint for interpretability tests (default: gpt-5.4)")
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
