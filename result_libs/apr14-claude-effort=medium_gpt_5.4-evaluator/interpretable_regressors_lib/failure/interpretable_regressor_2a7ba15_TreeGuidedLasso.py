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
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from interp_eval import run_all_interp_tests, ALL_TESTS, HARD_TESTS, INSIGHT_TESTS
from performance_eval import RESULTS_DIR, upsert_overall_results, evaluate_all_regressors, compute_rank_scores
from visualize import plot_interp_vs_performance

# ---------------------------------------------------------------------------
# Interpretable Regressor (edit this, everything in this class is fair game)
# ---------------------------------------------------------------------------


class TreeGuidedLassoRegressor(BaseEstimator, RegressorMixin):
    """
    Lasso regression augmented with tree-derived threshold features.

    1. Fit a decision tree to discover important split thresholds
    2. Create binary indicator features: I(x_j > threshold) for each split
    3. Combine original features + threshold indicators
    4. Fit LassoCV for sparse selection

    The result captures both linear effects AND threshold effects,
    presented as a clean linear equation:
      y = w1*x0 + w2*x1 + w3*I(x0>0.5) + w4*I(x1>1.2) + ...

    Each I(...) is just 1 if the condition is true, 0 otherwise.
    This is strictly more expressive than plain linear models but
    uses the same readable equation format.
    """

    def __init__(self, tree_max_depth=4, max_threshold_features=20):
        self.tree_max_depth = tree_max_depth
        self.max_threshold_features = max_threshold_features

    def fit(self, X, y):
        n_samples, n_feat = X.shape
        self.n_features_ = n_feat

        # Step 1: Fit a tree to discover thresholds
        tree = DecisionTreeRegressor(
            max_depth=self.tree_max_depth,
            min_samples_leaf=max(5, n_samples // 20),
            random_state=42,
        )
        tree.fit(X, y)

        # Extract all split thresholds from the tree
        thresholds = []
        tree_ = tree.tree_
        for node_id in range(tree_.node_count):
            if tree_.children_left[node_id] != -1:  # internal node
                feat = tree_.feature[node_id]
                thresh = tree_.threshold[node_id]
                thresholds.append((feat, thresh))

        # Limit number of threshold features
        if len(thresholds) > self.max_threshold_features:
            thresholds = thresholds[:self.max_threshold_features]

        self.thresholds_ = thresholds

        # Step 2: Build augmented feature matrix
        X_aug = self._augment(X)

        # Build feature names
        self.aug_names_ = [f"x{i}" for i in range(n_feat)]
        for feat, thresh in self.thresholds_:
            self.aug_names_.append(f"I(x{feat}>{thresh:.2f})")

        # Step 3: Fit LassoCV
        self.lasso_ = LassoCV(cv=3, max_iter=5000, random_state=42)
        self.lasso_.fit(X_aug, y)

        return self

    def _augment(self, X):
        """Augment X with binary threshold features."""
        indicators = []
        for feat, thresh in self.thresholds_:
            indicators.append((X[:, feat] > thresh).astype(float))
        if indicators:
            return np.column_stack([X] + indicators)
        return X

    def predict(self, X):
        check_is_fitted(self, "lasso_")
        X_aug = self._augment(X)
        return self.lasso_.predict(X_aug)

    def __str__(self):
        check_is_fitted(self, "lasso_")
        coefs = self.lasso_.coef_
        intercept = self.lasso_.intercept_
        alpha = self.lasso_.alpha_

        # Collect active terms
        active = [(self.aug_names_[i], coefs[i])
                  for i in range(len(coefs)) if abs(coefs[i]) > 1e-8]

        lines = [f"Sparse Linear Model with Threshold Features (L1, α={alpha:.4g}):"]

        if active:
            # Build equation
            equation = f"{intercept:.4f}"
            for name, c in sorted(active, key=lambda x: -abs(x[1])):
                equation += f" {c:+.4f}*{name}"
            lines.append(f"  y = {equation}")
            lines.append("")
            lines.append(f"  Active terms ({len(active)} non-zero, sorted by magnitude):")
            for name, c in sorted(active, key=lambda x: -abs(x[1])):
                lines.append(f"    {c:+.4f} * {name}")

            # Explain indicator features
            has_indicators = any("I(" in name for name, _ in active)
            if has_indicators:
                lines.append("")
                lines.append("  Note: I(condition) = 1 if condition is true, 0 otherwise")
        else:
            lines.append(f"  y = {intercept:.4f} (all terms zeroed)")

        lines.append(f"  intercept: {intercept:.4f}")

        # Unused features
        used = set()
        for name, _ in active:
            for i in range(self.n_features_):
                if f"x{i}" in name:
                    used.add(f"x{i}")
        unused = [f"x{i}" for i in range(self.n_features_) if f"x{i}" not in used]
        if unused:
            lines.append(f"  Features not used: {', '.join(unused)}")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
TreeGuidedLassoRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "TreeGuidedLasso"
model_description = "Lasso with tree-derived threshold indicator features, captures nonlinearity in linear equation format"
model_defs = [(model_shorthand_name, TreeGuidedLassoRegressor())]


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

    perf_csv = os.path.join(RESULTS_DIR, "performance_results.csv")
    perf_fields = ["dataset", "model", "rmse", "rank"]

    existing_perf = []
    if os.path.exists(perf_csv):
        with open(perf_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("model") != model_name:
                    existing_perf.append(row)

    for ds_name, model_rmses in dataset_rmses.items():
        rmse_val = model_rmses.get(model_name, float("nan"))
        existing_perf.append({
            "dataset": ds_name,
            "model": model_name,
            "rmse": "" if np.isnan(rmse_val) else f"{rmse_val:.6f}",
            "rank": "",
        })

    by_dataset = defaultdict(list)
    for row in existing_perf:
        by_dataset[row["dataset"]].append(row)

    for ds_name, rows in by_dataset.items():
        valid = [(r, float(r["rmse"])) for r in rows if r["rmse"] not in ("", None)]
        valid.sort(key=lambda x: x[1])
        for rank_idx, (r, _) in enumerate(valid, 1):
            r["rank"] = rank_idx
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
