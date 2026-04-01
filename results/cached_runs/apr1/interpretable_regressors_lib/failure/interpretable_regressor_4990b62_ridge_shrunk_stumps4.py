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


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
RidgePlusStumpsRegressor.__module__ = "interpretable_regressor"
RidgePlusTreeRegressor.__module__ = "interpretable_regressor"


# ---------------------------------------------------------------------------
# Evaluation (do not edit any of the evaluation functions, only the names and model descriptions below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    model_defs = [("RidgeShrunkStumps4", RidgePlusStumpsRegressor(n_stumps=4, stump_lr=0.5))]

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
    model_name = "RidgeShrunkStumps4"
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
        "model_name":                         "RidgeShrunkStumps4",
        "description":                        "RidgeCV + 4 greedy stumps with 0.5 learning rate on residuals",
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

