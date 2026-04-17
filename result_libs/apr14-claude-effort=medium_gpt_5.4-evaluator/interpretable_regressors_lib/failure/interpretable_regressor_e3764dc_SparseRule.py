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
from sklearn.ensemble import GradientBoostingRegressor
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


def _extract_rules_from_tree(tree, feature_names):
    """Extract all root-to-leaf rules from a decision tree."""
    tree_ = tree.tree_
    rules = []

    def _recurse(node, conditions):
        if tree_.children_left[node] == -1:
            # Leaf node — emit rule
            rules.append(list(conditions))
            return
        feat = tree_.feature[node]
        thresh = tree_.threshold[node]
        fname = feature_names[feat]

        # Left child: feat <= thresh
        conditions.append((fname, '<=', thresh))
        _recurse(tree_.children_left[node], conditions)
        conditions.pop()

        # Right child: feat > thresh
        conditions.append((fname, '>', thresh))
        _recurse(tree_.children_right[node], conditions)
        conditions.pop()

    _recurse(0, [])
    return rules


class SparseRuleRegressor(BaseEstimator, RegressorMixin):
    """
    Sparse rule-based regressor.

    1. Fit a GBM ensemble to generate diverse decision trees
    2. Extract all root-to-leaf rules as binary features
    3. Also include raw linear features for each variable
    4. Fit LassoCV to select the most useful rules + linear terms
    5. Present as: y = intercept + w1*rule1 + w2*rule2 + w3*x0 + ...

    Each rule is a conjunction of conditions (e.g., "x0 > 0.5 AND x1 <= 1.2")
    that evaluates to 1 (True) or 0 (False). The LLM just checks each rule
    and multiplies by the weight.

    Combines the nonlinear/interaction power of trees with the sparsity
    and readability of Lasso.
    """

    def __init__(self, n_trees=50, max_depth=3, max_rules=200):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_rules = max_rules

    def fit(self, X, y):
        n_samples, n_feat = X.shape
        self.n_features_ = n_feat
        feature_names = [f"x{i}" for i in range(n_feat)]

        # Step 1: Fit GBM to get diverse trees
        gbm = GradientBoostingRegressor(
            n_estimators=self.n_trees,
            max_depth=self.max_depth,
            learning_rate=0.1,
            random_state=42,
            subsample=0.8,
        )
        gbm.fit(X, y)

        # Step 2: Extract rules from all trees
        all_rules = []
        for tree_arr in gbm.estimators_:
            tree = tree_arr[0]
            rules = _extract_rules_from_tree(tree, feature_names)
            all_rules.extend(rules)

        # Deduplicate rules by their string representation
        seen = set()
        unique_rules = []
        for rule in all_rules:
            key = tuple((f, op, round(t, 6)) for f, op, t in rule)
            if key not in seen and len(rule) > 0:
                seen.add(key)
                unique_rules.append(rule)

        # Limit number of rules
        if len(unique_rules) > self.max_rules:
            # Select rules by correlation with y
            R = self._evaluate_rules(X, unique_rules)
            corrs = np.array([abs(np.corrcoef(R[:, j], y)[0, 1])
                              if np.std(R[:, j]) > 1e-10 else 0.0
                              for j in range(R.shape[1])])
            top_idx = np.argsort(corrs)[-self.max_rules:]
            unique_rules = [unique_rules[i] for i in sorted(top_idx)]

        self.rules_ = unique_rules

        # Step 3: Build feature matrix: rules + linear terms
        R = self._evaluate_rules(X, self.rules_)
        # Add linear features
        X_combined = np.hstack([X, R])

        # Feature names for combined
        self.rule_names_ = []
        for rule in self.rules_:
            parts = [f"{f}{op}{t:.2f}" for f, op, t in rule]
            self.rule_names_.append(" AND ".join(parts))

        # Step 4: Fit LassoCV
        self.lasso_ = LassoCV(cv=3, max_iter=5000, random_state=42)
        self.lasso_.fit(X_combined, y)

        return self

    def _evaluate_rules(self, X, rules):
        """Evaluate all rules on X, returning binary matrix."""
        n = X.shape[0]
        feature_idx = {}
        for i in range(self.n_features_):
            feature_idx[f"x{i}"] = i

        R = np.zeros((n, len(rules)))
        for j, rule in enumerate(rules):
            mask = np.ones(n, dtype=bool)
            for fname, op, thresh in rule:
                fi = feature_idx[fname]
                if op == '<=':
                    mask &= X[:, fi] <= thresh
                else:
                    mask &= X[:, fi] > thresh
            R[:, j] = mask.astype(float)
        return R

    def predict(self, X):
        check_is_fitted(self, "lasso_")
        R = self._evaluate_rules(X, self.rules_)
        X_combined = np.hstack([X, R])
        return self.lasso_.predict(X_combined)

    def __str__(self):
        check_is_fitted(self, "lasso_")
        coefs = self.lasso_.coef_
        intercept = self.lasso_.intercept_
        alpha = self.lasso_.alpha_
        n_feat = self.n_features_

        # Split coefficients into linear and rule parts
        linear_coefs = coefs[:n_feat]
        rule_coefs = coefs[n_feat:]

        # Collect active terms
        active_linear = [(f"x{i}", linear_coefs[i]) for i in range(n_feat)
                         if abs(linear_coefs[i]) > 1e-8]
        active_rules = [(self.rule_names_[j], rule_coefs[j]) for j in range(len(rule_coefs))
                        if abs(rule_coefs[j]) > 1e-8]

        lines = [f"Sparse Rule-Based Linear Model (L1 regularization, α={alpha:.4g}):"]
        lines.append(f"  intercept: {intercept:.4f}")
        lines.append("")

        if active_linear:
            lines.append(f"  Linear terms ({len(active_linear)} features):")
            for name, c in sorted(active_linear, key=lambda x: -abs(x[1])):
                lines.append(f"    {c:+.4f} * {name}")

        if active_rules:
            lines.append(f"\n  Rule terms ({len(active_rules)} rules, each evaluates to 1 if ALL conditions met, else 0):")
            for name, c in sorted(active_rules, key=lambda x: -abs(x[1])):
                lines.append(f"    {c:+.4f} * [{name}]")

        if not active_linear and not active_rules:
            lines.append(f"  y = {intercept:.4f} (all terms zeroed)")

        # Show unused features
        used_feats = set()
        for name, _ in active_linear:
            used_feats.add(name)
        for name, _ in active_rules:
            for part in name.split(" AND "):
                for i in range(n_feat):
                    if f"x{i}" in part:
                        used_feats.add(f"x{i}")
        unused = [f"x{i}" for i in range(n_feat) if f"x{i}" not in used_feats]
        if unused:
            lines.append(f"\n  Features not used: {', '.join(unused)}")

        lines.append("\n  To predict: evaluate each rule (1=all conditions True, 0=otherwise),")
        lines.append("  multiply each term by its weight, and sum with intercept.")

        return "\n".join(lines)


# Make class picklable when script is run as __main__ (required for joblib caching/parallel)
import sys as _sys
_sys.modules.setdefault("interpretable_regressor", _sys.modules[__name__])
SparseRuleRegressor.__module__ = "interpretable_regressor"

model_shorthand_name = "SparseRule"
model_description = "Sparse rule-based regressor: GBM-extracted rules + linear terms, Lasso-selected for sparsity"
model_defs = [(model_shorthand_name, SparseRuleRegressor())]


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

    # --- Upsert performance_results.csv and recompute ranks ---
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
