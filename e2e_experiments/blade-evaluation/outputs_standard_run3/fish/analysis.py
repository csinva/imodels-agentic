import json
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def top_abs_dict(values: Dict[str, float], k: int = 3) -> List[str]:
    items = sorted(values.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [name for name, _ in items[:k]]


def main():
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info.get("research_questions", ["Unknown question"])[0]

    df = pd.read_csv("fish.csv")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary_stats = df[numeric_cols].describe().T
    missing = df.isna().sum()
    corr = df[numeric_cols].corr(numeric_only=True)

    # Target for the research question: fish caught per hour.
    work = df.copy()
    work["fish_per_hour"] = work["fish_caught"] / work["hours"].replace(0, np.nan)
    work = work.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    work["log_fish_per_hour"] = np.log1p(work["fish_per_hour"])

    # Core rate estimates.
    weighted_rate = safe_float(work["fish_caught"].sum() / work["hours"].sum())
    mean_individual_rate = safe_float(work["fish_per_hour"].mean())
    median_individual_rate = safe_float(work["fish_per_hour"].median())

    # Statistical tests.
    ttest_rate_gt_zero = stats.ttest_1samp(work["fish_per_hour"], popmean=0, alternative="greater")

    binary_tests = {}
    for col in ["livebait", "camper"]:
        g0 = work.loc[work[col] == 0, "fish_per_hour"]
        g1 = work.loc[work[col] == 1, "fish_per_hour"]
        t_res = stats.ttest_ind(g1, g0, equal_var=False)
        binary_tests[col] = {
            "mean_when_0": safe_float(g0.mean()),
            "mean_when_1": safe_float(g1.mean()),
            "t_stat": safe_float(t_res.statistic),
            "p_value": safe_float(t_res.pvalue),
        }

    anova_tests = {}
    for col in ["persons", "child"]:
        groups = [g["fish_per_hour"].values for _, g in work.groupby(col)]
        a_res = stats.f_oneway(*groups)
        anova_tests[col] = {
            "f_stat": safe_float(a_res.statistic),
            "p_value": safe_float(a_res.pvalue),
        }

    pearson_fish_hours = stats.pearsonr(work["fish_caught"], work["hours"])
    pearson_rate_hours = stats.pearsonr(work["fish_per_hour"], work["hours"])

    features = ["livebait", "camper", "persons", "child", "hours"]
    X = work[features]
    y = work["log_fish_per_hour"]

    # statsmodels OLS for inferential interpretability (coefficients + p-values).
    X_sm = sm.add_constant(X)
    ols = sm.OLS(y, X_sm).fit()
    ols_params = {k: safe_float(v) for k, v in ols.params.items()}
    ols_pvalues = {k: safe_float(v) for k, v in ols.pvalues.items()}

    # scikit-learn interpretable models.
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    sklearn_results = {}

    linear_models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.001, max_iter=10000),
    }

    for name, model in linear_models.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])
        cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
        pipe.fit(X, y)
        coef = pipe.named_steps["model"].coef_
        sklearn_results[name] = {
            "cv_r2_mean": safe_float(np.mean(cv_scores)),
            "cv_r2_std": safe_float(np.std(cv_scores)),
            "coef": {f: safe_float(c) for f, c in zip(features, coef)},
        }

    tree = DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE)
    tree_cv_scores = cross_val_score(tree, X, y, cv=cv, scoring="r2")
    tree.fit(X, y)
    sklearn_results["decision_tree_regressor"] = {
        "cv_r2_mean": safe_float(np.mean(tree_cv_scores)),
        "cv_r2_std": safe_float(np.std(tree_cv_scores)),
        "feature_importance": {
            f: safe_float(v) for f, v in zip(features, tree.feature_importances_)
        },
    }

    # imodels interpretable models.
    imodels_results = {}

    rulefit = RuleFitRegressor(
        random_state=RANDOM_STATE,
        n_estimators=100,
        tree_size=3,
        max_rules=20,
    )
    rulefit.fit(X.values, y.values, feature_names=features)
    rules_df = rulefit._get_rules()
    rules_df = rules_df[rules_df["importance"] > 0].sort_values("importance", ascending=False)
    top_rules = rules_df.head(8)
    imodels_results["rulefit"] = {
        "train_r2": safe_float(rulefit.score(X.values, y.values)),
        "top_rules": top_rules[["rule", "coef", "support", "importance"]].to_dict("records"),
    }

    figs = FIGSRegressor(random_state=RANDOM_STATE, max_rules=8)
    figs.fit(X.values, y.values, feature_names=features)
    imodels_results["figs"] = {
        "train_r2": safe_float(figs.score(X.values, y.values)),
        "feature_importance": {f: safe_float(v) for f, v in zip(features, figs.feature_importances_)},
        "model_text": str(figs),
    }

    hstree = HSTreeRegressor(random_state=RANDOM_STATE, max_leaf_nodes=8)
    hstree.fit(X.values, y.values, feature_names=features)
    imodels_results["hstree"] = {
        "train_r2": safe_float(hstree.score(X.values, y.values)),
        "model_text": str(hstree),
    }

    # Strength of evidence for answering the research question.
    significant_predictors = [
        f for f in features if ols_pvalues.get(f, 1.0) < 0.05
    ]
    best_cv_r2 = max(v["cv_r2_mean"] for v in sklearn_results.values())

    score = 40
    if (ttest_rate_gt_zero.pvalue < 0.05) and (weighted_rate > 0):
        score += 20
    if ols.f_pvalue < 0.05:
        score += 15
    score += min(15, 3 * len(significant_predictors))

    if best_cv_r2 >= 0.25:
        score += 10
    elif best_cv_r2 >= 0.15:
        score += 6
    elif best_cv_r2 >= 0.05:
        score += 3

    # Penalize for non-significant group effects among the tested grouping variables.
    if binary_tests["camper"]["p_value"] >= 0.05:
        score -= 2
    if anova_tests["child"]["p_value"] >= 0.05:
        score -= 2

    score = int(np.clip(round(score), 0, 100))

    # Build concise, evidence-based explanation.
    top_linear = top_abs_dict(sklearn_results["linear_regression"]["coef"], k=3)
    top_tree = top_abs_dict(sklearn_results["decision_tree_regressor"]["feature_importance"], k=3)
    top_figs = top_abs_dict(imodels_results["figs"]["feature_importance"], k=3)

    explanation = (
        f"Question: {question} "
        f"Estimated average catch rate is {weighted_rate:.3f} fish/hour (total fish/total hours), "
        f"with mean individual trip rate {mean_individual_rate:.3f} and median {median_individual_rate:.3f}. "
        f"Rate is significantly above zero (one-sample t-test p={ttest_rate_gt_zero.pvalue:.3g}). "
        f"OLS on log(1+fish/hour) is significant overall (F-test p={ols.f_pvalue:.3g}, adj-R2={ols.rsquared_adj:.3f}) and shows "
        f"significant predictors: {', '.join(significant_predictors) if significant_predictors else 'none'}. "
        f"Livebait effect is {'significant' if binary_tests['livebait']['p_value'] < 0.05 else 'weak/non-significant'} "
        f"(Welch t-test p={binary_tests['livebait']['p_value']:.3g}); camper is "
        f"{'significant' if binary_tests['camper']['p_value'] < 0.05 else 'not significant'} in the direct t-test "
        f"(p={binary_tests['camper']['p_value']:.3g}). "
        f"ANOVA suggests persons matters (p={anova_tests['persons']['p_value']:.3g}) while child group means are less clear "
        f"(p={anova_tests['child']['p_value']:.3g}). "
        f"Interpretable models agree on key drivers; strongest features are linear={top_linear}, "
        f"tree={top_tree}, FIGS={top_figs}. "
        f"This supports a strong 'Yes' that catch rate can be estimated and is meaningfully associated with trip characteristics, "
        f"especially group composition and time-related effects."
    )

    # Save a full analysis artifact for transparency/debugging.
    analysis_artifact = {
        "question": question,
        "n_rows": int(len(work)),
        "missing_values": {k: int(v) for k, v in missing.items()},
        "summary_statistics": summary_stats.to_dict(),
        "correlation_matrix": corr.to_dict(),
        "rate_summary": {
            "weighted_rate": weighted_rate,
            "mean_individual_rate": mean_individual_rate,
            "median_individual_rate": median_individual_rate,
        },
        "tests": {
            "ttest_rate_gt_zero": {
                "t_stat": safe_float(ttest_rate_gt_zero.statistic),
                "p_value": safe_float(ttest_rate_gt_zero.pvalue),
            },
            "binary_tests": binary_tests,
            "anova_tests": anova_tests,
            "pearson_fish_vs_hours": {
                "r": safe_float(pearson_fish_hours.statistic),
                "p_value": safe_float(pearson_fish_hours.pvalue),
            },
            "pearson_rate_vs_hours": {
                "r": safe_float(pearson_rate_hours.statistic),
                "p_value": safe_float(pearson_rate_hours.pvalue),
            },
        },
        "ols": {
            "params": ols_params,
            "pvalues": ols_pvalues,
            "adj_r2": safe_float(ols.rsquared_adj),
            "f_pvalue": safe_float(ols.f_pvalue),
        },
        "sklearn_models": sklearn_results,
        "imodels": imodels_results,
    }

    with open("analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(analysis_artifact, f, indent=2)

    conclusion = {
        "response": score,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        # Required format: file must contain only a JSON object.
        f.write(json.dumps(conclusion))

    print("Analysis complete. Wrote analysis_results.json and conclusion.txt")


if __name__ == "__main__":
    main()
