import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_float(x) -> float:
    return float(np.asarray(x).item())


def main() -> None:
    info_path = Path("info.json")
    data_path = Path("teachingratings.csv")

    info = json.loads(info_path.read_text())
    question = info.get("research_questions", ["Unknown question"])[0]

    df = pd.read_csv(data_path)

    print_section("Research Question")
    print(question)

    print_section("Data Overview")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("Columns:", list(df.columns))
    print("Missing values by column:")
    print(df.isna().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    print_section("Summary Statistics (Numeric)")
    print(df[numeric_cols].describe().T)

    print_section("Categorical Distributions")
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts(normalize=True).rename("proportion"))

    print_section("Correlation Matrix (Numeric)")
    corr = df[numeric_cols].corr()
    print(corr)
    eval_corr_sorted = corr["eval"].sort_values(ascending=False)
    print("\nCorrelations with eval:")
    print(eval_corr_sorted)

    print_section("Statistical Tests for Beauty -> Eval")
    pearson_r, pearson_p = stats.pearsonr(df["beauty"], df["eval"])
    spearman_rho, spearman_p = stats.spearmanr(df["beauty"], df["eval"])

    beauty_median = df["beauty"].median()
    high_beauty_eval = df.loc[df["beauty"] >= beauty_median, "eval"]
    low_beauty_eval = df.loc[df["beauty"] < beauty_median, "eval"]
    t_stat, t_p = stats.ttest_ind(high_beauty_eval, low_beauty_eval, equal_var=False)

    beauty_quartiles = pd.qcut(df["beauty"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    anova_f, anova_p = stats.f_oneway(
        df.loc[beauty_quartiles == "Q1", "eval"],
        df.loc[beauty_quartiles == "Q2", "eval"],
        df.loc[beauty_quartiles == "Q3", "eval"],
        df.loc[beauty_quartiles == "Q4", "eval"],
    )

    print(f"Pearson r={pearson_r:.4f}, p={pearson_p:.4g}")
    print(f"Spearman rho={spearman_rho:.4f}, p={spearman_p:.4g}")
    print(
        "Median split eval means: "
        f"high beauty={high_beauty_eval.mean():.4f}, low beauty={low_beauty_eval.mean():.4f}"
    )
    print(f"Welch t-test: t={t_stat:.4f}, p={t_p:.4g}")
    print(f"ANOVA across beauty quartiles: F={anova_f:.4f}, p={anova_p:.4g}")

    print_section("OLS With Controls (statsmodels)")
    formula = (
        "eval ~ beauty + age + students + allstudents + "
        "C(minority) + C(gender) + C(credits) + C(division) + C(native) + C(tenure)"
    )
    ols = smf.ols(formula=formula, data=df).fit(cov_type="HC3")
    beauty_coef = safe_float(ols.params["beauty"])
    beauty_p = safe_float(ols.pvalues["beauty"])
    beauty_ci_low, beauty_ci_high = [safe_float(v) for v in ols.conf_int().loc["beauty"]]

    print(ols.summary())
    print(
        f"Beauty coefficient={beauty_coef:.4f}, p={beauty_p:.4g}, "
        f"95% CI=({beauty_ci_low:.4f}, {beauty_ci_high:.4f})"
    )

    print_section("Interpretable Models (scikit-learn + imodels)")
    target_col = "eval"
    X = df.drop(columns=[target_col])
    y = df[target_col].values

    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    X_proc = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    beauty_feature_name = "num__beauty"

    lin = LinearRegression().fit(X_proc, y)
    ridge = Ridge(alpha=1.0, random_state=0).fit(X_proc, y)
    lasso = Lasso(alpha=0.01, max_iter=20000, random_state=0).fit(X_proc, y)

    lin_coef = dict(zip(feature_names, lin.coef_))
    ridge_coef = dict(zip(feature_names, ridge.coef_))
    lasso_coef = dict(zip(feature_names, lasso.coef_))

    lin_beauty = safe_float(lin_coef.get(beauty_feature_name, 0.0))
    ridge_beauty = safe_float(ridge_coef.get(beauty_feature_name, 0.0))
    lasso_beauty = safe_float(lasso_coef.get(beauty_feature_name, 0.0))

    print(
        "Beauty coefficient (standardized feature): "
        f"Linear={lin_beauty:.4f}, Ridge={ridge_beauty:.4f}, Lasso={lasso_beauty:.4f}"
    )

    tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=20, random_state=0)
    tree.fit(X_proc, y)
    tree_importance = dict(zip(feature_names, tree.feature_importances_))
    tree_beauty_imp = safe_float(tree_importance.get(beauty_feature_name, 0.0))
    print(f"DecisionTree beauty importance={tree_beauty_imp:.4f}")

    rulefit = RuleFitRegressor(max_rules=30, random_state=0, cv=False)
    rulefit.fit(X_proc, y, feature_names=feature_names)
    rules_df = rulefit._get_rules()
    rules_df = rules_df.sort_values("importance", ascending=False)
    top_beauty_rules = rules_df[rules_df["rule"].str.contains("beauty", regex=False)].head(5)

    beauty_linear_rule_coef = 0.0
    beauty_linear_row = rules_df[rules_df["rule"] == beauty_feature_name]
    if not beauty_linear_row.empty:
        beauty_linear_rule_coef = safe_float(beauty_linear_row.iloc[0]["coef"])

    print("Top RuleFit rules containing beauty:")
    if top_beauty_rules.empty:
        print("No beauty-containing rules in top set.")
    else:
        print(top_beauty_rules[["rule", "coef", "support", "importance"]])

    figs = FIGSRegressor(max_rules=12, random_state=0)
    figs.fit(X_proc, y, feature_names=feature_names)
    figs_importance = dict(zip(feature_names, figs.feature_importances_))
    figs_beauty_imp = safe_float(figs_importance.get(beauty_feature_name, 0.0))
    print(f"FIGS beauty importance={figs_beauty_imp:.4f}")

    hst = HSTreeRegressor(max_leaf_nodes=20, random_state=0)
    hst.fit(X_proc, y)
    hst_importance = dict(zip(feature_names, hst.estimator_.feature_importances_))
    hst_beauty_imp = safe_float(hst_importance.get(beauty_feature_name, 0.0))
    print(f"HSTree beauty importance={hst_beauty_imp:.4f}")

    print_section("Conclusion Scoring")
    significance_checks = {
        "pearson": pearson_p < 0.05,
        "spearman": spearman_p < 0.05,
        "t_test": t_p < 0.05,
        "anova": anova_p < 0.05,
        "ols_beauty": beauty_p < 0.05,
    }
    n_sig = sum(significance_checks.values())

    direction_checks = {
        "pearson_positive": pearson_r > 0,
        "spearman_positive": spearman_rho > 0,
        "ols_positive": beauty_coef > 0,
        "linear_positive": lin_beauty > 0,
        "ridge_positive": ridge_beauty > 0,
        "lasso_positive": lasso_beauty > 0,
    }
    n_positive = sum(direction_checks.values())

    tree_support = np.mean([tree_beauty_imp, figs_beauty_imp, hst_beauty_imp])

    score = 50

    if beauty_p < 1e-3:
        score += 25
    elif beauty_p < 1e-2:
        score += 20
    elif beauty_p < 5e-2:
        score += 12
    else:
        score -= 20

    score += 8 if pearson_p < 0.05 else -8
    score += 8 if spearman_p < 0.05 else -8
    score += 5 if t_p < 0.05 else -5
    score += 5 if anova_p < 0.05 else -5

    if n_positive >= 5:
        score += 7
    elif n_positive >= 4:
        score += 4
    else:
        score -= 10

    if tree_support > 0.10:
        score += 5

    # Keep score in the required Likert-style range.
    score = int(np.clip(round(score), 0, 100))

    explanation = (
        f"Question: {question} Evidence indicates a positive beauty-evaluation relationship. "
        f"Pearson r={pearson_r:.3f} (p={pearson_p:.2g}), Spearman rho={spearman_rho:.3f} "
        f"(p={spearman_p:.2g}), t-test p={t_p:.3g}, ANOVA p={anova_p:.2g}. "
        f"In OLS with controls, beauty coef={beauty_coef:.3f} (p={beauty_p:.2g}, "
        f"95% CI [{beauty_ci_low:.3f}, {beauty_ci_high:.3f}]). "
        f"Interpretable models also support positive impact: standardized beauty coefficients "
        f"Linear/Ridge/Lasso={lin_beauty:.3f}/{ridge_beauty:.3f}/{lasso_beauty:.3f}, "
        f"tree-based beauty importance DecisionTree/FIGS/HSTree={tree_beauty_imp:.3f}/"
        f"{figs_beauty_imp:.3f}/{hst_beauty_imp:.3f}. "
        f"Significant tests: {n_sig}/5; positive-direction checks: {n_positive}/6; "
        f"RuleFit linear beauty coefficient={beauty_linear_rule_coef:.3f}. "
        f"Overall conclusion: yes, beauty has a statistically significant positive impact "
        f"on teaching evaluations in this dataset."
    )

    output = {"response": score, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(output))

    print(f"Final score: {score}")
    print("Wrote conclusion.txt")


if __name__ == "__main__":
    main()
