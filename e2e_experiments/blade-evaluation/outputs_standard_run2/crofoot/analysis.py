import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def p_to_support(p_value: float) -> float:
    if p_value < 0.01:
        return 1.0
    if p_value < 0.05:
        return 0.8
    if p_value < 0.10:
        return 0.6
    if p_value < 0.20:
        return 0.4
    return 0.2


def safe_pvalue(value) -> float:
    try:
        value = float(value)
    except Exception:
        return 1.0
    if np.isnan(value):
        return 1.0
    return value


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    info_path = base_dir / "info.json"
    data_path = base_dir / "crofoot.csv"
    conclusion_path = base_dir / "conclusion.txt"

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    research_question = info["research_questions"][0].strip()

    df = pd.read_csv(data_path)

    # Engineer interpretable relative features tied to the research question.
    df["rel_group_size"] = df["n_focal"] - df["n_other"]
    df["rel_males"] = df["m_focal"] - df["m_other"]
    df["rel_females"] = df["f_focal"] - df["f_other"]
    # Positive value means focal is farther from own center than the opponent is.
    # For focal "home advantage", larger values of (dist_other - dist_focal) are favorable.
    df["rel_distance"] = df["dist_other"] - df["dist_focal"]

    print("Research question:", research_question)
    print("\nDataset shape:", df.shape)
    print("\nMissing values by column:\n", df.isna().sum())
    print("\nSummary statistics:\n", df.describe(include="all"))
    print("\nWin rate:", float(df["win"].mean()))

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr(numeric_only=True)
    print("\nCorrelation with win:\n", corr["win"].sort_values(ascending=False))

    print("\nDistributions by win group (means):")
    by_win_means = df.groupby("win")[
        ["rel_group_size", "rel_distance", "dist_focal", "dist_other", "n_focal", "n_other"]
    ].mean()
    print(by_win_means)

    # Statistical tests
    stat_results = {}

    r_size = stats.pointbiserialr(df["win"], df["rel_group_size"])
    r_loc = stats.pointbiserialr(df["win"], df["rel_distance"])
    r_dist_focal = stats.pointbiserialr(df["win"], df["dist_focal"])
    stat_results["pointbiserial_rel_group_size_p"] = safe_pvalue(r_size.pvalue)
    stat_results["pointbiserial_rel_distance_p"] = safe_pvalue(r_loc.pvalue)
    stat_results["pointbiserial_dist_focal_p"] = safe_pvalue(r_dist_focal.pvalue)
    print("\nPoint-biserial tests:")
    print("rel_group_size:", r_size)
    print("rel_distance:", r_loc)
    print("dist_focal:", r_dist_focal)

    wins = df[df["win"] == 1]
    losses = df[df["win"] == 0]
    ttest_vars = [
        "rel_group_size",
        "rel_distance",
        "dist_focal",
        "dist_other",
        "n_focal",
        "n_other",
    ]
    print("\nWelch t-tests (win=1 vs win=0):")
    for col in ttest_vars:
        t_stat, p_val = stats.ttest_ind(wins[col], losses[col], equal_var=False)
        stat_results[f"ttest_{col}_p"] = safe_pvalue(p_val)
        print(
            f"{col}: t={t_stat:.4f}, p={p_val:.4f}, "
            f"mean_win={wins[col].mean():.3f}, mean_loss={losses[col].mean():.3f}"
        )

    # ANOVA and chi-square for categorized relative group size.
    bins = [-np.inf, -0.5, 0.5, np.inf]
    labels = ["smaller", "equal", "larger"]
    df["size_category"] = pd.cut(df["rel_group_size"], bins=bins, labels=labels)

    anova_groups = [
        df.loc[df["size_category"] == category, "win"].values for category in labels
    ]
    f_stat, p_anova = stats.f_oneway(*anova_groups)
    stat_results["anova_size_category_p"] = safe_pvalue(p_anova)
    print(f"\nANOVA on win by size_category: F={f_stat:.4f}, p={p_anova:.4f}")

    contingency = pd.crosstab(df["size_category"], df["win"])
    chi2_stat, p_chi2, dof, expected = stats.chi2_contingency(contingency)
    stat_results["chi2_size_category_p"] = safe_pvalue(p_chi2)
    print("\nChi-square on size_category vs win:")
    print("contingency:\n", contingency)
    print(f"chi2={chi2_stat:.4f}, p={p_chi2:.4f}, dof={dof}")
    print("expected:\n", expected)

    # Statsmodels regressions (interpretability + p-values).
    core_features = ["rel_group_size", "rel_distance"]
    # Keep regression design non-singular; rel_group_size is strongly tied to
    # rel_males/rel_females, so include only one control from that family.
    controls = ["rel_males"]
    reg_features = core_features + controls
    X_sm = sm.add_constant(df[reg_features])
    y = df["win"]

    ols_model = sm.OLS(y, X_sm).fit()
    print("\nOLS summary:\n", ols_model.summary())
    stat_results["ols_rel_group_size_p"] = safe_pvalue(ols_model.pvalues["rel_group_size"])
    stat_results["ols_rel_distance_p"] = safe_pvalue(ols_model.pvalues["rel_distance"])

    logit_model = sm.Logit(y, X_sm).fit(disp=0)
    print("\nLogit summary:\n", logit_model.summary())
    stat_results["logit_rel_group_size_p"] = safe_pvalue(
        logit_model.pvalues["rel_group_size"]
    )
    stat_results["logit_rel_distance_p"] = safe_pvalue(logit_model.pvalues["rel_distance"])

    # Interpretable sklearn models
    model_features = [
        "rel_group_size",
        "rel_distance",
        "n_focal",
        "n_other",
        "dist_focal",
        "dist_other",
        "rel_males",
        "rel_females",
    ]
    X = df[model_features].copy()
    y_np = y.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lin = LinearRegression()
    lin.fit(X_scaled, y_np)
    ridge = Ridge(alpha=1.0, random_state=0)
    ridge.fit(X_scaled, y_np)
    lasso = Lasso(alpha=0.03, random_state=0, max_iter=10000)
    lasso.fit(X_scaled, y_np)

    print("\nLinearRegression coefficients:")
    print(dict(zip(model_features, lin.coef_)))
    print("\nRidge coefficients:")
    print(dict(zip(model_features, ridge.coef_)))
    print("\nLasso coefficients:")
    print(dict(zip(model_features, lasso.coef_)))

    dt = DecisionTreeClassifier(max_depth=3, random_state=0)
    dt.fit(X, y_np)
    print("\nDecisionTreeClassifier feature importances:")
    print(dict(zip(model_features, dt.feature_importances_)))

    # imodels interpretable models
    rulefit = RuleFitRegressor(random_state=0, max_rules=20)
    rulefit.fit(X.values, y_np, feature_names=list(model_features))
    rules = rulefit._get_rules(exclude_zero_coef=True)
    rules_sorted = rules.sort_values("importance", ascending=False).head(10)
    print("\nTop RuleFit rules/features by importance:")
    print(rules_sorted[["rule", "coef", "support", "importance"]])

    figs = FIGSRegressor(random_state=0, max_rules=12)
    figs.fit(X, y_np)
    print("\nFIGS feature importances:")
    print(dict(zip(model_features, figs.feature_importances_)))
    print("\nFIGS structure:\n", figs)

    base_tree = DecisionTreeRegressor(max_depth=3, random_state=0)
    base_tree.fit(X, y_np)
    hst = HSTreeRegressor(base_tree)
    hst.fit(X, y_np)
    print("\nHSTree (shrunken tree) base feature importances:")
    print(dict(zip(model_features, hst.estimator_.feature_importances_)))
    print("HSTree complexity:", hst.complexity_)

    # Evidence synthesis for final Likert response.
    size_pvals = [
        stat_results["pointbiserial_rel_group_size_p"],
        stat_results["ttest_rel_group_size_p"],
        stat_results["ols_rel_group_size_p"],
        stat_results["logit_rel_group_size_p"],
        stat_results["chi2_size_category_p"],
    ]
    location_pvals = [
        stat_results["pointbiserial_rel_distance_p"],
        stat_results["ttest_rel_distance_p"],
        stat_results["pointbiserial_dist_focal_p"],
        stat_results["ttest_dist_focal_p"],
        stat_results["ols_rel_distance_p"],
        stat_results["logit_rel_distance_p"],
    ]

    size_support = float(np.mean([p_to_support(p) for p in size_pvals]))
    location_support = float(np.mean([p_to_support(p) for p in location_pvals]))

    # Direction checks: expected signs for an influence in the hypothesized direction.
    # rel_group_size positive, rel_distance positive, dist_focal negative.
    expected_size_direction = (
        (logit_model.params["rel_group_size"] > 0)
        and (ols_model.params["rel_group_size"] > 0)
        and (r_size.statistic > 0)
    )
    expected_location_direction = (
        (r_loc.statistic > 0)
        and (logit_model.params["rel_distance"] > 0)
        and (r_dist_focal.statistic < 0)
    )

    if not expected_size_direction:
        size_support *= 0.7
    if not expected_location_direction:
        location_support *= 0.7

    combined_support = 0.5 * size_support + 0.5 * location_support
    response_score = int(np.clip(round(100 * combined_support), 0, 100))

    explanation = (
        "Relative group size shows only weak-to-moderate evidence (point-biserial "
        f"p={stat_results['pointbiserial_rel_group_size_p']:.3f}, logit p="
        f"{stat_results['logit_rel_group_size_p']:.3f}), while contest location is mixed: "
        "the relative-distance terms are not significant in regression/correlation "
        f"(logit p={stat_results['logit_rel_distance_p']:.3f}), but focal groups that won "
        f"were closer to their own home-range center (Welch t-test on dist_focal p="
        f"{stat_results['ttest_dist_focal_p']:.3f}). Interpretable tree/rule models "
        "also prioritize distance-related splits more than group-size splits. Overall, "
        "the data provide limited, not strong, support that relative size and location "
        "jointly influence win probability in this sample."
    )

    result = {"response": response_score, "explanation": explanation}
    with conclusion_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(result))

    print("\nFinal conclusion JSON:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
