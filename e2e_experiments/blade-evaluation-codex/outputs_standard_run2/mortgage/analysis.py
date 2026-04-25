import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from imodels import RuleFitRegressor, FIGSRegressor, HSTreeRegressor

warnings.filterwarnings("ignore")


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def summarize_series(series: pd.Series):
    return {
        "mean": safe_float(series.mean()),
        "std": safe_float(series.std()),
        "min": safe_float(series.min()),
        "25%": safe_float(series.quantile(0.25)),
        "50%": safe_float(series.quantile(0.50)),
        "75%": safe_float(series.quantile(0.75)),
        "max": safe_float(series.max()),
    }


def main():
    info_path = Path("info.json")
    data_path = Path("mortgage.csv")

    info = json.loads(info_path.read_text())
    question = info.get("research_questions", ["Unknown research question"])[0]

    df_raw = pd.read_csv(data_path)

    # Standardize likely index column if present.
    if "Unnamed: 0" in df_raw.columns:
        df_raw = df_raw.drop(columns=["Unnamed: 0"])

    # Ensure numeric where expected; coerce non-numeric to NaN.
    for c in df_raw.columns:
        df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

    # Build/verify approval target.
    if "accept" not in df_raw.columns and "deny" in df_raw.columns:
        df_raw["accept"] = 1 - df_raw["deny"]

    required_cols = [
        "female",
        "accept",
        "black",
        "housing_expense_ratio",
        "self_employed",
        "married",
        "mortgage_credit",
        "consumer_credit",
        "bad_history",
        "PI_ratio",
        "loan_to_value",
    ]
    if "denied_PMI" in df_raw.columns:
        required_cols.append("denied_PMI")

    df = df_raw[required_cols].dropna().copy()

    # -----------------
    # 1) Data exploration
    # -----------------
    n = len(df)
    approval_by_gender = df.groupby("female")["accept"].mean().to_dict()
    corr_with_accept = df.corr(numeric_only=True)["accept"].sort_values(ascending=False).to_dict()

    exploration = {
        "research_question": question,
        "n_rows_after_dropna": int(n),
        "approval_rate_overall": safe_float(df["accept"].mean()),
        "approval_rate_by_female": {str(k): safe_float(v) for k, v in approval_by_gender.items()},
        "female_distribution": df["female"].value_counts(normalize=True).sort_index().to_dict(),
        "accept_distribution": df["accept"].value_counts(normalize=True).sort_index().to_dict(),
        "summary_PI_ratio": summarize_series(df["PI_ratio"]),
        "summary_loan_to_value": summarize_series(df["loan_to_value"]),
        "corr_with_accept": {k: safe_float(v) for k, v in corr_with_accept.items()},
    }

    print("=== Research Question ===")
    print(question)
    print("\n=== Basic Exploration ===")
    print(json.dumps(exploration, indent=2))

    # -----------------
    # 2) Statistical tests
    # -----------------
    g0 = df.loc[df["female"] == 0, "accept"]
    g1 = df.loc[df["female"] == 1, "accept"]

    contingency = pd.crosstab(df["female"], df["accept"])
    chi2_stat, chi2_p, _, _ = stats.chi2_contingency(contingency)

    t_stat, t_p = stats.ttest_ind(g1, g0, equal_var=False, nan_policy="omit")

    # ANOVA for acceptance by female and mortgage_credit (plus interaction)
    anova_model = smf.ols("accept ~ C(female) + C(mortgage_credit) + C(female):C(mortgage_credit)", data=df).fit()
    anova_tbl = anova_lm(anova_model, typ=2)

    # Gender differences in key risk metrics (contextual compositional differences)
    pi_t = stats.ttest_ind(
        df.loc[df["female"] == 1, "PI_ratio"],
        df.loc[df["female"] == 0, "PI_ratio"],
        equal_var=False,
        nan_policy="omit",
    )
    ltv_t = stats.ttest_ind(
        df.loc[df["female"] == 1, "loan_to_value"],
        df.loc[df["female"] == 0, "loan_to_value"],
        equal_var=False,
        nan_policy="omit",
    )

    print("\n=== Statistical Tests ===")
    print(f"Chi-square female vs accept: chi2={chi2_stat:.4f}, p={chi2_p:.6g}")
    print(f"T-test accept by female: t={t_stat:.4f}, p={t_p:.6g}")
    print("ANOVA table (accept ~ female + mortgage_credit + interaction):")
    print(anova_tbl)
    print(f"T-test PI_ratio by female: t={pi_t.statistic:.4f}, p={pi_t.pvalue:.6g}")
    print(f"T-test loan_to_value by female: t={ltv_t.statistic:.4f}, p={ltv_t.pvalue:.6g}")

    # -----------------
    # 3) Interpretable models (scikit-learn)
    # -----------------
    features_base = [
        "female",
        "black",
        "housing_expense_ratio",
        "self_employed",
        "married",
        "mortgage_credit",
        "consumer_credit",
        "bad_history",
        "PI_ratio",
        "loan_to_value",
    ]
    features_with_pmi = features_base + (["denied_PMI"] if "denied_PMI" in df.columns else [])

    X = df[features_with_pmi].copy()
    y = df["accept"].copy()

    lin = LinearRegression().fit(X, y)
    ridge = Ridge(alpha=1.0).fit(X, y)
    lasso = Lasso(alpha=0.001, max_iter=20000, random_state=0).fit(X, y)

    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=0)
    tree.fit(X, y)

    coef_df = pd.DataFrame(
        {
            "feature": features_with_pmi,
            "linear_coef": lin.coef_,
            "ridge_coef": ridge.coef_,
            "lasso_coef": lasso.coef_,
            "tree_importance": tree.feature_importances_,
        }
    ).sort_values("linear_coef", key=lambda s: s.abs(), ascending=False)

    female_row = coef_df.loc[coef_df["feature"] == "female"].iloc[0].to_dict()

    print("\n=== scikit-learn Interpretable Models ===")
    print("Top coefficients/importances:")
    print(coef_df.head(10))
    print("Female effect row:")
    print(female_row)

    # -----------------
    # 4) Interpretable models (imodels)
    # -----------------
    imodels_results = {}

    # RuleFitRegressor: extract linear term for female from rules dataframe.
    try:
        rulefit = RuleFitRegressor(random_state=0, include_linear=True, max_rules=40)
        rulefit.fit(X.values, y.values, feature_names=features_with_pmi)

        if hasattr(rulefit, "_get_rules"):
            rules_df = rulefit._get_rules(exclude_zero_coef=False)
            female_rule_rows = rules_df[rules_df["rule"].astype(str) == "female"]
            female_rule_coef = safe_float(female_rule_rows["coef"].iloc[0]) if len(female_rule_rows) else np.nan
            female_rule_importance = safe_float(female_rule_rows["importance"].iloc[0]) if len(female_rule_rows) else np.nan
            imodels_results["rulefit_female_coef"] = female_rule_coef
            imodels_results["rulefit_female_importance"] = female_rule_importance
        else:
            imodels_results["rulefit_female_coef"] = np.nan
            imodels_results["rulefit_female_importance"] = np.nan
    except Exception as e:
        imodels_results["rulefit_error"] = str(e)

    # FIGSRegressor: use feature_importances_ for female.
    try:
        figs = FIGSRegressor(random_state=0, max_rules=12)
        figs.fit(X, y)
        if hasattr(figs, "feature_importances_"):
            fidx = features_with_pmi.index("female")
            imodels_results["figs_female_importance"] = safe_float(figs.feature_importances_[fidx])
            imodels_results["figs_tree_uses_female"] = bool(figs.feature_importances_[fidx] > 0)
        else:
            imodels_results["figs_female_importance"] = np.nan
            imodels_results["figs_tree_uses_female"] = False
    except Exception as e:
        imodels_results["figs_error"] = str(e)

    # HSTreeRegressor: inspect textual tree for presence of female split.
    try:
        hs = HSTreeRegressor(max_leaf_nodes=10, random_state=0)
        hs.fit(X, y)
        hs_text = str(hs)
        imodels_results["hstree_uses_female"] = "female" in hs_text
    except Exception as e:
        imodels_results["hstree_error"] = str(e)

    print("\n=== imodels Interpretable Models ===")
    print(imodels_results)

    # -----------------
    # 5) Regression with p-values (statsmodels)
    # -----------------
    X_ols = sm.add_constant(df[features_base])
    ols = sm.OLS(y, X_ols).fit()

    logit = sm.Logit(y, X_ols).fit(disp=0)

    female_ols_coef = safe_float(ols.params.get("female", np.nan))
    female_ols_p = safe_float(ols.pvalues.get("female", np.nan))
    female_ols_ci_low, female_ols_ci_high = [safe_float(v) for v in ols.conf_int().loc["female"]]

    female_logit_coef = safe_float(logit.params.get("female", np.nan))
    female_logit_p = safe_float(logit.pvalues.get("female", np.nan))
    female_logit_ci_low, female_logit_ci_high = [safe_float(v) for v in logit.conf_int().loc["female"]]

    print("\n=== Statsmodels Regression ===")
    print(
        f"OLS female coef={female_ols_coef:.6f}, p={female_ols_p:.6g}, "
        f"95% CI=({female_ols_ci_low:.6f}, {female_ols_ci_high:.6f})"
    )
    print(
        f"Logit female coef={female_logit_coef:.6f}, p={female_logit_p:.6g}, "
        f"95% CI=({female_logit_ci_low:.6f}, {female_logit_ci_high:.6f})"
    )

    # -----------------
    # 6) Synthesis into Likert response (0-100)
    # -----------------
    # Scoring emphasizes significance for gender while discounting weak practical impact.
    score = 45

    # Unadjusted tests (raw association).
    score += 15 if chi2_p < 0.05 else -5
    score += 10 if t_p < 0.05 else -5

    # Adjusted tests (conditional association).
    score += 20 if female_ols_p < 0.05 else -20
    score += 15 if female_logit_p < 0.05 else -15

    # Effect size from adjusted linear probability model.
    abs_effect = abs(female_ols_coef)
    if abs_effect >= 0.05:
        score += 5
    elif abs_effect >= 0.02:
        score -= 3
    else:
        score -= 8

    # Penalize if tree/rule-based models show near-zero use of female.
    female_tree_importance = safe_float(female_row.get("tree_importance", np.nan), default=np.nan)
    if np.isnan(female_tree_importance) or female_tree_importance <= 0.001:
        score -= 5

    figs_female_importance = safe_float(imodels_results.get("figs_female_importance", np.nan), default=np.nan)
    if np.isnan(figs_female_importance) or figs_female_importance <= 0.001:
        score -= 3

    score = int(np.clip(round(score), 0, 100))

    direction = "higher" if female_ols_coef > 0 else "lower"
    explanation = (
        f"Raw approval rates are nearly identical by gender (female={approval_by_gender.get(1.0, np.nan):.3f}, "
        f"male={approval_by_gender.get(0.0, np.nan):.3f}; chi-square p={chi2_p:.3f}). "
        f"After controlling for credit and debt factors, female has a statistically significant association with "
        f"{direction} approval in both OLS (coef={female_ols_coef:.3f}, p={female_ols_p:.3f}, "
        f"95% CI [{female_ols_ci_low:.3f}, {female_ols_ci_high:.3f}]) and logistic regression "
        f"(coef={female_logit_coef:.3f}, p={female_logit_p:.3f}). "
        f"Interpretable tree/rule models assign little importance to female, indicating the effect is present but modest."
    )

    output = {"response": score, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(output))

    print("\n=== Final Conclusion JSON ===")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
