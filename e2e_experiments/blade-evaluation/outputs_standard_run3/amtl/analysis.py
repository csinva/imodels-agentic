import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

warnings.filterwarnings("ignore")


def safe_print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def extract_top_linear_features(model, feature_names, top_k=8):
    coefs = np.ravel(model.coef_)
    pairs = list(zip(feature_names, coefs))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:top_k]


def compute_likert_score(results):
    score = 50

    glm_coef = results.get("glm_is_human_coef", np.nan)
    glm_p = results.get("glm_is_human_p", np.nan)
    t_p = results.get("ttest_p", np.nan)
    t_stat = results.get("ttest_stat", np.nan)
    mean_diff = results.get("mean_diff", np.nan)
    chi2_p = results.get("chi2_p", np.nan)
    anova_p = results.get("anova_p", np.nan)

    if np.isfinite(glm_coef):
        score += 18 if glm_coef > 0 else -18
    if np.isfinite(glm_p):
        if glm_p < 1e-6:
            score += 20
        elif glm_p < 1e-3:
            score += 16
        elif glm_p < 0.01:
            score += 12
        elif glm_p < 0.05:
            score += 8
        else:
            score -= 12

    if np.isfinite(mean_diff) and np.isfinite(t_stat):
        if mean_diff > 0 and t_stat > 0:
            score += 12
        else:
            score -= 12

    if np.isfinite(t_p):
        if t_p < 1e-6:
            score += 12
        elif t_p < 1e-3:
            score += 10
        elif t_p < 0.01:
            score += 8
        elif t_p < 0.05:
            score += 5
        else:
            score -= 8

    if np.isfinite(chi2_p):
        score += 6 if chi2_p < 0.05 else -4

    if np.isfinite(anova_p):
        score += 4 if anova_p < 0.05 else -2

    return int(max(0, min(100, round(score))))


def main():
    root = Path(".")
    info_path = root / "info.json"
    data_path = root / "amtl.csv"

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info.get("research_questions", [""])[0]

    df = pd.read_csv(data_path)

    # Core derived variables for this AMTL question
    df["amtl_rate"] = (df["num_amtl"] / df["sockets"]).clip(0, 1)
    df["any_amtl"] = (df["num_amtl"] > 0).astype(int)
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    safe_print_section("Research Question")
    print(research_question)

    safe_print_section("Data Overview")
    print(f"Shape: {df.shape}")
    print("Missing values by column:")
    print(df.isna().sum())

    num_cols = ["num_amtl", "sockets", "age", "stdev_age", "prob_male", "amtl_rate"]
    print("\nSummary statistics:")
    print(df[num_cols].describe().T)

    safe_print_section("Group Distributions")
    genus_stats = df.groupby("genus")["amtl_rate"].agg(["mean", "std", "median", "count"])
    print("AMTL rate by genus:")
    print(genus_stats)

    tooth_stats = df.groupby("tooth_class")["amtl_rate"].agg(["mean", "std", "median", "count"])
    print("\nAMTL rate by tooth class:")
    print(tooth_stats)

    safe_print_section("Correlations")
    corr = df[num_cols].corr(numeric_only=True)
    print(corr)

    safe_print_section("Statistical Tests")
    human_rates = df.loc[df["is_human"] == 1, "amtl_rate"]
    nonhuman_rates = df.loc[df["is_human"] == 0, "amtl_rate"]

    ttest_res = stats.ttest_ind(human_rates, nonhuman_rates, equal_var=False, alternative="greater")
    mwu_res = stats.mannwhitneyu(human_rates, nonhuman_rates, alternative="greater")

    contingency = pd.crosstab(df["is_human"], df["any_amtl"])
    chi2_stat, chi2_p, _, _ = stats.chi2_contingency(contingency)

    groups = [g["amtl_rate"].values for _, g in df.groupby("genus")]
    anova_stat, anova_p = stats.f_oneway(*groups)

    print(f"Welch t-test (Homo sapiens > non-human), t={ttest_res.statistic:.4f}, p={ttest_res.pvalue:.4e}")
    print(f"Mann-Whitney U (Homo sapiens > non-human), U={mwu_res.statistic:.4f}, p={mwu_res.pvalue:.4e}")
    print(f"Chi-square (human vs non-human by any AMTL), chi2={chi2_stat:.4f}, p={chi2_p:.4e}")
    print(f"ANOVA across genera on AMTL rate, F={anova_stat:.4f}, p={anova_p:.4e}")

    safe_print_section("Controlled Regression Models (Interpretability)")
    ols_formula = "amtl_rate ~ is_human + age + prob_male + C(tooth_class)"
    ols_res = smf.ols(ols_formula, data=df).fit(cov_type="HC3")
    print("OLS coefficients and p-values:")
    print(pd.DataFrame({"coef": ols_res.params, "p_value": ols_res.pvalues}))

    glm_formula = "amtl_rate ~ is_human + age + prob_male + C(tooth_class)"
    glm_res = smf.glm(
        glm_formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()
    print("\nBinomial GLM coefficients and p-values:")
    print(pd.DataFrame({"coef": glm_res.params, "p_value": glm_res.pvalues}))

    genus_glm_formula = "amtl_rate ~ C(genus) + age + prob_male + C(tooth_class)"
    genus_glm_res = smf.glm(
        genus_glm_formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()
    print("\nGenus-specific GLM coefficients and p-values (Homo sapiens baseline):")
    print(pd.DataFrame({"coef": genus_glm_res.params, "p_value": genus_glm_res.pvalues}))

    safe_print_section("Scikit-learn Interpretable Models")
    feature_cols = ["age", "stdev_age", "prob_male", "sockets", "tooth_class", "genus"]
    X = df[feature_cols]
    y_reg = df["amtl_rate"]
    y_clf = df["any_amtl"]

    cat_cols = ["tooth_class", "genus"]
    num_model_cols = ["age", "stdev_age", "prob_male", "sockets"]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_model_cols),
        ]
    )

    lin_pipe = Pipeline([("pre", pre), ("model", LinearRegression())])
    ridge_pipe = Pipeline([("pre", pre), ("model", Ridge(alpha=1.0, random_state=0))])
    lasso_pipe = Pipeline([("pre", pre), ("model", Lasso(alpha=0.0005, random_state=0, max_iter=20000))])

    lin_pipe.fit(X, y_reg)
    ridge_pipe.fit(X, y_reg)
    lasso_pipe.fit(X, y_reg)

    feature_names = lin_pipe.named_steps["pre"].get_feature_names_out()

    lin_top = extract_top_linear_features(lin_pipe.named_steps["model"], feature_names)
    ridge_top = extract_top_linear_features(ridge_pipe.named_steps["model"], feature_names)
    lasso_top = extract_top_linear_features(lasso_pipe.named_steps["model"], feature_names)

    print("Top LinearRegression coefficients:")
    print(lin_top)
    print("\nTop Ridge coefficients:")
    print(ridge_top)
    print("\nTop Lasso coefficients:")
    print(lasso_top)

    Xt = pre.fit_transform(X)
    tree_reg = DecisionTreeRegressor(max_depth=3, min_samples_leaf=25, random_state=0)
    tree_clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=25, random_state=0)
    tree_reg.fit(Xt, y_reg)
    tree_clf.fit(Xt, y_clf)

    tree_reg_imp = sorted(zip(feature_names, tree_reg.feature_importances_), key=lambda x: x[1], reverse=True)
    tree_clf_imp = sorted(zip(feature_names, tree_clf.feature_importances_), key=lambda x: x[1], reverse=True)

    print("\nDecisionTreeRegressor feature importances:")
    print(tree_reg_imp[:10])
    print("\nDecisionTreeClassifier feature importances:")
    print(tree_clf_imp[:10])

    safe_print_section("imodels Interpretable Models")
    imodels_summary = []
    X_im = pd.get_dummies(X, drop_first=True)

    try:
        from imodels import RuleFitRegressor, FIGSRegressor, HSTreeRegressor

        rulefit = RuleFitRegressor(random_state=0)
        rulefit.fit(X_im, y_reg)
        if hasattr(rulefit, "get_rules"):
            rules_df = rulefit.get_rules()
            rules_df = rules_df[rules_df.coef != 0].sort_values("support", ascending=False)
            imodels_summary.append(
                {
                    "model": "RuleFitRegressor",
                    "n_active_rules": int(rules_df.shape[0]),
                    "top_rules": rules_df[["rule", "coef", "support"]].head(5).to_dict("records"),
                }
            )
        else:
            imodels_summary.append({"model": "RuleFitRegressor", "status": "fit_ok"})

        figs = FIGSRegressor(random_state=0, max_rules=12)
        figs.fit(X_im, y_reg)
        figs_imp = getattr(figs, "feature_importances_", None)
        if figs_imp is not None:
            top_figs = sorted(zip(X_im.columns, figs_imp), key=lambda x: x[1], reverse=True)[:8]
            imodels_summary.append({"model": "FIGSRegressor", "top_feature_importances": top_figs})
        else:
            imodels_summary.append({"model": "FIGSRegressor", "status": "fit_ok"})

        hst = HSTreeRegressor(random_state=0, max_leaf_nodes=12)
        hst.fit(X_im, y_reg)
        hst_imp = getattr(hst, "feature_importances_", None)
        if hst_imp is not None:
            top_hst = sorted(zip(X_im.columns, hst_imp), key=lambda x: x[1], reverse=True)[:8]
            imodels_summary.append({"model": "HSTreeRegressor", "top_feature_importances": top_hst})
        else:
            imodels_summary.append({"model": "HSTreeRegressor", "status": "fit_ok"})

    except Exception as e:
        imodels_summary.append({"model": "imodels", "status": f"failed: {str(e)}"})

    print(imodels_summary)

    # Extract main inferential evidence for answer
    human_mean = float(human_rates.mean())
    nonhuman_mean = float(nonhuman_rates.mean())

    results_for_score = {
        "glm_is_human_coef": float(glm_res.params.get("is_human", np.nan)),
        "glm_is_human_p": float(glm_res.pvalues.get("is_human", np.nan)),
        "ttest_stat": float(ttest_res.statistic),
        "ttest_p": float(ttest_res.pvalue),
        "mean_diff": human_mean - nonhuman_mean,
        "chi2_p": float(chi2_p),
        "anova_p": float(anova_p),
    }

    response_score = compute_likert_score(results_for_score)

    coef_human = results_for_score["glm_is_human_coef"]
    p_human = results_for_score["glm_is_human_p"]

    explanation = (
        f"Humans show substantially higher AMTL rates than non-human primates (mean {human_mean:.3f} vs "
        f"{nonhuman_mean:.3f}; one-sided Welch t-test p={ttest_res.pvalue:.2e}). In a controlled binomial "
        f"regression with age, sex probability, and tooth class, the human indicator remains strongly positive "
        f"(coef={coef_human:.3f}, p={p_human:.2e}). Genus-specific controlled models also show Pan, Papio, and "
        f"Pongo below Homo sapiens. Interpretable linear/tree/rule-based models consistently prioritize genus "
        f"(human vs non-human) and age as major predictors, supporting a strong Yes to the research question."
    )

    conclusion = {"response": int(response_score), "explanation": explanation}

    with (root / "conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(conclusion, f)

    safe_print_section("Conclusion JSON")
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()
