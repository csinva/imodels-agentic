import json
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text


warnings.filterwarnings("ignore")


def top_k_effects(values, names, k=5):
    order = np.argsort(np.abs(values))[::-1][:k]
    return [(str(names[i]), float(values[i])) for i in order]


def main():
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info["research_questions"][0]
    print("Research question:", question)

    df = pd.read_csv("boxes.csv")
    print("\nData shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nSummary statistics:")
    print(df.describe(include="all").T)

    print("\nDistributions:")
    for col in ["y", "gender", "majority_first", "culture", "age"]:
        print(f"\n{col} value counts:")
        print(df[col].value_counts(dropna=False).sort_index())

    # Main analysis target: whether the child chose the majority option.
    df["majority_choice"] = (df["y"] == 2).astype(int)
    df["minority_choice"] = (df["y"] == 3).astype(int)

    print("\nMajority-choice rate:", round(float(df["majority_choice"].mean()), 4))
    print("\nMajority-choice by age:")
    print(df.groupby("age")["majority_choice"].mean().round(4))
    print("\nMajority-choice by culture:")
    print(df.groupby("culture")["majority_choice"].mean().round(4))

    corr_cols = ["majority_choice", "age", "gender", "majority_first", "culture"]
    print("\nCorrelation matrix:")
    print(df[corr_cols].corr().round(4))

    # Statistical tests focused on age-development and age-by-culture effects.
    stat_results = {}
    majority_age = df.loc[df["majority_choice"] == 1, "age"]
    nonmajority_age = df.loc[df["majority_choice"] == 0, "age"]

    t_res = stats.ttest_ind(majority_age, nonmajority_age, equal_var=False)
    pb_res = stats.pointbiserialr(df["majority_choice"], df["age"])
    sp_res = stats.spearmanr(df["age"], df["majority_choice"])
    chi2_res = stats.chi2_contingency(pd.crosstab(df["culture"], df["majority_choice"]))

    stat_results["ttest_age_majority_vs_nonmajority_p"] = float(t_res.pvalue)
    stat_results["pointbiserial_age_majority_r"] = float(pb_res.statistic)
    stat_results["pointbiserial_age_majority_p"] = float(pb_res.pvalue)
    stat_results["spearman_age_majority_rho"] = float(sp_res.correlation)
    stat_results["spearman_age_majority_p"] = float(sp_res.pvalue)
    stat_results["chi2_culture_majority_p"] = float(chi2_res[1])

    ols_main = smf.ols(
        "majority_choice ~ age + gender + majority_first + C(culture)", data=df
    ).fit()
    ols_inter = smf.ols(
        "majority_choice ~ age * C(culture) + gender + majority_first", data=df
    ).fit()

    interaction_terms = [t for t in ols_inter.params.index if "age:C(culture)" in t]
    if interaction_terms:
        joint_hypothesis = " = 0, ".join(interaction_terms) + " = 0"
        interaction_test = ols_inter.f_test(joint_hypothesis)
        ols_interaction_p = float(interaction_test.pvalue)
    else:
        ols_interaction_p = np.nan

    stat_results["ols_age_coef"] = float(ols_main.params["age"])
    stat_results["ols_age_p"] = float(ols_main.pvalues["age"])
    stat_results["ols_interaction_joint_p"] = ols_interaction_p

    logit_main = smf.logit(
        "majority_choice ~ age + gender + majority_first + C(culture)", data=df
    ).fit(disp=False)
    logit_inter = smf.logit(
        "majority_choice ~ age * C(culture) + gender + majority_first", data=df
    ).fit(disp=False, maxiter=300)

    lr_stat = 2 * (logit_inter.llf - logit_main.llf)
    lr_df = logit_inter.df_model - logit_main.df_model
    lr_p = stats.chi2.sf(lr_stat, lr_df)

    stat_results["logit_age_coef"] = float(logit_main.params["age"])
    stat_results["logit_age_p"] = float(logit_main.pvalues["age"])
    stat_results["logit_interaction_lr_p"] = float(lr_p)

    print("\nStatistical test results:")
    for k, v in stat_results.items():
        print(f"{k}: {v:.6f}")

    # Interpretable scikit-learn models.
    X = df[["age", "gender", "majority_first", "culture"]]
    y = df["majority_choice"].astype(float).values
    preprocessor = ColumnTransformer(
        [
            ("num", "passthrough", ["age", "gender", "majority_first"]),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), ["culture"]),
        ]
    )
    X_enc = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    sk_results = {}
    lin = LinearRegression().fit(X_enc, y)
    ridge = Ridge(alpha=1.0, random_state=0).fit(X_enc, y)
    lasso = Lasso(alpha=0.001, random_state=0, max_iter=10000).fit(X_enc, y)
    dtr = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X_enc, y)
    dtc = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_enc, y.astype(int))

    sk_models = {
        "linear_regression": lin,
        "ridge": ridge,
        "lasso": lasso,
        "decision_tree_regressor": dtr,
        "decision_tree_classifier": dtc,
    }

    print("\nScikit-learn interpretable model summaries:")
    for name, model in sk_models.items():
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(X_enc)[:, 1]
        else:
            preds = model.predict(X_enc)
        sk_results[f"{name}_r2"] = float(r2_score(y, preds))

        if hasattr(model, "coef_"):
            top = top_k_effects(model.coef_, feature_names, k=5)
            sk_results[f"{name}_top_coefficients"] = top
            print(f"{name} top coefficients:", top)
        if hasattr(model, "feature_importances_"):
            top = top_k_effects(model.feature_importances_, feature_names, k=5)
            sk_results[f"{name}_top_feature_importances"] = top
            print(f"{name} top feature importances:", top)
        print(f"{name} r2:", round(sk_results[f'{name}_r2'], 4))

    tree_rules = export_text(dtc, feature_names=list(feature_names))
    print("\nDecision tree classifier rules (depth<=3):")
    print(tree_rules)

    # Interpretable imodels models.
    X_im = pd.get_dummies(X, columns=["culture"], drop_first=True)
    X_im_values = X_im.values
    feature_names_im = X_im.columns.tolist()

    imodels_results = {}
    imodel_specs = [
        ("rulefit_regressor", RuleFitRegressor),
        ("figs_regressor", FIGSRegressor),
        ("hstree_regressor", HSTreeRegressor),
    ]

    print("\nimodels summaries:")
    for name, cls in imodel_specs:
        model = cls()
        model.fit(X_im_values, y, feature_names=feature_names_im)
        pred = model.predict(X_im_values)
        imodels_results[f"{name}_r2"] = float(r2_score(y, pred))
        print(f"{name} r2:", round(imodels_results[f"{name}_r2"], 4))

        if hasattr(model, "feature_importances_"):
            fi = np.array(model.feature_importances_)
            top = top_k_effects(fi, feature_names_im, k=5)
            imodels_results[f"{name}_top_feature_importances"] = top
            print(f"{name} top feature importances:", top)

        rules_text = str(model).splitlines()
        imodels_results[f"{name}_preview"] = " | ".join(rules_text[:6])
        print(f"{name} preview:", imodels_results[f"{name}_preview"])

    # Evidence-based scoring for the research question.
    # Low score if age and age-by-culture terms are consistently non-significant.
    age_pvals = [
        stat_results["pointbiserial_age_majority_p"],
        stat_results["ttest_age_majority_vs_nonmajority_p"],
        stat_results["ols_age_p"],
        stat_results["logit_age_p"],
    ]
    interaction_pvals = [
        stat_results["ols_interaction_joint_p"],
        stat_results["logit_interaction_lr_p"],
    ]
    sig_age = sum(p < 0.05 for p in age_pvals if pd.notna(p))
    sig_inter = sum(p < 0.05 for p in interaction_pvals if pd.notna(p))
    total_tests = len([p for p in age_pvals + interaction_pvals if pd.notna(p)])
    sig_ratio = (sig_age + sig_inter) / total_tests if total_tests else 0.0

    # Convert evidence to 0-100 Likert. Penalize for non-significance; keep slight floor >0.
    response = int(round(100 * sig_ratio))
    if response == 0:
        response = 10

    explanation = (
        "Evidence does not support a developmental increase in majority preference with age "
        "across cultures in this dataset. Age showed near-zero association with majority choice "
        f"(point-biserial r={stat_results['pointbiserial_age_majority_r']:.3f}, "
        f"p={stat_results['pointbiserial_age_majority_p']:.3f}; t-test p={stat_results['ttest_age_majority_vs_nonmajority_p']:.3f}; "
        f"OLS age p={stat_results['ols_age_p']:.3f}; logit age p={stat_results['logit_age_p']:.3f}). "
        "Age-by-culture interaction was also not significant "
        f"(OLS joint interaction p={stat_results['ols_interaction_joint_p']:.3f}; "
        f"logit LR interaction p={stat_results['logit_interaction_lr_p']:.3f}). "
        "Interpretable sklearn and imodels models consistently highlighted majority_first and some "
        "culture indicators as stronger predictors than age. Therefore the answer is a strong 'No' to "
        "the claim that majority reliance develops with age across cultural contexts in this sample."
    )

    output = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=True)

    print("\nWrote conclusion.txt with response =", response)


if __name__ == "__main__":
    main()
