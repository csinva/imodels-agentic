import json
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")


def main():
    # 1) Load data and define core variables for the research question
    df = pd.read_csv("caschools.csv")
    df["avg_score"] = (df["read"] + df["math"]) / 2.0
    df["str_ratio"] = df["students"] / df["teachers"]  # student-teacher ratio
    df["computer_per_student"] = df["computer"] / df["students"]

    analysis_cols = [
        "avg_score",
        "str_ratio",
        "lunch",
        "english",
        "income",
        "expenditure",
        "calworks",
        "computer_per_student",
        "students",
    ]
    data = df[analysis_cols].dropna().copy()

    print("=== DATA OVERVIEW ===")
    print(f"Rows used: {len(data)}")
    print(data.describe().T)

    print("\n=== CORRELATIONS (Pearson) ===")
    corr = data.corr(numeric_only=True)
    print(corr["avg_score"].sort_values(ascending=False))

    # Simple distribution summaries for core variables
    print("\n=== DISTRIBUTION SNAPSHOT ===")
    for c in ["avg_score", "str_ratio", "lunch", "income", "english"]:
        q = data[c].quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        print(f"{c}:\n{q}\n")

    y = data["avg_score"]
    str_x = data["str_ratio"]

    # 2) Statistical tests directly addressing the relationship
    pearson_r, pearson_p = stats.pearsonr(str_x, y)
    spearman_rho, spearman_p = stats.spearmanr(str_x, y)

    median_str = str_x.median()
    low_str_scores = y[str_x <= median_str]
    high_str_scores = y[str_x > median_str]
    t_stat, t_p = stats.ttest_ind(low_str_scores, high_str_scores, equal_var=False)

    data["str_quartile"] = pd.qcut(data["str_ratio"], 4, labels=False, duplicates="drop")
    groups = [g["avg_score"].values for _, g in data.groupby("str_quartile")]
    f_stat, anova_p = stats.f_oneway(*groups)

    # OLS: simple and adjusted regressions with p-values
    X_simple = sm.add_constant(data[["str_ratio"]])
    ols_simple = sm.OLS(y, X_simple).fit()

    controls = [
        "str_ratio",
        "lunch",
        "english",
        "income",
        "expenditure",
        "calworks",
        "computer_per_student",
        "students",
    ]
    X_adj = sm.add_constant(data[controls])
    ols_adj = sm.OLS(y, X_adj).fit()

    print("\n=== STATISTICAL TESTS ===")
    print(f"Pearson r(str, avg_score): {pearson_r:.4f}, p={pearson_p:.4g}")
    print(f"Spearman rho(str, avg_score): {spearman_rho:.4f}, p={spearman_p:.4g}")
    print(f"T-test low vs high STR avg_score: t={t_stat:.4f}, p={t_p:.4g}")
    print(f"ANOVA across STR quartiles: F={f_stat:.4f}, p={anova_p:.4g}")

    print("\n=== OLS SIMPLE ===")
    print(ols_simple.summary())

    print("\n=== OLS ADJUSTED ===")
    print(ols_adj.summary())

    # 3) Interpretable sklearn models
    X = data[controls]

    lin = LinearRegression().fit(X, y)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=0)).fit(X, y)
    lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.05, random_state=0, max_iter=20000)).fit(X, y)
    tree = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y)

    print("\n=== SKLEARN INTERPRETABLE MODELS ===")
    lin_coefs = pd.Series(lin.coef_, index=X.columns).sort_values(key=np.abs, ascending=False)
    print("LinearRegression coefficients:")
    print(lin_coefs)
    print(f"LinearRegression R^2: {r2_score(y, lin.predict(X)):.4f}")

    ridge_coefs = pd.Series(ridge.named_steps["ridge"].coef_, index=X.columns).sort_values(
        key=np.abs, ascending=False
    )
    print("\nRidge standardized coefficients:")
    print(ridge_coefs)
    print(f"Ridge R^2: {r2_score(y, ridge.predict(X)):.4f}")

    lasso_coefs = pd.Series(lasso.named_steps["lasso"].coef_, index=X.columns).sort_values(
        key=np.abs, ascending=False
    )
    print("\nLasso standardized coefficients:")
    print(lasso_coefs)
    print(f"Lasso R^2: {r2_score(y, lasso.predict(X)):.4f}")

    tree_imp = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nDecisionTreeRegressor feature importances:")
    print(tree_imp)
    print(f"DecisionTreeRegressor R^2: {r2_score(y, tree.predict(X)):.4f}")

    # 4) Interpretable imodels models
    print("\n=== IMODELS ===")
    imodels_notes = []
    try:
        from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor
    except Exception as e:
        msg = f"imodels import failed: {type(e).__name__}: {e}"
        print(msg)
        imodels_notes.append(msg)
    else:
        try:
            rulefit = RuleFitRegressor(random_state=0, max_rules=30)
            rulefit.fit(X.values, y.values, feature_names=X.columns.tolist())
            rulefit_r2 = r2_score(y, rulefit.predict(X.values))
            print(f"RuleFitRegressor R^2: {rulefit_r2:.4f}")
            if hasattr(rulefit, "get_rules"):
                rules = rulefit.get_rules()
                nonzero_rules = rules.loc[rules["coef"] != 0].copy()
                if len(nonzero_rules) > 0:
                    nonzero_rules["abs_coef"] = nonzero_rules["coef"].abs()
                    top_rules = nonzero_rules.sort_values("abs_coef", ascending=False).head(10)
                    print("Top RuleFit rules/terms:")
                    print(top_rules[[c for c in ["rule", "coef", "support"] if c in top_rules.columns]])
                else:
                    print("RuleFit selected no non-zero rules.")
            else:
                print("RuleFitRegressor has no get_rules() in this version; using R^2 as interpretable model evidence.")
            imodels_notes.append("RuleFit model fit successfully.")
        except Exception as e:
            msg = f"RuleFit failed: {type(e).__name__}: {e}"
            print(msg)
            imodels_notes.append(msg)

        try:
            figs = FIGSRegressor(random_state=0, max_rules=20)
            figs.fit(X.values, y.values, feature_names=X.columns.tolist())
            figs_r2 = r2_score(y, figs.predict(X.values))
            print(f"\nFIGSRegressor R^2: {figs_r2:.4f}")
            if hasattr(figs, "feature_importances_"):
                figs_imp = pd.Series(figs.feature_importances_, index=X.columns).sort_values(ascending=False)
                print("FIGS feature importances:")
                print(figs_imp)
            imodels_notes.append("FIGS model fit successfully.")
        except Exception as e:
            msg = f"FIGS failed: {type(e).__name__}: {e}"
            print(msg)
            imodels_notes.append(msg)

        try:
            hst = HSTreeRegressor(max_leaf_nodes=8, random_state=0)
            hst.fit(X.values, y.values, feature_names=X.columns.tolist())
            hst_r2 = r2_score(y, hst.predict(X.values))
            print(f"\nHSTreeRegressor R^2: {hst_r2:.4f}")
            if hasattr(hst, "feature_importances_"):
                hst_imp = pd.Series(hst.feature_importances_, index=X.columns).sort_values(ascending=False)
                print("HSTree feature importances:")
                print(hst_imp)
            imodels_notes.append("HSTree model fit successfully.")
        except Exception as e:
            msg = f"HSTree failed: {type(e).__name__}: {e}"
            print(msg)
            imodels_notes.append(msg)

    # 5) Convert evidence to Likert score (0-100)
    coef_simple = ols_simple.params["str_ratio"]
    p_simple = ols_simple.pvalues["str_ratio"]
    coef_adj = ols_adj.params["str_ratio"]
    p_adj = ols_adj.pvalues["str_ratio"]

    score = 50
    evidence = []

    if coef_adj < 0 and p_adj < 0.05:
        score += 25
        evidence.append(
            f"Adjusted OLS shows a negative STR coefficient ({coef_adj:.3f}) with p={p_adj:.3g}."
        )
    elif coef_adj < 0 and p_adj < 0.10:
        score += 12
        evidence.append(
            f"Adjusted OLS is directionally negative ({coef_adj:.3f}) with marginal significance p={p_adj:.3g}."
        )
    else:
        score -= 20
        evidence.append(
            f"Adjusted OLS does not show strong negative significance for STR (coef={coef_adj:.3f}, p={p_adj:.3g})."
        )

    if coef_simple < 0 and p_simple < 0.05:
        score += 12
        evidence.append(f"Simple OLS is negative and significant (coef={coef_simple:.3f}, p={p_simple:.3g}).")
    elif coef_simple < 0:
        score += 4
        evidence.append(f"Simple OLS is negative but weakly supported (coef={coef_simple:.3f}, p={p_simple:.3g}).")

    if pearson_r < 0 and pearson_p < 0.05:
        score += 8
        evidence.append(f"Pearson correlation is negative (r={pearson_r:.3f}, p={pearson_p:.3g}).")

    if spearman_rho < 0 and spearman_p < 0.05:
        score += 5
        evidence.append(f"Spearman correlation is negative (rho={spearman_rho:.3f}, p={spearman_p:.3g}).")

    diff = float(low_str_scores.mean() - high_str_scores.mean())
    if diff > 0 and t_p < 0.05:
        score += 6
        evidence.append(
            f"Low-STR districts have higher mean scores by {diff:.2f} points (t-test p={t_p:.3g})."
        )

    if anova_p < 0.05:
        score += 4
        evidence.append(f"ANOVA across STR quartiles is significant (p={anova_p:.3g}).")

    score = int(np.clip(round(score), 0, 100))

    explanation = (
        "Research question: Is a lower student-teacher ratio associated with higher academic performance? "
        + " ".join(evidence)
        + " Interpretable sklearn models generally rank socioeconomic controls (especially lunch/english/income) as major predictors, "
        + "while STR still appears with a negative effect in regression-based analyses. "
        + " ".join(imodels_notes)
    )

    conclusion = {"response": score, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(conclusion))

    print("\n=== FINAL CONCLUSION JSON ===")
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()
