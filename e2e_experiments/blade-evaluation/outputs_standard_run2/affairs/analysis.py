import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from imodels import RuleFitRegressor, FIGSRegressor, HSTreeRegressor


RANDOM_STATE = 42


def safe_corr(df_num: pd.DataFrame, target: str, top_n: int = 8):
    corr = df_num.corr(numeric_only=True)[target].sort_values(key=np.abs, ascending=False)
    return corr.head(top_n)


def format_feature_weights(series: pd.Series, top_n: int = 8):
    s = series.sort_values(key=np.abs, ascending=False).head(top_n)
    return {k: float(v) for k, v in s.items()}


def main():
    info = json.loads(Path("info.json").read_text())
    question = info.get("research_questions", [""])[0].strip()
    print(f"Research question: {question}")

    df = pd.read_csv("affairs.csv")
    print("\nData shape:", df.shape)
    print("\nColumns:", list(df.columns))

    # Basic preprocessing
    df["children_yes"] = (df["children"].str.lower() == "yes").astype(int)
    df["affair_binary"] = (df["affairs"] > 0).astype(int)
    df["gender_male"] = (df["gender"].str.lower() == "male").astype(int)

    print("\nSummary stats (numeric):")
    print(df.describe().T)

    print("\nDistribution of affairs:")
    print(df["affairs"].value_counts().sort_index())

    print("\nGroup means by children:")
    print(df.groupby("children")["affairs"].agg(["count", "mean", "median", "std"]))

    # Correlation exploration
    numeric_cols = [
        "affairs",
        "affair_binary",
        "children_yes",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
        "gender_male",
    ]
    corr_top = safe_corr(df[numeric_cols], target="affairs", top_n=10)
    print("\nTop correlations with affairs:")
    print(corr_top)

    # Statistical tests directly on question variable
    affairs_child_yes = df.loc[df["children_yes"] == 1, "affairs"]
    affairs_child_no = df.loc[df["children_yes"] == 0, "affairs"]

    t_stat, t_p = stats.ttest_ind(affairs_child_yes, affairs_child_no, equal_var=False)
    mw_u, mw_p = stats.mannwhitneyu(affairs_child_yes, affairs_child_no, alternative="two-sided")

    contingency = pd.crosstab(df["children_yes"], df["affair_binary"])
    chi2, chi2_p, _, _ = stats.chi2_contingency(contingency)

    print("\nTwo-sample tests (children yes vs no on affairs):")
    print(f"Welch t-test: t={t_stat:.4f}, p={t_p:.6f}")
    print(f"Mann-Whitney U: U={mw_u:.4f}, p={mw_p:.6f}")
    print("\nChi-square test (children vs any-affair):")
    print(contingency)
    print(f"chi2={chi2:.4f}, p={chi2_p:.6f}")

    # OLS regression with controls
    ols_features = [
        "children_yes",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
        "gender_male",
    ]
    X_ols = sm.add_constant(df[ols_features])
    y = df["affairs"]
    ols_model = sm.OLS(y, X_ols).fit()

    print("\nOLS coefficient table:")
    print(ols_model.summary().tables[1])

    # Logistic regression for any affair
    y_bin = df["affair_binary"]
    logit_model = sm.Logit(y_bin, X_ols).fit(disp=False)
    print("\nLogit coefficient table (any affair):")
    print(logit_model.summary().tables[1])

    # scikit-learn interpretable models
    X_ml = pd.get_dummies(
        df[["gender", "age", "yearsmarried", "children", "religiousness", "education", "occupation", "rating"]],
        drop_first=True,
    )
    y_ml = df["affairs"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_ml, y_ml, test_size=0.3, random_state=RANDOM_STATE
    )

    lr = LinearRegression().fit(X_train, y_train)
    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE).fit(X_train, y_train)
    lasso = Lasso(alpha=0.01, random_state=RANDOM_STATE, max_iter=10000).fit(X_train, y_train)
    dtr = DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE).fit(X_train, y_train)

    y_train_bin = (y_train > 0).astype(int)
    y_test_bin = (y_test > 0).astype(int)
    dtc = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE).fit(X_train, y_train_bin)

    print("\nModel performance:")
    print(f"LinearRegression R2(test): {r2_score(y_test, lr.predict(X_test)):.4f}")
    print(f"Ridge R2(test): {r2_score(y_test, ridge.predict(X_test)):.4f}")
    print(f"Lasso R2(test): {r2_score(y_test, lasso.predict(X_test)):.4f}")
    print(f"DecisionTreeRegressor R2(test): {r2_score(y_test, dtr.predict(X_test)):.4f}")
    print(f"DecisionTreeClassifier Accuracy(test, any-affair): {accuracy_score(y_test_bin, dtc.predict(X_test)):.4f}")

    lr_coef = pd.Series(lr.coef_, index=X_ml.columns)
    ridge_coef = pd.Series(ridge.coef_, index=X_ml.columns)
    lasso_coef = pd.Series(lasso.coef_, index=X_ml.columns)
    dtr_imp = pd.Series(dtr.feature_importances_, index=X_ml.columns)
    dtc_imp = pd.Series(dtc.feature_importances_, index=X_ml.columns)

    print("\nTop linear coefficients (LR):")
    print(format_feature_weights(lr_coef))
    print("\nTop linear coefficients (Ridge):")
    print(format_feature_weights(ridge_coef))
    print("\nTop linear coefficients (Lasso):")
    print(format_feature_weights(lasso_coef))
    print("\nTop feature importances (DecisionTreeRegressor):")
    print(format_feature_weights(dtr_imp))
    print("\nTop feature importances (DecisionTreeClassifier):")
    print(format_feature_weights(dtc_imp))

    # imodels interpretable models
    rf = RuleFitRegressor(random_state=RANDOM_STATE, max_rules=30).fit(X_train.values, y_train.values, feature_names=list(X_train.columns))
    figs = FIGSRegressor(random_state=RANDOM_STATE, max_rules=12).fit(X_train.values, y_train.values, feature_names=list(X_train.columns))
    hst = HSTreeRegressor(random_state=RANDOM_STATE, max_leaf_nodes=20).fit(X_train.values, y_train.values, feature_names=list(X_train.columns))

    print("\nimodels performance:")
    print(f"RuleFitRegressor R2(test): {r2_score(y_test, rf.predict(X_test.values)):.4f}")
    print(f"FIGSRegressor R2(test): {r2_score(y_test, figs.predict(X_test.values)):.4f}")
    print(f"HSTreeRegressor R2(test): {r2_score(y_test, hst.predict(X_test.values)):.4f}")

    try:
        rules_df = rf.get_rules()
        active_rules = rules_df[(rules_df["coef"] != 0) & (rules_df["type"] == "rule")].copy()
        active_rules = active_rules.sort_values("importance", ascending=False).head(8)
        print("\nTop RuleFit rules:")
        if active_rules.empty:
            print("No active non-zero rules found.")
        else:
            print(active_rules[["rule", "coef", "support", "importance"]])
    except Exception as e:
        print(f"Could not extract RuleFit rules: {e}")

    # Evidence synthesis for final Likert score
    children_mean_diff = affairs_child_yes.mean() - affairs_child_no.mean()  # negative would support decrease
    ols_children_coef = float(ols_model.params["children_yes"])
    ols_children_p = float(ols_model.pvalues["children_yes"])
    logit_children_coef = float(logit_model.params["children_yes"])
    logit_children_p = float(logit_model.pvalues["children_yes"])

    child_feature_name = "children_yes" if "children_yes" in X_ml.columns else None
    lr_child_coef = float(lr_coef.get(child_feature_name, np.nan))
    ridge_child_coef = float(ridge_coef.get(child_feature_name, np.nan))
    lasso_child_coef = float(lasso_coef.get(child_feature_name, np.nan))

    # Score calibration anchored on statistical significance + direction consistency.
    score = 50

    # Direction from raw mean difference
    if children_mean_diff < 0:
        score += 8
    else:
        score -= 8

    # Significance from direct tests
    if t_p < 0.05:
        score += 15 if children_mean_diff < 0 else -15
    if mw_p < 0.05:
        score += 10 if children_mean_diff < 0 else -10
    if chi2_p < 0.05:
        score += 8 if logit_children_coef < 0 else -8

    # Regression-controlled evidence
    if ols_children_p < 0.05:
        score += 18 if ols_children_coef < 0 else -18
    if logit_children_p < 0.05:
        score += 18 if logit_children_coef < 0 else -18

    # Interpretable model directional consistency
    directional_votes = 0
    total_votes = 0
    for coef in [lr_child_coef, ridge_child_coef, lasso_child_coef]:
        if np.isfinite(coef):
            total_votes += 1
            if coef < 0:
                directional_votes += 1
    if total_votes:
        vote_ratio = directional_votes / total_votes
        if vote_ratio >= 2 / 3:
            score += 8
        elif vote_ratio <= 1 / 3:
            score -= 8

    score = int(np.clip(round(score), 0, 100))

    explanation = (
        f"Question: {question} "
        f"Mean affairs (children=yes)={affairs_child_yes.mean():.3f} vs (children=no)={affairs_child_no.mean():.3f} "
        f"(difference yes-no={children_mean_diff:.3f}). "
        f"Welch t-test p={t_p:.4g}, Mann-Whitney p={mw_p:.4g}, chi-square p={chi2_p:.4g}. "
        f"Controlled OLS children coefficient={ols_children_coef:.3f} (p={ols_children_p:.4g}); "
        f"Logit(any affair) children coefficient={logit_children_coef:.3f} (p={logit_children_p:.4g}). "
        f"Interpretable sklearn linear coefficients for children were: "
        f"LinearRegression={lr_child_coef:.3f}, Ridge={ridge_child_coef:.3f}, Lasso={lasso_child_coef:.3f}. "
        f"These results indicate the direction and significance of association used to compute the Likert score."
    )

    result = {"response": score, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(result))
    print("\nWrote conclusion.txt:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
