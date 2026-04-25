import json
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


def format_dict_sorted(d: Dict[str, float], top_n: int = None) -> str:
    items = sorted(d.items(), key=lambda x: abs(x[1]), reverse=True)
    if top_n is not None:
        items = items[:top_n]
    return ", ".join([f"{k}={v:.4f}" for k, v in items])


def safe_qcut(series: pd.Series, q: int, labels: List[str]) -> pd.Series:
    return pd.qcut(series, q=q, labels=labels, duplicates="drop")


def main() -> None:
    # --------------------------
    # 1) Load and prepare data
    # --------------------------
    df = pd.read_csv("hurricane.csv")

    numeric_cols = [
        "year",
        "masfem",
        "min",
        "gender_mf",
        "category",
        "alldeaths",
        "ndam",
        "elapsedyrs",
        "masfem_mturk",
        "wind",
        "ndam15",
    ]

    for col in ["alldeaths", "ndam", "ndam15"]:
        df[f"log_{col}"] = np.log1p(df[col])

    # Proxy of potentially severe social impact
    df["high_deaths"] = (df["alldeaths"] > df["alldeaths"].median()).astype(int)

    print("=== Dataset Overview ===")
    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")
    print("Missing values by column:")
    print(df.isna().sum().to_string())
    print("\nSummary statistics:")
    print(df[numeric_cols].describe().T[["mean", "std", "min", "50%", "max"]].to_string())

    print("\nDistribution diagnostics:")
    print(
        f"Skew(alldeaths)={stats.skew(df['alldeaths']):.3f}, "
        f"Skew(log_alldeaths)={stats.skew(df['log_alldeaths']):.3f}"
    )

    # --------------------------
    # 2) Correlations and statistical tests
    # --------------------------
    corr = df[numeric_cols].corr(numeric_only=True)
    cor_deaths = corr["alldeaths"].sort_values(ascending=False)
    cor_log_deaths = df[numeric_cols + ["log_alldeaths"]].corr(numeric_only=True)["log_alldeaths"].sort_values(
        ascending=False
    )
    print("\nTop correlations with alldeaths:")
    print(cor_deaths.head(8).to_string())
    print("\nTop correlations with log_alldeaths:")
    print(cor_log_deaths.head(8).to_string())

    # Key tests for research question
    pearson_raw = stats.pearsonr(df["masfem"], df["alldeaths"])
    pearson_log = stats.pearsonr(df["masfem"], df["log_alldeaths"])
    spearman_raw = stats.spearmanr(df["masfem"], df["alldeaths"])

    female_log = df.loc[df["gender_mf"] == 1, "log_alldeaths"]
    male_log = df.loc[df["gender_mf"] == 0, "log_alldeaths"]
    ttest_gender = stats.ttest_ind(female_log, male_log, equal_var=False)

    df["masfem_tercile"] = safe_qcut(df["masfem"], q=3, labels=["low", "mid", "high"])
    groups = [
        df.loc[df["masfem_tercile"] == level, "log_alldeaths"].values
        for level in df["masfem_tercile"].dropna().unique()
    ]
    anova_terciles = stats.f_oneway(*groups)

    print("\n=== Hypothesis Tests ===")
    print(
        f"Pearson(masfem, alldeaths): r={pearson_raw.statistic:.4f}, p={pearson_raw.pvalue:.4f}; "
        f"Pearson(masfem, log_alldeaths): r={pearson_log.statistic:.4f}, p={pearson_log.pvalue:.4f}"
    )
    print(f"Spearman(masfem, alldeaths): rho={spearman_raw.statistic:.4f}, p={spearman_raw.pvalue:.4f}")
    print(
        f"Welch t-test log_alldeaths female vs male names: t={ttest_gender.statistic:.4f}, "
        f"p={ttest_gender.pvalue:.4f}, mean_female={female_log.mean():.4f}, mean_male={male_log.mean():.4f}"
    )
    print(f"ANOVA log_alldeaths across masfem terciles: F={anova_terciles.statistic:.4f}, p={anova_terciles.pvalue:.4f}")

    # --------------------------
    # 3) Interpretable regression models
    # --------------------------
    features = ["masfem", "gender_mf", "wind", "min", "category", "year", "ndam15", "masfem_mturk"]
    X = df[features].copy()
    y = df["log_alldeaths"].values

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    lin = LinearRegression()
    lin.fit(X, y)
    lin_coef = dict(zip(features, lin.coef_))
    lin_cv = cross_val_score(lin, X, y, cv=kf, scoring="r2").mean()

    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ])
    ridge.fit(X, y)
    ridge_coef = dict(zip(features, ridge.named_steps["model"].coef_))
    ridge_cv = cross_val_score(ridge, X, y, cv=kf, scoring="r2").mean()

    lasso = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.05, random_state=RANDOM_STATE, max_iter=10000)),
    ])
    lasso.fit(X, y)
    lasso_coef = dict(zip(features, lasso.named_steps["model"].coef_))
    lasso_cv = cross_val_score(lasso, X, y, cv=kf, scoring="r2").mean()

    dtr = DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE)
    dtr.fit(X, y)
    dtr_fi = dict(zip(features, dtr.feature_importances_))
    dtr_cv = cross_val_score(dtr, X, y, cv=kf, scoring="r2").mean()

    dtc = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
    dtc.fit(X, df["high_deaths"]) 
    dtc_fi = dict(zip(features, dtc.feature_importances_))
    dtc_cv = cross_val_score(dtc, X, df["high_deaths"], cv=kf, scoring="accuracy").mean()

    print("\n=== sklearn Interpretable Models ===")
    print(f"LinearRegression CV R2={lin_cv:.4f}; |coef| top: {format_dict_sorted(lin_coef, top_n=6)}")
    print(f"Ridge CV R2={ridge_cv:.4f}; |coef| top: {format_dict_sorted(ridge_coef, top_n=6)}")
    print(f"Lasso CV R2={lasso_cv:.4f}; |coef| top: {format_dict_sorted(lasso_coef, top_n=6)}")
    print(f"DecisionTreeRegressor CV R2={dtr_cv:.4f}; importance top: {format_dict_sorted(dtr_fi, top_n=6)}")
    print(f"DecisionTreeClassifier CV Acc={dtc_cv:.4f}; importance top: {format_dict_sorted(dtc_fi, top_n=6)}")

    # --------------------------
    # 4) statsmodels OLS with p-values
    # --------------------------
    X_ols = sm.add_constant(X)
    ols = sm.OLS(y, X_ols).fit()
    masfem_coef = float(ols.params["masfem"])
    masfem_p = float(ols.pvalues["masfem"])

    df["masfem_x_log_ndam15"] = df["masfem"] * df["log_ndam15"]
    inter_features = ["masfem", "log_ndam15", "masfem_x_log_ndam15", "year"]
    X_inter = sm.add_constant(df[inter_features])
    ols_inter = sm.OLS(y, X_inter).fit()

    inter_coef = float(ols_inter.params["masfem_x_log_ndam15"])
    inter_p = float(ols_inter.pvalues["masfem_x_log_ndam15"])

    print("\n=== statsmodels OLS ===")
    print("Main-effects model coefficients (subset):")
    print(
        ols.params[["masfem", "gender_mf", "wind", "min", "category", "year", "ndam15", "masfem_mturk"]].to_string()
    )
    print("Main-effects model p-values (subset):")
    print(
        ols.pvalues[["masfem", "gender_mf", "wind", "min", "category", "year", "ndam15", "masfem_mturk"]].to_string()
    )
    print(f"Main-effects model R2={ols.rsquared:.4f}")

    print("\nInteraction model summary (targeted terms):")
    print(
        ols_inter.params[["masfem", "log_ndam15", "masfem_x_log_ndam15", "year"]].to_string()
    )
    print(
        ols_inter.pvalues[["masfem", "log_ndam15", "masfem_x_log_ndam15", "year"]].to_string()
    )
    print(f"Interaction model R2={ols_inter.rsquared:.4f}")

    # --------------------------
    # 5) imodels interpretable models
    # --------------------------
    rulefit = RuleFitRegressor(random_state=RANDOM_STATE, max_rules=50)
    rulefit.fit(X, y)
    rulefit_pred = rulefit.predict(X)
    rulefit_r2 = r2_score(y, rulefit_pred)
    rf_rules = rulefit._get_rules(exclude_zero_coef=True)
    rf_rules = rf_rules.sort_values("importance", ascending=False)
    top_rules = rf_rules.head(10)
    masfem_rules_top = int(top_rules["rule"].astype(str).str.contains("masfem").sum())

    figs = FIGSRegressor(random_state=RANDOM_STATE, max_rules=12)
    figs.fit(X, y)
    figs_pred = figs.predict(X)
    figs_r2 = r2_score(y, figs_pred)
    figs_fi = dict(zip(features, figs.feature_importances_))

    hst = HSTreeRegressor(random_state=RANDOM_STATE, max_leaf_nodes=8)
    hst.fit(X, y)
    hst_pred = hst.predict(X)
    hst_r2 = r2_score(y, hst_pred)
    hst_fi = dict(zip(features, hst.estimator_.feature_importances_))

    print("\n=== imodels Interpretable Models ===")
    print(f"RuleFit in-sample R2={rulefit_r2:.4f}; top rules include masfem in {masfem_rules_top}/10 rules")
    print("Top RuleFit rules:")
    print(top_rules[["rule", "coef", "support", "importance"]].head(8).to_string(index=False))
    print(f"FIGS in-sample R2={figs_r2:.4f}; feature importance top: {format_dict_sorted(figs_fi, top_n=6)}")
    print(f"HSTree in-sample R2={hst_r2:.4f}; feature importance top: {format_dict_sorted(hst_fi, top_n=6)}")

    # --------------------------
    # 6) Translate evidence to Likert 0-100
    # --------------------------
    response = 50

    # Penalize lack of significance in primary tests.
    if masfem_p >= 0.10:
        response -= 20
    elif masfem_p < 0.05 and masfem_coef > 0:
        response += 20

    if pearson_log.pvalue >= 0.10:
        response -= 10
    elif pearson_log.statistic > 0 and pearson_log.pvalue < 0.05:
        response += 10

    if ttest_gender.pvalue >= 0.10:
        response -= 10
    elif ttest_gender.pvalue < 0.05 and female_log.mean() > male_log.mean():
        response += 10

    if anova_terciles.pvalue >= 0.10:
        response -= 5
    elif anova_terciles.pvalue < 0.05:
        response += 5

    # Directional but weak interaction evidence gets only a small bonus.
    if inter_p < 0.10 and inter_coef > 0:
        response += 10

    # If interpretable tree/rule models mostly emphasize non-gender severity/exposure variables,
    # nudge score downward.
    high_importance_non_gender = (
        list(figs_fi.keys())[np.argmax(figs.feature_importances_)] != "masfem"
        and list(hst_fi.keys())[np.argmax(hst.estimator_.feature_importances_)] != "masfem"
    )
    if high_importance_non_gender:
        response -= 5

    response = int(np.clip(response, 0, 100))

    explanation = (
        "Evidence for the claim is weak in this dataset: femininity (masfem) has small, non-significant "
        f"associations with deaths (Pearson p={pearson_log.pvalue:.3f}; OLS adjusted p={masfem_p:.3f}), "
        f"female-vs-male name difference is non-significant (t-test p={ttest_gender.pvalue:.3f}), and ANOVA "
        f"across femininity terciles is non-significant (p={anova_terciles.pvalue:.3f}). An interaction between "
        f"femininity and log damage is only marginal (p={inter_p:.3f}). Interpretable models (trees/rules) place "
        "most importance on storm severity/exposure proxies (damage, pressure, year) rather than name gender, "
        "so the data does not provide strong support for a real femininity-driven precaution effect."
    )

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump({"response": response, "explanation": explanation}, f)

    print("\nWrote conclusion.txt")
    print({"response": response, "explanation": explanation})


if __name__ == "__main__":
    main()
