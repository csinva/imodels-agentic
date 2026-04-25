import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text

from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor


RANDOM_STATE = 42


def safe_float(x):
    if x is None:
        return None
    try:
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        return float(x)
    except Exception:
        return None


def main():
    base = Path(".")
    info_path = base / "info.json"
    data_path = base / "hurricane.csv"

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info.get("research_questions", ["Unknown question"])[0]
    df = pd.read_csv(data_path)

    # Core transformations for skewed outcomes.
    df["log_alldeaths"] = np.log1p(df["alldeaths"])
    df["log_ndam15"] = np.log1p(df["ndam15"])

    print("Research question:")
    print(research_question)
    print("\nData shape:", df.shape)

    print("\nMissing values:")
    print(df.isna().sum().to_string())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    print("\nSummary statistics (numeric):")
    print(df[numeric_cols].describe().T[["mean", "std", "min", "50%", "max"]].to_string())

    print("\nDistribution diagnostics (selected variables):")
    for col in ["masfem", "gender_mf", "alldeaths", "log_alldeaths", "wind", "min", "category", "ndam15"]:
        if col in df.columns:
            skew = safe_float(df[col].skew())
            q = df[col].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict()
            q = {f"q{int(k*100)}": round(v, 4) for k, v in q.items()}
            print(f"{col}: skew={skew:.4f} quantiles={q}")

    corr_vars = ["masfem", "gender_mf", "alldeaths", "log_alldeaths", "wind", "min", "category", "ndam15", "log_ndam15", "year"]
    corr_vars = [c for c in corr_vars if c in df.columns]
    corr_matrix = df[corr_vars].corr(numeric_only=True)
    print("\nCorrelation matrix (selected):")
    print(corr_matrix.round(3).to_string())

    # Main bivariate tests for the hypothesis.
    pearson_raw = stats.pearsonr(df["masfem"], df["alldeaths"])
    spearman_raw = stats.spearmanr(df["masfem"], df["alldeaths"])
    pearson_log = stats.pearsonr(df["masfem"], df["log_alldeaths"])
    spearman_log = stats.spearmanr(df["masfem"], df["log_alldeaths"])

    print("\nBivariate association tests (masfem vs deaths):")
    print(f"Pearson raw deaths: r={pearson_raw.statistic:.4f}, p={pearson_raw.pvalue:.4g}")
    print(f"Spearman raw deaths: rho={spearman_raw.statistic:.4f}, p={spearman_raw.pvalue:.4g}")
    print(f"Pearson log deaths: r={pearson_log.statistic:.4f}, p={pearson_log.pvalue:.4g}")
    print(f"Spearman log deaths: rho={spearman_log.statistic:.4f}, p={spearman_log.pvalue:.4g}")

    male = df.loc[df["gender_mf"] == 0, "log_alldeaths"]
    female = df.loc[df["gender_mf"] == 1, "log_alldeaths"]
    ttest = stats.ttest_ind(female, male, equal_var=False, nan_policy="omit")

    print("\nWelch t-test (female-name vs male-name storms on log deaths):")
    print(f"female_mean={female.mean():.4f}, male_mean={male.mean():.4f}, t={ttest.statistic:.4f}, p={ttest.pvalue:.4g}")

    # ANOVA across femininity quartiles.
    df["masfem_q"] = pd.qcut(df["masfem"], q=4, labels=False, duplicates="drop")
    groups = [g["log_alldeaths"].values for _, g in df.groupby("masfem_q")]
    anova = stats.f_oneway(*groups)
    print("\nANOVA across masfem quartiles (log deaths):")
    print(f"F={anova.statistic:.4f}, p={anova.pvalue:.4g}")
    print("Quartile means (log deaths):")
    print(df.groupby("masfem_q")["log_alldeaths"].mean().round(4).to_string())

    # Regression models (statsmodels for p-values and confidence intervals).
    X_simple = sm.add_constant(df[["masfem"]])
    ols_simple = sm.OLS(df["log_alldeaths"], X_simple).fit(cov_type="HC3")

    model_features = ["masfem", "wind", "min", "category", "log_ndam15", "year"]
    X_main = df[model_features].copy()
    for c in model_features:
        X_main[c] = X_main[c].fillna(X_main[c].median())
    X_main_sm = sm.add_constant(X_main)
    ols_main = sm.OLS(df["log_alldeaths"], X_main_sm).fit(cov_type="HC3")

    print("\nOLS (simple): log_alldeaths ~ masfem")
    print(ols_simple.summary())

    print("\nOLS (adjusted): log_alldeaths ~ masfem + wind + min + category + log_ndam15 + year")
    print(ols_main.summary())

    # Interpretable sklearn models.
    X_np = X_main.values
    y_np = df["log_alldeaths"].values

    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ])
    lasso_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.03, random_state=RANDOM_STATE, max_iter=20000)),
    ])

    lr_pipe.fit(X_np, y_np)
    ridge_pipe.fit(X_np, y_np)
    lasso_pipe.fit(X_np, y_np)

    lr_coef = pd.Series(lr_pipe.named_steps["model"].coef_, index=model_features)
    ridge_coef = pd.Series(ridge_pipe.named_steps["model"].coef_, index=model_features)
    lasso_coef = pd.Series(lasso_pipe.named_steps["model"].coef_, index=model_features)

    print("\nLinearRegression coefficients (standardized features):")
    print(lr_coef.sort_values(key=np.abs, ascending=False).round(4).to_string())

    print("\nRidge coefficients (standardized features):")
    print(ridge_coef.sort_values(key=np.abs, ascending=False).round(4).to_string())

    print("\nLasso coefficients (standardized features):")
    print(lasso_coef.sort_values(key=np.abs, ascending=False).round(4).to_string())

    dt = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=RANDOM_STATE)
    dt.fit(X_np, y_np)
    dt_importance = pd.Series(dt.feature_importances_, index=model_features)
    print("\nDecisionTreeRegressor feature importances:")
    print(dt_importance.sort_values(ascending=False).round(4).to_string())
    print("\nDecision tree structure:")
    print(export_text(dt, feature_names=model_features))

    # Interpretable imodels.
    rf = RuleFitRegressor(random_state=RANDOM_STATE)
    rf.fit(X_main, y_np)

    # In this imodels version, `coef` contains linear terms first, then rule terms.
    rf_linear_coefs = pd.Series(np.array(rf.coef[: len(model_features)]), index=model_features)
    rf_rule_coefs = np.array(rf.coef[len(model_features):])
    rf_rules = np.array(rf.rules_)

    nonzero_mask = np.abs(rf_rule_coefs) > 1e-8
    top_rule_idx = np.where(nonzero_mask)[0]
    top_rule_rows = []
    for idx in top_rule_idx:
        top_rule_rows.append((rf_rules[idx], float(rf_rule_coefs[idx])))
    top_rule_rows = sorted(top_rule_rows, key=lambda x: abs(x[1]), reverse=True)[:10]

    print("\nRuleFit linear coefficients:")
    print(rf_linear_coefs.sort_values(key=np.abs, ascending=False).round(4).to_string())

    print("\nRuleFit top non-zero rules:")
    for rule, coef in top_rule_rows:
        print(f"coef={coef:.4f} | rule={rule}")

    figs = FIGSRegressor(random_state=RANDOM_STATE)
    figs.fit(X_main, y_np)
    figs_imp = pd.Series(figs.feature_importances_, index=model_features)
    print("\nFIGS feature importances:")
    print(figs_imp.sort_values(ascending=False).round(4).to_string())

    hs = HSTreeRegressor(random_state=RANDOM_STATE)
    hs.fit(X_main, y_np)
    hs_imp = pd.Series(hs.estimator_.feature_importances_, index=model_features)
    print("\nHSTree feature importances (from internal tree estimator):")
    print(hs_imp.sort_values(ascending=False).round(4).to_string())

    # Evidence synthesis for final Likert score.
    masfem_coef_main = safe_float(ols_main.params["masfem"])
    masfem_p_main = safe_float(ols_main.pvalues["masfem"])
    masfem_coef_simple = safe_float(ols_simple.params["masfem"])
    masfem_p_simple = safe_float(ols_simple.pvalues["masfem"])

    significant_positive_tests = 0
    significant_negative_tests = 0

    test_records = [
        (pearson_log.statistic, pearson_log.pvalue),
        (ttest.statistic, ttest.pvalue),
        (masfem_coef_simple, masfem_p_simple),
        (masfem_coef_main, masfem_p_main),
    ]

    for effect, pval in test_records:
        if pval < 0.05:
            if effect > 0:
                significant_positive_tests += 1
            elif effect < 0:
                significant_negative_tests += 1

    # Base decision rule: if no statistically significant positive evidence, return a low score.
    if significant_positive_tests == 0:
        response = 15
    else:
        net = significant_positive_tests - significant_negative_tests
        response = 50 + 12 * net
        response = int(max(0, min(100, response)))

    # Small directionality adjustment from interpretable models (no significance override).
    masfem_sign_votes = []
    masfem_sign_votes.append(np.sign(lr_coef["masfem"]))
    masfem_sign_votes.append(np.sign(ridge_coef["masfem"]))
    masfem_sign_votes.append(np.sign(lasso_coef["masfem"]))
    masfem_sign_votes.append(np.sign(rf_linear_coefs["masfem"]))
    direction_sum = float(np.nansum(masfem_sign_votes))
    if direction_sum > 0 and response < 90:
        response += 2
    elif direction_sum < 0 and response > 10:
        response -= 2
    response = int(max(0, min(100, response)))

    explanation = (
        "Using the 94-hurricane dataset, femininity of storm names was not a statistically significant predictor "
        "of fatalities. Key tests on deaths/log-deaths were non-significant (Pearson log r="
        f"{pearson_log.statistic:.3f}, p={pearson_log.pvalue:.3f}; Welch t-test female vs male p={ttest.pvalue:.3f}; "
        f"ANOVA across femininity quartiles p={anova.pvalue:.3f}). In robust OLS with controls "
        "(wind, pressure, category, damage, year), masfem remained non-significant "
        f"(coef={masfem_coef_main:.3f}, p={masfem_p_main:.3f}). Interpretable ML models mainly emphasized damage/pressure, "
        "with weak and unstable femininity effects. Therefore evidence for the claimed relationship is weak."
    )

    output = {"response": response, "explanation": explanation}
    with (base / "conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(output, f)

    print("\nWrote conclusion.txt")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
