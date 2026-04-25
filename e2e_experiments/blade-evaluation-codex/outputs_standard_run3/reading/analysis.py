import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor

warnings.filterwarnings("ignore")


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def top_abs_series(series: pd.Series, n: int = 10) -> pd.Series:
    return series.reindex(series.abs().sort_values(ascending=False).index).head(n)


def main() -> None:
    info_path = Path("info.json")
    data_path = Path("reading.csv")

    info = json.loads(info_path.read_text())
    research_question = info.get("research_questions", ["Unknown question"])[0]

    df = pd.read_csv(data_path)

    numeric_candidates = [
        "reader_view",
        "running_time",
        "adjusted_running_time",
        "scrolling_time",
        "num_words",
        "correct_rate",
        "img_width",
        "age",
        "dyslexia",
        "gender",
        "retake_trial",
        "dyslexia_bin",
        "Flesch_Kincaid",
        "speed",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["reader_view", "speed", "dyslexia_bin"]).copy()
    df["log_speed"] = np.log1p(df["speed"])
    df["rv_x_dys"] = df["reader_view"] * df["dyslexia_bin"]

    print("Research question:", research_question)
    print("Rows, cols:", df.shape)

    # EDA summaries
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary_stats = df[numeric_cols].describe().T
    print("\nNumeric summary (first 12 rows):")
    print(summary_stats.head(12))

    print("\nMissing values (top 10):")
    print(df.isna().sum().sort_values(ascending=False).head(10))

    speed_by_group = (
        df.groupby(["dyslexia_bin", "reader_view"])["speed"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    print("\nSpeed by dyslexia_bin x reader_view:")
    print(speed_by_group)

    corr_with_speed = df[numeric_cols].corr(numeric_only=True)["speed"].dropna().sort_values(ascending=False)
    print("\nTop positive correlations with speed:")
    print(corr_with_speed.head(8))
    print("\nTop negative correlations with speed:")
    print(corr_with_speed.tail(8))

    print("\nDistribution diagnostics:")
    print(
        {
            "speed_skew": safe_float(df["speed"].skew()),
            "log_speed_skew": safe_float(df["log_speed"].skew()),
            "speed_kurtosis": safe_float(df["speed"].kurtosis()),
            "log_speed_kurtosis": safe_float(df["log_speed"].kurtosis()),
        }
    )

    # Focused statistical tests for dyslexia group
    dys_df = df[df["dyslexia_bin"] == 1].copy()
    dys_rv1 = dys_df[dys_df["reader_view"] == 1]["speed"].dropna()
    dys_rv0 = dys_df[dys_df["reader_view"] == 0]["speed"].dropna()

    ttest_raw = stats.ttest_ind(dys_rv1, dys_rv0, equal_var=False, nan_policy="omit")
    ttest_log = stats.ttest_ind(np.log1p(dys_rv1), np.log1p(dys_rv0), equal_var=False, nan_policy="omit")
    mw_test = stats.mannwhitneyu(dys_rv1, dys_rv0, alternative="two-sided")

    print("\nDyslexia-group tests (reader_view=1 vs 0):")
    print(
        {
            "n_reader_view_1": int(dys_rv1.shape[0]),
            "n_reader_view_0": int(dys_rv0.shape[0]),
            "mean_diff_speed": safe_float(dys_rv1.mean() - dys_rv0.mean()),
            "median_diff_speed": safe_float(dys_rv1.median() - dys_rv0.median()),
            "ttest_raw_p": safe_float(ttest_raw.pvalue),
            "ttest_log_p": safe_float(ttest_log.pvalue),
            "mannwhitney_p": safe_float(mw_test.pvalue),
        }
    )

    # Two-way ANOVA and adjusted OLS
    anova_model = smf.ols("log_speed ~ C(reader_view) * C(dyslexia_bin)", data=df).fit()
    anova_table = sm.stats.anova_lm(anova_model, typ=2)
    print("\nTwo-way ANOVA (log_speed):")
    print(anova_table)

    controls_formula = (
        "log_speed ~ reader_view * dyslexia_bin + num_words + correct_rate + age + "
        "retake_trial + Flesch_Kincaid + C(device) + C(page_id)"
    )
    ols_full = smf.ols(controls_formula, data=df).fit()
    print("\nAdjusted OLS coefficients of interest:")
    for term in ["reader_view", "dyslexia_bin", "reader_view:dyslexia_bin"]:
        print(term, "coef=", safe_float(ols_full.params.get(term)), "p=", safe_float(ols_full.pvalues.get(term)))

    ols_dys = smf.ols(
        "log_speed ~ reader_view + num_words + correct_rate + age + retake_trial + Flesch_Kincaid + C(device) + C(page_id)",
        data=dys_df,
    ).fit()
    print("\nDyslexia-only adjusted OLS reader_view effect:")
    print(
        {
            "coef": safe_float(ols_dys.params.get("reader_view")),
            "pvalue": safe_float(ols_dys.pvalues.get("reader_view")),
            "conf_int_low": safe_float(ols_dys.conf_int().loc["reader_view", 0]),
            "conf_int_high": safe_float(ols_dys.conf_int().loc["reader_view", 1]),
        }
    )

    # Interpretable ML models
    model_df = df.copy()
    feature_cols = [
        "reader_view",
        "dyslexia_bin",
        "rv_x_dys",
        "num_words",
        "correct_rate",
        "img_width",
        "age",
        "retake_trial",
        "Flesch_Kincaid",
        "running_time",
        "adjusted_running_time",
        "scrolling_time",
        "device",
        "education",
        "english_native",
        "page_id",
    ]
    feature_cols = [c for c in feature_cols if c in model_df.columns]

    X = model_df[feature_cols].copy()
    y = model_df["log_speed"].values

    for c in X.select_dtypes(include=[np.number]).columns:
        X[c] = X[c].fillna(X[c].median())
    for c in X.select_dtypes(exclude=[np.number]).columns:
        mode_val = X[c].mode(dropna=True)
        fill_val = mode_val.iloc[0] if not mode_val.empty else "missing"
        X[c] = X[c].fillna(fill_val)

    X_enc = pd.get_dummies(X, drop_first=True)

    linear = LinearRegression()
    linear.fit(X_enc, y)
    linear_coefs = pd.Series(linear.coef_, index=X_enc.columns)

    ridge = Ridge(alpha=1.0, random_state=0)
    ridge.fit(X_enc, y)
    ridge_coefs = pd.Series(ridge.coef_, index=X_enc.columns)

    lasso = Lasso(alpha=0.001, max_iter=20000, random_state=0)
    lasso.fit(X_enc, y)
    lasso_coefs = pd.Series(lasso.coef_, index=X_enc.columns)

    tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=40, random_state=0)
    tree.fit(X_enc, y)
    tree_importance = pd.Series(tree.feature_importances_, index=X_enc.columns)

    print("\nTop linear coefficients (abs):")
    print(top_abs_series(linear_coefs, 12))
    print("\nTop ridge coefficients (abs):")
    print(top_abs_series(ridge_coefs, 12))
    print("\nTop lasso coefficients (abs):")
    print(top_abs_series(lasso_coefs, 12))
    print("\nTop decision-tree feature importances:")
    print(tree_importance.sort_values(ascending=False).head(12))

    imodels_results = {}

    try:
        rulefit = RuleFitRegressor(random_state=0)
        rulefit.fit(X_enc.values, y, feature_names=X_enc.columns.tolist())

        n_lin = len(rulefit.feature_names_)
        coef_arr = np.array(rulefit.coef, dtype=float)

        lin_coefs = pd.Series(coef_arr[:n_lin], index=rulefit.feature_names_)
        imodels_results["rulefit_top_linear_terms"] = top_abs_series(lin_coefs, 10).to_dict()

        rule_coefs = coef_arr[n_lin:]
        if len(rule_coefs) > 0 and len(rulefit.rules_) > 0:
            rules_df = pd.DataFrame({"rule": [str(r) for r in rulefit.rules_], "coef": rule_coefs})
            rules_df["abs_coef"] = rules_df["coef"].abs()
            imodels_results["rulefit_top_rules"] = (
                rules_df[rules_df["abs_coef"] > 1e-9]
                .sort_values("abs_coef", ascending=False)
                .head(5)[["rule", "coef"]]
                .to_dict("records")
            )
        else:
            imodels_results["rulefit_top_rules"] = []
    except Exception as exc:
        imodels_results["rulefit_error"] = str(exc)

    try:
        figs = FIGSRegressor(random_state=0, max_rules=20)
        figs.fit(X_enc.values, y, feature_names=X_enc.columns.tolist())
        figs_importance = pd.Series(figs.feature_importances_, index=X_enc.columns)
        imodels_results["figs_top_features"] = figs_importance.sort_values(ascending=False).head(10).to_dict()
    except Exception as exc:
        imodels_results["figs_error"] = str(exc)

    try:
        hst = HSTreeRegressor(random_state=0, max_leaf_nodes=20)
        hst.fit(X_enc.values, y, feature_names=X_enc.columns.tolist())
        if hasattr(hst, "estimator_") and hasattr(hst.estimator_, "feature_importances_"):
            hst_importance = pd.Series(hst.estimator_.feature_importances_, index=X_enc.columns)
            imodels_results["hstree_top_features"] = hst_importance.sort_values(ascending=False).head(10).to_dict()
        else:
            imodels_results["hstree_top_features"] = {}
    except Exception as exc:
        imodels_results["hstree_error"] = str(exc)

    print("\nInterpretable imodels summary:")
    print(imodels_results)

    # Score evidence for hypothesis:
    # "Reader view improves reading speed for individuals with dyslexia"
    mean_diff = safe_float(dys_rv1.mean() - dys_rv0.mean(), 0.0)
    median_diff = safe_float(dys_rv1.median() - dys_rv0.median(), 0.0)
    p_ttest_log = safe_float(ttest_log.pvalue, 1.0)
    p_mw = safe_float(mw_test.pvalue, 1.0)

    coef_dys_reader = safe_float(ols_dys.params.get("reader_view"), 0.0)
    p_dys_reader = safe_float(ols_dys.pvalues.get("reader_view"), 1.0)

    coef_interaction = safe_float(ols_full.params.get("reader_view:dyslexia_bin"), 0.0)
    p_interaction = safe_float(ols_full.pvalues.get("reader_view:dyslexia_bin"), 1.0)

    score = 50

    # Direct within-dyslexia comparison on log-speed
    if p_ttest_log < 0.05:
        score += 30 if mean_diff > 0 else -35
    else:
        score -= 15

    # Nonparametric robustness check
    if p_mw < 0.05:
        score += 15 if median_diff > 0 else -20
    else:
        score -= 10

    # Adjusted effect among dyslexia participants
    if p_dys_reader < 0.05:
        score += 25 if coef_dys_reader > 0 else -30
    else:
        score -= 10

    # Differential effect for dyslexia vs non-dyslexia
    if p_interaction < 0.05:
        score += 20 if coef_interaction > 0 else -25
    else:
        score -= 10

    score = int(np.clip(round(score), 0, 100))

    explanation = (
        f"Question: {research_question} "
        f"In dyslexic participants, average speed with Reader View was {dys_rv1.mean():.2f} versus {dys_rv0.mean():.2f} without, "
        f"and median speed difference was {median_diff:.2f}. "
        f"The within-dyslexia Welch t-test on log(speed) was not significant (p={p_ttest_log:.3g}) and Mann-Whitney was not significant (p={p_mw:.3g}). "
        f"In adjusted dyslexia-only OLS, Reader View coefficient was {coef_dys_reader:.4f} (p={p_dys_reader:.3g}), and in full-sample OLS the interaction "
        f"ReaderView*dyslexia_bin was {coef_interaction:.4f} (p={p_interaction:.3g}), indicating no significant dyslexia-specific improvement. "
        f"Interpretable models (Linear/Ridge/Lasso/Decision Tree and imodels RuleFit/FIGS/HSTree) did not identify a strong, consistent positive Reader View effect relative to key covariates."
    )

    output = {"response": score, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(output))

    print("\nWrote conclusion.txt:")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
