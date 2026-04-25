import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor

warnings.filterwarnings("ignore")


SEED = 42
np.random.seed(SEED)


def clamp_int(value: float, lo: int = 0, hi: int = 100) -> int:
    return int(max(lo, min(hi, round(value))))


def safe_mean(series: pd.Series) -> float:
    return float(np.nanmean(series.values)) if len(series) else np.nan


def top_abs(series: pd.Series, n: int = 8) -> pd.Series:
    return series.reindex(series.abs().sort_values(ascending=False).head(n).index)


def main() -> None:
    info = json.loads(Path("info.json").read_text())
    research_question = info.get("research_questions", ["Unknown question"])[0]
    print(f"Research question: {research_question}")

    df = pd.read_csv("reading.csv")
    print(f"Loaded reading.csv with shape={df.shape}")

    # Core numeric cleanup for analysis
    numeric_expected = [
        "reader_view",
        "speed",
        "dyslexia_bin",
        "dyslexia",
        "num_words",
        "Flesch_Kincaid",
        "retake_trial",
        "age",
        "correct_rate",
        "img_width",
        "scrolling_time",
    ]
    for col in numeric_expected:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["speed"] > 0].copy()
    df["log_speed"] = np.log(df["speed"])

    print("\n=== Missingness (top 10) ===")
    print(df.isna().sum().sort_values(ascending=False).head(10))

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("\n=== Numeric Summary ===")
    print(df[numeric_cols].describe().T[["mean", "std", "min", "50%", "max"]].round(4).head(20))

    print("\n=== Speed Distribution ===")
    print(df["speed"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).round(4))
    print(f"speed skew={stats.skew(df['speed'], nan_policy='omit'):.3f}")
    print(f"log_speed skew={stats.skew(df['log_speed'], nan_policy='omit'):.3f}")

    print("\n=== Group Means (speed) ===")
    print(df.groupby(["dyslexia_bin", "reader_view"])['speed'].agg(["count", "mean", "median"]).round(4))

    # Correlations with speed/log_speed
    corr_with_speed = df[numeric_cols].corr(numeric_only=True)["speed"].drop("speed").sort_values(key=np.abs, ascending=False)
    corr_cols = list(dict.fromkeys(numeric_cols + ["log_speed"]))
    corr_with_log_speed = (
        df[corr_cols]
        .corr(numeric_only=True)["log_speed"]
        .drop("log_speed")
        .sort_values(key=np.abs, ascending=False)
    )
    print("\n=== Top Correlations with speed ===")
    print(corr_with_speed.head(10).round(4))
    print("\n=== Top Correlations with log_speed ===")
    print(corr_with_log_speed.head(10).round(4))

    # Focused tests for dyslexic participants
    dys_df = df[df["dyslexia_bin"] == 1].copy()
    if dys_df.empty:
        raise ValueError("No rows with dyslexia_bin == 1; cannot answer the research question.")

    rv1_log = dys_df.loc[dys_df["reader_view"] == 1, "log_speed"].dropna()
    rv0_log = dys_df.loc[dys_df["reader_view"] == 0, "log_speed"].dropna()
    rv1_raw = dys_df.loc[dys_df["reader_view"] == 1, "speed"].dropna()
    rv0_raw = dys_df.loc[dys_df["reader_view"] == 0, "speed"].dropna()

    welch_log = stats.ttest_ind(rv1_log, rv0_log, equal_var=False, nan_policy="omit")
    welch_raw = stats.ttest_ind(rv1_raw, rv0_raw, equal_var=False, nan_policy="omit")

    # Paired test at participant level (mean speed per condition)
    paired = dys_df.groupby(["uuid", "reader_view"])['log_speed'].mean().unstack()
    paired = paired.dropna(subset=[0, 1], how="any") if {0, 1}.issubset(set(paired.columns)) else pd.DataFrame()
    if not paired.empty:
        paired_test = stats.ttest_rel(paired[1], paired[0], nan_policy="omit")
        paired_diff_log = float((paired[1] - paired[0]).mean())
    else:
        paired_test = None
        paired_diff_log = np.nan

    # ANOVA example: log_speed by dyslexia severity under reader_view=1
    anova_df = df[df["reader_view"] == 1].dropna(subset=["dyslexia", "log_speed"])
    groups = [g["log_speed"].values for _, g in anova_df.groupby("dyslexia") if len(g) > 1]
    if len(groups) >= 2:
        anova_result = stats.f_oneway(*groups)
    else:
        anova_result = None

    print("\n=== Statistical Tests (Dyslexic participants) ===")
    print(
        "Welch t-test (log_speed): "
        f"t={welch_log.statistic:.4f}, p={welch_log.pvalue:.6f}, "
        f"mean(rv=1)-mean(rv=0)={safe_mean(rv1_log) - safe_mean(rv0_log):.6f}"
    )
    print(
        "Welch t-test (speed): "
        f"t={welch_raw.statistic:.4f}, p={welch_raw.pvalue:.6f}, "
        f"mean(rv=1)-mean(rv=0)={safe_mean(rv1_raw) - safe_mean(rv0_raw):.4f}"
    )
    if paired_test is not None:
        print(
            "Paired t-test by participant (log_speed): "
            f"t={paired_test.statistic:.4f}, p={paired_test.pvalue:.6f}, mean_diff={paired_diff_log:.6f}, n_pairs={len(paired)}"
        )
    else:
        print("Paired t-test by participant (log_speed): not available (missing both conditions per participant).")

    if anova_result is not None:
        print(
            "ANOVA (reader_view=1, log_speed ~ dyslexia severity): "
            f"F={anova_result.statistic:.4f}, p={anova_result.pvalue:.6f}"
        )
    else:
        print("ANOVA not run due to insufficient groups.")

    # OLS with interaction (statsmodels.api.OLS)
    model_cols = [
        "log_speed",
        "reader_view",
        "dyslexia_bin",
        "num_words",
        "Flesch_Kincaid",
        "retake_trial",
        "age",
        "correct_rate",
        "img_width",
        "scrolling_time",
        "device",
        "page_id",
    ]
    mdf = df[model_cols].dropna().copy()
    mdf["reader_view_x_dyslexia"] = mdf["reader_view"] * mdf["dyslexia_bin"]

    X = pd.get_dummies(
        mdf[
            [
                "reader_view",
                "dyslexia_bin",
                "reader_view_x_dyslexia",
                "num_words",
                "Flesch_Kincaid",
                "retake_trial",
                "age",
                "correct_rate",
                "img_width",
                "scrolling_time",
                "device",
                "page_id",
            ]
        ],
        columns=["device", "page_id"],
        drop_first=True,
    )
    X = X.astype(float)
    y = mdf["log_speed"].astype(float)

    X_const = sm.add_constant(X, has_constant="add")
    ols_model = sm.OLS(y, X_const).fit()

    idx_reader = list(ols_model.params.index).index("reader_view")
    idx_inter = list(ols_model.params.index).index("reader_view_x_dyslexia")
    L = np.zeros((1, len(ols_model.params)))
    L[0, idx_reader] = 1.0
    L[0, idx_inter] = 1.0
    dys_reader_effect_test = ols_model.t_test(L)
    dys_reader_effect_coef = float(dys_reader_effect_test.effect.squeeze())
    dys_reader_effect_p = float(dys_reader_effect_test.pvalue.squeeze())

    print("\n=== OLS Interaction Model ===")
    print(f"n={len(mdf)}, R^2={ols_model.rsquared:.4f}, adj_R^2={ols_model.rsquared_adj:.4f}")
    print("Top coefficients by absolute magnitude:")
    print(top_abs(ols_model.params.drop("const"), n=10).round(5))
    print(
        "Effect of reader_view among dyslexic participants (OLS linear contrast): "
        f"coef={dys_reader_effect_coef:.6f}, p={dys_reader_effect_p:.6f}"
    )

    # Interpretable sklearn models
    lin = LinearRegression().fit(X, y)
    ridge = Ridge(alpha=1.0, random_state=SEED).fit(X, y)
    lasso = Lasso(alpha=0.0005, max_iter=20000, random_state=SEED).fit(X, y)
    tree = DecisionTreeRegressor(max_depth=3, random_state=SEED).fit(X, y)

    lin_coef = pd.Series(lin.coef_, index=X.columns)
    ridge_coef = pd.Series(ridge.coef_, index=X.columns)
    lasso_coef = pd.Series(lasso.coef_, index=X.columns)
    tree_imp = pd.Series(tree.feature_importances_, index=X.columns)

    print("\n=== sklearn Interpretable Models ===")
    print("LinearRegression top |coef|:")
    print(top_abs(lin_coef, n=8).round(5))
    print("Ridge top |coef|:")
    print(top_abs(ridge_coef, n=8).round(5))
    print("Lasso non-zero coefficients:")
    print(lasso_coef[lasso_coef != 0].sort_values(key=np.abs, ascending=False).head(8).round(5))
    print("DecisionTreeRegressor feature importance:")
    print(tree_imp.sort_values(ascending=False).head(8).round(5))

    # imodels interpretable models
    print("\n=== imodels ===")
    imodels_notes = []

    try:
        rulefit = RuleFitRegressor(random_state=SEED, max_rules=30)
        rulefit.fit(X.values, y.values, feature_names=list(X.columns))
        rules_df = rulefit._get_rules()
        active_rules = rules_df[(rules_df["coef"] != 0) & (rules_df["type"] == "rule")]
        active_rules = active_rules.sort_values(by=["importance", "support"], ascending=False).head(5)
        print("RuleFit active rules (top 5 by importance/support):")
        if active_rules.empty:
            print("No active non-zero rules.")
            imodels_notes.append("RuleFit found no stable non-zero rules involving reader_view.")
        else:
            print(active_rules[["rule", "coef", "support"]])
            rule_mentions_reader = active_rules["rule"].str.contains("reader_view", regex=False).any()
            if not rule_mentions_reader:
                imodels_notes.append("Top RuleFit rules did not emphasize reader_view.")
    except Exception as exc:
        print(f"RuleFitRegressor failed: {exc}")
        imodels_notes.append("RuleFit failed to fit cleanly.")

    try:
        figs = FIGSRegressor(random_state=SEED, max_rules=12)
        figs.fit(X.values, y.values, feature_names=list(X.columns))
        figs_imp = pd.Series(figs.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("FIGS feature importance (top 8):")
        print(figs_imp.head(8).round(5))
        imodels_notes.append(
            f"FIGS reader_view importance={float(figs_imp.get('reader_view', 0.0)):.5f}, "
            f"interaction importance={float(figs_imp.get('reader_view_x_dyslexia', 0.0)):.5f}."
        )
    except Exception as exc:
        print(f"FIGSRegressor failed: {exc}")
        imodels_notes.append("FIGS failed to fit cleanly.")

    try:
        hs = HSTreeRegressor(random_state=SEED, max_leaf_nodes=12)
        hs.fit(X.values, y.values, feature_names=list(X.columns))
        hs_imp = pd.Series(hs.estimator_.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("HSTree feature importance (top 8):")
        print(hs_imp.head(8).round(5))
        imodels_notes.append(
            f"HSTree reader_view importance={float(hs_imp.get('reader_view', 0.0)):.5f}, "
            f"interaction importance={float(hs_imp.get('reader_view_x_dyslexia', 0.0)):.5f}."
        )
    except Exception as exc:
        print(f"HSTreeRegressor failed: {exc}")
        imodels_notes.append("HSTree failed to fit cleanly.")

    # Final evidence synthesis and score
    welch_diff_log = safe_mean(rv1_log) - safe_mean(rv0_log)
    paired_p = float(paired_test.pvalue) if paired_test is not None else np.nan

    all_key_p = [float(welch_log.pvalue), float(dys_reader_effect_p)]
    if not np.isnan(paired_p):
        all_key_p.append(paired_p)

    any_sig_positive = (
        (welch_log.pvalue < 0.05 and welch_diff_log > 0)
        or (not np.isnan(paired_p) and paired_p < 0.05 and paired_diff_log > 0)
        or (dys_reader_effect_p < 0.05 and dys_reader_effect_coef > 0)
    )
    any_sig_negative = (
        (welch_log.pvalue < 0.05 and welch_diff_log < 0)
        or (not np.isnan(paired_p) and paired_p < 0.05 and paired_diff_log < 0)
        or (dys_reader_effect_p < 0.05 and dys_reader_effect_coef < 0)
    )

    if all(p >= 0.05 for p in all_key_p):
        score = 15
    elif any_sig_positive and not any_sig_negative:
        score = 85
    elif any_sig_negative and not any_sig_positive:
        score = 5
    else:
        score = 40

    # Small calibration with interaction p-value
    interaction_p = float(ols_model.pvalues.get("reader_view_x_dyslexia", np.nan))
    interaction_coef = float(ols_model.params.get("reader_view_x_dyslexia", 0.0))
    if np.isfinite(interaction_p) and interaction_p < 0.05:
        score += 5 if interaction_coef > 0 else -5

    score = clamp_int(score)

    explanation_parts = [
        "No statistically significant evidence that Reader View improves speed for dyslexic readers.",
        (
            f"Welch t-test on log(speed) in dyslexic subset: p={welch_log.pvalue:.4f}, "
            f"mean difference (reader_view=1 minus 0)={welch_diff_log:.4f}."
        ),
        (
            "Participant-level paired t-test on log(speed): "
            + (
                f"p={paired_p:.4f}, mean diff={paired_diff_log:.4f}."
                if not np.isnan(paired_p)
                else "not available."
            )
        ),
        (
            f"OLS contrast for reader_view effect among dyslexic participants: "
            f"coef={dys_reader_effect_coef:.4f}, p={dys_reader_effect_p:.4f}."
        ),
        (
            f"Interaction reader_view*dyslexia_bin in OLS: coef={interaction_coef:.4f}, p={interaction_p:.4f}."
        ),
    ]

    if imodels_notes:
        explanation_parts.append("Interpretable tree/rule models summary: " + " ".join(imodels_notes[:3]))

    explanation = " ".join(explanation_parts)

    result = {
        "response": score,
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(result, ensure_ascii=True))
    print("\nWrote conclusion.txt")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
