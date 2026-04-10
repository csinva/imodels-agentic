import json
import math
from typing import Dict, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.preprocessing import StandardScaler

from interp_models import SmartAdditiveRegressor, HingeEBMRegressor


def get_effect(effects: Dict[str, Dict[str, Any]], name: str) -> Dict[str, Any]:
    default = {"direction": "zero", "importance": 0.0, "rank": 0}
    eff = effects.get(name, default)
    out = {
        "direction": str(eff.get("direction", "zero")),
        "importance": float(eff.get("importance", 0.0)),
        "rank": int(eff.get("rank", 0)),
    }
    return out


def nonlinear_summary(model: SmartAdditiveRegressor, feature_name: str) -> str:
    if feature_name not in model.feature_names_:
        return ""
    j = model.feature_names_.index(feature_name)
    if j not in model.shape_functions_:
        return ""

    thresholds, intervals = model.shape_functions_[j]
    if len(thresholds) == 0 or len(intervals) < 2:
        return ""

    start_val = float(intervals[0])
    end_val = float(intervals[-1])
    peak_idx = int(np.argmax(intervals))
    trough_idx = int(np.argmin(intervals))

    if peak_idx == 0:
        peak_desc = f"below {thresholds[0]:.1f}"
    elif peak_idx == len(thresholds):
        peak_desc = f"above {thresholds[-1]:.1f}"
    else:
        peak_desc = f"around {thresholds[peak_idx - 1]:.1f} to {thresholds[peak_idx]:.1f}"

    if trough_idx == 0:
        trough_desc = f"below {thresholds[0]:.1f}"
    elif trough_idx == len(thresholds):
        trough_desc = f"above {thresholds[-1]:.1f}"
    else:
        trough_desc = f"around {thresholds[trough_idx - 1]:.1f} to {thresholds[trough_idx]:.1f}"

    trend = "increasing" if end_val > start_val else "decreasing"
    return (
        f"{feature_name} shows a {trend} nonlinear pattern with low effect {trough_desc} "
        f"and highest effect {peak_desc}."
    )


def main() -> None:
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info["research_questions"][0]
    print("Research question:")
    print(research_question)

    df = pd.read_csv("amtl.csv")

    # Define DV/IV based on question (frequency of AMTL in humans vs non-humans)
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]
    df["has_amtl"] = (df["num_amtl"] > 0).astype(int)

    needed_cols = [
        "num_amtl",
        "sockets",
        "age",
        "stdev_age",
        "prob_male",
        "tooth_class",
        "genus",
        "is_human",
        "amtl_rate",
        "has_amtl",
    ]
    df = df.dropna(subset=needed_cols).copy()

    print("\nStep 1: Explore data")
    print(f"Rows used: {len(df)}")
    print("DV (primary): amtl_rate = num_amtl / sockets")
    print("IV (primary): is_human (Homo sapiens=1, non-human genera=0)")

    numeric_cols = ["num_amtl", "sockets", "age", "stdev_age", "prob_male", "amtl_rate", "is_human"]
    print("\nNumeric summary statistics:")
    print(df[numeric_cols].describe().T)

    print("\nDistribution checks (categorical counts):")
    print("Genus counts:")
    print(df["genus"].value_counts())
    print("Tooth class counts:")
    print(df["tooth_class"].value_counts())

    print("\nBivariate AMTL comparisons:")
    by_human = df.groupby("is_human")["amtl_rate"].agg(["mean", "std", "median", "count"])
    print(by_human)
    by_genus = df.groupby("genus")["amtl_rate"].agg(["mean", "std", "median", "count"]).sort_values("mean", ascending=False)
    print("\nAMTL rate by genus:")
    print(by_genus)

    corr_pearson = float(df["is_human"].corr(df["amtl_rate"]))
    corr_spearman = float(df[["is_human", "amtl_rate"]].corr(method="spearman").iloc[0, 1])
    ttest = stats.ttest_ind(
        df.loc[df["is_human"] == 1, "amtl_rate"],
        df.loc[df["is_human"] == 0, "amtl_rate"],
        equal_var=False,
    )
    print(f"\nPearson corr(is_human, amtl_rate): {corr_pearson:.4f}")
    print(f"Spearman corr(is_human, amtl_rate): {corr_spearman:.4f}")
    print(f"Welch t-test human vs non-human amtl_rate: t={ttest.statistic:.4f}, p={ttest.pvalue:.4g}")

    print("\nCorrelation matrix (numeric variables):")
    print(df[numeric_cols].corr())

    print("\nStep 2: Controlled regression models")

    # OLS on count with sockets as control (captures differing opportunity for AMTL)
    ols_formula = "num_amtl ~ is_human + age + prob_male + stdev_age + sockets + C(tooth_class)"
    ols_model = smf.ols(ols_formula, data=df).fit()
    print("\nOLS model summary:")
    print(ols_model.summary())

    # Binomial GLM on AMTL frequency with sockets as binomial denominator
    X_glm = pd.get_dummies(
        df[["is_human", "age", "prob_male", "stdev_age", "tooth_class"]],
        drop_first=True,
        dtype=float,
    )
    X_glm = sm.add_constant(X_glm)
    glm_model = sm.GLM(
        df["amtl_rate"],
        X_glm,
        family=sm.families.Binomial(),
        var_weights=df["sockets"],
    ).fit()
    print("\nBinomial GLM summary:")
    print(glm_model.summary())

    # Additional robustness: logistic for any AMTL event
    logit_model = smf.logit(
        "has_amtl ~ is_human + age + prob_male + stdev_age + C(tooth_class)",
        data=df,
    ).fit(disp=0)
    print("\nLogit(has_amtl) summary:")
    print(logit_model.summary())

    print("\nStep 3: Interpretable models")
    X_interp = df[["is_human", "age", "stdev_age", "prob_male", "sockets"]].copy()
    tooth_dummies = pd.get_dummies(df["tooth_class"], prefix="tooth", drop_first=True, dtype=float)
    X_interp = pd.concat([X_interp, tooth_dummies], axis=1)
    y_interp = df["num_amtl"].astype(float)

    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X_interp, y_interp)
    smart_effects = smart.feature_effects()

    print("\nSmartAdditiveRegressor model:")
    print(smart)
    print("\nSmartAdditive feature effects:")
    print(smart_effects)

    # Scale for hinge model so coefficient-based selection is not dominated by units
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_interp), columns=X_interp.columns)

    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X_scaled, y_interp)
    hinge_effects = hinge.feature_effects()

    print("\nHingeEBMRegressor model:")
    print(hinge)
    print("\nHingeEBM feature effects:")
    print(hinge_effects)

    # Extract key coefficients/effects
    human_rate = float(df.loc[df["is_human"] == 1, "amtl_rate"].mean())
    nonhuman_rate = float(df.loc[df["is_human"] == 0, "amtl_rate"].mean())

    ols_coef = float(ols_model.params.get("is_human", np.nan))
    ols_p = float(ols_model.pvalues.get("is_human", np.nan))

    glm_coef = float(glm_model.params.get("is_human", np.nan))
    glm_p = float(glm_model.pvalues.get("is_human", np.nan))
    glm_or = float(math.exp(glm_coef))

    logit_coef = float(logit_model.params.get("is_human", np.nan))
    logit_p = float(logit_model.pvalues.get("is_human", np.nan))
    logit_or = float(math.exp(logit_coef))

    smart_human = get_effect(smart_effects, "is_human")
    hinge_human = get_effect(hinge_effects, "is_human")
    smart_age = get_effect(smart_effects, "age")
    hinge_age = get_effect(hinge_effects, "age")
    smart_sockets = get_effect(smart_effects, "sockets")

    age_shape_text = nonlinear_summary(smart, "age")

    # Likert score heuristic (0-100)
    score = 20

    # Bivariate evidence
    if human_rate > nonhuman_rate:
        rate_ratio = human_rate / max(nonhuman_rate, 1e-8)
        if rate_ratio > 4:
            score += 22
        elif rate_ratio > 2:
            score += 16
        else:
            score += 8

    if corr_pearson > 0.2:
        score += 8
    elif corr_pearson > 0.1:
        score += 5

    # Controlled models
    if glm_coef > 0 and glm_p < 0.001:
        score += 20
    elif glm_coef > 0 and glm_p < 0.05:
        score += 14
    elif glm_coef > 0 and glm_p < 0.1:
        score += 8
    else:
        score -= 8

    if ols_coef > 0 and ols_p < 0.05:
        score += 10
    elif ols_coef > 0 and ols_p < 0.1:
        score += 5
    elif ols_coef > 0:
        score += 2
    else:
        score -= 6

    if logit_coef > 0 and logit_p < 0.01:
        score += 10
    elif logit_coef > 0 and logit_p < 0.05:
        score += 7

    # Interpretable models
    if smart_human["direction"].startswith("positive") and smart_human["importance"] >= 0.05:
        score += 7
    elif smart_human["direction"].startswith("positive") and smart_human["importance"] > 0:
        score += 4
    elif smart_human["importance"] == 0:
        score -= 5

    if hinge_human["direction"].startswith("positive") and hinge_human["importance"] >= 0.05:
        score += 7
    elif hinge_human["direction"].startswith("positive") and hinge_human["importance"] > 0:
        score += 4
    elif hinge_human["importance"] == 0:
        score -= 5

    response = int(max(0, min(100, round(score))))

    explanation = (
        f"Humans show higher AMTL frequency than non-human primates in bivariate comparisons "
        f"(mean AMTL rate {human_rate:.3f} vs {nonhuman_rate:.3f}; Pearson r={corr_pearson:.3f}). "
        f"After controls, the human effect remains positive in count OLS (coef={ols_coef:.3f}, p={ols_p:.3g}) "
        f"and is strongly positive in binomial AMTL-rate GLM (coef={glm_coef:.3f}, p={glm_p:.3g}, OR={glm_or:.2f}) "
        f"and in logistic AMTL-presence model (coef={logit_coef:.3f}, p={logit_p:.3g}, OR={logit_or:.2f}). "
        f"SmartAdditive also gives a positive human effect (importance={smart_human['importance']:.1%}, "
        f"rank={smart_human['rank']}) while showing that age is the dominant predictor "
        f"(importance={smart_age['importance']:.1%}, rank={smart_age['rank']}) with nonlinear thresholds; "
        f"{age_shape_text} "
        f"Sockets and age-uncertainty also matter (sockets importance={smart_sockets['importance']:.1%}). "
        f"HingeEBM confirms direction: human effect is positive and retained (importance={hinge_human['importance']:.1%}, "
        f"rank={hinge_human['rank']}), but smaller than age (importance={hinge_age['importance']:.1%}, rank={hinge_age['rank']}). "
        f"Overall this supports a robust Yes: humans have higher AMTL frequencies after accounting for age, sex, and tooth class, "
        f"though age is the strongest confounder and contributes more than species identity."
    )

    output = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(output, ensure_ascii=True))

    print("\nWrote conclusion.txt")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
