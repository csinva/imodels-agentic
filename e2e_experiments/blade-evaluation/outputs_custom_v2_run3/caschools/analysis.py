import json
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm

from interp_models import HingeEBMRegressor, SmartAdditiveRegressor


def fit_ols(df: pd.DataFrame, dv: str, features: List[str], label: str):
    data = df[[dv] + features].dropna()
    X = sm.add_constant(data[features], has_constant="add")
    y = data[dv]
    model = sm.OLS(y, X).fit()
    print(f"\n=== OLS: {label} ===")
    print(model.summary())
    return model


def safe_effect(effects: Dict, feature: str):
    if feature not in effects:
        return {"direction": "zero", "importance": 0.0, "rank": 0}
    e = effects[feature]
    return {
        "direction": str(e.get("direction", "zero")),
        "importance": float(e.get("importance", 0.0)),
        "rank": int(e.get("rank", 0) or 0),
    }


def smart_shape_summary(model: SmartAdditiveRegressor, feature: str) -> str:
    if feature not in model.feature_names_:
        return "feature unavailable in SmartAdditive model"

    j = model.feature_names_.index(feature)
    if j not in model.shape_functions_:
        return "no learned nonlinear pattern for this feature"

    thresholds, intervals = model.shape_functions_[j]
    if not intervals:
        return "no learned shape"

    max_idx = int(np.argmax(intervals))
    min_idx = int(np.argmin(intervals))

    def interval_label(idx: int) -> str:
        if len(thresholds) == 0:
            return "all values"
        if idx == 0:
            return f"<= {thresholds[0]:.2f}"
        if idx == len(thresholds):
            return f"> {thresholds[-1]:.2f}"
        return f"({thresholds[idx - 1]:.2f}, {thresholds[idx]:.2f}]"

    return (
        f"highest contribution around {feature} {interval_label(max_idx)} ({intervals[max_idx]:+.2f}), "
        f"lowest around {feature} {interval_label(min_idx)} ({intervals[min_idx]:+.2f})"
    )


def evidence_strength(coef: float, pval: float) -> float:
    if coef >= 0:
        return 0.0
    if pval < 0.01:
        return 1.0
    if pval < 0.05:
        return 0.8
    if pval < 0.10:
        return 0.4
    return 0.0


def model_effect_strength(direction: str, importance: float) -> float:
    if "decreasing" not in direction and direction != "negative":
        return 0.0
    if importance >= 0.10:
        return 1.0
    if importance >= 0.05:
        return 0.7
    if importance >= 0.02:
        return 0.4
    if importance > 0:
        return 0.2
    return 0.0


def main():
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info.get("research_questions", ["Unknown question"])[0]
    print("Research question:")
    print(question)

    df = pd.read_csv("caschools.csv")

    # Build IV and DV for the question.
    iv = "student_teacher_ratio"
    dv = "avg_score"
    df[iv] = df["students"] / df["teachers"]
    df[dv] = (df["read"] + df["math"]) / 2.0

    print("\n=== Step 1: Exploration ===")
    cols_for_summary = [
        iv,
        dv,
        "students",
        "teachers",
        "calworks",
        "lunch",
        "english",
        "income",
        "expenditure",
        "computer",
    ]
    print("Summary statistics:")
    print(df[cols_for_summary].describe().T)

    print("\nDistribution quantiles (IV and DV):")
    print(df[[iv, dv]].quantile([0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]))

    corr = df[[iv, dv, "lunch", "income", "english", "calworks", "expenditure", "computer"]].corr()[dv]
    print("\nCorrelations with DV:")
    print(corr.sort_values(ascending=False))

    # Step 2: OLS models with increasing controls.
    m1 = fit_ols(df, dv, [iv], "Bivariate")
    core_controls = [iv, "lunch", "income", "english"]
    m2 = fit_ols(df, dv, core_controls, "Core controls")
    full_controls = [iv, "lunch", "income", "english", "expenditure", "computer", "calworks", "students"]
    m3 = fit_ols(df, dv, full_controls, "Extended controls")

    # Step 3: Interpretable models.
    print("\n=== Step 3: Interpretable Models ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude identifiers and direct components of the constructed DV.
    exclude = {"rownames", "district", dv, "read", "math"}
    interp_features = [c for c in numeric_cols if c not in exclude]
    X_interp = df[interp_features]
    y = df[dv]

    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X_interp, y)
    smart_effects = smart.feature_effects()
    print("\nSmartAdditiveRegressor model:")
    print(smart)
    print("\nSmartAdditive feature effects:")
    print(smart_effects)

    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X_interp, y)
    hinge_effects = hinge.feature_effects()
    print("\nHingeEBMRegressor model:")
    print(hinge)
    print("\nHingeEBM feature effects:")
    print(hinge_effects)

    iv_smart = safe_effect(smart_effects, iv)
    iv_hinge = safe_effect(hinge_effects, iv)

    # Important confounders from model importances.
    smart_ranked = sorted(
        [(k, float(v.get("importance", 0.0)), str(v.get("direction", "")), int(v.get("rank", 0) or 0))
         for k, v in smart_effects.items() if k != iv],
        key=lambda x: x[1],
        reverse=True,
    )
    hinge_ranked = sorted(
        [(k, float(v.get("importance", 0.0)), str(v.get("direction", "")), int(v.get("rank", 0) or 0))
         for k, v in hinge_effects.items() if k != iv],
        key=lambda x: x[1],
        reverse=True,
    )

    top_smart = smart_ranked[:3]
    top_hinge = hinge_ranked[:3]

    # Scoring rubric weighted toward controlled robustness.
    score = 0.0
    score += 20.0 * evidence_strength(float(m1.params[iv]), float(m1.pvalues[iv]))
    score += 25.0 * evidence_strength(float(m2.params[iv]), float(m2.pvalues[iv]))
    score += 35.0 * evidence_strength(float(m3.params[iv]), float(m3.pvalues[iv]))
    score += 10.0 * model_effect_strength(iv_smart["direction"], iv_smart["importance"])
    score += 10.0 * model_effect_strength(iv_hinge["direction"], iv_hinge["importance"])
    score_int = int(np.clip(round(score), 0, 100))

    shape_text = smart_shape_summary(smart, iv)

    explanation = (
        f"Bivariate evidence supports the hypothesis: {iv} is negatively correlated with {dv} "
        f"(r={df[iv].corr(df[dv]):.3f}) and the unadjusted OLS slope is {m1.params[iv]:.3f} "
        f"(p={m1.pvalues[iv]:.3g}). After core socioeconomic controls (lunch, income, english), "
        f"the association remains negative but smaller ({m2.params[iv]:.3f}, p={m2.pvalues[iv]:.3g}). "
        f"With extended controls (including expenditure, computer, calworks, students), it stays negative "
        f"but is not statistically robust ({m3.params[iv]:.3f}, p={m3.pvalues[iv]:.3g}). "
        f"SmartAdditive ranks {iv} at #{iv_smart['rank']} with {iv_smart['importance']:.1%} importance and "
        f"{iv_smart['direction']} shape; {shape_text}. HingeEBM gives {iv} direction={iv_hinge['direction']} "
        f"with {iv_hinge['importance']:.1%} importance, effectively shrinking it toward zero relative to stronger predictors. "
        f"The dominant confounders are socioeconomic variables: SmartAdditive top features are "
        f"{top_smart[0][0]} ({top_smart[0][1]:.1%}), {top_smart[1][0]} ({top_smart[1][1]:.1%}), "
        f"{top_smart[2][0]} ({top_smart[2][1]:.1%}); HingeEBM top features are "
        f"{top_hinge[0][0]} ({top_hinge[0][1]:.1%}), {top_hinge[1][0]} ({top_hinge[1][1]:.1%}), "
        f"{top_hinge[2][0]} ({top_hinge[2][1]:.1%}). Overall, lower student-teacher ratio shows a directionally "
        f"favorable but only partially robust relationship with performance once confounders are controlled."
    )

    payload = {"response": score_int, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print("\n=== Final conclusion payload ===")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
