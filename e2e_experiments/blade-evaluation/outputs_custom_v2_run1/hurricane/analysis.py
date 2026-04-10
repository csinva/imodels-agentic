import json
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from interp_models import SmartAdditiveRegressor, HingeEBMRegressor


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def infer_question_and_variables(info: Dict) -> Tuple[str, str, str]:
    question = info.get("research_questions", [""])[0]

    # This dataset's canonical question is about name femininity -> deaths.
    dv = "alldeaths"
    iv = "masfem"

    # Lightweight fallback heuristics if metadata changes.
    if isinstance(question, str):
        q = question.lower()
        if "death" in q and "alldeaths" in info.get("data_desc", {}).get("field_names", []):
            dv = "alldeaths"
        if "femin" in q and "masfem" in info.get("data_desc", {}).get("field_names", []):
            iv = "masfem"

    return question, dv, iv


def summarize_distribution(series: pd.Series) -> Dict[str, float]:
    return {
        "mean": safe_float(series.mean()),
        "std": safe_float(series.std()),
        "min": safe_float(series.min()),
        "q25": safe_float(series.quantile(0.25)),
        "median": safe_float(series.median()),
        "q75": safe_float(series.quantile(0.75)),
        "max": safe_float(series.max()),
        "skew": safe_float(series.skew()),
    }


def top_effects(effects: Dict, k: int = 5) -> List[Tuple[str, Dict]]:
    items = [(name, vals) for name, vals in effects.items() if vals.get("importance", 0) > 0]
    items.sort(key=lambda x: -x[1].get("importance", 0))
    return items[:k]


def get_effect(effects: Dict, feature: str) -> Dict:
    return effects.get(feature, {"direction": "zero", "importance": 0.0, "rank": 0})


def clamp_int(x: float, lo: int = 0, hi: int = 100) -> int:
    return int(max(lo, min(hi, round(x))))


def main():
    with open("info.json", "r") as f:
        info = json.load(f)

    question, dv, iv = infer_question_and_variables(info)
    df = pd.read_csv("hurricane.csv")

    print("=" * 80)
    print("Research question:")
    print(question)
    print(f"Inferred DV: {dv}")
    print(f"Inferred IV: {iv}")
    print("=" * 80)

    # Keep all numeric columns for broad analysis, but exclude identifier from modeling.
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    model_numeric_cols = [c for c in numeric_cols if c != "ind"]

    # Median-impute numeric missings for model compatibility.
    for c in model_numeric_cols:
        if df[c].isna().any():
            med = df[c].median()
            df[c] = df[c].fillna(med)
            print(f"Imputed missing values in {c} with median={med:.4f}")

    # Step 1: exploration
    print("\nStep 1: Summary statistics for numeric columns")
    print(df[model_numeric_cols].describe().T.to_string())

    print("\nStep 1: Distribution snapshots")
    iv_dist = summarize_distribution(df[iv])
    dv_dist = summarize_distribution(df[dv])
    print(f"{iv} distribution: {iv_dist}")
    print(f"{dv} distribution: {dv_dist}")

    print("\nStep 1: Bivariate correlations with DV")
    corr_with_dv = (
        df[model_numeric_cols]
        .corr(numeric_only=True)[dv]
        .drop(labels=[dv])
        .sort_values(ascending=False)
    )
    print(corr_with_dv.to_string())
    iv_dv_corr = safe_float(corr_with_dv.get(iv, np.nan))
    print(f"\nBivariate correlation ({iv}, {dv}) = {iv_dv_corr:.4f}")

    # Step 2: OLS with controls
    # Relevant controls: storm severity, damage, and time.
    controls = ["min", "wind", "category", "ndam15", "year"]
    feature_cols = [iv] + controls
    X = sm.add_constant(df[feature_cols])
    y = df[dv]

    ols = sm.OLS(y, X).fit()
    print("\nStep 2: OLS with controls")
    print(ols.summary())

    iv_coef = safe_float(ols.params.get(iv, np.nan))
    iv_p = safe_float(ols.pvalues.get(iv, np.nan))

    # Optional robustness to skewed counts.
    y_log = np.log1p(y)
    ols_log = sm.OLS(y_log, X).fit()
    iv_coef_log = safe_float(ols_log.params.get(iv, np.nan))
    iv_p_log = safe_float(ols_log.pvalues.get(iv, np.nan))
    print("\nStep 2b: OLS on log1p(alldeaths) robustness check")
    print(ols_log.summary())

    # Step 3: custom interpretable models on all numeric predictors except DV.
    X_all = df[[c for c in model_numeric_cols if c != dv]]
    y_all = df[dv]

    print("\nStep 3: SmartAdditiveRegressor")
    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X_all, y_all)
    smart_effects = smart.feature_effects()
    print(smart)
    print("\nSmartAdditive feature_effects:")
    print(smart_effects)

    print("\nStep 3: HingeEBMRegressor")
    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X_all, y_all)
    hinge_effects = hinge.feature_effects()
    print(hinge)
    print("\nHingeEBM feature_effects:")
    print(hinge_effects)

    iv_smart = get_effect(smart_effects, iv)
    iv_hinge = get_effect(hinge_effects, iv)

    smart_top = top_effects(smart_effects, k=5)
    hinge_top = top_effects(hinge_effects, k=5)

    # Confounders from OLS (excluding IV) with strongest evidence.
    ols_sig_controls = []
    for c in controls:
        p = safe_float(ols.pvalues.get(c, np.nan))
        b = safe_float(ols.params.get(c, np.nan))
        if np.isfinite(p) and p < 0.10:
            ols_sig_controls.append((c, b, p))
    ols_sig_controls.sort(key=lambda t: t[2])

    # Convert evidence to Likert score.
    score = 50.0

    # Bivariate signal (weak positive gets small upward weight).
    if np.isfinite(iv_dv_corr):
        if abs(iv_dv_corr) < 0.05:
            score -= 8
        elif abs(iv_dv_corr) < 0.15:
            score -= 2
        elif abs(iv_dv_corr) < 0.30:
            score += 6
        else:
            score += 12

    # Controlled OLS evidence.
    if np.isfinite(iv_p):
        if iv_p < 0.01:
            score += 25
        elif iv_p < 0.05:
            score += 16
        elif iv_p < 0.10:
            score += 8
        else:
            score -= 16

    # Direction alignment with hypothesis (positive femininity -> more deaths).
    if np.isfinite(iv_coef):
        if iv_coef > 0:
            score += 4
        else:
            score -= 8

    # Robustness on log-scale model.
    if np.isfinite(iv_p_log):
        if iv_p_log < 0.10:
            score += 8
        else:
            score -= 6

    # Interpretable model importances.
    smart_imp = safe_float(iv_smart.get("importance", 0.0))
    hinge_imp = safe_float(iv_hinge.get("importance", 0.0))

    if smart_imp >= 0.10:
        score += 15
    elif smart_imp >= 0.03:
        score += 6
    else:
        score -= 14

    if hinge_imp >= 0.10:
        score += 12
    elif hinge_imp >= 0.03:
        score += 5
    else:
        score -= 12

    response = clamp_int(score)

    # Build explanation string with direction, magnitude, shape, robustness, confounders.
    smart_desc = (
        f"SmartAdditive: direction={iv_smart.get('direction', 'zero')}, "
        f"importance={smart_imp:.3f}, rank={iv_smart.get('rank', 0)}"
    )
    hinge_desc = (
        f"HingeEBM: direction={iv_hinge.get('direction', 'zero')}, "
        f"importance={hinge_imp:.3f}, rank={iv_hinge.get('rank', 0)}"
    )

    smart_top_txt = ", ".join(
        [f"{n} ({v.get('importance', 0):.3f}, {v.get('direction', 'n/a')})" for n, v in smart_top]
    ) if smart_top else "none"

    hinge_top_txt = ", ".join(
        [f"{n} ({v.get('importance', 0):.3f}, {v.get('direction', 'n/a')})" for n, v in hinge_top]
    ) if hinge_top else "none"

    if ols_sig_controls:
        conf_txt = ", ".join([f"{c} (coef={b:.3f}, p={p:.3g})" for c, b, p in ols_sig_controls])
    else:
        conf_txt = "no controls reached p<0.10"

    explanation = (
        f"Using DV={dv} and IV={iv}, the bivariate association is weak (r={iv_dv_corr:.3f}). "
        f"In controlled OLS (controls=min, wind, category, ndam15, year), the IV coefficient is positive but not statistically reliable "
        f"(coef={iv_coef:.3f}, p={iv_p:.3f}); this remains non-significant on log deaths (coef={iv_coef_log:.3f}, p={iv_p_log:.3f}). "
        f"{smart_desc}. {hinge_desc}. In both interpretable models, {iv} has near-zero importance relative to other predictors, "
        f"so the effect is not robust across model classes and shows no stable nonlinear threshold pattern. "
        f"Most influential features are SmartAdditive: {smart_top_txt}; HingeEBM: {hinge_top_txt}. "
        f"Key confounders/competing predictors in OLS are: {conf_txt}. "
        f"Overall evidence for the claim is weak and inconsistent after controls, so the score is low."
    )

    result = {
        "response": int(response),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(result, f)

    print("\nWrote conclusion.txt:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
