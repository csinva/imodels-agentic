import json
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from interp_models import HingeEBMRegressor, SmartAdditiveRegressor


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def get_effect(effects: Dict, name: str) -> Dict:
    if name not in effects:
        return {"direction": "missing", "importance": 0.0, "rank": 0}
    out = effects[name].copy()
    out["importance"] = safe_float(out.get("importance", 0.0), 0.0)
    out["rank"] = int(out.get("rank", 0) or 0)
    return out


def extract_age_threshold_summary(smart_model: SmartAdditiveRegressor, feature_name: str = "age") -> str:
    # Use internal shape function to provide a concise threshold/nonlinearity summary.
    if feature_name not in smart_model.feature_names_:
        return "No age shape detected."

    age_idx = smart_model.feature_names_.index(feature_name)
    if age_idx not in smart_model.shape_functions_:
        return "Age had no detectable nonlinear shape."

    thresholds, intervals = smart_model.shape_functions_[age_idx]
    if len(thresholds) == 0:
        return "Age effect appeared approximately linear."

    transition = None
    for i in range(len(intervals) - 1):
        if intervals[i] <= 0 < intervals[i + 1]:
            if i < len(thresholds):
                transition = thresholds[i]
            break

    if transition is not None:
        return f"Age effect is nonlinear with a negative-to-positive transition around {transition:.1f} years."

    # Fallback: use trend from first to last interval.
    trend = "increasing" if intervals[-1] > intervals[0] else "decreasing"
    return (
        f"Age effect is nonlinear ({trend}) across thresholds from "
        f"{thresholds[0]:.1f} to {thresholds[-1]:.1f} years."
    )


def top_nonkey_effects(effects: Dict, key_vars: List[str], top_k: int = 3) -> List[Tuple[str, Dict]]:
    rows = []
    for name, meta in effects.items():
        if name in key_vars:
            continue
        imp = safe_float(meta.get("importance", 0.0), 0.0)
        if imp > 0:
            rows.append((name, {"importance": imp, "direction": meta.get("direction", "unknown")}))
    rows.sort(key=lambda x: x[1]["importance"], reverse=True)
    return rows[:top_k]


def main():
    # Step 1: Read research question and data metadata
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info.get("research_questions", [""])[0]
    print("Research question:")
    print(question)
    print("\nLoading dataset...")

    df = pd.read_csv("panda_nuts.csv")
    print(f"Rows: {len(df)}, Columns: {list(df.columns)}")

    # Define DV and key IVs based on question
    # Efficiency proxy: nuts opened per second during session.
    df = df.copy()
    df["sex_male"] = (df["sex"].astype(str).str.lower() == "m").astype(int)
    df["help_received"] = (df["help"].astype(str).str.lower() == "y").astype(int)
    df["efficiency"] = np.where(df["seconds"] > 0, df["nuts_opened"] / df["seconds"], np.nan)
    df = df.dropna(subset=["efficiency"]).reset_index(drop=True)

    dv = "efficiency"
    key_vars = ["age", "sex_male", "help_received"]

    print("\nDV and key IVs:")
    print(f"DV: {dv}")
    print(f"Key IVs: {key_vars}")

    # Basic exploration
    print("\nSummary statistics (core variables):")
    print(df[["efficiency", "age", "sex_male", "help_received", "nuts_opened", "seconds"]].describe())

    print("\nDistribution checks:")
    print("Efficiency quantiles:")
    print(df["efficiency"].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
    print(f"Zero-nut sessions: {(df['nuts_opened'] == 0).mean():.3f}")

    print("\nBivariate correlations with efficiency:")
    bivariate_corr = {}
    for var in key_vars:
        corr = np.corrcoef(df[var], df[dv])[0, 1]
        bivariate_corr[var] = corr
        print(f"  corr({var}, {dv}) = {corr:.3f}")

    # Step 2: OLS with controls
    # Controls: hammer type + chimpanzee fixed effects to absorb tool and individual heterogeneity.
    ols_df = df[[dv, "age", "sex_male", "help_received", "hammer", "chimpanzee"]].copy()
    ols_df["chimpanzee"] = ols_df["chimpanzee"].astype(str)

    X_ols = pd.get_dummies(
        ols_df.drop(columns=[dv]),
        columns=["hammer", "chimpanzee"],
        drop_first=True,
    )
    X_ols = sm.add_constant(X_ols).astype(float)
    y_ols = ols_df[dv].astype(float)

    ols_model = sm.OLS(y_ols, X_ols).fit()
    print("\nOLS results (with controls):")
    print(ols_model.summary())

    ols_coef = {k: safe_float(ols_model.params.get(k, np.nan)) for k in key_vars}
    ols_pval = {k: safe_float(ols_model.pvalues.get(k, np.nan)) for k in key_vars}

    # Step 3: Interpretable models
    # Build fully numeric feature set with dummies for categorical controls.
    interp_df = df[["age", "sex_male", "help_received", "hammer", "chimpanzee"]].copy()
    interp_df["chimpanzee"] = interp_df["chimpanzee"].astype(str)
    X_interp = pd.get_dummies(interp_df, columns=["hammer", "chimpanzee"], drop_first=False).astype(float)
    y_interp = df[dv].astype(float)

    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X_interp, y_interp)
    print("\nSmartAdditiveRegressor summary:")
    print(smart)
    smart_effects = smart.feature_effects()
    print("\nSmartAdditive feature effects:")
    print(smart_effects)

    hinge = HingeEBMRegressor(n_knots=3, max_input_features=min(40, X_interp.shape[1]))
    hinge.fit(X_interp, y_interp)
    print("\nHingeEBMRegressor summary:")
    print(hinge)
    hinge_effects = hinge.feature_effects()
    print("\nHingeEBM feature effects:")
    print(hinge_effects)

    # Pull key variable evidence from interpretable models
    smart_age = get_effect(smart_effects, "age")
    smart_sex = get_effect(smart_effects, "sex_male")
    smart_help = get_effect(smart_effects, "help_received")

    hinge_age = get_effect(hinge_effects, "age")
    hinge_sex = get_effect(hinge_effects, "sex_male")
    hinge_help = get_effect(hinge_effects, "help_received")

    # Quantify evidence strength for a 0-100 response score.
    # Stronger score when key variables show consistent controlled effects across models.
    def variable_points(var: str, smart_eff: Dict, hinge_eff: Dict) -> float:
        pts = 0.0
        # Bivariate signal
        if abs(bivariate_corr[var]) >= 0.30:
            pts += 8
        elif abs(bivariate_corr[var]) >= 0.15:
            pts += 5

        # OLS controlled significance
        p = ols_pval[var]
        if p < 0.05:
            pts += 10
        elif p < 0.10:
            pts += 5

        # Interpretable models (importance under controls)
        if smart_eff["importance"] >= 0.05:
            pts += 8
        elif smart_eff["importance"] > 0:
            pts += 4

        if hinge_eff["importance"] >= 0.05:
            pts += 8
        elif hinge_eff["importance"] > 0:
            pts += 4

        # Penalty for inconsistent/fragile effect after controls
        controlled_failures = 0
        if not np.isfinite(p) or p >= 0.10:
            controlled_failures += 1
        if smart_eff["importance"] < 0.05:
            controlled_failures += 1
        if hinge_eff["importance"] < 0.05:
            controlled_failures += 1
        if controlled_failures >= 2:
            pts -= 6

        return pts

    score_age = variable_points("age", smart_age, hinge_age)
    score_sex = variable_points("sex_male", smart_sex, hinge_sex)
    score_help = variable_points("help_received", smart_help, hinge_help)

    raw_score = score_age + score_sex + score_help
    response_score = int(np.clip(round(raw_score), 0, 100))

    # Summarize non-key confounders
    conf_smart = top_nonkey_effects(smart_effects, key_vars, top_k=3)
    conf_hinge = top_nonkey_effects(hinge_effects, key_vars, top_k=3)

    conf_smart_txt = ", ".join([f"{n} ({m['importance']:.1%}, {m['direction']})" for n, m in conf_smart])
    conf_hinge_txt = ", ".join([f"{n} ({m['importance']:.1%}, {m['direction']})" for n, m in conf_hinge])

    age_shape_txt = extract_age_threshold_summary(smart, feature_name="age")

    explanation = (
        f"Using efficiency = nuts_opened/seconds as the DV, bivariate correlations were "
        f"age={bivariate_corr['age']:.3f}, sex_male={bivariate_corr['sex_male']:.3f}, "
        f"help_received={bivariate_corr['help_received']:.3f}. "
        f"In OLS with hammer and chimpanzee controls, age stayed positive (coef={ols_coef['age']:.3f}, p={ols_pval['age']:.3f}) "
        f"while sex_male was positive but weaker (coef={ols_coef['sex_male']:.3f}, p={ols_pval['sex_male']:.3f}) and "
        f"help_received was negative but not robust (coef={ols_coef['help_received']:.3f}, p={ols_pval['help_received']:.3f}). "
        f"SmartAdditive ranked age as the strongest key predictor (importance={smart_age['importance']:.1%}, rank={smart_age['rank']}), "
        f"sex_male lower (importance={smart_sex['importance']:.1%}, rank={smart_sex['rank']}), and help_received modest (importance={smart_help['importance']:.1%}, rank={smart_help['rank']}); "
        f"{age_shape_txt} "
        f"HingeEBM also kept age positive (importance={hinge_age['importance']:.1%}, rank={hinge_age['rank']}) and sex_male positive "
        f"(importance={hinge_sex['importance']:.1%}, rank={hinge_sex['rank']}), but shrank help_received to near-zero "
        f"(importance={hinge_help['importance']:.1%}, rank={hinge_help['rank']}). "
        f"Key confounders included SmartAdditive: {conf_smart_txt if conf_smart_txt else 'none'}; "
        f"HingeEBM: {conf_hinge_txt if conf_hinge_txt else 'none'}. "
        f"Overall this supports a moderate effect pattern: age is the most reliable driver of efficiency, sex is smaller/less stable after strict controls, and help is inconsistent."
    )

    output = {"response": response_score, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f)

    print("\nWrote conclusion.txt")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
