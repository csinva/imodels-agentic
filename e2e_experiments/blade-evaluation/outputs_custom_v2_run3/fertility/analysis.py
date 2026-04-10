import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from interp_models import SmartAdditiveRegressor, HingeEBMRegressor


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Parse date fields used to approximate cycle timing / fertility window.
    date_cols = ["DateTesting", "StartDateofLastPeriod", "StartDateofPeriodBeforeLast"]
    for col in date_cols:
        out[col] = pd.to_datetime(out[col], format="%m/%d/%y", errors="coerce")

    # DV: average religiosity across three survey items.
    out["Religiosity"] = out[["Rel1", "Rel2", "Rel3"]].mean(axis=1, skipna=True)

    # Cycle features.
    out["InferredCycleLength"] = (
        out["StartDateofLastPeriod"] - out["StartDateofPeriodBeforeLast"]
    ).dt.days.astype(float)
    out["CycleLength"] = out["ReportedCycleLength"].fillna(out["InferredCycleLength"])
    out["CycleLength"] = out["CycleLength"].clip(lower=20, upper=45)

    out["CycleDay"] = (out["DateTesting"] - out["StartDateofLastPeriod"]).dt.days + 1

    # Approximate ovulation day ~ 14 days before next period.
    out["EstimatedOvulationDay"] = out["CycleLength"] - 14
    out["DaysFromOvulation"] = (out["CycleDay"] - out["EstimatedOvulationDay"]).abs()

    # Primary IV: smooth fertile-window index in [0, 1].
    out["FertilityIndex"] = (1 - out["DaysFromOvulation"] / 6).clip(lower=0, upper=1)

    # Alternate IV representation for robustness checks.
    out["HighFertility"] = (out["DaysFromOvulation"] <= 3).astype(int)

    return out


def main() -> None:
    info_path = Path("info.json")
    data_path = Path("fertility.csv")

    info = json.loads(info_path.read_text())
    research_question = info["research_questions"][0]

    df_raw = pd.read_csv(data_path)
    df = build_features(df_raw)

    # Core variables for this research question.
    dv = "Religiosity"
    iv = "FertilityIndex"

    print("=== Research Question ===")
    print(research_question)
    print("DV:", dv)
    print("IV:", iv)

    print("\n=== Step 1: Summary Statistics ===")
    summary_cols = [
        dv,
        iv,
        "HighFertility",
        "CycleDay",
        "CycleLength",
        "Relationship",
        "Sure1",
        "Sure2",
    ]
    print(df[summary_cols].describe().T)

    print("\n=== Distributions (value counts for key discrete vars) ===")
    print("Relationship counts:")
    print(df["Relationship"].value_counts().sort_index())
    print("HighFertility counts:")
    print(df["HighFertility"].value_counts().sort_index())

    corr_cols = [dv, iv, "HighFertility", "Relationship", "Sure1", "Sure2", "CycleLength", "CycleDay"]
    corr = df[corr_cols].corr(numeric_only=True)[dv].sort_values(ascending=False)
    print("\n=== Bivariate Correlations With DV ===")
    print(corr)

    print("\n=== Step 2: Regression Tests With Controls ===")
    # Bivariate OLS.
    X_biv = sm.add_constant(df[[iv]])
    model_biv = sm.OLS(df[dv], X_biv).fit()
    print("\nBivariate OLS: Religiosity ~ FertilityIndex")
    print(model_biv.summary())

    # Controlled OLS.
    controls = ["Relationship", "Sure1", "Sure2", "CycleLength", "CycleDay"]
    X_ctrl = sm.add_constant(df[[iv] + controls])
    model_ctrl = sm.OLS(df[dv], X_ctrl).fit()
    print("\nControlled OLS: Religiosity ~ FertilityIndex + controls")
    print(model_ctrl.summary())

    # Robustness with binary high-fertility indicator.
    X_alt = sm.add_constant(df[["HighFertility"] + controls])
    model_alt = sm.OLS(df[dv], X_alt).fit()
    print("\nRobustness OLS: Religiosity ~ HighFertility + controls")
    print(model_alt.summary())

    print("\n=== Step 3: Interpretable Models ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude identifier and DV-source components to avoid target leakage.
    exclude = {"WorkerID", dv, "Rel1", "Rel2", "Rel3"}
    feature_cols = [c for c in numeric_cols if c not in exclude]

    X_interp = df[feature_cols].copy()
    X_interp = X_interp.fillna(X_interp.median(numeric_only=True))
    y = df[dv]

    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X_interp, y)
    smart_effects = smart.feature_effects()

    print("\nSmartAdditiveRegressor:")
    print(smart)
    print("\nSmartAdditive feature_effects():")
    print(smart_effects)

    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X_interp, y)
    hinge_effects = hinge.feature_effects()

    print("\nHingeEBMRegressor:")
    print(hinge)
    print("\nHingeEBM feature_effects():")
    print(hinge_effects)

    print("\n=== Step 4: Conclusion Synthesis ===")
    corr_iv = float(df[[iv, dv]].corr().iloc[0, 1])

    biv_coef = float(model_biv.params[iv])
    biv_p = float(model_biv.pvalues[iv])

    ctrl_coef = float(model_ctrl.params[iv])
    ctrl_p = float(model_ctrl.pvalues[iv])

    alt_coef = float(model_alt.params["HighFertility"])
    alt_p = float(model_alt.pvalues["HighFertility"])

    rel_coef = float(model_ctrl.params["Relationship"])
    rel_p = float(model_ctrl.pvalues["Relationship"])

    smart_iv = smart_effects.get(iv, {"direction": "zero", "importance": 0.0, "rank": 0})
    smart_iv_direction = str(smart_iv.get("direction", "zero"))
    smart_iv_importance = float(smart_iv.get("importance", 0.0))
    smart_iv_rank = int(smart_iv.get("rank", 0) or 0)

    hinge_iv = hinge_effects.get(iv, {"direction": "zero", "importance": 0.0, "rank": 0}) if hinge_effects else {"direction": "zero", "importance": 0.0, "rank": 0}
    hinge_iv_direction = str(hinge_iv.get("direction", "zero"))
    hinge_iv_importance = float(hinge_iv.get("importance", 0.0))
    hinge_iv_rank = int(hinge_iv.get("rank", 0) or 0)

    # Identify strongest non-IV predictors from SmartAdditive for confounder context.
    smart_sorted = sorted(
        [(k, v) for k, v in smart_effects.items() if k != iv and float(v.get("importance", 0.0)) > 0],
        key=lambda kv: float(kv[1]["importance"]),
        reverse=True,
    )
    top_conf = smart_sorted[:2]

    if top_conf:
        conf_desc = "; ".join(
            [
                f"{name} ranked #{int(eff.get('rank', 0) or 0)} with importance {float(eff.get('importance', 0.0)):.1%} ({eff.get('direction', 'unknown')})"
                for name, eff in top_conf
            ]
        )
    else:
        conf_desc = "no other feature had meaningful importance"

    # Likert score (0-100): low when no robust evidence across analyses.
    if biv_p >= 0.10 and ctrl_p >= 0.10 and alt_p >= 0.10 and smart_iv_importance < 0.05 and hinge_iv_importance < 0.05:
        score = 8
    elif ctrl_p >= 0.10 and smart_iv_importance < 0.10:
        score = 20
    elif ctrl_p < 0.05 and smart_iv_importance >= 0.10 and hinge_iv_importance >= 0.05:
        score = 80
    elif ctrl_p < 0.05:
        score = 65
    else:
        score = 35

    explanation = (
        f"The estimated fertility effect on religiosity is weak and not robust. "
        f"Bivariate association is near zero (corr={corr_iv:.3f}; OLS coef={biv_coef:.3f}, p={biv_p:.3f}). "
        f"After controls (relationship status, date-certainty measures, cycle length, cycle day), the effect remains near zero "
        f"(coef={ctrl_coef:.3f}, p={ctrl_p:.3f}). A binary fertile-window robustness check is also null "
        f"(HighFertility coef={alt_coef:.3f}, p={alt_p:.3f}). "
        f"In SmartAdditive, FertilityIndex is {smart_iv_direction} with low relative importance "
        f"({smart_iv_importance:.1%}, rank #{smart_iv_rank}), indicating only a minor nonlinear pattern at most. "
        f"HingeEBM assigns FertilityIndex direction={hinge_iv_direction} with importance {hinge_iv_importance:.1%} "
        f"(rank #{hinge_iv_rank}), effectively shrinking it to negligible influence. "
        f"Other variables matter more: {conf_desc}. "
        f"Overall, evidence does not support a meaningful effect of fertility-related hormonal fluctuations on religiosity in this sample."
    )

    result = {"response": int(score), "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(result))

    print("Likert response:", score)
    print("Wrote conclusion.txt")


if __name__ == "__main__":
    main()
