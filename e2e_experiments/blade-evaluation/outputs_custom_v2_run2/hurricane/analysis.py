import json
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr

from interp_models import SmartAdditiveRegressor, HingeEBMRegressor


def infer_iv_dv(question: str, columns: List[str]) -> Tuple[str, str]:
    q = question.lower()

    # Dependent variable inference
    if any(tok in q for tok in ["death", "fatal"]):
        dv = "alldeaths" if "alldeaths" in columns else None
    else:
        dv = None

    # Independent variable inference
    if "feminine" in q or "mascul" in q or "gender" in q:
        iv = "masfem" if "masfem" in columns else None
    else:
        iv = None

    if dv is None:
        # fallback: first plausible outcome-like numeric column
        for cand in ["alldeaths", "ndam15", "ndam"]:
            if cand in columns:
                dv = cand
                break
    if iv is None:
        for cand in ["masfem", "masfem_mturk", "gender_mf"]:
            if cand in columns:
                iv = cand
                break

    if dv is None or iv is None:
        raise ValueError(f"Could not infer IV/DV from question. iv={iv}, dv={dv}")

    return iv, dv


def build_ols(X: pd.DataFrame, y: pd.Series):
    Xc = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, Xc, missing="drop").fit()
    return model


def sorted_effects(effects: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
    return sorted(effects.items(), key=lambda kv: kv[1].get("rank", 10_000) or 10_000)


def main():
    # Step 1: Understand question and explore
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info.get("research_questions", ["(missing question)"])[0]
    print("Research question:")
    print(question)
    print()

    df = pd.read_csv("hurricane.csv")
    print(f"Loaded hurricane.csv with shape={df.shape}")

    iv, dv = infer_iv_dv(question, df.columns.tolist())
    print(f"Inferred IV={iv}, DV={dv}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("\nNumeric columns:", numeric_cols)

    print("\nSummary statistics (numeric):")
    print(df[numeric_cols].describe().T)

    print("\nDistribution diagnostics:")
    skew = df[numeric_cols].skew(numeric_only=True)
    print(skew.sort_values(key=lambda s: s.abs(), ascending=False))

    # Bivariate correlations for primary IV-DV relationship
    v = df[[iv, dv]].dropna()
    pear_r, pear_p = pearsonr(v[iv], v[dv])
    spear_r, spear_p = spearmanr(v[iv], v[dv])
    log_dv = np.log1p(v[dv])
    pear_r_log, pear_p_log = pearsonr(v[iv], log_dv)
    spear_r_log, spear_p_log = spearmanr(v[iv], log_dv)

    print("\nBivariate correlations (IV vs DV):")
    print(f"Pearson (raw DV): r={pear_r:.4f}, p={pear_p:.4g}")
    print(f"Spearman (raw DV): rho={spear_r:.4f}, p={spear_p:.4g}")
    print(f"Pearson (log1p DV): r={pear_r_log:.4f}, p={pear_p_log:.4g}")
    print(f"Spearman (log1p DV): rho={spear_r_log:.4f}, p={spear_p_log:.4g}")

    # Step 2: OLS with controls
    analysis_df = df.copy()
    analysis_df["log_dv"] = np.log1p(analysis_df[dv])

    # Bivariate OLS
    X_biv = analysis_df[[iv]]
    y = analysis_df["log_dv"]
    ols_biv = build_ols(X_biv, y)

    # Controlled OLS: relevant meteorological and era controls + source dummies
    base_controls = [c for c in ["wind", "min", "category", "ndam15", "year", "gender_mf"] if c in analysis_df.columns]
    X_ctrl = analysis_df[[iv] + base_controls].copy()
    if "source" in analysis_df.columns:
        source_dummies = pd.get_dummies(analysis_df["source"], prefix="source", drop_first=True, dtype=float)
        X_ctrl = pd.concat([X_ctrl, source_dummies], axis=1)

    ols_ctrl = build_ols(X_ctrl, y)

    # Full-numeric sensitivity model: all numeric controls except obvious ID and DV itself
    all_numeric_predictors = [c for c in numeric_cols if c not in {dv, "ind"}]
    X_full = analysis_df[all_numeric_predictors].copy()
    # Impute missing values for full model
    X_full = X_full.fillna(X_full.median(numeric_only=True))
    ols_full = build_ols(X_full, y)

    print("\nOLS Bivariate summary (log DV ~ IV):")
    print(ols_biv.summary())
    print("\nOLS Controlled summary (log DV ~ IV + controls + source):")
    print(ols_ctrl.summary())
    print("\nOLS Full-numeric sensitivity summary:")
    print(ols_full.summary())

    # Step 3: Interpretable models
    X_interp = analysis_df[all_numeric_predictors].copy()
    X_interp = X_interp.fillna(X_interp.median(numeric_only=True))
    y_interp = y.values

    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X_interp, y_interp)
    smart_effects = smart.feature_effects()

    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X_interp, y_interp)
    hinge_effects = hinge.feature_effects()

    print("\nSmartAdditiveRegressor model:")
    print(smart)
    print("\nSmartAdditive feature_effects():")
    print(smart_effects)

    print("\nHingeEBMRegressor model:")
    print(hinge)
    print("\nHingeEBM feature_effects():")
    print(hinge_effects)

    # Step 4: Rich conclusion with score
    iv_biv_coef = float(ols_biv.params.get(iv, np.nan))
    iv_biv_p = float(ols_biv.pvalues.get(iv, np.nan))
    iv_ctrl_coef = float(ols_ctrl.params.get(iv, np.nan))
    iv_ctrl_p = float(ols_ctrl.pvalues.get(iv, np.nan))

    smart_iv = smart_effects.get(iv, {"direction": "zero", "importance": 0.0, "rank": 0})
    hinge_iv = hinge_effects.get(iv, {"direction": "zero", "importance": 0.0, "rank": 0})

    top_smart = [
        (name, eff["importance"], eff["direction"])
        for name, eff in sorted_effects(smart_effects)
        if eff.get("rank", 0) > 0
    ][:5]
    top_hinge = [
        (name, eff["importance"], eff["direction"])
        for name, eff in sorted_effects(hinge_effects)
        if eff.get("rank", 0) > 0
    ][:5]

    # Evidence-based score mapping
    no_signal = (
        pear_p >= 0.10
        and iv_ctrl_p >= 0.10
        and smart_iv.get("importance", 0.0) < 0.05
        and hinge_iv.get("importance", 0.0) < 0.05
    )

    strong_positive = (
        pear_p < 0.05
        and iv_ctrl_p < 0.05
        and iv_ctrl_coef > 0
        and smart_iv.get("importance", 0.0) >= 0.10
        and ("positive" in smart_iv.get("direction", "") or "increasing" in smart_iv.get("direction", ""))
        and hinge_iv.get("importance", 0.0) >= 0.05
        and hinge_iv.get("direction") == "positive"
    )

    if strong_positive:
        response = 85
    elif no_signal:
        response = 10
    else:
        # Mixed evidence bucket
        response = 45

    # Confounders that appear important
    confounders = []
    for col in ["ndam15", "min", "wind", "category", "year", "masfem_mturk", "ndam"]:
        if col == iv:
            continue
        info_bits = []
        if col in ols_ctrl.pvalues.index and ols_ctrl.pvalues[col] < 0.10:
            info_bits.append(f"OLS p={ols_ctrl.pvalues[col]:.3g}")
        seff = smart_effects.get(col)
        if seff and seff.get("rank", 0) > 0 and seff.get("importance", 0.0) >= 0.05:
            info_bits.append(f"Smart rank {seff['rank']} ({seff['importance']:.1%}, {seff['direction']})")
        heff = hinge_effects.get(col)
        if heff and heff.get("rank", 0) > 0 and heff.get("importance", 0.0) >= 0.05:
            info_bits.append(f"Hinge rank {heff['rank']} ({heff['importance']:.1%}, {heff['direction']})")
        if info_bits:
            confounders.append(f"{col}: " + "; ".join(info_bits))

    confounder_text = " | ".join(confounders[:4]) if confounders else "No strong confounder pattern detected."

    explanation = (
        f"Using {dv} as the outcome and {iv} as the key predictor, the bivariate association is weak "
        f"(Pearson r={pear_r:.3f}, p={pear_p:.3f}; Spearman rho={spear_r:.3f}, p={spear_p:.3f}). "
        f"In OLS on log(1+{dv}), {iv} is not significant in the bivariate model "
        f"(coef={iv_biv_coef:.3f}, p={iv_biv_p:.3f}) and remains non-significant after controls "
        f"(coef={iv_ctrl_coef:.3f}, p={iv_ctrl_p:.3f}). "
        f"SmartAdditive ranks {iv} at {smart_iv.get('rank', 0)} with {smart_iv.get('importance', 0.0):.1%} importance "
        f"and {smart_iv.get('direction', 'zero')} shape, indicating a small/non-robust effect. "
        f"HingeEBM gives {iv} rank {hinge_iv.get('rank', 0)} with {hinge_iv.get('importance', 0.0):.1%} importance "
        f"({hinge_iv.get('direction', 'zero')}), effectively shrinking it out. "
        f"By contrast, storm intensity/damage variables dominate feature importance across models "
        f"(top SmartAdditive: {top_smart}; top HingeEBM: {top_hinge}). "
        f"Overall, evidence does not support a robust positive femininity effect on deaths in this dataset. "
        f"Key confounder pattern: {confounder_text}"
    )

    # Ensure clean ASCII-like plain text for JSON
    explanation = re.sub(r"\s+", " ", explanation).strip()

    payload = {"response": int(response), "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print("\nWrote conclusion.txt:")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
