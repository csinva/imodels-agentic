import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from interp_models import SmartAdditiveRegressor, HingeEBMRegressor


INFO_PATH = Path("info.json")
DATA_PATH = Path("soccer.csv")
CONCLUSION_PATH = Path("conclusion.txt")


def safe_float(x):
    try:
        if x is None:
            return None
        val = float(x)
        if np.isfinite(val):
            return val
        return None
    except Exception:
        return None


def fmt_num(x, digits=4):
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"


def summarize_top_effects(effects, exclude=None, top_k=3):
    exclude = exclude or set()
    rows = []
    for name, info in effects.items():
        if name in exclude:
            continue
        imp = info.get("importance", 0.0) or 0.0
        if imp > 0:
            rows.append((name, info.get("direction", "unknown"), imp, info.get("rank", 0)))
    rows.sort(key=lambda x: x[2], reverse=True)
    return rows[:top_k]


def rank_of_feature(effects, feature):
    info = effects.get(feature, {})
    return info.get("rank", 0), info.get("importance", 0.0), info.get("direction", "zero")


def main():
    with INFO_PATH.open() as f:
        info = json.load(f)

    research_question = info["research_questions"][0]
    print("Research question:")
    print(research_question)
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset with shape: {df.shape}")

    # Define IV and DV from question/context
    dv_col = "redCards"
    iv_col = "skin_tone"

    # Build skin tone from two raters
    df[iv_col] = df[["rater1", "rater2"]].mean(axis=1, skipna=True)

    # Keep rows where IV and DV are observed
    analysis_df = df.dropna(subset=[iv_col, dv_col]).copy()

    # Binary dark-skin indicator for descriptive comparison
    analysis_df["dark_skin"] = (analysis_df[iv_col] >= 0.5).astype(int)
    analysis_df["any_red_card"] = (analysis_df[dv_col] > 0).astype(int)

    print(f"Rows after requiring non-missing skin tone and red cards: {len(analysis_df)}")

    # Step 1: Explore
    print("\nStep 1: Summary statistics and bivariate relationships")
    vars_for_summary = [
        iv_col,
        "dark_skin",
        dv_col,
        "any_red_card",
        "games",
        "yellowCards",
        "yellowReds",
        "goals",
        "meanIAT",
        "meanExp",
    ]
    print(analysis_df[vars_for_summary].describe().T)

    corr = analysis_df[[iv_col, dv_col, "any_red_card", "games", "yellowCards"]].corr(numeric_only=True)
    print("\nCorrelation matrix (key variables):")
    print(corr)

    # Group descriptive stats for dark vs light
    group_stats = analysis_df.groupby("dark_skin")[dv_col].agg(["mean", "std", "count"])
    red_rate = analysis_df.groupby("dark_skin")["any_red_card"].mean()
    print("\nRed card counts by dark_skin (0=lighter, 1=darker):")
    print(group_stats)
    print("\nAny-red-card rate by dark_skin (0=lighter, 1=darker):")
    print(red_rate)

    # Bivariate OLS
    X_biv = sm.add_constant(analysis_df[[iv_col]])
    y = analysis_df[dv_col]
    biv_model = sm.OLS(y, X_biv).fit(cov_type="HC3")
    print("\nBivariate OLS: redCards ~ skin_tone")
    print(biv_model.summary())

    # Step 2: OLS with controls
    print("\nStep 2: Controlled OLS")
    base_numeric_controls = [
        "games",
        "goals",
        "yellowCards",
        "yellowReds",
        "victories",
        "ties",
        "defeats",
        "height",
        "weight",
        "meanIAT",
        "meanExp",
        "seIAT",
        "seExp",
        "nIAT",
        "nExp",
    ]

    # Include league and position fixed effects via dummies
    cat_cols = ["leagueCountry", "position"]
    ols_df = analysis_df[[dv_col, iv_col] + base_numeric_controls + cat_cols].copy()

    # Missing-value handling for OLS
    for c in [iv_col] + base_numeric_controls:
        med = ols_df[c].median()
        ols_df[c] = ols_df[c].fillna(med)

    for c in cat_cols:
        ols_df[c] = ols_df[c].fillna("Missing")

    cat_dummies = pd.get_dummies(ols_df[cat_cols], drop_first=True)
    X_ctrl = pd.concat([ols_df[[iv_col] + base_numeric_controls], cat_dummies], axis=1)
    X_ctrl = sm.add_constant(X_ctrl).astype(float)
    y_ctrl = ols_df[dv_col].astype(float)

    ctrl_model = sm.OLS(y_ctrl, X_ctrl).fit(cov_type="HC3")
    print("Controlled OLS: redCards ~ skin_tone + controls")
    print(ctrl_model.summary())

    # Step 3: Interpretable models on numeric columns with valid controls
    # Exclude target-derived columns and direct components of IV construction.
    print("\nStep 3: Interpretable models")
    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
    excluded_for_leakage_or_redundancy = {
        dv_col,          # target
        "any_red_card",  # target-derived
        "dark_skin",     # derived threshold of IV
        "rater1",        # IV component
        "rater2",        # IV component
    }
    feature_cols = [c for c in numeric_cols if c not in excluded_for_leakage_or_redundancy]

    model_df = analysis_df[feature_cols + [dv_col]].copy()
    for c in feature_cols:
        model_df[c] = model_df[c].fillna(model_df[c].median())

    X_model = model_df[feature_cols]
    y_model = model_df[dv_col]

    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X_model, y_model)
    smart_effects = smart.feature_effects()
    print("\nSmartAdditiveRegressor model:")
    print(smart)
    print("\nSmartAdditive feature effects:")
    print(smart_effects)

    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X_model, y_model)
    hinge_effects = hinge.feature_effects()
    print("\nHingeEBMRegressor model:")
    print(hinge)
    print("\nHingeEBM feature effects:")
    print(hinge_effects)

    # Gather IV evidence
    biv_coef = safe_float(biv_model.params.get(iv_col))
    biv_p = safe_float(biv_model.pvalues.get(iv_col))

    ctrl_coef = safe_float(ctrl_model.params.get(iv_col))
    ctrl_p = safe_float(ctrl_model.pvalues.get(iv_col))

    smart_rank, smart_imp, smart_dir = rank_of_feature(smart_effects, iv_col)
    hinge_rank, hinge_imp, hinge_dir = rank_of_feature(hinge_effects, iv_col)

    # Try to recover nonlinear threshold details for skin_tone from SmartAdditive
    shape_desc = ""
    if iv_col in getattr(smart, "feature_names_", []):
        j = smart.feature_names_.index(iv_col)
        if j in getattr(smart, "shape_functions_", {}):
            thresholds, intervals = smart.shape_functions_[j]
            if len(thresholds) > 0:
                first_t = thresholds[0]
                low_val = intervals[0]
                high_val = intervals[-1]
                shape_desc = (
                    f"SmartAdditive shows threshold-like changes around skin_tone={first_t:.3f}; "
                    f"effect level shifts from {low_val:.4f} (lowest bin) to {high_val:.4f} (highest bin)."
                )

    # Scoring logic (0-100)
    score = 50

    # Controlled OLS gets highest weight
    if ctrl_coef is not None and ctrl_coef > 0:
        if ctrl_p is not None:
            if ctrl_p < 0.001:
                score = 85
            elif ctrl_p < 0.01:
                score = 78
            elif ctrl_p < 0.05:
                score = 70
            elif ctrl_p < 0.10:
                score = 60
            else:
                score = 48
        else:
            score = 55
    else:
        if ctrl_p is not None and ctrl_p < 0.05:
            score = 15
        else:
            score = 25

    # Bivariate support adjustment
    if biv_coef is not None and biv_coef > 0 and biv_p is not None and biv_p < 0.05:
        score += 5
    elif biv_coef is not None and biv_coef < 0:
        score -= 8

    # Interpretable model support adjustment
    supportive = 0
    if smart_imp > 0 and (
        smart_dir.startswith("positive")
        or smart_dir.startswith("nonlinear (increasing")
    ):
        supportive += 1
    elif smart_imp > 0:
        supportive -= 1

    if hinge_imp > 0 and hinge_dir == "positive":
        supportive += 1
    elif hinge_imp > 0 and hinge_dir == "negative":
        supportive -= 1

    score += 6 * supportive

    # If IV is very low-importance in both interpretable models, dampen
    if smart_imp < 0.01 and hinge_imp < 0.01:
        score -= 10

    score = int(max(0, min(100, round(score))))

    # Confounders/top features
    smart_top = summarize_top_effects(smart_effects, exclude={iv_col}, top_k=3)
    hinge_top = summarize_top_effects(hinge_effects, exclude={iv_col}, top_k=3)

    def top_to_text(rows):
        if not rows:
            return "none stood out"
        parts = []
        for name, direction, imp, rank in rows:
            parts.append(f"{name} (rank {rank}, {direction}, importance={imp:.1%})")
        return "; ".join(parts)

    dark_mean = safe_float(group_stats.loc[1, "mean"]) if 1 in group_stats.index else None
    light_mean = safe_float(group_stats.loc[0, "mean"]) if 0 in group_stats.index else None
    dark_rate = safe_float(red_rate.loc[1]) if 1 in red_rate.index else None
    light_rate = safe_float(red_rate.loc[0]) if 0 in red_rate.index else None

    explanation = (
        f"Question: {research_question} "
        f"Using {len(analysis_df)} player-referee dyads with rated skin tone, darker-toned players had "
        f"higher raw red-card counts than lighter-toned players (mean {fmt_num(dark_mean, 4)} vs {fmt_num(light_mean, 4)}; "
        f"any-red-card rate {fmt_num(dark_rate, 4)} vs {fmt_num(light_rate, 4)}). "
        f"Bivariate OLS shows skin_tone -> redCards coef={fmt_num(biv_coef, 4)}, p={fmt_num(biv_p, 4)}. "
        f"After controls (exposure, performance/discipline, body metrics, league/position, referee-country bias proxies), "
        f"the skin_tone effect is coef={fmt_num(ctrl_coef, 4)}, p={fmt_num(ctrl_p, 4)}. "
        f"SmartAdditive ranks skin_tone #{smart_rank} with importance={smart_imp:.1%} and direction='{smart_dir}'; "
        f"HingeEBM ranks it #{hinge_rank} with importance={hinge_imp:.1%} and direction='{hinge_dir}'. "
        f"{shape_desc} "
        f"Key confounders in SmartAdditive: {top_to_text(smart_top)}. "
        f"Key confounders in HingeEBM: {top_to_text(hinge_top)}. "
        f"Overall, the IV-DV relationship is {'robust' if score >= 75 else 'moderate/partial' if score >= 40 else 'weak/inconsistent'} across models, yielding a Likert score of {score}/100."
    )

    result = {
        "response": score,
        "explanation": explanation,
    }

    with CONCLUSION_PATH.open("w") as f:
        json.dump(result, f)

    print("\nWrote conclusion.txt")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
