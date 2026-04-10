import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pointbiserialr

from interp_models import SmartAdditiveRegressor, HingeEBMRegressor

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def fmt(x, nd=4):
    if x is None:
        return "nan"
    try:
        xv = float(x)
    except Exception:
        return str(x)
    if np.isnan(xv) or np.isinf(xv):
        return "nan"
    return f"{xv:.{nd}f}"


def p_sig_label(p):
    if p is None or np.isnan(p):
        return "unavailable"
    if p < 0.01:
        return "strong"
    if p < 0.05:
        return "significant"
    if p < 0.10:
        return "marginal"
    return "weak"


def safe_pointbiserial(y, x):
    try:
        r, p = pointbiserialr(y, x)
        return float(r), float(p)
    except Exception:
        return np.nan, np.nan


def safe_logit(formula, data):
    model = smf.logit(formula, data=data)
    res = model.fit(disp=False, maxiter=200)
    return res


def get_effect(effects, feature):
    e = effects.get(feature, {})
    return {
        "direction": e.get("direction", "zero"),
        "importance": float(e.get("importance", 0.0) or 0.0),
        "rank": int(e.get("rank", 0) or 0),
    }


def threshold_summary_from_smart(model, feature_names, feature):
    if feature not in feature_names:
        return ""
    j = feature_names.index(feature)
    if not hasattr(model, "shape_functions_") or j not in model.shape_functions_:
        return ""

    thresholds, intervals = model.shape_functions_[j]
    thresholds = [float(t) for t in thresholds]
    intervals = [float(v) for v in intervals]
    if not thresholds or not intervals:
        return ""

    i_max = int(np.argmax(intervals))
    i_min = int(np.argmin(intervals))

    def region(idx):
        if idx == 0:
            return f"<= {thresholds[0]:.1f}"
        if idx == len(thresholds):
            return f"> {thresholds[-1]:.1f}"
        return f"({thresholds[idx-1]:.1f}, {thresholds[idx]:.1f}]"

    return (
        f"strongest positive region at {feature} {region(i_max)} and most negative region at "
        f"{feature} {region(i_min)}"
    )


def main():
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info.get("research_questions", ["Unknown research question"])[0].strip()
    df = pd.read_csv("crofoot.csv")

    # Research-aligned engineered variables
    # Positive rel_group_size means focal group is larger.
    # Positive loc_advantage means contest is relatively closer to focal group's range center.
    df["rel_group_size"] = df["n_focal"] - df["n_other"]
    df["loc_advantage"] = df["dist_other"] - df["dist_focal"]

    dv = "win"
    iv_size = "rel_group_size"
    iv_loc = "loc_advantage"

    print("=== RESEARCH QUESTION ===")
    print(question)
    print("DV:", dv)
    print("IVs:", iv_size, "(relative group size),", iv_loc, "(relative contest location)")

    print("\n=== STEP 1: EXPLORATION ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("Numeric columns:", numeric_cols)

    print("\nSummary statistics:")
    print(df[numeric_cols].describe().T)

    print("\nWin distribution:")
    print(df[dv].value_counts(dropna=False).sort_index())
    print(df[dv].value_counts(normalize=True, dropna=False).sort_index())

    print("\nCorrelations with DV (win):")
    corr_with_dv = df[numeric_cols].corr(numeric_only=True)[dv].sort_values(ascending=False)
    print(corr_with_dv)

    size_r, size_p_bi = safe_pointbiserial(df[dv], df[iv_size])
    loc_r, loc_p_bi = safe_pointbiserial(df[dv], df[iv_loc])
    print("\nPoint-biserial correlations:")
    print(f"{iv_size}: r={fmt(size_r)}, p={fmt(size_p_bi)}")
    print(f"{iv_loc}: r={fmt(loc_r)}, p={fmt(loc_p_bi)}")

    print("\n=== STEP 2: CONTROLLED STATISTICAL MODELS ===")
    # Main controlled model with group-level controls.
    formula_main = "win ~ rel_group_size + loc_advantage + focal + other"
    logit_main = safe_logit(formula_main, df)
    print("Main logistic model:", formula_main)
    print(logit_main.summary())

    # Robustness model: decompose location into the two raw distance terms.
    formula_alt = "win ~ rel_group_size + dist_focal + dist_other + focal + other"
    logit_alt = safe_logit(formula_alt, df)
    print("\nRobustness logistic model:", formula_alt)
    print(logit_alt.summary())

    size_coef = float(logit_main.params.get(iv_size, np.nan))
    size_p = float(logit_main.pvalues.get(iv_size, np.nan))
    loc_coef = float(logit_main.params.get(iv_loc, np.nan))
    loc_p = float(logit_main.pvalues.get(iv_loc, np.nan))

    dist_focal_coef = float(logit_alt.params.get("dist_focal", np.nan))
    dist_focal_p = float(logit_alt.pvalues.get("dist_focal", np.nan))
    dist_other_coef = float(logit_alt.params.get("dist_other", np.nan))
    dist_other_p = float(logit_alt.pvalues.get("dist_other", np.nan))

    print("\n=== STEP 3: INTERPRETABLE MODELS ===")
    feature_cols = [c for c in numeric_cols if c != dv]
    X = df[feature_cols]
    y = df[dv]

    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X, y)
    smart_effects = smart.feature_effects()
    print("SmartAdditiveRegressor:")
    print(smart)
    print("Smart feature effects:")
    print(smart_effects)

    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X, y)
    hinge_effects = hinge.feature_effects()
    print("\nHingeEBMRegressor:")
    print(hinge)
    print("Hinge feature effects:")
    print(hinge_effects)

    smart_size = get_effect(smart_effects, iv_size)
    smart_loc = get_effect(smart_effects, iv_loc)
    hinge_size = get_effect(hinge_effects, iv_size)
    hinge_loc = get_effect(hinge_effects, iv_loc)
    hinge_dist_focal = get_effect(hinge_effects, "dist_focal")
    hinge_dist_other = get_effect(hinge_effects, "dist_other")

    top_confounders = [
        (k, v) for k, v in smart_effects.items() if k not in {iv_size, iv_loc} and float(v.get("importance", 0) or 0) > 0
    ]
    top_confounders.sort(key=lambda kv: float(kv[1].get("importance", 0) or 0), reverse=True)
    top_confounders = top_confounders[:3]

    size_shape = threshold_summary_from_smart(smart, feature_cols, iv_size)
    loc_shape = threshold_summary_from_smart(smart, feature_cols, iv_loc)

    # Score synthesis (0-100), balancing consistency across methods.
    score = 5

    # Relative group size evidence.
    if size_p < 0.01:
        score += 30
    elif size_p < 0.05:
        score += 22
    elif size_p < 0.10:
        score += 14
    elif size_p < 0.20:
        score += 8
    else:
        score += 3

    if size_p_bi < 0.10:
        score += 8
    elif size_p_bi < 0.20:
        score += 4
    else:
        score += 1

    if smart_size["importance"] >= 0.10:
        score += 8
    elif smart_size["importance"] >= 0.05:
        score += 5
    else:
        score += 2

    if hinge_size["importance"] > 0:
        score += 5
    else:
        score -= 2

    # Location evidence.
    if loc_p < 0.01:
        score += 26
    elif loc_p < 0.05:
        score += 18
    elif loc_p < 0.10:
        score += 10
    elif loc_p < 0.20:
        score += 6
    else:
        score += 2

    if loc_p_bi < 0.10:
        score += 8
    elif loc_p_bi < 0.20:
        score += 4
    else:
        score += 1

    if smart_loc["importance"] >= 0.20:
        score += 12
    elif smart_loc["importance"] >= 0.10:
        score += 8
    elif smart_loc["importance"] >= 0.05:
        score += 4
    else:
        score += 1

    if hinge_loc["importance"] > 0:
        score += 5
    else:
        # Hinge may use raw distance components instead of the engineered difference.
        if (hinge_dist_focal["importance"] + hinge_dist_other["importance"]) >= 0.20:
            score += 4
        else:
            score -= 2

    if (dist_focal_p < 0.10) or (dist_other_p < 0.10):
        score += 5

    # Mild inconsistency penalty.
    if loc_p > 0.20 and smart_loc["importance"] >= 0.20:
        score -= 2

    response = int(max(0, min(100, round(score))))

    confounder_text = ", ".join(
        [
            f"{name} (importance={100*float(meta.get('importance', 0) or 0):.1f}%, rank={int(meta.get('rank', 0) or 0)})"
            for name, meta in top_confounders
        ]
    )
    if not confounder_text:
        confounder_text = "none with meaningful importance"

    explanation = (
        f"DV is win (focal group victory). Relative group size shows a positive but only marginal controlled effect "
        f"(logit coef={fmt(size_coef)}, p={fmt(size_p)}; bivariate r={fmt(size_r)}, p={fmt(size_p_bi)}). "
        f"SmartAdditive ranks rel_group_size #{smart_size['rank']} with {100*smart_size['importance']:.1f}% importance and direction "
        f"'{smart_size['direction']}' ({size_shape}). HingeEBM assigns rel_group_size {100*hinge_size['importance']:.1f}% importance "
        f"(often zeroed out), so size effects are present but not robustly strong. "
        f"Contest location is mixed in linear models: loc_advantage is weak in controlled logit (coef={fmt(loc_coef)}, p={fmt(loc_p)}) and bivariate "
        f"correlation (r={fmt(loc_r)}, p={fmt(loc_p_bi)}), but SmartAdditive ranks it #{smart_loc['rank']} with {100*smart_loc['importance']:.1f}% "
        f"importance and a '{smart_loc['direction']}' pattern ({loc_shape}), indicating threshold-like home-range advantage. "
        f"In robustness logit, dist_focal is negative and marginal (coef={fmt(dist_focal_coef)}, p={fmt(dist_focal_p)}), while dist_other is weaker "
        f"(coef={fmt(dist_other_coef)}, p={fmt(dist_other_p)}). HingeEBM captures location mostly through raw distances "
        f"(dist_focal importance={100*hinge_dist_focal['importance']:.1f}%, dist_other={100*hinge_dist_other['importance']:.1f}%) rather than the "
        f"constructed difference variable. Key confounders also matter: {confounder_text}. Overall, evidence supports a moderate/partial influence: "
        f"relative size is weak-to-marginally positive and location effects appear nonlinear rather than consistently linear across models."
    )

    result = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=True)

    print("\n=== FINAL CONCLUSION JSON ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
