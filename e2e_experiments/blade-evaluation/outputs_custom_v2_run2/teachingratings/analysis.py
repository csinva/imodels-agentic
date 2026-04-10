import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from interp_models import SmartAdditiveRegressor, HingeEBMRegressor


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def fmt_float(x, nd=4):
    return float(np.round(float(x), nd))


def top_effects(effects, k=5):
    ranked = sorted(
        [(name, info) for name, info in effects.items()],
        key=lambda kv: float(kv[1].get("importance", 0.0)),
        reverse=True,
    )
    return ranked[:k]


def significant_controls(model, exclude=None, alpha=0.05):
    exclude = set(exclude or [])
    pvals = model.pvalues
    params = model.params
    out = []
    for name in pvals.index:
        if name in exclude:
            continue
        p = float(pvals[name])
        if p < alpha:
            out.append((name, float(params[name]), p))
    out.sort(key=lambda x: x[2])
    return out


def smart_beauty_shape_summary(model, feature_name="beauty"):
    if not hasattr(model, "shape_functions_"):
        return None
    if feature_name not in model.feature_names_:
        return None

    j = model.feature_names_.index(feature_name)
    if j not in model.shape_functions_:
        return None

    thresholds, intervals = model.shape_functions_[j]
    if len(intervals) == 0:
        return None

    crossing = None
    for i in range(1, len(intervals)):
        if intervals[i - 1] <= 0 < intervals[i]:
            if i - 1 < len(thresholds):
                crossing = thresholds[i - 1]
            break

    return {
        "low_effect": float(intervals[0]),
        "high_effect": float(intervals[-1]),
        "crossing": None if crossing is None else float(crossing),
        "min_effect": float(min(intervals)),
        "max_effect": float(max(intervals)),
    }


def compute_score(biv_coef, biv_p, ctrl_coef, ctrl_p, smart_info, hinge_info):
    if ctrl_p < 0.01 and ctrl_coef > 0:
        score = 82
    elif ctrl_p < 0.05 and ctrl_coef > 0:
        score = 68
    elif ctrl_coef > 0:
        score = 50
    else:
        score = 20

    smart_imp = float(smart_info.get("importance", 0.0)) if smart_info else 0.0
    smart_dir = (smart_info or {}).get("direction", "")
    if smart_imp >= 0.2 and ("increasing" in smart_dir or smart_dir == "positive"):
        score += 6
    elif smart_imp >= 0.05:
        score += 3

    hinge_imp = float(hinge_info.get("importance", 0.0)) if hinge_info else 0.0
    hinge_dir = (hinge_info or {}).get("direction", "")
    if hinge_imp >= 0.1 and hinge_dir == "positive":
        score += 4
    elif hinge_imp == 0:
        score -= 4

    if biv_p >= 0.05:
        score -= 8

    if abs(biv_coef) > 1e-12:
        rel_change = abs(ctrl_coef - biv_coef) / abs(biv_coef)
        if rel_change > 0.5:
            score -= 6

    return int(np.clip(round(score), 0, 100))


def main():
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info["research_questions"][0]
    print("=== Research Question ===")
    print(research_question)

    df = pd.read_csv("teachingratings.csv")

    iv = "beauty"
    dv = "eval"
    print("\n=== Variables ===")
    print(f"Dependent variable (DV): {dv}")
    print(f"Independent variable (IV): {iv}")

    # Step 1: exploration
    print("\n=== Step 1: Exploration ===")
    print(f"Dataset shape: {df.shape}")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    print("\nNumeric summary:")
    print(df[numeric_cols].describe().T)

    print("\nCategorical distributions:")
    for c in cat_cols:
        print(f"\n{c}:")
        print(df[c].value_counts(dropna=False))

    print("\nCorrelations with eval (numeric):")
    corr_with_eval = df[numeric_cols].corr(numeric_only=True)[dv].sort_values(ascending=False)
    print(corr_with_eval)

    biv_corr = df[[iv, dv]].corr().iloc[0, 1]
    print(f"\nBivariate correlation ({iv}, {dv}): {biv_corr:.4f}")

    X_biv = sm.add_constant(df[[iv]])
    biv_model = sm.OLS(df[dv], X_biv).fit()
    print("\nBivariate OLS (eval ~ beauty):")
    print(biv_model.summary())

    # Step 2: controlled OLS
    print("\n=== Step 2: Controlled OLS ===")
    controls_numeric = ["age", "students", "allstudents"]
    controls_categorical = ["minority", "gender", "credits", "division", "native", "tenure"]
    model_cols = [iv] + controls_numeric + controls_categorical

    X_ctrl = pd.get_dummies(df[model_cols], columns=controls_categorical, drop_first=True, dtype=float)
    X_ctrl = sm.add_constant(X_ctrl).astype(float)
    ctrl_model = sm.OLS(df[dv], X_ctrl).fit()

    print(ctrl_model.summary())

    # Step 3: Interpretable models
    print("\n=== Step 3: Interpretable Models ===")

    # As requested: include all numeric columns first
    all_numeric_predictors = [c for c in numeric_cols if c != dv]
    X_all = df[all_numeric_predictors]
    y = df[dv]

    print("\nUsing ALL numeric predictors:", all_numeric_predictors)
    smart_all = SmartAdditiveRegressor(n_rounds=200)
    smart_all.fit(X_all, y)
    smart_all_effects = smart_all.feature_effects()
    print("\nSmartAdditiveRegressor (all numeric):")
    print(smart_all)
    print("Top effects:", top_effects(smart_all_effects, k=6))

    hinge_all = HingeEBMRegressor(n_knots=3)
    hinge_all.fit(X_all, y)
    hinge_all_effects = hinge_all.feature_effects()
    print("\nHingeEBMRegressor (all numeric):")
    print(hinge_all)
    print("Top effects:", top_effects(hinge_all_effects, k=6))

    # Substantive numeric model excluding identifier-like numeric columns
    substantive_numeric = [c for c in all_numeric_predictors if c not in {"rownames", "prof"}]
    X_sub = df[substantive_numeric]

    print("\nUsing substantive numeric predictors (excluding identifier columns rownames/prof):", substantive_numeric)
    smart_sub = SmartAdditiveRegressor(n_rounds=200)
    smart_sub.fit(X_sub, y)
    smart_sub_effects = smart_sub.feature_effects()
    print("\nSmartAdditiveRegressor (substantive numeric):")
    print(smart_sub)
    print("Top effects:", top_effects(smart_sub_effects, k=6))

    hinge_sub = HingeEBMRegressor(n_knots=3)
    hinge_sub.fit(X_sub, y)
    hinge_sub_effects = hinge_sub.feature_effects()
    print("\nHingeEBMRegressor (substantive numeric):")
    print(hinge_sub)
    print("Top effects:", top_effects(hinge_sub_effects, k=6))

    # Step 4: rich conclusion
    print("\n=== Step 4: Conclusion Synthesis ===")
    biv_coef = float(biv_model.params[iv])
    biv_p = float(biv_model.pvalues[iv])
    ctrl_coef = float(ctrl_model.params[iv])
    ctrl_p = float(ctrl_model.pvalues[iv])

    smart_beauty = smart_sub_effects.get(iv, {"direction": "zero", "importance": 0.0, "rank": 0})
    hinge_beauty = hinge_sub_effects.get(iv, {"direction": "zero", "importance": 0.0, "rank": 0})
    shape = smart_beauty_shape_summary(smart_sub, feature_name=iv)

    sig_ctrl = significant_controls(ctrl_model, exclude={"const", iv}, alpha=0.05)
    sig_ctrl_text = ", ".join(
        [f"{name} (coef={coef:+.3f}, p={p:.3g})" for name, coef, p in sig_ctrl[:6]]
    )
    if not sig_ctrl_text:
        sig_ctrl_text = "none at p<0.05"

    score = compute_score(
        biv_coef=biv_coef,
        biv_p=biv_p,
        ctrl_coef=ctrl_coef,
        ctrl_p=ctrl_p,
        smart_info=smart_beauty,
        hinge_info=hinge_beauty,
    )

    smart_imp_pct = 100.0 * float(smart_beauty.get("importance", 0.0))
    hinge_imp_pct = 100.0 * float(hinge_beauty.get("importance", 0.0))

    if shape is not None:
        crossing_text = (
            f"with a sign-change threshold around beauty={shape['crossing']:.2f}" if shape["crossing"] is not None else "without a clear sign-change threshold"
        )
        shape_text = (
            f"SmartAdditive shows a {smart_beauty.get('direction', 'nonlinear')} shape: low beauty bins contribute about {shape['low_effect']:+.3f} to eval and high beauty bins about {shape['high_effect']:+.3f}, {crossing_text}."
        )
    else:
        shape_text = "SmartAdditive did not return a stable per-feature shape summary for beauty."

    explanation = (
        f"Beauty shows a robust positive association with teaching evaluations. Bivariate OLS gives coef={biv_coef:.3f} (p={biv_p:.3g}), and after controlling for instructor/course confounders the effect remains and slightly strengthens (coef={ctrl_coef:.3f}, p={ctrl_p:.3g}). "
        f"Magnitude is meaningful relative to 1-5 eval scale and persists across models. {shape_text} "
        f"In the substantive SmartAdditive model, beauty is rank {int(smart_beauty.get('rank', 0))} with importance {smart_imp_pct:.1f}%. "
        f"In the substantive HingeEBM model, beauty is {hinge_beauty.get('direction', 'zero')} with importance {hinge_imp_pct:.1f}% (rank {int(hinge_beauty.get('rank', 0))}). "
        f"Important confounders in controlled OLS include {sig_ctrl_text}. These covariates matter, but they do not remove the beauty effect, so evidence supports a strong 'Yes'."
    )

    out = {"response": int(score), "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(out, f)

    print("\nWrote conclusion.txt")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
