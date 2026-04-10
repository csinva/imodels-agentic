import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from interp_models import SmartAdditiveRegressor, HingeEBMRegressor


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def fmt_p(p):
    if pd.isna(p):
        return "nan"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def top_effects(effects, k=3):
    ranked = [
        (name, info)
        for name, info in effects.items()
        if info.get("importance", 0) > 0 and info.get("rank", 0) > 0
    ]
    ranked.sort(key=lambda t: t[1]["rank"])
    return ranked[:k]


def get_effect(effects, key):
    return effects.get(key, {"direction": "zero", "importance": 0.0, "rank": 0})


def build_score(age_p, age_corr, age_imp_smart, age_imp_hinge, sign_consistent):
    score = 10

    if not pd.isna(age_p):
        if age_p < 0.01:
            score += 45
        elif age_p < 0.05:
            score += 35
        elif age_p < 0.10:
            score += 20
        else:
            score += 5

    mean_imp = np.nanmean([age_imp_smart, age_imp_hinge])
    if mean_imp >= 0.20:
        score += 30
    elif mean_imp >= 0.10:
        score += 20
    elif mean_imp >= 0.05:
        score += 12
    elif mean_imp >= 0.02:
        score += 7
    else:
        score += 2

    abs_corr = abs(age_corr) if not pd.isna(age_corr) else 0.0
    if abs_corr >= 0.25:
        score += 15
    elif abs_corr >= 0.15:
        score += 10
    elif abs_corr >= 0.08:
        score += 6
    elif abs_corr >= 0.03:
        score += 3

    if not sign_consistent:
        score -= 12

    return int(max(0, min(100, round(score))))


def main():
    info = json.loads(Path("info.json").read_text())
    question = info.get("research_questions", [""])[0]

    print("=" * 80)
    print("Research question")
    print(question)
    print("=" * 80)

    df = pd.read_csv("boxes.csv")

    # DV: reliance on majority option (binary)
    df["majority_choice"] = (df["y"] == 2).astype(int)

    dv = "majority_choice"
    iv = "age"
    control_cols = ["gender", "majority_first", "culture"]
    numeric_cols = ["age", "gender", "majority_first", "culture"]

    print("\nStep 1: Exploration")
    print(f"Rows: {len(df)}")
    print("Columns:", df.columns.tolist())

    print("\nOutcome distribution (original y):")
    print(df["y"].value_counts().sort_index())
    print("\nOutcome distribution (majority_choice):")
    print(df[dv].value_counts().sort_index())

    print("\nSummary statistics:")
    print(df[["y", dv] + numeric_cols].describe().T)

    print("\nBivariate correlations with majority_choice:")
    corr_series = df[[dv] + numeric_cols].corr(numeric_only=True)[dv].drop(dv)
    print(corr_series)

    age_corr = safe_float(corr_series.get(iv, np.nan))
    age_corr_p = stats.pearsonr(df[iv], df[dv]).pvalue
    print(
        f"\nAge vs majority_choice Pearson r={age_corr:.3f}, p={fmt_p(age_corr_p)}"
    )

    print("\nStep 2: Controlled models")
    # Logistic regression with culture as categorical controls
    X_logit = pd.concat(
        [
            df[[iv, "gender", "majority_first"]],
            pd.get_dummies(df["culture"], prefix="culture", drop_first=True, dtype=float),
        ],
        axis=1,
    )
    X_logit = sm.add_constant(X_logit)

    logit_model = sm.Logit(df[dv], X_logit).fit(disp=False)
    print("\nLogit model summary:")
    print(logit_model.summary())

    # OLS linear probability model as robustness check
    X_ols = sm.add_constant(df[[iv] + control_cols])
    ols_model = sm.OLS(df[dv], X_ols).fit()
    print("\nOLS model summary:")
    print(ols_model.summary())

    age_coef_logit = safe_float(logit_model.params.get(iv, np.nan))
    age_p_logit = safe_float(logit_model.pvalues.get(iv, np.nan))
    age_coef_ols = safe_float(ols_model.params.get(iv, np.nan))
    age_p_ols = safe_float(ols_model.pvalues.get(iv, np.nan))

    print("\nStep 3: Interpretable models")
    X_interp = df[numeric_cols]
    y_interp = df[dv]

    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X_interp, y_interp)
    print("\nSmartAdditiveRegressor:")
    print(smart)
    smart_effects = smart.feature_effects()
    print("\nSmartAdditive feature effects:")
    print(smart_effects)

    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X_interp, y_interp)
    print("\nHingeEBMRegressor:")
    print(hinge)
    hinge_effects = hinge.feature_effects()
    print("\nHingeEBM feature effects:")
    print(hinge_effects)

    age_smart = get_effect(smart_effects, iv)
    age_hinge = get_effect(hinge_effects, iv)

    age_imp_smart = safe_float(age_smart.get("importance", 0.0), 0.0)
    age_imp_hinge = safe_float(age_hinge.get("importance", 0.0), 0.0)

    # Characterize age shape from SmartAdditive thresholds
    shape_note = age_smart.get("direction", "zero")
    if "nonlinear" in shape_note:
        try:
            age_idx = X_interp.columns.get_loc(iv)
            thresholds = smart.shape_functions_[age_idx][0]
            if len(thresholds) > 0:
                shape_note += (
                    f" with thresholds spanning roughly {min(thresholds):.2f} to {max(thresholds):.2f} years"
                )
        except Exception:
            pass

    sign_logit = np.sign(age_coef_logit) if not pd.isna(age_coef_logit) else 0
    sign_ols = np.sign(age_coef_ols) if not pd.isna(age_coef_ols) else 0
    sign_smart = 0
    if "positive" in age_smart.get("direction", "") or "increasing" in age_smart.get("direction", ""):
        sign_smart = 1
    elif "negative" in age_smart.get("direction", "") or "decreasing" in age_smart.get("direction", ""):
        sign_smart = -1
    sign_hinge = 0
    if age_hinge.get("direction") == "positive":
        sign_hinge = 1
    elif age_hinge.get("direction") == "negative":
        sign_hinge = -1

    nonzero_signs = [s for s in [sign_logit, sign_ols, sign_smart, sign_hinge] if s != 0]
    sign_consistent = len(set(nonzero_signs)) <= 1 if nonzero_signs else True

    score = build_score(
        age_p=age_p_logit,
        age_corr=age_corr,
        age_imp_smart=age_imp_smart,
        age_imp_hinge=age_imp_hinge,
        sign_consistent=sign_consistent,
    )

    # Identify notable confounders from controlled models and feature importance
    sig_controls = []
    for name, p in logit_model.pvalues.items():
        if name in {"const", iv}:
            continue
        if p < 0.05:
            sig_controls.append(f"{name} (p={fmt_p(p)})")

    smart_top = top_effects(smart_effects, k=4)
    hinge_top = top_effects(hinge_effects, k=4)

    smart_top_str = ", ".join(
        [f"{n} (rank {d['rank']}, imp {100*d['importance']:.1f}%, {d['direction']})" for n, d in smart_top]
    ) or "none"
    hinge_top_str = ", ".join(
        [f"{n} (rank {d['rank']}, imp {100*d['importance']:.1f}%, {d['direction']})" for n, d in hinge_top]
    ) or "none"

    explanation = (
        f"IV=age, DV=majority_choice (y==2). Bivariate age-majority association is "
        f"r={age_corr:.3f} (p={fmt_p(age_corr_p)}). In controlled logistic regression "
        f"(adjusting for gender, majority_first, and culture dummies), age has coef={age_coef_logit:.3f} "
        f"(p={fmt_p(age_p_logit)}); OLS robustness gives coef={age_coef_ols:.3f} "
        f"(p={fmt_p(age_p_ols)}). SmartAdditive ranks age #{age_smart.get('rank', 0)} "
        f"with {100*age_imp_smart:.1f}% importance and {shape_note}. HingeEBM ranks age "
        f"#{age_hinge.get('rank', 0)} with {100*age_imp_hinge:.1f}% importance and "
        f"{age_hinge.get('direction', 'zero')} direction. Top SmartAdditive features: {smart_top_str}. "
        f"Top HingeEBM features: {hinge_top_str}. "
        f"Sign consistency across models is {'high' if sign_consistent else 'mixed'}. "
        f"Notable controlled confounders: {', '.join(sig_controls) if sig_controls else 'none at p<0.05'}. "
        f"Overall, this implies a {'robust' if score >= 75 else 'moderate' if score >= 40 else 'weak'} "
        f"age-related development in majority preference after accounting for other factors."
    )

    result = {"response": int(score), "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(result, ensure_ascii=True))

    print("\nStep 4: Conclusion JSON")
    print(json.dumps(result, indent=2))
    print("\nWrote conclusion.txt")


if __name__ == "__main__":
    main()
