import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

from agentic_imodels import (
    HingeEBMRegressor,
    SmartAdditiveRegressor,
    WinsorizedSparseOLSRegressor,
)


@dataclass
class FemaleEffect:
    sign: str
    magnitude: float
    selected: bool
    detail: str


def safe_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def get_hingebm_effective_coef(model: HingeEBMRegressor, feature_idx: int) -> float:
    # Reconstruct the same effective linearized coefficient used in __str__.
    n_sel = len(model.selected_)
    coefs = model.lasso_.coef_
    effective = {}

    for i in range(n_sel):
        j_orig = int(model.selected_[i])
        effective[j_orig] = float(coefs[i])

    for idx, (feat_idx, knot, direction) in enumerate(model.hinge_info_):
        j_orig = int(model.selected_[feat_idx])
        c = float(coefs[n_sel + idx])
        if abs(c) < 1e-12:
            continue
        if direction == "pos":
            effective[j_orig] = effective.get(j_orig, 0.0) + c
        else:
            effective[j_orig] = effective.get(j_orig, 0.0) - c

    return float(effective.get(feature_idx, 0.0))


def summarize_female_from_smart(model: SmartAdditiveRegressor, female_idx: int) -> FemaleEffect:
    importance = float(model.feature_importances_[female_idx]) if female_idx < len(model.feature_importances_) else 0.0

    if female_idx not in model.shape_functions_:
        return FemaleEffect(
            sign="zero",
            magnitude=0.0,
            selected=False,
            detail="female excluded from SmartAdditive shape functions",
        )

    slope, _, r2 = model.linear_approx_.get(female_idx, (0.0, 0.0, 0.0))
    if r2 > 0.70:
        sign = "positive" if slope > 0 else ("negative" if slope < 0 else "zero")
        return FemaleEffect(
            sign=sign,
            magnitude=float(abs(slope)),
            selected=importance > 1e-8,
            detail=f"SmartAdditive linear approx slope={slope:.6f}, r2={r2:.3f}, importance={importance:.6f}",
        )

    thresholds, intervals = model.shape_functions_[female_idx]
    if len(intervals) >= 2:
        delta = float(intervals[-1] - intervals[0])
    else:
        delta = 0.0
    sign = "positive" if delta > 0 else ("negative" if delta < 0 else "zero")
    return FemaleEffect(
        sign=sign,
        magnitude=abs(delta),
        selected=importance > 1e-8,
        detail=f"SmartAdditive piecewise delta(1-0)~{delta:.6f}, thresholds={thresholds}, importance={importance:.6f}",
    )


def summarize_female_from_hingebm(model: HingeEBMRegressor, female_idx: int) -> FemaleEffect:
    coef = get_hingebm_effective_coef(model, female_idx)
    sign = "positive" if coef > 0 else ("negative" if coef < 0 else "zero")
    selected = abs(coef) > 1e-8
    return FemaleEffect(
        sign=sign,
        magnitude=abs(coef),
        selected=selected,
        detail=f"HingeEBM effective linearized female coef={coef:.6f}",
    )


def summarize_female_from_winsor(model: WinsorizedSparseOLSRegressor, female_idx: int) -> FemaleEffect:
    support = list(model.support_)
    if female_idx in support:
        pos = support.index(female_idx)
        coef = float(model.ols_coef_[pos])
        sign = "positive" if coef > 0 else ("negative" if coef < 0 else "zero")
        return FemaleEffect(
            sign=sign,
            magnitude=abs(coef),
            selected=True,
            detail=f"WinsorizedSparseOLS selected female coef={coef:.6f}",
        )
    return FemaleEffect(
        sign="zero",
        magnitude=0.0,
        selected=False,
        detail="WinsorizedSparseOLS excluded female (lasso zeroed)",
    )


def calibrate_score(
    pval: float,
    ci_low: float,
    ci_high: float,
    chi2_p: float,
    z_p: float,
    smart: FemaleEffect,
    hingebm: FemaleEffect,
    winsor: FemaleEffect,
) -> int:
    score = 45

    # Classical significance anchor.
    if pval < 0.01:
        score += 25
    elif pval < 0.05:
        score += 18
    elif pval < 0.10:
        score += 8
    else:
        score -= 22

    # CI exclusion of zero.
    if ci_low > 0 or ci_high < 0:
        score += 10
    else:
        score -= 10

    # Bivariate anchor: if raw association is absent, dampen confidence.
    if chi2_p >= 0.10 and z_p >= 0.10:
        score -= 12
    elif chi2_p < 0.05 or z_p < 0.05:
        score += 8

    # Interpretable model corroboration / null evidence.
    effects = [smart, hingebm, winsor]
    nonzero = [e for e in effects if e.selected and e.sign != "zero"]
    zeros = [e for e in effects if not e.selected or e.sign == "zero"]

    if len(nonzero) >= 2:
        score += 15
    if len(zeros) >= 2:
        score -= 20
    elif len(zeros) == 1:
        score -= 8

    # Sign consistency across non-zero effects.
    signs = {e.sign for e in nonzero}
    if len(signs) == 1 and len(nonzero) >= 2:
        score += 5
    elif len(signs) > 1:
        score -= 8

    return int(max(0, min(100, round(score))))


def main() -> None:
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info["research_questions"][0]
    print(f"Research question: {research_question}")

    df_raw = pd.read_csv("mortgage.csv")
    df = safe_to_numeric(df_raw)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Outcome is mortgage approval (accept=1), as asked in the question.
    if "accept" in df.columns:
        outcome = "accept"
    else:
        # Fallback if only deny is present.
        df["accept"] = 1 - df["deny"]
        outcome = "accept"

    iv = "female"
    all_features = [c for c in df.columns if c != outcome]

    # Controls for formal test: applicant characteristics, excluding complementary target column.
    controls = [c for c in all_features if c not in {"deny", "accept", iv}]
    model_features = [iv] + controls

    required_cols = [outcome] + model_features
    dfm = df[required_cols].dropna().copy()

    print("\n=== Data overview ===")
    print(f"Rows (raw): {len(df_raw)}, rows used after NA filtering: {len(dfm)}")
    print(f"Columns used ({len(required_cols)}): {required_cols}")
    print("\nSummary stats:")
    print(dfm.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]])

    print("\n=== Distributions ===")
    for col in required_cols:
        nunique = dfm[col].nunique()
        if nunique <= 8:
            print(f"\nValue counts for {col}:")
            print(dfm[col].value_counts(dropna=False).sort_index())
        else:
            q = dfm[col].quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
            print(f"\nQuantiles for {col}:")
            print(q)

    print("\n=== Correlations with approval ===")
    corr_to_outcome = dfm.corr(numeric_only=True)[outcome].sort_values(ascending=False)
    print(corr_to_outcome)

    # Bivariate gender-approval association.
    print("\n=== Bivariate tests (female vs approval) ===")
    ctab = pd.crosstab(dfm[iv], dfm[outcome])
    print("Contingency table (rows=female, cols=accept):")
    print(ctab)

    chi2, chi2_p, _, _ = stats.chi2_contingency(ctab)
    print(f"Chi-square test p-value: {chi2_p:.6g}")

    female_1 = dfm[dfm[iv] == 1][outcome]
    female_0 = dfm[dfm[iv] == 0][outcome]
    z_stat, z_p = proportions_ztest(
        count=np.array([female_1.sum(), female_0.sum()]),
        nobs=np.array([len(female_1), len(female_0)]),
    )
    print(f"Two-proportion z-test p-value: {z_p:.6g}, z={z_stat:.4f}")
    print(f"Approval rate female=1: {female_1.mean():.4f}")
    print(f"Approval rate female=0: {female_0.mean():.4f}")

    # Classical controlled model.
    print("\n=== Controlled logistic regression (statsmodels Logit) ===")
    X = sm.add_constant(dfm[model_features], has_constant="add")
    y = dfm[outcome]
    logit = sm.Logit(y, X).fit(disp=False)
    print(logit.summary())

    female_coef = float(logit.params[iv])
    female_pval = float(logit.pvalues[iv])
    ci_low, ci_high = logit.conf_int().loc[iv].tolist()
    odds_ratio = float(np.exp(female_coef))
    or_low, or_high = float(np.exp(ci_low)), float(np.exp(ci_high))
    print(
        f"\nFemale (controlled) log-odds coef={female_coef:.6f}, p={female_pval:.6g}, "
        f"95% CI=({ci_low:.6f}, {ci_high:.6f}), OR={odds_ratio:.4f}, OR 95% CI=({or_low:.4f}, {or_high:.4f})"
    )

    # Sensitivity: include denied_PMI if available.
    if "denied_PMI" in dfm.columns:
        sens_features = [iv] + [c for c in controls if c != "denied_PMI"] + ["denied_PMI"]
        X_sens = sm.add_constant(dfm[sens_features], has_constant="add")
        try:
            logit_sens = sm.Logit(y, X_sens).fit(disp=False)
            sc = float(logit_sens.params[iv])
            sp = float(logit_sens.pvalues[iv])
            print(
                "Sensitivity model with denied_PMI included: "
                f"female coef={sc:.6f}, p={sp:.6g}"
            )
        except Exception as e:
            print(f"Sensitivity model with denied_PMI failed: {e}")

    print("\n=== Interpretable models (agentic_imodels) ===")
    print("Feature index mapping used by model prints:")
    for i, col in enumerate(model_features):
        print(f"  x{i} -> {col}")

    X_model = dfm[model_features].to_numpy()
    y_model = y.to_numpy()
    female_idx = model_features.index(iv)

    smart = SmartAdditiveRegressor().fit(X_model, y_model)
    print("\n--- SmartAdditiveRegressor (honest) ---")
    print(smart)

    hingebm = HingeEBMRegressor().fit(X_model, y_model)
    print("\n--- HingeEBMRegressor (high-rank, decoupled) ---")
    print(hingebm)

    winsor = WinsorizedSparseOLSRegressor().fit(X_model, y_model)
    print("\n--- WinsorizedSparseOLSRegressor (honest sparse linear / Lasso-selected) ---")
    print(winsor)

    smart_eff = summarize_female_from_smart(smart, female_idx)
    hingebm_eff = summarize_female_from_hingebm(hingebm, female_idx)
    winsor_eff = summarize_female_from_winsor(winsor, female_idx)

    print("\n=== Female effect summary across interpretable models ===")
    print(smart_eff.detail)
    print(hingebm_eff.detail)
    print(winsor_eff.detail)

    response = calibrate_score(
        pval=female_pval,
        ci_low=ci_low,
        ci_high=ci_high,
        chi2_p=chi2_p,
        z_p=z_p,
        smart=smart_eff,
        hingebm=hingebm_eff,
        winsor=winsor_eff,
    )

    explanation = (
        f"Question: {research_question} "
        f"Bivariate tests show gender-approval association with chi-square p={chi2_p:.4g} "
        f"and two-proportion z-test p={z_p:.4g}. "
        f"In controlled logistic regression (approval ~ female + controls), female coef={female_coef:.4f} "
        f"(OR={odds_ratio:.3f}, 95% CI OR {or_low:.3f} to {or_high:.3f}, p={female_pval:.4g}). "
        f"Interpretable models: SmartAdditive says female is {smart_eff.sign} (selected={smart_eff.selected}); "
        f"HingeEBM says {hingebm_eff.sign} (selected={hingebm_eff.selected}); "
        f"WinsorizedSparseOLS says {winsor_eff.sign} (selected={winsor_eff.selected}). "
        "Score reflects combined evidence from significance, controlled effect size, cross-model direction consistency, "
        "and sparse-model zeroing/null evidence."
    )

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump({"response": response, "explanation": explanation}, f)

    print("\nWrote conclusion.txt")


if __name__ == "__main__":
    main()
