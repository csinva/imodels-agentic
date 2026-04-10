import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score

from interp_models import HingeEBMRegressor, SmartAdditiveRegressor


def choose_iv(question: str, columns: list[str]) -> str:
    q = question.lower()
    if "hour" in q and "hours" in columns:
        return "hours"
    # fallback: first non-DV numeric candidate by convention
    for c in ["hours", "livebait", "camper", "persons", "child"]:
        if c in columns:
            return c
    return columns[0]


def safe_effect(effects: dict, feature: str) -> dict:
    return effects.get(feature, {"direction": "zero", "importance": 0.0, "rank": 0})


def score_from_evidence(
    corr_iv_dv: float,
    ols_coef: float,
    ols_p: float,
    smart_eff: dict,
    hinge_eff: dict,
) -> int:
    # Anchor on controlled OLS first (per rubric).
    if ols_p < 0.01:
        score = 82.0
    elif ols_p < 0.05:
        score = 70.0
    elif ols_p < 0.10:
        score = 55.0
    else:
        score = 25.0

    # Penalize if adjusted direction is opposite to hypothesized positive effect.
    score += 5 if ols_coef > 0 else -15

    # Bivariate support (small adjustment).
    if np.isfinite(corr_iv_dv):
        if corr_iv_dv > 0:
            score += min(8, abs(corr_iv_dv) * 20)
        else:
            score -= min(10, abs(corr_iv_dv) * 25)

    # SmartAdditive evidence (shape + relative importance; moderate adjustment)
    smart_dir = smart_eff.get("direction", "zero")
    smart_imp = float(smart_eff.get("importance", 0.0) or 0.0)
    if smart_dir in {"positive", "nonlinear (increasing trend)"}:
        score += 4 + min(10, smart_imp * 100 * 0.15)
    elif smart_dir in {"negative", "nonlinear (decreasing trend)"}:
        score -= 5 + min(10, smart_imp * 100 * 0.15)
    elif "nonlinear" in smart_dir:
        score += 2
    else:
        score -= 4

    # Hinge evidence (moderate adjustment)
    hinge_dir = hinge_eff.get("direction", "zero")
    hinge_imp = float(hinge_eff.get("importance", 0.0) or 0.0)
    if hinge_dir == "positive":
        score += 4 + min(8, hinge_imp * 100 * 0.12)
    elif hinge_dir == "negative":
        score -= 5 + min(8, hinge_imp * 100 * 0.12)
    else:
        score -= 3

    # Robustness bonus/penalty for agreement across models.
    directions = []
    if ols_coef > 0:
        directions.append("pos")
    elif ols_coef < 0:
        directions.append("neg")

    if smart_dir in {"positive", "nonlinear (increasing trend)"}:
        directions.append("pos")
    elif smart_dir in {"negative", "nonlinear (decreasing trend)"}:
        directions.append("neg")

    if hinge_dir == "positive":
        directions.append("pos")
    elif hinge_dir == "negative":
        directions.append("neg")

    if len(directions) >= 2 and len(set(directions)) == 1:
        score += 8
    elif "pos" in directions and "neg" in directions:
        score -= 10

    # If adjusted OLS is not conventionally significant, cap extreme certainty.
    if ols_p >= 0.05:
        score = min(score, 72.0)

    return int(np.clip(round(score), 0, 100))


def top_features(effects: dict, exclude: set[str], k: int = 3) -> list[tuple[str, dict]]:
    ranked = [(f, v) for f, v in effects.items() if f not in exclude and (v.get("importance", 0) or 0) > 0]
    ranked.sort(key=lambda x: (-x[1].get("importance", 0), x[0]))
    return ranked[:k]


def summarize_ranked_features(items: list[tuple[str, dict]]) -> str:
    if not items:
        return "none"
    out = []
    for name, meta in items:
        rank = meta.get("rank", 0)
        imp = float(meta.get("importance", 0) or 0) * 100
        out.append(f"{name} (rank {rank}, {imp:.1f}%)")
    return ", ".join(out)


def main() -> None:
    # Step 1: understand question + explore
    info = json.loads(Path("info.json").read_text())
    question = info["research_questions"][0]

    df = pd.read_csv("fish.csv")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    dv = "fish_caught" if "fish_caught" in df.columns else numeric_cols[0]
    iv = choose_iv(question, [c for c in df.columns if c != dv])

    feature_cols = [c for c in numeric_cols if c != dv]

    print("Research question:", question)
    print(f"Dependent variable (DV): {dv}")
    print(f"Independent variable (IV): {iv}")
    print(f"Control/features used in models: {feature_cols}")
    print()

    print("=== Summary statistics ===")
    print(df[numeric_cols].describe().T)
    print()

    print("=== Distribution checks (quantiles) ===")
    for c in numeric_cols:
        q = df[c].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict()
        print(f"{c}: {q}")
    print()

    print("=== Bivariate correlations ===")
    corr = df[numeric_cols].corr(numeric_only=True)
    print(corr)
    print()

    corr_iv_dv = float(corr.loc[iv, dv])
    print(f"Correlation({iv}, {dv}) = {corr_iv_dv:.4f}")

    # Per-hour descriptive target for context in the question wording
    fish_per_hour = np.where(df["hours"] > 0, df[dv] / df["hours"], np.nan)
    mean_fph = float(np.nanmean(fish_per_hour))
    median_fph = float(np.nanmedian(fish_per_hour))
    print(f"Average fish per hour: {mean_fph:.4f}")
    print(f"Median fish per hour: {median_fph:.4f}")
    print()

    # Step 2: controlled OLS
    print("=== OLS with controls ===")
    X = sm.add_constant(df[feature_cols])
    y = df[dv]
    ols_model = sm.OLS(y, X).fit()
    print(ols_model.summary())
    print()

    # simple bivariate OLS as a check
    print("=== Bivariate OLS (DV ~ IV only) ===")
    X_bi = sm.add_constant(df[[iv]])
    bi_model = sm.OLS(y, X_bi).fit()
    print(bi_model.summary())
    print()

    ols_coef = float(ols_model.params.get(iv, np.nan))
    ols_p = float(ols_model.pvalues.get(iv, np.nan))
    ols_ci_lo, ols_ci_hi = ols_model.conf_int().loc[iv].tolist()

    # Step 3: custom interpretable models
    print("=== SmartAdditiveRegressor ===")
    X_all = df[feature_cols]
    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X_all, y)
    print(smart)
    smart_effects = smart.feature_effects()
    print("SmartAdditive feature effects:")
    print(smart_effects)
    print()

    print("=== HingeEBMRegressor ===")
    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X_all, y)
    print(hinge)
    hinge_effects = hinge.feature_effects()
    print("HingeEBM feature effects:")
    print(hinge_effects)
    print()

    smart_iv = safe_effect(smart_effects, iv)
    hinge_iv = safe_effect(hinge_effects, iv)

    # Fit quality context
    r2_ols = float(ols_model.rsquared)
    r2_smart = float(r2_score(y, smart.predict(X_all)))
    r2_hinge = float(r2_score(y, hinge.predict(X_all)))

    # Rank/importance context for confounders
    top_smart_controls = top_features(smart_effects, exclude={iv}, k=3)
    top_hinge_controls = top_features(hinge_effects, exclude={iv}, k=3)
    top_smart_txt = summarize_ranked_features(top_smart_controls)
    top_hinge_txt = summarize_ranked_features(top_hinge_controls)

    response = score_from_evidence(
        corr_iv_dv=corr_iv_dv,
        ols_coef=ols_coef,
        ols_p=ols_p,
        smart_eff=smart_iv,
        hinge_eff=hinge_iv,
    )

    expl = (
        f"Using DV={dv} and IV={iv}, the bivariate relationship is "
        f"corr={corr_iv_dv:.3f}. In the controlled OLS (controls: "
        f"{', '.join([c for c in feature_cols if c != iv])}), the IV coefficient is "
        f"{ols_coef:.3f} with p={ols_p:.4f} and 95% CI [{ols_ci_lo:.3f}, {ols_ci_hi:.3f}], "
        f"so the direction is {'positive' if ols_coef > 0 else 'negative'} after adjustment. "
        f"SmartAdditive ranks {iv} at #{smart_iv.get('rank', 0)} with "
        f"importance={float(smart_iv.get('importance', 0.0))*100:.1f}% and direction "
        f"'{smart_iv.get('direction', 'zero')}', indicating "
        f"{'a nonlinear threshold/increasing pattern' if 'nonlinear' in str(smart_iv.get('direction', '')) else 'a mostly linear effect'}. "
        f"HingeEBM ranks {iv} at #{hinge_iv.get('rank', 0)} with "
        f"importance={float(hinge_iv.get('importance', 0.0))*100:.1f}% and direction "
        f"'{hinge_iv.get('direction', 'zero')}'. "
        f"Model fit is R^2={r2_ols:.3f} (OLS), {r2_smart:.3f} (SmartAdditive), and {r2_hinge:.3f} (HingeEBM). "
        f"Top other predictors are SmartAdditive: "
        f"{top_smart_txt}; "
        f"HingeEBM: {top_hinge_txt}. "
        f"Average catch rate is {mean_fph:.2f} fish/hour (median {median_fph:.2f}), and the score reflects "
        f"the IV effect strength and consistency across bivariate, controlled, and interpretable models."
    )

    payload = {"response": int(response), "explanation": expl}
    Path("conclusion.txt").write_text(json.dumps(payload))

    print("=== Final conclusion payload ===")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
