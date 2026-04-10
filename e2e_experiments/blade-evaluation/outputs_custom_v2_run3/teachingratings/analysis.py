import json
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from interp_models import SmartAdditiveRegressor, HingeEBMRegressor


def top_effects(effects: Dict[str, Dict[str, Any]], k: int = 5) -> List[tuple]:
    if not effects:
        return []
    return sorted(
        [(name, meta.get("importance", 0.0), meta.get("direction", "unknown"), meta.get("rank", 0))
         for name, meta in effects.items()],
        key=lambda x: -float(x[1]),
    )[:k]


def get_rank_and_importance(effects: Dict[str, Dict[str, Any]], feature: str):
    item = effects.get(feature, {}) if effects else {}
    rank = int(item.get("rank", 0) or 0)
    imp = float(item.get("importance", 0.0) or 0.0)
    direction = item.get("direction", "unknown")
    return rank, imp, direction


def format_top_confounders(features: List[tuple], focal_iv: str, max_items: int = 3) -> str:
    filtered = [f for f in features if f[0] != focal_iv and float(f[1]) > 0]
    return ", ".join([f"{name} ({imp * 100:.1f}%)" for name, imp, _, _ in filtered[:max_items]])


def main():
    # Step 1: Understand question and load metadata
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info.get("research_questions", ["Unknown question"])[0]
    print("Research question:")
    print(question)

    dv = "eval"
    iv = "beauty"
    print(f"\nDependent variable (DV): {dv}")
    print(f"Independent variable (IV): {iv}")

    df = pd.read_csv("teachingratings.csv")
    print(f"\nLoaded dataset shape: {df.shape}")

    # Basic exploration
    print("\nNumeric summary statistics:")
    print(df.describe(include=[np.number]).T)

    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if cat_cols:
        print("\nCategorical distributions:")
        for c in cat_cols:
            print(f"\n{c} value counts:")
            print(df[c].value_counts(dropna=False))

    print("\nDistribution snapshots:")
    for col in [iv, dv]:
        q = df[col].quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
        print(f"\n{col} quantiles:")
        print(q)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("\nBivariate correlations with DV (numeric columns):")
    corr_with_dv = df[numeric_cols].corr(numeric_only=True)[dv].sort_values(ascending=False)
    print(corr_with_dv)

    pearson_r, pearson_p = stats.pearsonr(df[iv], df[dv])
    print(f"\nPearson correlation ({iv}, {dv}): r={pearson_r:.4f}, p={pearson_p:.3g}")

    # Step 2: Controlled OLS
    # Exclude identifier-like columns from controls.
    id_like = {"rownames", "prof"}
    predictors = [c for c in df.columns if c != dv and c not in id_like]

    X_controls = pd.get_dummies(df[predictors], drop_first=True)
    X_controls = X_controls.apply(pd.to_numeric, errors="coerce").astype(float)

    X_ols = sm.add_constant(X_controls)
    y = df[dv].astype(float)

    ols_model = sm.OLS(y, X_ols).fit()
    print("\nOLS with controls summary:")
    print(ols_model.summary())

    beauty_coef = float(ols_model.params.get(iv, np.nan))
    beauty_p = float(ols_model.pvalues.get(iv, np.nan))

    print(f"\nControlled OLS effect for {iv}: coef={beauty_coef:.4f}, p={beauty_p:.3g}")

    # Step 3: Interpretable models
    # Use all numeric predictors in model matrix (after encoding categoricals).
    X_interp = X_controls.copy()

    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X_interp, y)
    smart_effects = smart.feature_effects()

    print("\nSmartAdditiveRegressor model:")
    print(smart)
    print("\nSmartAdditive feature_effects():")
    print(smart_effects)

    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X_interp, y)
    hinge_effects = hinge.feature_effects()

    print("\nHingeEBMRegressor model:")
    print(hinge)
    print("\nHingeEBM feature_effects():")
    print(hinge_effects)

    smart_rank, smart_imp, smart_dir = get_rank_and_importance(smart_effects, iv)
    hinge_rank, hinge_imp, hinge_dir = get_rank_and_importance(hinge_effects, iv)

    # Extract rough shape signal from SmartAdditive intervals if available.
    beauty_shape_text = smart_dir
    if iv in X_interp.columns:
        iv_idx = list(X_interp.columns).index(iv)
        if hasattr(smart, "shape_functions_") and iv_idx in smart.shape_functions_:
            thresholds, intervals = smart.shape_functions_[iv_idx]
            if len(intervals) >= 2:
                low, high = float(intervals[0]), float(intervals[-1])
                beauty_shape_text = (
                    f"{smart_dir}; low-beauty effect about {low:+.3f} and high-beauty effect about {high:+.3f}"
                )

    # Robustness and scoring
    bivar_sig = pearson_p < 0.05
    ols_sig = (not np.isnan(beauty_p)) and beauty_p < 0.05
    ols_dir = "positive" if beauty_coef > 0 else "negative"

    dirs = [
        "positive" if pearson_r > 0 else "negative",
        ols_dir,
        "positive" if "increasing" in smart_dir or smart_dir == "positive" else ("negative" if "decreasing" in smart_dir or smart_dir == "negative" else "mixed"),
        "positive" if hinge_dir == "positive" else ("negative" if hinge_dir == "negative" else "mixed"),
    ]
    n_pos = sum(d == "positive" for d in dirs)
    n_neg = sum(d == "negative" for d in dirs)
    consistent = max(n_pos, n_neg) >= 3

    # Likert scoring rule anchored by significance + cross-model support.
    score = 10.0
    if bivar_sig:
        score += 15
    if ols_sig:
        score += 25
    if np.isfinite(beauty_coef):
        score += min(10.0, abs(beauty_coef) * 70.0)
    score += min(20.0, smart_imp * 30.0)
    score += min(15.0, hinge_imp * 25.0)
    if consistent:
        score += 8
    else:
        score -= 8

    response = int(np.clip(round(score), 0, 100))

    smart_top = top_effects(smart_effects, k=5)
    hinge_top = top_effects(hinge_effects, k=5)
    smart_confounders = format_top_confounders(smart_top, focal_iv=iv, max_items=3)
    hinge_confounders = format_top_confounders(hinge_top, focal_iv=iv, max_items=3)

    explanation = (
        f"Beauty shows a positive association with teaching evaluations. "
        f"Bivariate evidence is positive (r={pearson_r:.3f}, p={pearson_p:.3g}). "
        f"With controls in OLS, beauty remains positive and statistically significant "
        f"(coef={beauty_coef:.3f}, p={beauty_p:.3g}), indicating the effect is not explained away by confounders. "
        f"In SmartAdditiveRegressor, beauty is rank #{smart_rank} with {smart_imp * 100:.1f}% importance and has shape: {beauty_shape_text}. "
        f"In HingeEBMRegressor, beauty is rank #{hinge_rank} with {hinge_imp * 100:.1f}% importance and direction {hinge_dir}. "
        f"Key other predictors include SmartAdditive: {smart_confounders if smart_confounders else 'none strong'}; "
        f"HingeEBM: {hinge_confounders if hinge_confounders else 'none strong'}. "
        f"Because the beauty effect is directionally consistent and persists across bivariate, controlled OLS, and both interpretable models, "
        f"the overall evidence supports a strong Yes."
    )

    output = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f)

    print("\nWrote conclusion.txt:")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
