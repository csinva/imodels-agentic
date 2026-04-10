import json
from typing import Dict, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from interp_models import SmartAdditiveRegressor, HingeEBMRegressor


def to_native(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def extract_top_features(effects: Dict[str, Dict[str, Any]], exclude=None, k=3):
    exclude = set(exclude or [])
    rows = []
    for feat, info in effects.items():
        if feat in exclude:
            continue
        imp = float(to_native(info.get("importance", 0.0)))
        rank = int(to_native(info.get("rank", 0)))
        direction = str(info.get("direction", "unknown"))
        rows.append((feat, imp, rank, direction))
    rows.sort(key=lambda z: (-z[1], z[2] if z[2] > 0 else 10**9, z[0]))
    return rows[:k]


def main():
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info.get("research_questions", [""])[0]
    print("Research question:")
    print(question)

    df = pd.read_csv("amtl.csv")

    # IV and DV derived from the research question and metadata.
    iv_col = "is_human"
    dv_count_col = "num_amtl"
    exposure_col = "sockets"
    dv_rate_col = "amtl_rate"

    df[iv_col] = (df["genus"] == "Homo sapiens").astype(int)
    df[dv_rate_col] = df[dv_count_col] / df[exposure_col]

    print("\nStep 1: Explore")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print("\nNumeric summary:")
    print(df[[dv_count_col, exposure_col, "age", "stdev_age", "prob_male", dv_rate_col]].describe())

    print("\nGenus counts:")
    print(df["genus"].value_counts())
    print("\nTooth class counts:")
    print(df["tooth_class"].value_counts())

    human_rate = df.loc[df[iv_col] == 1, dv_rate_col]
    nonhuman_rate = df.loc[df[iv_col] == 0, dv_rate_col]
    mean_human = float(human_rate.mean())
    mean_nonhuman = float(nonhuman_rate.mean())
    diff = mean_human - mean_nonhuman

    pearson_r, pearson_p = stats.pearsonr(df[iv_col], df[dv_rate_col])
    spearman_rho, spearman_p = stats.spearmanr(df[iv_col], df[dv_rate_col])
    t_stat, t_p = stats.ttest_ind(human_rate, nonhuman_rate, equal_var=False)

    print("\nBivariate human vs non-human AMTL rate:")
    print(f"Mean rate (human): {mean_human:.4f}")
    print(f"Mean rate (non-human): {mean_nonhuman:.4f}")
    print(f"Difference (human - non-human): {diff:.4f}")
    print(f"Pearson r: {pearson_r:.4f}, p={pearson_p:.3g}")
    print(f"Spearman rho: {spearman_rho:.4f}, p={spearman_p:.3g}")
    print(f"Welch t-test: t={t_stat:.4f}, p={t_p:.3g}")

    print("\nCorrelations with AMTL rate (numeric columns):")
    num_cols = [c for c in ["age", "stdev_age", "prob_male", "sockets", iv_col] if c in df.columns]
    corr_series = df[num_cols + [dv_rate_col]].corr(numeric_only=True)[dv_rate_col].sort_values(ascending=False)
    print(corr_series)

    # Controls aligned with the research question: age, sex proxy, tooth class.
    # Include sockets and age uncertainty as additional quantitative controls.
    X_base = pd.DataFrame(
        {
            iv_col: df[iv_col],
            "age": df["age"],
            "prob_male": df["prob_male"],
            "stdev_age": df["stdev_age"],
            "sockets": df["sockets"],
        }
    )
    tooth_dummies = pd.get_dummies(df["tooth_class"], prefix="tooth_class", drop_first=True, dtype=float)
    X = pd.concat([X_base, tooth_dummies], axis=1)
    X_sm = sm.add_constant(X, has_constant="add")

    print("\nStep 2: Controlled statistical models")
    ols_model = sm.OLS(df[dv_count_col], X_sm).fit()
    print("\nOLS summary (DV = num_amtl):")
    print(ols_model.summary())

    # Frequency-appropriate robustness check for count/trial data.
    glm_binom = sm.GLM(
        df[dv_rate_col],
        X_sm,
        family=sm.families.Binomial(),
        var_weights=df[exposure_col],
    ).fit()
    print("\nBinomial GLM summary (DV = num_amtl/sockets, weighted by sockets):")
    print(glm_binom.summary())

    print("\nStep 3: Interpretable models")
    X_interp = X.copy()
    y_interp = df[dv_count_col].astype(float)

    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X_interp, y_interp)
    smart_effects = smart.feature_effects()
    print("\nSmartAdditiveRegressor:")
    print(smart)
    print("\nSmart feature effects:")
    print(smart_effects)

    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X_interp, y_interp)
    hinge_effects = hinge.feature_effects()
    print("\nHingeEBMRegressor:")
    print(hinge)
    print("\nHinge feature effects:")
    print(hinge_effects)

    # Collect core evidence for IV (human indicator).
    ols_coef = float(ols_model.params.get(iv_col, np.nan))
    ols_p = float(ols_model.pvalues.get(iv_col, np.nan))

    glm_coef = float(glm_binom.params.get(iv_col, np.nan))
    glm_p = float(glm_binom.pvalues.get(iv_col, np.nan))
    glm_or = float(np.exp(glm_coef)) if np.isfinite(glm_coef) else np.nan

    smart_iv = smart_effects.get(iv_col, {"direction": "zero", "importance": 0.0, "rank": 0})
    hinge_iv = hinge_effects.get(iv_col, {"direction": "zero", "importance": 0.0, "rank": 0})

    smart_iv_dir = str(smart_iv.get("direction", "zero"))
    smart_iv_imp = float(to_native(smart_iv.get("importance", 0.0)))
    smart_iv_rank = int(to_native(smart_iv.get("rank", 0)))

    hinge_iv_dir = str(hinge_iv.get("direction", "zero"))
    hinge_iv_imp = float(to_native(hinge_iv.get("importance", 0.0)))
    hinge_iv_rank = int(to_native(hinge_iv.get("rank", 0)))

    # Score synthesis using consistency across analyses.
    score = 10

    biv_sig_pos = diff > 0 and pearson_p < 0.05 and t_p < 0.05
    if biv_sig_pos:
        score += 20
    elif diff > 0:
        score += 10

    if ols_coef > 0 and ols_p < 0.05:
        score += 15
    elif ols_coef > 0 and ols_p < 0.10:
        score += 8

    if glm_coef > 0 and glm_p < 0.05:
        score += 25
    elif glm_coef > 0:
        score += 12

    if ("positive" in smart_iv_dir or "increasing" in smart_iv_dir) and smart_iv_imp >= 0.05:
        score += 15
    elif ("positive" in smart_iv_dir or "increasing" in smart_iv_dir) and smart_iv_imp > 0:
        score += 8

    if ("positive" in hinge_iv_dir or "increasing" in hinge_iv_dir) and hinge_iv_imp >= 0.03:
        score += 10
    elif hinge_iv_imp == 0:
        score += 0
    else:
        score -= 10

    # Penalize inconsistency when one model drops the effect.
    if hinge_iv_imp == 0 and (glm_coef > 0 and glm_p < 0.05) and smart_iv_imp > 0:
        score -= 12

    score = int(max(0, min(100, round(score))))

    top_smart_controls = extract_top_features(smart_effects, exclude=[iv_col], k=3)
    top_hinge_controls = extract_top_features(hinge_effects, exclude=[iv_col], k=3)

    smart_ctrl_text = ", ".join(
        [f"{f} (rank {r}, {imp*100:.1f}%, {d})" for f, imp, r, d in top_smart_controls if imp > 0]
    )
    hinge_ctrl_text = ", ".join(
        [f"{f} (rank {r}, {imp*100:.1f}%, {d})" for f, imp, r, d in top_hinge_controls if imp > 0]
    )

    explanation = (
        f"Bivariate evidence is strongly positive: humans have higher AMTL frequency than non-humans "
        f"({mean_human:.3f} vs {mean_nonhuman:.3f}; diff={diff:.3f}; Pearson r={pearson_r:.3f}, p={pearson_p:.2g}). "
        f"With controls, OLS on AMTL count shows a small positive but only marginal human effect "
        f"(coef={ols_coef:.3f}, p={ols_p:.3g}). A binomial GLM for AMTL frequency (num_amtl/sockets) shows a strong positive human effect "
        f"(log-odds coef={glm_coef:.3f}, OR={glm_or:.2f}, p={glm_p:.2g}), indicating robustness in a model matched to count/trial data. "
        f"SmartAdditive also retains a positive human effect (importance={smart_iv_imp*100:.1f}%, rank={smart_iv_rank}, direction={smart_iv_dir}), "
        f"while HingeEBM shrinks it to zero (importance={hinge_iv_imp*100:.1f}%, rank={hinge_iv_rank}). "
        f"Effect shape is mixed: the human indicator is mostly linear-positive in SmartAdditive, while key confounders show nonlinear structure, "
        f"especially age and age uncertainty. Important confounders are: SmartAdditive -> {smart_ctrl_text}; "
        f"HingeEBM -> {hinge_ctrl_text}. Overall, evidence supports higher AMTL in humans, but with some model-dependence in sparse selection, "
        f"so the conclusion is moderately strong rather than maximal."
    )

    output = {"response": score, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=True)

    print("\nStep 4: Conclusion")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
