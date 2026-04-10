import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from interp_models import SmartAdditiveRegressor, HingeEBMRegressor

warnings.filterwarnings("ignore")


def p_to_strength(p: float) -> int:
    if p < 0.001:
        return 4
    if p < 0.01:
        return 3
    if p < 0.05:
        return 2
    if p < 0.10:
        return 1
    return 0


def sign_from_direction(direction: str) -> int:
    if direction is None:
        return 0
    d = direction.lower()
    if "positive" in d or "increasing" in d:
        return 1
    if "negative" in d or "decreasing" in d:
        return -1
    return 0


def top_effects(effects: dict, exclude: str, k: int = 3):
    items = []
    for name, vals in effects.items():
        if name == exclude:
            continue
        imp = float(vals.get("importance", 0.0) or 0.0)
        if imp <= 0:
            continue
        items.append((name, imp, vals.get("direction", "unknown"), vals.get("rank", 0)))
    items.sort(key=lambda x: (-x[1], x[0]))
    return items[:k]


def safe_fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "nan"


def main():
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info["research_questions"][0]
    print("Research question:", question)

    df = pd.read_csv("soccer.csv")
    print(f"Loaded soccer.csv with shape={df.shape}")

    # Core variables
    df["skin_tone"] = df[["rater1", "rater2"]].mean(axis=1, skipna=True)
    df["birthday_parsed"] = pd.to_datetime(df["birthday"], format="%d.%m.%Y", errors="coerce")
    df["age"] = 2013 - df["birthday_parsed"].dt.year
    df["red_any"] = (df["redCards"] > 0).astype(int)

    dv = "redCards"
    iv = "skin_tone"

    # Step 1: Explore
    explore_cols = [
        dv,
        "red_any",
        iv,
        "games",
        "yellowCards",
        "yellowReds",
        "goals",
        "height",
        "weight",
        "age",
        "meanIAT",
        "meanExp",
    ]
    explore_df = df[explore_cols].dropna()

    print("\n=== STEP 1: Summary statistics ===")
    print(explore_df.describe().T)

    pearson_r, pearson_p = stats.pearsonr(explore_df[iv], explore_df[dv])
    spearman_rho, spearman_p = stats.spearmanr(explore_df[iv], explore_df[dv])
    print(f"\nPearson corr({iv}, {dv}) = {pearson_r:.4f}, p={pearson_p:.4g}")
    print(f"Spearman corr({iv}, {dv}) = {spearman_rho:.4f}, p={spearman_p:.4g}")

    q25 = explore_df[iv].quantile(0.25)
    q75 = explore_df[iv].quantile(0.75)
    light = explore_df.loc[explore_df[iv] <= q25, dv]
    dark = explore_df.loc[explore_df[iv] >= q75, dv]
    t_stat, t_p = stats.ttest_ind(dark, light, equal_var=False, nan_policy="omit")
    print(
        "Dark-vs-light mean redCards "
        f"(top quartile vs bottom quartile): dark={dark.mean():.4f}, light={light.mean():.4f}, "
        f"diff={dark.mean() - light.mean():.4f}, p={t_p:.4g}"
    )

    # Simple bivariate OLS
    X_biv = sm.add_constant(explore_df[[iv]], has_constant="add")
    biv_ols = sm.OLS(explore_df[dv], X_biv).fit(cov_type="HC3")
    print("\nBivariate OLS (robust SE):")
    print(biv_ols.summary())

    # Step 2: controlled models
    print("\n=== STEP 2: Controlled models ===")
    numeric_controls = [
        "games",
        "yellowCards",
        "yellowReds",
        "goals",
        "height",
        "weight",
        "age",
        "meanIAT",
        "meanExp",
        "seIAT",
        "seExp",
    ]
    cat_controls = ["position", "leagueCountry"]

    model_cols = [dv, "red_any", iv] + numeric_controls + cat_controls
    model_df = df[model_cols].dropna().copy()
    print(f"Modeling rows after NA drop: {len(model_df)}")

    X_num = model_df[[iv] + numeric_controls]
    X_cat = pd.get_dummies(model_df[cat_controls], drop_first=True, dtype=float)
    X_full = pd.concat([X_num, X_cat], axis=1)
    X_full = sm.add_constant(X_full, has_constant="add")

    ols = sm.OLS(model_df[dv], X_full).fit(cov_type="HC3")
    print("\nControlled OLS on redCards (robust SE):")
    print(ols.summary())

    # Logistic robustness check for probability of any red card
    logit = None
    try:
        logit = sm.Logit(model_df["red_any"], X_full).fit(disp=False, maxiter=200)
        print("\nControlled Logistic on red_any:")
        print(logit.summary())
    except Exception as e:
        print(f"Logistic model failed: {e}")

    # Step 3: interpretable models with broad numeric feature set
    print("\n=== STEP 3: Interpretable models ===")
    df_for_interp = df.copy()

    numeric_cols = df_for_interp.select_dtypes(include=[np.number]).columns.tolist()
    # Remove IDs/duplicates of the IV and targets
    drop_from_features = {dv, "red_any", "refNum", "refCountry", "rater1", "rater2"}
    feature_cols = [c for c in numeric_cols if c not in drop_from_features]
    if iv not in feature_cols:
        feature_cols = [iv] + feature_cols

    interp_df = df_for_interp[feature_cols + [dv]].dropna().copy()
    # Keep runtime manageable while preserving signal
    if len(interp_df) > 60000:
        interp_df = interp_df.sample(n=60000, random_state=42)
        print("Sampled 60,000 rows for interpretable model fitting.")
    print(f"Interpretable modeling rows: {len(interp_df)}, features: {len(feature_cols)}")

    X_interp = interp_df[feature_cols]
    y_interp = interp_df[dv]

    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X_interp, y_interp)
    smart_effects = smart.feature_effects()
    print("\nSmartAdditiveRegressor:")
    print(smart)
    print("\nSmart effects dict:")
    print(smart_effects)

    hinge = HingeEBMRegressor(n_knots=3, max_input_features=min(15, len(feature_cols)), ebm_max_rounds=400)
    hinge.fit(X_interp, y_interp)
    hinge_effects = hinge.feature_effects()
    print("\nHingeEBMRegressor:")
    print(hinge)
    print("\nHinge effects dict:")
    print(hinge_effects)

    # Pull key stats for IV
    biv_coef = float(biv_ols.params.get(iv, np.nan))
    biv_p = float(biv_ols.pvalues.get(iv, np.nan))

    ols_coef = float(ols.params.get(iv, np.nan))
    ols_p = float(ols.pvalues.get(iv, np.nan))

    if logit is not None and iv in logit.params.index:
        logit_coef = float(logit.params[iv])
        logit_p = float(logit.pvalues[iv])
        logit_or = float(np.exp(logit_coef))
    else:
        logit_coef = np.nan
        logit_p = np.nan
        logit_or = np.nan

    smart_iv = smart_effects.get(iv, {"direction": "zero", "importance": 0.0, "rank": 0})
    hinge_iv = hinge_effects.get(iv, {"direction": "zero", "importance": 0.0, "rank": 0})

    smart_dir = smart_iv.get("direction", "zero")
    smart_imp = float(smart_iv.get("importance", 0.0) or 0.0)
    smart_rank = int(smart_iv.get("rank", 0) or 0)

    hinge_dir = hinge_iv.get("direction", "zero")
    hinge_imp = float(hinge_iv.get("importance", 0.0) or 0.0)
    hinge_rank = int(hinge_iv.get("rank", 0) or 0)

    # Determine shape/threshold evidence from SmartAdditive internals
    shape_summary = "effect too small for stable shape"
    threshold_summary = "no clear threshold"
    if smart_imp >= 0.01:
        try:
            iv_idx = smart.feature_names_.index(iv)
            if iv_idx in smart.shape_functions_:
                thresholds, intervals = smart.shape_functions_[iv_idx]
                _, _, r2 = smart.linear_approx_.get(iv_idx, (0.0, 0.0, 1.0))
                if r2 > 0.90:
                    shape_summary = "approximately linear"
                else:
                    shape_summary = smart_dir
                    if len(thresholds) > 0:
                        threshold_summary = f"threshold behavior appears around {thresholds[0]:.3f}"
        except Exception:
            pass

    # Confounders from model importance
    top_smart = top_effects(smart_effects, exclude=iv, k=3)
    top_hinge = top_effects(hinge_effects, exclude=iv, k=3)

    # Score synthesis
    score = 50

    # Controlled OLS and logistic are highest weight
    ols_strength = p_to_strength(ols_p) if np.isfinite(ols_p) else 0
    logit_strength = p_to_strength(logit_p) if np.isfinite(logit_p) else 0

    score += (10 * ols_strength) * (1 if ols_coef > 0 else -1 if ols_coef < 0 else 0)
    score += (6 * logit_strength) * (1 if logit_coef > 0 else -1 if logit_coef < 0 else 0)

    # Bivariate evidence (lighter weight)
    biv_strength = p_to_strength(biv_p) if np.isfinite(biv_p) else 0
    score += (4 * biv_strength) * (1 if biv_coef > 0 else -1 if biv_coef < 0 else 0)

    # Interpretable model evidence
    smart_sign = sign_from_direction(smart_dir)
    hinge_sign = sign_from_direction(hinge_dir)

    if smart_imp >= 0.10:
        score += 12 * smart_sign
    elif smart_imp >= 0.05:
        score += 8 * smart_sign
    elif smart_imp < 0.01:
        score -= 4

    if hinge_imp >= 0.10:
        score += 12 * hinge_sign
    elif hinge_imp >= 0.05:
        score += 8 * hinge_sign
    elif hinge_imp < 0.01:
        score -= 4

    # Penalize inconsistency across models
    signs = []
    if np.isfinite(ols_coef) and ols_strength > 0:
        signs.append(1 if ols_coef > 0 else -1)
    if np.isfinite(logit_coef) and logit_strength > 0:
        signs.append(1 if logit_coef > 0 else -1)
    if smart_imp >= 0.03 and smart_sign != 0:
        signs.append(smart_sign)
    if hinge_imp >= 0.03 and hinge_sign != 0:
        signs.append(hinge_sign)

    if signs and (any(s > 0 for s in signs) and any(s < 0 for s in signs)):
        score = int(round((score + 50) / 2))

    # If there is little evidence in all controlled models, force low score range
    weak_all = (
        (not np.isfinite(ols_p) or ols_p >= 0.10)
        and (not np.isfinite(logit_p) or logit_p >= 0.10)
        and smart_imp < 0.03
        and hinge_imp < 0.03
    )
    if weak_all:
        score = min(score, 20)

    # If interpretable models both null out skin tone, temper confidence
    if smart_imp < 0.01 and hinge_imp < 0.01:
        if (np.isfinite(ols_p) and ols_p < 0.05) or (np.isfinite(logit_p) and logit_p < 0.05):
            score = min(score, 65)
        else:
            score = min(score, 30)

    score = int(np.clip(round(score), 0, 100))

    # Compose explanation
    conf_smart_txt = ", ".join(
        [f"{n} ({d}, imp={i:.1%}, rank={r})" for n, i, d, r in top_smart]
    ) or "none"
    conf_hinge_txt = ", ".join(
        [f"{n} ({d}, imp={i:.1%}, rank={r})" for n, i, d, r in top_hinge]
    ) or "none"

    explanation = (
        f"Question: whether darker skin tone predicts more red cards. "
        f"Bivariate evidence is {'positive' if biv_coef > 0 else 'negative' if biv_coef < 0 else 'null'} "
        f"(OLS coef={safe_fmt(biv_coef)}, p={safe_fmt(biv_p)}; Pearson r={safe_fmt(pearson_r)}, p={safe_fmt(pearson_p)}). "
        f"After controls (games, discipline, performance, body metrics, league/position, and country-bias covariates), "
        f"the skin-tone coefficient is {safe_fmt(ols_coef)} (p={safe_fmt(ols_p)}) in OLS. "
        f"Logistic robustness on any red card gives coef={safe_fmt(logit_coef)} (OR={safe_fmt(logit_or, 3)}, p={safe_fmt(logit_p)}). "
        f"SmartAdditive ranks skin tone #{smart_rank} with importance {smart_imp:.1%}, direction '{smart_dir}', "
        f"shape '{shape_summary}' and {threshold_summary}. "
        f"HingeEBM ranks skin tone #{hinge_rank} with importance {hinge_imp:.1%}, direction '{hinge_dir}'. "
        f"Key confounders in SmartAdditive: {conf_smart_txt}. "
        f"Key confounders in HingeEBM: {conf_hinge_txt}. "
        f"Final score reflects direction, magnitude, nonlinearity, and consistency across bivariate, controlled, and interpretable models."
    )

    result = {"response": score, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=True)

    print("\nWrote conclusion.txt:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
