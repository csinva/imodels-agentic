import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm

from interp_models import SmartAdditiveRegressor, HingeEBMRegressor

warnings.filterwarnings("ignore")


def _safe_float(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def _extract_linear_term(model, var_name):
    try:
        coef = _safe_float(model.params.get(var_name, np.nan))
        pval = _safe_float(model.pvalues.get(var_name, np.nan))
    except Exception:
        coef, pval = np.nan, np.nan
    return coef, pval


def _smart_shape_summary(smart_model, feature_name):
    if feature_name not in smart_model.feature_names_:
        return "feature not present"

    effects = smart_model.feature_effects()
    feat_effect = effects.get(feature_name, {"importance": 0.0})
    if feat_effect.get("importance", 0.0) < 0.01:
        return "no meaningful learned effect"

    j = smart_model.feature_names_.index(feature_name)
    if j not in smart_model.shape_functions_:
        return "no learned effect"

    slope, _, r2 = smart_model.linear_approx_.get(j, (0.0, 0.0, 1.0))
    if r2 > 0.9:
        direction = "positive" if slope > 0 else "negative"
        return f"approximately linear {direction} (slope={slope:.4f}, r2={r2:.3f})"

    thresholds, intervals = smart_model.shape_functions_[j]
    if len(intervals) >= 2:
        trend = "increasing" if intervals[-1] > intervals[0] else "decreasing"
    else:
        trend = "flat"
    th = ", ".join([f"{t:.3f}" for t in thresholds[:4]])
    if len(thresholds) > 4:
        th += ", ..."
    return f"nonlinear {trend} with thresholds around [{th}]"


def main():
    info_path = Path("info.json")
    data_path = Path("soccer.csv")

    info = json.loads(info_path.read_text())
    question = info["research_questions"][0]

    print("Research question:")
    print(question)

    df = pd.read_csv(data_path)
    print(f"\nLoaded data with shape: {df.shape}")

    # DV and IV
    dv = "redCards"
    iv = "skin_tone"

    # Build IV: mean of two raters
    df[iv] = df[["rater1", "rater2"]].mean(axis=1, skipna=True)

    # Parse age at season midpoint (2012-07-01)
    bday = pd.to_datetime(df["birthday"], format="%d.%m.%Y", errors="coerce")
    season_mid = pd.Timestamp("2012-07-01")
    df["age"] = (season_mid - bday).dt.days / 365.25

    # Dark vs light subset for direct question wording
    df["dark_skin"] = np.where(df[iv] > 0.5, 1, np.where(df[iv] < 0.5, 0, np.nan))
    dl = df.dropna(subset=["dark_skin", dv]).copy()

    print("\nStep 1: Summary statistics")
    print(df[[dv, iv, "rater1", "rater2", "games", "yellowCards", "yellowReds", "meanIAT", "meanExp"]].describe())
    print("\nredCards distribution:")
    print(df[dv].value_counts(dropna=False).sort_index())

    # Bivariate correlations
    corr_skin_pearson = stats.pearsonr(df[iv].dropna(), df.loc[df[iv].notna(), dv])
    corr_skin_spearman = stats.spearmanr(df[iv], df[dv], nan_policy="omit")
    print("\nBivariate correlations with redCards")
    print(f"Pearson(skin_tone, redCards): r={corr_skin_pearson.statistic:.4f}, p={corr_skin_pearson.pvalue:.4g}")
    print(f"Spearman(skin_tone, redCards): rho={corr_skin_spearman.statistic:.4f}, p={corr_skin_spearman.pvalue:.4g}")

    # Dark vs light mean difference
    dark_mean = dl.loc[dl["dark_skin"] == 1, dv].mean()
    light_mean = dl.loc[dl["dark_skin"] == 0, dv].mean()
    ttest = stats.ttest_ind(
        dl.loc[dl["dark_skin"] == 1, dv],
        dl.loc[dl["dark_skin"] == 0, dv],
        equal_var=False,
        nan_policy="omit",
    )
    print("\nDark vs light raw comparison")
    print(f"Mean redCards (dark): {dark_mean:.4f}")
    print(f"Mean redCards (light): {light_mean:.4f}")
    print(f"Difference (dark-light): {dark_mean - light_mean:.4f}")
    print(f"Welch t-test p-value: {ttest.pvalue:.4g}")

    # Step 2: Controlled OLS (continuous IV)
    print("\nStep 2: OLS with controls")

    base_cols = [
        dv,
        iv,
        "games",
        "yellowCards",
        "yellowReds",
        "goals",
        "age",
        "height",
        "weight",
        "meanIAT",
        "meanExp",
        "position",
        "leagueCountry",
    ]
    model_df = df[base_cols].dropna().copy()

    X = model_df[[iv, "games", "yellowCards", "yellowReds", "goals", "age", "height", "weight", "meanIAT", "meanExp"]]
    X_cat = pd.get_dummies(model_df[["position", "leagueCountry"]], drop_first=True, dtype=float)
    X = pd.concat([X, X_cat], axis=1)
    X = sm.add_constant(X)
    y = model_df[dv].astype(float)

    ols_model = sm.OLS(y, X).fit(cov_type="HC3")
    print("\nOLS (continuous skin_tone) summary:")
    print(ols_model.summary())

    coef_skin, p_skin = _extract_linear_term(ols_model, iv)

    # Step 2b: Direct binary dark-vs-light controlled model
    dl_cols = [
        dv,
        "dark_skin",
        "games",
        "yellowCards",
        "yellowReds",
        "goals",
        "age",
        "height",
        "weight",
        "meanIAT",
        "meanExp",
        "position",
        "leagueCountry",
    ]
    dl_model_df = df[dl_cols].dropna().copy()

    X_dl = dl_model_df[["dark_skin", "games", "yellowCards", "yellowReds", "goals", "age", "height", "weight", "meanIAT", "meanExp"]]
    X_dl_cat = pd.get_dummies(dl_model_df[["position", "leagueCountry"]], drop_first=True, dtype=float)
    X_dl = pd.concat([X_dl, X_dl_cat], axis=1)
    X_dl = sm.add_constant(X_dl)
    y_dl = dl_model_df[dv].astype(float)

    ols_dark_model = sm.OLS(y_dl, X_dl).fit(cov_type="HC3")
    print("\nOLS (dark vs light indicator) summary:")
    print(ols_dark_model.summary())

    coef_dark, p_dark = _extract_linear_term(ols_dark_model, "dark_skin")

    # Step 3: Interpretable models on numeric columns
    print("\nStep 3: Interpretable models")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if dv in numeric_cols:
        numeric_cols.remove(dv)

    # Keep all numeric columns and add engineered skin_tone if missing
    if iv not in numeric_cols:
        numeric_cols.append(iv)

    interp_df = df[numeric_cols + [dv]].copy()
    for c in interp_df.columns:
        if interp_df[c].isna().any():
            interp_df[c] = interp_df[c].fillna(interp_df[c].median())

    X_interp = interp_df[numeric_cols]
    y_interp = interp_df[dv]

    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X_interp, y_interp)
    smart_effects = smart.feature_effects()
    print("\nSmartAdditiveRegressor:")
    print(smart)
    print("\nSmart feature effects:")
    print(smart_effects)

    hinge = HingeEBMRegressor(n_knots=3, ebm_max_rounds=300)
    hinge.fit(X_interp, y_interp)
    hinge_effects = hinge.feature_effects()
    print("\nHingeEBMRegressor:")
    print(hinge)
    print("\nHinge feature effects:")
    print(hinge_effects)

    # Pull model evidence for skin_tone
    smart_skin = smart_effects.get(iv, {"direction": "zero", "importance": 0.0, "rank": 0})
    hinge_skin = hinge_effects.get(iv, {"direction": "zero", "importance": 0.0, "rank": 0})
    smart_shape = _smart_shape_summary(smart, iv)

    # Evidence synthesis -> score (0-100)
    score = 10

    # Bivariate evidence
    if (dark_mean - light_mean) > 0:
        score += 5
    if ttest.pvalue < 0.05 and (dark_mean - light_mean) > 0:
        score += 10

    # Correlation and controlled OLS evidence
    if corr_skin_pearson.pvalue < 0.05 and corr_skin_pearson.statistic > 0:
        score += 10
    if np.isfinite(coef_skin) and coef_skin > 0:
        score += 10
    if np.isfinite(p_skin) and p_skin < 0.05 and coef_skin > 0:
        score += 10

    # Direct dark-vs-light controlled estimate
    if np.isfinite(coef_dark) and coef_dark > 0:
        score += 5
    if np.isfinite(p_dark) and p_dark < 0.05 and coef_dark > 0:
        score += 15

    # Interpretable-model robustness
    smart_imp = smart_skin.get("importance", 0.0)
    hinge_imp = hinge_skin.get("importance", 0.0)
    smart_pos = "positive" in str(smart_skin.get("direction", "")) or "increasing" in str(smart_skin.get("direction", ""))
    hinge_pos = hinge_skin.get("direction", "") == "positive"

    if smart_imp >= 0.01 and smart_pos:
        score += 10
    if hinge_imp >= 0.01 and hinge_pos:
        score += 10

    # Penalize inconsistency: significant only in one specification, absent in interpretable models
    if (p_dark >= 0.05) and (smart_imp < 0.01) and (hinge_imp < 0.01):
        score -= 15

    score = int(np.clip(score, 0, 100))

    # Main confounders from OLS by absolute coefficient magnitude among standardized-ish counts
    top_pvals = ols_model.pvalues.sort_values()
    strongest_controls = [
        name for name in top_pvals.index
        if name not in {"const", iv} and top_pvals[name] < 0.05
    ][:4]

    explanation = (
        f"Research question: whether darker skin tone predicts more red cards. "
        f"Bivariate comparison shows dark-skin players average {dark_mean:.4f} red cards vs {light_mean:.4f} for light-skin "
        f"(difference={dark_mean - light_mean:.4f}, Welch p={ttest.pvalue:.4g}). "
        f"Correlation with continuous skin tone is Pearson r={corr_skin_pearson.statistic:.4f} (p={corr_skin_pearson.pvalue:.4g}). "
        f"In controlled OLS, skin_tone coef={coef_skin:.4f} (p={p_skin:.4g}); binary dark_skin coef={coef_dark:.4f} (p={p_dark:.4g}). "
        f"SmartAdditive ranks skin_tone #{smart_skin.get('rank', 0)} with importance={smart_skin.get('importance', 0.0):.4f}, "
        f"direction={smart_skin.get('direction', 'zero')}, shape={smart_shape}. "
        f"HingeEBM ranks skin_tone #{hinge_skin.get('rank', 0)} with importance={hinge_skin.get('importance', 0.0):.4f}, "
        f"direction={hinge_skin.get('direction', 'zero')}. "
        f"Key confounders in OLS include {', '.join(strongest_controls) if strongest_controls else 'no strong controls at p<0.05'}. "
        f"Overall score reflects direction, magnitude, nonlinear shape, and robustness across bivariate, controlled, and interpretable models."
    )

    out = {"response": int(score), "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(out, ensure_ascii=True))

    print("\nWrote conclusion.txt")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
