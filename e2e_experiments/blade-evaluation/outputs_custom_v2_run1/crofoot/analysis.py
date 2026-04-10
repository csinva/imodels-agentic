import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr

from interp_models import HingeEBMRegressor, SmartAdditiveRegressor


def safe_pearson(x: pd.Series, y: pd.Series):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.isclose(x.std(ddof=0), 0) or np.isclose(y.std(ddof=0), 0):
        return np.nan, np.nan
    r, p = pearsonr(x, y)
    return float(r), float(p)


def fit_binomial_glm(y: pd.Series, X: pd.DataFrame):
    Xc = sm.add_constant(X, has_constant="add")
    model = sm.GLM(y, Xc, family=sm.families.Binomial()).fit()
    return model


def get_coef_p(model, name: str):
    coef = float(model.params[name]) if name in model.params else np.nan
    pval = float(model.pvalues[name]) if name in model.pvalues else np.nan
    return coef, pval


def direction_matches(direction: str, expected_sign: int):
    d = (direction or "").lower()
    if expected_sign > 0:
        if "positive" in d or "increasing" in d:
            return True
        if "negative" in d or "decreasing" in d:
            return False
        if "non-monotonic" in d:
            return True
        return False
    if expected_sign < 0:
        if "negative" in d or "decreasing" in d:
            return True
        if "positive" in d or "increasing" in d:
            return False
        if "non-monotonic" in d:
            return True
        return False
    return False


def score_parametric(sign_ok: bool, pval: float, max_points: float):
    if not sign_ok or np.isnan(pval):
        return 0.0
    if pval < 0.05:
        return max_points
    if pval < 0.10:
        return 0.75 * max_points
    if pval < 0.20:
        return 0.45 * max_points
    return 0.20 * max_points


def construct_support_from_effects(effects: dict, feature_expectations: dict):
    support_imp = 0.0
    oppose_imp = 0.0
    for feat, expected_sign in feature_expectations.items():
        e = effects.get(feat)
        if not e:
            continue
        imp = float(e.get("importance", 0.0) or 0.0)
        if imp <= 0:
            continue
        ok = direction_matches(str(e.get("direction", "")), expected_sign)
        if ok:
            support_imp += imp
        else:
            oppose_imp += imp

    delta = support_imp - oppose_imp
    if delta >= 0.08:
        return 1.0
    if delta >= 0.03:
        return 0.5
    if support_imp > oppose_imp and support_imp > 0:
        return 0.25
    return 0.0


def fmt_num(x, digits=3):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "NA"
    return f"{x:.{digits}f}"


def main():
    info_path = Path("info.json")
    data_path = Path("crofoot.csv")

    info = json.loads(info_path.read_text())
    question = info.get("research_questions", ["Research question unavailable"])[0]

    df = pd.read_csv(data_path)

    # Identify DV: prefer explicit 'win' for this dataset, else first binary numeric column.
    dv = "win" if "win" in df.columns else None
    if dv is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        binary_candidates = [
            c for c in numeric_cols if df[c].dropna().nunique() == 2
        ]
        if not binary_candidates:
            raise ValueError("Could not identify binary dependent variable.")
        dv = binary_candidates[0]

    # Construct interpretable IVs tied to the research question wording.
    if {"n_focal", "n_other"}.issubset(df.columns):
        df["rel_group_size"] = df["n_focal"] - df["n_other"]
    if {"dist_focal", "dist_other"}.issubset(df.columns):
        # Positive = contest closer to focal home-range center than to other group center.
        df["rel_location_adv"] = df["dist_other"] - df["dist_focal"]

    print("=" * 80)
    print("Research question:")
    print(question)
    print("=" * 80)
    print(f"Data shape: {df.shape}")
    print(f"Dependent variable (DV): {dv}")

    # Step 1: Explore
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    predictors = [c for c in numeric_cols if c != dv]

    print("\nStep 1: Summary statistics")
    print(df[numeric_cols].describe().T)

    print("\nDV distribution:")
    print(df[dv].value_counts(dropna=False).sort_index())
    print(df[dv].value_counts(normalize=True, dropna=False).sort_index())

    corr_rows = []
    for col in predictors:
        r, p = safe_pearson(df[col], df[dv])
        corr_rows.append({"feature": col, "pearson_r": r, "p_value": p, "abs_r": abs(r) if not np.isnan(r) else np.nan})
    corr_df = pd.DataFrame(corr_rows).sort_values("abs_r", ascending=False)

    print("\nTop bivariate correlations with DV:")
    print(corr_df[["feature", "pearson_r", "p_value"]].head(12).to_string(index=False))

    # Step 2: Controlled regression models
    iv_size = "rel_group_size" if "rel_group_size" in df.columns else None
    iv_loc = "rel_location_adv" if "rel_location_adv" in df.columns else None

    id_controls = [c for c in ["focal", "other", "dyad"] if c in df.columns]

    rel_model = None
    rel_features = [c for c in [iv_size, iv_loc] if c is not None] + id_controls
    if rel_features:
        rel_model = fit_binomial_glm(df[dv], df[rel_features])
        print("\nStep 2A: Binomial GLM with relative IVs + ID controls")
        print(rel_model.summary())

    raw_features = [c for c in ["n_focal", "n_other", "dist_focal", "dist_other"] if c in df.columns] + id_controls
    raw_model = None
    if raw_features:
        raw_model = fit_binomial_glm(df[dv], df[raw_features])
        print("\nStep 2B: Binomial GLM with raw size/location vars + ID controls")
        print(raw_model.summary())

    # Step 3: Custom interpretable models using all numeric predictors
    X_custom = df[predictors].copy()
    y = df[dv].copy()

    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X_custom, y)
    smart_effects = smart.feature_effects()

    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X_custom, y)
    hinge_effects = hinge.feature_effects()

    print("\nStep 3A: SmartAdditiveRegressor")
    print(smart)
    print("Feature effects:")
    print(smart_effects)

    print("\nStep 3B: HingeEBMRegressor")
    print(hinge)
    print("Feature effects:")
    print(hinge_effects)

    # Extract core evidence for question variables
    bivar_size_r, bivar_size_p = safe_pearson(df[iv_size], df[dv]) if iv_size else (np.nan, np.nan)
    bivar_loc_r, bivar_loc_p = safe_pearson(df[iv_loc], df[dv]) if iv_loc else (np.nan, np.nan)

    rel_size_coef = rel_size_p = rel_loc_coef = rel_loc_p = np.nan
    if rel_model is not None and iv_size in rel_features:
        rel_size_coef, rel_size_p = get_coef_p(rel_model, iv_size)
    if rel_model is not None and iv_loc in rel_features:
        rel_loc_coef, rel_loc_p = get_coef_p(rel_model, iv_loc)

    # Raw model components
    n_focal_coef = n_focal_p = n_other_coef = n_other_p = np.nan
    dist_focal_coef = dist_focal_p = dist_other_coef = dist_other_p = np.nan
    if raw_model is not None:
        if "n_focal" in raw_features:
            n_focal_coef, n_focal_p = get_coef_p(raw_model, "n_focal")
        if "n_other" in raw_features:
            n_other_coef, n_other_p = get_coef_p(raw_model, "n_other")
        if "dist_focal" in raw_features:
            dist_focal_coef, dist_focal_p = get_coef_p(raw_model, "dist_focal")
        if "dist_other" in raw_features:
            dist_other_coef, dist_other_p = get_coef_p(raw_model, "dist_other")

    # Scoring rubric (0-100)
    total_points = 0.0
    max_points = 12.0

    # Bivariate evidence (2 points total)
    total_points += score_parametric(sign_ok=(bivar_size_r > 0), pval=bivar_size_p, max_points=1.0)
    total_points += score_parametric(sign_ok=(bivar_loc_r > 0), pval=bivar_loc_p, max_points=1.0)

    # Relative controlled GLM evidence (4 points total)
    total_points += score_parametric(sign_ok=(rel_size_coef > 0), pval=rel_size_p, max_points=2.0)
    total_points += score_parametric(sign_ok=(rel_loc_coef > 0), pval=rel_loc_p, max_points=2.0)

    # Raw controlled GLM evidence (2 points total)
    size_raw_sign_ok = (n_focal_coef > 0) and (n_other_coef < 0)
    size_raw_p = np.nanmin([n_focal_p, n_other_p]) if not (np.isnan(n_focal_p) and np.isnan(n_other_p)) else np.nan
    total_points += score_parametric(sign_ok=size_raw_sign_ok, pval=size_raw_p, max_points=1.0)

    loc_raw_sign_ok = (dist_focal_coef < 0) and (dist_other_coef > 0)
    loc_raw_p = np.nanmin([dist_focal_p, dist_other_p]) if not (np.isnan(dist_focal_p) and np.isnan(dist_other_p)) else np.nan
    # partial credit if one of the two signs matches expected
    if not loc_raw_sign_ok:
        partial_loc_sign_ok = (dist_focal_coef < 0) or (dist_other_coef > 0)
        total_points += score_parametric(sign_ok=partial_loc_sign_ok, pval=loc_raw_p, max_points=0.5)
    else:
        total_points += score_parametric(sign_ok=True, pval=loc_raw_p, max_points=1.0)

    # Interpretable model construct-level support (4 points total)
    size_expectations = {k: v for k, v in {
        "rel_group_size": +1,
        "n_focal": +1,
        "n_other": -1,
    }.items() if k in predictors}

    loc_expectations = {k: v for k, v in {
        "rel_location_adv": +1,
        "dist_focal": -1,
        "dist_other": +1,
    }.items() if k in predictors}

    smart_size_support = construct_support_from_effects(smart_effects, size_expectations)
    smart_loc_support = construct_support_from_effects(smart_effects, loc_expectations)
    hinge_size_support = construct_support_from_effects(hinge_effects, size_expectations)
    hinge_loc_support = construct_support_from_effects(hinge_effects, loc_expectations)

    total_points += smart_size_support
    total_points += smart_loc_support
    total_points += hinge_size_support
    total_points += hinge_loc_support

    score = int(np.clip(round(100.0 * (total_points / max_points)), 0, 100))

    # Summaries for explanation
    def safe_effect(effects, feat):
        e = effects.get(feat, {}) if isinstance(effects, dict) else {}
        return {
            "direction": str(e.get("direction", "zero")),
            "importance": float(e.get("importance", 0.0) or 0.0),
            "rank": int(e.get("rank", 0) or 0),
        }

    smart_size = safe_effect(smart_effects, "rel_group_size")
    smart_loc = safe_effect(smart_effects, "rel_location_adv")
    hinge_size = safe_effect(hinge_effects, "rel_group_size")
    hinge_loc = safe_effect(hinge_effects, "rel_location_adv")

    # Top non-target predictors from SmartAdditive as confounders
    target_vars = {"rel_group_size", "rel_location_adv", "n_focal", "n_other", "dist_focal", "dist_other"}
    smart_ranked = sorted(
        [(k, v) for k, v in smart_effects.items() if isinstance(v, dict)],
        key=lambda kv: float(kv[1].get("importance", 0.0) or 0.0),
        reverse=True,
    )
    confounders = []
    for feat, eff in smart_ranked:
        if feat in target_vars:
            continue
        imp = float(eff.get("importance", 0.0) or 0.0)
        if imp <= 0:
            continue
        confounders.append(f"{feat} ({imp:.1%}, {eff.get('direction', 'unknown')})")
        if len(confounders) == 2:
            break
    confounder_text = ", ".join(confounders) if confounders else "none with meaningful importance"

    explanation = (
        f"Bivariate evidence is weak: rel_group_size has r={fmt_num(bivar_size_r)} (p={fmt_num(bivar_size_p)}), "
        f"and rel_location_adv has r={fmt_num(bivar_loc_r)} (p={fmt_num(bivar_loc_p)}). "
        f"In controlled binomial GLM with relative predictors, rel_group_size is {fmt_num(rel_size_coef)} "
        f"(p={fmt_num(rel_size_p)}) and rel_location_adv is {fmt_num(rel_loc_coef, 4)} (p={fmt_num(rel_loc_p)}), "
        f"so classical significance is limited. A raw-feature GLM is directionally suggestive for size "
        f"(n_focal={fmt_num(n_focal_coef)}, n_other={fmt_num(n_other_coef)}) and partly for location "
        f"(dist_focal={fmt_num(dist_focal_coef,4)}, dist_other={fmt_num(dist_other_coef,4)}), but mostly not significant. "
        f"SmartAdditive indicates nonlinear positive patterns for both focal advantages: rel_group_size rank {smart_size['rank']} "
        f"(importance={smart_size['importance']:.1%}, {smart_size['direction']}) and rel_location_adv rank {smart_loc['rank']} "
        f"(importance={smart_loc['importance']:.1%}, {smart_loc['direction']}), suggesting threshold-like effects. "
        f"HingeEBM is more conservative and zeros both relative variables directly "
        f"(rel_group_size importance={hinge_size['importance']:.1%}, rel_location_adv importance={hinge_loc['importance']:.1%}), "
        f"so robustness across models is mixed. Important additional predictors include {confounder_text}. "
        f"Overall, evidence supports at most a moderate influence, stronger for contest location than for relative group size."
    )

    payload = {"response": int(score), "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(payload, ensure_ascii=True))

    print("\nWrote conclusion.txt:")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
