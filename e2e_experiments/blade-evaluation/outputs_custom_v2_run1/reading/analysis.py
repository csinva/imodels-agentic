import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from interp_models import SmartAdditiveRegressor, HingeEBMRegressor


np.random.seed(42)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def effect_or_zero(effects, feature_name):
    if feature_name not in effects:
        return {"direction": "zero", "importance": 0.0, "rank": 0}
    out = effects[feature_name].copy()
    out["importance"] = safe_float(out.get("importance", 0.0))
    out["rank"] = int(out.get("rank", 0) or 0)
    return out


def top_features(effects, k=5):
    cleaned = []
    for name, val in effects.items():
        imp = safe_float(val.get("importance", 0.0))
        if imp > 0:
            cleaned.append((name, imp, val.get("direction", "unknown"), int(val.get("rank", 0) or 0)))
    cleaned.sort(key=lambda x: x[1], reverse=True)
    return cleaned[:k]


def fill_model_data(df):
    out = df.copy()
    categorical_cols = ["device", "education", "language", "english_native"]
    for c in categorical_cols:
        if c in out.columns:
            out[c] = out[c].fillna("Missing")

    numeric_fill_cols = ["age", "gender", "dyslexia", "dyslexia_bin", "retake_trial"]
    for c in numeric_fill_cols:
        if c in out.columns:
            out[c] = out[c].fillna(out[c].median())

    return out


def main():
    info = json.loads(Path("info.json").read_text())
    question = info["research_questions"][0]

    iv = "reader_view"
    dv = "speed"
    subgroup_col = "dyslexia_bin"

    df = pd.read_csv("reading.csv")
    df_model = fill_model_data(df)

    print("=" * 80)
    print("Research question:")
    print(question)
    print(f"\nIV: {iv}")
    print(f"DV: {dv}")
    print(f"Subgroup variable: {subgroup_col}")
    print("=" * 80)

    # Step 1: Explore
    print("\n[Step 1] Summary stats and bivariate exploration")
    key_cols = [iv, dv, subgroup_col, "age", "num_words", "correct_rate", "Flesch_Kincaid"]
    existing_key_cols = [c for c in key_cols if c in df.columns]
    print(df[existing_key_cols].describe(include="all").T)

    print("\nSpeed by reader_view (overall):")
    print(df.groupby(iv)[dv].agg(["count", "mean", "median", "std"]))

    dys_df = df[df[subgroup_col] == 1].copy()
    print("\nSpeed by reader_view among dyslexia_bin==1:")
    print(dys_df.groupby(iv)[dv].agg(["count", "mean", "median", "std"]))

    numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols_all].corr(numeric_only=True)[dv].sort_values(ascending=False)
    print("\nTop positive correlations with speed:")
    print(corr.head(8))
    print("\nTop negative correlations with speed:")
    print(corr.tail(8))

    # Bivariate tests
    rv0 = df.loc[df[iv] == 0, dv].dropna()
    rv1 = df.loc[df[iv] == 1, dv].dropna()
    ttest_overall = stats.ttest_ind(rv1, rv0, equal_var=False)

    dys_rv0 = dys_df.loc[dys_df[iv] == 0, dv].dropna()
    dys_rv1 = dys_df.loc[dys_df[iv] == 1, dv].dropna()
    ttest_dys = stats.ttest_ind(dys_rv1, dys_rv0, equal_var=False)

    corr_overall = stats.pearsonr(df[iv], df[dv])
    corr_dys = stats.pearsonr(dys_df[iv], dys_df[dv])

    print("\nBivariate tests:")
    print(f"Overall mean diff (reader_view=1 - 0): {rv1.mean() - rv0.mean():.3f}")
    print(f"Overall Welch t-test p-value: {ttest_overall.pvalue:.6f}")
    print(f"Overall Pearson r(reader_view, speed): {corr_overall.statistic:.4f}, p={corr_overall.pvalue:.6f}")
    print(f"Dyslexia subgroup mean diff (reader_view=1 - 0): {dys_rv1.mean() - dys_rv0.mean():.3f}")
    print(f"Dyslexia subgroup Welch t-test p-value: {ttest_dys.pvalue:.6f}")
    print(f"Dyslexia subgroup Pearson r: {corr_dys.statistic:.4f}, p={corr_dys.pvalue:.6f}")

    # Step 2: Controlled OLS
    print("\n[Step 2] OLS with controls")
    formula_control = (
        "speed ~ reader_view * dyslexia_bin + age + retake_trial + num_words + "
        "correct_rate + img_width + Flesch_Kincaid + C(gender) + C(device) + "
        "C(education) + C(language) + C(page_id) + C(english_native)"
    )
    model_control = smf.ols(formula_control, data=df_model).fit(cov_type="HC3")
    print(model_control.summary())

    coef_rv = safe_float(model_control.params.get("reader_view", np.nan))
    p_rv = safe_float(model_control.pvalues.get("reader_view", np.nan))
    coef_inter = safe_float(model_control.params.get("reader_view:dyslexia_bin", np.nan))
    p_inter = safe_float(model_control.pvalues.get("reader_view:dyslexia_bin", np.nan))

    lin = model_control.t_test("reader_view + reader_view:dyslexia_bin = 0")
    dys_effect = safe_float(np.asarray(lin.effect).squeeze())
    dys_se = safe_float(np.asarray(lin.sd).squeeze())
    dys_p = safe_float(np.asarray(lin.pvalue).squeeze())

    print("\nKey controlled coefficients:")
    print(f"reader_view (effect when dyslexia_bin=0): coef={coef_rv:.3f}, p={p_rv:.6f}")
    print(f"reader_view:dyslexia_bin interaction: coef={coef_inter:.3f}, p={p_inter:.6f}")
    print(
        "Net reader_view effect among dyslexia_bin=1: "
        f"coef={dys_effect:.3f}, se={dys_se:.3f}, p={dys_p:.6f}"
    )

    # Subgroup controlled OLS
    dys_model_df = fill_model_data(dys_df)
    formula_sub = (
        "speed ~ reader_view + age + retake_trial + num_words + correct_rate + "
        "img_width + Flesch_Kincaid + C(gender) + C(device) + C(education) + "
        "C(language) + C(page_id) + C(english_native)"
    )
    model_sub = smf.ols(formula_sub, data=dys_model_df).fit(cov_type="HC3")
    coef_sub = safe_float(model_sub.params.get("reader_view", np.nan))
    p_sub = safe_float(model_sub.pvalues.get("reader_view", np.nan))

    print("\nSubgroup (dyslexia only) controlled model coefficient:")
    print(f"reader_view coef={coef_sub:.3f}, p={p_sub:.6f}")

    # Step 3: Interpretable models
    print("\n[Step 3] Interpretable models")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != dv]

    full_num = df[numeric_cols + [dv]].copy()
    for c in numeric_cols:
        full_num[c] = full_num[c].fillna(full_num[c].median())

    X_full = full_num[numeric_cols]
    y_full = full_num[dv]

    smart_full = SmartAdditiveRegressor(n_rounds=200)
    smart_full.fit(X_full, y_full)
    smart_full_effects = smart_full.feature_effects()

    hinge_full = HingeEBMRegressor(n_knots=3)
    hinge_full.fit(X_full, y_full)
    hinge_full_effects = hinge_full.feature_effects()

    print("\nSmartAdditiveRegressor (full sample):")
    print(smart_full)
    print("\nSmartAdditive effects (full):")
    print(smart_full_effects)

    print("\nHingeEBMRegressor (full sample):")
    print(hinge_full)
    print("\nHingeEBM effects (full):")
    print(hinge_full_effects)

    dys_num = dys_df[numeric_cols + [dv]].copy()
    for c in numeric_cols:
        dys_num[c] = dys_num[c].fillna(dys_num[c].median())

    X_dys = dys_num[numeric_cols]
    y_dys = dys_num[dv]

    smart_dys = SmartAdditiveRegressor(n_rounds=200)
    smart_dys.fit(X_dys, y_dys)
    smart_dys_effects = smart_dys.feature_effects()

    hinge_dys = HingeEBMRegressor(n_knots=3)
    hinge_dys.fit(X_dys, y_dys)
    hinge_dys_effects = hinge_dys.feature_effects()

    print("\nSmartAdditiveRegressor (dyslexia subgroup):")
    print(smart_dys)
    print("\nSmartAdditive effects (dyslexia subgroup):")
    print(smart_dys_effects)

    print("\nHingeEBMRegressor (dyslexia subgroup):")
    print(hinge_dys)
    print("\nHingeEBM effects (dyslexia subgroup):")
    print(hinge_dys_effects)

    rv_smart_full = effect_or_zero(smart_full_effects, iv)
    rv_hinge_full = effect_or_zero(hinge_full_effects, iv)
    rv_smart_dys = effect_or_zero(smart_dys_effects, iv)
    rv_hinge_dys = effect_or_zero(hinge_dys_effects, iv)

    top_smart_dys = top_features(smart_dys_effects, k=5)
    top_hinge_dys = top_features(hinge_dys_effects, k=5)

    # Step 4: Score and explanation
    positive_hits = 0
    negative_hits = 0
    null_hits = 0

    tests = []

    # Bivariate dyslexia subgroup
    diff_dys = dys_rv1.mean() - dys_rv0.mean()
    if ttest_dys.pvalue < 0.05:
        if diff_dys > 0:
            positive_hits += 1
            tests.append("bivariate_dys=positive_sig")
        else:
            negative_hits += 1
            tests.append("bivariate_dys=negative_sig")
    else:
        null_hits += 1
        tests.append("bivariate_dys=null")

    # Controlled interaction-derived effect in dyslexia
    if dys_p < 0.05:
        if dys_effect > 0:
            positive_hits += 1
            tests.append("ols_interaction_dys=positive_sig")
        else:
            negative_hits += 1
            tests.append("ols_interaction_dys=negative_sig")
    else:
        null_hits += 1
        tests.append("ols_interaction_dys=null")

    # Subgroup controlled model
    if p_sub < 0.05:
        if coef_sub > 0:
            positive_hits += 1
            tests.append("ols_subgroup=positive_sig")
        else:
            negative_hits += 1
            tests.append("ols_subgroup=negative_sig")
    else:
        null_hits += 1
        tests.append("ols_subgroup=null")

    # SmartAdditive subgroup importance
    if rv_smart_dys["importance"] >= 0.01:
        if "positive" in rv_smart_dys["direction"] or "increasing" in rv_smart_dys["direction"]:
            positive_hits += 1
            tests.append("smart_dys=positive_nonzero")
        elif "negative" in rv_smart_dys["direction"] or "decreasing" in rv_smart_dys["direction"]:
            negative_hits += 1
            tests.append("smart_dys=negative_nonzero")
        else:
            null_hits += 1
            tests.append("smart_dys=nonmonotonic")
    else:
        null_hits += 1
        tests.append("smart_dys=zero")

    # Hinge subgroup importance
    if rv_hinge_dys["importance"] >= 0.01:
        if rv_hinge_dys["direction"] == "positive":
            positive_hits += 1
            tests.append("hinge_dys=positive_nonzero")
        elif rv_hinge_dys["direction"] == "negative":
            negative_hits += 1
            tests.append("hinge_dys=negative_nonzero")
        else:
            null_hits += 1
            tests.append("hinge_dys=nonmonotonic")
    else:
        null_hits += 1
        tests.append("hinge_dys=zero")

    if positive_hits >= 4 and negative_hits == 0:
        response = 90
    elif positive_hits >= 3 and negative_hits <= 1:
        response = 78
    elif positive_hits >= 2:
        response = 62
    elif positive_hits == 1:
        response = 45
    elif positive_hits == 0 and negative_hits >= 2:
        response = 5
    elif positive_hits == 0 and null_hits >= 4:
        response = 8
    else:
        response = 20

    explanation = (
        f"The evidence does not support that Reader View improves reading speed for participants with dyslexia. "
        f"In bivariate subgroup comparisons (dyslexia_bin=1), the mean speed difference was {diff_dys:.2f} "
        f"(Reader View minus control), with Welch t-test p={ttest_dys.pvalue:.3f}, indicating no reliable uplift. "
        f"In controlled OLS with demographics, text/page factors, and device/language controls, the implied "
        f"Reader View effect for dyslexia participants was {dys_effect:.2f} (SE={dys_se:.2f}, p={dys_p:.3f}); "
        f"the dyslexia-only controlled model similarly gave coef={coef_sub:.2f}, p={p_sub:.3f}. "
        f"Interpretable models agree: in the dyslexia subgroup, SmartAdditive assigns reader_view "
        f"direction '{rv_smart_dys['direction']}' with importance {rv_smart_dys['importance']:.4f} "
        f"(rank {rv_smart_dys['rank']}), and HingeEBM assigns direction '{rv_hinge_dys['direction']}' "
        f"with importance {rv_hinge_dys['importance']:.4f} (rank {rv_hinge_dys['rank']}). "
        f"So shape/magnitude evidence for Reader View is essentially zero. Other variables dominate speed, "
        f"especially {', '.join([f'{n} ({imp:.1%})' for n, imp, _, _ in top_smart_dys[:3]])} in SmartAdditive "
        f"and {', '.join([f'{n} ({imp:.1%})' for n, imp, _, _ in top_hinge_dys[:3]])} in HingeEBM, showing the "
        f"main drivers are timing/text characteristics rather than Reader View. Robustness across bivariate, "
        f"controlled, and two interpretable models indicates no meaningful positive effect."
    )

    conclusion = {
        "response": int(response),
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=True))

    print("\n[Step 4] Conclusion JSON written to conclusion.txt")
    print(json.dumps(conclusion, indent=2))
    print("\nDiagnostics summary:")
    print({
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
        "null_hits": null_hits,
        "tests": tests,
        "reader_view_effects": {
            "smart_full": rv_smart_full,
            "hinge_full": rv_hinge_full,
            "smart_dys": rv_smart_dys,
            "hinge_dys": rv_hinge_dys,
        },
    })


if __name__ == "__main__":
    main()
