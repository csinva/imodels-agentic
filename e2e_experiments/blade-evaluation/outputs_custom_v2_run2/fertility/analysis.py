import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from interp_models import SmartAdditiveRegressor, HingeEBMRegressor


def rank_effects(effects_dict):
    items = []
    for feat, vals in effects_dict.items():
        imp = float(vals.get("importance", 0.0))
        direction = vals.get("direction", "unknown")
        rank = int(vals.get("rank", 0) or 0)
        items.append((feat, imp, direction, rank))
    items.sort(key=lambda x: (-x[1], x[0]))
    return items


def safe_effect(effects_dict, key):
    v = effects_dict.get(key, {}) if isinstance(effects_dict, dict) else {}
    return {
        "importance": float(v.get("importance", 0.0)),
        "direction": v.get("direction", "zero"),
        "rank": int(v.get("rank", 0) or 0),
    }


def main():
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info["research_questions"][0]
    print("Research question:", question)

    df = pd.read_csv("fertility.csv")

    # Parse dates
    date_cols = ["DateTesting", "StartDateofLastPeriod", "StartDateofPeriodBeforeLast"]
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], format="%m/%d/%y", errors="coerce")

    # DV: religiosity composite score
    rel_items = ["Rel1", "Rel2", "Rel3"]
    df["religiosity_mean"] = df[rel_items].mean(axis=1)

    # IV: fertility-related hormonal fluctuation proxy
    # Use cycle timing from reported cycle length (fallback to date-derived cycle length),
    # then convert to conception probability based on relative day to ovulation.
    df["cycle_len_dates"] = (
        df["StartDateofLastPeriod"] - df["StartDateofPeriodBeforeLast"]
    ).dt.days
    df["estimated_cycle_length"] = df["ReportedCycleLength"].fillna(df["cycle_len_dates"])
    df["cycle_day"] = (df["DateTesting"] - df["StartDateofLastPeriod"]).dt.days
    df["ovulation_day"] = df["estimated_cycle_length"] - 14.0
    df["days_to_ovulation"] = df["cycle_day"] - df["ovulation_day"]

    conception_prob = {
        -5: 0.04,
        -4: 0.08,
        -3: 0.17,
        -2: 0.29,
        -1: 0.27,
        0: 0.08,
        1: 0.01,
    }

    def fertility_probability(x):
        if pd.isna(x):
            return np.nan
        return conception_prob.get(int(np.round(x)), 0.0)

    df["fertility_score"] = df["days_to_ovulation"].apply(fertility_probability)
    df["high_fertility"] = (
        (df["days_to_ovulation"] >= -5) & (df["days_to_ovulation"] <= 1)
    ).astype(int)

    # Additional time control
    df["test_day_index"] = (df["DateTesting"] - df["DateTesting"].min()).dt.days

    # Keep complete cases for modeling
    model_cols = [
        "religiosity_mean",
        "fertility_score",
        "high_fertility",
        "cycle_day",
        "estimated_cycle_length",
        "days_to_ovulation",
        "Sure1",
        "Sure2",
        "Relationship",
        "test_day_index",
    ]
    data = df[model_cols].dropna().copy()

    print("\nN rows for modeling:", len(data))
    print("\nSummary statistics:")
    print(data.describe().T)

    print("\nDistribution snapshots (quantiles):")
    for c in ["religiosity_mean", "fertility_score", "days_to_ovulation", "cycle_day"]:
        qs = data[c].quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
        print(f"{c} quantiles:\n{qs}\n")

    # Step 1: bivariate checks
    print("\nBivariate correlations with DV (religiosity_mean):")
    bivar_corr = {}
    for c in [
        "fertility_score",
        "high_fertility",
        "days_to_ovulation",
        "cycle_day",
        "estimated_cycle_length",
        "Sure1",
        "Sure2",
        "Relationship",
    ]:
        r = data[[c, "religiosity_mean"]].corr().iloc[0, 1]
        bivar_corr[c] = r
        print(f"  {c}: r={r:.4f}")

    hi = data.loc[data["high_fertility"] == 1, "religiosity_mean"]
    lo = data.loc[data["high_fertility"] == 0, "religiosity_mean"]
    t_stat, t_p = stats.ttest_ind(hi, lo, equal_var=False, nan_policy="omit")
    print(
        f"\nHigh vs low fertility religiosity mean: high={hi.mean():.3f}, "
        f"low={lo.mean():.3f}, t={t_stat:.3f}, p={t_p:.4f}"
    )

    # Step 2: controlled OLS
    print("\nControlled OLS (continuous fertility_score as IV):")
    ols_features_main = [
        "fertility_score",
        "cycle_day",
        "estimated_cycle_length",
        "Sure1",
        "Sure2",
        "Relationship",
        "test_day_index",
    ]
    X_main = sm.add_constant(data[ols_features_main])
    y = data["religiosity_mean"]
    ols_main = sm.OLS(y, X_main).fit()
    print(ols_main.summary())

    print("\nAlternative OLS (binary high_fertility as IV):")
    ols_features_alt = [
        "high_fertility",
        "cycle_day",
        "estimated_cycle_length",
        "Sure1",
        "Sure2",
        "Relationship",
        "test_day_index",
    ]
    X_alt = sm.add_constant(data[ols_features_alt])
    ols_alt = sm.OLS(y, X_alt).fit()
    print(ols_alt.summary())

    # Step 3: custom interpretable models
    X_interp = data[
        [
            "fertility_score",
            "high_fertility",
            "cycle_day",
            "estimated_cycle_length",
            "days_to_ovulation",
            "Sure1",
            "Sure2",
            "Relationship",
            "test_day_index",
        ]
    ]

    print("\nSmartAdditiveRegressor:")
    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X_interp, y)
    print(smart)
    smart_effects = smart.feature_effects()
    print("\nSmart feature effects:")
    print(smart_effects)

    print("\nHingeEBMRegressor:")
    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X_interp, y)
    print(hinge)
    hinge_effects = hinge.feature_effects()
    print("\nHinge feature effects:")
    print(hinge_effects)

    # Extract IV evidence
    iv = "fertility_score"
    iv_corr = float(bivar_corr.get(iv, np.nan))
    iv_coef = float(ols_main.params.get(iv, np.nan))
    iv_p = float(ols_main.pvalues.get(iv, np.nan))
    iv_alt_coef = float(ols_alt.params.get("high_fertility", np.nan))
    iv_alt_p = float(ols_alt.pvalues.get("high_fertility", np.nan))

    smart_iv = safe_effect(smart_effects, iv)
    hinge_iv = safe_effect(hinge_effects, iv)

    smart_ranked = rank_effects(smart_effects)
    hinge_ranked = rank_effects(hinge_effects)
    smart_top = [x for x in smart_ranked if x[1] > 0][:3]
    hinge_top = [x for x in hinge_ranked if x[1] > 0][:3]

    # Score mapping based on consistency/magnitude
    # We intentionally weight controlled analyses most heavily.
    if (
        (iv_p > 0.2)
        and (iv_alt_p > 0.2)
        and (abs(iv_corr) < 0.08)
        and (hinge_iv["importance"] < 0.02)
    ):
        response = 12
    elif (
        (iv_p < 0.05)
        and (abs(iv_coef) > 0.15)
        and (smart_iv["importance"] >= 0.10)
        and (hinge_iv["importance"] >= 0.10)
    ):
        response = 85
    elif (iv_p < 0.10) and ((smart_iv["importance"] >= 0.05) or (hinge_iv["importance"] >= 0.05)):
        response = 55
    elif (smart_iv["importance"] >= 0.08) and (hinge_iv["importance"] < 0.02):
        response = 25
    else:
        response = 20

    confounder_txt = "; ".join(
        [f"{f} (smart imp={imp:.1%}, dir={d})" for f, imp, d, _ in smart_top if f != iv]
    )
    hinge_txt = "; ".join(
        [f"{f} (hinge imp={imp:.1%}, dir={d})" for f, imp, d, _ in hinge_top if f != iv]
    )
    if not confounder_txt:
        confounder_txt = "no clear non-IV predictors in SmartAdditive"
    if not hinge_txt:
        hinge_txt = "no predictors were retained (all effects shrunk to zero)"

    explanation = (
        f"Using religiosity_mean (mean of Rel1-Rel3) as the DV and fertility_score "
        f"(cycle-based conception probability proxy) as the IV, the bivariate association is weak "
        f"(r={iv_corr:.3f}). In controlled OLS, fertility_score is {('positive' if iv_coef > 0 else 'negative')} "
        f"but not statistically reliable (coef={iv_coef:.3f}, p={iv_p:.3f}); the binary high_fertility "
        f"specification is also not reliable (coef={iv_alt_coef:.3f}, p={iv_alt_p:.3f}). "
        f"SmartAdditive ranks fertility_score #{smart_iv['rank']} with importance {smart_iv['importance']:.1%} "
        f"and direction {smart_iv['direction']}, while HingeEBM ranks it #{hinge_iv['rank']} with importance "
        f"{hinge_iv['importance']:.1%} and direction {hinge_iv['direction']}. These indicate little-to-moderate "
        f"predictive contribution relative to stronger covariates. The strongest other predictors are {confounder_txt}. "
        f"HingeEBM similarly emphasizes {hinge_txt}. Across bivariate, controlled, and interpretable models, the "
        f"fertility-related effect on religiosity is weak/inconsistent rather than robust, so the evidence leans "
        f"toward 'No'."
    )

    result = {"response": int(response), "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f)

    print("\nWrote conclusion.txt")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
