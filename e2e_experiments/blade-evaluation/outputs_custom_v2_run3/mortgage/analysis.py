import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from interp_models import HingeEBMRegressor, SmartAdditiveRegressor

warnings.filterwarnings("ignore")


def fmt(x, digits=4):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "nan"
    return f"{x:.{digits}f}"


def top_effects(effects, exclude=None, top_k=5):
    exclude = exclude or set()
    rows = []
    for name, meta in effects.items():
        if name in exclude:
            continue
        imp = float(meta.get("importance", 0.0) or 0.0)
        if imp <= 0:
            continue
        rows.append((name, meta.get("direction", "unknown"), imp, int(meta.get("rank", 0) or 0)))
    rows.sort(key=lambda r: (-r[2], r[0]))
    return rows[:top_k]


def main():
    with open("info.json", "r") as f:
        info = json.load(f)

    question = info.get("research_questions", [""])[0]
    print("Research question:", question)

    df = pd.read_csv("mortgage.csv")
    print("\nLoaded mortgage.csv with shape:", df.shape)

    # Research question mapping
    # IV: female (1=female), DV: accept (1=approved)
    iv = "female"
    dv = "accept"

    # Use numeric features, remove ID and leakage duplicate outcome variable.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_from_features = {dv, "deny", "Unnamed: 0"}
    feature_cols = [c for c in numeric_cols if c not in drop_from_features]

    model_cols = [dv] + feature_cols
    analysis_df = df[model_cols].dropna().copy()

    print("Rows after dropping missing values on model columns:", len(analysis_df))
    print("Dropped rows:", len(df) - len(analysis_df))

    # Step 1: exploration
    print("\n=== Step 1: Exploration ===")
    print("Summary stats (numeric columns):")
    print(analysis_df.describe().T)

    print("\nDistribution of DV (accept):")
    print(analysis_df[dv].value_counts(normalize=True).sort_index())

    print("\nDistribution of IV (female):")
    print(analysis_df[iv].value_counts(normalize=True).sort_index())

    corr_with_dv = analysis_df[[dv] + feature_cols].corr(numeric_only=True)[dv].sort_values(ascending=False)
    print("\nCorrelations with DV (accept):")
    print(corr_with_dv)

    group_means = analysis_df.groupby(iv)[dv].mean()
    print("\nBivariate approval rate by female:")
    print(group_means)

    # Bivariate models
    X_bi = sm.add_constant(analysis_df[[iv]], has_constant="add")
    logit_bi = sm.Logit(analysis_df[dv], X_bi).fit(disp=False, maxiter=200)
    ols_bi = sm.OLS(analysis_df[dv], X_bi).fit()

    print("\nBivariate Logit summary:")
    print(logit_bi.summary())
    print("\nBivariate OLS summary:")
    print(ols_bi.summary())

    # Step 2: controlled models
    print("\n=== Step 2: Controlled models ===")
    X_ctrl = sm.add_constant(analysis_df[feature_cols], has_constant="add")
    y = analysis_df[dv]

    logit_ctrl = sm.Logit(y, X_ctrl).fit(disp=False, maxiter=500)
    ols_ctrl = sm.OLS(y, X_ctrl).fit()

    print("Controlled Logit summary:")
    print(logit_ctrl.summary())
    print("\nControlled OLS summary:")
    print(ols_ctrl.summary())

    # Step 3: interpretable models
    print("\n=== Step 3: Interpretable models ===")
    X_interp = analysis_df[feature_cols]

    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X_interp, y)
    smart_effects = smart.feature_effects()

    print("SmartAdditiveRegressor model:")
    print(smart)
    print("\nSmartAdditive feature effects:")
    print(smart_effects)

    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X_interp, y)
    hinge_effects = hinge.feature_effects()

    print("\nHingeEBMRegressor model:")
    print(hinge)
    print("\nHingeEBM feature effects:")
    print(hinge_effects)

    # Pull key metrics for conclusion
    corr_iv_dv = float(analysis_df[[iv, dv]].corr().loc[iv, dv])

    bi_logit_coef = float(logit_bi.params.get(iv, np.nan))
    bi_logit_p = float(logit_bi.pvalues.get(iv, np.nan))
    bi_ols_coef = float(ols_bi.params.get(iv, np.nan))
    bi_ols_p = float(ols_bi.pvalues.get(iv, np.nan))

    ctrl_logit_coef = float(logit_ctrl.params.get(iv, np.nan))
    ctrl_logit_p = float(logit_ctrl.pvalues.get(iv, np.nan))
    ctrl_or = float(np.exp(ctrl_logit_coef))
    ctrl_ols_coef = float(ols_ctrl.params.get(iv, np.nan))
    ctrl_ols_p = float(ols_ctrl.pvalues.get(iv, np.nan))

    smart_iv = smart_effects.get(iv, {"direction": "zero", "importance": 0.0, "rank": 0})
    hinge_iv = hinge_effects.get(iv, {"direction": "zero", "importance": 0.0, "rank": 0})

    smart_top = top_effects(smart_effects, exclude={iv}, top_k=4)
    hinge_top = top_effects(hinge_effects, exclude={iv}, top_k=4)

    smart_top_text = ", ".join([
        f"{name} ({direction}, {imp*100:.1f}%, rank {rank})"
        for name, direction, imp, rank in smart_top
    ])
    hinge_top_text = ", ".join([
        f"{name} ({direction}, {imp*100:.1f}%, rank {rank})"
        for name, direction, imp, rank in hinge_top
    ])

    # Likert scoring rubric
    score = 50
    # Controlled evidence
    if ctrl_logit_p < 0.01 and ctrl_ols_p < 0.01:
        score += 12
    elif ctrl_logit_p < 0.05 and ctrl_ols_p < 0.05:
        score += 8
    elif ctrl_logit_p < 0.10 or ctrl_ols_p < 0.10:
        score += 4
    else:
        score -= 10

    # Bivariate evidence
    if bi_logit_p < 0.05 and bi_ols_p < 0.05:
        score += 10
    elif bi_logit_p > 0.10 and bi_ols_p > 0.10:
        score -= 8

    # Interpretable-model robustness
    smart_imp = float(smart_iv.get("importance", 0.0) or 0.0)
    hinge_imp = float(hinge_iv.get("importance", 0.0) or 0.0)
    if smart_imp >= 0.03:
        score += 8
    elif smart_imp < 0.01:
        score -= 7

    if hinge_imp >= 0.03:
        score += 6
    elif hinge_imp < 0.02:
        score -= 4

    # Direction consistency
    if ctrl_logit_coef > 0 and ctrl_ols_coef > 0 and "positive" in str(hinge_iv.get("direction", "")):
        score += 2

    score = int(np.clip(round(score), 0, 100))

    explanation = (
        f"Question: does gender (female=1) affect mortgage approval (accept=1)? "
        f"Bivariate evidence is essentially null: corr(female, accept)={fmt(corr_iv_dv,4)}, "
        f"bivariate logit coef={fmt(bi_logit_coef,3)} (p={fmt(bi_logit_p,3)}), and bivariate OLS coef={fmt(bi_ols_coef,3)} "
        f"(p={fmt(bi_ols_p,3)}). After controlling for financial and application covariates, female becomes a positive predictor: "
        f"controlled logit coef={fmt(ctrl_logit_coef,3)} (OR={fmt(ctrl_or,2)}, p={fmt(ctrl_logit_p,3)}) and controlled OLS coef={fmt(ctrl_ols_coef,3)} "
        f"(about {fmt(ctrl_ols_coef*100,1)} percentage-point higher approval, p={fmt(ctrl_ols_p,3)}). "
        f"Shape/magnitude from interpretable models is mixed: SmartAdditive assigns female {smart_iv.get('direction','zero')} effect with "
        f"importance={float(smart_iv.get('importance',0.0) or 0.0)*100:.1f}% (rank {smart_iv.get('rank',0)}), while HingeEBM gives a "
        f"{hinge_iv.get('direction','zero')} female effect with importance={float(hinge_iv.get('importance',0.0) or 0.0)*100:.1f}% "
        f"(rank {hinge_iv.get('rank',0)}). The strongest confounders are not gender: SmartAdditive highlights {smart_top_text}; "
        f"HingeEBM highlights {hinge_top_text}. Several top predictors are nonlinear in SmartAdditive (notably PI_ratio and loan_to_value with "
        f"decreasing threshold-like patterns), indicating credit-risk structure dominates approval decisions. Overall, gender shows a modest "
        f"positive controlled association, but weak/inconsistent importance across interpretable models, so evidence is partial rather than strong."
    )

    result = {"response": score, "explanation": explanation}

    with open("conclusion.txt", "w") as f:
        json.dump(result, f)

    print("\nWrote conclusion.txt:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
