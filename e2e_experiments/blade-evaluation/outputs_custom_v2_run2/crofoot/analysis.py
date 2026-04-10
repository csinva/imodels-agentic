import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pointbiserialr

from interp_models import HingeEBMRegressor, SmartAdditiveRegressor

warnings.filterwarnings("ignore")


def to_py(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def safe_effect(effects, name):
    return effects.get(name, {"direction": "zero", "importance": 0.0, "rank": 0})


def summarize_thresholds(model, feature_name):
    if feature_name not in model.feature_names_:
        return "shape unavailable"
    j = model.feature_names_.index(feature_name)
    if j not in model.shape_functions_:
        return "approximately no learned nonlinear shape"

    thresholds, intervals = model.shape_functions_[j]
    if len(intervals) == 0:
        return "approximately no learned nonlinear shape"

    low = float(intervals[0])
    high = float(intervals[-1])
    if len(thresholds) > 0:
        first_t = float(thresholds[0])
        last_t = float(thresholds[-1])
        return (
            f"nonlinear with thresholds from about {first_t:.1f} to {last_t:.1f}; "
            f"effect shifts from {low:+.3f} (low values) to {high:+.3f} (high values)"
        )
    return f"nonlinear with effect from {low:+.3f} to {high:+.3f}"


def var_score(var_name, logit_p, smart_imp, hinge_imp):
    # 0-100 evidence score for whether this variable has a robust effect
    score = 50

    if logit_p < 0.05:
        score += 20
    elif logit_p < 0.10:
        score += 10
    else:
        score -= 10

    if smart_imp > 0.20:
        score += 15
    elif smart_imp > 0.10:
        score += 8
    elif smart_imp > 0.03:
        score += 3
    else:
        score -= 6

    if hinge_imp > 0.20:
        score += 10
    elif hinge_imp > 0.05:
        score += 5
    elif hinge_imp > 0.0:
        score += 1
    else:
        score -= 8

    return int(np.clip(round(score), 0, 100))


def main():
    # Step 1: Understand question + explore
    with open("info.json", "r") as f:
        info = json.load(f)

    question = info["research_questions"][0]
    print("Research question:")
    print(question)

    df = pd.read_csv("crofoot.csv")

    dv = "win"
    iv_size = "rel_group_size"
    iv_loc = "rel_location_adv"

    # Engineer IVs directly from metadata definitions
    df[iv_size] = df["n_focal"] - df["n_other"]
    # Positive means contest is closer to focal center than the other center
    df[iv_loc] = df["dist_other"] - df["dist_focal"]

    print("\nDataset shape:", df.shape)
    print("\nSummary statistics:")
    print(df.describe(include="all").T)

    print("\nDV distribution (win):")
    print(df[dv].value_counts(dropna=False).sort_index())

    print("\nIV distributions:")
    print(df[[iv_size, iv_loc]].describe().T)

    print("\nBivariate correlations with DV (point-biserial):")
    corr_rows = []
    for col in [iv_size, iv_loc, "dist_focal", "dist_other", "n_focal", "n_other"]:
        r, p = pointbiserialr(df[dv], df[col])
        corr_rows.append((col, float(r), float(p)))
    corr_df = pd.DataFrame(corr_rows, columns=["variable", "correlation", "p_value"]).sort_values(
        "p_value"
    )
    print(corr_df.to_string(index=False))

    # Step 2: Logistic regression with controls (binary DV)
    # Controls chosen to avoid exact multicollinearity while adjusting for group identity and base context.
    feature_columns = [iv_size, iv_loc, "n_other", "dist_focal", "focal", "other"]
    X = sm.add_constant(df[feature_columns])
    y = df[dv]

    logit_model = sm.Logit(y, X).fit(disp=False)
    print("\nLogistic regression with controls:")
    print(logit_model.summary())

    coef = logit_model.params
    pvals = logit_model.pvalues
    odds = np.exp(coef)

    # Step 3: Interpretable models
    numeric_columns = [c for c in df.select_dtypes(include=[np.number]).columns if c != dv]
    X_num = df[numeric_columns]

    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X_num, y)
    smart_effects = smart.feature_effects()

    print("\nSmartAdditiveRegressor model:")
    print(smart)
    print("\nSmartAdditive feature effects:")
    print(smart_effects)

    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X_num, y)
    hinge_effects = hinge.feature_effects()

    print("\nHingeEBMRegressor model:")
    print(hinge)
    print("\nHingeEBM feature effects:")
    print(hinge_effects)

    # Pull evidence for key IVs
    size_smart = safe_effect(smart_effects, iv_size)
    loc_smart = safe_effect(smart_effects, iv_loc)
    size_hinge = safe_effect(hinge_effects, iv_size)
    loc_hinge = safe_effect(hinge_effects, iv_loc)

    size_score = var_score(
        iv_size,
        float(pvals[iv_size]),
        float(to_py(size_smart.get("importance", 0.0))),
        float(to_py(size_hinge.get("importance", 0.0))),
    )
    loc_score = var_score(
        iv_loc,
        float(pvals[iv_loc]),
        float(to_py(loc_smart.get("importance", 0.0))),
        float(to_py(loc_hinge.get("importance", 0.0))),
    )

    response = int(np.clip(round((size_score + loc_score) / 2), 0, 100))

    # Identify major confounders from Smart/Hinge excluding IVs
    smart_ranked = sorted(
        [(k, v) for k, v in smart_effects.items() if k not in {iv_size, iv_loc}],
        key=lambda x: float(to_py(x[1].get("importance", 0.0))),
        reverse=True,
    )
    hinge_ranked = sorted(
        [(k, v) for k, v in hinge_effects.items() if k not in {iv_size, iv_loc}],
        key=lambda x: float(to_py(x[1].get("importance", 0.0))),
        reverse=True,
    )

    top_smart = [f"{k} ({100*float(to_py(v['importance'])):.1f}%)" for k, v in smart_ranked[:3] if float(to_py(v.get("importance", 0.0))) > 0]
    top_hinge = [f"{k} ({100*float(to_py(v['importance'])):.1f}%)" for k, v in hinge_ranked[:3] if float(to_py(v.get("importance", 0.0))) > 0]

    size_shape = summarize_thresholds(smart, iv_size)
    loc_shape = summarize_thresholds(smart, iv_loc)

    explanation = (
        f"DV is win (focal group victory) with IVs relative group size ({iv_size}) and contest location advantage ({iv_loc}=dist_other-dist_focal). "
        f"Bivariate evidence is weak: corr(win,{iv_size})={corr_df.loc[corr_df['variable']==iv_size,'correlation'].iloc[0]:.3f} "
        f"(p={corr_df.loc[corr_df['variable']==iv_size,'p_value'].iloc[0]:.3f}) and corr(win,{iv_loc})={corr_df.loc[corr_df['variable']==iv_loc,'correlation'].iloc[0]:.3f} "
        f"(p={corr_df.loc[corr_df['variable']==iv_loc,'p_value'].iloc[0]:.3f}). "
        f"In controlled logistic regression, {iv_size} is positive but not significant (coef={coef[iv_size]:.3f}, OR={odds[iv_size]:.3f}, p={pvals[iv_size]:.3f}), "
        f"while {iv_loc} is slightly negative and not significant (coef={coef[iv_loc]:.4f}, OR={odds[iv_loc]:.4f}, p={pvals[iv_loc]:.3f}). "
        f"SmartAdditive shows richer nonlinearity: {iv_size} has {size_shape} with modest importance "
        f"({100*float(to_py(size_smart.get('importance',0))):.1f}%, rank {to_py(size_smart.get('rank',0))}); "
        f"{iv_loc} has {loc_shape} with larger importance ({100*float(to_py(loc_smart.get('importance',0))):.1f}%, "
        f"rank {to_py(loc_smart.get('rank',0))}), indicating threshold-like home-field effects. "
        f"However, HingeEBM (sparse model) zeroes out both IVs (importance 0% each), so evidence is not robust across model classes. "
        f"Confounders matter: SmartAdditive ranks {', '.join(top_smart) if top_smart else 'none'} highest, and HingeEBM emphasizes "
        f"{', '.join(top_hinge) if top_hinge else 'none'}. Overall, contest location shows some nonlinear signal but relative size is weak; "
        f"because effects are inconsistent after controls and sparsity selection, the answer is only weak-to-moderate Yes."
    )

    result = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w") as f:
        json.dump(result, f)

    print("\nWrote conclusion.txt:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
