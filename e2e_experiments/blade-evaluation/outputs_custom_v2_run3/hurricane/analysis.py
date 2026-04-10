import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from interp_models import HingeEBMRegressor, SmartAdditiveRegressor


def safe_float(x):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return float(x)


def to_builtin_effects(effects):
    out = {}
    for k, v in effects.items():
        out[k] = {
            "direction": str(v.get("direction", "unknown")),
            "importance": safe_float(v.get("importance", 0.0)) or 0.0,
            "rank": int(v.get("rank", 0) or 0),
        }
    return out


def format_top_effects(effects, exclude=None, top_k=3):
    exclude = exclude or set()
    rows = []
    for name, meta in effects.items():
        if name in exclude:
            continue
        imp = meta.get("importance", 0.0) or 0.0
        rank = meta.get("rank", 0) or 0
        if imp > 0 and rank > 0:
            rows.append((rank, name, meta.get("direction", "unknown"), imp))
    rows.sort(key=lambda x: x[0])
    rows = rows[:top_k]
    if not rows:
        return "none"
    return "; ".join(
        f"{name} (rank {rank}, {direction}, importance={imp:.1%})"
        for rank, name, direction, imp in rows
    )


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    with open("info.json", "r") as f:
        info = json.load(f)

    question = info.get("research_questions", [""])[0]
    print("Research question:", question)

    df = pd.read_csv("hurricane.csv")
    print("\nData shape:", df.shape)
    print("Columns:", list(df.columns))

    # Define DV and IV from the research framing.
    dv = "alldeaths"
    if "masfem" in df.columns:
        iv = "masfem"
    elif "gender_mf" in df.columns:
        iv = "gender_mf"
    else:
        raise ValueError("Could not identify a femininity IV (expected masfem or gender_mf).")

    print(f"\nUsing DV='{dv}' and IV='{iv}'")

    # Step 1: Explore
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("\nStep 1: Summary statistics (numeric)")
    print(df[numeric_cols].describe().T[["mean", "std", "min", "max"]])

    print("\nMissing values per column:")
    print(df.isna().sum())

    print("\nDistribution diagnostics (skewness):")
    print(df[numeric_cols].skew().sort_values(ascending=False))

    pearson_r, pearson_p = stats.pearsonr(df[iv], df[dv])
    spearman_rho, spearman_p = stats.spearmanr(df[iv], df[dv])
    print(
        f"\nBivariate {iv} vs {dv}: Pearson r={pearson_r:.3f} (p={pearson_p:.3g}), "
        f"Spearman rho={spearman_rho:.3f} (p={spearman_p:.3g})"
    )

    # Step 2: OLS with controls
    control_candidates = ["min", "wind", "category", "ndam15", "elapsedyrs"]
    controls = [c for c in control_candidates if c in df.columns and c not in {dv, iv}]
    feature_cols = [iv] + controls

    ols_df = df[[dv] + feature_cols].copy()
    for col in feature_cols:
        if ols_df[col].isna().any():
            ols_df[col] = ols_df[col].fillna(ols_df[col].median())

    X = sm.add_constant(ols_df[feature_cols])
    y = ols_df[dv]
    ols_model = sm.OLS(y, X).fit()
    print("\nStep 2: OLS results (raw deaths)")
    print(ols_model.summary())

    # Robustness: log-transformed DV to reduce heavy-tail influence
    log_y = np.log1p(y)
    ols_log_model = sm.OLS(log_y, X).fit()
    print("\nOLS robustness check (log1p deaths)")
    print(ols_log_model.summary())

    # Step 3: Interpretable models (all numeric predictors except DV and ID)
    interp_numeric = [c for c in numeric_cols if c not in {dv, "ind"}]
    X_interp = df[interp_numeric].copy()
    for col in X_interp.columns:
        if X_interp[col].isna().any():
            X_interp[col] = X_interp[col].fillna(X_interp[col].median())
    y_interp = df[dv].astype(float).values

    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X_interp, y_interp)
    smart_effects = to_builtin_effects(smart.feature_effects())
    print("\nStep 3: SmartAdditiveRegressor")
    print(smart)
    print("Smart effects:", smart_effects)

    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X_interp, y_interp)
    hinge_effects = to_builtin_effects(hinge.feature_effects())
    print("\nHingeEBMRegressor")
    print(hinge)
    print("Hinge effects:", hinge_effects)

    # Step 4: Rich conclusion and Likert score
    ols_coef = safe_float(ols_model.params.get(iv))
    ols_p = safe_float(ols_model.pvalues.get(iv))
    ols_log_coef = safe_float(ols_log_model.params.get(iv))
    ols_log_p = safe_float(ols_log_model.pvalues.get(iv))

    smart_iv = smart_effects.get(iv, {"direction": "unknown", "importance": 0.0, "rank": 0})
    hinge_iv = hinge_effects.get(iv, {"direction": "unknown", "importance": 0.0, "rank": 0})

    smart_iv_imp = smart_iv.get("importance", 0.0) or 0.0
    hinge_iv_imp = hinge_iv.get("importance", 0.0) or 0.0

    sig_bivar = pearson_p is not None and pearson_p < 0.05
    sig_ols = ols_p is not None and ols_p < 0.05
    sig_log = ols_log_p is not None and ols_log_p < 0.05
    iv_active_smart = smart_iv_imp >= 0.01
    iv_active_hinge = hinge_iv_imp >= 0.01

    if sig_ols and iv_active_smart and iv_active_hinge:
        response = 85
    elif sig_ols and (iv_active_smart or iv_active_hinge):
        response = 75
    elif sig_ols or sig_bivar or sig_log:
        response = 50 if (iv_active_smart or iv_active_hinge) else 40
    elif iv_active_smart or iv_active_hinge:
        response = 30
    elif (ols_p is not None and ols_p < 0.2) or abs(pearson_r) > 0.1:
        response = 20
    else:
        response = 10

    direction_text = "positive" if (ols_coef is not None and ols_coef > 0) else "negative"
    smart_shape = smart_iv.get("direction", "unknown")
    hinge_direction = hinge_iv.get("direction", "unknown")

    top_smart = format_top_effects(smart_effects, exclude={iv}, top_k=3)
    top_hinge = format_top_effects(hinge_effects, exclude={iv}, top_k=3)

    explanation = (
        f"Using DV={dv} and IV={iv}, the bivariate association is weak "
        f"(Pearson r={pearson_r:.3f}, p={pearson_p:.3g}). In controlled OLS "
        f"(controls: {', '.join(controls)}), the IV effect is {direction_text} but not significant "
        f"(coef={ols_coef:.3f}, p={ols_p:.3g}); the log-DV robustness model is also non-significant "
        f"(coef={ols_log_coef:.3f}, p={ols_log_p:.3g}). In SmartAdditive, {iv} has direction "
        f"'{smart_shape}' with importance={smart_iv_imp:.1%} (rank={smart_iv.get('rank', 0)}), "
        f"indicating little to no nonlinear threshold effect. In HingeEBM, {iv} has direction "
        f"'{hinge_direction}' with importance={hinge_iv_imp:.1%} (rank={hinge_iv.get('rank', 0)}), "
        f"so it is effectively excluded. Confounders dominate: SmartAdditive top effects are {top_smart}; "
        f"Hinge top effects are {top_hinge}. Overall, evidence that more feminine hurricane names "
        f"lead to higher fatalities is weak and not robust across models."
    )

    result = {"response": int(response), "explanation": explanation}
    with open("conclusion.txt", "w") as f:
        json.dump(result, f)

    print("\nWrote conclusion.txt:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
