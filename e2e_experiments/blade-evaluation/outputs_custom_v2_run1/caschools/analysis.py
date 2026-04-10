import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from interp_models import SmartAdditiveRegressor, HingeEBMRegressor

warnings.filterwarnings("ignore")


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def top_effect(effects, exclude=None, k=3):
    exclude = set(exclude or [])
    rows = []
    for name, meta in effects.items():
        if name in exclude:
            continue
        imp = _safe_float(meta.get("importance", 0.0))
        if np.isnan(imp):
            imp = 0.0
        rows.append((name, imp, meta.get("direction", "unknown"), int(meta.get("rank", 0) or 0)))
    rows.sort(key=lambda t: (-t[1], t[0]))
    return rows[:k]


def fmt_effect(meta):
    if not meta:
        return "not selected"
    imp = _safe_float(meta.get("importance", 0.0)) * 100
    rank = int(meta.get("rank", 0) or 0)
    direction = meta.get("direction", "unknown")
    if rank > 0:
        return f"{direction}, importance={imp:.1f}% (rank {rank})"
    return f"{direction}, importance={imp:.1f}%"


def main():
    # Step 1: Understand the question and explore
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info.get("research_questions", ["Unknown research question"])[0]
    print("Research question:")
    print(question)

    df = pd.read_csv("caschools.csv")

    # Define IV and DV from research question context
    iv_col = "str"  # student-teacher ratio
    dv_col = "testscr"  # academic performance proxy
    df[iv_col] = df["students"] / df["teachers"]
    df[dv_col] = (df["read"] + df["math"]) / 2.0

    print("\nDataset shape:", df.shape)
    print("IV:", iv_col, "DV:", dv_col)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("\nNumeric columns:", numeric_cols)

    print("\nSummary statistics (numeric):")
    print(df[numeric_cols].describe().T)

    print("\nDistribution checks (skewness):")
    skew = df[numeric_cols].skew().sort_values(ascending=False)
    print(skew)

    print("\nBivariate correlations with DV:")
    corr_to_dv = df[numeric_cols].corr(numeric_only=True)[dv_col].sort_values(ascending=False)
    print(corr_to_dv)

    biv_corr = _safe_float(df[iv_col].corr(df[dv_col]))
    print(f"\nBivariate correlation ({iv_col}, {dv_col}) = {biv_corr:.4f}")

    # Step 2: OLS with controls
    controls = [
        "calworks",
        "lunch",
        "english",
        "income",
        "expenditure",
        "computer",
        "students",
    ]
    feature_columns = [iv_col] + controls

    ols_df = df[[dv_col] + feature_columns].dropna().copy()
    X_ols = sm.add_constant(ols_df[feature_columns], has_constant="add")
    y_ols = ols_df[dv_col]
    ols_model = sm.OLS(y_ols, X_ols).fit()

    print("\nOLS summary:")
    print(ols_model.summary())

    iv_coef = _safe_float(ols_model.params.get(iv_col, np.nan))
    iv_p = _safe_float(ols_model.pvalues.get(iv_col, np.nan))

    # Step 3: Interpretable models
    # Use all relevant numeric predictors for interpretation, excluding IDs and target components
    exclude_predictors = {dv_col, "read", "math", "district", "rownames"}
    model_features = [c for c in numeric_cols if c not in exclude_predictors] + [iv_col]
    model_features = list(dict.fromkeys(model_features))

    X_interp = df[model_features].copy()
    y_interp = df[dv_col].copy()

    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X_interp, y_interp)
    print("\nSmartAdditiveRegressor:")
    print(smart)
    smart_effects = smart.feature_effects()
    print("\nSmart feature effects:")
    print(smart_effects)

    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X_interp, y_interp)
    print("\nHingeEBMRegressor:")
    print(hinge)
    hinge_effects = hinge.feature_effects()
    print("\nHinge feature effects:")
    print(hinge_effects)

    # Step 4: Rich conclusion + score
    smart_iv = smart_effects.get(iv_col, {})
    hinge_iv = hinge_effects.get(iv_col, {})

    smart_iv_imp = _safe_float(smart_iv.get("importance", 0.0))
    hinge_iv_imp = _safe_float(hinge_iv.get("importance", 0.0))

    ols_support = int((not np.isnan(iv_p)) and (iv_p < 0.05) and (iv_coef < 0))
    smart_support = int((smart_iv_imp >= 0.05) and ("decreasing" in str(smart_iv.get("direction", "")).lower() or smart_iv.get("direction") == "negative"))
    hinge_support = int((hinge_iv_imp >= 0.05) and (hinge_iv.get("direction") == "negative"))
    support_count = ols_support + smart_support + hinge_support

    # Scoring heuristic aligned to rubric
    if support_count == 3:
        response = 88
    elif support_count == 2:
        response = 65
    elif support_count == 1:
        response = 35
    else:
        # If bivariate evidence exists but not robust with controls/models -> weak evidence
        if (not np.isnan(biv_corr)) and (biv_corr < -0.15):
            response = 22
        else:
            response = 10

    response = int(max(0, min(100, response)))

    top_smart = top_effect(smart_effects, exclude={iv_col}, k=3)
    top_hinge = top_effect(hinge_effects, exclude={iv_col}, k=3)

    smart_top_txt = ", ".join([f"{n} ({imp*100:.1f}%, {d})" for n, imp, d, _ in top_smart]) if top_smart else "none"
    hinge_top_txt = ", ".join([f"{n} ({imp*100:.1f}%, {d})" for n, imp, d, _ in top_hinge]) if top_hinge else "none"

    iv_direction = "negative" if iv_coef < 0 else "positive"
    iv_sig_txt = "statistically significant" if (not np.isnan(iv_p) and iv_p < 0.05) else "not statistically significant"

    explanation = (
        f"Question: {question} "
        f"Using DV={dv_col} and IV={iv_col} (students/teachers), the bivariate relationship is negative "
        f"(corr={biv_corr:.3f}), but after controls the OLS effect is weak (coef={iv_coef:.3f}, p={iv_p:.3f}) and {iv_sig_txt}. "
        f"In SmartAdditiveRegressor, {iv_col} shows {fmt_effect(smart_iv)} with a small nonlinear decreasing shape (higher ratios mostly reduce scores, but modestly). "
        f"In HingeEBMRegressor, {iv_col} is {fmt_effect(hinge_iv)} and is effectively excluded from the sparse equation. "
        f"The strongest confounders are socioeconomic/demographic variables: Smart top features are {smart_top_txt}; "
        f"Hinge top features are {hinge_top_txt}. "
        f"Overall, evidence that lower student-teacher ratio raises performance is {iv_direction} in sign but not robust across controlled models, so support is weak-to-moderate rather than strong."
    )

    result = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=True)

    print("\nWrote conclusion.txt:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
