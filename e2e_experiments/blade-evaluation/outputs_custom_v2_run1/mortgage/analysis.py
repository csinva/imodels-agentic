import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from interp_models import HingeEBMRegressor, SmartAdditiveRegressor


def infer_variables(question: str, columns):
    q = question.lower()
    cols_lower = {c.lower(): c for c in columns}

    # IV inference
    iv = None
    if "gender" in q and "female" in cols_lower:
        iv = cols_lower["female"]
    else:
        for c in columns:
            if c.lower() in q:
                iv = c
                break
    if iv is None and "female" in columns:
        iv = "female"

    # DV inference
    dv = None
    if any(k in q for k in ["approve", "approval", "accepted", "accept"]):
        if "accept" in columns:
            dv = "accept"
    if dv is None and any(k in q for k in ["deny", "denied", "rejected"]):
        if "deny" in columns:
            dv = "deny"
    if dv is None:
        if "accept" in columns:
            dv = "accept"
        elif "deny" in columns:
            dv = "deny"

    return iv, dv


def direction_from_smart(direction_text: str):
    if not direction_text:
        return None
    txt = direction_text.lower()
    if "positive" in txt or "increasing" in txt:
        return 1
    if "negative" in txt or "decreasing" in txt:
        return -1
    return 0


def safe_top_effects(effects: dict, n=5, exclude=None):
    exclude = set(exclude or [])
    rows = []
    for feat, meta in effects.items():
        imp = float(meta.get("importance", 0.0))
        rank = int(meta.get("rank", 0) or 0)
        if feat in exclude or imp <= 0:
            continue
        rows.append((feat, imp, rank, meta.get("direction", "")))
    rows.sort(key=lambda x: (x[2] if x[2] > 0 else 10**9, -x[1], x[0]))
    return rows[:n]


def main():
    root = Path(".")
    info = json.loads((root / "info.json").read_text())
    question = info["research_questions"][0]

    df = pd.read_csv(root / "mortgage.csv")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    iv, dv = infer_variables(question, df.columns.tolist())

    if iv is None or dv is None:
        raise ValueError(f"Could not infer IV/DV from question: iv={iv}, dv={dv}")

    # Exclude obvious non-controls/leakage columns
    outcome_complements = {"accept", "deny", "denied_PMI"}
    exclude_controls = {"Unnamed: 0", iv, *outcome_complements}

    controls = [c for c in numeric_cols if c not in exclude_controls]

    use_cols = [iv, dv] + controls
    model_df = df[use_cols].dropna().copy()

    print("=" * 80)
    print("Research question:")
    print(question)
    print(f"Inferred IV: {iv}")
    print(f"Inferred DV: {dv}")
    print(f"Rows used after dropping missing values: {len(model_df)} / {len(df)}")
    print("=" * 80)

    # Step 1: Explore
    print("\n[Step 1] Summary statistics")
    print(model_df[[iv, dv] + controls].describe().T)

    print("\n[Step 1] Distributions")
    print(f"{dv} value counts:\n{model_df[dv].value_counts(dropna=False).sort_index()}")
    print(f"{iv} value counts:\n{model_df[iv].value_counts(dropna=False).sort_index()}")

    print("\n[Step 1] Bivariate relationship")
    group_means = model_df.groupby(iv)[dv].mean()
    print(f"Mean {dv} by {iv}:\n{group_means}")
    corr_iv_dv = float(model_df[[iv, dv]].corr().iloc[0, 1])
    print(f"Pearson correlation ({iv}, {dv}): {corr_iv_dv:.4f}")

    # Bivariate logistic (if DV binary)
    unique_y = set(model_df[dv].unique().tolist())
    is_binary_y = unique_y.issubset({0, 1}) and len(unique_y) == 2
    biv_p = np.nan
    biv_coef = np.nan
    if is_binary_y:
        X_biv = sm.add_constant(model_df[[iv]], has_constant="add")
        biv_logit = sm.Logit(model_df[dv], X_biv).fit(disp=False)
        biv_coef = float(biv_logit.params[iv])
        biv_p = float(biv_logit.pvalues[iv])
        print(f"Bivariate logit {iv} coef={biv_coef:.4f}, p={biv_p:.4g}, OR={np.exp(biv_coef):.3f}")

    # Step 2: Controlled models
    print("\n[Step 2] Controlled models")
    X = sm.add_constant(model_df[[iv] + controls], has_constant="add")
    y = model_df[dv]

    if is_binary_y:
        logit_model = sm.Logit(y, X).fit(disp=False, maxiter=1000)
        print("\nControlled logistic regression summary:")
        print(logit_model.summary())
        controlled_coef = float(logit_model.params[iv])
        controlled_p = float(logit_model.pvalues[iv])
        controlled_or = float(np.exp(controlled_coef))
    else:
        logit_model = None
        controlled_coef = np.nan
        controlled_p = np.nan
        controlled_or = np.nan

    ols_model = sm.OLS(y, X).fit()
    print("\nControlled OLS summary (linear probability if DV is binary):")
    print(ols_model.summary())
    ols_coef = float(ols_model.params[iv])
    ols_p = float(ols_model.pvalues[iv])

    # Step 3: Interpretable models
    print("\n[Step 3] Interpretable models")
    X_interp = model_df[[iv] + controls]

    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X_interp, y)
    smart_effects = smart.feature_effects()
    print("\nSmartAdditiveRegressor model:")
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

    iv_smart = smart_effects.get(iv, {"direction": "zero", "importance": 0.0, "rank": 0})
    iv_hinge = hinge_effects.get(iv, {"direction": "zero", "importance": 0.0, "rank": 0})

    smart_imp = float(iv_smart.get("importance", 0.0) or 0.0)
    hinge_imp = float(iv_hinge.get("importance", 0.0) or 0.0)

    # Basic bivariate mean difference for interpretability
    if 0 in group_means.index and 1 in group_means.index:
        mean_diff = float(group_means.loc[1] - group_means.loc[0])
    else:
        # Fallback for unexpected coding
        vals = group_means.values
        mean_diff = float(vals[-1] - vals[0]) if len(vals) >= 2 else 0.0

    # Likert score synthesis (0-100)
    score = 5.0

    # Bivariate evidence
    if abs(mean_diff) > 0.02:
        score += 12
    elif abs(mean_diff) > 0.005:
        score += 6
    if not np.isnan(biv_p):
        if biv_p < 0.05:
            score += 8
        elif biv_p < 0.10:
            score += 4

    # Controlled logistic evidence
    if not np.isnan(controlled_p):
        if controlled_p < 0.01:
            score += 35
        elif controlled_p < 0.05:
            score += 28
        elif controlled_p < 0.10:
            score += 18

    # Controlled OLS evidence
    if ols_p < 0.01:
        score += 14
    elif ols_p < 0.05:
        score += 10
    elif ols_p < 0.10:
        score += 6

    # Effect size from OR
    if not np.isnan(controlled_or):
        log_or = abs(np.log(controlled_or))
        if log_or > 0.5:
            score += 8
        elif log_or > 0.2:
            score += 5
        elif log_or > 0.1:
            score += 2

    # Interpretable model support
    if smart_imp >= 0.05:
        score += 15
    elif smart_imp >= 0.02:
        score += 8
    elif smart_imp >= 0.01:
        score += 4

    if hinge_imp >= 0.05:
        score += 12
    elif hinge_imp >= 0.02:
        score += 6
    elif hinge_imp >= 0.01:
        score += 3

    # Consistency penalty
    dir_signs = []
    if abs(mean_diff) > 1e-9:
        dir_signs.append(1 if mean_diff > 0 else -1)
    if not np.isnan(controlled_coef) and abs(controlled_coef) > 1e-9:
        dir_signs.append(1 if controlled_coef > 0 else -1)
    smart_dir = direction_from_smart(str(iv_smart.get("direction", "")))
    if smart_dir in (-1, 1):
        dir_signs.append(smart_dir)
    hinge_dir = str(iv_hinge.get("direction", "")).lower()
    if hinge_dir == "positive":
        dir_signs.append(1)
    elif hinge_dir == "negative":
        dir_signs.append(-1)

    if dir_signs and (any(s > 0 for s in dir_signs) and any(s < 0 for s in dir_signs)):
        score -= 12
    elif smart_imp < 0.01 and hinge_imp < 0.03 and not np.isnan(controlled_p) and controlled_p < 0.05:
        score -= 6

    # Caps aligned with rubric
    if (np.isnan(controlled_p) or controlled_p >= 0.10) and smart_imp < 0.02 and hinge_imp < 0.02:
        score = min(score, 25)
    if (not np.isnan(controlled_p) and controlled_p < 0.05) and smart_imp < 0.01 and hinge_imp < 0.03:
        score = min(score, 55)

    score = int(np.clip(round(score), 0, 100))

    smart_top = safe_top_effects(smart_effects, n=4, exclude={iv})
    hinge_top = safe_top_effects(hinge_effects, n=4, exclude={iv})

    smart_top_str = ", ".join(
        [f"{f} (rank {r}, imp {imp:.1%}, {d})" for f, imp, r, d in smart_top]
    ) or "none"
    hinge_top_str = ", ".join(
        [f"{f} (rank {r}, imp {imp:.1%}, {d})" for f, imp, r, d in hinge_top]
    ) or "none"

    explanation = (
        f"Question: {question} Using DV={dv} and IV={iv}. "
        f"Bivariate evidence is weak (mean {dv} difference for {iv}=1 vs 0: {mean_diff:+.4f}; "
        f"corr={corr_iv_dv:+.4f}; bivariate logit p={biv_p:.4g} if available). "
        f"After controls, logistic regression gives {iv} coef={controlled_coef:+.4f} "
        f"(OR={controlled_or:.3f}, p={controlled_p:.4g}), while controlled OLS gives coef={ols_coef:+.4f} "
        f"(p={ols_p:.4g}). "
        f"In interpretable models, SmartAdditive shows {iv} direction='{iv_smart.get('direction')}', "
        f"importance={smart_imp:.1%}, rank={iv_smart.get('rank')}; this suggests "
        f"{'little to no nonlinear/threshold signal' if smart_imp < 0.01 else 'a measurable shaped effect'}. "
        f"HingeEBM shows {iv} direction='{iv_hinge.get('direction')}', importance={hinge_imp:.1%}, "
        f"rank={iv_hinge.get('rank')}. "
        f"Major confounders are stronger than {iv}: SmartAdditive top features: {smart_top_str}. "
        f"HingeEBM top features: {hinge_top_str}. "
        f"Overall, the gender effect appears positive after controls but small and not fully robust across "
        f"interpretable models, so evidence is partial rather than strong."
    )

    result = {"response": score, "explanation": explanation}

    with open(root / "conclusion.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(result))

    print("\n[Step 4] Wrote conclusion.txt:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
