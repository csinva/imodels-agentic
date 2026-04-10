import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr

from interp_models import SmartAdditiveRegressor, HingeEBMRegressor


def _to_native(x):
    if isinstance(x, (np.generic,)):
        return x.item()
    return x


def _clean_effects(effects):
    cleaned = {}
    for feat, vals in effects.items():
        cleaned[feat] = {k: _to_native(v) for k, v in vals.items()}
    return cleaned


def main():
    info_path = Path("info.json")
    data_path = Path("fish.csv")

    info = json.loads(info_path.read_text())
    df = pd.read_csv(data_path)

    question = info.get("research_questions", ["Unknown question"])[0]
    print("Research question:")
    print(question)
    print()

    # Identify variables for this task.
    dv = "fish_caught" if "fish_caught" in df.columns else df.columns[0]
    iv = "hours" if "hours" in df.columns else [c for c in df.columns if c != dv][0]

    print(f"Dependent variable (DV): {dv}")
    print(f"Primary independent variable (IV): {iv}")
    print()

    # Step 1: explore
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    print("=== Step 1: Summary statistics ===")
    print(df[numeric_cols].describe().T)
    print()

    print("=== Step 1: Distribution diagnostics (skewness) ===")
    print(df[numeric_cols].skew().sort_values(ascending=False))
    print()

    print("=== Step 1: Correlation matrix ===")
    print(df[numeric_cols].corr())
    print()

    r, r_p = pearsonr(df[iv], df[dv])
    print(f"Bivariate Pearson correlation ({iv}, {dv}): r={r:.4f}, p={r_p:.4g}")
    print()

    # Simple model for raw relationship
    X_simple = sm.add_constant(df[[iv]])
    model_simple = sm.OLS(df[dv], X_simple).fit()
    print("=== Simple OLS (DV ~ IV) ===")
    print(model_simple.summary())
    print()

    # Step 2: controlled model
    controls = [c for c in df.columns if c != dv]
    X_full = sm.add_constant(df[controls])

    unique_dv = set(df[dv].dropna().unique().tolist())
    is_binary_dv = unique_dv.issubset({0, 1}) and len(unique_dv) <= 2

    if is_binary_dv:
        model_full = sm.Logit(df[dv], X_full).fit(disp=False)
        model_type = "Logit"
    else:
        model_full = sm.OLS(df[dv], X_full).fit()
        model_type = "OLS"

    print(f"=== Step 2: Controlled {model_type} ({dv} ~ all predictors) ===")
    print(model_full.summary())
    print()

    # Step 3: custom interpretable models
    feature_cols = [c for c in numeric_cols if c != dv]
    X_num = df[feature_cols]
    y = df[dv]

    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X_num, y)
    smart_effects = _clean_effects(smart.feature_effects())

    print("=== Step 3: SmartAdditiveRegressor ===")
    print(smart)
    print("Feature effects:")
    print(smart_effects)
    print()

    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X_num, y)
    hinge_effects = _clean_effects(hinge.feature_effects())

    print("=== Step 3: HingeEBMRegressor ===")
    print(hinge)
    print("Feature effects:")
    print(hinge_effects)
    print()

    # Collect metrics for conclusion
    simple_coef = float(model_simple.params.get(iv, np.nan))
    simple_p = float(model_simple.pvalues.get(iv, np.nan))
    full_coef = float(model_full.params.get(iv, np.nan))
    full_p = float(model_full.pvalues.get(iv, np.nan))

    smart_iv = smart_effects.get(iv, {"direction": "unknown", "importance": 0.0, "rank": 0})
    hinge_iv = hinge_effects.get(iv, {"direction": "unknown", "importance": 0.0, "rank": 0})

    # Shape details from SmartAdditive thresholds.
    shape_note = ""
    if hasattr(smart, "shape_functions_") and iv in feature_cols:
        iv_idx = feature_cols.index(iv)
        if iv_idx in smart.shape_functions_:
            thresholds, intervals = smart.shape_functions_[iv_idx]
            thresholds = np.array(thresholds, dtype=float)
            intervals = np.array(intervals, dtype=float)
            if len(thresholds) > 0 and len(intervals) > 1:
                pos_idx = np.where(intervals > 0)[0]
                if len(pos_idx) > 0:
                    first_pos = int(pos_idx[0])
                    if first_pos == 0:
                        cross_text = f"effect already positive at very low {iv}"
                    else:
                        cross_text = f"effect turns positive around {iv}>{thresholds[first_pos - 1]:.2f}"
                else:
                    cross_text = "effect does not turn positive in observed range"

                diffs = np.diff(intervals)
                if len(diffs) > 0:
                    j = int(np.argmax(diffs))
                    jump_t = thresholds[j] if j < len(thresholds) else thresholds[-1]
                    shape_note = (
                        f"SmartAdditive shows a nonlinear increasing pattern; {cross_text}, "
                        f"with the largest upward step near {iv}≈{jump_t:.2f}."
                    )
                else:
                    shape_note = f"SmartAdditive shows a nonlinear increasing pattern; {cross_text}."

    # Confounders from controlled model
    pvals = model_full.pvalues.drop(labels=["const"], errors="ignore")
    coefs = model_full.params.drop(labels=["const"], errors="ignore")
    control_items = []
    for col in controls:
        if col == iv:
            continue
        if col in pvals.index:
            control_items.append((col, float(coefs[col]), float(pvals[col])))

    # Most influential significant controls
    sig_controls = [x for x in control_items if x[2] < 0.05]
    sig_controls.sort(key=lambda t: t[2])
    conf_text_parts = []
    for col, coef, p in sig_controls[:3]:
        direction = "positive" if coef > 0 else "negative"
        conf_text_parts.append(f"{col} ({direction}, coef={coef:.3f}, p={p:.3g})")
    conf_text = "; ".join(conf_text_parts) if conf_text_parts else "no strong control variables at p<0.05"

    # Fish/hour descriptive rates
    fish_per_hour = df[dv] / df[iv]
    avg_ratio = float(np.mean(fish_per_hour.replace([np.inf, -np.inf], np.nan).dropna()))
    weighted_ratio = float(df[dv].sum() / df[iv].sum())

    # Score strength of evidence for IV effect on DV
    score = 0

    # Controlled model gets highest weight
    if full_p < 0.01:
        score += 45
    elif full_p < 0.05:
        score += 35
    elif full_p < 0.10:
        score += 25
    else:
        score += 10 if full_coef > 0 else 0

    # Bivariate evidence
    if simple_p < 0.01:
        score += 20
    elif simple_p < 0.05:
        score += 15
    elif simple_p < 0.10:
        score += 8

    # Interpretable models
    smart_imp = float(smart_iv.get("importance", 0.0) or 0.0)
    hinge_imp = float(hinge_iv.get("importance", 0.0) or 0.0)
    smart_dir = str(smart_iv.get("direction", "")).lower()
    hinge_dir = str(hinge_iv.get("direction", "")).lower()

    score += min(20, int(round(30 * smart_imp)))
    score += min(15, int(round(20 * hinge_imp)))

    if "positive" in smart_dir or "increasing" in smart_dir:
        score += 5
    elif "negative" in smart_dir or "decreasing" in smart_dir:
        score -= 5

    if "positive" in hinge_dir:
        score += 3
    elif "negative" in hinge_dir:
        score -= 3

    # Penalize weak controlled evidence
    if 0.05 <= full_p < 0.10:
        score -= 7
    elif full_p >= 0.10:
        score -= 15

    response = int(np.clip(score, 0, 100))

    explanation = (
        f"Primary test of {iv} on {dv}: bivariate association is positive (r={r:.3f}, p={r_p:.4g}); "
        f"simple OLS gives coef={simple_coef:.3f} fish/hour (p={simple_p:.4g}). "
        f"With controls (livebait, camper, persons, child), the {iv} coefficient stays positive but weakens "
        f"to coef={full_coef:.3f} (p={full_p:.4g}), so evidence is moderate rather than definitive. "
        f"Magnitude/importance across interpretable models: SmartAdditive rank={smart_iv.get('rank', 0)} "
        f"importance={smart_imp:.1%} direction={smart_iv.get('direction', 'unknown')}; "
        f"HingeEBM rank={hinge_iv.get('rank', 0)} importance={hinge_imp:.1%} direction={hinge_iv.get('direction', 'unknown')}. "
        f"{shape_note} Confounders that materially affect catch include {conf_text}. "
        f"For rate context, mean individual fish/hour is {avg_ratio:.3f} (skewed), while total fish/total hours is {weighted_ratio:.3f}. "
        f"Overall: {iv} likely increases catches, but part of the raw effect is explained by group composition and trip characteristics."
    )

    conclusion = {"response": response, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(conclusion))

    print("=== Step 4: Conclusion JSON ===")
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()
