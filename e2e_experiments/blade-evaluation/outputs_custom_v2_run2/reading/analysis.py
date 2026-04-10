import json
import re
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from interp_models import HingeEBMRegressor, SmartAdditiveRegressor

warnings.filterwarnings("ignore")


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def infer_iv_dv(question: str, columns: List[str]) -> Tuple[str, str]:
    q = _normalize(question)
    cols = list(columns)

    iv = None
    if "reader view" in q and "reader_view" in cols:
        iv = "reader_view"
    else:
        for c in cols:
            c_norm = _normalize(c)
            if c_norm and c_norm in q:
                iv = c
                break
    if iv is None:
        iv = "reader_view" if "reader_view" in cols else cols[0]

    dv = None
    speed_candidates = [c for c in cols if any(k in c.lower() for k in ["speed", "wpm", "words_per_min"])]
    if "reading speed" in q and speed_candidates:
        non_iv = [c for c in speed_candidates if c != iv]
        dv = non_iv[0] if non_iv else speed_candidates[0]

    if dv is None:
        non_iv_speed = [c for c in speed_candidates if c != iv]
        if non_iv_speed:
            dv = non_iv_speed[0]

    if dv is None:
        numeric_like = [c for c in cols if c != iv]
        dv = numeric_like[-1]

    return iv, dv


def format_p(p: float) -> str:
    if pd.isna(p):
        return "nan"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"


def top_effects(effects: Dict[str, Dict], exclude: str, k: int = 3) -> List[Tuple[str, Dict]]:
    rows = [(f, e) for f, e in effects.items() if f != exclude and e.get("importance", 0) > 0]
    rows.sort(key=lambda x: x[1].get("importance", 0), reverse=True)
    return rows[:k]


def smart_shape_summary(model: SmartAdditiveRegressor, feature_name: str, feature_order: List[str]) -> str:
    if feature_name not in feature_order:
        return "shape unavailable"
    j = feature_order.index(feature_name)
    if not hasattr(model, "shape_functions_") or j not in model.shape_functions_:
        return "no meaningful learned shape"

    thresholds, intervals = model.shape_functions_[j]
    slope, _, r2 = model.linear_approx_.get(j, (0.0, 0.0, 1.0))

    if r2 > 0.90:
        return f"approximately linear (slope={slope:.4f})"

    if len(thresholds) == 0 or len(intervals) < 2:
        return "nonlinear but weakly structured"

    diffs = np.diff(intervals)
    if len(diffs) == 0:
        return "nonlinear but weakly structured"

    max_idx = int(np.argmax(np.abs(diffs)))
    t = thresholds[max_idx]
    delta = diffs[max_idx]
    trend = "increase" if delta > 0 else "decrease"
    return f"nonlinear with strongest step near {feature_name}={t:.3f} ({trend}, delta={delta:.4f})"


def score_effect(
    biv_coef: float,
    biv_p: float,
    ctrl_coef: float,
    ctrl_p: float,
    smart_dir: str,
    smart_imp: float,
    hinge_dir: str,
    hinge_imp: float,
) -> int:
    score = 10

    # Bivariate evidence
    if biv_coef > 0 and biv_p < 0.05:
        score += 20
    elif biv_coef > 0 and biv_p < 0.10:
        score += 14
    elif biv_coef > 0:
        score += 8
    elif biv_coef < 0 and biv_p < 0.05:
        score -= 12

    # Controlled evidence (stronger weight)
    if ctrl_coef > 0 and ctrl_p < 0.05:
        score += 30
    elif ctrl_coef > 0 and ctrl_p < 0.10:
        score += 20
    elif ctrl_coef > 0:
        score += 8
    elif ctrl_coef < 0 and ctrl_p < 0.05:
        score -= 15

    # SmartAdditive evidence
    if ("positive" in smart_dir or "increasing" in smart_dir) and smart_imp >= 0.05:
        score += 20
    elif ("positive" in smart_dir or "increasing" in smart_dir) and smart_imp >= 0.01:
        score += 12
    elif smart_imp == 0:
        score += 0
    elif "negative" in smart_dir or "decreasing" in smart_dir:
        score -= 8

    # HingeEBM evidence
    if hinge_dir == "positive" and hinge_imp >= 0.05:
        score += 20
    elif hinge_dir == "positive" and hinge_imp >= 0.01:
        score += 12
    elif hinge_imp == 0:
        score += 0
    elif hinge_dir == "negative":
        score -= 8

    return int(np.clip(round(score), 0, 100))


def main() -> None:
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info.get("research_questions", ["Unknown question"])[0]
    df = pd.read_csv("reading.csv")

    iv, dv = infer_iv_dv(question, df.columns.tolist())

    print("=" * 80)
    print("Research question:")
    print(question)
    print(f"\nIdentified IV: {iv}")
    print(f"Identified DV: {dv}")
    print("=" * 80)

    # Step 1: Exploration
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("\nStep 1: Summary statistics for numeric variables")
    print(df[numeric_cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]])

    print("\nDV distribution diagnostics")
    dv_series = df[dv].dropna()
    print(dv_series.describe())
    print(f"Skewness: {stats.skew(dv_series):.4f}")
    print(f"Kurtosis: {stats.kurtosis(dv_series):.4f}")

    print(f"\nIV distribution diagnostics ({iv})")
    if df[iv].nunique(dropna=True) <= 10:
        print(df[iv].value_counts(dropna=False).sort_index())
    else:
        print(df[iv].describe())

    print("\nBivariate correlations with DV (numeric only)")
    corr_rows = []
    for c in numeric_cols:
        if c == dv:
            continue
        s = df[[c, dv]].dropna()
        if s[c].nunique() <= 1:
            continue
        r, p = stats.pearsonr(s[c], s[dv])
        corr_rows.append((c, r, p))
    corr_rows.sort(key=lambda x: abs(x[1]), reverse=True)
    corr_df = pd.DataFrame(corr_rows, columns=["feature", "pearson_r", "p_value"])
    print(corr_df.head(15).to_string(index=False))

    # Step 2: OLS with controls
    print("\nStep 2: OLS regression")
    biv_df = df[[dv, iv]].dropna()
    X_biv = sm.add_constant(biv_df[[iv]])
    y_biv = biv_df[dv]
    biv_model = sm.OLS(y_biv, X_biv).fit(cov_type="HC3")
    print("\nBivariate OLS (DV ~ IV)")
    print(biv_model.summary())

    # Relevant controls: demographics/page/device/text features, excluding obvious post-treatment timing mediators
    excluded_numeric_controls = {
        dv,
        iv,
        "running_time",
        "adjusted_running_time",
        "scrolling_time",
        "correct_rate",
    }
    num_controls = [c for c in numeric_cols if c not in excluded_numeric_controls]

    cat_controls = []
    for c in df.columns:
        if c in {dv, iv, "uuid"}:
            continue
        if df[c].dtype == "object" or str(df[c].dtype).startswith("category"):
            cat_controls.append(c)

    model_cols = [dv, iv] + num_controls + cat_controls
    ctrl_df = df[model_cols].dropna().copy()

    X_num = ctrl_df[[iv] + num_controls]
    X_cat = pd.get_dummies(ctrl_df[cat_controls], drop_first=True, dtype=float) if cat_controls else pd.DataFrame(index=ctrl_df.index)
    X_ctrl = pd.concat([X_num, X_cat], axis=1)
    X_ctrl = sm.add_constant(X_ctrl)
    y_ctrl = ctrl_df[dv]

    if y_ctrl.nunique() == 2:
        ctrl_model = sm.Logit(y_ctrl, X_ctrl).fit(disp=False)
        model_type = "Logit"
    else:
        ctrl_model = sm.OLS(y_ctrl, X_ctrl).fit(cov_type="HC3")
        model_type = "OLS"

    print(f"\nControlled {model_type} ({dv} ~ {iv} + controls)")
    print(ctrl_model.summary())

    biv_coef = float(biv_model.params.get(iv, np.nan))
    biv_p = float(biv_model.pvalues.get(iv, np.nan))
    ctrl_coef = float(ctrl_model.params.get(iv, np.nan))
    ctrl_p = float(ctrl_model.pvalues.get(iv, np.nan))

    # Step 3: Interpretable models using ALL numeric columns except DV
    print("\nStep 3: Interpretable models")
    interp_features = [c for c in numeric_cols if c != dv]
    X_interp = df[interp_features].copy()
    for c in interp_features:
        if X_interp[c].isna().any():
            X_interp[c] = X_interp[c].fillna(X_interp[c].median())

    y_interp = df[dv]
    valid = ~y_interp.isna()
    X_interp = X_interp.loc[valid]
    y_interp = y_interp.loc[valid]

    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X_interp, y_interp)
    smart_effects = smart.feature_effects()
    print("\nSmartAdditiveRegressor summary:")
    print(smart)
    print("\nSmartAdditive feature_effects:")
    print(json.dumps(smart_effects, indent=2))

    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X_interp, y_interp)
    hinge_effects = hinge.feature_effects()
    print("\nHingeEBMRegressor summary:")
    print(hinge)
    print("\nHingeEBM feature_effects:")
    print(json.dumps(hinge_effects, indent=2))

    smart_iv = smart_effects.get(iv, {"direction": "zero", "importance": 0.0, "rank": 0})
    hinge_iv = hinge_effects.get(iv, {"direction": "zero", "importance": 0.0, "rank": 0})

    iv_shape = smart_shape_summary(smart, iv, interp_features)

    top_smart = top_effects(smart_effects, exclude=iv, k=3)
    top_hinge = top_effects(hinge_effects, exclude=iv, k=3)

    top_smart_text = ", ".join(
        [f"{f} (rank {e.get('rank', 0)}, {100 * e.get('importance', 0):.1f}%)" for f, e in top_smart]
    ) or "none"
    top_hinge_text = ", ".join(
        [f"{f} (rank {e.get('rank', 0)}, {100 * e.get('importance', 0):.1f}%)" for f, e in top_hinge]
    ) or "none"

    score = score_effect(
        biv_coef=biv_coef,
        biv_p=biv_p,
        ctrl_coef=ctrl_coef,
        ctrl_p=ctrl_p,
        smart_dir=smart_iv.get("direction", "zero"),
        smart_imp=float(smart_iv.get("importance", 0.0)),
        hinge_dir=hinge_iv.get("direction", "zero"),
        hinge_imp=float(hinge_iv.get("importance", 0.0)),
    )

    explanation = (
        f"Question: whether {iv} improves {dv}. "
        f"Bivariate OLS shows a {('positive' if biv_coef > 0 else 'negative')} association "
        f"(coef={biv_coef:.3f}, p={format_p(biv_p)}). "
        f"After adding controls (demographics, text/page properties, and device/language covariates), "
        f"the {iv} effect is {('positive' if ctrl_coef > 0 else 'negative')} "
        f"(coef={ctrl_coef:.3f}, p={format_p(ctrl_p)}), indicating "
        f"{('the effect persists' if (ctrl_coef > 0 and ctrl_p < 0.05) else 'a weaker/less robust controlled effect')}. "
        f"In SmartAdditive, {iv} is {smart_iv.get('direction', 'zero')} with importance "
        f"{100 * float(smart_iv.get('importance', 0.0)):.1f}% (rank {smart_iv.get('rank', 0)}), and the learned shape is {iv_shape}. "
        f"In HingeEBM, {iv} is {hinge_iv.get('direction', 'zero')} with importance "
        f"{100 * float(hinge_iv.get('importance', 0.0)):.1f}% (rank {hinge_iv.get('rank', 0)}). "
        f"Other influential predictors are SmartAdditive: {top_smart_text}; HingeEBM: {top_hinge_text}. "
        f"Overall score={score} reflects direction, magnitude/rank, nonlinear shape evidence, and consistency across models."
    )

    result = {"response": int(score), "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(result))

    print("\nStep 4: Wrote conclusion.txt")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
