import json
import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.exceptions import ConvergenceWarning

from agentic_imodels import (
    HingeEBMRegressor,
    HingeGAMRegressor,
    SmartAdditiveRegressor,
    WinsorizedSparseOLSRegressor,
)


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def print_distribution(name: str, values: pd.Series, bins: int = 8) -> None:
    vals = values.dropna().to_numpy()
    hist, edges = np.histogram(vals, bins=bins)
    print(f"\\n{name} distribution (counts by bin):")
    for i, count in enumerate(hist):
        left = edges[i]
        right = edges[i + 1]
        print(f"  [{left:8.3f}, {right:8.3f}): {int(count)}")


def extract_zeroed_features(model_text: str) -> set[str]:
    m = re.search(r"Features with zero coefficients \(excluded\):\s*(.+)", model_text)
    if not m:
        return set()
    return {tok.strip() for tok in m.group(1).split(",") if tok.strip()}


def extract_named_coef(model_text: str, feature_token: str) -> float | None:
    m = re.search(rf"\n\s*{re.escape(feature_token)}:\s*([+-]?\d+(?:\.\d+)?)", model_text)
    if not m:
        return None
    return float(m.group(1))


def extract_piecewise_values(model_text: str, feature_token: str) -> list[float]:
    tag = f"f({feature_token}):"
    if tag not in model_text:
        return []
    part = model_text.split(tag, 1)[1]
    lines = part.splitlines()[1:]

    vals = []
    for line in lines:
        if line.strip().startswith("f("):
            break
        nums = re.findall(r"([+-]?\d+\.\d+)\s*$", line)
        if nums:
            vals.append(float(nums[-1]))
    return vals


def main() -> None:
    question = "Is a lower student-teacher ratio associated with higher academic performance?"
    print("Research question:", question)

    df = pd.read_csv("caschools.csv")
    df["student_teacher_ratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0
    df["grade_KK_08"] = (df["grades"] == "KK-08").astype(int)

    iv = "student_teacher_ratio"
    dv = "avg_score"
    controls = [
        "calworks",
        "lunch",
        "english",
        "income",
        "expenditure",
        "computer",
        "students",
        "grade_KK_08",
    ]
    features = [iv] + controls

    print("\n=== DATA OVERVIEW ===")
    print("Rows, columns:", df.shape)
    print("\nMissing values by column:")
    print(df[features + ["read", "math"]].isna().sum().to_string())

    print("\nSummary statistics for key variables:")
    print(df[features + [dv]].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(3).to_string())

    print_distribution(iv, df[iv])
    print_distribution(dv, df[dv])

    corr_to_outcome = (
        df[features + [dv]]
        .corr(numeric_only=True)[dv]
        .sort_values(ascending=False)
        .round(3)
    )
    print("\nCorrelation of predictors with avg_score:")
    print(corr_to_outcome.to_string())

    print("\n=== CLASSICAL TESTS ===")
    pearson_r, pearson_p = stats.pearsonr(df[iv], df[dv])
    print(f"Pearson correlation ({iv}, {dv}): r={pearson_r:.4f}, p={pearson_p:.4g}")

    X_biv = sm.add_constant(df[[iv]])
    ols_biv = sm.OLS(df[dv], X_biv).fit()
    print("\nBivariate OLS summary:")
    print(ols_biv.summary())

    X_ctrl = sm.add_constant(df[features])
    ols_ctrl = sm.OLS(df[dv], X_ctrl).fit()
    print("\nControlled OLS summary:")
    print(ols_ctrl.summary())

    iv_beta_ctrl = float(ols_ctrl.params[iv])
    iv_p_ctrl = float(ols_ctrl.pvalues[iv])
    iv_ci_low, iv_ci_high = map(float, ols_ctrl.conf_int().loc[iv])

    iv_beta_biv = float(ols_biv.params[iv])
    iv_p_biv = float(ols_biv.pvalues[iv])

    print("\n=== INTERPRETABLE MODELS (agentic_imodels) ===")
    X_interpret = df[[
        iv,
        "calworks",
        "lunch",
        "english",
        "income",
        "expenditure",
        "computer",
        "students",
    ]]
    y_interpret = df[dv]

    model_classes = [
        WinsorizedSparseOLSRegressor,  # honest sparse linear
        SmartAdditiveRegressor,        # honest shape model
        HingeGAMRegressor,             # honest hinge GAM
        HingeEBMRegressor,             # high-rank decoupled model
    ]

    model_strings: dict[str, str] = {}
    for cls in model_classes:
        model = cls().fit(X_interpret, y_interpret)
        s = str(model)
        model_strings[cls.__name__] = s
        print(f"\n--- {cls.__name__} ---")
        print(model)

    iv_token = "x0"  # first feature in X_interpret is student_teacher_ratio

    zeroed_by = []
    iv_coefs = {}
    for name, text in model_strings.items():
        if iv_token in extract_zeroed_features(text):
            zeroed_by.append(name)
        coef = extract_named_coef(text, iv_token)
        if coef is not None:
            iv_coefs[name] = coef

    smart_vals = extract_piecewise_values(model_strings["SmartAdditiveRegressor"], iv_token)
    shape_amplitude = float(max(smart_vals) - min(smart_vals)) if smart_vals else 0.0

    print("\n=== EVIDENCE SYNTHESIS ===")
    print(
        f"Controlled OLS for {iv}: beta={iv_beta_ctrl:.4f}, p={iv_p_ctrl:.4g}, "
        f"95% CI=[{iv_ci_low:.4f}, {iv_ci_high:.4f}]"
    )
    print(f"Bivariate OLS for {iv}: beta={iv_beta_biv:.4f}, p={iv_p_biv:.4g}")
    print(f"agentic_imodels linear coefficients for x0 ({iv}): {iv_coefs}")
    print(f"Models zeroing out x0 ({iv}): {zeroed_by if zeroed_by else 'None'}")
    print(f"SmartAdditive piecewise amplitude for x0: {shape_amplitude:.4f}")

    # Likert score calibration (0=strong No, 100=strong Yes)
    score = 50.0

    # Bivariate evidence
    if iv_beta_biv < 0 and iv_p_biv < 0.05:
        score += 12
    elif iv_beta_biv < 0 and iv_p_biv < 0.10:
        score += 6

    # Controlled evidence (primary)
    if iv_beta_ctrl < 0 and iv_p_ctrl < 0.05:
        score += 30
    elif iv_beta_ctrl < 0 and iv_p_ctrl < 0.10:
        score += 12
    else:
        score -= 22

    # Magnitude under controls
    if abs(iv_beta_ctrl) < 0.5:
        score -= 6
    elif abs(iv_beta_ctrl) > 1.5 and iv_beta_ctrl < 0:
        score += 6

    # Null evidence from hinge/regularized zeroing
    score -= 10 * len(zeroed_by)

    # Direction consistency across non-zero coefficients in interpretable models
    if iv_coefs:
        neg = sum(c < 0 for c in iv_coefs.values())
        pos = sum(c > 0 for c in iv_coefs.values())
        if neg >= 2:
            score += 4
        if pos > 0:
            score -= 8

    # Shape evidence: if nonlinear amplitude is tiny, association is likely weak
    if smart_vals and shape_amplitude < 1.0:
        score -= 4

    response = int(np.clip(round(score), 0, 100))

    explanation = (
        f"Bivariate analysis shows a negative association between student-teacher ratio and average scores "
        f"(Pearson r={pearson_r:.3f}, p={pearson_p:.2g}; bivariate OLS beta={iv_beta_biv:.3f}, p={iv_p_biv:.2g}). "
        f"However, after controlling for socioeconomic and resource covariates, the ratio effect becomes small and not statistically "
        f"significant (beta={iv_beta_ctrl:.3f}, p={iv_p_ctrl:.3f}, 95% CI [{iv_ci_low:.3f}, {iv_ci_high:.3f}]). "
        f"In interpretable models, WinsorizedSparseOLS gives a small negative linear coefficient for the ratio, SmartAdditive shows only "
        f"modest nonlinear movement, and both HingeGAM and HingeEBM zero out the ratio term, which is strong null evidence under sparse/hinge selection. "
        f"Overall, evidence supports at most a weak association once confounding is addressed, so the answer leans No rather than Yes."
    )

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump({"response": response, "explanation": explanation}, f)

    print("\nWrote conclusion.txt")
    print(json.dumps({"response": response, "explanation": explanation}, indent=2))


if __name__ == "__main__":
    main()
