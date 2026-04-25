import json
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr

from agentic_imodels import (
    HingeEBMRegressor,
    SmartAdditiveRegressor,
    WinsorizedSparseOLSRegressor,
)


def summarize_data(df: pd.DataFrame) -> None:
    print("=== DATA OVERVIEW ===")
    print(f"Shape: {df.shape}")
    print("\nColumn dtypes:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isna().sum())

    print("\n=== SUMMARY STATS (NUMERIC) ===")
    print(df.describe().T)

    print("\n=== EVAL DISTRIBUTION ===")
    print(df["eval"].value_counts().sort_index())

    print("\n=== BEAUTY DISTRIBUTION (BINS) ===")
    beauty_bins = pd.cut(df["beauty"], bins=10)
    print(beauty_bins.value_counts().sort_index())

    print("\n=== NUMERIC CORRELATIONS WITH EVAL ===")
    corr_with_eval = df.corr(numeric_only=True)["eval"].sort_values(ascending=False)
    print(corr_with_eval)


def run_classical_tests(df: pd.DataFrame) -> Dict[str, float]:
    print("\n=== CLASSICAL TESTS ===")

    # Bivariate association
    r, p_corr = pearsonr(df["beauty"], df["eval"])
    print(f"Pearson correlation beauty vs eval: r={r:.4f}, p={p_corr:.4g}")

    biv_model = smf.ols("eval ~ beauty", data=df).fit()
    print("\nBivariate OLS: eval ~ beauty")
    print(biv_model.summary())

    # Controlled OLS with relevant confounders
    controlled_formula = (
        "eval ~ beauty + age + students + allstudents + "
        "C(minority) + C(gender) + C(credits) + C(division) + C(native) + C(tenure)"
    )
    controlled_model = smf.ols(controlled_formula, data=df).fit(cov_type="HC3")
    print("\nControlled OLS (HC3 robust SE):")
    print(controlled_model.summary())

    beauty_coef = float(controlled_model.params["beauty"])
    beauty_p = float(controlled_model.pvalues["beauty"])
    beauty_ci_low, beauty_ci_high = controlled_model.conf_int().loc["beauty"].tolist()

    print(
        f"\nControlled effect for beauty: coef={beauty_coef:.4f}, "
        f"p={beauty_p:.4g}, 95% CI=({beauty_ci_low:.4f}, {beauty_ci_high:.4f})"
    )

    return {
        "pearson_r": float(r),
        "pearson_p": float(p_corr),
        "bivariate_beauty_coef": float(biv_model.params["beauty"]),
        "bivariate_beauty_p": float(biv_model.pvalues["beauty"]),
        "controlled_beauty_coef": beauty_coef,
        "controlled_beauty_p": beauty_p,
        "controlled_beauty_ci_low": float(beauty_ci_low),
        "controlled_beauty_ci_high": float(beauty_ci_high),
        "controlled_r2": float(controlled_model.rsquared),
    }


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Keep interpretable and relevant predictors; exclude row index and professor ID.
    feature_cols = [
        "beauty",
        "age",
        "students",
        "allstudents",
        "minority",
        "gender",
        "credits",
        "division",
        "native",
        "tenure",
    ]
    X = df[feature_cols].copy()
    X = pd.get_dummies(X, drop_first=True)
    y = df["eval"].copy()
    return X, y


def extract_beauty_signal(
    model_text: str,
    feature_name: str = "beauty",
    feature_aliases: List[str] | None = None,
) -> Dict[str, object]:
    lines = model_text.splitlines()
    tokens = [feature_name]
    if feature_aliases:
        tokens.extend(feature_aliases)
    beauty_lines: List[str] = [ln for ln in lines if any(tok in ln for tok in tokens)]
    present = len(beauty_lines) > 0

    signs: List[str] = []
    for ln in beauty_lines:
        # Lightweight sign extraction for common printable formats
        if re.search(r":\s*\+?[0-9\.]+", ln) and not re.search(r":\s*-", ln):
            signs.append("positive")
        elif re.search(r":\s*-[0-9\.]+", ln):
            signs.append("negative")
        elif re.search(rf"{feature_name}.*\+\s*[0-9\.]+", ln) or re.search(
            rf"\+\s*[0-9\.]+\s*\*?\s*{feature_name}", ln
        ):
            signs.append("positive")
        elif re.search(rf"{feature_name}.*-\s*[0-9\.]+", ln) or re.search(
            rf"-\s*[0-9\.]+\s*\*?\s*{feature_name}", ln
        ):
            signs.append("negative")
        elif "<=" in ln or ">=" in ln or "<" in ln or ">" in ln:
            signs.append("nonlinear_or_threshold")
        else:
            signs.append("mentioned")

    return {
        "present": present,
        "lines": beauty_lines[:6],
        "sign_hints": signs,
    }


def run_agentic_models(X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, object]]:
    print("\n=== AGENTIC_IMODELS INTERPRETABLE REGRESSORS ===")

    models = [
        ("SmartAdditiveRegressor", SmartAdditiveRegressor()),
        ("HingeEBMRegressor", HingeEBMRegressor()),
        ("WinsorizedSparseOLSRegressor", WinsorizedSparseOLSRegressor()),
    ]

    signals: Dict[str, Dict[str, object]] = {}
    beauty_aliases = [f"x{i}" for i, col in enumerate(X.columns) if "beauty" in col]
    print("\nFeature index map used by many printed models:")
    for i, col in enumerate(X.columns):
        print(f"  x{i} -> {col}")
    print(f"Beauty aliases in printed models: {beauty_aliases}")

    for name, model in models:
        fitted = model.fit(X, y)
        print(f"\n=== {name} ===")
        print(fitted)  # Required interpretable form output
        model_text = str(fitted)
        beauty_signal = extract_beauty_signal(
            model_text, feature_name="beauty", feature_aliases=beauty_aliases
        )
        signals[name] = beauty_signal
        print(f"\nBeauty signal in {name}: {json.dumps(beauty_signal, indent=2)}")

    return signals


def calibrate_score(
    classical: Dict[str, float], model_signals: Dict[str, Dict[str, object]]
) -> Tuple[int, str]:
    score = 50
    rationale: List[str] = []

    controlled_beta = classical["controlled_beauty_coef"]
    controlled_p = classical["controlled_beauty_p"]

    if controlled_p < 0.001:
        score += 22
        rationale.append("Controlled OLS shows a highly significant beauty effect (p < 0.001).")
    elif controlled_p < 0.01:
        score += 16
        rationale.append("Controlled OLS shows a significant beauty effect (p < 0.01).")
    elif controlled_p < 0.05:
        score += 10
        rationale.append("Controlled OLS shows a statistically significant beauty effect (p < 0.05).")
    elif controlled_p < 0.10:
        score += 4
        rationale.append("Controlled OLS gives only marginal evidence for beauty (0.05 <= p < 0.10).")
    else:
        score -= 24
        rationale.append("Controlled OLS does not support a reliable beauty effect (p >= 0.10).")

    if controlled_beta > 0:
        score += 6
        rationale.append("The controlled coefficient for beauty is positive.")
    elif controlled_beta < 0:
        score -= 6
        rationale.append("The controlled coefficient for beauty is negative.")

    if classical["pearson_p"] < 0.05 and classical["pearson_r"] > 0:
        score += 5
        rationale.append("Bivariate correlation is positive and significant.")
    elif classical["pearson_p"] >= 0.05:
        score -= 5
        rationale.append("Bivariate correlation is not statistically significant.")

    present_count = sum(1 for v in model_signals.values() if v["present"])
    absent_count = len(model_signals) - present_count

    score += 7 * present_count
    score -= 9 * absent_count

    if present_count == len(model_signals):
        rationale.append("Beauty appears across all interpretable models, supporting robustness.")
    elif present_count >= 2:
        rationale.append("Beauty appears in most interpretable models, supporting moderate robustness.")
    elif present_count == 1:
        rationale.append("Beauty appears in only one interpretable model, so robustness is limited.")
    else:
        rationale.append("Beauty is zeroed out or absent across interpretable models, strong null evidence.")

    score = int(np.clip(round(score), 0, 100))

    explanation = (
        f"Research question: impact of beauty on teaching evaluations. "
        f"Controlled OLS coefficient for beauty is {controlled_beta:.3f} "
        f"(p={controlled_p:.4g}, 95% CI [{classical['controlled_beauty_ci_low']:.3f}, "
        f"{classical['controlled_beauty_ci_high']:.3f}]). "
        f"Bivariate correlation is r={classical['pearson_r']:.3f} (p={classical['pearson_p']:.4g}). "
        f"Across interpretable models, beauty is present in {present_count}/{len(model_signals)} models. "
        + " ".join(rationale)
    )

    return score, explanation


def main() -> None:
    print("Loading data...")
    df = pd.read_csv("teachingratings.csv")

    summarize_data(df)
    classical = run_classical_tests(df)
    X, y = build_features(df)
    model_signals = run_agentic_models(X, y)
    score, explanation = calibrate_score(classical, model_signals)

    output = {"response": int(score), "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f)

    print("\n=== FINAL CONCLUSION JSON ===")
    print(json.dumps(output, indent=2))
    print("\nWrote conclusion.txt")


if __name__ == "__main__":
    main()
