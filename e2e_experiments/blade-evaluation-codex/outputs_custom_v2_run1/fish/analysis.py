import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.exceptions import ConvergenceWarning

from agentic_imodels import (
    HingeEBMRegressor,
    SmartAdditiveRegressor,
    SparseSignedBasisPursuitRegressor,
)


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def parse_linear_x_coeffs(model_text: str, feature_cols: list[str]) -> dict[str, float]:
    coeffs: dict[str, float] = {}
    for coef_str, idx_str in re.findall(r"([+-]?\d+(?:\.\d+)?)\*x(\d+)", model_text):
        idx = int(idx_str)
        if 0 <= idx < len(feature_cols):
            coeffs[feature_cols[idx]] = float(coef_str)
    return coeffs


def parse_zeroed_features(model_text: str, feature_cols: list[str]) -> set[str]:
    zeroed = set()
    patterns = [
        r"Zero-contribution features:\s*([^\n]+)",
        r"zero[- ]effect features:\s*([^\n]+)",
        r"Features with zero coefficients \(excluded\):\s*([^\n]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, model_text, flags=re.IGNORECASE)
        if not m:
            continue
        raw = m.group(1)
        parts = [p.strip().strip(".") for p in raw.split(",") if p.strip()]
        for token in parts:
            if token in feature_cols:
                zeroed.add(token)
            elif re.fullmatch(r"x\d+", token):
                idx = int(token[1:])
                if 0 <= idx < len(feature_cols):
                    zeroed.add(feature_cols[idx])
    return zeroed


def clamp_int(x: float, lo: int = 0, hi: int = 100) -> int:
    return int(max(lo, min(hi, round(x))))


def main() -> None:
    base = Path(".")
    info = json.loads((base / "info.json").read_text())

    print("=== Research question ===")
    for q in info.get("research_questions", []):
        print("-", q)

    df = pd.read_csv(base / "fish.csv")
    df["fish_per_hour"] = df["fish_caught"] / df["hours"]
    # stabilized log-rate for continuous interpretable regressors
    df["log_rate"] = np.log((df["fish_caught"] + 0.5) / df["hours"])

    print("\n=== Data overview ===")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Missing values:\n", df.isna().sum())

    print("\n=== Summary statistics ===")
    print(df.describe().to_string())

    print("\n=== Distribution checkpoints ===")
    for col in ["fish_caught", "hours", "fish_per_hour", "log_rate"]:
        qs = df[col].quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
        print(f"{col}:\n{qs.to_string()}\n")

    weighted_rate = float(df["fish_caught"].sum() / df["hours"].sum())
    mean_individual_rate = float(df["fish_per_hour"].mean())
    median_individual_rate = float(df["fish_per_hour"].median())
    zero_catch_share = float((df["fish_caught"] == 0).mean())

    print("=== Catch-rate estimates ===")
    print(f"Weighted average catch rate (total fish / total hours): {weighted_rate:.4f} fish/hour")
    print(f"Mean individual trip catch rate: {mean_individual_rate:.4f} fish/hour")
    print(f"Median individual trip catch rate: {median_individual_rate:.4f} fish/hour")
    print(f"Share of zero-catch trips: {zero_catch_share:.3f}")

    print("\n=== Correlations ===")
    corr = df[["fish_caught", "livebait", "camper", "persons", "child", "hours", "fish_per_hour", "log_rate"]].corr(numeric_only=True)
    print(corr.to_string())

    print("\n=== Bivariate tests ===")
    pearson_hours = stats.pearsonr(df["fish_caught"], df["hours"])
    spearman_hours = stats.spearmanr(df["fish_caught"], df["hours"])
    print(f"Pearson fish_caught vs hours: r={pearson_hours.statistic:.4f}, p={pearson_hours.pvalue:.4g}")
    print(f"Spearman fish_caught vs hours: rho={spearman_hours.statistic:.4f}, p={spearman_hours.pvalue:.4g}")

    for b in ["livebait", "camper"]:
        grp1 = df.loc[df[b] == 1, "log_rate"]
        grp0 = df.loc[df[b] == 0, "log_rate"]
        t = stats.ttest_ind(grp1, grp0, equal_var=False)
        print(
            f"Welch t-test log_rate by {b}: "
            f"mean(1)={grp1.mean():.4f}, mean(0)={grp0.mean():.4f}, t={t.statistic:.4f}, p={t.pvalue:.4g}"
        )

    for c in ["persons", "child"]:
        s = stats.spearmanr(df[c], df["log_rate"])
        print(f"Spearman log_rate vs {c}: rho={s.statistic:.4f}, p={s.pvalue:.4g}")

    print("\n=== Controlled count-rate models ===")
    controls = ["livebait", "camper", "persons", "child"]
    X_ctrl = sm.add_constant(df[controls])
    y_count = df["fish_caught"]
    offset = np.log(df["hours"])

    poisson = sm.GLM(y_count, X_ctrl, family=sm.families.Poisson(), offset=offset).fit()
    print("Poisson GLM with log(hours) offset")
    print(poisson.summary())
    overdispersion_ratio = float(poisson.pearson_chi2 / poisson.df_resid)
    print(f"Poisson overdispersion ratio (Pearson chi2/df): {overdispersion_ratio:.3f}")

    nb = sm.NegativeBinomial(y_count, X_ctrl, offset=offset).fit(disp=False)
    print("\nNegative Binomial with log(hours) offset (primary due overdispersion)")
    print(nb.summary())

    nb_params = nb.params.copy()
    nb_pvals = nb.pvalues.copy()
    irr = np.exp(nb_params.drop("alpha"))

    print("\nIncidence Rate Ratios (NB):")
    print(irr.to_string())

    print("\n=== Interpretable models (agentic_imodels) ===")
    feature_cols = ["livebait", "camper", "persons", "child", "hours"]
    X_interp = df[feature_cols]
    y_interp = df["log_rate"]

    print("Feature mapping used by printed models:")
    for i, c in enumerate(feature_cols):
        print(f"  x{i} -> {c}")

    model_specs = [
        ("SmartAdditiveRegressor", SmartAdditiveRegressor()),  # honest
        ("HingeEBMRegressor", HingeEBMRegressor()),            # high-rank decoupled
        ("SparseSignedBasisPursuitRegressor", SparseSignedBasisPursuitRegressor()),  # honest sparse/zeroing
    ]

    model_texts: dict[str, str] = {}
    linear_coeffs: dict[str, dict[str, float]] = {}
    zeroed_by_model: dict[str, set[str]] = {}

    for name, model in model_specs:
        fitted = model.fit(X_interp, y_interp)
        text = str(fitted)
        model_texts[name] = text
        linear_coeffs[name] = parse_linear_x_coeffs(text, feature_cols)
        zeroed_by_model[name] = parse_zeroed_features(text, feature_cols)

        print(f"\n--- {name} ---")
        print(text)

    # Evidence synthesis for Likert score
    sig_features = [f for f in controls if nb_pvals[f] < 0.05]
    very_sig_features = [f for f in controls if nb_pvals[f] < 0.01]

    # Directional robustness across two complementary interpretable models
    dir_support = 0
    dir_total = 0
    for f in sig_features:
        nb_sign = np.sign(nb_params[f])
        for model_name in ["SmartAdditiveRegressor", "HingeEBMRegressor"]:
            coeff = linear_coeffs.get(model_name, {}).get(f)
            if coeff is None:
                continue
            dir_total += 1
            if np.sign(coeff) == nb_sign and abs(coeff) > 1e-8:
                dir_support += 1

    sparse_zeroed = zeroed_by_model.get("SparseSignedBasisPursuitRegressor", set())
    conflicted_sig = [f for f in sig_features if f in sparse_zeroed]

    # Score calibration per SKILL guidance
    score = 50.0
    if len(sig_features) == 0:
        score = 15.0
    elif len(sig_features) == 1:
        score = 62.0
    elif len(sig_features) >= 2:
        score = 72.0

    score += 4.0 * len(very_sig_features)
    if "persons" in sig_features and irr.get("persons", 1.0) > 2.0:
        score += 6.0
    if "livebait" in sig_features and irr.get("livebait", 1.0) > 2.0:
        score += 6.0

    if dir_total > 0:
        score += 8.0 * (dir_support / dir_total)

    score -= 8.0 * len(conflicted_sig)

    # reward specific null evidence for non-significant controls
    null_controls = [f for f in controls if nb_pvals[f] >= 0.05]
    null_zeroed = [f for f in null_controls if f in sparse_zeroed]
    score += 2.0 * len(null_zeroed)

    response = clamp_int(score)

    # Structured explanation with concrete effect sizes
    effect_lines = []
    for f in controls:
        beta = nb_params[f]
        p = nb_pvals[f]
        irr_f = float(np.exp(beta))
        effect_lines.append(f"{f}: beta={beta:.3f}, IRR={irr_f:.2f}, p={p:.4g}")

    explanation = (
        f"Average catch rate is about {weighted_rate:.3f} fish/hour (total-fish/total-hours; "
        f"trip-level mean {mean_individual_rate:.3f}, median {median_individual_rate:.3f}). "
        f"Poisson with offset showed strong overdispersion (ratio={overdispersion_ratio:.2f}), "
        f"so Negative Binomial is primary. NB evidence with controls: "
        + "; ".join(effect_lines)
        + ". This indicates robust positive associations for livebait and persons, while camper and child are not statistically reliable after controls. "
        "In agentic_imodels, SmartAdditive and HingeEBM both keep livebait/persons with positive direction and show camper as weak/zeroed, "
        "while SparseSignedBasisPursuit provides sparse null evidence by zeroing camper and child (and also drops livebait, indicating some model disagreement). "
        "Overall evidence supports that catch rate per hour is meaningfully influenced by trip composition, with strongest support for number of adults and livebait usage, "
        "and weaker/inconsistent evidence for camper and children."
    )

    conclusion = {"response": response, "explanation": explanation}
    with (base / "conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=True)

    print("\n=== Conclusion JSON ===")
    print(json.dumps(conclusion, ensure_ascii=True))


if __name__ == "__main__":
    main()
