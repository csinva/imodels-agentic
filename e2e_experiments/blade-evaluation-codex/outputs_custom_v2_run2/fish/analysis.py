import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from agentic_imodels import (
    HingeEBMRegressor,
    SmartAdditiveRegressor,
    SparseSignedBasisPursuitRegressor,
)


FEATURES = ["livebait", "camper", "persons", "child"]
TARGET = "fish_caught"
EXPOSURE = "hours"
FEATURE_MAP = {0: "livebait", 1: "camper", 2: "persons", 3: "child"}


def parse_linear_x_coeffs(model_text: str, feature_map: dict[int, str]) -> dict[str, float]:
    """Parse terms like '+0.123*x2' from model string output."""
    compact = model_text.replace(" ", "")
    matches = re.findall(r"([+-]?\d+(?:\.\d+)?)\*x(\d+)", compact)
    out: dict[str, float] = {}
    for coef_str, idx_str in matches:
        idx = int(idx_str)
        if idx in feature_map:
            out[feature_map[idx]] = float(coef_str)
    return out


def parse_sparse_active_zero(model_text: str, feature_map: dict[int, str]) -> tuple[list[str], list[str]]:
    active: list[str] = []
    zero: list[str] = []
    for line in model_text.splitlines():
        line = line.strip()
        if line.startswith("Active features:"):
            rhs = line.split(":", 1)[1].strip()
            toks = [t.strip() for t in rhs.split(",") if t.strip()]
            for tok in toks:
                if tok.startswith("x") and tok[1:].isdigit():
                    active.append(feature_map.get(int(tok[1:]), tok))
                else:
                    active.append(tok)
        if line.startswith("Zero-contribution features:"):
            rhs = line.split(":", 1)[1].strip()
            toks = [t.strip() for t in rhs.split(",") if t.strip()]
            for tok in toks:
                if tok.startswith("x") and tok[1:].isdigit():
                    zero.append(feature_map.get(int(tok[1:]), tok))
                else:
                    zero.append(tok)
    return active, zero


def main() -> None:
    np.random.seed(0)

    info = json.loads(Path("info.json").read_text())
    research_question = info["research_questions"][0]

    df = pd.read_csv("fish.csv")

    print("=== Research Question ===")
    print(research_question)

    print("\n=== Data Overview ===")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Missing values:\n", df.isna().sum())

    print("\n=== Summary Statistics ===")
    print(df.describe().T)

    # Distribution-focused stats
    print("\n=== Distribution Diagnostics ===")
    zero_share = (df[TARGET] == 0).mean()
    print(f"Share of zero catches: {zero_share:.3f}")
    print("fish_caught quantiles:")
    print(df[TARGET].quantile([0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]))
    print(f"Mean fish_caught: {df[TARGET].mean():.3f}")
    print(f"Variance fish_caught: {df[TARGET].var():.3f}")
    print(f"Variance/mean ratio: {df[TARGET].var() / df[TARGET].mean():.3f}")

    # Per-hour rate diagnostics
    df["fish_per_hour"] = df[TARGET] / df[EXPOSURE]
    df["log1p_fish_per_hour"] = np.log1p(df["fish_per_hour"])
    print(f"\nMean fish/hour: {df['fish_per_hour'].mean():.3f}")
    print(f"Median fish/hour: {df['fish_per_hour'].median():.3f}")

    print("\n=== Correlation Matrix ===")
    print(df[[TARGET, "fish_per_hour", *FEATURES, EXPOSURE]].corr(numeric_only=True))

    print("\n=== Bivariate Tests (Fish/Hour) ===")
    bivariate = {}

    for col in ["livebait", "camper"]:
        g1 = df.loc[df[col] == 1, "fish_per_hour"]
        g0 = df.loc[df[col] == 0, "fish_per_hour"]
        t_stat, p_val = stats.ttest_ind(g1, g0, equal_var=False)
        bivariate[col] = {
            "group1_mean": float(g1.mean()),
            "group0_mean": float(g0.mean()),
            "p": float(p_val),
        }
        print(
            f"{col}: mean(1)={g1.mean():.3f}, mean(0)={g0.mean():.3f}, "
            f"Welch t-test p={p_val:.4g}"
        )

    for col in ["persons", "child", "hours"]:
        r, p_val = stats.pearsonr(df[col], df["fish_per_hour"])
        bivariate[col] = {"r": float(r), "p": float(p_val)}
        print(f"{col}: Pearson r={r:.3f}, p={p_val:.4g}")

    print("\n=== Classical Controlled Model: Count GLM with Exposure Offset ===")
    X = sm.add_constant(df[FEATURES])
    y = df[TARGET]
    offset = np.log(df[EXPOSURE])

    poisson = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit()
    overdisp_ratio = float(np.sum(poisson.resid_pearson**2) / poisson.df_resid)
    print("\nPoisson GLM summary:")
    print(poisson.summary())
    print(f"Poisson Pearson overdispersion ratio: {overdisp_ratio:.3f}")

    nb2 = sm.NegativeBinomial(y, X, loglike_method="nb2", offset=offset).fit(disp=False)
    print("\nNegative Binomial (NB2) summary:")
    print(nb2.summary())

    nb_pvals = nb2.pvalues.to_dict()
    nb_coefs = nb2.params.to_dict()

    # Convert to incidence-rate ratios for interpretability
    irr = {k: float(np.exp(v)) for k, v in nb_coefs.items() if k != "alpha"}
    print("\nNB2 Incidence-Rate Ratios (exp(beta)):")
    for k, v in irr.items():
        print(f"{k}: {v:.3f}")

    print("\n=== agentic_imodels (Interpretable Regressors) ===")
    X_rate = df[FEATURES]
    y_rate = df["log1p_fish_per_hour"]

    smart = SmartAdditiveRegressor().fit(X_rate, y_rate)
    print("\n--- SmartAdditiveRegressor ---")
    print(smart)

    hinge_ebm = HingeEBMRegressor().fit(X_rate, y_rate)
    print("\n--- HingeEBMRegressor ---")
    print(hinge_ebm)

    sparse_basis = SparseSignedBasisPursuitRegressor().fit(X_rate, y_rate)
    print("\n--- SparseSignedBasisPursuitRegressor ---")
    print(sparse_basis)

    smart_text = str(smart)
    hinge_text = str(hinge_ebm)
    sparse_text = str(sparse_basis)

    smart_coefs = parse_linear_x_coeffs(smart_text, FEATURE_MAP)
    hinge_coefs = parse_linear_x_coeffs(hinge_text, FEATURE_MAP)
    sparse_active, sparse_zero = parse_sparse_active_zero(sparse_text, FEATURE_MAP)

    print("\nParsed SmartAdditive coefficients:", smart_coefs)
    print("Parsed HingeEBM coefficients:", hinge_coefs)
    print("Parsed SparseSigned active features:", sparse_active)
    print("Parsed SparseSigned zero features:", sparse_zero)

    # Evidence synthesis + Likert calibration
    sig = {
        f: (nb_pvals.get(f, 1.0) < 0.05)
        for f in FEATURES
    }

    # Strong evidence factors (significant in NB2 and same sign in two agentic models)
    agreement = {}
    for f in FEATURES:
        s = smart_coefs.get(f)
        h = hinge_coefs.get(f)
        if s is None or h is None:
            agreement[f] = False
        else:
            agreement[f] = np.sign(s) == np.sign(h)

    # Score starts from a moderate prior and is adjusted by consistency/strength
    score = 50

    # Controlled significance in appropriate count model
    if sig.get("persons", False):
        score += 18
    if sig.get("livebait", False):
        score += 14
    if sig.get("camper", False):
        score -= 4
    if sig.get("child", False):
        score += 4

    # Bivariate support
    if bivariate["persons"]["p"] < 0.05:
        score += 6
    if bivariate["livebait"]["p"] < 0.05:
        score += 5

    # Interpretable-model robustness agreement (upweight only significant controlled effects)
    for f in FEATURES:
        if agreement.get(f, False) and sig.get(f, False):
            score += 2

    # Null evidence from sparse basis zeroing
    if "livebait" in sparse_zero:
        score -= 6
    if "camper" in sparse_zero:
        score -= 3

    # Mixed-evidence penalties: only some predictors are stable after controls
    n_sig = sum(int(v) for v in sig.values())
    if n_sig <= 2:
        score -= 6
    if sig.get("livebait", False) and "livebait" in sparse_zero:
        score -= 4

    # Clamp 0..100
    score = int(max(0, min(100, round(score))))

    explanation = (
        "Using a count model with exposure (NB2 with log(hours) offset), the fish-catch rate is "
        "strongly associated with group composition: persons is positive and highly significant "
        f"(beta={nb_coefs['persons']:.3f}, p={nb_pvals['persons']:.3g}, IRR={irr['persons']:.2f}), "
        "and livebait is also positive/significant "
        f"(beta={nb_coefs['livebait']:.3f}, p={nb_pvals['livebait']:.3g}, IRR={irr['livebait']:.2f}). "
        f"Camper (p={nb_pvals['camper']:.3g}) and child (p={nb_pvals['child']:.3g}) are not significant "
        "after controls. Bivariate checks on fish/hour are consistent for persons and livebait, but not "
        "for camper/child. Interpretable models broadly agree on direction for persons (positive) and "
        "child (negative), while SparseSignedBasisPursuit zeroes out livebait and camper, which is null "
        "evidence against those effects and lowers certainty. Overall this supports a moderate-to-strong "
        "Yes: some factors (especially number of adults, plus livebait in the controlled count model) "
        "meaningfully influence catch rate per hour, but not all candidate predictors are robust."
    )

    payload = {"response": score, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(payload))

    print("\n=== Final Conclusion JSON ===")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
