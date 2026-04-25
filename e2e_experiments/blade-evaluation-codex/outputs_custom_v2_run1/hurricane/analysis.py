import json
import os
import re
import shutil
import sys


def _ensure_runtime() -> None:
    """Allow `python3 analysis.py` in environments where packages are on python3.11."""
    try:
        import pandas  # noqa: F401
    except ModuleNotFoundError:
        py311 = shutil.which("python3.11")
        if py311 and os.path.realpath(sys.executable) != os.path.realpath(py311):
            os.execv(py311, [py311, *sys.argv])
        raise


_ensure_runtime()

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

from agentic_imodels import (
    HingeEBMRegressor,
    SmartAdditiveRegressor,
    WinsorizedSparseOLSRegressor,
)


def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def active_in_hinge_text(model_text: str, feature_token: str) -> bool:
    pattern = rf"\b{re.escape(feature_token)}\b:"
    return re.search(pattern, model_text) is not None


def main() -> None:
    # ------------------------------------------------------------------
    # 1) Load question metadata and data
    # ------------------------------------------------------------------
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info.get("research_questions", [""])[0]
    print_section("Research Question")
    print(question)

    df = pd.read_csv("hurricane.csv")
    print_section("Data Overview")
    print(f"Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("\nMissing values:")
    print(df.isna().sum().to_string())

    # Core transforms for analysis
    df["log_alldeaths"] = np.log1p(df["alldeaths"])
    df["log_ndam15"] = np.log1p(df["ndam15"])

    # ------------------------------------------------------------------
    # 2) EDA: summary, distributions, correlations
    # ------------------------------------------------------------------
    numeric_cols = [
        "masfem",
        "gender_mf",
        "wind",
        "min",
        "category",
        "ndam15",
        "log_ndam15",
        "year",
        "alldeaths",
        "log_alldeaths",
    ]
    print_section("Summary Statistics")
    print(df[numeric_cols].describe().T.to_string(float_format=lambda x: f"{x:,.4f}"))

    print_section("Distribution Checks")
    zero_share = (df["alldeaths"] == 0).mean()
    print(f"Share of storms with zero deaths: {zero_share:.3f}")
    print("Deaths quantiles:")
    print(df["alldeaths"].quantile([0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]).to_string())
    print("\nMasfem quantiles:")
    print(df["masfem"].quantile([0, 0.25, 0.5, 0.75, 1.0]).to_string())

    print_section("Correlations")
    corr_cols = ["masfem", "gender_mf", "wind", "min", "category", "log_ndam15", "year", "log_alldeaths"]
    print(df[corr_cols].corr(numeric_only=True).to_string(float_format=lambda x: f"{x:,.3f}"))

    # ------------------------------------------------------------------
    # 3) Classical statistical tests (bivariate + controlled)
    # ------------------------------------------------------------------
    print_section("Classical Statistical Tests")
    pearson_r, pearson_p = stats.pearsonr(df["masfem"], df["log_alldeaths"])
    spearman_rho, spearman_p = stats.spearmanr(df["masfem"], df["alldeaths"])
    print(f"Pearson r(masfem, log_alldeaths) = {pearson_r:.4f}, p = {pearson_p:.4g}")
    print(f"Spearman rho(masfem, alldeaths) = {spearman_rho:.4f}, p = {spearman_p:.4g}")

    # Bivariate OLS
    ols_biv = smf.ols("log_alldeaths ~ masfem", data=df).fit()
    print("\nBivariate OLS (log_alldeaths ~ masfem):")
    print(ols_biv.summary())

    # Controlled OLS
    controls_formula = "wind + min + category + log_ndam15 + year + C(source)"
    ols_ctrl = smf.ols(f"log_alldeaths ~ masfem + {controls_formula}", data=df).fit()
    print("\nControlled OLS:")
    print(ols_ctrl.summary())

    # Controlled count model: GLM Negative Binomial
    nb_ctrl = smf.glm(
        f"alldeaths ~ masfem + {controls_formula}",
        data=df,
        family=sm.families.NegativeBinomial(),
    ).fit()
    print("\nControlled GLM Negative Binomial:")
    print(nb_ctrl.summary())

    ols_beta = float(ols_ctrl.params["masfem"])
    ols_p = float(ols_ctrl.pvalues["masfem"])
    nb_beta = float(nb_ctrl.params["masfem"])
    nb_p = float(nb_ctrl.pvalues["masfem"])
    nb_pct_per_unit = float(np.expm1(nb_beta) * 100.0)

    # ------------------------------------------------------------------
    # 4) Interpretable modeling with agentic_imodels
    # ------------------------------------------------------------------
    print_section("Interpretable Regressors (agentic_imodels)")
    # Keep features numeric and aligned with controlled design.
    feature_cols = ["masfem", "wind", "min", "category", "log_ndam15", "year"]
    X = df[feature_cols].copy()
    y = df["log_alldeaths"].to_numpy()

    model_specs = [
        ("SmartAdditiveRegressor", SmartAdditiveRegressor()),
        ("HingeEBMRegressor", HingeEBMRegressor()),
        ("WinsorizedSparseOLSRegressor", WinsorizedSparseOLSRegressor()),
    ]

    model_texts = {}
    model_r2 = {}
    perm_masfem = {}

    print("Feature index mapping used by model printouts:")
    for i, col in enumerate(feature_cols):
        print(f"  x{i} -> {col}")

    for name, model in model_specs:
        print_section(f"{name} Fit")
        model.fit(X, y)
        y_hat = model.predict(X)
        r2 = float(r2_score(y, y_hat))
        model_r2[name] = r2
        print(f"In-sample R^2: {r2:.4f}")
        print(model)  # Required by instructions for interpretable form.
        model_texts[name] = str(model)

        pi = permutation_importance(model, X, y, n_repeats=30, random_state=0)
        masfem_imp = float(pi.importances_mean[0])
        perm_masfem[name] = masfem_imp
        print(f"Permutation importance (masfem): {masfem_imp:.6f}")

    # Model-specific evidence extraction
    smart = [m for n, m in model_specs if n == "SmartAdditiveRegressor"][0]
    smart_total_imp = float(np.sum(smart.feature_importances_))
    smart_masfem_imp = float(smart.feature_importances_[0])
    smart_rel_imp = smart_masfem_imp / smart_total_imp if smart_total_imp > 0 else 0.0
    smart_rank = int(np.argsort(-smart.feature_importances_).tolist().index(0) + 1)

    hinge_text = model_texts["HingeEBMRegressor"]
    hinge_masfem_active = active_in_hinge_text(hinge_text, "x0")

    wins = [m for n, m in model_specs if n == "WinsorizedSparseOLSRegressor"][0]
    wins_masfem_active = 0 in list(wins.support_)

    print_section("Evidence Synthesis")
    print(f"Controlled OLS masfem beta={ols_beta:.4f}, p={ols_p:.4g}")
    print(f"Controlled NegBin masfem beta={nb_beta:.4f}, p={nb_p:.4g}, percent change/unit={nb_pct_per_unit:.2f}%")
    print(f"SmartAdditive masfem relative importance={smart_rel_imp:.4f}, rank={smart_rank}")
    print(f"HingeEBM masfem active in displayed sparse equation: {hinge_masfem_active}")
    print(f"WinsorizedSparseOLS masfem selected (non-zero): {wins_masfem_active}")
    print("Permutation importance of masfem by model:")
    for n in perm_masfem:
        print(f"  {n}: {perm_masfem[n]:.6f}")

    # ------------------------------------------------------------------
    # 5) Calibrated Likert response (0-100) + explanation
    # ------------------------------------------------------------------
    # Scoring rubric from SKILL.md:
    # - Non-significant controlled tests + lasso/hinge zeroing + low importance => 0-15.
    weak_significance = (ols_p >= 0.10) and (nb_p >= 0.10)
    strong_zeroing = (not wins_masfem_active) and (not hinge_masfem_active)
    low_importance = smart_rank > 3 or smart_rel_imp < 0.10

    if weak_significance and strong_zeroing and low_importance:
        response = 12
    elif weak_significance and (strong_zeroing or low_importance):
        response = 22
    elif (ols_p < 0.05) and (nb_p < 0.05) and (wins_masfem_active or hinge_masfem_active):
        response = 82
    else:
        response = 40

    explanation = (
        f"Question: {question} "
        f"Bivariate association is near zero (Pearson r={pearson_r:.3f}, p={pearson_p:.3f}; "
        f"Spearman rho={spearman_rho:.3f}, p={spearman_p:.3f}). "
        f"With controls, masfem is not significant in OLS on log deaths "
        f"(beta={ols_beta:.4f}, p={ols_p:.3f}) nor in Negative Binomial for count deaths "
        f"(beta={nb_beta:.4f}, p={nb_p:.3f}, about {nb_pct_per_unit:.1f}% change per unit but uncertain). "
        f"Interpretable models also provide mostly null evidence: "
        f"WinsorizedSparseOLS excludes masfem (lasso-style zeroing), HingeEBM also excludes masfem in its sparse displayed equation, "
        f"and SmartAdditive gives only modest masfem importance (rank {smart_rank}/{len(feature_cols)}, "
        f"relative importance {smart_rel_imp:.3f}). "
        f"Across classical and interpretable analyses, severity/exposure controls (pressure, wind, damage, year) dominate fatalities, "
        f"so evidence for a robust femininity-name effect is weak."
    )

    result = {"response": int(response), "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=True)

    print_section("Final JSON Written To conclusion.txt")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
