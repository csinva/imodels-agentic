import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from agentic_imodels import (
    HingeEBMRegressor,
    SmartAdditiveRegressor,
    WinsorizedSparseOLSRegressor,
)
from scipy import stats
from sklearn.metrics import r2_score


def parse_support(support, n_features):
    """Normalize support output to zero-based selected column indices."""
    arr = np.asarray(support)
    if arr.size == 0:
        return []

    # Boolean mask case
    if arr.dtype == bool and arr.size == n_features:
        return [i for i, keep in enumerate(arr.tolist()) if keep]

    # Integer index list case
    idx = [int(x) for x in arr.tolist()]
    if min(idx) >= 0 and max(idx) <= n_features - 1:
        return idx
    if min(idx) >= 1 and max(idx) <= n_features:
        return [i - 1 for i in idx]
    return [i for i in idx if 0 <= i < n_features]


def main():
    info = json.loads(Path("info.json").read_text())
    question = info.get("research_questions", ["Unknown question"])[0]

    print("Research question:")
    print(question)
    print()

    df = pd.read_csv("hurricane.csv")
    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nMissing values:\n", df.isna().sum().to_string())

    # Core engineered variables for count outcome and interaction structure
    df["log_deaths"] = np.log1p(df["alldeaths"])
    df["log_ndam15"] = np.log1p(df["ndam15"])
    df["masfem_x_log_ndam15"] = df["masfem"] * df["log_ndam15"]

    print("\nOutcome distribution (alldeaths):")
    print(df["alldeaths"].describe().to_string())
    print("Skew(alldeaths):", float(df["alldeaths"].skew()))

    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr(numeric_only=True)["alldeaths"].sort_values(ascending=False)
    print("\nCorrelations with alldeaths:")
    print(corr.to_string())

    # Step 2: bivariate tests
    pearson_raw = stats.pearsonr(df["masfem"], df["alldeaths"])
    spearman_raw = stats.spearmanr(df["masfem"], df["alldeaths"])
    pearson_log = stats.pearsonr(df["masfem"], df["log_deaths"])

    print("\nBivariate tests for masfem -> deaths:")
    print(
        f"Pearson(masfem, alldeaths): r={pearson_raw.statistic:.3f}, p={pearson_raw.pvalue:.4f}"
    )
    print(
        f"Spearman(masfem, alldeaths): rho={spearman_raw.statistic:.3f}, p={spearman_raw.pvalue:.4f}"
    )
    print(
        f"Pearson(masfem, log1p(alldeaths)): r={pearson_log.statistic:.3f}, p={pearson_log.pvalue:.4f}"
    )

    # Step 2: controlled models (statsmodels)
    model_df = df[
        [
            "alldeaths",
            "log_deaths",
            "masfem",
            "log_ndam15",
            "min",
            "category",
            "wind",
            "year",
        ]
    ].dropna()

    ols_main = smf.ols(
        "log_deaths ~ masfem + min + category + wind + log_ndam15 + year",
        data=model_df,
    ).fit()

    ols_interaction = smf.ols(
        "log_deaths ~ masfem * log_ndam15 + min + category + wind + year",
        data=model_df,
    ).fit()

    nb_interaction = smf.glm(
        "alldeaths ~ masfem * log_ndam15 + min + category + wind + year",
        data=model_df,
        family=sm.families.NegativeBinomial(),
    ).fit()

    print("\nControlled OLS (main effect model) summary:")
    print(ols_main.summary())

    print("\nControlled OLS (interaction model) coefficient table:")
    print(ols_interaction.summary().tables[1])

    print("\nControlled Negative Binomial GLM (interaction model) coefficient table:")
    print(nb_interaction.summary().tables[1])

    # Step 3: agentic_imodels for shape/direction/robustness
    feature_cols = [
        "masfem",
        "log_ndam15",
        "masfem_x_log_ndam15",
        "min",
        "category",
        "wind",
        "year",
    ]
    feature_map = {f"x{i}": col for i, col in enumerate(feature_cols)}

    X = df[feature_cols].copy()
    for c in feature_cols:
        X[c] = X[c].fillna(X[c].median())
    y = df["log_deaths"]

    print("\nFeature index map used by model printouts:")
    for k, v in feature_map.items():
        print(f"  {k} -> {v}")

    model_classes = [
        SmartAdditiveRegressor,
        HingeEBMRegressor,
        WinsorizedSparseOLSRegressor,
    ]
    fitted = {}

    for cls in model_classes:
        m = cls()
        m.fit(X, y)
        fitted[cls.__name__] = m
        pred = m.predict(X)
        print(f"\n=== {cls.__name__} (train R^2={r2_score(y, pred):.3f}) ===")
        print(m)

    # Extract key interpretable evidence
    smart = fitted["SmartAdditiveRegressor"]
    hinge = fitted["HingeEBMRegressor"]
    winsor = fitted["WinsorizedSparseOLSRegressor"]

    smart_imp = {
        col: float(val) for col, val in zip(feature_cols, smart.feature_importances_)
    }
    smart_imp_ranked = sorted(smart_imp.items(), key=lambda kv: kv[1], reverse=True)

    hinge_base_coefs = {
        feature_cols[i]: float(hinge.lasso_.coef_[i]) for i in range(len(feature_cols))
    }

    winsor_selected_idx = parse_support(winsor.support_, len(feature_cols))
    winsor_selected_cols = [feature_cols[i] for i in winsor_selected_idx]
    winsor_coef_map = {}
    if len(winsor_selected_cols) == len(winsor.ols_coef_):
        winsor_coef_map = {
            col: float(coef) for col, coef in zip(winsor_selected_cols, winsor.ols_coef_)
        }

    print("\nModel-derived evidence summary:")
    print("SmartAdditive feature importances (descending):")
    for name, val in smart_imp_ranked:
        print(f"  {name}: {val:.4f}")

    print("\nHingeEBM base (lasso) coefficients for original features:")
    for name, coef in hinge_base_coefs.items():
        print(f"  {name}: {coef:.4f}")

    print("\nWinsorizedSparseOLS selected features and coefficients:")
    if winsor_coef_map:
        for name, coef in winsor_coef_map.items():
            print(f"  {name}: {coef:.4f}")
    else:
        print("  Could not map selected features to coefficients cleanly.")

    # Step 4: calibrated Likert score
    score = 50

    # Bivariate evidence
    if pearson_raw.pvalue > 0.1 and spearman_raw.pvalue > 0.1:
        score -= 15

    # Controlled main-effect evidence
    ols_main_coef = float(ols_main.params["masfem"])
    ols_main_p = float(ols_main.pvalues["masfem"])
    if ols_main_p > 0.1:
        score -= 15
    else:
        score += 10
    if ols_main_coef < 0:
        score -= 5

    # Interaction evidence (conditional pattern)
    ols_int_p = float(ols_interaction.pvalues["masfem:log_ndam15"])
    nb_int_p = float(nb_interaction.pvalues["masfem:log_ndam15"])
    if nb_int_p < 0.05:
        score += 10
    elif nb_int_p < 0.1:
        score += 5

    if ols_int_p < 0.05:
        score += 5
    elif ols_int_p < 0.1:
        score += 3

    # Zeroing / sparse evidence for null main effect
    masfem_zero_count = 0
    if abs(hinge_base_coefs.get("masfem", 0.0)) < 1e-6:
        masfem_zero_count += 1
    if "masfem" not in winsor_selected_cols:
        masfem_zero_count += 1
    if smart_imp.get("masfem", 0.0) < 0.1 * max(smart_imp.values()):
        masfem_zero_count += 1

    if masfem_zero_count >= 2:
        score -= 10

    # Consistency for interaction term across interpretable models
    interaction_supported = 0
    if smart_imp.get("masfem_x_log_ndam15", 0.0) >= np.median(list(smart_imp.values())):
        interaction_supported += 1
    if abs(hinge_base_coefs.get("masfem_x_log_ndam15", 0.0)) > 1e-4:
        interaction_supported += 1
    if "masfem_x_log_ndam15" in winsor_selected_cols:
        interaction_supported += 1
    if interaction_supported >= 2:
        score += 6

    score = int(np.clip(round(score), 0, 100))

    explanation = (
        f"Bivariate evidence for femininity alone is weak/non-significant "
        f"(Pearson r={pearson_raw.statistic:.3f}, p={pearson_raw.pvalue:.3f}; "
        f"Spearman rho={spearman_raw.statistic:.3f}, p={spearman_raw.pvalue:.3f}). "
        f"In controlled OLS on log deaths, the main masfem term is small and non-significant "
        f"(beta={ols_main_coef:.3f}, p={ols_main_p:.3f}). "
        f"A conditional effect appears with damage: masfem*log_ndam15 is marginal in OLS "
        f"(p={ols_int_p:.3f}) and significant in negative-binomial GLM (p={nb_int_p:.3f}). "
        f"Interpretable models agree that direct masfem is not a dominant standalone driver "
        f"(HingeEBM lasso base coef={hinge_base_coefs.get('masfem', 0.0):.4f}; "
        f"SmartAdditive importance={smart_imp.get('masfem', 0.0):.4f}; "
        f"WinsorizedSparseOLS selected={('masfem' in winsor_selected_cols)}), while the interaction term is retained/important across models. "
        f"Overall this supports at most a conditional, not strong general, femininity effect."
    )

    output = {"response": score, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(output))

    print("\nLikert response:", score)
    print("Wrote conclusion.txt")


if __name__ == "__main__":
    main()
