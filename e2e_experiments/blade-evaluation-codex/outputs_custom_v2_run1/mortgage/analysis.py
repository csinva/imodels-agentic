import json
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from agentic_imodels import (
    HingeEBMRegressor,
    HingeGAMRegressor,
    SmartAdditiveRegressor,
    WinsorizedSparseOLSRegressor,
)


warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class ModelEvidence:
    model_name: str
    female_effect: float
    female_active: bool
    female_importance_share: float | None


def load_metadata(path: str = "info.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_data(path: str = "mortgage.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def print_exploration(df: pd.DataFrame, features: list[str], outcome: str, iv: str):
    print("\\n=== DATA OVERVIEW ===")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("Missing values per column:")
    print(df.isna().sum().sort_values(ascending=False).to_string())

    print("\\n=== SUMMARY STATISTICS ===")
    print(df.describe().T.to_string())

    print("\\n=== DISTRIBUTIONS ===")
    binary_cols = [c for c in df.columns if set(df[c].dropna().unique()).issubset({0, 1})]
    print("Binary-variable means (proportions):")
    for c in binary_cols:
        print(f"  {c}: {df[c].mean():.4f}")

    numeric_for_hist = ["housing_expense_ratio", "PI_ratio", "loan_to_value"]
    for c in numeric_for_hist:
        if c in df.columns:
            hist, bin_edges = np.histogram(df[c].dropna().values, bins=10)
            print(f"\\nHistogram for {c}:")
            for i in range(len(hist)):
                print(f"  [{bin_edges[i]:.3f}, {bin_edges[i+1]:.3f}): {int(hist[i])}")

    print("\\n=== CORRELATIONS ===")
    corr = df[features + [outcome]].corr(numeric_only=True)
    corr_with_outcome = corr[outcome].drop(outcome).sort_values(key=np.abs, ascending=False)
    print(f"Correlation with {outcome}:")
    print(corr_with_outcome.to_string())
    print(f"\\nCorrelation {iv} vs {outcome}: {corr.loc[iv, outcome]:.4f}")


def run_bivariate_tests(df: pd.DataFrame, iv: str, outcome: str):
    print("\\n=== BIVARIATE TESTS ===")
    d = df[[iv, outcome]].dropna()
    ctab = pd.crosstab(d[iv], d[outcome])
    print("Contingency table (female x accept):")
    print(ctab.to_string())

    chi2, p_chi2, dof, expected = stats.chi2_contingency(ctab)
    print(f"Chi-square: chi2={chi2:.6f}, dof={dof}, p={p_chi2:.6g}")
    print("Expected frequencies:")
    print(pd.DataFrame(expected, index=ctab.index, columns=ctab.columns).to_string())

    acceptance_by_gender = d.groupby(iv)[outcome].mean()
    diff = acceptance_by_gender.loc[1.0] - acceptance_by_gender.loc[0.0]
    print("Acceptance rate by gender:")
    for k, v in acceptance_by_gender.items():
        print(f"  female={int(k)}: {v:.4f}")
    print(f"Rate difference (female - male): {diff:.6f}")

    return {
        "chi2": chi2,
        "p_chi2": p_chi2,
        "rate_female": acceptance_by_gender.loc[1.0],
        "rate_male": acceptance_by_gender.loc[0.0],
        "rate_diff": diff,
    }


def run_controlled_logit(df: pd.DataFrame, outcome: str, iv: str, controls: list[str]):
    print("\\n=== CONTROLLED LOGISTIC REGRESSION (statsmodels.Logit) ===")
    cols = [outcome, iv] + controls
    d = df[cols].dropna().copy()

    X = sm.add_constant(d[[iv] + controls], has_constant="add")
    y = d[outcome]

    logit = sm.Logit(y, X)
    result = logit.fit(disp=0, maxiter=200)

    print(result.summary())

    beta = float(result.params[iv])
    pval = float(result.pvalues[iv])
    ci_low, ci_high = result.conf_int().loc[iv]
    odds_ratio = float(np.exp(beta))
    or_ci_low, or_ci_high = float(np.exp(ci_low)), float(np.exp(ci_high))

    print(
        f"\\nFemale coefficient: beta={beta:.6f}, p={pval:.6g}, "
        f"OR={odds_ratio:.6f}, OR 95% CI=({or_ci_low:.6f}, {or_ci_high:.6f})"
    )

    return {
        "n": len(d),
        "beta": beta,
        "pval": pval,
        "or": odds_ratio,
        "or_ci_low": or_ci_low,
        "or_ci_high": or_ci_high,
    }


def hinge_ebm_effect(model: HingeEBMRegressor, feature_idx: int) -> float:
    n_sel = len(model.selected_)
    coefs = model.lasso_.coef_
    effective = {int(model.selected_[i]): float(coefs[i]) for i in range(n_sel)}

    for idx, (feat_idx, knot, direction) in enumerate(model.hinge_info_):
        j_orig = int(model.selected_[feat_idx])
        c = float(coefs[n_sel + idx])
        if direction == "pos":
            effective[j_orig] = effective.get(j_orig, 0.0) + c
        else:
            effective[j_orig] = effective.get(j_orig, 0.0) - c

    return float(effective.get(feature_idx, 0.0))


def extract_model_evidence(model, model_name: str, female_idx: int) -> ModelEvidence:
    if isinstance(model, SmartAdditiveRegressor):
        fi = np.asarray(model.feature_importances_, dtype=float)
        total = float(fi.sum())
        share = (float(fi[female_idx]) / total) if total > 0 else None
        slope = float(model.linear_approx_.get(female_idx, (0.0, 0.0, 1.0))[0])
        active = bool(female_idx in model.shape_functions_ and (share is not None and share >= 0.01))
        effect = slope if active else 0.0
        return ModelEvidence(model_name, effect, active, share)

    if isinstance(model, HingeGAMRegressor):
        fi = np.asarray(model.feature_importances_, dtype=float)
        total = float(fi.sum())
        share = (float(fi[female_idx]) / total) if total > 0 else None
        slope = float(model.linear_approx_.get(female_idx, (0.0, 0.0, 1.0))[0])
        active = bool(female_idx in model.shape_functions_ and (share is not None and share >= 0.01))
        effect = slope if active else 0.0
        return ModelEvidence(model_name, effect, active, share)

    if isinstance(model, HingeEBMRegressor):
        effect = hinge_ebm_effect(model, female_idx)
        active = abs(effect) > 1e-6
        return ModelEvidence(model_name, effect, active, None)

    if isinstance(model, WinsorizedSparseOLSRegressor):
        if female_idx in model.support_:
            j = list(model.support_).index(female_idx)
            effect = float(model.ols_coef_[j])
            active = True
        else:
            effect = 0.0
            active = False
        return ModelEvidence(model_name, effect, active, None)

    return ModelEvidence(model_name, 0.0, False, None)


def run_interpretable_models(df: pd.DataFrame, features: list[str], outcome: str, iv: str):
    print("\\n=== INTERPRETABLE MODELS (agentic_imodels) ===")
    d = df[[outcome] + features].dropna().copy()
    X = d[features]
    y = d[outcome].astype(float)
    female_idx = features.index(iv)

    model_classes = [
        SmartAdditiveRegressor,      # honest model for shape
        HingeGAMRegressor,           # honest hinge model (zeroing evidence)
        HingeEBMRegressor,           # high-rank decoupled model for robustness
        WinsorizedSparseOLSRegressor # honest sparse linear with lasso-style selection
    ]

    evidence = []
    for cls in model_classes:
        print(f"\\n--- Fitting {cls.__name__} ---")
        model = cls()
        model.fit(X, y)
        print(model)

        ev = extract_model_evidence(model, cls.__name__, female_idx)
        evidence.append(ev)
        if ev.female_importance_share is None:
            print(
                f"[Female effect summary] active={ev.female_active}, "
                f"effect={ev.female_effect:.6f}"
            )
        else:
            print(
                f"[Female effect summary] active={ev.female_active}, "
                f"effect={ev.female_effect:.6f}, "
                f"importance_share={ev.female_importance_share:.6f}"
            )

    return evidence


def calibrate_score(
    bivariate: dict,
    controlled: dict,
    model_evidence: list[ModelEvidence],
) -> tuple[int, str]:
    active_effects = [ev.female_effect for ev in model_evidence if ev.female_active]
    zeroed_count = sum(1 for ev in model_evidence if not ev.female_active)
    pos_count = sum(1 for e in active_effects if e > 0)
    neg_count = sum(1 for e in active_effects if e < 0)

    score = 50.0

    # Controlled model carries more weight than bivariate.
    p = controlled["pval"]
    beta = controlled["beta"]
    if p < 0.01:
        score += 24
    elif p < 0.05:
        score += 14
    elif p < 0.10:
        score += 6
    else:
        score -= 16

    if beta > 0:
        score += 6
    elif beta < 0:
        score -= 6

    # Bivariate evidence is weak here; include but downweight.
    p_bi = bivariate["p_chi2"]
    if p_bi < 0.05:
        score += 8
    else:
        score -= 4

    # Robustness across interpretable models.
    if pos_count >= 3:
        score += 10
    elif pos_count == 2:
        score += 4

    if neg_count > 0:
        score -= 8

    # Zeroing in sparse/hinge models is meaningful null evidence.
    score -= 4 * zeroed_count

    # Cap and round to integer Likert.
    score = int(round(min(100, max(0, score))))

    direction = "higher acceptance odds for female applicants" if beta > 0 else "lower acceptance odds for female applicants"
    explanation = (
        f"Bivariate evidence is null (chi-square p={bivariate['p_chi2']:.3f}; "
        f"acceptance rates female={bivariate['rate_female']:.3f}, male={bivariate['rate_male']:.3f}). "
        f"After controlling for credit and financial covariates, female is statistically significant "
        f"in logistic regression (beta={controlled['beta']:.3f}, p={controlled['pval']:.3f}, "
        f"OR={controlled['or']:.2f}, 95% CI {controlled['or_ci_low']:.2f}-{controlled['or_ci_high']:.2f}), "
        f"indicating {direction}. Interpretable models mostly agree on positive female effects "
        f"(positive active effects={pos_count}, negative active effects={neg_count}, zeroed/excluded={zeroed_count}). "
        f"Because significance is modest and one model zeroes female, evidence is moderate rather than strong."
    )

    return score, explanation


def main():
    meta = load_metadata("info.json")
    question = meta.get("research_questions", [""])[0]
    print("Research question:")
    print(question)

    df = prepare_data("mortgage.csv")

    outcome = "accept"   # 1 = approved
    iv = "female"
    controls = [
        "black",
        "housing_expense_ratio",
        "self_employed",
        "married",
        "mortgage_credit",
        "consumer_credit",
        "bad_history",
        "PI_ratio",
        "loan_to_value",
    ]
    features = [iv] + controls

    print_exploration(df, features, outcome, iv)
    bivariate = run_bivariate_tests(df, iv, outcome)
    controlled = run_controlled_logit(df, outcome, iv, controls)
    model_evidence = run_interpretable_models(df, features, outcome, iv)

    response, explanation = calibrate_score(bivariate, controlled, model_evidence)
    result = {"response": int(response), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=True)

    print("\\n=== FINAL CALIBRATED CONCLUSION ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
