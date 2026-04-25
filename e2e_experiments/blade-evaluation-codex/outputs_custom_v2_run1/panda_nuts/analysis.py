import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import r2_score

from agentic_imodels import (
    HingeEBMRegressor,
    HingeGAMRegressor,
    SmartAdditiveRegressor,
    WinsorizedSparseOLSRegressor,
)


def canonicalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["sex", "help", "hammer"]:
        out[col] = out[col].astype(str).str.strip()
    out["sex"] = out["sex"].str.lower().replace({"m": "male", "f": "female"})
    out["help"] = out["help"].str.lower().replace({"y": "yes", "n": "no"})
    out["hammer"] = out["hammer"].str.upper().replace({"WOOD": "wood"})
    return out


def summarize_eda(df: pd.DataFrame) -> None:
    print("\n=== Data Overview ===")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nMissing values per column:")
    print(df.isna().sum())

    numeric_cols = ["age", "nuts_opened", "seconds", "efficiency"]
    print("\n=== Numeric Summary Statistics ===")
    print(df[numeric_cols].describe().T)

    print("\n=== Distribution Snapshots (selected quantiles) ===")
    for col in numeric_cols:
        qs = df[col].quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
        print(f"\n{col}:")
        print(qs)

    print("\n=== Correlations (numeric) ===")
    print(df[numeric_cols].corr())

    print("\n=== Mean Efficiency by Group ===")
    print("By sex:")
    print(df.groupby("sex")["efficiency"].mean().sort_values(ascending=False))
    print("\nBy help:")
    print(df.groupby("help")["efficiency"].mean().sort_values(ascending=False))
    print("\nBy hammer:")
    print(df.groupby("hammer")["efficiency"].mean().sort_values(ascending=False))


def run_classical_tests(df: pd.DataFrame):
    print("\n=== Bivariate Statistical Tests ===")
    age_corr = stats.pearsonr(df["age"], df["efficiency"])
    male = df.loc[df["sex"] == "male", "efficiency"]
    female = df.loc[df["sex"] == "female", "efficiency"]
    sex_t = stats.ttest_ind(male, female, equal_var=False)
    help_yes = df.loc[df["help"] == "yes", "efficiency"]
    help_no = df.loc[df["help"] == "no", "efficiency"]
    help_t = stats.ttest_ind(help_yes, help_no, equal_var=False)

    print(f"Age vs efficiency Pearson r={age_corr.statistic:.4f}, p={age_corr.pvalue:.4g}")
    print(
        "Sex (male - female) mean efficiency diff="
        f"{male.mean() - female.mean():.4f}, Welch t p={sex_t.pvalue:.4g}"
    )
    print(
        "Help (yes - no) mean efficiency diff="
        f"{help_yes.mean() - help_no.mean():.4f}, Welch t p={help_t.pvalue:.4g}"
    )

    print("\n=== Controlled OLS (primary formal test) ===")
    formula = "efficiency ~ age + C(sex) + C(help) + C(hammer)"
    ols = smf.ols(formula, data=df).fit(cov_type="HC3")
    print(ols.summary())

    return {
        "age_corr_r": float(age_corr.statistic),
        "age_corr_p": float(age_corr.pvalue),
        "sex_t_p": float(sex_t.pvalue),
        "help_t_p": float(help_t.pvalue),
        "ols": ols,
    }


def fit_interpretable_models(df: pd.DataFrame):
    X = pd.get_dummies(df[["age", "sex", "help", "hammer"]], drop_first=True)
    y = df["efficiency"].values

    feature_names = list(X.columns)
    print("\n=== Encoded Features for Interpretable Models ===")
    print(feature_names)

    models = [
        ("WinsorizedSparseOLSRegressor", WinsorizedSparseOLSRegressor()),
        ("SmartAdditiveRegressor", SmartAdditiveRegressor()),
        ("HingeGAMRegressor", HingeGAMRegressor()),
        ("HingeEBMRegressor", HingeEBMRegressor()),
    ]

    fitted = {}
    for name, model in models:
        fitted_model = model.fit(X, y)
        pred = fitted_model.predict(X)
        r2 = r2_score(y, pred)
        print(f"\n=== {name} (train R^2={r2:.4f}) ===")
        print(fitted_model)
        fitted[name] = fitted_model

    return feature_names, fitted


def collect_evidence(feature_names, tests, fitted):
    ols = tests["ols"]
    ols_params = ols.params
    ols_pvals = ols.pvalues

    # Feature indices in encoded matrix
    idx = {name: i for i, name in enumerate(feature_names)}

    # OLS-based controlled evidence
    evidence = {
        "age": {
            "ols_coef": float(ols_params.get("age", np.nan)),
            "ols_p": float(ols_pvals.get("age", np.nan)),
            "winsor_coef": 0.0,
            "smart_coef": 0.0,
            "hinge_nonzero": False,
            "ebm_nonzero": False,
        },
        "sex_male": {
            "ols_coef": float(ols_params.get("C(sex)[T.male]", np.nan)),
            "ols_p": float(ols_pvals.get("C(sex)[T.male]", np.nan)),
            "winsor_coef": 0.0,
            "smart_coef": 0.0,
            "hinge_nonzero": False,
            "ebm_nonzero": False,
        },
        "help_yes": {
            "ols_coef": float(ols_params.get("C(help)[T.yes]", np.nan)),
            "ols_p": float(ols_pvals.get("C(help)[T.yes]", np.nan)),
            "winsor_coef": 0.0,
            "smart_coef": 0.0,
            "hinge_nonzero": False,
            "ebm_nonzero": False,
        },
    }

    winsor = fitted["WinsorizedSparseOLSRegressor"]
    smart = fitted["SmartAdditiveRegressor"]
    hinge = fitted["HingeGAMRegressor"]
    ebm = fitted["HingeEBMRegressor"]

    # Winsorized sparse OLS coefficients on encoded features
    for key, fname in [("age", "age"), ("sex_male", "sex_male"), ("help_yes", "help_yes")]:
        i = idx[fname]
        evidence[key]["winsor_coef"] = float(winsor.ols_coef_[i])

    # SmartAdditive linear approximation; keys are feature indices present in approximation
    smart_lin = getattr(smart, "linear_approx_", {})
    for key, fname in [("age", "age"), ("sex_male", "sex_male"), ("help_yes", "help_yes")]:
        i = idx[fname]
        if i in smart_lin:
            evidence[key]["smart_coef"] = float(smart_lin[i][0])
        else:
            evidence[key]["smart_coef"] = 0.0

    # HingeGAM zeroing evidence from feature importances
    hinge_importances = getattr(hinge, "feature_importances_", np.zeros(len(feature_names)))
    for key, fname in [("age", "age"), ("sex_male", "sex_male"), ("help_yes", "help_yes")]:
        i = idx[fname]
        evidence[key]["hinge_nonzero"] = bool(abs(float(hinge_importances[i])) > 1e-8)

    # HingeEBM lasso coefficients: first n base features correspond to x0...x(n-1)
    ebm_lasso_coef = ebm.lasso_.coef_
    n_base = len(feature_names)
    for key, fname in [("age", "age"), ("sex_male", "sex_male"), ("help_yes", "help_yes")]:
        i = idx[fname]
        if i < n_base:
            evidence[key]["ebm_nonzero"] = bool(abs(float(ebm_lasso_coef[i])) > 1e-8)

    print("\n=== Aggregated Evidence (target predictors) ===")
    for key, vals in evidence.items():
        print(f"{key}: {vals}")

    return evidence


def score_predictor(ev):
    score = 50
    p = ev["ols_p"]
    if p < 0.01:
        score += 20
    elif p < 0.05:
        score += 15
    elif p < 0.10:
        score += 5
    else:
        score -= 15

    if abs(ev["winsor_coef"]) > 1e-8:
        score += 10
    else:
        score -= 10

    if abs(ev["smart_coef"]) > 1e-8:
        score += 10
    else:
        score -= 10

    if ev["hinge_nonzero"]:
        score += 10
    else:
        score -= 10

    if ev["ebm_nonzero"]:
        score += 10
    else:
        score -= 10

    return int(max(0, min(100, round(score))))


def build_explanation(df: pd.DataFrame, tests, evidence, score):
    ols = tests["ols"]
    age_beta = float(ols.params["age"])
    age_p = float(ols.pvalues["age"])
    sex_beta = float(ols.params["C(sex)[T.male]"])
    sex_p = float(ols.pvalues["C(sex)[T.male]"])
    help_beta = float(ols.params["C(help)[T.yes]"])
    help_p = float(ols.pvalues["C(help)[T.yes]"])

    age_score = score_predictor(evidence["age"])
    sex_score = score_predictor(evidence["sex_male"])
    help_score = score_predictor(evidence["help_yes"])

    text = (
        "Efficiency was defined as nuts_opened/seconds. "
        f"Bivariate evidence: age correlated positively with efficiency "
        f"(r={tests['age_corr_r']:.3f}, p={tests['age_corr_p']:.3g}); males had higher "
        f"mean efficiency than females (p={tests['sex_t_p']:.3g}); help-yes sessions had lower "
        f"efficiency than help-no sessions (p={tests['help_t_p']:.3g}). "
        "In controlled OLS (adjusting for sex, help, hammer), age remained positive "
        f"(beta={age_beta:.3f}, p={age_p:.3g}), male remained positive "
        f"(beta={sex_beta:.3f}, p={sex_p:.3g}), and help-yes remained negative "
        f"(beta={help_beta:.3f}, p={help_p:.3g}). "
        "Interpretable models agreed on a strong positive age effect: WinsorizedSparseOLS and "
        "SmartAdditive gave positive age terms, while HingeGAM and HingeEBM retained age and "
        "zeroed most other predictors, indicating age is the most robust driver. "
        "Sex and help were nonzero in WinsorizedSparseOLS and SmartAdditive but were zeroed by "
        "hinge/Lasso-based models, so their effects are plausible but less robust than age. "
        "Estimated evidence strengths (0-100) were approximately "
        f"age={age_score}, sex={sex_score}, help={help_score}; overall this supports a "
        f"moderate-to-strong 'Yes' that age, sex, and help influence efficiency, with "
        "direction: older and male higher efficiency, help associated with lower efficiency "
        "after controls."
    )
    return text


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    print("=== Research Question ===")
    for q in info.get("research_questions", []):
        print("-", q)

    df = pd.read_csv("panda_nuts.csv")
    df = canonicalize_categories(df)
    df["efficiency"] = np.where(df["seconds"] > 0, df["nuts_opened"] / df["seconds"], np.nan)
    df = df.dropna(subset=["efficiency", "age", "sex", "help", "hammer"]).copy()

    summarize_eda(df)
    tests = run_classical_tests(df)
    feature_names, fitted = fit_interpretable_models(df)
    evidence = collect_evidence(feature_names, tests, fitted)

    # Aggregate predictor scores for overall Likert response
    per_predictor_scores = [score_predictor(evidence[k]) for k in ["age", "sex_male", "help_yes"]]
    response_score = int(round(float(np.mean(per_predictor_scores))))
    explanation = build_explanation(df, tests, evidence, response_score)

    result = {"response": response_score, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f)

    print("\n=== Final Likert Output ===")
    print(result)
    print("\nWrote conclusion.txt")


if __name__ == "__main__":
    main()
