import json
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf

from agentic_imodels import (
    HingeEBMRegressor,
    HingeGAMRegressor,
    SmartAdditiveRegressor,
    WinsorizedSparseOLSRegressor,
)


warnings.filterwarnings("ignore")
np.random.seed(42)


@dataclass
class FeatureEvidence:
    name: str
    bivariate_p: float
    controlled_coef: float
    controlled_p: float
    honest_effects: dict


def describe_data(df: pd.DataFrame) -> None:
    print("\n=== Data Overview ===")
    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")
    print("\nHead:")
    print(df.head())

    print("\nNumeric summary:")
    print(df[["age", "nuts_opened", "seconds", "efficiency"]].describe().T)

    print("\nCategorical distributions:")
    for col in ["sex", "help", "hammer", "chimpanzee"]:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False).sort_index())

    print("\nCorrelations (numeric variables):")
    corr_cols = ["age", "sex_m", "help_y", "nuts_opened", "seconds", "efficiency"]
    print(df[corr_cols].corr().round(3))

    print("\nMean efficiency by group:")
    print(df.groupby("sex")["efficiency"].mean().rename("eff_by_sex"))
    print(df.groupby("help")["efficiency"].mean().rename("eff_by_help"))
    print(df.groupby("hammer")["efficiency"].mean().rename("eff_by_hammer"))


def bivariate_tests(df: pd.DataFrame) -> dict:
    out = {}

    pear = stats.pearsonr(df["age"], df["efficiency"])
    spear = stats.spearmanr(df["age"], df["efficiency"])
    out["age_pearson_r"] = float(pear.statistic)
    out["age_pearson_p"] = float(pear.pvalue)
    out["age_spearman_rho"] = float(spear.statistic)
    out["age_spearman_p"] = float(spear.pvalue)

    eff_m = df.loc[df["sex_m"] == 1, "efficiency"]
    eff_f = df.loc[df["sex_m"] == 0, "efficiency"]
    sex_t = stats.ttest_ind(eff_m, eff_f, equal_var=False)
    out["sex_t_stat"] = float(sex_t.statistic)
    out["sex_t_p"] = float(sex_t.pvalue)

    eff_help = df.loc[df["help_y"] == 1, "efficiency"]
    eff_nohelp = df.loc[df["help_y"] == 0, "efficiency"]
    help_t = stats.ttest_ind(eff_help, eff_nohelp, equal_var=False)
    out["help_t_stat"] = float(help_t.statistic)
    out["help_t_p"] = float(help_t.pvalue)

    print("\n=== Bivariate Tests ===")
    print(f"Age vs efficiency (Pearson): r={out['age_pearson_r']:.3f}, p={out['age_pearson_p']:.4g}")
    print(f"Age vs efficiency (Spearman): rho={out['age_spearman_rho']:.3f}, p={out['age_spearman_p']:.4g}")
    print(f"Sex (male vs female) efficiency t-test: t={out['sex_t_stat']:.3f}, p={out['sex_t_p']:.4g}")
    print(f"Help (yes vs no) efficiency t-test: t={out['help_t_stat']:.3f}, p={out['help_t_p']:.4g}")

    return out


def fit_classical_model(df: pd.DataFrame):
    formula = "efficiency ~ age + sex_m + help_y + C(hammer)"
    ols = smf.ols(formula, data=df).fit()
    ols_cluster = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["chimpanzee"]}
    )

    print("\n=== Classical Model: OLS with Controls (clustered by chimpanzee) ===")
    print(ols_cluster.summary())

    return ols, ols_cluster


def extract_hinge_ebm_effective_coefs(model: HingeEBMRegressor) -> dict:
    coefs = model.lasso_.coef_
    n_sel = len(model.selected_)

    effective = {}
    for i in range(n_sel):
        j = int(model.selected_[i])
        effective[j] = float(coefs[i])

    for idx, (feat_idx, knot, direction) in enumerate(model.hinge_info_):
        j = int(model.selected_[feat_idx])
        c = float(coefs[n_sel + idx])
        if abs(c) < 1e-8:
            continue
        if direction == "pos":
            effective[j] = effective.get(j, 0.0) + c
        else:
            effective[j] = effective.get(j, 0.0) - c

    for k in list(effective):
        if abs(effective[k]) < 1e-8:
            del effective[k]

    return effective


def fit_agentic_models(X: pd.DataFrame, y: pd.Series):
    model_classes = [
        SmartAdditiveRegressor,
        HingeGAMRegressor,
        WinsorizedSparseOLSRegressor,
        HingeEBMRegressor,
    ]

    print("\n=== agentic_imodels Fits ===")
    print("Feature index mapping for printed equations:")
    for i, c in enumerate(X.columns):
        print(f"  x{i} -> {c}")

    fitted = {}
    for cls in model_classes:
        name = cls.__name__
        model = cls()
        model.fit(X, y)
        fitted[name] = model

        print(f"\n--- {name} ---")
        print(model)

    return fitted


def collect_honest_effects(models: dict, feature_index: dict) -> dict:
    effects = {f: {} for f in feature_index}

    smart = models.get("SmartAdditiveRegressor")
    if smart is not None:
        for f, j in feature_index.items():
            slope = 0.0
            if hasattr(smart, "linear_approx_") and j in smart.linear_approx_:
                slope = float(smart.linear_approx_[j][0])
            effects[f]["SmartAdditiveRegressor"] = slope

    hinge = models.get("HingeGAMRegressor")
    if hinge is not None:
        for f, j in feature_index.items():
            slope = 0.0
            if hasattr(hinge, "linear_approx_") and j in hinge.linear_approx_:
                slope = float(hinge.linear_approx_[j][0])
            effects[f]["HingeGAMRegressor"] = slope

    sparse = models.get("WinsorizedSparseOLSRegressor")
    if sparse is not None:
        support = list(getattr(sparse, "support_", []))
        coefs = list(getattr(sparse, "ols_coef_", []))
        support_to_coef = {int(j): float(c) for j, c in zip(support, coefs)}
        for f, j in feature_index.items():
            effects[f]["WinsorizedSparseOLSRegressor"] = support_to_coef.get(j, 0.0)

    ebm = models.get("HingeEBMRegressor")
    if ebm is not None:
        ebm_effects = extract_hinge_ebm_effective_coefs(ebm)
        for f, j in feature_index.items():
            effects[f]["HingeEBMRegressor_display"] = float(ebm_effects.get(j, 0.0))

    print("\n=== Extracted Feature Effects from agentic_imodels ===")
    for f, eff_dict in effects.items():
        vals = ", ".join([f"{k}: {v:+.4f}" for k, v in eff_dict.items()])
        print(f"{f}: {vals}")

    return effects


def calibrate_feature_score(controlled_p: float, bivariate_p: float, honest_effects: dict, is_small_group: bool = False) -> float:
    if controlled_p < 0.01:
        classical = 1.0
    elif controlled_p < 0.05:
        classical = 0.85
    elif controlled_p < 0.10:
        classical = 0.60
    else:
        classical = 0.25

    nz_effects = [v for v in honest_effects.values() if abs(v) > 1e-6]
    model_strength = len(nz_effects) / max(1, len(honest_effects))

    if bivariate_p < 0.01:
        biv = 1.0
    elif bivariate_p < 0.05:
        biv = 0.8
    elif bivariate_p < 0.10:
        biv = 0.5
    else:
        biv = 0.2

    signs = np.sign(nz_effects)
    if len(signs) > 1 and len(set(signs)) > 1:
        consistency_penalty = 0.75
    else:
        consistency_penalty = 1.0

    score = 100.0 * (0.50 * classical + 0.30 * model_strength + 0.20 * biv)
    score *= consistency_penalty

    if is_small_group:
        score *= 0.90

    return float(np.clip(score, 0, 100))


def main() -> None:
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info["research_questions"][0]
    print("Research question:")
    print(question)

    df = pd.read_csv("panda_nuts.csv")

    # Encode focal binary predictors consistently.
    df["sex_m"] = (df["sex"].astype(str).str.lower() == "m").astype(int)
    df["help_y"] = (df["help"].astype(str).str.lower() == "y").astype(int)

    # Outcome: nut-cracking efficiency (nuts opened per second).
    df["efficiency"] = df["nuts_opened"] / df["seconds"]

    describe_data(df)
    biv = bivariate_tests(df)
    ols, ols_cluster = fit_classical_model(df)

    X = pd.get_dummies(df[["age", "sex", "help", "hammer"]], drop_first=True, dtype=float)
    y = df["efficiency"]

    models = fit_agentic_models(X, y)

    if len(models) < 2:
        raise RuntimeError("Need at least two fitted agentic_imodels models.")

    feature_index = {
        "age": int(list(X.columns).index("age")),
        "sex_m": int(list(X.columns).index("sex_m")),
        "help_y": int(list(X.columns).index("help_y")),
    }

    effects = collect_honest_effects(models, feature_index)

    # Focus evidence on honest models for zeroing/sign checks.
    honest_model_names = [
        "SmartAdditiveRegressor",
        "HingeGAMRegressor",
        "WinsorizedSparseOLSRegressor",
    ]

    age_honest = {k: effects["age"][k] for k in honest_model_names if k in effects["age"]}
    sex_honest = {k: effects["sex_m"][k] for k in honest_model_names if k in effects["sex_m"]}
    help_honest = {k: effects["help_y"][k] for k in honest_model_names if k in effects["help_y"]}

    age_score = calibrate_feature_score(
        controlled_p=float(ols_cluster.pvalues["age"]),
        bivariate_p=float(biv["age_pearson_p"]),
        honest_effects=age_honest,
        is_small_group=False,
    )
    sex_score = calibrate_feature_score(
        controlled_p=float(ols_cluster.pvalues["sex_m"]),
        bivariate_p=float(biv["sex_t_p"]),
        honest_effects=sex_honest,
        is_small_group=False,
    )
    help_score = calibrate_feature_score(
        controlled_p=float(ols_cluster.pvalues["help_y"]),
        bivariate_p=float(biv["help_t_p"]),
        honest_effects=help_honest,
        is_small_group=(int(df["help_y"].sum()) < 10),
    )

    overall_score = int(round(float(np.mean([age_score, sex_score, help_score]))))

    explanation = (
        "Efficiency was modeled as nuts_opened/seconds. In the controlled OLS with "
        "clustered SEs by chimpanzee and hammer controls, age had a positive coefficient "
        f"(beta={ols_cluster.params['age']:.3f}, p={ols_cluster.pvalues['age']:.3f}), male sex had "
        f"higher efficiency than female (beta={ols_cluster.params['sex_m']:.3f}, p={ols_cluster.pvalues['sex_m']:.3f}), "
        f"and receiving help was associated with lower efficiency (beta={ols_cluster.params['help_y']:.3f}, "
        f"p={ols_cluster.pvalues['help_y']:.3f}). Bivariate tests were also significant for age "
        f"(Pearson r={biv['age_pearson_r']:.3f}, p={biv['age_pearson_p']:.3g}), sex (p={biv['sex_t_p']:.3g}), "
        f"and help (p={biv['help_t_p']:.3g}). Interpretable models were mixed but informative: "
        "SmartAdditive and WinsorizedSparseOLS retained non-zero effects for age/sex/help, while "
        "HingeGAM zeroed sex/help and kept age, and HingeEBM display was sparse. This yields moderate-to-strong "
        "overall evidence that age, sex, and help relate to nut-cracking efficiency, with robustness caveats "
        "for help due to only 7 helped sessions and some model disagreement on non-age predictors."
    )

    payload = {
        "response": overall_score,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True))

    print("\n=== Final Likert Output ===")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
