import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from agentic_imodels import (
    HingeEBMRegressor,
    SmartAdditiveRegressor,
    WinsorizedSparseOLSRegressor,
)

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def summarize_feature_correlations(df_num: pd.DataFrame, target_col: str) -> pd.Series:
    corr = df_num.corr(numeric_only=True)[target_col].sort_values(ascending=False)
    return corr


def binary_counterfactual_effect(model, X: pd.DataFrame, col: str, lo: float = 0.0, hi: float = 1.0):
    x_lo = X.copy()
    x_hi = X.copy()
    x_lo[col] = lo
    x_hi[col] = hi
    pred_lo = model.predict(x_lo)
    pred_hi = model.predict(x_hi)
    diff = pred_hi - pred_lo
    return {
        "mean": float(np.mean(diff)),
        "median": float(np.median(diff)),
        "std": float(np.std(diff)),
    }


def feature_sensitivity(model, X: pd.DataFrame, binary_cols: set[str]) -> dict[str, dict[str, float]]:
    base = model.predict(X)
    out = {}
    for col in X.columns:
        x2 = X.copy()
        if col in binary_cols:
            x2[col] = 1.0 - x2[col]
        else:
            step = float(X[col].std())
            if step == 0 or np.isnan(step):
                out[col] = {"direction": 0.0, "magnitude": 0.0}
                continue
            lo = float(X[col].min())
            hi = float(X[col].max())
            x2[col] = np.clip(x2[col] + step, lo, hi)
        delta = model.predict(x2) - base
        out[col] = {
            "direction": float(np.mean(delta)),
            "magnitude": float(np.mean(np.abs(delta))),
        }
    return out


def rank_from_sensitivity(sensitivity: dict[str, dict[str, float]]) -> list[tuple[str, float]]:
    return sorted(
        [(k, v["magnitude"]) for k, v in sensitivity.items()],
        key=lambda x: x[1],
        reverse=True,
    )


def safe_rank(feature: str, ranked: list[tuple[str, float]]) -> int:
    names = [x[0] for x in ranked]
    return names.index(feature) + 1 if feature in names else len(ranked) + 1


def build_score(
    p_bivar: float,
    p_ols: float,
    p_glm: float,
    coef_glm: float,
    child_effects: dict[str, float],
    child_ranks: list[int],
    zeroed_by_sparse: bool,
) -> int:
    score = 50.0

    if p_glm < 0.001:
        score += 28
    elif p_glm < 0.01:
        score += 22
    elif p_glm < 0.05:
        score += 15
    elif p_glm < 0.10:
        score += 8
    else:
        score -= 15

    if p_ols < 0.05:
        score += 8
    else:
        score -= 5

    if p_bivar < 0.05:
        score += 6
    else:
        score -= 4

    neg_votes = sum(1 for v in child_effects.values() if v < -1e-4)
    pos_votes = sum(1 for v in child_effects.values() if v > 1e-4)

    if neg_votes >= 2 and coef_glm < 0:
        score += 12
    elif pos_votes >= 2 and coef_glm > 0:
        score -= 15
    else:
        score -= 4

    avg_rank = float(np.mean(child_ranks))
    if avg_rank <= 3:
        score += 10
    elif avg_rank <= 5:
        score += 3
    else:
        score -= 8

    if zeroed_by_sparse:
        score -= 16

    return int(np.clip(round(score), 0, 100))


def main() -> None:
    info_path = Path("info.json")
    data_path = Path("affairs.csv")

    info = json.loads(info_path.read_text())
    question = info.get("research_questions", ["Unknown question"])[0]

    df = pd.read_csv(data_path)

    print_section("Research Question")
    print(question)

    print_section("Data Overview")
    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")
    print("Missing values per column:")
    print(df.isna().sum())

    numeric_cols = [
        "affairs",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]

    print("\nSummary statistics (numeric):")
    print(df[numeric_cols].describe().T)

    print("\nOutcome distribution (affairs):")
    print(df["affairs"].value_counts().sort_index())
    print(f"Zero-rate: {(df['affairs'] == 0).mean():.3f}")

    print("\nChildren distribution:")
    print(df["children"].value_counts())

    df_model = df.copy()
    df_model["children_yes"] = (df_model["children"] == "yes").astype(float)
    df_model["gender_male"] = (df_model["gender"] == "male").astype(float)

    print_section("Correlations")
    corr_df = df_model[["affairs", "children_yes", "gender_male", "age", "yearsmarried", "religiousness", "education", "occupation", "rating"]]
    corr_with_target = summarize_feature_correlations(corr_df, "affairs")
    print("Correlation with affairs:")
    print(corr_with_target)

    print_section("Bivariate Test: Children vs Affairs")
    y_yes = df_model.loc[df_model["children_yes"] == 1, "affairs"]
    y_no = df_model.loc[df_model["children_yes"] == 0, "affairs"]

    t_stat, p_ttest = stats.ttest_ind(y_yes, y_no, equal_var=False)
    u_stat, p_mwu = stats.mannwhitneyu(y_yes, y_no, alternative="two-sided")
    r_pb, p_pb = stats.pointbiserialr(df_model["children_yes"], df_model["affairs"])

    print(f"Mean affairs (children=yes): {y_yes.mean():.4f}")
    print(f"Mean affairs (children=no):  {y_no.mean():.4f}")
    print(f"Welch t-test: t={t_stat:.4f}, p={p_ttest:.6g}")
    print(f"Mann-Whitney U: U={u_stat:.2f}, p={p_mwu:.6g}")
    print(f"Point-biserial correlation: r={r_pb:.4f}, p={p_pb:.6g}")

    print_section("Classical Controlled Models")
    formula = (
        "affairs ~ children_yes + gender_male + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )

    ols = smf.ols(formula=formula, data=df_model).fit(cov_type="HC3")
    print("OLS with HC3 robust SE:")
    print(ols.summary())

    mean_affairs = float(df_model["affairs"].mean())
    var_affairs = float(df_model["affairs"].var())
    overdispersion_ratio = var_affairs / mean_affairs if mean_affairs > 0 else np.nan
    print(f"\nCount outcome check: mean={mean_affairs:.4f}, variance={var_affairs:.4f}, var/mean={overdispersion_ratio:.4f}")

    poisson = smf.glm(
        formula=formula,
        data=df_model,
        family=sm.families.Poisson(),
    ).fit()
    nb = smf.glm(
        formula=formula,
        data=df_model,
        family=sm.families.NegativeBinomial(),
    ).fit()

    print("\nPoisson GLM coefficients:")
    print(poisson.summary().tables[1])

    print("\nNegative-Binomial GLM coefficients:")
    print(nb.summary().tables[1])

    print_section("Interpretable Models (agentic_imodels)")
    feature_cols = [
        "children_yes",
        "gender_male",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]
    X = df_model[feature_cols].astype(float)
    y = df_model["affairs"].astype(float)

    models = {
        "SmartAdditiveRegressor": SmartAdditiveRegressor(),
        "WinsorizedSparseOLSRegressor": WinsorizedSparseOLSRegressor(max_features=8, cv=5),
        "HingeEBMRegressor": HingeEBMRegressor(),
    }

    fitted = {}
    for name, model in models.items():
        model.fit(X, y)
        fitted[name] = model
        print(f"\n--- {name} ---")
        print(model)

    print_section("Interpretable Model Effects: Direction, Magnitude, Shape, Robustness")
    binary_cols = {"children_yes", "gender_male"}

    child_effects = {}
    child_ranks = []

    for name, model in fitted.items():
        child_eff = binary_counterfactual_effect(model, X, "children_yes", 0.0, 1.0)
        child_effects[name] = child_eff["mean"]

        sens = feature_sensitivity(model, X, binary_cols=binary_cols)
        ranked = rank_from_sensitivity(sens)
        rank_child = safe_rank("children_yes", ranked)
        child_ranks.append(rank_child)

        print(f"\n{name}:")
        print(
            f"  children_yes counterfactual effect (yes - no): "
            f"mean={child_eff['mean']:.4f}, median={child_eff['median']:.4f}, sd={child_eff['std']:.4f}"
        )
        print(f"  children_yes sensitivity rank: {rank_child}/{len(feature_cols)}")
        print("  Top feature sensitivities:")
        for f, mag in ranked[:5]:
            print(f"    {f}: {mag:.4f}")

    # Shape readout from SmartAdditive (best model for explicit shape)
    print("\nSmartAdditive shape details for children_yes:")
    smart = fitted["SmartAdditiveRegressor"]
    idx_children = feature_cols.index("children_yes")
    if hasattr(smart, "shape_functions_") and idx_children in smart.shape_functions_:
        thresholds, intervals = smart.shape_functions_[idx_children]
        print(f"  thresholds={thresholds}")
        print(f"  intervals={intervals}")
    if hasattr(smart, "linear_approx_") and idx_children in smart.linear_approx_:
        slope, offset, r2 = smart.linear_approx_[idx_children]
        print(f"  linear approximation: slope={slope:.4f}, offset={offset:.4f}, R2={r2:.4f}")

    # Sparse zeroing evidence from WinsorizedSparseOLS
    wso = fitted["WinsorizedSparseOLSRegressor"]
    child_idx = feature_cols.index("children_yes")
    zeroed_by_sparse = child_idx not in set(getattr(wso, "support_", []))
    print("\nWinsorizedSparseOLS sparse-selection evidence:")
    print(f"  selected feature indices: {list(getattr(wso, 'support_', []))}")
    print(f"  children_yes selected: {not zeroed_by_sparse}")

    # Formal inference anchor: Negative Binomial coefficient for children_yes
    coef_ols = float(ols.params["children_yes"])
    p_ols = float(ols.pvalues["children_yes"])

    coef_nb = float(nb.params["children_yes"])
    p_nb = float(nb.pvalues["children_yes"])

    bivar_p = float(p_ttest)

    likert = build_score(
        p_bivar=bivar_p,
        p_ols=p_ols,
        p_glm=p_nb,
        coef_glm=coef_nb,
        child_effects=child_effects,
        child_ranks=child_ranks,
        zeroed_by_sparse=zeroed_by_sparse,
    )

    explanation = (
        f"Question: {question.strip()} "
        f"Bivariate evidence is {'significant' if bivar_p < 0.05 else 'not significant'} "
        f"(Welch p={bivar_p:.3g}; mean affairs yes={y_yes.mean():.3f}, no={y_no.mean():.3f}). "
        f"With controls, OLS gives children_yes beta={coef_ols:.3f} (p={p_ols:.3g}) and "
        f"Negative-Binomial GLM gives beta={coef_nb:.3f} (p={p_nb:.3g}). "
        f"Across interpretable models, average counterfactual effect of children_yes is "
        f"{np.mean(list(child_effects.values())):.3f} affairs/year (yes-no), with per-model effects "
        f"{', '.join([f'{k}:{v:.3f}' for k, v in child_effects.items()])}. "
        f"children_yes average sensitivity rank is {np.mean(child_ranks):.2f} of {len(feature_cols)} features. "
        f"WinsorizedSparseOLS {'zeroed out children_yes (null evidence)' if zeroed_by_sparse else 'retained children_yes (non-zero contribution)'}. "
        f"Combining controlled significance, direction consistency, magnitude, shape readout, and sparse-null evidence yields a calibrated Likert score of {likert}/100."
    )

    payload = {"response": int(likert), "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(payload))

    print_section("Conclusion JSON")
    print(json.dumps(payload, indent=2))
    print("\nWrote conclusion.txt")


if __name__ == "__main__":
    main()
