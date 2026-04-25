import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from agentic_imodels import (
    HingeEBMRegressor,
    SmartAdditiveRegressor,
    WinsorizedSparseOLSRegressor,
)


SEED = 42
RNG = np.random.default_rng(SEED)


def format_p(p: float) -> str:
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def counterfactual_dark_effect(model, X: pd.DataFrame, dark_col: str = "dark_skin") -> float:
    x1 = X.copy()
    x0 = X.copy()
    x1[dark_col] = 1
    x0[dark_col] = 0
    return float(np.mean(model.predict(x1) - model.predict(x0)))


def main() -> None:
    info = json.loads(Path("info.json").read_text())
    question = info["research_questions"][0]
    print("Research question:")
    print(question)
    print()

    df = pd.read_csv("soccer.csv")
    print(f"Loaded soccer.csv with shape: {df.shape}")

    # Core construction
    df["skin_tone"] = df[["rater1", "rater2"]].mean(axis=1)
    df["birthday_dt"] = pd.to_datetime(df["birthday"], format="%d.%m.%Y", errors="coerce")
    df["age_2013"] = 2013 - df["birthday_dt"].dt.year

    base = df[df["skin_tone"].notna() & (df["games"] > 0)].copy()
    base["red_rate"] = base["redCards"] / base["games"]
    base["dark_skin_50"] = (base["skin_tone"] > 0.5).astype(int)

    # Primary contrast for the question: dark (>=0.75) vs light (<=0.25)
    ext = base[(base["skin_tone"] <= 0.25) | (base["skin_tone"] >= 0.75)].copy()
    ext["dark_skin"] = (ext["skin_tone"] >= 0.75).astype(int)
    ext["red_any"] = (ext["redCards"] > 0).astype(int)

    for col in ["yellowCards", "yellowReds", "goals", "victories", "ties", "defeats"]:
        ext[f"{col}_pg"] = ext[col] / ext["games"]

    print("\n=== Data coverage ===")
    print(f"Rows with usable skin-tone ratings: {len(base):,}")
    print(f"Rows in dark-vs-light extreme contrast: {len(ext):,}")
    print(f"Dark share (extreme contrast): {ext['dark_skin'].mean():.3f}")

    print("\n=== Summary statistics (extreme contrast) ===")
    summary_cols = [
        "redCards",
        "red_rate",
        "skin_tone",
        "games",
        "yellowCards",
        "yellowReds",
        "goals",
        "height",
        "weight",
        "meanIAT",
        "meanExp",
    ]
    print(ext[summary_cols].describe().T[["mean", "std", "min", "max"]])

    print("\n=== Key distributions ===")
    print("redCards value counts:")
    print(ext["redCards"].value_counts().sort_index())
    print("\nred card rate by dark_skin:")
    print(ext.groupby("dark_skin")["red_rate"].mean())

    corr_cols = [
        "red_rate",
        "skin_tone",
        "games",
        "yellowCards",
        "yellowReds",
        "goals",
        "height",
        "weight",
        "meanIAT",
        "meanExp",
    ]
    print("\n=== Correlations with red_rate ===")
    corr = ext[corr_cols].corr(numeric_only=True)["red_rate"].sort_values(ascending=False)
    print(corr)

    # Bivariate test
    dark_rates = ext.loc[ext["dark_skin"] == 1, "red_rate"]
    light_rates = ext.loc[ext["dark_skin"] == 0, "red_rate"]
    t_stat, t_p = stats.ttest_ind(dark_rates, light_rates, equal_var=False, nan_policy="omit")

    print("\n=== Bivariate test ===")
    print(f"Welch t-test on red_rate (dark vs light): t={t_stat:.4f}, p={format_p(float(t_p))}")

    # Classical statistical tests with controls (Poisson count with exposure offset)
    bivar_formula = "redCards ~ dark_skin"
    ctrl_formula = (
        "redCards ~ dark_skin + yellowCards + yellowReds + goals + "
        "height + weight + age_2013 + meanIAT + meanExp + "
        "C(position) + C(leagueCountry)"
    )
    full_formula_dark50 = (
        "redCards ~ dark_skin_50 + yellowCards + yellowReds + goals + "
        "height + weight + age_2013 + meanIAT + meanExp + "
        "C(position) + C(leagueCountry)"
    )
    full_formula_cont = (
        "redCards ~ skin_tone + yellowCards + yellowReds + goals + "
        "height + weight + age_2013 + meanIAT + meanExp + "
        "C(position) + C(leagueCountry)"
    )

    bivar_glm = smf.glm(
        formula=bivar_formula,
        data=ext,
        family=sm.families.Poisson(),
        offset=np.log(ext["games"]),
    ).fit()

    ctrl_glm = smf.glm(
        formula=ctrl_formula,
        data=ext,
        family=sm.families.Poisson(),
        offset=np.log(ext["games"]),
    ).fit()

    full_dark_glm = smf.glm(
        formula=full_formula_dark50,
        data=base,
        family=sm.families.Poisson(),
        offset=np.log(base["games"]),
    ).fit()

    full_cont_glm = smf.glm(
        formula=full_formula_cont,
        data=base,
        family=sm.families.Poisson(),
        offset=np.log(base["games"]),
    ).fit()

    bivar_coef = float(bivar_glm.params["dark_skin"])
    bivar_p = float(bivar_glm.pvalues["dark_skin"])
    ctrl_coef = float(ctrl_glm.params["dark_skin"])
    ctrl_p = float(ctrl_glm.pvalues["dark_skin"])
    ctrl_irr = float(np.exp(ctrl_coef))

    full_dark_coef = float(full_dark_glm.params["dark_skin_50"])
    full_dark_p = float(full_dark_glm.pvalues["dark_skin_50"])

    full_cont_coef = float(full_cont_glm.params["skin_tone"])
    full_cont_p = float(full_cont_glm.pvalues["skin_tone"])

    print("\n=== Classical models (Poisson GLM with log(games) offset) ===")
    print(
        f"Extreme dark-vs-light bivariate: coef={bivar_coef:.4f}, "
        f"IRR={np.exp(bivar_coef):.3f}, p={format_p(bivar_p)}"
    )
    print(
        f"Extreme dark-vs-light controlled: coef={ctrl_coef:.4f}, "
        f"IRR={ctrl_irr:.3f}, p={format_p(ctrl_p)}"
    )
    print(
        f"Full sample threshold dark_skin_50 controlled: coef={full_dark_coef:.4f}, "
        f"IRR={np.exp(full_dark_coef):.3f}, p={format_p(full_dark_p)}"
    )
    print(
        f"Full sample continuous skin_tone controlled: coef={full_cont_coef:.4f}, "
        f"IRR={np.exp(full_cont_coef):.3f}, p={format_p(full_cont_p)}"
    )

    # Interpretable models (heavy use per instructions)
    model_df = ext.copy()
    feature_cols = [
        "dark_skin",
        "skin_tone",
        "height",
        "weight",
        "age_2013",
        "meanIAT",
        "meanExp",
        "yellowCards_pg",
        "yellowReds_pg",
        "goals_pg",
        "victories_pg",
        "ties_pg",
        "defeats_pg",
    ]
    X = model_df[feature_cols].copy()
    X = pd.concat(
        [
            X,
            pd.get_dummies(model_df["position"], prefix="pos", drop_first=True),
            pd.get_dummies(model_df["leagueCountry"], prefix="league", drop_first=True),
        ],
        axis=1,
    )

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    y = model_df["red_rate"].astype(float)

    # Keep agentic_imodels runtime stable on large dataset
    max_agentic_rows = 12000
    if len(X) > max_agentic_rows:
        idx = RNG.choice(len(X), size=max_agentic_rows, replace=False)
        Xm = X.iloc[idx].reset_index(drop=True)
        ym = y.iloc[idx].reset_index(drop=True)
    else:
        Xm = X.reset_index(drop=True)
        ym = y.reset_index(drop=True)

    print("\n=== agentic_imodels setup ===")
    print(f"Interpretable model sample size: {len(Xm):,}")
    print("Feature index mapping used by printed models:")
    for i, name in enumerate(Xm.columns):
        print(f"x{i}: {name}")

    smart = SmartAdditiveRegressor().fit(Xm, ym)
    hinge_ebm = HingeEBMRegressor().fit(Xm, ym)
    sparse_ols = WinsorizedSparseOLSRegressor().fit(Xm, ym)

    print("\n=== SmartAdditiveRegressor (honest) ===")
    print(smart)
    print("\n=== HingeEBMRegressor (decoupled) ===")
    print(hinge_ebm)
    print("\n=== WinsorizedSparseOLSRegressor (honest sparse linear) ===")
    print(sparse_ols)

    # Model-derived direction / magnitude / robustness checks
    smart_delta_dark = counterfactual_dark_effect(smart, Xm, dark_col="dark_skin")
    hinge_delta_dark = counterfactual_dark_effect(hinge_ebm, Xm, dark_col="dark_skin")
    sparse_delta_dark = counterfactual_dark_effect(sparse_ols, Xm, dark_col="dark_skin")

    smart_importance = pd.Series(
        smart.feature_importances_, index=Xm.columns, dtype=float
    ).sort_values(ascending=False)
    dark_rank = int(smart_importance.index.get_loc("dark_skin") + 1)
    dark_importance = float(smart_importance.loc["dark_skin"])

    selected_features = [Xm.columns[i] for i in sparse_ols.support_]
    dark_selected_by_sparse = "dark_skin" in selected_features
    sparse_dark_coef = 0.0
    if dark_selected_by_sparse:
        sparse_dark_idx = selected_features.index("dark_skin")
        sparse_dark_coef = float(sparse_ols.ols_coef_[sparse_dark_idx])

    # In HingeEBM, the first len(selected_) coefficients correspond to original features.
    dark_col_idx = int(np.where(Xm.columns == "dark_skin")[0][0])
    dark_linear_coef_hinge = 0.0
    if dark_col_idx in hinge_ebm.selected_:
        local_idx = int(np.where(hinge_ebm.selected_ == dark_col_idx)[0][0])
        dark_linear_coef_hinge = float(hinge_ebm.lasso_.coef_[local_idx])

    print("\n=== Interpretable-model evidence summary ===")
    print(
        f"SmartAdditive dark_skin average counterfactual effect on red_rate: {smart_delta_dark:.6f}"
    )
    print(
        f"HingeEBM dark_skin average counterfactual effect on red_rate: {hinge_delta_dark:.6f}"
    )
    print(
        f"WinsorizedSparseOLS dark_skin average counterfactual effect on red_rate: {sparse_delta_dark:.6f}"
    )
    print(
        f"SmartAdditive dark_skin importance rank: {dark_rank}/{len(smart_importance)} "
        f"(importance={dark_importance:.6f})"
    )
    print(f"WinsorizedSparseOLS selected features: {selected_features}")
    print(f"WinsorizedSparseOLS dark_skin coefficient: {sparse_dark_coef:.6f}")
    print(f"HingeEBM dark_skin linear/hinge coefficient (Lasso stage): {dark_linear_coef_hinge:.6f}")

    # Calibrated Likert scoring with explicit penalties for null/inconsistent evidence
    score = 35

    # Primary controlled evidence on directly relevant dark-vs-light contrast
    if ctrl_p < 0.01:
        score += 12
    elif ctrl_p < 0.05:
        score += 10
    elif ctrl_p < 0.10:
        score += 5
    else:
        score -= 8

    if bivar_p < 0.05:
        score += 8
    elif bivar_p < 0.10:
        score += 4
    else:
        score -= 4

    if ctrl_irr > 1.20:
        score += 6
    elif ctrl_irr > 1.08:
        score += 4
    elif ctrl_irr > 1.02:
        score += 2

    # Sensitivity penalty if broad threshold is weak
    if full_dark_p >= 0.05:
        score -= 8

    # Interpretable-model corroboration vs null evidence
    if smart_delta_dark > 0:
        score += 5
    else:
        score -= 5

    if dark_rank <= max(3, int(0.35 * len(smart_importance))):
        score += 3

    if dark_selected_by_sparse and sparse_dark_coef > 0:
        score += 8
    else:
        score -= 6

    if abs(dark_linear_coef_hinge) < 1e-6:
        score -= 5  # explicit null evidence from hinge/lasso zeroing
    else:
        score += 3

    if hinge_delta_dark > 0:
        score += 2

    response = int(np.clip(round(score), 0, 100))

    explanation = (
        f"The evidence is moderately supportive, not definitive. In the primary dark-vs-light "
        f"contrast (skin_tone >= 0.75 vs <= 0.25), dark skin predicts more red cards per game: "
        f"bivariate Poisson coef={bivar_coef:.3f} (p={format_p(bivar_p)}) and controlled Poisson "
        f"coef={ctrl_coef:.3f}, IRR={ctrl_irr:.3f} (p={format_p(ctrl_p)}). However, robustness is mixed: "
        f"using a broader threshold (skin_tone > 0.5) the controlled effect is not significant "
        f"(p={format_p(full_dark_p)}), though continuous skin tone remains positive and significant "
        f"(coef={full_cont_coef:.3f}, p={format_p(full_cont_p)}). Interpretable models also mix support and nulls: "
        f"SmartAdditive and WinsorizedSparseOLS both show positive dark_skin effects "
        f"(deltas {smart_delta_dark:.4f} and {sparse_delta_dark:.4f}; sparse model keeps dark_skin), "
        f"but HingeEBM's sparse linear/hinge stage zeros dark_skin (null evidence), with only a small positive "
        f"counterfactual effect from its residual corrector ({hinge_delta_dark:.4f}). Overall this indicates a real but "
        f"definition-sensitive positive association, so the conclusion is above neutral but below strong yes."
    )

    out = {"response": response, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(out, ensure_ascii=True))

    print("\n=== Final calibrated conclusion ===")
    print(json.dumps(out, indent=2))
    print("\nWrote conclusion.txt")


if __name__ == "__main__":
    main()
