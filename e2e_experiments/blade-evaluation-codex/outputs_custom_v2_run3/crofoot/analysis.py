import json
import warnings

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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def clip01(x: float) -> int:
    return int(np.clip(np.round(x), 0, 100))


def active_features_hinge_ebm(model: HingeEBMRegressor) -> set[int]:
    """Recover which original feature indices are active in stage-1 hinge-lasso."""
    coefs = np.asarray(model.lasso_.coef_)
    n_selected = len(model.selected_)
    active = set()
    for idx, coef in enumerate(coefs):
        if abs(coef) < 1e-6:
            continue
        if idx < n_selected:
            active.add(int(model.selected_[idx]))
        else:
            feat_idx, _knot, _direction = model.hinge_info_[idx - n_selected]
            active.add(int(model.selected_[feat_idx]))
    return active


def main() -> None:
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info["research_questions"][0]
    print("Research question:")
    print(question)
    print("\nLoading data...\n")

    df = pd.read_csv("crofoot.csv")

    # Features aligned with the research question.
    df["size_diff"] = df["n_focal"] - df["n_other"]
    df["size_ratio"] = df["n_focal"] / df["n_other"]
    df["dist_diff"] = df["dist_other"] - df["dist_focal"]
    df["dist_ratio"] = df["dist_other"] / df["dist_focal"]
    df["male_diff"] = df["m_focal"] - df["m_other"]
    df["female_diff"] = df["f_focal"] - df["f_other"]

    print("Data shape:", df.shape)
    print("Missing values by column:")
    print(df.isna().sum())

    print("\nOutcome distribution (win):")
    print(df["win"].value_counts().sort_index())
    print("Win rate:", round(float(df["win"].mean()), 4))

    print("\nSummary statistics:")
    print(df.describe(include="all").T)

    print("\nCorrelations with win (Pearson):")
    corr_cols = [
        "win",
        "size_diff",
        "size_ratio",
        "dist_diff",
        "dist_ratio",
        "male_diff",
        "female_diff",
        "dyad",
    ]
    print(df[corr_cols].corr(numeric_only=True)["win"].sort_values(ascending=False))

    print("\nBivariate tests:")
    pb_size = stats.pointbiserialr(df["win"], df["size_diff"])
    pb_dist = stats.pointbiserialr(df["win"], df["dist_diff"])
    t_size = stats.ttest_ind(
        df.loc[df["win"] == 1, "size_diff"],
        df.loc[df["win"] == 0, "size_diff"],
        equal_var=False,
    )
    t_dist = stats.ttest_ind(
        df.loc[df["win"] == 1, "dist_diff"],
        df.loc[df["win"] == 0, "dist_diff"],
        equal_var=False,
    )
    print(f"Point-biserial(win, size_diff): r={pb_size.statistic:.3f}, p={pb_size.pvalue:.4f}")
    print(f"Point-biserial(win, dist_diff): r={pb_dist.statistic:.3f}, p={pb_dist.pvalue:.4f}")
    print(f"Welch t-test size_diff by win: t={t_size.statistic:.3f}, p={t_size.pvalue:.4f}")
    print(f"Welch t-test dist_diff by win: t={t_dist.statistic:.3f}, p={t_dist.pvalue:.4f}")

    print("\nClassical regression tests (Binomial GLM / logit link):")
    glm_base = smf.glm("win ~ size_diff + dist_diff", data=df, family=sm.families.Binomial()).fit()
    glm_ctrl = smf.glm(
        "win ~ size_diff + dist_diff + male_diff + female_diff",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    glm_dyad = smf.glm(
        "win ~ size_diff + dist_diff + C(dyad)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    print("\n--- GLM: win ~ size_diff + dist_diff ---")
    print(glm_base.summary())
    print("\n--- GLM: + male/female composition controls ---")
    print(glm_ctrl.summary())
    print("\n--- GLM: + dyad fixed effects ---")
    print(glm_dyad.summary())

    # Interpretable models focused on the research variables and a structural control.
    model_features = ["size_diff", "dist_diff", "dyad"]
    X = df[model_features].to_numpy(dtype=float)
    y = df["win"].to_numpy(dtype=float)

    print("\nInterpretable model feature map:")
    for i, name in enumerate(model_features):
        print(f"  x{i} = {name}")

    print("\n=== SmartAdditiveRegressor (honest; shape + direction) ===")
    smart = SmartAdditiveRegressor().fit(X, y)
    print(smart)

    print("\n=== WinsorizedSparseOLSRegressor (honest sparse linear; zeroing evidence) ===")
    sparse = WinsorizedSparseOLSRegressor().fit(X, y)
    print(sparse)

    print("\n=== HingeEBMRegressor (high-rank decoupled benchmark) ===")
    hinge = HingeEBMRegressor().fit(X, y)
    print(hinge)

    # Pull interpretable evidence for score calibration.
    size_slope_smart = smart.linear_approx_.get(0, (0.0, 0.0, 0.0))[0]
    dist_slope_smart = smart.linear_approx_.get(1, (0.0, 0.0, 0.0))[0]
    size_imp_smart = float(smart.feature_importances_[0]) if len(smart.feature_importances_) > 0 else 0.0
    dist_imp_smart = float(smart.feature_importances_[1]) if len(smart.feature_importances_) > 1 else 0.0

    sparse_size = 0 in set(int(i) for i in sparse.support_)
    sparse_dist = 1 in set(int(i) for i in sparse.support_)

    hinge_active = active_features_hinge_ebm(hinge)
    hinge_size = 0 in hinge_active
    hinge_dist = 1 in hinge_active

    p_base_size = float(glm_base.pvalues["size_diff"])
    p_base_dist = float(glm_base.pvalues["dist_diff"])
    p_ctrl_size = float(glm_ctrl.pvalues["size_diff"])
    p_ctrl_dist = float(glm_ctrl.pvalues["dist_diff"])
    p_dyad_size = float(glm_dyad.pvalues["size_diff"])
    p_dyad_dist = float(glm_dyad.pvalues["dist_diff"])

    b_size = float(glm_base.params["size_diff"])
    b_dist = float(glm_base.params["dist_diff"])
    c_size = float(glm_ctrl.params["size_diff"])
    c_dist = float(glm_ctrl.params["dist_diff"])
    d_size = float(glm_dyad.params["size_diff"])
    d_dist = float(glm_dyad.params["dist_diff"])

    # Separate strength scores for each driver, then combine.
    size_score = 50.0
    dist_score = 50.0

    # Classical evidence (weights emphasize controlled models).
    size_score += 10 if p_base_size < 0.10 else (-4 if p_base_size > 0.20 else 0)
    size_score += 16 if p_ctrl_size < 0.05 else (8 if p_ctrl_size < 0.10 else -5)
    size_score += 8 if p_dyad_size < 0.10 else -6

    dist_score += 10 if p_base_dist < 0.10 else (-6 if p_base_dist > 0.20 else 0)
    dist_score += 12 if p_ctrl_dist < 0.05 else (5 if p_ctrl_dist < 0.10 else -8)
    dist_score += 8 if p_dyad_dist < 0.10 else -7

    # Direction consistency across GLMs.
    size_signs = np.sign([b_size, c_size, d_size])
    dist_signs = np.sign([b_dist, c_dist, d_dist])
    if np.all(size_signs == size_signs[0]):
        size_score += 5
    if np.all(dist_signs == dist_signs[0]):
        dist_score += 4

    # Interpretable model corroboration / null evidence.
    size_score += 7 if size_slope_smart > 0 else -7
    dist_score += 3 if abs(dist_slope_smart) > 1e-4 else -3

    size_score += 7 if sparse_size else -7
    dist_score += 7 if sparse_dist else -9

    size_score += 4 if hinge_size else -6
    dist_score += 4 if hinge_dist else -6

    # If SmartAdditive importance is tiny, discount that feature.
    if size_imp_smart < 0.03:
        size_score -= 6
    if dist_imp_smart < 0.03:
        dist_score -= 6

    size_score = float(np.clip(size_score, 0, 100))
    dist_score = float(np.clip(dist_score, 0, 100))

    # Question asks about BOTH relative group size and location influence.
    combined_score = 0.5 * size_score + 0.5 * dist_score
    response = clip01(combined_score)

    explanation = (
        f"Bivariate tests were weak (size_diff r={pb_size.statistic:.2f}, p={pb_size.pvalue:.3f}; "
        f"dist_diff r={pb_dist.statistic:.2f}, p={pb_dist.pvalue:.3f}). "
        f"In Binomial GLM without controls, size_diff was positive but not significant "
        f"(beta={b_size:.3f}, p={p_base_size:.3f}) and dist_diff was near zero/non-significant "
        f"(beta={b_dist:.4f}, p={p_base_dist:.3f}). "
        f"With male/female composition controls, size_diff became significant and positive "
        f"(beta={c_size:.3f}, p={p_ctrl_size:.3f}) but dist_diff remained non-significant "
        f"(beta={c_dist:.4f}, p={p_ctrl_dist:.3f}). "
        f"With dyad fixed effects, both were non-significant (size p={p_dyad_size:.3f}, dist p={p_dyad_dist:.3f}). "
        f"Interpretable models were mixed for size and mostly null for location: SmartAdditive showed a positive "
        f"size slope ({size_slope_smart:.4f}) with some nonlinear location pattern, WinsorizedSparseOLS retained "
        f"size_diff but zeroed dist_diff, and HingeEBM zeroed both. "
        f"Overall this supports at most a moderate, not robust, influence of relative group size and weak evidence "
        f"for contest location as an independent predictor."
    )

    output = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f)

    print("\nFinal calibrated scores:")
    print(f"size_score={size_score:.1f}, dist_score={dist_score:.1f}, combined={response}")
    print("Wrote conclusion.txt")


if __name__ == "__main__":
    main()
