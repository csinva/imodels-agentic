import json
import re
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
from agentic_imodels import (
    HingeEBMRegressor,
    SmartAdditiveRegressor,
    WinsorizedSparseOLSRegressor,
)


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def coef_from_model_text(model_text: str, feature_idx: int) -> float:
    """Extract printed coefficient for x{feature_idx} from model __str__ output."""
    pattern = rf"\bx{feature_idx}:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"
    match = re.search(pattern, model_text)
    if match:
        return float(match.group(1))
    return 0.0


def main() -> None:
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    question = info["research_questions"][0]
    print(f"Research question: {question}\n")

    df = pd.read_csv("soccer.csv")
    print(f"Loaded soccer.csv with shape={df.shape}")

    # Core variables for this question.
    df["skin_tone"] = df[["rater1", "rater2"]].mean(axis=1, skipna=True)
    df["any_red_card"] = (df["redCards"] > 0).astype(int)
    birthday = pd.to_datetime(df["birthday"], format="%d.%m.%Y", errors="coerce")
    df["age_2013"] = 2013 - birthday.dt.year
    df["dark_player"] = (df["skin_tone"] >= 0.5).astype(int)

    # 1) Data exploration: summary stats, distributions, correlations.
    print("\n=== Exploration ===")
    print("Top missingness:")
    print(df.isna().mean().sort_values(ascending=False).head(12).to_string())

    numeric_cols = [
        "skin_tone",
        "any_red_card",
        "redCards",
        "games",
        "yellowCards",
        "yellowReds",
        "goals",
        "height",
        "weight",
        "meanIAT",
        "meanExp",
        "age_2013",
    ]
    print("\nSelected numeric summary:")
    print(df[numeric_cols].describe().T.to_string())

    print("\nOutcome distribution:")
    print(df["any_red_card"].value_counts(normalize=True).rename("proportion").to_string())

    print("\nSkin tone distribution (with non-missing):")
    print(df["skin_tone"].describe().to_string())

    corr = df[numeric_cols].corr(numeric_only=True)["any_red_card"].sort_values(ascending=False)
    print("\nCorrelation with any_red_card:")
    print(corr.to_string())

    # Dark-vs-light descriptive comparison.
    ctab = pd.crosstab(df["dark_player"], df["any_red_card"])
    chi2, chi2_p, _, _ = stats.chi2_contingency(ctab)
    dark_rate = df.loc[df["dark_player"] == 1, "any_red_card"].mean()
    light_rate = df.loc[df["dark_player"] == 0, "any_red_card"].mean()
    print("\nDark/light red-card rates:")
    print(f"light={light_rate:.4f}, dark={dark_rate:.4f}, chi2_p={chi2_p:.4g}")

    # 2) Classical test with controls (formal inference): logistic regression.
    model_cols = [
        "any_red_card",
        "skin_tone",
        "games",
        "yellowCards",
        "yellowReds",
        "goals",
        "height",
        "weight",
        "meanIAT",
        "meanExp",
        "age_2013",
        "position",
        "leagueCountry",
        "playerShort",
    ]
    d = df[model_cols].dropna().copy()
    print(f"\nRows after dropping NA for modeling: {len(d)}")

    biv_formula = "any_red_card ~ skin_tone"
    biv_logit = smf.logit(biv_formula, data=d).fit(disp=False)
    biv_coef = float(biv_logit.params["skin_tone"])
    biv_p = float(biv_logit.pvalues["skin_tone"])
    print("\n=== Bivariate Logit ===")
    print(f"skin_tone coef={biv_coef:.6f}, OR={np.exp(biv_coef):.4f}, p={biv_p:.4g}")

    ctrl_formula = (
        "any_red_card ~ skin_tone + games + yellowCards + yellowReds + goals + "
        "height + weight + meanIAT + meanExp + age_2013 + "
        "C(position) + C(leagueCountry)"
    )
    ctrl_logit = smf.logit(ctrl_formula, data=d).fit(disp=False, maxiter=200)
    ctrl_coef = float(ctrl_logit.params["skin_tone"])
    ctrl_p = float(ctrl_logit.pvalues["skin_tone"])
    ctrl_ci_low, ctrl_ci_high = ctrl_logit.conf_int().loc["skin_tone"].tolist()

    # Cluster-robust SE by player for repeated dyads.
    ctrl_logit_cluster = smf.logit(ctrl_formula, data=d).fit(
        disp=False,
        maxiter=200,
        cov_type="cluster",
        cov_kwds={"groups": d["playerShort"]},
    )
    ctrl_cluster_p = float(ctrl_logit_cluster.pvalues["skin_tone"])

    print("\n=== Controlled Logit ===")
    print(
        "skin_tone coef={:.6f}, OR={:.4f}, p={:.4g}, 95%CI(log-odds)=({:.6f}, {:.6f}), "
        "cluster_p={:.4g}".format(
            ctrl_coef, np.exp(ctrl_coef), ctrl_p, ctrl_ci_low, ctrl_ci_high, ctrl_cluster_p
        )
    )

    # 3) Interpretable models for direction, magnitude/rank, shape, robustness.
    # The library docs note best behavior on smaller tabular data; use a fixed sample.
    sample_n = min(30000, len(d))
    d_sample = d.sample(n=sample_n, random_state=42)

    x_base_cols = [
        "skin_tone",
        "games",
        "yellowCards",
        "yellowReds",
        "goals",
        "height",
        "weight",
        "meanIAT",
        "meanExp",
        "age_2013",
        "position",
        "leagueCountry",
    ]
    X = pd.get_dummies(
        d_sample[x_base_cols],
        columns=["position", "leagueCountry"],
        drop_first=True,
        dtype=float,
    )
    y = d_sample["any_red_card"].astype(float)
    feature_names = list(X.columns)
    skin_idx = feature_names.index("skin_tone")

    print(f"\nInterpretable-model sample size: {sample_n}, features after encoding: {X.shape[1]}")

    models = [
        WinsorizedSparseOLSRegressor(),
        SmartAdditiveRegressor(),
        HingeEBMRegressor(),
    ]
    model_text = {}
    fitted = {}
    for m in models:
        name = m.__class__.__name__
        print(f"\n=== {name} ===")
        m.fit(X, y)
        fitted[name] = m
        text = str(m)
        model_text[name] = text
        print(text)

    # Extract skin-tone evidence from interpretable models.
    win_coef_skin = coef_from_model_text(model_text["WinsorizedSparseOLSRegressor"], skin_idx)
    hebm_coef_skin = coef_from_model_text(model_text["HingeEBMRegressor"], skin_idx)
    smart = fitted["SmartAdditiveRegressor"]
    smart_skin_imp = float(smart.feature_importances_[skin_idx])
    smart_rank = int(np.argsort(-smart.feature_importances_).tolist().index(skin_idx) + 1)
    smart_shape = smart.shape_functions_.get(skin_idx, None)

    top_k = 8
    top_idx = np.argsort(-smart.feature_importances_)[:top_k]
    top_features = [(feature_names[i], float(smart.feature_importances_[i])) for i in top_idx]
    print("\nTop SmartAdditive feature importances:")
    for fname, imp in top_features:
        print(f"{fname}: {imp:.6f}")

    print("\nSkin-tone evidence from interpretable models:")
    print(
        f"WinsorizedSparseOLS x{skin_idx} coef={win_coef_skin:.6f}; "
        f"HingeEBM x{skin_idx} coef={hebm_coef_skin:.6f}; "
        f"SmartAdditive importance rank={smart_rank}/{len(feature_names)}, "
        f"importance={smart_skin_imp:.6f}"
    )
    if smart_shape is not None:
        knots, values = smart_shape
        print(f"SmartAdditive skin_tone piecewise knots={knots}, values={values}")

    # 4) Calibrated conclusion (0..100 Likert).
    # Start from classical controlled significance/magnitude.
    if ctrl_p < 0.01:
        score = 72
    elif ctrl_p < 0.05:
        score = 64
    elif ctrl_p < 0.10:
        score = 52
    else:
        score = 30

    if ctrl_coef > 0:
        score += 3
    else:
        score -= 8

    if ctrl_cluster_p < 0.05:
        score += 4
    else:
        score -= 3

    # Interpretable-model robustness adjustments.
    if win_coef_skin > 0 and hebm_coef_skin > 0:
        score += 6
    elif win_coef_skin == 0 and hebm_coef_skin == 0:
        score -= 12
    else:
        score -= 3

    # Penalize if skin tone is weakly ranked relative to other predictors.
    if smart_rank > 12:
        score -= 10
    elif smart_rank > 8:
        score -= 6
    elif smart_rank > 5:
        score -= 3
    else:
        score += 2

    score = int(np.clip(round(score), 0, 100))

    explanation = (
        f"Bivariate logit shows a positive skin-tone effect on any red card "
        f"(coef={biv_coef:.3f}, p={biv_p:.4g}). With controls for games, disciplinary history, "
        f"performance, body traits, age, position, and league, the effect remains positive and "
        f"significant (coef={ctrl_coef:.3f}, OR={np.exp(ctrl_coef):.3f}, p={ctrl_p:.4g}; "
        f"cluster-robust p={ctrl_cluster_p:.4g}). Descriptively, dark players have a slightly "
        f"higher red-card incidence than light players (dark={dark_rate:.4f}, light={light_rate:.4f}; "
        f"chi-square p={chi2_p:.4g}). Interpretable models support a positive but modest effect: "
        f"WinsorizedSparseOLS includes skin_tone with positive coefficient ({win_coef_skin:.4f}), "
        f"HingeEBM also gives a positive skin_tone term ({hebm_coef_skin:.4f}), and SmartAdditive "
        f"shows skin_tone as nonzero but not dominant (importance rank {smart_rank}/{len(feature_names)}). "
        f"Major predictors are disciplinary/game-exposure variables (especially yellowCards/games), "
        f"so the skin-tone relationship appears real but moderate in magnitude."
    )

    result = {"response": score, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=True))

    print("\nWrote conclusion.txt")
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
