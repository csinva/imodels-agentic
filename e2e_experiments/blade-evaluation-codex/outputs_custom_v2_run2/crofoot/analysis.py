import json
import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import brier_score_loss, roc_auc_score

from agentic_imodels import (
    HingeEBMRegressor,
    SmartAdditiveRegressor,
    WinsorizedSparseOLSRegressor,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def pointbiserial_and_ttest(df: pd.DataFrame, target: str, columns: list[str]) -> pd.DataFrame:
    rows = []
    y = df[target].astype(float)
    for col in columns:
        x = df[col].astype(float)
        pb = stats.pointbiserialr(y, x)
        g1 = x[df[target] == 1]
        g0 = x[df[target] == 0]
        t_res = stats.ttest_ind(g1, g0, equal_var=False)
        rows.append(
            {
                "feature": col,
                "pointbiserial_r": pb.statistic,
                "pointbiserial_p": pb.pvalue,
                "mean_when_win1": float(g1.mean()),
                "mean_when_win0": float(g0.mean()),
                "ttest_stat": float(t_res.statistic),
                "ttest_p": float(t_res.pvalue),
            }
        )
    return pd.DataFrame(rows)


def summarize_glm(model, name: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "coef": model.params,
            "std_err": model.bse,
            "p_value": model.pvalues,
            "odds_ratio": np.exp(model.params),
        }
    )
    print_section(name)
    print(model.summary())
    print("\nCoefficient table:")
    print(out.round(6).to_string())
    return out


def extract_zeroed_features(model_text: str, feature_names: list[str]) -> set[str]:
    zeroed = set()
    marker_patterns = [
        r"Features excluded \(zero effect\):\s*(.+)",
        r"Features with zero coefficients \(excluded\):\s*(.+)",
    ]
    for pat in marker_patterns:
        m = re.search(pat, model_text)
        if not m:
            continue
        rhs = m.group(1).strip()
        if rhs.lower() in {"none", ""}:
            continue
        for raw in rhs.split(","):
            token = raw.strip().split()[0]
            if token.startswith("x") and token[1:].isdigit():
                idx = int(token[1:])
                if 0 <= idx < len(feature_names):
                    zeroed.add(feature_names[idx])
            elif token in feature_names:
                zeroed.add(token)
    return zeroed


def marginal_profile(
    model,
    X: pd.DataFrame,
    feature: str,
    n_grid: int = 11,
) -> dict:
    base = X.median(numeric_only=True).to_frame().T
    grid = np.quantile(X[feature], np.linspace(0.05, 0.95, n_grid))
    grid = np.unique(grid)

    preds = []
    for val in grid:
        row = base.copy()
        row.loc[:, feature] = val
        pred = float(model.predict(row)[0])
        preds.append(pred)
    preds = np.clip(np.array(preds), 0.0, 1.0)

    if len(grid) > 1:
        rho = stats.spearmanr(grid, preds).statistic
        rho = 0.0 if np.isnan(rho) else float(rho)
    else:
        rho = 0.0

    if abs(rho) < 0.2:
        direction = "mixed_or_flat"
    elif rho > 0:
        direction = "positive"
    else:
        direction = "negative"

    return {
        "feature": feature,
        "grid_min": float(grid.min()),
        "grid_max": float(grid.max()),
        "pred_min": float(preds.min()),
        "pred_max": float(preds.max()),
        "effect_span": float(preds.max() - preds.min()),
        "direction": direction,
        "spearman_rho": float(rho),
    }


def clipped_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    p = np.clip(y_pred, 0.0, 1.0)
    return {
        "auc": float(roc_auc_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
    }


def main() -> None:
    print_section("1) Load Data")
    df = pd.read_csv("crofoot.csv")

    # Variables directly tied to the research question.
    df["size_diff"] = df["n_focal"] - df["n_other"]
    df["dist_adv"] = df["dist_other"] - df["dist_focal"]

    # Additional controls describing composition.
    df["male_diff"] = df["m_focal"] - df["m_other"]
    df["female_diff"] = df["f_focal"] - df["f_other"]

    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")
    print("Columns:", list(df.columns))
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())

    print_section("2) Exploratory Analysis")
    key_cols = [
        "win",
        "size_diff",
        "dist_adv",
        "dist_focal",
        "dist_other",
        "male_diff",
        "female_diff",
    ]
    print("Summary statistics (key columns):")
    print(df[key_cols].describe().round(4).to_string())

    print("\nOutcome distribution (win):")
    print(df["win"].value_counts().sort_index().to_string())

    print("\nQuantiles for focal predictors:")
    print(df[["size_diff", "dist_adv"]].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(3).to_string())

    corr = df[key_cols].corr(numeric_only=True)
    print("\nCorrelation matrix (key columns):")
    print(corr.round(3).to_string())

    print_section("3) Bivariate Statistical Tests")
    biv = pointbiserial_and_ttest(
        df,
        target="win",
        columns=["size_diff", "dist_adv", "dist_focal", "dist_other", "male_diff", "female_diff"],
    )
    print(biv.round(6).to_string(index=False))

    print_section("4) Classical Binomial GLM (Logistic)")
    glm_core = smf.glm("win ~ size_diff + dist_adv", data=df, family=sm.families.Binomial()).fit()
    glm_location = smf.glm(
        "win ~ size_diff + dist_focal + dist_other",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    glm_controlled = smf.glm(
        "win ~ size_diff + dist_focal + dist_other + male_diff",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    core_tab = summarize_glm(glm_core, "GLM Core: win ~ size_diff + dist_adv")
    loc_tab = summarize_glm(glm_location, "GLM Location Split: win ~ size_diff + dist_focal + dist_other")
    ctrl_tab = summarize_glm(
        glm_controlled,
        "GLM Controlled: win ~ size_diff + dist_focal + dist_other + male_diff",
    )

    print_section("5) Interpretable Models (agentic_imodels)")
    y = df["win"].to_numpy()

    core_features = ["size_diff", "dist_adv"]
    ext_features = ["size_diff", "dist_adv", "dist_focal", "dist_other", "male_diff", "female_diff"]

    model_specs = [
        ("SmartAdditiveRegressor", SmartAdditiveRegressor(), core_features),
        ("HingeEBMRegressor", HingeEBMRegressor(), ext_features),
        ("WinsorizedSparseOLSRegressor", WinsorizedSparseOLSRegressor(), ext_features),
    ]

    model_results = {}

    for name, model, feats in model_specs:
        X = df[feats]
        model.fit(X, y)
        preds = model.predict(X)
        mtxt = str(model)
        zeros = extract_zeroed_features(mtxt, feats)
        metrics = clipped_metrics(y, preds)

        print(f"\n--- {name} ---")
        print("Feature map:", {f"x{i}": feat for i, feat in enumerate(feats)})
        print(model)  # Required by prompt: keep interpretable fitted form in output.
        print("Train metrics:", {k: round(v, 4) for k, v in metrics.items()})
        print("Zeroed features detected from printed form:", sorted(zeros) if zeros else "None")

        profiles = {}
        for feat in ["size_diff", "dist_adv"]:
            if feat in feats:
                profiles[feat] = marginal_profile(model, X, feat)
        print("Marginal profiles (predicted win probability while varying one feature):")
        for feat, prof in profiles.items():
            print(
                f"  {feat}: direction={prof['direction']}, "
                f"span={prof['effect_span']:.4f}, rho={prof['spearman_rho']:.3f}, "
                f"grid=[{prof['grid_min']:.3f}, {prof['grid_max']:.3f}]"
            )

        model_results[name] = {
            "features": feats,
            "text": mtxt,
            "zeroed": zeros,
            "metrics": metrics,
            "profiles": profiles,
        }

    print_section("6) Synthesis + Likert Calibration")
    p_size_core = float(core_tab.loc["size_diff", "p_value"])
    p_dist_adv_core = float(core_tab.loc["dist_adv", "p_value"])
    p_size_ctrl = float(ctrl_tab.loc["size_diff", "p_value"])
    p_dist_focal_ctrl = float(ctrl_tab.loc["dist_focal", "p_value"])
    p_dist_other_ctrl = float(ctrl_tab.loc["dist_other", "p_value"])

    zero_size = sum("size_diff" in r["zeroed"] for r in model_results.values())
    zero_dist_adv = sum("dist_adv" in r["zeroed"] for r in model_results.values())

    smart_profiles = model_results["SmartAdditiveRegressor"]["profiles"]
    smart_size_span = smart_profiles["size_diff"]["effect_span"]
    smart_dist_span = smart_profiles["dist_adv"]["effect_span"]

    # Evidence-calibrated score (0=No, 100=Yes)
    score = 50.0

    # Relative group size evidence.
    if p_size_core < 0.05:
        score += 8
    elif p_size_core < 0.10:
        score += 3
    else:
        score -= 5

    if p_size_ctrl < 0.05:
        score += 8
    elif p_size_ctrl < 0.10:
        score += 2
    else:
        score -= 6

    # Contest location evidence.
    strongest_location_p = min(p_dist_adv_core, p_dist_focal_ctrl, p_dist_other_ctrl)
    if strongest_location_p < 0.05:
        score += 8
    elif strongest_location_p < 0.10:
        score += 3
    else:
        score -= 8

    # If location is mostly weak in aggregate, slight penalty.
    if p_dist_adv_core > 0.20 and p_dist_other_ctrl > 0.20:
        score -= 4

    # Sparse/hinge zeroing counts as null evidence.
    score -= 4 * zero_size
    score -= 4 * zero_dist_adv

    # Nonlinear shape in SmartAdditive provides limited positive evidence.
    if smart_size_span > 0.08:
        score += 3
    if smart_dist_span > 0.08:
        score += 2

    score = int(np.clip(round(score), 0, 100))

    explanation = (
        f"Bivariate GLM gave weak evidence for relative group size (size_diff p={p_size_core:.3f}) "
        f"and little evidence for relative location advantage as dist_adv (p={p_dist_adv_core:.3f}). "
        f"In controlled GLM (size_diff + dist_focal + dist_other + male_diff), size_diff remained weak/non-"
        f"significant (p={p_size_ctrl:.3f}), while location was mixed: dist_focal was stronger "
        f"(p={p_dist_focal_ctrl:.3f}) but dist_other was not (p={p_dist_other_ctrl:.3f}). "
        f"Interpretable models were not robust for the target predictors: size_diff was zeroed in {zero_size} model(s), "
        f"and dist_adv was zeroed in {zero_dist_adv} model(s). SmartAdditive showed some nonlinear shape "
        f"(size span={smart_size_span:.3f}, location span={smart_dist_span:.3f}), but sparse models mostly dropped these terms. "
        f"Overall, evidence is mixed and leans weak rather than strongly affirmative."
    )

    conclusion = {"response": score, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)

    print(f"Final Likert score: {score}")
    print("Wrote conclusion.txt")


if __name__ == "__main__":
    main()
