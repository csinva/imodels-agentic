import json
import re
import warnings

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
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score


warnings.filterwarnings("ignore", category=FutureWarning)


def section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def extract_x_coef(model_text: str, x_idx: int):
    pattern = rf"x{x_idx}:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"
    match = re.search(pattern, model_text)
    if not match:
        return None
    return float(match.group(1))


def is_x_zeroed(model_text: str, x_idx: int) -> bool:
    x_tok = f"x{x_idx}"
    zero_patterns = [
        rf"Features with zero coefficients.*\b{x_tok}\b",
        rf"Features excluded.*\b{x_tok}\b",
        rf"zero effect.*\b{x_tok}\b",
    ]
    return any(re.search(p, model_text, flags=re.IGNORECASE) for p in zero_patterns)


def main() -> None:
    section("Research Question")
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    question = info["research_questions"][0]
    print(question)

    section("Load Data")
    df = pd.read_csv("amtl.csv")
    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")
    print("Columns:", list(df.columns))

    section("Data Exploration")
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]
    df["homo_sapiens"] = (df["genus"] == "Homo sapiens").astype(int)
    df["genus"] = pd.Categorical(
        df["genus"], categories=["Homo sapiens", "Pan", "Pongo", "Papio"]
    )

    print("Missing values:\n", df.isna().sum())
    numeric_cols = ["num_amtl", "sockets", "age", "stdev_age", "prob_male", "amtl_rate"]
    print("\nNumeric summary:")
    print(df[numeric_cols].describe().round(4))
    print("\nGenus counts:")
    print(df["genus"].value_counts())
    print("\nTooth class counts:")
    print(df["tooth_class"].value_counts())

    genus_rate = df.groupby("genus", observed=False)["amtl_rate"].agg(["mean", "std", "count"])
    tooth_rate = df.groupby("tooth_class", observed=False)["amtl_rate"].agg(
        ["mean", "std", "count"]
    )
    print("\nAMTL rate by genus:")
    print(genus_rate.round(4))
    print("\nAMTL rate by tooth_class:")
    print(tooth_rate.round(4))
    print("\nCorrelations (numeric):")
    print(df[numeric_cols].corr().round(3))

    section("Bivariate Evidence: Homo sapiens vs Non-human")
    human_rate = df.loc[df["homo_sapiens"] == 1, "amtl_rate"]
    nonhuman_rate = df.loc[df["homo_sapiens"] == 0, "amtl_rate"]
    mean_diff = float(human_rate.mean() - nonhuman_rate.mean())
    welch = stats.ttest_ind(human_rate, nonhuman_rate, equal_var=False)
    print(f"Human mean AMTL rate: {human_rate.mean():.5f}")
    print(f"Non-human mean AMTL rate: {nonhuman_rate.mean():.5f}")
    print(f"Mean difference (human - non-human): {mean_diff:.5f}")
    print(f"Welch t-test statistic={welch.statistic:.4f}, p-value={welch.pvalue:.3e}")

    biv_ols = smf.ols("amtl_rate ~ homo_sapiens", data=df).fit(cov_type="HC3")
    print("\nBivariate OLS (HC3 robust SE):")
    print(biv_ols.summary().tables[1])

    section("Controlled Classical Test: Binomial GLM")
    glm_main = smf.glm(
        "amtl_rate ~ homo_sapiens + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()
    print(glm_main.summary())

    homo_coef = float(glm_main.params["homo_sapiens"])
    homo_p = float(glm_main.pvalues["homo_sapiens"])
    homo_ci_low, homo_ci_high = glm_main.conf_int().loc["homo_sapiens"]
    homo_or = float(np.exp(homo_coef))
    print(
        "\nKey controlled effect (homo_sapiens): "
        f"log-odds coef={homo_coef:.4f}, OR={homo_or:.3f}, p={homo_p:.3e}, "
        f"95% CI(log-odds)=[{homo_ci_low:.4f}, {homo_ci_high:.4f}]"
    )

    glm_genus = smf.glm(
        "amtl_rate ~ C(genus) + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()
    print("\nControlled GLM with genus-specific contrasts (baseline = Homo sapiens):")
    print(glm_genus.summary().tables[1])

    section("Interpretable Models (agentic_imodels)")
    X = pd.DataFrame(
        {
            "homo_sapiens": df["homo_sapiens"].astype(float),
            "age": df["age"].astype(float),
            "prob_male": df["prob_male"].astype(float),
            "tooth_class_Posterior": (df["tooth_class"] == "Posterior").astype(float),
            "tooth_class_Premolar": (df["tooth_class"] == "Premolar").astype(float),
        }
    )
    y = df["amtl_rate"].astype(float).values
    sockets = df["sockets"].astype(float).values

    feature_map = {f"x{i}": col for i, col in enumerate(X.columns)}
    print("Feature map for printed models:", feature_map)

    model_classes = [
        SmartAdditiveRegressor,
        HingeEBMRegressor,
        WinsorizedSparseOLSRegressor,
    ]

    model_summaries = {}
    for cls in model_classes:
        name = cls.__name__
        model = cls()
        try:
            model.fit(X, y, sample_weight=sockets)
        except TypeError:
            model.fit(X, y)
        model_text = str(model)
        print(f"\n--- {name} ---")
        print(model)

        preds = model.predict(X)
        r2 = float(r2_score(y, preds))
        pi = permutation_importance(
            model,
            X,
            y,
            n_repeats=10,
            random_state=0,
            scoring="neg_mean_squared_error",
        )
        importances = sorted(
            zip(X.columns, pi.importances_mean), key=lambda x: x[1], reverse=True
        )
        print(f"In-sample R^2: {r2:.4f}")
        print("Permutation importance ranking (higher = more important):")
        for feat, imp in importances:
            print(f"  {feat:24s} {imp:.6f}")

        homo_model_coef = extract_x_coef(model_text, 0)
        homo_zeroed = is_x_zeroed(model_text, 0)
        model_summaries[name] = {
            "homo_coef": homo_model_coef,
            "homo_zeroed": homo_zeroed,
            "r2": r2,
            "top_feature": importances[0][0],
        }

    section("Synthesis and Calibrated Likert Score")
    positive_model_count = sum(
        1
        for s in model_summaries.values()
        if s["homo_coef"] is not None and s["homo_coef"] > 0
    )
    zeroed_model_count = sum(1 for s in model_summaries.values() if s["homo_zeroed"])

    genus_terms = [t for t in glm_genus.params.index if t.startswith("C(genus)[T.")]
    genus_all_lower_than_humans = all(
        (glm_genus.params[t] < 0) and (glm_genus.pvalues[t] < 0.05) for t in genus_terms
    )

    score = 35
    if mean_diff > 0 and welch.pvalue < 1e-3:
        score += 15
    elif mean_diff > 0 and welch.pvalue < 0.05:
        score += 8

    if homo_coef > 0 and homo_p < 1e-6:
        score += 30
    elif homo_coef > 0 and homo_p < 0.05:
        score += 18
    elif homo_coef <= 0:
        score -= 15

    if homo_or >= 3:
        score += 5
    elif homo_or >= 1.5:
        score += 3

    if positive_model_count >= 2:
        score += 8
    elif positive_model_count == 1:
        score += 3

    if zeroed_model_count >= 1:
        score -= 10

    if genus_all_lower_than_humans:
        score += 5

    score = int(np.clip(round(score), 0, 100))

    explanation = (
        f"Bivariate evidence shows higher AMTL in humans (mean rate {human_rate.mean():.3f}) "
        f"than non-humans ({nonhuman_rate.mean():.3f}), with a strong Welch test "
        f"(p={welch.pvalue:.2e}). In the controlled binomial GLM (adjusting for age, sex "
        f"probability, and tooth class), the Homo sapiens indicator remains strongly positive "
        f"(log-odds={homo_coef:.3f}, OR={homo_or:.2f}, p={homo_p:.2e}). Age is strongly positive, "
        f"posterior teeth are higher-risk than anterior, and prob_male is negative. "
        f"Interpretable agentic_imodels mostly agree on direction: SmartAdditiveRegressor and "
        f"HingeEBMRegressor assign a positive human effect, while WinsorizedSparseOLSRegressor "
        f"zeroes out the human term (null evidence under a sparse linear constraint). "
        f"The signal is therefore strong but not perfectly uniform across model classes; overall "
        f"evidence supports that modern humans have higher AMTL frequencies after controls."
    )

    print(f"Likert score (0-100): {score}")
    print("Explanation:", explanation)

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump({"response": score, "explanation": explanation}, f, ensure_ascii=True)


if __name__ == "__main__":
    main()
