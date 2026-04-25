import json
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

from agentic_imodels import (
    HingeEBMRegressor,
    HingeGAMRegressor,
    WinsorizedSparseOLSRegressor,
)


def safe_corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def counterfactual_effect(model, X, feature, low, high):
    x_low = X.copy()
    x_high = X.copy()
    x_low[feature] = low
    x_high[feature] = high
    pred_low = model.predict(x_low.values)
    pred_high = model.predict(x_high.values)
    return float(np.mean(pred_high - pred_low))


def extract_hinge_ebm_effective_coefs(model):
    coefs = model.lasso_.coef_
    n_sel = len(model.selected_)
    eff = {}
    for i in range(n_sel):
        j_orig = int(model.selected_[i])
        eff[j_orig] = float(coefs[i])
    for idx, (feat_idx, _, direction) in enumerate(model.hinge_info_):
        j_orig = int(model.selected_[feat_idx])
        c = float(coefs[n_sel + idx])
        if abs(c) < 1e-8:
            continue
        if direction == "pos":
            eff[j_orig] = eff.get(j_orig, 0.0) + c
        else:
            eff[j_orig] = eff.get(j_orig, 0.0) - c
    return eff


def top_features_from_dict(feature_names, coef_dict, k=5):
    pairs = []
    for j, v in coef_dict.items():
        name = feature_names[j]
        pairs.append((name, float(v)))
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)
    return pairs[:k]


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info["research_questions"][0]
    print("Research question:")
    print(question)
    print("\nLoading amtl.csv ...")

    df = pd.read_csv("amtl.csv")
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]
    df["non_amtl"] = df["sockets"] - df["num_amtl"]

    print("\n=== Data overview ===")
    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print("\nNumeric summary:")
    print(
        df[
            [
                "num_amtl",
                "sockets",
                "amtl_rate",
                "age",
                "stdev_age",
                "prob_male",
                "is_human",
            ]
        ]
        .describe()
        .T.round(4)
        .to_string()
    )
    print("\nGenus counts:")
    print(df["genus"].value_counts().to_string())
    print("\nTooth class counts:")
    print(df["tooth_class"].value_counts().to_string())
    print("\nAMTL rate by genus:")
    print(df.groupby("genus")["amtl_rate"].mean().sort_values(ascending=False).round(4).to_string())
    print("\nAMTL rate by tooth class:")
    print(df.groupby("tooth_class")["amtl_rate"].mean().sort_values(ascending=False).round(4).to_string())

    corr_df = pd.DataFrame(
        {
            "amtl_rate": df["amtl_rate"],
            "is_human": df["is_human"],
            "age": df["age"],
            "prob_male": df["prob_male"],
            "stdev_age": df["stdev_age"],
            "num_amtl": df["num_amtl"],
            "sockets": df["sockets"],
        }
    )
    print("\nCorrelation matrix (numeric):")
    print(corr_df.corr().round(3).to_string())

    print("\n=== Bivariate tests ===")
    human = df["is_human"] == 1
    t_res = stats.ttest_ind(
        df.loc[human, "amtl_rate"],
        df.loc[~human, "amtl_rate"],
        equal_var=False,
    )
    print(
        "Welch t-test on per-row AMTL rate (human vs non-human): "
        f"t={t_res.statistic:.3f}, p={t_res.pvalue:.3e}"
    )

    contingency = np.array(
        [
            [df.loc[human, "num_amtl"].sum(), df.loc[human, "non_amtl"].sum()],
            [df.loc[~human, "num_amtl"].sum(), df.loc[~human, "non_amtl"].sum()],
        ],
        dtype=float,
    )
    chi2, chi_p, _, _ = stats.chi2_contingency(contingency)
    print(
        "Chi-square on aggregated sockets (human vs non-human): "
        f"chi2={chi2:.3f}, p={chi_p:.3e}"
    )

    y_counts = np.column_stack([df["num_amtl"].values, df["non_amtl"].values])

    X_biv = patsy.dmatrix("is_human", data=df, return_type="dataframe")
    glm_biv = sm.GLM(y_counts, X_biv, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": df["specimen"]}
    )
    print("\nBivariate binomial GLM summary:")
    print(glm_biv.summary())

    print("\n=== Controlled classical model ===")
    X_ctrl = patsy.dmatrix(
        "is_human + age + prob_male + C(tooth_class)",
        data=df,
        return_type="dataframe",
    )
    glm_ctrl = sm.GLM(y_counts, X_ctrl, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": df["specimen"]}
    )
    print(glm_ctrl.summary())

    beta_h = float(glm_ctrl.params["is_human"])
    p_h = float(glm_ctrl.pvalues["is_human"])
    ci_h = glm_ctrl.conf_int().loc["is_human"].astype(float).values
    or_h = float(np.exp(beta_h))
    or_ci = np.exp(ci_h)
    print(
        "\nHuman effect (controlled GLM): "
        f"log-odds={beta_h:.4f}, OR={or_h:.3f}, p={p_h:.3e}, "
        f"95% OR CI=[{or_ci[0]:.3f}, {or_ci[1]:.3f}]"
    )

    print("\n=== Interpretable agentic_imodels ===")
    X_model = pd.get_dummies(
        df[["is_human", "age", "prob_male", "tooth_class"]],
        columns=["tooth_class"],
        drop_first=True,
    )
    feature_names = list(X_model.columns)
    y_model = df["amtl_rate"].values

    models = [
        ("WinsorizedSparseOLSRegressor", WinsorizedSparseOLSRegressor(max_features=8)),
        ("HingeGAMRegressor", HingeGAMRegressor()),
        ("HingeEBMRegressor", HingeEBMRegressor()),
    ]

    fitted = {}
    for name, model in models:
        model.fit(X_model.values, y_model)
        pred = model.predict(X_model.values)
        rmse = mean_squared_error(y_model, pred) ** 0.5
        r2 = r2_score(y_model, pred)
        fitted[name] = model

        print(f"\n{name} train RMSE={rmse:.4f}, R^2={r2:.4f}")
        print(model)

    print("\n=== Feature direction/magnitude checks ===")
    model_effects = {}
    for name, model in fitted.items():
        eff_human = counterfactual_effect(model, X_model, "is_human", 0, 1)
        eff_age = counterfactual_effect(
            model,
            X_model,
            "age",
            float(np.percentile(X_model["age"], 25)),
            float(np.percentile(X_model["age"], 75)),
        )
        model_effects[name] = {"is_human": eff_human, "age_q75_minus_q25": eff_age}
        print(
            f"{name}: mean prediction delta human(1)-human(0)={eff_human:+.4f}; "
            f"age(Q75-Q25) delta={eff_age:+.4f}"
        )

    wso = fitted["WinsorizedSparseOLSRegressor"]
    kept = {feature_names[i]: float(c) for i, c in zip(wso.support_, wso.ols_coef_)}
    zeroed = [f for f in feature_names if f not in kept]
    print("\nWinsorizedSparseOLS kept coefficients:")
    print(json.dumps(kept, indent=2))
    print("WinsorizedSparseOLS zeroed features:", ", ".join(zeroed))

    hgam = fitted["HingeGAMRegressor"]
    hgam_slopes = {}
    for j, (slope, _, _) in hgam.linear_approx_.items():
        hgam_slopes[feature_names[int(j)]] = float(slope)
    hgam_importance = {
        feature_names[i]: float(v) for i, v in enumerate(hgam.feature_importances_)
    }
    print("\nHingeGAM linear slopes:")
    print(json.dumps(hgam_slopes, indent=2))
    print("HingeGAM importances (range of partial effect):")
    print(json.dumps(hgam_importance, indent=2))

    hebm = fitted["HingeEBMRegressor"]
    hebm_eff = extract_hinge_ebm_effective_coefs(hebm)
    hebm_named = {feature_names[k]: float(v) for k, v in hebm_eff.items()}
    print("\nHingeEBM effective displayed coefficients:")
    print(json.dumps(hebm_named, indent=2))

    print("\nTop displayed features by model:")
    print(
        "HingeGAM:",
        top_features_from_dict(feature_names, {i: v for i, v in enumerate(hgam.feature_importances_)}),
    )
    print("HingeEBM:", top_features_from_dict(feature_names, hebm_eff))

    print("\n=== Synthesis ===")
    positive_models = sum(1 for m in model_effects.values() if m["is_human"] > 1e-4)
    near_zero_models = sum(1 for m in model_effects.values() if abs(m["is_human"]) <= 1e-4)
    negative_models = sum(1 for m in model_effects.values() if m["is_human"] < -1e-4)

    # Calibrated Likert score per SKILL.md guidance.
    if p_h < 1e-3 and or_h > 3:
        score = 82
    elif p_h < 1e-2 and or_h > 2:
        score = 74
    elif p_h < 5e-2:
        score = 62
    else:
        score = 28

    if positive_models == len(model_effects):
        score += 8
    elif positive_models >= 2:
        score += 2
    else:
        score -= 8

    if "is_human" in zeroed:
        score -= 6
    if near_zero_models > 0:
        score -= 2
    if negative_models > 0:
        score -= 8

    score = int(max(0, min(100, round(score))))

    explanation = (
        f"Bivariate evidence indicates higher AMTL in humans (binomial GLM p={glm_biv.pvalues['is_human']:.2e}; "
        f"chi-square p={chi_p:.2e}). In the controlled binomial model (age, sex probability, tooth class), "
        f"the human indicator remains positive and significant (log-odds={beta_h:.2f}, OR={or_h:.2f}, "
        f"95% CI {or_ci[0]:.2f}-{or_ci[1]:.2f}, p={p_h:.2e}). Interpretable models are mostly consistent but not "
        f"unanimous: HingeGAM and HingeEBM both assign positive human effects "
        f"({model_effects['HingeGAMRegressor']['is_human']:+.3f}, "
        f"{model_effects['HingeEBMRegressor']['is_human']:+.3f} predicted rate delta for human=1 vs 0), while "
        f"WinsorizedSparseOLS zeroes out is_human, which is notable null evidence under strong sparsity pressure. "
        f"Age is a robust positive predictor across models, and posterior teeth generally show higher AMTL than anterior. "
        f"Overall this supports a real positive human-vs-nonhuman difference after controls, with some shrinkage-related "
        f"uncertainty in one sparse model."
    )

    result = {"response": score, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f)

    print("\nWrote conclusion.txt:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
