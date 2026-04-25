import json
import re
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from agentic_imodels import HingeEBMRegressor, HingeGAMRegressor, SmartAdditiveRegressor

warnings.filterwarnings("ignore")


def parse_zeroed_features(model_text: str, feature_names: List[str]) -> List[str]:
    """Parse lines like: Features with zero coefficients (excluded): x0, x1"""
    match = re.search(r"Features with zero coefficients \(excluded\):\s*(.*)", model_text)
    if not match:
        return []

    zero_tokens = [tok.strip() for tok in match.group(1).split(",") if tok.strip()]
    zero_features = []
    for token in zero_tokens:
        idx_match = re.fullmatch(r"x(\d+)", token)
        if idx_match:
            idx = int(idx_match.group(1))
            if 0 <= idx < len(feature_names):
                zero_features.append(feature_names[idx])
    return zero_features


def counterfactual_children_effect(model, X: pd.DataFrame) -> float:
    """Average predicted change in affairs when toggling children indicator 0->1."""
    X0 = X.copy()
    X1 = X.copy()
    X0["children_yes"] = 0
    X1["children_yes"] = 1
    pred0 = np.asarray(model.predict(X0), dtype=float)
    pred1 = np.asarray(model.predict(X1), dtype=float)
    return float(np.mean(pred1 - pred0))


def main():
    # ---------------------------------------------------------------------
    # 1) Read question metadata + load data
    # ---------------------------------------------------------------------
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    question = info.get("research_questions", ["Unknown question"])[0]

    df = pd.read_csv("affairs.csv")

    print("=" * 88)
    print("Research question:")
    print(question)
    print("=" * 88)

    # Basic encoding for analysis
    df = df.copy()
    df["children_yes"] = (df["children"].str.lower() == "yes").astype(int)
    df["gender_male"] = (df["gender"].str.lower() == "male").astype(int)

    # ---------------------------------------------------------------------
    # 2) Exploration: summary stats, distributions, correlations
    # ---------------------------------------------------------------------
    print("\n[EDA] Data shape:", df.shape)
    print("[EDA] Missing values per column:")
    print(df.isna().sum())

    numeric_cols = [
        "affairs",
        "children_yes",
        "gender_male",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]

    print("\n[EDA] Numeric summary statistics:")
    print(df[numeric_cols].describe().T)

    print("\n[EDA] Affairs distribution (counts):")
    print(df["affairs"].value_counts().sort_index())

    print("\n[EDA] Affairs by children group:")
    print(df.groupby("children")["affairs"].describe())

    corr = df[numeric_cols].corr(numeric_only=True)
    print("\n[EDA] Correlation of predictors with affairs:")
    print(corr["affairs"].sort_values(ascending=False))

    # ---------------------------------------------------------------------
    # 3) Classical tests: bivariate + controlled GLM for counts
    # ---------------------------------------------------------------------
    y_yes = df.loc[df["children_yes"] == 1, "affairs"]
    y_no = df.loc[df["children_yes"] == 0, "affairs"]

    t_stat, t_p = stats.ttest_ind(y_yes, y_no, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(y_yes, y_no, alternative="two-sided")
    mean_diff = float(y_yes.mean() - y_no.mean())

    print("\n[Bivariate] children_yes vs affairs")
    print(f"Mean(yes)={y_yes.mean():.4f}, Mean(no)={y_no.mean():.4f}, Diff(yes-no)={mean_diff:.4f}")
    print(f"Welch t-test: t={t_stat:.4f}, p={t_p:.6g}")
    print(f"Mann-Whitney U: U={u_stat:.4f}, p={u_p:.6g}")

    # Controlled model: Poisson diagnostic -> Negative Binomial GLM
    control_cols = [
        "children_yes",
        "gender_male",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]
    X_glm = sm.add_constant(df[control_cols])
    y = df["affairs"]

    poisson_model = sm.GLM(y, X_glm, family=sm.families.Poisson()).fit()
    dispersion = float(poisson_model.pearson_chi2 / poisson_model.df_resid)

    # Method-of-moments style alpha from Poisson dispersion: Var ~= mu + alpha*mu^2
    mu_bar = float(np.mean(poisson_model.fittedvalues))
    alpha_est = max((dispersion - 1.0) / max(mu_bar, 1e-8), 1e-6)

    nb_model = sm.GLM(
        y,
        X_glm,
        family=sm.families.NegativeBinomial(alpha=alpha_est),
    ).fit(cov_type="HC3")

    print("\n[Controlled count model] Negative Binomial GLM (with controls)")
    print(f"Poisson dispersion diagnostic: {dispersion:.3f} (>1 suggests overdispersion)")
    print(f"Estimated NB alpha used: {alpha_est:.4f}")
    print(nb_model.summary())

    beta_child = float(nb_model.params["children_yes"])
    p_child = float(nb_model.pvalues["children_yes"])
    ci_low, ci_high = nb_model.conf_int().loc["children_yes"].tolist()
    irr_child = float(np.exp(beta_child))

    print(
        f"\n[Controlled effect] children_yes beta={beta_child:.4f}, p={p_child:.6g}, "
        f"95% CI=({ci_low:.4f}, {ci_high:.4f}), IRR={irr_child:.4f}"
    )

    # OLS sensitivity (robust SE)
    ols_model = sm.OLS(y, X_glm).fit(cov_type="HC3")
    beta_child_ols = float(ols_model.params["children_yes"])
    p_child_ols = float(ols_model.pvalues["children_yes"])
    print(
        f"[OLS sensitivity] children_yes beta={beta_child_ols:.4f}, p={p_child_ols:.6g}"
    )

    # ---------------------------------------------------------------------
    # 4) Interpretable models: shape/direction/magnitude/robustness
    # ---------------------------------------------------------------------
    feature_cols = control_cols  # keep identical variables to controlled model
    X_model = df[feature_cols]
    y_model = y

    x_index_map = {f"x{i}": col for i, col in enumerate(feature_cols)}
    print("\n[Interpretable models] x-index mapping:")
    print(x_index_map)

    model_classes = [SmartAdditiveRegressor, HingeGAMRegressor, HingeEBMRegressor]
    fitted_models: Dict[str, object] = {}
    model_texts: Dict[str, str] = {}

    for cls in model_classes:
        name = cls.__name__
        model = cls()
        model.fit(X_model, y_model)
        fitted_models[name] = model
        text = str(model)
        model_texts[name] = text
        print("\n" + "=" * 30 + f" {name} " + "=" * 30)
        print(model)  # required: capture interpretable form verbatim

    # Counterfactual children effect across models
    model_child_effects = {
        name: counterfactual_children_effect(model, X_model)
        for name, model in fitted_models.items()
    }
    print("\n[Robustness] Counterfactual avg effect of children_yes (1 vs 0) on predicted affairs:")
    for name, eff in model_child_effects.items():
        print(f"  {name}: {eff:.4f}")

    # Zeroing / feature-importance null evidence
    hinge_zeroed = parse_zeroed_features(model_texts["HingeGAMRegressor"], feature_cols)
    print("\n[Null evidence] HingeGAM zeroed-out features:", hinge_zeroed)

    smart_importances = None
    smart_rank = None
    if hasattr(fitted_models["SmartAdditiveRegressor"], "feature_importances_"):
        smart_importances = np.asarray(fitted_models["SmartAdditiveRegressor"].feature_importances_, dtype=float)
        imp_table = pd.DataFrame({"feature": feature_cols, "importance": smart_importances})
        imp_table = imp_table.sort_values("importance", ascending=False).reset_index(drop=True)
        print("\n[SmartAdditive] Feature importance ranking:")
        print(imp_table)

        rank_map = {row.feature: i + 1 for i, row in imp_table.iterrows()}
        smart_rank = rank_map.get("children_yes", None)
        smart_child_imp = float(imp_table.loc[imp_table["feature"] == "children_yes", "importance"].iloc[0])
    else:
        smart_child_imp = float("nan")

    # ---------------------------------------------------------------------
    # 5) Calibrated conclusion score (0-100 Likert)
    # ---------------------------------------------------------------------
    score = 50

    # Controlled significance is primary
    if p_child < 0.01:
        score += 30 if beta_child < 0 else -30
    elif p_child < 0.05:
        score += 20 if beta_child < 0 else -20
    elif p_child < 0.10:
        score += 8 if beta_child < 0 else -8
    else:
        score -= 15

    # Bivariate corroboration
    if t_p < 0.05:
        score += 8 if mean_diff < 0 else -8
    elif t_p >= 0.10:
        score -= 5

    # Robustness across interpretable models
    for eff in model_child_effects.values():
        if eff < -0.05:
            score += 5
        elif eff > 0.05:
            score -= 5

    # Null evidence from sparse/hinge exclusion and low importance
    if "children_yes" in hinge_zeroed:
        score -= 15

    if smart_importances is not None:
        if smart_child_imp <= 1e-9:
            score -= 10
        if smart_rank is not None and smart_rank >= len(feature_cols) - 1:
            score -= 5

    # OLS sensitivity (small auxiliary weight)
    if p_child_ols < 0.05:
        score += 5 if beta_child_ols < 0 else -5
    elif p_child_ols >= 0.10:
        score -= 3

    response = int(np.clip(np.round(score), 0, 100))

    explanation = (
        f"Question: {question} "
        f"Bivariate evidence was weak/mixed (children mean diff yes-no={mean_diff:.3f}, Welch p={t_p:.3g}, "
        f"Mann-Whitney p={u_p:.3g}). In the controlled Negative Binomial GLM, children had "
        f"beta={beta_child:.3f} (IRR={irr_child:.3f}, p={p_child:.3g}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]), "
        f"so the adjusted effect was {'negative' if beta_child < 0 else 'positive'} but "
        f"{'not statistically significant' if p_child >= 0.05 else 'statistically significant'}. "
        f"Interpretable models corroborated weak impact: counterfactual children effects were "
        + ", ".join([f"{k}={v:.3f}" for k, v in model_child_effects.items()])
        + ". "
        f"HingeGAM zeroed out {('children_yes' if 'children_yes' in hinge_zeroed else 'no key child effect')} and "
        f"SmartAdditive ranked children importance at {smart_rank if smart_rank is not None else 'NA'} "
        f"(importance={smart_child_imp:.3f}). Overall evidence that having children decreases affairs is weak."
    )

    out = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(out, f)

    print("\n[Final] Likert response:", response)
    print("[Final] Wrote conclusion.txt")


if __name__ == "__main__":
    main()
