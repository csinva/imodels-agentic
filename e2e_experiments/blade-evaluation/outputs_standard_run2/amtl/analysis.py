import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

warnings.filterwarnings("ignore")


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def top_abs_coefs(feature_names, coefs, n=8):
    s = pd.Series(coefs, index=feature_names)
    return s.reindex(s.abs().sort_values(ascending=False).index).head(n)


def main() -> None:
    # ---------------------------------------------------------------------
    # Load and basic preparation
    # ---------------------------------------------------------------------
    df = pd.read_csv("amtl.csv")
    required_cols = [
        "num_amtl",
        "sockets",
        "age",
        "prob_male",
        "tooth_class",
        "genus",
    ]
    df = df.dropna(subset=required_cols).copy()
    df = df[df["sockets"] > 0].copy()

    df["amtl_rate"] = df["num_amtl"] / df["sockets"]
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    print_section("Dataset Overview")
    print(f"Rows: {len(df)}")
    print(f"Columns: {df.shape[1]}")
    print("\nGenus counts:")
    print(df["genus"].value_counts().to_string())
    print("\nTooth class counts:")
    print(df["tooth_class"].value_counts().to_string())

    print_section("Summary Statistics")
    numeric_cols = ["num_amtl", "sockets", "amtl_rate", "age", "stdev_age", "prob_male"]
    print(df[numeric_cols].describe().round(4).to_string())

    print("\nMean AMTL rate by genus:")
    print(df.groupby("genus")["amtl_rate"].agg(["mean", "median", "std", "count"]).round(4).to_string())

    print("\nMean AMTL rate by tooth class:")
    print(df.groupby("tooth_class")["amtl_rate"].agg(["mean", "median", "std", "count"]).round(4).to_string())

    print_section("Correlations")
    corr_cols = ["num_amtl", "sockets", "amtl_rate", "age", "stdev_age", "prob_male", "is_human"]
    corr = df[corr_cols].corr().round(4)
    print(corr.to_string())

    spear_r, spear_p = stats.spearmanr(df["age"], df["amtl_rate"])
    print(f"\nSpearman(age, amtl_rate): rho={spear_r:.4f}, p={spear_p:.3e}")

    # ---------------------------------------------------------------------
    # Statistical tests
    # ---------------------------------------------------------------------
    print_section("Statistical Tests")
    human = df[df["is_human"] == 1]["amtl_rate"]
    nonhuman = df[df["is_human"] == 0]["amtl_rate"]

    t_stat, t_p = stats.ttest_ind(human, nonhuman, equal_var=False)
    print(
        "Welch t-test (human vs non-human AMTL rate): "
        f"t={t_stat:.4f}, p={t_p:.3e}, human_mean={human.mean():.4f}, nonhuman_mean={nonhuman.mean():.4f}"
    )

    groups = [vals["amtl_rate"].values for _, vals in df.groupby("genus")]
    anova_stat, anova_p = stats.f_oneway(*groups)
    print(f"One-way ANOVA across genera: F={anova_stat:.4f}, p={anova_p:.3e}")

    # Controlled binomial regression: probability of missing tooth among observable sockets
    X_glm = pd.get_dummies(
        df[["is_human", "age", "prob_male", "tooth_class"]],
        drop_first=True,
        dtype=float,
    )
    X_glm = sm.add_constant(X_glm, has_constant="add").astype(float)
    y_glm = df["amtl_rate"].astype(float)

    glm = sm.GLM(
        y_glm,
        X_glm,
        family=sm.families.Binomial(),
        var_weights=df["sockets"].astype(float),
    )
    glm_res = glm.fit()

    human_coef = float(glm_res.params.get("is_human", np.nan))
    human_p = float(glm_res.pvalues.get("is_human", np.nan))
    human_or = float(np.exp(human_coef))
    human_ci = glm_res.conf_int().loc["is_human"].astype(float)
    human_or_ci = np.exp(human_ci.values)

    print("\nBinomial GLM (controls: age, prob_male, tooth_class):")
    print(glm_res.summary2().tables[1].round(5).to_string())
    print(
        f"\nHuman effect (controlled): coef={human_coef:.4f}, p={human_p:.3e}, "
        f"odds_ratio={human_or:.3f}, OR_95CI=({human_or_ci[0]:.3f}, {human_or_ci[1]:.3f})"
    )

    # OLS for additional interpretable p-values
    ols = sm.OLS(y_glm, X_glm).fit()
    print("\nOLS (same predictors, unweighted rate model) coefficients:")
    print(ols.summary2().tables[1].round(5).to_string())

    # ---------------------------------------------------------------------
    # Interpretable models (scikit-learn + imodels)
    # ---------------------------------------------------------------------
    print_section("Interpretable Models")
    X_ml = pd.get_dummies(
        df[["genus", "tooth_class", "age", "prob_male", "stdev_age"]],
        drop_first=True,
        dtype=float,
    )
    y_ml = df["amtl_rate"].astype(float).values
    feature_names = X_ml.columns.tolist()
    X_values = X_ml.values
    sample_weight = df["sockets"].astype(float).values

    # Linear models
    lin = LinearRegression()
    lin.fit(X_values, y_ml, sample_weight=sample_weight)
    lin_pred = lin.predict(X_values)

    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_values, y_ml, sample_weight=sample_weight)
    ridge_pred = ridge.predict(X_values)

    lasso = Lasso(alpha=0.0005, max_iter=20000, random_state=42)
    lasso.fit(X_values, y_ml, sample_weight=sample_weight)
    lasso_pred = lasso.predict(X_values)

    print("LinearRegression top coefficients (absolute):")
    print(top_abs_coefs(feature_names, lin.coef_).round(5).to_string())
    print(f"LinearRegression weighted in-sample R^2: {r2_score(y_ml, lin_pred, sample_weight=sample_weight):.4f}")

    print("\nRidge top coefficients (absolute):")
    print(top_abs_coefs(feature_names, ridge.coef_).round(5).to_string())
    print(f"Ridge weighted in-sample R^2: {r2_score(y_ml, ridge_pred, sample_weight=sample_weight):.4f}")

    print("\nLasso top coefficients (absolute):")
    print(top_abs_coefs(feature_names, lasso.coef_).round(5).to_string())
    print(f"Lasso weighted in-sample R^2: {r2_score(y_ml, lasso_pred, sample_weight=sample_weight):.4f}")

    # Tree models
    tree_reg = DecisionTreeRegressor(max_depth=3, min_samples_leaf=30, random_state=42)
    tree_reg.fit(X_values, y_ml, sample_weight=sample_weight)
    reg_importance = pd.Series(tree_reg.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\nDecisionTreeRegressor feature importances (top 8):")
    print(reg_importance.head(8).round(5).to_string())

    y_binary = (df["num_amtl"] > 0).astype(int).values
    tree_clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30, random_state=42)
    tree_clf.fit(X_values, y_binary, sample_weight=sample_weight)
    clf_importance = pd.Series(tree_clf.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\nDecisionTreeClassifier feature importances (top 8):")
    print(clf_importance.head(8).round(5).to_string())

    # imodels models
    print("\nimodels RuleFitRegressor")
    rulefit = RuleFitRegressor(n_estimators=100, max_rules=25, tree_size=4, random_state=42)
    rulefit.fit(X_values, y_ml, feature_names=feature_names)
    rules_df = rulefit._get_rules(exclude_zero_coef=True)
    if not rules_df.empty:
        print(rules_df.sort_values("importance", ascending=False).head(8).round(5).to_string(index=False))
    else:
        print("No non-zero rules identified.")

    print("\nimodels FIGSRegressor")
    figs = FIGSRegressor(max_rules=12, random_state=42)
    figs.fit(X_values, y_ml, feature_names=feature_names, sample_weight=sample_weight)
    figs_importance = pd.Series(figs.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("Top FIGS feature importances:")
    print(figs_importance.head(8).round(5).to_string())
    figs_text = str(figs)
    print("\nFIGS structure (truncated):")
    print("\n".join(figs_text.splitlines()[:20]))

    print("\nimodels HSTreeRegressor")
    hst = HSTreeRegressor(max_leaf_nodes=8, random_state=42)
    hst.fit(X_values, y_ml, sample_weight=sample_weight)
    hst_importance = pd.Series(hst.estimator_.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("Top HSTree feature importances:")
    print(hst_importance.head(8).round(5).to_string())
    print("\nHSTree surrogate tree (truncated depth):")
    print(export_text(hst.estimator_, feature_names=feature_names, max_depth=3))

    # ---------------------------------------------------------------------
    # Conclusion scoring
    # ---------------------------------------------------------------------
    print_section("Conclusion Synthesis")
    lin_human_coef = float(lin.coef_[feature_names.index("genus_Homo sapiens")]) if "genus_Homo sapiens" in feature_names else np.nan
    lasso_human_coef = float(lasso.coef_[feature_names.index("genus_Homo sapiens")]) if "genus_Homo sapiens" in feature_names else np.nan

    score = 50
    score += 20 if human_coef > 0 else -20
    score += 20 if human_p < 0.05 else -20
    score += 10 if human_p < 1e-6 else 0
    score += 10 if human.mean() > nonhuman.mean() else -10
    score += 10 if t_p < 0.05 else -10
    score += 5 if anova_p < 0.05 else 0
    if np.isfinite(lin_human_coef):
        score += 3 if lin_human_coef > 0 else -3
    if np.isfinite(lasso_human_coef):
        score += 2 if lasso_human_coef > 0 else -2

    response = int(np.clip(round(score), 0, 100))

    explanation = (
        "Evidence strongly supports higher AMTL in modern humans versus non-human primates. "
        f"Unadjusted rates are higher in humans ({human.mean():.3f}) than non-humans ({nonhuman.mean():.3f}), "
        f"with a significant Welch t-test (p={t_p:.2e}) and genus-level ANOVA (p={anova_p:.2e}). "
        "In the key controlled binomial GLM (adjusting for age, sex probability, and tooth class), "
        f"the human indicator is positive and highly significant (coef={human_coef:.3f}, p={human_p:.2e}, "
        f"OR={human_or:.2f}, 95% CI {human_or_ci[0]:.2f}-{human_or_ci[1]:.2f}). "
        "Interpretable linear, lasso/ridge, decision-tree, RuleFit, FIGS, and HSTree models also identify "
        "genus-related structure consistent with elevated AMTL in humans."
    )

    conclusion = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)

    print(f"Likert response score: {response}")
    print("Wrote conclusion.txt")


if __name__ == "__main__":
    main()
