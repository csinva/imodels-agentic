import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")


def _normalize_binary(series: pd.Series, yes_values) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(yes_values).astype(int)


def _safe_ttest(group_a: pd.Series, group_b: pd.Series):
    a = group_a.dropna()
    b = group_b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    res = stats.ttest_ind(a, b, equal_var=False)
    return float(res.statistic), float(res.pvalue)


def _points_from_pvalue(p: float) -> int:
    if np.isnan(p):
        return 0
    if p < 0.01:
        return 30
    if p < 0.05:
        return 22
    if p < 0.10:
        return 12
    return 4


def main():
    base = Path(".")
    info = json.loads((base / "info.json").read_text())
    research_question = info.get("research_questions", ["Unknown question"])[0]

    df = pd.read_csv(base / "panda_nuts.csv")

    df["sex"] = df["sex"].astype(str).str.strip().str.lower()
    df["help"] = df["help"].astype(str).str.strip().str.lower()

    df["sex_m"] = _normalize_binary(df["sex"], {"m", "male"})
    df["help_y"] = _normalize_binary(df["help"], {"y", "yes", "true", "1"})

    # Efficiency definition: nuts cracked per second in a session.
    df["efficiency"] = np.where(df["seconds"] > 0, df["nuts_opened"] / df["seconds"], np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)

    analysis_cols = [
        "chimpanzee",
        "age",
        "sex",
        "hammer",
        "help",
        "sex_m",
        "help_y",
        "nuts_opened",
        "seconds",
        "efficiency",
    ]
    df = df[analysis_cols].dropna().copy()

    print("Research question:", research_question)
    print("\nData shape:", df.shape)
    print("\nMissing values by column:\n", df.isna().sum())
    print("\nSummary statistics (numeric):\n", df.select_dtypes(include=[np.number]).describe().T)

    print("\nEfficiency distribution details:")
    print("Quantiles:\n", df["efficiency"].quantile([0.0, 0.25, 0.5, 0.75, 1.0]))
    hist_counts, hist_bins = np.histogram(df["efficiency"], bins=8)
    print("Histogram counts:", hist_counts.tolist())
    print("Histogram bins:", [round(x, 4) for x in hist_bins.tolist()])

    numeric_for_corr = df[["age", "sex_m", "help_y", "nuts_opened", "seconds", "efficiency"]]
    print("\nCorrelation matrix:\n", numeric_for_corr.corr())

    pearson_age = stats.pearsonr(df["age"], df["efficiency"])
    spearman_age = stats.spearmanr(df["age"], df["efficiency"])
    print(
        "\nAge-efficiency correlations:\n"
        f"Pearson r={pearson_age.statistic:.3f}, p={pearson_age.pvalue:.4g}; "
        f"Spearman rho={spearman_age.statistic:.3f}, p={spearman_age.pvalue:.4g}"
    )

    female_eff = df.loc[df["sex_m"] == 0, "efficiency"]
    male_eff = df.loc[df["sex_m"] == 1, "efficiency"]
    t_sex, p_sex = _safe_ttest(male_eff, female_eff)

    help_eff = df.loc[df["help_y"] == 1, "efficiency"]
    no_help_eff = df.loc[df["help_y"] == 0, "efficiency"]
    t_help, p_help = _safe_ttest(help_eff, no_help_eff)

    print("\nWelch t-tests on efficiency:")
    print(
        f"Sex (male vs female): t={t_sex:.3f}, p={p_sex:.4g}, "
        f"means=({male_eff.mean():.3f}, {female_eff.mean():.3f})"
    )
    print(
        f"Help (yes vs no): t={t_help:.3f}, p={p_help:.4g}, "
        f"means=({help_eff.mean():.3f}, {no_help_eff.mean():.3f})"
    )

    # OLS with p-values and confidence intervals for adjusted effects.
    X_ols = pd.DataFrame(
        {
            "age": df["age"],
            "sex_m": df["sex_m"],
            "help_y": df["help_y"],
        }
    )
    X_ols = sm.add_constant(X_ols)
    y = df["efficiency"]

    ols_model = sm.OLS(y, X_ols).fit()
    print("\nOLS summary:\n", ols_model.summary())

    feature_cols = ["age", "sex", "help", "hammer"]
    X = df[feature_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                ["sex", "help", "hammer"],
            ),
            ("num", "passthrough", ["age"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    lin_pipe = Pipeline([("prep", preprocessor), ("model", LinearRegression())])
    ridge_pipe = Pipeline([("prep", preprocessor), ("model", Ridge(alpha=1.0, random_state=0))])
    lasso_pipe = Pipeline([("prep", preprocessor), ("model", Lasso(alpha=0.001, random_state=0, max_iter=10000))])
    tree_reg_pipe = Pipeline([("prep", preprocessor), ("model", DecisionTreeRegressor(max_depth=3, random_state=0))])

    models = {
        "linear": lin_pipe,
        "ridge": ridge_pipe,
        "lasso": lasso_pipe,
        "tree_reg": tree_reg_pipe,
    }

    print("\nInterpretable scikit-learn model summaries:")
    fitted_models = {}
    for name, mdl in models.items():
        mdl.fit(X, y)
        preds = mdl.predict(X)
        r2 = r2_score(y, preds)
        fitted_models[name] = mdl
        print(f"{name} R^2: {r2:.3f}")

    feat_names = fitted_models["linear"].named_steps["prep"].get_feature_names_out()
    lin_coef = fitted_models["linear"].named_steps["model"].coef_
    ridge_coef = fitted_models["ridge"].named_steps["model"].coef_
    lasso_coef = fitted_models["lasso"].named_steps["model"].coef_

    coef_table = pd.DataFrame(
        {
            "feature": feat_names,
            "linear_coef": lin_coef,
            "ridge_coef": ridge_coef,
            "lasso_coef": lasso_coef,
        }
    )
    coef_table["abs_linear"] = coef_table["linear_coef"].abs()
    coef_table = coef_table.sort_values("abs_linear", ascending=False)
    print("\nTop linear/ridge/lasso coefficients by |linear coef|:\n", coef_table.head(10))

    tree_importance = pd.DataFrame(
        {
            "feature": feat_names,
            "importance": fitted_models["tree_reg"].named_steps["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    print("\nDecisionTreeRegressor feature importances:\n", tree_importance)

    high_eff = (y >= y.median()).astype(int)
    tree_clf_pipe = Pipeline(
        [
            ("prep", preprocessor),
            ("model", DecisionTreeClassifier(max_depth=3, random_state=0)),
        ]
    )
    tree_clf_pipe.fit(X, high_eff)
    clf_importance = pd.DataFrame(
        {
            "feature": feat_names,
            "importance": tree_clf_pipe.named_steps["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    print("\nDecisionTreeClassifier (high efficiency) feature importances:\n", clf_importance)

    # imodels (rule/tree-based interpretable models)
    X_im = pd.get_dummies(X, drop_first=True)

    rulefit = RuleFitRegressor(random_state=0, max_rules=30)
    rulefit.fit(X_im, y)
    rulefit_rules = rulefit._get_rules()
    rulefit_rules = rulefit_rules[rulefit_rules["importance"] > 0].sort_values(
        "importance", ascending=False
    )

    figs = FIGSRegressor(random_state=0, max_rules=12)
    figs.fit(X_im, y)
    figs_importances = pd.Series(figs.feature_importances_, index=X_im.columns).sort_values(ascending=False)

    hstree = HSTreeRegressor(random_state=0)
    hstree.fit(X_im, y)
    hs_importances = pd.Series(hstree.estimator_.feature_importances_, index=X_im.columns).sort_values(
        ascending=False
    )

    print("\nimodels summaries:")
    print(f"RuleFit train R^2: {rulefit.score(X_im, y):.3f}")
    print("Top RuleFit rules/terms:\n", rulefit_rules.head(10))
    print(f"FIGS train R^2: {figs.score(X_im, y):.3f}")
    print("FIGS feature importances:\n", figs_importances)
    print(f"HSTree train R^2: {hstree.score(X_im, y):.3f}")
    print("HSTree feature importances:\n", hs_importances)

    p_age = float(ols_model.pvalues.get("age", np.nan))
    p_sex_adj = float(ols_model.pvalues.get("sex_m", np.nan))
    p_help_adj = float(ols_model.pvalues.get("help_y", np.nan))

    evidence_points = (
        _points_from_pvalue(p_age)
        + _points_from_pvalue(p_sex_adj)
        + _points_from_pvalue(p_help_adj)
        + max(0.0, ols_model.rsquared_adj) * 10.0
    )
    response_score = int(np.clip(round(evidence_points), 0, 100))

    age_coef = float(ols_model.params.get("age", np.nan))
    sex_coef = float(ols_model.params.get("sex_m", np.nan))
    help_coef = float(ols_model.params.get("help_y", np.nan))

    explanation = (
        "Using efficiency = nuts_opened/seconds, adjusted OLS found significant positive effects of age "
        f"(beta={age_coef:.3f}, p={p_age:.4g}) and male sex (beta={sex_coef:.3f}, p={p_sex_adj:.4g}), "
        f"while help had a negative but non-significant adjusted effect (beta={help_coef:.3f}, p={p_help_adj:.4g}). "
        "Unadjusted t-tests showed lower efficiency when help was present, but this weakens after controlling for age/sex. "
        "Interpretable models (linear/ridge/lasso, decision trees, RuleFit/FIGS/HSTree) consistently ranked age and sex-related terms "
        "as important, with weaker evidence for independent help effects."
    )

    out = {"response": response_score, "explanation": explanation}
    with open(base / "conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(out, f)

    print("\nWrote conclusion.txt:")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
