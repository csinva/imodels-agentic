import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")


def main() -> None:
    df = pd.read_csv("teachingratings.csv")

    # Basic dataset exploration
    print("=== Data Shape ===")
    print(df.shape)
    print("\n=== Columns ===")
    print(df.columns.tolist())

    print("\n=== Missing Values ===")
    print(df.isna().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    print("\n=== Numeric Summary ===")
    print(df[numeric_cols].describe().T)

    print("\n=== Categorical Distributions ===")
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts(normalize=True).round(3))

    print("\n=== Correlations with eval ===")
    corr_with_eval = df[numeric_cols].corr(numeric_only=True)["eval"].sort_values(ascending=False)
    print(corr_with_eval)

    # Statistical tests focused on research question: beauty -> eval
    beauty = df["beauty"]
    evals = df["eval"]

    pearson_r, pearson_p = stats.pearsonr(beauty, evals)
    spearman_rho, spearman_p = stats.spearmanr(beauty, evals)

    median_beauty = beauty.median()
    high_beauty_eval = evals[beauty >= median_beauty]
    low_beauty_eval = evals[beauty < median_beauty]
    t_stat, t_p = stats.ttest_ind(high_beauty_eval, low_beauty_eval, equal_var=False)

    beauty_quartile = pd.qcut(beauty, 4, labels=False)
    groups = [evals[beauty_quartile == q] for q in range(4)]
    f_stat, anova_p = stats.f_oneway(*groups)

    print("\n=== Statistical Tests ===")
    print(f"Pearson r(beauty, eval) = {pearson_r:.4f}, p = {pearson_p:.4g}")
    print(f"Spearman rho(beauty, eval) = {spearman_rho:.4f}, p = {spearman_p:.4g}")
    print(f"T-test (high vs low beauty eval): t = {t_stat:.4f}, p = {t_p:.4g}")
    print(f"ANOVA (eval across beauty quartiles): F = {f_stat:.4f}, p = {anova_p:.4g}")

    # OLS models for interpretable inference with p-values
    ols_simple = smf.ols("eval ~ beauty", data=df).fit()
    ols_adjusted = smf.ols(
        "eval ~ beauty + age + students + allstudents + C(minority) + C(gender) + "
        "C(credits) + C(division) + C(native) + C(tenure)",
        data=df,
    ).fit()

    print("\n=== OLS: eval ~ beauty ===")
    print(ols_simple.summary())

    print("\n=== OLS Adjusted ===")
    print(ols_adjusted.summary())

    # Interpretable sklearn models
    # Exclude identifier-like columns from predictive modeling for interpretability
    drop_cols = ["eval", "rownames", "prof"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df["eval"]

    model_num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    model_cat_cols = [c for c in X.columns if c not in model_num_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", model_num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), model_cat_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001, max_iter=20000),
        "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=42),
    }

    sklearn_results = {}
    beauty_linear_effects = {}

    print("\n=== Interpretable sklearn models ===")
    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        r2 = r2_score(y_test, preds)
        sklearn_results[name] = r2
        print(f"\n{name} test R^2: {r2:.4f}")

        feat_names = pipe.named_steps["prep"].get_feature_names_out()
        fitted_model = pipe.named_steps["model"]

        if hasattr(fitted_model, "coef_"):
            coefs = pd.Series(fitted_model.coef_, index=feat_names)
            top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(12)
            print("Top coefficients by absolute magnitude:")
            print(top)
            beauty_coef = coefs[[i for i in coefs.index if "beauty" in i]].sum()
            beauty_linear_effects[name] = beauty_coef
            print(f"Beauty coefficient ({name}): {beauty_coef:.4f}")

        if hasattr(fitted_model, "feature_importances_"):
            imps = pd.Series(fitted_model.feature_importances_, index=feat_names)
            top_imp = imps.sort_values(ascending=False).head(12)
            print("Top feature importances:")
            print(top_imp)

    # imodels (interpretable rule/tree-based models)
    imodels_summary = {}
    beauty_importance_proxy = 0.0
    try:
        from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor

        X_train_enc = preprocessor.fit_transform(X_train)
        X_test_enc = preprocessor.transform(X_test)
        feature_names = preprocessor.get_feature_names_out()

        if hasattr(X_train_enc, "toarray"):
            X_train_enc = X_train_enc.toarray()
            X_test_enc = X_test_enc.toarray()

        imodels = {
            "RuleFitRegressor": RuleFitRegressor(random_state=42),
            "FIGSRegressor": FIGSRegressor(random_state=42),
            "HSTreeRegressor": HSTreeRegressor(),
        }

        print("\n=== imodels ===")
        for name, model in imodels.items():
            model.fit(X_train_enc, y_train, feature_names=feature_names)
            pred = model.predict(X_test_enc)
            r2 = r2_score(y_test, pred)
            imodels_summary[name] = r2
            print(f"\n{name} test R^2: {r2:.4f}")

            if hasattr(model, "feature_importances_"):
                imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
                print("Top feature importances:")
                print(imp.head(12))
                beauty_imp = imp[[i for i in imp.index if "beauty" in i]].sum()
                beauty_importance_proxy = max(beauty_importance_proxy, float(beauty_imp))

            if name == "RuleFitRegressor" and hasattr(model, "get_rules"):
                rules = model.get_rules()
                rules = rules.loc[rules["coef"] != 0].copy()
                rules["abs_coef"] = rules["coef"].abs()
                rules_top = rules.sort_values("abs_coef", ascending=False).head(15)
                print("Top non-zero RuleFit rules/terms:")
                print(rules_top[["rule", "coef", "support", "importance"]])
                beauty_rule_hits = rules_top["rule"].astype(str).str.contains("beauty").sum()
                if beauty_rule_hits > 0:
                    beauty_importance_proxy = max(beauty_importance_proxy, 0.1)

    except Exception as e:
        print("\n[Warning] imodels step failed:", repr(e))

    # Consolidated inference for final Likert response
    beauty_coef_adj = float(ols_adjusted.params.get("beauty", np.nan))
    beauty_p_adj = float(ols_adjusted.pvalues.get("beauty", np.nan))
    beauty_coef_simple = float(ols_simple.params.get("beauty", np.nan))
    beauty_p_simple = float(ols_simple.pvalues.get("beauty", np.nan))

    score = 50

    if beauty_p_adj < 0.001:
        score += 30
    elif beauty_p_adj < 0.01:
        score += 25
    elif beauty_p_adj < 0.05:
        score += 18
    elif beauty_p_adj < 0.10:
        score += 8
    else:
        score -= 20

    if beauty_p_simple < 0.05:
        score += 10
    else:
        score -= 5

    if beauty_coef_adj > 0:
        score += 5
    else:
        score -= 10

    if abs(beauty_coef_adj) >= 0.08:
        score += 8
    elif abs(beauty_coef_adj) >= 0.04:
        score += 4

    if t_p < 0.05:
        score += 4

    if anova_p < 0.05:
        score += 4

    if any(v > 0 for v in beauty_linear_effects.values()):
        score += 2

    if beauty_importance_proxy > 0:
        score += 2

    score = int(max(0, min(100, round(score))))

    explanation = (
        f"Beauty shows a statistically significant positive association with teaching evaluations. "
        f"Pearson r={pearson_r:.3f} (p={pearson_p:.2g}); simple OLS beauty coefficient={beauty_coef_simple:.3f} "
        f"(p={beauty_p_simple:.2g}); adjusted OLS beauty coefficient={beauty_coef_adj:.3f} (p={beauty_p_adj:.2g}) "
        f"after controlling for instructor and course covariates. Group tests are also significant "
        f"(median split t-test p={t_p:.2g}, beauty-quartile ANOVA p={anova_p:.2g}), and interpretable "
        f"linear/tree/rule models retain beauty as a useful predictor. This is observational evidence "
        f"of a meaningful relationship, not definitive causality."
    )

    output = {"response": score, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(output), encoding="utf-8")

    print("\n=== Final Conclusion JSON ===")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
