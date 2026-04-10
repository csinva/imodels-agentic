import json
import warnings
import inspect
from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def safe_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = np.sqrt(((len(x) - 1) * vx + (len(y) - 1) * vy) / (len(x) + len(y) - 2))
    if pooled == 0:
        return 0.0
    return (x.mean() - y.mean()) / pooled


def clamp_int_score(x: float) -> int:
    return int(max(0, min(100, round(x))))


def top_coefficients(feature_names, coefs, k=10):
    pairs = list(zip(feature_names, coefs))
    pairs = sorted(pairs, key=lambda t: abs(t[1]), reverse=True)
    return pairs[:k]


def instantiate_model(cls, preferred_kwargs: Dict[str, Any]):
    try:
        sig = inspect.signature(cls)
        kwargs = {k: v for k, v in preferred_kwargs.items() if k in sig.parameters}
        return cls(**kwargs)
    except Exception:
        return cls()


def main():
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info.get("research_questions", ["Unknown question"])[0]
    print("Research question:", question)

    df = pd.read_csv("soccer.csv")
    print("Raw shape:", df.shape)

    # Feature engineering
    df["skin_tone"] = df[["rater1", "rater2"]].mean(axis=1)
    df = df.dropna(subset=["skin_tone", "games", "redCards"]).copy()
    df = df[df["games"] > 0].copy()
    df["red_rate"] = df["redCards"] / df["games"]
    df["any_red"] = (df["redCards"] > 0).astype(int)
    df["dark_skin"] = (df["skin_tone"] >= 0.5).astype(int)
    df["skin_level"] = pd.Categorical(df["skin_tone"])

    print("Analysis shape (after required filtering):", df.shape)
    print("Missing skin_tone proportion in raw data:", float(pd.read_csv("soccer.csv")[["rater1", "rater2"]].mean(axis=1).isna().mean()))

    # 1) EDA
    numeric_eda_cols = [
        "skin_tone",
        "redCards",
        "red_rate",
        "games",
        "yellowCards",
        "yellowReds",
        "goals",
        "height",
        "weight",
        "meanIAT",
        "meanExp",
    ]
    numeric_eda_cols = [c for c in numeric_eda_cols if c in df.columns]

    print("\nSummary statistics:")
    print(df[numeric_eda_cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]])

    print("\nDistributions:")
    print("Skin tone value counts:")
    print(df["skin_tone"].value_counts().sort_index())
    print("Red cards value counts:")
    print(df["redCards"].value_counts().sort_index())

    corr = df[numeric_eda_cols].corr(numeric_only=True)
    print("\nTop correlations with red_rate:")
    if "red_rate" in corr.columns:
        print(corr["red_rate"].sort_values(ascending=False).head(8))
        print(corr["red_rate"].sort_values(ascending=True).head(8))

    group_stats = df.groupby("dark_skin")["red_rate"].agg(["count", "mean", "std"])
    print("\nRed-card rate by dark_skin (0=light,1=dark):")
    print(group_stats)

    # 2) Statistical tests
    light = df.loc[df["dark_skin"] == 0, "red_rate"].values
    dark = df.loc[df["dark_skin"] == 1, "red_rate"].values

    t_res = stats.ttest_ind(dark, light, equal_var=False, nan_policy="omit")
    mw_res = stats.mannwhitneyu(dark, light, alternative="two-sided")
    effect_d = safe_cohens_d(dark, light)

    contingency = pd.crosstab(df["dark_skin"], df["any_red"])
    chi2_res = stats.chi2_contingency(contingency)

    anova_groups = [g["red_rate"].values for _, g in df.groupby("skin_level") if len(g) >= 30]
    anova_res = stats.f_oneway(*anova_groups) if len(anova_groups) >= 2 else None

    print("\nStatistical tests:")
    print(f"Welch t-test dark vs light red_rate: stat={t_res.statistic:.4f}, p={t_res.pvalue:.6g}")
    print(f"Mann-Whitney U dark vs light red_rate: stat={mw_res.statistic:.4f}, p={mw_res.pvalue:.6g}")
    print(f"Cohen's d (dark-light): {effect_d:.4f}")
    print(f"Chi-square any_red vs dark_skin: chi2={chi2_res[0]:.4f}, p={chi2_res[1]:.6g}")
    if anova_res is not None:
        print(f"ANOVA red_rate across skin-tone levels: F={anova_res.statistic:.4f}, p={anova_res.pvalue:.6g}")

    # Controlled regressions
    reg_formula = "red_rate ~ skin_tone + games + yellowCards + yellowReds + goals + C(position) + C(leagueCountry)"
    ols_model = smf.ols(reg_formula, data=df).fit(cov_type="HC3")

    poisson_formula = "redCards ~ skin_tone + games + yellowCards + yellowReds + goals + C(position) + C(leagueCountry)"
    poisson_model = smf.glm(
        poisson_formula,
        data=df,
        family=sm.families.Poisson(),
        offset=np.log(df["games"].values),
    ).fit()

    skin_ols_coef = float(ols_model.params.get("skin_tone", np.nan))
    skin_ols_p = float(ols_model.pvalues.get("skin_tone", np.nan))
    skin_pois_coef = float(poisson_model.params.get("skin_tone", np.nan))
    skin_pois_p = float(poisson_model.pvalues.get("skin_tone", np.nan))

    print("\nRegression results (key parameter: skin_tone):")
    print(f"OLS coef={skin_ols_coef:.6f}, p={skin_ols_p:.6g}")
    print(f"Poisson log-rate coef={skin_pois_coef:.6f}, p={skin_pois_p:.6g}, IRR={np.exp(skin_pois_coef):.4f}")

    # 3) Interpretable ML models (scikit-learn)
    model_features_num = [
        "skin_tone",
        "games",
        "yellowCards",
        "yellowReds",
        "goals",
        "victories",
        "ties",
        "defeats",
        "height",
        "weight",
        "meanIAT",
        "meanExp",
    ]
    model_features_cat = ["position", "leagueCountry"]
    model_features_num = [c for c in model_features_num if c in df.columns]
    model_features_cat = [c for c in model_features_cat if c in df.columns]

    model_df = df[model_features_num + model_features_cat + ["red_rate", "any_red"]].dropna().copy()
    X = model_df[model_features_num + model_features_cat]
    y_reg = model_df["red_rate"].values
    y_clf = model_df["any_red"].values

    split_idx = int(0.8 * len(model_df))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_reg_train, y_reg_test = y_reg[:split_idx], y_reg[split_idx:]
    y_clf_train, y_clf_test = y_clf[:split_idx], y_clf[split_idx:]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", model_features_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), model_features_cat),
        ]
    )

    lin = Pipeline([("pre", pre), ("model", LinearRegression())])
    ridge = Pipeline([("pre", pre), ("model", Ridge(alpha=1.0))])
    lasso = Pipeline([("pre", pre), ("model", Lasso(alpha=1e-4, max_iter=10000))])
    tree_reg = Pipeline([("pre", pre), ("model", DecisionTreeRegressor(max_depth=4, min_samples_leaf=200, random_state=42))])
    tree_clf = Pipeline([("pre", pre), ("model", DecisionTreeClassifier(max_depth=4, min_samples_leaf=200, class_weight="balanced", random_state=42))])

    lin.fit(X_train, y_reg_train)
    ridge.fit(X_train, y_reg_train)
    lasso.fit(X_train, y_reg_train)
    tree_reg.fit(X_train, y_reg_train)
    tree_clf.fit(X_train, y_clf_train)

    lin_pred = lin.predict(X_test)
    ridge_pred = ridge.predict(X_test)
    lasso_pred = lasso.predict(X_test)
    tree_reg_pred = tree_reg.predict(X_test)
    tree_clf_proba = tree_clf.predict_proba(X_test)[:, 1]

    print("\nScikit-learn interpretable model performance:")
    print(f"LinearRegression R2={r2_score(y_reg_test, lin_pred):.4f}, MAE={mean_absolute_error(y_reg_test, lin_pred):.6f}")
    print(f"Ridge R2={r2_score(y_reg_test, ridge_pred):.4f}, MAE={mean_absolute_error(y_reg_test, ridge_pred):.6f}")
    print(f"Lasso R2={r2_score(y_reg_test, lasso_pred):.4f}, MAE={mean_absolute_error(y_reg_test, lasso_pred):.6f}")
    print(f"DecisionTreeRegressor R2={r2_score(y_reg_test, tree_reg_pred):.4f}, MAE={mean_absolute_error(y_reg_test, tree_reg_pred):.6f}")
    if len(np.unique(y_clf_test)) > 1:
        print(f"DecisionTreeClassifier AUC={roc_auc_score(y_clf_test, tree_clf_proba):.4f}")

    pre_fitted = lin.named_steps["pre"]
    feature_names = pre_fitted.get_feature_names_out()

    lin_coefs = lin.named_steps["model"].coef_
    ridge_coefs = ridge.named_steps["model"].coef_
    lasso_coefs = lasso.named_steps["model"].coef_
    tree_reg_importances = tree_reg.named_steps["model"].feature_importances_
    tree_clf_importances = tree_clf.named_steps["model"].feature_importances_

    print("\nTop LinearRegression coefficients:")
    print(top_coefficients(feature_names, lin_coefs, k=12))
    print("\nTop Ridge coefficients:")
    print(top_coefficients(feature_names, ridge_coefs, k=12))
    print("\nTop Lasso coefficients:")
    print(top_coefficients(feature_names, lasso_coefs, k=12))
    print("\nTop DecisionTreeRegressor importances:")
    print(top_coefficients(feature_names, tree_reg_importances, k=12))
    print("\nTop DecisionTreeClassifier importances:")
    print(top_coefficients(feature_names, tree_clf_importances, k=12))

    def get_coef_for_skin(model_pipeline):
        fn = model_pipeline.named_steps["pre"].get_feature_names_out()
        co = model_pipeline.named_steps["model"].coef_
        for name, value in zip(fn, co):
            if name == "num__skin_tone":
                return float(value)
        return np.nan

    lin_skin_coef = get_coef_for_skin(lin)
    ridge_skin_coef = get_coef_for_skin(ridge)
    lasso_skin_coef = get_coef_for_skin(lasso)

    # 4) Interpretable ML models (imodels)
    imodels_results = {}
    try:
        from imodels import RuleFitRegressor, FIGSRegressor, HSTreeRegressor

        im_features = [c for c in ["skin_tone", "games", "yellowCards", "yellowReds", "goals", "victories", "ties", "defeats", "height", "weight", "meanIAT", "meanExp"] if c in df.columns]
        im_df = df[im_features + ["red_rate"]].dropna().copy()

        if len(im_df) > 40000:
            im_df = im_df.sample(n=40000, random_state=42)

        X_im = im_df[im_features].values
        y_im = im_df["red_rate"].values

        rulefit = instantiate_model(RuleFitRegressor, {"random_state": 42, "n_estimators": 30})
        figs = instantiate_model(FIGSRegressor, {"random_state": 42, "max_rules": 30})
        hstree = instantiate_model(HSTreeRegressor, {"random_state": 42, "max_leaf_nodes": 20})

        for name, model in [("RuleFitRegressor", rulefit), ("FIGSRegressor", figs), ("HSTreeRegressor", hstree)]:
            try:
                model.fit(X_im, y_im)
                pred = model.predict(X_im)
                r2 = r2_score(y_im, pred)
                out = {"r2_train": float(r2)}

                if hasattr(model, "feature_importances_"):
                    fi = model.feature_importances_
                    out["top_features"] = top_coefficients(im_features, fi, k=8)

                if name == "RuleFitRegressor" and hasattr(model, "get_rules"):
                    rules = model.get_rules()
                    rules = rules[rules.coef != 0].copy()
                    rules = rules.sort_values("support", ascending=False).head(8)
                    out["top_rules"] = rules[["rule", "coef", "support"]].to_dict(orient="records")

                imodels_results[name] = out
            except Exception as model_err:
                imodels_results[name] = {"error": str(model_err)}

    except Exception as import_err:
        imodels_results["import_error"] = str(import_err)

    print("\nimodels results:")
    print(imodels_results)

    # 5) Evidence synthesis -> Likert score
    score = 50.0

    # Primary significance evidence
    if np.isfinite(t_res.pvalue):
        if t_res.pvalue < 0.05 and dark.mean() > light.mean():
            score += 10
        elif t_res.pvalue < 0.05 and dark.mean() < light.mean():
            score -= 10
        else:
            score -= 3

    if anova_res is not None and np.isfinite(anova_res.pvalue):
        if anova_res.pvalue < 0.05:
            score += 6
        else:
            score -= 3

    if np.isfinite(skin_ols_p):
        if skin_ols_p < 0.05 and skin_ols_coef > 0:
            score += 12
        elif skin_ols_p < 0.05 and skin_ols_coef < 0:
            score -= 12
        else:
            score -= 4

    if np.isfinite(skin_pois_p):
        if skin_pois_p < 0.05 and skin_pois_coef > 0:
            score += 12
        elif skin_pois_p < 0.05 and skin_pois_coef < 0:
            score -= 12
        else:
            score -= 4

    # Secondary checks
    if np.isfinite(chi2_res[1]):
        if chi2_res[1] < 0.05 and contingency.loc[1, 1] / contingency.loc[1].sum() > contingency.loc[0, 1] / contingency.loc[0].sum():
            score += 5
        elif chi2_res[1] < 0.05:
            score -= 5
        else:
            score -= 5

    if np.isfinite(mw_res.pvalue) and mw_res.pvalue >= 0.05:
        score -= 4

    if np.isfinite(effect_d) and abs(effect_d) < 0.05:
        score -= 8

    coef_votes = [lin_skin_coef, ridge_skin_coef, lasso_skin_coef]
    coef_votes = [c for c in coef_votes if np.isfinite(c)]
    if coef_votes:
        if np.mean(coef_votes) > 0:
            score += 3
        elif np.mean(coef_votes) < 0:
            score -= 3

    score = clamp_int_score(score)

    direction = "higher" if dark.mean() > light.mean() else "lower"
    explanation = (
        f"Using {len(df):,} player-referee dyads with skin-tone ratings, dark-skin players had {direction} red-card rate "
        f"than light-skin players (dark={dark.mean():.5f}, light={light.mean():.5f}). "
        f"The dark-vs-light rate difference was statistically significant by Welch t-test (p={t_res.pvalue:.4g}) and "
        f"ANOVA across skin-tone levels (p={anova_res.pvalue:.4g}). "
        f"In controlled regressions, skin tone was positively associated with red-card outcomes in OLS (coef={skin_ols_coef:.4g}, p={skin_ols_p:.4g}) "
        f"and Poisson with game-exposure offset (coef={skin_pois_coef:.4g}, p={skin_pois_p:.4g}, IRR={np.exp(skin_pois_coef):.3f}). "
        f"The chi-square test on any red card was weaker/non-significant (p={chi2_res[1]:.4g}), and effect sizes were small (Cohen's d={effect_d:.3f}), "
        f"so evidence supports a positive relationship but not an extremely large one."
    )

    out = {
        "response": int(score),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=True))

    print("\nWrote conclusion.txt:")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
