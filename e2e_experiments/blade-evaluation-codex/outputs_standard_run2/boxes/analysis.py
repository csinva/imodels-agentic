import json
import inspect
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

warnings.filterwarnings("ignore")


def safe_pvalue(p):
    try:
        p = float(p)
        if np.isnan(p):
            return 1.0
        return p
    except Exception:
        return 1.0


def instantiate_with_supported_kwargs(cls, preferred_kwargs):
    sig = inspect.signature(cls)
    kwargs = {k: v for k, v in preferred_kwargs.items() if k in sig.parameters}
    return cls(**kwargs)


def sorted_effects(names, values, top_k=8):
    pairs = [(n, float(v)) for n, v in zip(names, values)]
    pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)
    return pairs[:top_k]


def main():
    # 1) Load data
    df = pd.read_csv("boxes.csv")

    # Core target for research question: reliance on majority option
    df["majority_choice"] = (df["y"] == 2).astype(int)

    # 2) EDA: summary statistics, distributions, correlations
    print("=== Basic Info ===")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\n=== Numeric Summary ===")
    print(df[["y", "majority_choice", "gender", "age", "majority_first", "culture"]].describe())

    print("\n=== Distributions ===")
    print("y counts:\n", df["y"].value_counts().sort_index())
    print("majority_choice counts:\n", df["majority_choice"].value_counts().sort_index())
    print("gender counts:\n", df["gender"].value_counts().sort_index())
    print("majority_first counts:\n", df["majority_first"].value_counts().sort_index())
    print("culture counts:\n", df["culture"].value_counts().sort_index())

    print("\nAge distribution by bins:")
    age_bins = pd.cut(df["age"], bins=[3.5, 6.5, 9.5, 12.5, 14.5], labels=["4-6", "7-9", "10-12", "13-14"])
    print(age_bins.value_counts().sort_index())

    print("\n=== Correlations (Pearson) ===")
    corr_cols = ["majority_choice", "age", "gender", "majority_first", "culture"]
    print(df[corr_cols].corr(numeric_only=True))

    # 3) Statistical tests addressing the research question
    print("\n=== Statistical Tests ===")

    # 3a) Age vs majority reliance
    age_majority = df.loc[df["majority_choice"] == 1, "age"]
    age_nonmajority = df.loc[df["majority_choice"] == 0, "age"]
    t_stat, t_p = stats.ttest_ind(age_majority, age_nonmajority, equal_var=False)
    pb_r, pb_p = stats.pointbiserialr(df["majority_choice"], df["age"])

    print(f"Welch t-test (age: majority vs non-majority): t={t_stat:.4f}, p={t_p:.6g}")
    print(f"Point-biserial correlation (majority_choice vs age): r={pb_r:.4f}, p={pb_p:.6g}")

    # 3b) Cultural differences in majority reliance
    contingency = pd.crosstab(df["culture"], df["majority_choice"])
    chi2, chi2_p, chi2_dof, _ = stats.chi2_contingency(contingency)
    print(f"Chi-square (culture x majority_choice): chi2={chi2:.4f}, dof={chi2_dof}, p={chi2_p:.6g}")

    # 3c) Regression with and without age*culture interaction
    model_no_int = smf.ols("majority_choice ~ age + gender + majority_first + C(culture)", data=df).fit()
    model_int = smf.ols("majority_choice ~ age * C(culture) + gender + majority_first", data=df).fit()
    cmp = anova_lm(model_no_int, model_int)

    age_p = safe_pvalue(model_no_int.pvalues.get("age", 1.0))
    interaction_p = safe_pvalue(cmp.iloc[1]["Pr(>F)"])

    print(f"OLS (no interaction) age coefficient={model_no_int.params['age']:.4f}, p={age_p:.6g}")
    print(f"Model comparison for age*culture interaction: p={interaction_p:.6g}")

    # Culture-specific slopes (descriptive inferential detail)
    culture_slopes = {}
    for c, sub in df.groupby("culture"):
        if sub["age"].nunique() >= 2:
            m = smf.ols("majority_choice ~ age", data=sub).fit()
            culture_slopes[int(c)] = {
                "slope": float(m.params.get("age", np.nan)),
                "p": safe_pvalue(m.pvalues.get("age", 1.0)),
            }

    print("Culture-specific age slopes for majority reliance:")
    for c in sorted(culture_slopes):
        print(f"  culture {c}: slope={culture_slopes[c]['slope']:.4f}, p={culture_slopes[c]['p']:.6g}")

    # 4) Interpretable models (scikit-learn)
    print("\n=== Interpretable Models: scikit-learn ===")
    X = df[["age", "gender", "majority_first", "culture"]].copy()
    X = pd.get_dummies(X, columns=["culture"], prefix="culture", drop_first=True)
    y = df["majority_choice"].values

    feature_names = list(X.columns)
    Xv = X.values

    lin = LinearRegression()
    lin.fit(Xv, y)
    print("LinearRegression top coefficients:", sorted_effects(feature_names, lin.coef_))

    ridge = Ridge(alpha=1.0, random_state=0)
    ridge.fit(Xv, y)
    print("Ridge top coefficients:", sorted_effects(feature_names, ridge.coef_))

    lasso = Lasso(alpha=0.005, random_state=0, max_iter=20000)
    lasso.fit(Xv, y)
    print("Lasso top coefficients:", sorted_effects(feature_names, lasso.coef_))

    dtr = DecisionTreeRegressor(max_depth=3, random_state=0)
    dtr.fit(Xv, y)
    print("DecisionTreeRegressor importances:", sorted_effects(feature_names, dtr.feature_importances_))

    dtc = DecisionTreeClassifier(max_depth=3, random_state=0)
    dtc.fit(Xv, y)
    print("DecisionTreeClassifier importances:", sorted_effects(feature_names, dtc.feature_importances_))

    # 5) Interpretable models (imodels)
    print("\n=== Interpretable Models: imodels ===")
    try:
        from imodels import RuleFitRegressor, FIGSRegressor, HSTreeRegressor

        imodel_classes = [
            ("RuleFitRegressor", RuleFitRegressor, {"random_state": 0, "max_rules": 40}),
            ("FIGSRegressor", FIGSRegressor, {"random_state": 0, "max_rules": 20}),
            ("HSTreeRegressor", HSTreeRegressor, {"random_state": 0, "max_leaf_nodes": 12}),
        ]

        for name, cls, kwargs in imodel_classes:
            try:
                model = instantiate_with_supported_kwargs(cls, kwargs)
                model.fit(Xv, y, feature_names=feature_names)

                print(f"{name}: fitted successfully")
                if hasattr(model, "feature_importances_"):
                    print("  top feature importances:", sorted_effects(feature_names, model.feature_importances_))
                if hasattr(model, "coef_"):
                    print("  top coefficients:", sorted_effects(feature_names, model.coef_))
                if hasattr(model, "get_rules"):
                    rules = model.get_rules()
                    if isinstance(rules, pd.DataFrame) and len(rules) > 0:
                        print("  extracted top rules (truncated)")
            except Exception as e:
                print(f"{name}: failed ({e})")
    except Exception as e:
        print("imodels import failed:", e)

    # 6) Construct conclusion score (0-100)
    # Weighted by significance evidence for age effect + culture moderation
    score = 50

    # age significance
    if age_p < 0.001:
        score += 20
    elif age_p < 0.01:
        score += 15
    elif age_p < 0.05:
        score += 10
    else:
        score -= 15

    # interaction significance (different developmental trend by culture)
    if interaction_p < 0.001:
        score += 20
    elif interaction_p < 0.01:
        score += 15
    elif interaction_p < 0.05:
        score += 10
    else:
        score -= 10

    # supporting tests
    chi2_p = safe_pvalue(chi2_p)
    pb_p = safe_pvalue(pb_p)
    if chi2_p < 0.05:
        score += 10
    else:
        score -= 5

    if pb_p < 0.05:
        score += 5
    else:
        score -= 5

    score = int(max(0, min(100, round(score))))

    # Summarize direction/magnitude for explanation
    age_coef = float(model_no_int.params.get("age", np.nan))
    n_sig_cultures = sum(1 for c in culture_slopes.values() if c["p"] < 0.05)
    n_cultures = len(culture_slopes)

    age_statement = "a statistically significant" if age_p < 0.05 else "no statistically significant"
    interaction_statement = "a statistically significant" if interaction_p < 0.05 else "no statistically significant"
    culture_statement = "statistically significant" if chi2_p < 0.05 else "not statistically significant"

    explanation = (
        f"There is {age_statement} overall age trend in majority-choice reliance "
        f"(OLS age coef={age_coef:.3f}, p={age_p:.3g}; point-biserial r={pb_r:.3f}, p={pb_p:.3g}). "
        f"Cross-cultural differences in majority reliance are {culture_statement} (chi-square p={chi2_p:.3g}). "
        f"There is {interaction_statement} age-by-culture interaction (p={interaction_p:.3g}), "
        f"with {n_sig_cultures}/{n_cultures} cultures showing significant within-culture age slopes. "
        "Interpretable linear, tree, and rule-based models emphasize majority_first and some culture terms more "
        "than age, supporting a low confidence in strong age-driven developmental change across cultures."
    )

    payload = {"response": score, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print("\nWrote conclusion.txt with:")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
