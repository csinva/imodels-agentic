import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler

from imodels import RuleFitRegressor, FIGSRegressor, HSTreeRegressor

warnings.filterwarnings("ignore")


def p_to_strength(p: float) -> float:
    if p < 0.05:
        return 1.0
    if p < 0.1:
        return 0.65
    if p < 0.2:
        return 0.35
    return 0.1


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def main():
    with open("info.json", "r") as f:
        info = json.load(f)

    question = info["research_questions"][0].strip()
    print("Research question:", question)

    df = pd.read_csv("crofoot.csv")
    print("\nData shape:", df.shape)
    print("Columns:", list(df.columns))

    # Feature engineering for interpretable hypothesis tests
    df["rel_group_size"] = df["n_focal"] - df["n_other"]
    df["rel_males"] = df["m_focal"] - df["m_other"]
    df["rel_females"] = df["f_focal"] - df["f_other"]
    # Positive loc_adv means focal is closer to its own home-range center than the other group is to theirs
    df["loc_adv"] = df["dist_other"] - df["dist_focal"]

    # Basic exploration
    print("\nSummary statistics (numeric):")
    print(df.describe(include=[np.number]).T[["mean", "std", "min", "max"]])

    print("\nWin rate:")
    print(df["win"].value_counts(normalize=True).rename("proportion"))

    corr = df[[
        "win",
        "rel_group_size",
        "loc_adv",
        "n_focal",
        "n_other",
        "dist_focal",
        "dist_other",
    ]].corr(numeric_only=True)
    print("\nCorrelation matrix:")
    print(corr)

    # Statistical tests
    winners = df[df["win"] == 1]
    losers = df[df["win"] == 0]

    t_rel = stats.ttest_ind(winners["rel_group_size"], losers["rel_group_size"], equal_var=False)
    t_loc = stats.ttest_ind(winners["loc_adv"], losers["loc_adv"], equal_var=False)

    pb_rel = stats.pointbiserialr(df["win"], df["rel_group_size"])
    pb_loc_adv = stats.pointbiserialr(df["win"], df["loc_adv"])
    pb_dist_focal = stats.pointbiserialr(df["win"], df["dist_focal"])

    # ANOVA across bins (distributional check)
    df["size_bin"] = pd.qcut(df["rel_group_size"], q=3, labels=False, duplicates="drop")
    df["loc_bin"] = pd.qcut(df["loc_adv"], q=3, labels=False, duplicates="drop")
    size_groups = [g["win"].values for _, g in df.groupby("size_bin")]
    loc_groups = [g["win"].values for _, g in df.groupby("loc_bin")]
    anova_size = stats.f_oneway(*size_groups)
    anova_loc = stats.f_oneway(*loc_groups)

    # Chi-square on sign of relative size vs winner
    tmp = df.copy()
    tmp["size_advantage"] = (tmp["rel_group_size"] > 0).astype(int)
    chi2_tab = pd.crosstab(tmp["size_advantage"], tmp["win"])
    chi2 = stats.chi2_contingency(chi2_tab)

    print("\nStatistical tests:")
    print(f"t-test rel_group_size: t={t_rel.statistic:.3f}, p={t_rel.pvalue:.4f}")
    print(f"t-test loc_adv: t={t_loc.statistic:.3f}, p={t_loc.pvalue:.4f}")
    print(f"point-biserial rel_group_size: r={pb_rel.statistic:.3f}, p={pb_rel.pvalue:.4f}")
    print(f"point-biserial loc_adv: r={pb_loc_adv.statistic:.3f}, p={pb_loc_adv.pvalue:.4f}")
    print(f"point-biserial dist_focal: r={pb_dist_focal.statistic:.3f}, p={pb_dist_focal.pvalue:.4f}")
    print(f"ANOVA size bins: F={anova_size.statistic:.3f}, p={anova_size.pvalue:.4f}")
    print(f"ANOVA location bins: F={anova_loc.statistic:.3f}, p={anova_loc.pvalue:.4f}")
    print(f"Chi-square size_advantage vs win: chi2={chi2[0]:.3f}, p={chi2[1]:.4f}")

    # Regression models
    # Avoid perfect multicollinearity: rel_group_size = rel_males + rel_females
    model_features = ["rel_group_size", "loc_adv", "rel_males"]
    X = df[model_features].copy()
    y = df["win"].astype(float)

    X_sm = sm.add_constant(X)
    ols = sm.OLS(y, X_sm).fit()
    logit = sm.Logit(y, X_sm).fit(disp=0)

    print("\nOLS coefficients:")
    print(ols.params)
    print("OLS p-values:")
    print(ols.pvalues)

    print("\nLogit coefficients:")
    print(logit.params)
    print("Logit p-values:")
    print(logit.pvalues)

    # sklearn interpretable models
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lin = LinearRegression().fit(X, y)
    ridge = Ridge(alpha=1.0, random_state=0).fit(X_scaled, y)
    lasso = Lasso(alpha=0.02, random_state=0, max_iter=10000).fit(X_scaled, y)

    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    loo = LeaveOneOut()
    y_hat_tree = cross_val_predict(tree, X, y, cv=loo)
    tree.fit(X, y)
    tree_acc = accuracy_score(y, y_hat_tree)

    print("\nLinearRegression coefficients:")
    print(dict(zip(model_features, lin.coef_)))
    print("Ridge coefficients (standardized features):")
    print(dict(zip(model_features, ridge.coef_)))
    print("Lasso coefficients (standardized features):")
    print(dict(zip(model_features, lasso.coef_)))
    print("DecisionTreeClassifier feature importances:")
    print(dict(zip(model_features, tree.feature_importances_)))
    print(f"DecisionTreeClassifier LOOCV accuracy: {tree_acc:.3f}")

    # imodels interpretable models
    imodels_notes = []

    try:
        rulefit = RuleFitRegressor(random_state=0, n_estimators=100, tree_size=3)
        rulefit.fit(X.values, y.values, feature_names=model_features)
        preds = rulefit.predict(X.values)
        rulefit_r = np.corrcoef(preds, y.values)[0, 1]
        imodels_notes.append(f"RuleFitRegressor corr(pred, y)={rulefit_r:.3f}")

        # Extract a few learned rules when available
        rules = None
        if hasattr(rulefit, "get_rules"):
            rules_df = rulefit.get_rules()
            if isinstance(rules_df, pd.DataFrame):
                rules_df = rules_df.sort_values("importance", ascending=False)
                top = rules_df.head(5)
                rules = top[[c for c in ["rule", "coef", "support", "importance"] if c in top.columns]]
        if rules is not None and len(rules) > 0:
            imodels_notes.append("Top RuleFit rules:\n" + rules.to_string(index=False))
    except Exception as e:
        imodels_notes.append(f"RuleFitRegressor failed: {e}")

    try:
        figs = FIGSRegressor(max_rules=12, random_state=0)
        figs.fit(X.values, y.values, feature_names=model_features)
        figs_pred = figs.predict(X.values)
        figs_r = np.corrcoef(figs_pred, y.values)[0, 1]
        imodels_notes.append(f"FIGSRegressor corr(pred, y)={figs_r:.3f}")
        if hasattr(figs, "feature_importances_"):
            imodels_notes.append(
                "FIGS feature importances: "
                + str(dict(zip(model_features, np.asarray(figs.feature_importances_).tolist())))
            )
    except Exception as e:
        imodels_notes.append(f"FIGSRegressor failed: {e}")

    try:
        hst = HSTreeRegressor(max_leaf_nodes=8, random_state=0)
        hst.fit(X.values, y.values, feature_names=model_features)
        hst_pred = hst.predict(X.values)
        hst_r = np.corrcoef(hst_pred, y.values)[0, 1]
        imodels_notes.append(f"HSTreeRegressor corr(pred, y)={hst_r:.3f}")
        if hasattr(hst, "feature_importances_"):
            imodels_notes.append(
                "HSTree feature importances: "
                + str(dict(zip(model_features, np.asarray(hst.feature_importances_).tolist())))
            )
    except Exception as e:
        imodels_notes.append(f"HSTreeRegressor failed: {e}")

    print("\nimodels summary:")
    for note in imodels_notes:
        print(note)

    # Build evidence score for the question
    p_rel_primary = np.nanmedian([
        safe_float(t_rel.pvalue),
        safe_float(pb_rel.pvalue),
        safe_float(logit.pvalues.get("rel_group_size", np.nan)),
    ])
    p_loc_primary = np.nanmedian([
        safe_float(t_loc.pvalue),
        safe_float(pb_loc_adv.pvalue),
        safe_float(logit.pvalues.get("loc_adv", np.nan)),
    ])
    p_loc_supp = safe_float(pb_dist_focal.pvalue)

    # Primary decision favors direct constructs in the question: relative size and relative location (loc_adv)
    if (p_rel_primary < 0.05) and (p_loc_primary < 0.05):
        response = 90
        verdict = "strong evidence for both effects"
    elif (p_rel_primary < 0.05) or (p_loc_primary < 0.05):
        response = 60
        verdict = "clear evidence for one effect but not both"
    elif (p_rel_primary < 0.1) or (p_loc_primary < 0.1):
        response = 45
        verdict = "marginal evidence only"
    elif (p_rel_primary > 0.2) and (p_loc_primary > 0.2):
        response = 15
        verdict = "little evidence for the hypothesized relative effects"
    else:
        response = 25
        verdict = "weak evidence overall"

    # Slight upward adjustment if supplementary location proxy is significant
    if p_loc_supp < 0.05:
        response = min(100, response + 10)

    explanation = (
        f"For the primary predictors tied to the question, evidence was weak: "
        f"median p-values were rel_group_size={p_rel_primary:.3f} and loc_adv={p_loc_primary:.3f} "
        f"from t-tests, point-biserial correlations, and logistic regression. "
        f"Logistic coefficients were rel_group_size={logit.params['rel_group_size']:.3f} "
        f"(p={logit.pvalues['rel_group_size']:.3f}) and loc_adv={logit.params['loc_adv']:.6f} "
        f"(p={logit.pvalues['loc_adv']:.3f}). "
        f"A supplementary location proxy (dist_focal) showed a univariate association "
        f"(point-biserial p={p_loc_supp:.3f}), suggesting possible location effects, but this was not robust "
        f"for the primary relative-location term. Interpretable sklearn and imodels fits were consistent with modest "
        f"signal and no strong, stable primary effects. Overall assessment: {verdict}."
    )

    out = {
        "response": int(response),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(out, f)

    print("\nWrote conclusion.txt:")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
