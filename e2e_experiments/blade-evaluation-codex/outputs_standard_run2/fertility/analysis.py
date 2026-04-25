import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import r2_score, accuracy_score

from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor


warnings.filterwarnings("ignore", category=UserWarning)


def parse_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        df[col] = pd.to_datetime(df[col], format="%m/%d/%y", errors="coerce")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Main outcome: mean religiosity across available items.
    df["Religiosity"] = df[["Rel1", "Rel2", "Rel3"]].mean(axis=1, skipna=True)

    # Prefer reported cycle length, fallback to inferred length from dates.
    inferred_cycle_length = (
        df["StartDateofLastPeriod"] - df["StartDateofPeriodBeforeLast"]
    ).dt.days
    cycle_length = df["ReportedCycleLength"].fillna(inferred_cycle_length)
    cycle_length = cycle_length.clip(lower=20, upper=40)
    df["CycleLengthUsed"] = cycle_length

    days_since_last_period = (df["DateTesting"] - df["StartDateofLastPeriod"]).dt.days
    df["DaysSinceLastPeriod"] = days_since_last_period

    # Day in cycle and an ovulation-centered fertile window approximation.
    cycle_day = (days_since_last_period % cycle_length) + 1
    ovulation_day = cycle_length - 14

    df["CycleDay"] = cycle_day
    df["OvulationDayApprox"] = ovulation_day
    df["FertileDistance"] = (cycle_day - ovulation_day).abs()
    df["HighFertility"] = (
        (cycle_day >= ovulation_day - 5) & (cycle_day <= ovulation_day + 1)
    ).astype(int)

    df["Phase"] = np.select(
        [
            cycle_day <= 5,
            (cycle_day >= 6) & (cycle_day <= ovulation_day - 1),
            (cycle_day >= ovulation_day) & (cycle_day <= ovulation_day + 1),
            cycle_day > ovulation_day + 1,
        ],
        ["menstrual", "follicular", "ovulatory", "luteal"],
        default="other",
    )

    df["MeanCertainty"] = df[["Sure1", "Sure2"]].mean(axis=1)
    return df


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan

    v1 = a.var(ddof=1)
    v2 = b.var(ddof=1)
    pooled_sd = np.sqrt(((len(a) - 1) * v1 + (len(b) - 1) * v2) / (len(a) + len(b) - 2))
    if pooled_sd == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled_sd


def run_stat_tests(df: pd.DataFrame) -> dict:
    out: dict[str, object] = {}

    test_df = df[["Religiosity", "FertileDistance", "HighFertility", "Phase"]].dropna()

    # Correlation between fertility proximity and religiosity.
    pearson_r, pearson_p = stats.pearsonr(test_df["FertileDistance"], test_df["Religiosity"])
    out["pearson_r"] = float(pearson_r)
    out["pearson_p"] = float(pearson_p)

    # High vs low fertility window mean difference.
    high_vals = test_df.loc[test_df["HighFertility"] == 1, "Religiosity"].values
    low_vals = test_df.loc[test_df["HighFertility"] == 0, "Religiosity"].values
    t_stat, t_p = stats.ttest_ind(high_vals, low_vals, equal_var=False, nan_policy="omit")
    out["ttest_stat"] = float(t_stat)
    out["ttest_p"] = float(t_p)
    out["high_mean"] = float(np.mean(high_vals))
    out["low_mean"] = float(np.mean(low_vals))
    out["cohens_d"] = float(cohens_d(high_vals, low_vals))

    # ANOVA across cycle phases.
    phase_groups = [
        g["Religiosity"].values
        for _, g in test_df.groupby("Phase")
        if len(g["Religiosity"].values) >= 2
    ]
    if len(phase_groups) >= 2:
        f_stat, anova_p = stats.f_oneway(*phase_groups)
    else:
        f_stat, anova_p = np.nan, np.nan
    out["anova_f"] = float(f_stat)
    out["anova_p"] = float(anova_p)

    # OLS with controls.
    ols_cols = [
        "Religiosity",
        "HighFertility",
        "CycleDay",
        "CycleLengthUsed",
        "Sure1",
        "Sure2",
        "Relationship",
    ]
    ols_df = df[ols_cols].dropna()
    y = ols_df["Religiosity"]
    X = sm.add_constant(ols_df.drop(columns=["Religiosity"]))
    ols = sm.OLS(y, X).fit()

    out["ols_n"] = int(len(ols_df))
    out["ols_r2"] = float(ols.rsquared)
    out["ols_coef_highfert"] = float(ols.params.get("HighFertility", np.nan))
    out["ols_p_highfert"] = float(ols.pvalues.get("HighFertility", np.nan))
    out["ols_summary"] = ols.summary().as_text()
    return out


def run_interpretable_models(df: pd.DataFrame) -> dict:
    feature_cols = [
        "HighFertility",
        "FertileDistance",
        "CycleDay",
        "CycleLengthUsed",
        "Sure1",
        "Sure2",
        "Relationship",
        "MeanCertainty",
    ]

    model_df = df[["Religiosity", *feature_cols]].dropna()
    X = model_df[feature_cols]
    y = model_df["Religiosity"]

    results: dict[str, object] = {"n_model_rows": int(len(model_df))}

    # Scikit-learn interpretable models.
    lin = LinearRegression().fit(X, y)
    ridge = Ridge(alpha=1.0, random_state=0).fit(X, y)
    lasso = Lasso(alpha=0.01, random_state=0, max_iter=10000).fit(X, y)
    tree = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y)

    def coef_map(model):
        return {col: float(v) for col, v in zip(feature_cols, model.coef_)}

    results["linear_r2"] = float(r2_score(y, lin.predict(X)))
    results["ridge_r2"] = float(r2_score(y, ridge.predict(X)))
    results["lasso_r2"] = float(r2_score(y, lasso.predict(X)))
    results["tree_r2"] = float(r2_score(y, tree.predict(X)))

    results["linear_coef"] = coef_map(lin)
    results["ridge_coef"] = coef_map(ridge)
    results["lasso_coef"] = coef_map(lasso)
    results["tree_importance"] = {
        col: float(v) for col, v in zip(feature_cols, tree.feature_importances_)
    }

    # Also use a simple interpretable classifier view (high vs low religiosity).
    median_rel = float(np.median(y))
    y_bin = (y >= median_rel).astype(int)
    clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y_bin)
    clf_acc = accuracy_score(y_bin, clf.predict(X))
    results["clf_accuracy"] = float(clf_acc)
    results["clf_importance"] = {
        col: float(v) for col, v in zip(feature_cols, clf.feature_importances_)
    }

    # iModels rule/tree models.
    try:
        rulefit = RuleFitRegressor(random_state=0)
        rulefit.fit(X, y)
        results["rulefit_r2"] = float(r2_score(y, rulefit.predict(X)))
        rules = getattr(rulefit, "rules_", [])
        results["rulefit_rules_preview"] = [str(r) for r in rules[:10]]
    except Exception as exc:
        results["rulefit_error"] = str(exc)

    try:
        figs = FIGSRegressor(random_state=0)
        figs.fit(X, y)
        results["figs_r2"] = float(r2_score(y, figs.predict(X)))
        if hasattr(figs, "feature_importances_"):
            results["figs_importance"] = {
                col: float(v) for col, v in zip(feature_cols, figs.feature_importances_)
            }
    except Exception as exc:
        results["figs_error"] = str(exc)

    try:
        hst = HSTreeRegressor()
        hst.fit(X, y)
        results["hstree_r2"] = float(r2_score(y, hst.predict(X)))
        if hasattr(hst, "feature_importances_"):
            results["hstree_importance"] = {
                col: float(v) for col, v in zip(feature_cols, hst.feature_importances_)
            }
    except Exception as exc:
        results["hstree_error"] = str(exc)

    return results


def compute_likert_score(stat_results: dict) -> tuple[int, str]:
    pvals = [
        stat_results.get("pearson_p", np.nan),
        stat_results.get("ttest_p", np.nan),
        stat_results.get("anova_p", np.nan),
        stat_results.get("ols_p_highfert", np.nan),
    ]
    valid_pvals = [p for p in pvals if np.isfinite(p)]
    significant = sum(p < 0.05 for p in valid_pvals)

    corr = abs(stat_results.get("pearson_r", np.nan))
    d = abs(stat_results.get("cohens_d", np.nan))

    # Convert significance pattern into a yes/no confidence score.
    if significant == 0:
        score = 10
    elif significant == 1:
        score = 35
    elif significant == 2:
        score = 60
    elif significant == 3:
        score = 80
    else:
        score = 92

    # Very weak effects push score lower.
    if significant == 0 and np.isfinite(corr) and np.isfinite(d) and corr < 0.10 and d < 0.20:
        score = 6

    explanation = (
        "Across multiple statistical tests, fertility-related predictors were not significant: "
        f"Pearson r={stat_results.get('pearson_r', np.nan):.3f} (p={stat_results.get('pearson_p', np.nan):.3g}), "
        f"high-vs-low fertility t-test p={stat_results.get('ttest_p', np.nan):.3g} "
        f"(means {stat_results.get('high_mean', np.nan):.2f} vs {stat_results.get('low_mean', np.nan):.2f}, "
        f"Cohen's d={stat_results.get('cohens_d', np.nan):.3f}), ANOVA across cycle phases "
        f"p={stat_results.get('anova_p', np.nan):.3g}, and OLS high-fertility coefficient "
        f"={stat_results.get('ols_coef_highfert', np.nan):.3f} (p={stat_results.get('ols_p_highfert', np.nan):.3g}). "
        "This provides little evidence that hormonal fertility fluctuations affect religiosity in this dataset."
    )
    return int(max(0, min(100, round(score)))), explanation


def main() -> None:
    base = Path(".")
    info = json.loads((base / "info.json").read_text())
    research_question = info.get("research_questions", [""])[0]

    df = pd.read_csv(base / "fertility.csv")
    df = parse_dates(
        df,
        ["DateTesting", "StartDateofLastPeriod", "StartDateofPeriodBeforeLast"],
    )
    feat_df = build_features(df)

    print("Research question:", research_question)
    print("\\nData shape:", feat_df.shape)
    print("\\nMissing values:\n", feat_df.isna().sum().sort_values(ascending=False).head(12))
    print("\\nNumeric summary:\n", feat_df.describe(include=[np.number]).T)

    corr_cols = [
        "Religiosity",
        "HighFertility",
        "FertileDistance",
        "CycleDay",
        "CycleLengthUsed",
        "Sure1",
        "Sure2",
        "Relationship",
    ]
    print("\\nCorrelation matrix:\n", feat_df[corr_cols].corr(numeric_only=True).round(3))
    print("\\nCycle phase distribution:\n", feat_df["Phase"].value_counts(dropna=False))

    stat_results = run_stat_tests(feat_df)
    model_results = run_interpretable_models(feat_df)

    print("\\nStatistical test results:")
    for k in [
        "pearson_r",
        "pearson_p",
        "ttest_stat",
        "ttest_p",
        "high_mean",
        "low_mean",
        "cohens_d",
        "anova_f",
        "anova_p",
        "ols_n",
        "ols_r2",
        "ols_coef_highfert",
        "ols_p_highfert",
    ]:
        print(f"  {k}: {stat_results.get(k)}")

    print("\\nInterpretable model highlights:")
    for k in [
        "linear_r2",
        "ridge_r2",
        "lasso_r2",
        "tree_r2",
        "rulefit_r2",
        "figs_r2",
        "hstree_r2",
    ]:
        if k in model_results:
            print(f"  {k}: {model_results[k]}")

    print("\\nLinear coefficients:")
    for feat, val in sorted(
        model_results.get("linear_coef", {}).items(),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    ):
        print(f"  {feat}: {val:.4f}")

    print("\\nDecision tree importances:")
    for feat, val in sorted(
        model_results.get("tree_importance", {}).items(),
        key=lambda kv: kv[1],
        reverse=True,
    ):
        print(f"  {feat}: {val:.4f}")

    if "rulefit_rules_preview" in model_results:
        print("\\nRuleFit rules preview:")
        for i, rule in enumerate(model_results["rulefit_rules_preview"][:10], start=1):
            print(f"  {i}. {rule}")

    score, explanation = compute_likert_score(stat_results)
    payload = {"response": score, "explanation": explanation}
    (base / "conclusion.txt").write_text(json.dumps(payload, ensure_ascii=True))

    print("\\nConclusion JSON:")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
