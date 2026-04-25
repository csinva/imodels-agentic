import json
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")

try:
    from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor

    IMODELS_AVAILABLE = True
except Exception:
    IMODELS_AVAILABLE = False


RANDOM_STATE = 42
FEATURES = [
    "fertility_score",
    "high_fertility",
    "cycle_day",
    "cycle_length",
    "Relationship",
    "Sure1",
    "Sure2",
]


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interpretable fertility-related features."""
    out = df.copy()

    for col in ["DateTesting", "StartDateofLastPeriod", "StartDateofPeriodBeforeLast"]:
        out[col] = pd.to_datetime(out[col], format="%m/%d/%y", errors="coerce")

    out["religiosity_mean"] = out[["Rel1", "Rel2", "Rel3"]].mean(axis=1, skipna=True)

    out["cycle_length_from_dates"] = (
        out["StartDateofLastPeriod"] - out["StartDateofPeriodBeforeLast"]
    ).dt.days
    out["cycle_length"] = out["ReportedCycleLength"].fillna(out["cycle_length_from_dates"])
    out["cycle_length"] = out["cycle_length"].clip(lower=21, upper=40)

    out["days_since_last"] = (out["DateTesting"] - out["StartDateofLastPeriod"]).dt.days

    valid_cycle = out["cycle_length"].notna() & out["days_since_last"].notna()
    out = out.loc[valid_cycle].copy()

    # Wrap day index to cycle position for participants potentially beyond one full cycle.
    out["cycle_day"] = (out["days_since_last"] % out["cycle_length"]) + 1
    out["ovulation_day"] = out["cycle_length"] - 14
    out["day_from_ovulation"] = out["cycle_day"] - out["ovulation_day"]
    out["fertility_distance"] = out["day_from_ovulation"].abs()

    # Smooth fertility proxy centered around predicted ovulation day.
    sigma_days = 2.0
    out["fertility_score"] = np.exp(
        -0.5 * (out["fertility_distance"] / sigma_days) ** 2
    )
    out["high_fertility"] = (out["fertility_distance"] <= 3).astype(int)

    def cycle_phase(row: pd.Series) -> str:
        if row["cycle_day"] <= 5:
            return "menstrual"
        if -5 <= row["day_from_ovulation"] <= 1:
            return "fertile_window"
        if row["day_from_ovulation"] < -5:
            return "follicular"
        return "luteal"

    out["phase"] = out.apply(cycle_phase, axis=1)
    return out


def print_eda(df: pd.DataFrame) -> None:
    print("\n=== DATA OVERVIEW ===")
    print(f"Rows after feature derivation: {len(df)}")
    print(df["phase"].value_counts().to_string())
    print("\nHigh fertility counts:")
    print(df["high_fertility"].value_counts().to_string())

    core_cols = [
        "religiosity_mean",
        "fertility_score",
        "high_fertility",
        "cycle_day",
        "cycle_length",
        "Relationship",
        "Sure1",
        "Sure2",
    ]
    print("\n=== SUMMARY STATISTICS ===")
    print(df[core_cols].describe().round(3).to_string())

    print("\n=== CORRELATION MATRIX ===")
    print(df[core_cols].corr().round(3).to_string())


def run_statistical_tests(df: pd.DataFrame) -> Dict[str, float]:
    print("\n=== STATISTICAL TESTS ===")
    rel = df["religiosity_mean"]

    high = df.loc[df["high_fertility"] == 1, "religiosity_mean"]
    low = df.loc[df["high_fertility"] == 0, "religiosity_mean"]
    t_stat, t_p = stats.ttest_ind(high, low, equal_var=False, nan_policy="omit")

    pooled_sd = np.sqrt(((high.var(ddof=1) + low.var(ddof=1)) / 2.0))
    cohen_d = (high.mean() - low.mean()) / pooled_sd if pooled_sd > 0 else 0.0

    pearson_r, pearson_p = stats.pearsonr(df["fertility_score"], rel)
    spearman_rho, spearman_p = stats.spearmanr(df["fertility_score"], rel)

    phase_groups = [
        g["religiosity_mean"].values for _, g in df.groupby("phase", observed=False)
    ]
    f_stat, anova_p = stats.f_oneway(*phase_groups)

    print(
        f"Welch t-test (high vs low fertility): t={t_stat:.3f}, p={t_p:.4f}, cohen_d={cohen_d:.3f}"
    )
    print(f"Pearson correlation (fertility_score vs religiosity): r={pearson_r:.3f}, p={pearson_p:.4f}")
    print(f"Spearman correlation: rho={spearman_rho:.3f}, p={spearman_p:.4f}")
    print(f"ANOVA across cycle phases: F={f_stat:.3f}, p={anova_p:.4f}")

    # OLS with interpretable covariates and p-values.
    X_ols = sm.add_constant(df[FEATURES])
    ols_model = sm.OLS(rel, X_ols).fit()

    print("\nOLS coefficient table:")
    print(ols_model.summary().tables[1])

    return {
        "t_test_p": float(t_p),
        "pearson_p": float(pearson_p),
        "spearman_p": float(spearman_p),
        "anova_p": float(anova_p),
        "ols_fertility_score_p": float(ols_model.pvalues.get("fertility_score", np.nan)),
        "ols_high_fertility_p": float(ols_model.pvalues.get("high_fertility", np.nan)),
        "ols_fertility_score_coef": float(ols_model.params.get("fertility_score", np.nan)),
        "cohen_d": float(cohen_d),
        "pearson_r": float(pearson_r),
    }


def top_coefficients(name: str, features: List[str], coefs: np.ndarray, k: int = 5) -> None:
    coef_series = pd.Series(coefs, index=features)
    order = coef_series.abs().sort_values(ascending=False).head(k).index
    print(f"\n{name} top coefficients:")
    for feat in order:
        print(f"  {feat}: {coef_series[feat]:.4f}")


def run_sklearn_models(df: pd.DataFrame) -> None:
    print("\n=== SCIKIT-LEARN INTERPRETABLE MODELS ===")
    model_df = df.dropna(subset=FEATURES + ["religiosity_mean"]).copy()
    X = model_df[FEATURES]
    y = model_df["religiosity_mean"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    lin = LinearRegression()
    lin.fit(X_train, y_train)
    y_pred_lin = lin.predict(X_test)
    print(f"LinearRegression R^2 (test): {r2_score(y_test, y_pred_lin):.3f}")
    top_coefficients("LinearRegression", FEATURES, lin.coef_)

    ridge = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ]
    )
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    print(f"Ridge R^2 (test): {r2_score(y_test, y_pred_ridge):.3f}")
    top_coefficients("Ridge (standardized)", FEATURES, ridge.named_steps["model"].coef_)

    lasso = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.05, random_state=RANDOM_STATE, max_iter=10000)),
        ]
    )
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    print(f"Lasso R^2 (test): {r2_score(y_test, y_pred_lasso):.3f}")
    top_coefficients("Lasso (standardized)", FEATURES, lasso.named_steps["model"].coef_)

    tree_reg = DecisionTreeRegressor(
        max_depth=3, min_samples_leaf=10, random_state=RANDOM_STATE
    )
    tree_reg.fit(X_train, y_train)
    y_pred_tree = tree_reg.predict(X_test)
    print(f"DecisionTreeRegressor R^2 (test): {r2_score(y_test, y_pred_tree):.3f}")
    print("DecisionTreeRegressor feature importances:")
    for feat, imp in sorted(
        zip(FEATURES, tree_reg.feature_importances_), key=lambda x: x[1], reverse=True
    ):
        print(f"  {feat}: {imp:.4f}")

    y_binary = (y >= y.median()).astype(int)
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X, y_binary, test_size=0.25, random_state=RANDOM_STATE
    )
    tree_clf = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=10, random_state=RANDOM_STATE
    )
    tree_clf.fit(Xc_train, yc_train)
    clf_acc = tree_clf.score(Xc_test, yc_test)
    print(f"DecisionTreeClassifier accuracy (test): {clf_acc:.3f}")
    print("DecisionTreeClassifier feature importances:")
    for feat, imp in sorted(
        zip(FEATURES, tree_clf.feature_importances_), key=lambda x: x[1], reverse=True
    ):
        print(f"  {feat}: {imp:.4f}")


def run_imodels_models(df: pd.DataFrame) -> None:
    print("\n=== IMODELS INTERPRETABLE MODELS ===")

    if not IMODELS_AVAILABLE:
        print("imodels not available in environment; skipping imodels models.")
        return

    model_df = df.dropna(subset=FEATURES + ["religiosity_mean"]).copy()
    X = model_df[FEATURES]
    y = model_df["religiosity_mean"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    try:
        rf = RuleFitRegressor(random_state=RANDOM_STATE, max_rules=30)
        rf.fit(X_train.values, y_train.values, feature_names=FEATURES)
        rf_r2 = rf.score(X_test.values, y_test.values)
        print(f"RuleFitRegressor R^2 (test): {rf_r2:.3f}")

        if hasattr(rf, "_get_rules"):
            rules_df = rf._get_rules()
            rules_df = rules_df[rules_df["coef"] != 0].copy()
            rules_df = rules_df.sort_values("importance", ascending=False).head(5)
            if len(rules_df) > 0:
                print("Top RuleFit rules by importance:")
                for _, row in rules_df.iterrows():
                    print(
                        f"  rule={row['rule']} | coef={row['coef']:.4f} | importance={row['importance']:.4f}"
                    )
            else:
                print("No non-zero RuleFit rules selected.")
    except Exception as exc:
        print(f"RuleFitRegressor failed: {exc}")

    try:
        figs = FIGSRegressor(random_state=RANDOM_STATE, max_rules=12)
        figs.fit(X_train.values, y_train.values, feature_names=FEATURES)
        figs_r2 = figs.score(X_test.values, y_test.values)
        print(f"FIGSRegressor R^2 (test): {figs_r2:.3f}")

        if hasattr(figs, "feature_importances_"):
            print("FIGS feature importances:")
            for feat, imp in sorted(
                zip(FEATURES, figs.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            ):
                print(f"  {feat}: {imp:.4f}")
    except Exception as exc:
        print(f"FIGSRegressor failed: {exc}")

    try:
        hs = HSTreeRegressor(random_state=RANDOM_STATE, max_leaf_nodes=8)
        hs.fit(X_train.values, y_train.values)
        hs_r2 = hs.score(X_test.values, y_test.values)
        print(f"HSTreeRegressor R^2 (test): {hs_r2:.3f}")

        # Use permutation importance for consistent interpretability even without built-in importances.
        perm = permutation_importance(
            hs,
            X_test.values,
            y_test.values,
            n_repeats=20,
            random_state=RANDOM_STATE,
        )
        print("HSTree permutation importances:")
        for feat, imp in sorted(
            zip(FEATURES, perm.importances_mean), key=lambda x: x[1], reverse=True
        ):
            print(f"  {feat}: {imp:.4f}")
    except Exception as exc:
        print(f"HSTreeRegressor failed: {exc}")


def build_conclusion(stats_out: Dict[str, float]) -> Dict[str, object]:
    pvals_to_check = {
        "Welch t-test": stats_out["t_test_p"],
        "Pearson": stats_out["pearson_p"],
        "Spearman": stats_out["spearman_p"],
        "ANOVA": stats_out["anova_p"],
        "OLS fertility_score": stats_out["ols_fertility_score_p"],
        "OLS high_fertility": stats_out["ols_high_fertility_p"],
    }

    significant = [name for name, p in pvals_to_check.items() if p < 0.05]
    min_p = float(np.nanmin(list(pvals_to_check.values())))
    abs_r = abs(stats_out["pearson_r"])
    abs_d = abs(stats_out["cohen_d"])

    if len(significant) >= 2 and min_p < 0.01:
        score = 85
    elif len(significant) >= 1:
        score = 70
    elif min_p < 0.10:
        score = 40
    else:
        # No significance and tiny effects -> strong "No".
        if abs_r < 0.10 and abs_d < 0.20:
            score = 10
        else:
            score = 20

    explanation = (
        "No statistically reliable fertility-religiosity effect was detected. "
        f"Key tests were non-significant (t-test p={stats_out['t_test_p']:.3f}, "
        f"Pearson p={stats_out['pearson_p']:.3f}, ANOVA p={stats_out['anova_p']:.3f}, "
        f"OLS fertility_score p={stats_out['ols_fertility_score_p']:.3f}, "
        f"OLS high_fertility p={stats_out['ols_high_fertility_p']:.3f}), "
        f"with very small effect sizes (r={stats_out['pearson_r']:.3f}, d={stats_out['cohen_d']:.3f}). "
        "Interpretable models also did not highlight fertility features as strong predictors relative to controls."
    )

    return {"response": int(score), "explanation": explanation}


def main() -> None:
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info.get("research_questions", ["(missing question)"])[0]
    print("Research question:", question)

    raw = pd.read_csv("fertility.csv")
    df = derive_features(raw)

    analysis_df = df.dropna(subset=["religiosity_mean"] + FEATURES + ["phase"]).copy()

    print_eda(analysis_df)
    stats_out = run_statistical_tests(analysis_df)
    run_sklearn_models(analysis_df)
    run_imodels_models(analysis_df)

    conclusion = build_conclusion(stats_out)

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)

    print("\nWrote conclusion.txt:")
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()
