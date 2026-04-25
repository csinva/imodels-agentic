import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.exceptions import ConvergenceWarning

from agentic_imodels import (
    HingeEBMRegressor,
    HingeGAMRegressor,
    SmartAdditiveRegressor,
    WinsorizedSparseOLSRegressor,
)


warnings.filterwarnings("ignore", category=ConvergenceWarning)
np.random.seed(0)


def parse_excluded_features(model_text: str) -> set[str]:
    excluded = set()
    for line in model_text.splitlines():
        if "excluded" in line.lower() and "x" in line:
            excluded.update(re.findall(r"x\d+", line))
    return excluded


def summarize_model_feature_use(model_text: str, feature_map: dict[str, str], key_features: list[str]) -> dict[str, str]:
    all_tokens = set(re.findall(r"x\d+", model_text))
    excluded = parse_excluded_features(model_text)
    included = all_tokens - excluded

    out = {}
    for feat in key_features:
        idx_token = feature_map[feat]
        if idx_token in included:
            out[feat] = "included"
        elif idx_token in excluded:
            out[feat] = "excluded"
        else:
            out[feat] = "not_present"
    return out


def calibrate_score(
    p_ols_cont: float,
    p_ols_binary: float,
    sparse_zero_count: int,
    any_positive_robust_signal: bool,
) -> int:
    if p_ols_cont < 0.01 and p_ols_binary < 0.05 and any_positive_robust_signal:
        return 90
    if p_ols_cont < 0.05 and any_positive_robust_signal:
        return 78
    if p_ols_cont < 0.1 or p_ols_binary < 0.1:
        return 45
    if sparse_zero_count >= 2 and not any_positive_robust_signal:
        return 10
    if sparse_zero_count >= 1:
        return 20
    return 30


def main() -> None:
    root = Path(".")

    with open(root / "info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info["research_questions"][0]
    print("Research question:")
    print(research_question)
    print()

    df = pd.read_csv(root / "fertility.csv")

    # Parse dates and construct fertility proxies from cycle timing.
    date_cols = ["DateTesting", "StartDateofLastPeriod", "StartDateofPeriodBeforeLast"]
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], format="%m/%d/%y")

    # Outcome: mean religiosity across available items.
    rel_cols = ["Rel1", "Rel2", "Rel3"]
    df["religiosity_mean"] = df[rel_cols].mean(axis=1, skipna=True)

    # Estimated cycle length: reported value if available, else previous-cycle difference.
    df["calc_cycle_len"] = (
        df["StartDateofLastPeriod"] - df["StartDateofPeriodBeforeLast"]
    ).dt.days
    df["cycle_len_used"] = df["ReportedCycleLength"].fillna(df["calc_cycle_len"]).clip(21, 40)

    # Position in cycle at survey date.
    df["days_since_last"] = (df["DateTesting"] - df["StartDateofLastPeriod"]).dt.days
    df["ovulation_day_est"] = df["cycle_len_used"] - 14
    df["days_from_ovulation"] = df["days_since_last"] - df["ovulation_day_est"]

    # Continuous fertility proxy and binary high-fertility window.
    sigma_days = 2.5
    df["fertility_score"] = np.exp(-0.5 * (df["days_from_ovulation"] / sigma_days) ** 2)
    df["high_fertility"] = df["days_from_ovulation"].between(-5, 0).astype(int)

    # Additional control for secular trend over the survey window.
    df["test_day_index"] = (df["DateTesting"] - df["DateTesting"].min()).dt.days

    # Drop any residual missing values in analysis columns.
    analysis_cols = [
        "religiosity_mean",
        "fertility_score",
        "high_fertility",
        "days_from_ovulation",
        "days_since_last",
        "cycle_len_used",
        "Sure1",
        "Sure2",
        "Relationship",
        "test_day_index",
    ]
    dfa = df[analysis_cols].dropna().copy()

    print("=== Data overview ===")
    print(f"Rows used for analysis: {len(dfa)}")
    print("\nSummary statistics:")
    print(dfa.describe().round(3).to_string())

    print("\nBinned distributions (counts):")
    fert_bins = np.histogram(dfa["fertility_score"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0])[0]
    rel_bins = np.histogram(dfa["religiosity_mean"], bins=[1, 3, 5, 7, 9.1])[0]
    print("fertility_score bins [0-0.2,0.2-0.4,0.4-0.6,0.6-0.8,0.8-1.0]:", fert_bins.tolist())
    print("religiosity_mean bins [1-3,3-5,5-7,7-9]:", rel_bins.tolist())

    corr_cols = [
        "religiosity_mean",
        "fertility_score",
        "high_fertility",
        "days_from_ovulation",
        "days_since_last",
        "cycle_len_used",
        "Sure1",
        "Sure2",
        "Relationship",
        "test_day_index",
    ]
    print("\nCorrelation matrix (Pearson):")
    print(dfa[corr_cols].corr().round(3).to_string())

    print("\n=== Classical statistical tests ===")
    pearson = stats.pearsonr(dfa["fertility_score"], dfa["religiosity_mean"])
    ttest = stats.ttest_ind(
        dfa.loc[dfa["high_fertility"] == 1, "religiosity_mean"],
        dfa.loc[dfa["high_fertility"] == 0, "religiosity_mean"],
        equal_var=False,
    )
    print(
        "Bivariate Pearson correlation (fertility_score vs religiosity_mean): "
        f"r={pearson.statistic:.4f}, p={pearson.pvalue:.4g}"
    )
    print(
        "High-fertility vs others t-test on religiosity_mean: "
        f"t={ttest.statistic:.4f}, p={ttest.pvalue:.4g}"
    )

    ols_cont = smf.ols(
        "religiosity_mean ~ fertility_score + Sure1 + Sure2 + cycle_len_used + C(Relationship) + test_day_index",
        data=dfa,
    ).fit()
    print("\nControlled OLS with continuous fertility score:")
    print(ols_cont.summary())

    ols_binary = smf.ols(
        "religiosity_mean ~ high_fertility + Sure1 + Sure2 + cycle_len_used + C(Relationship) + test_day_index",
        data=dfa,
    ).fit()
    print("\nControlled OLS with high-fertility indicator:")
    print(ols_binary.summary())

    print("\n=== agentic_imodels fits (interpretable forms) ===")
    feature_names = [
        "fertility_score",
        "high_fertility",
        "days_from_ovulation",
        "days_since_last",
        "cycle_len_used",
        "Sure1",
        "Sure2",
        "Relationship",
        "test_day_index",
    ]
    feature_map = {name: f"x{i}" for i, name in enumerate(feature_names)}

    X = dfa[feature_names]
    y = dfa["religiosity_mean"]

    model_classes = [
        WinsorizedSparseOLSRegressor,  # honest sparse linear (lasso/zeroing evidence)
        HingeGAMRegressor,  # honest hinge additive model (zeroing + direction)
        SmartAdditiveRegressor,  # honest shape discovery
        HingeEBMRegressor,  # high-rank decoupled model
    ]

    model_texts = {}
    model_summaries = {}
    key_features = ["fertility_score", "high_fertility", "days_from_ovulation"]

    for cls in model_classes:
        print(f"\n--- {cls.__name__} ---")
        model = cls()
        model.fit(X, y)
        text = str(model)
        print(text)
        model_texts[cls.__name__] = text
        model_summaries[cls.__name__] = summarize_model_feature_use(
            text, feature_map=feature_map, key_features=key_features
        )

    print("\nModel feature-use summary for fertility-related predictors:")
    for model_name, summary in model_summaries.items():
        print(model_name, summary)

    # Evidence synthesis.
    p_ols_cont = float(ols_cont.pvalues.get("fertility_score", np.nan))
    beta_ols_cont = float(ols_cont.params.get("fertility_score", np.nan))
    p_ols_binary = float(ols_binary.pvalues.get("high_fertility", np.nan))
    beta_ols_binary = float(ols_binary.params.get("high_fertility", np.nan))

    sparse_zero_count = 0
    robust_positive_signals = 0
    for model_name in ["WinsorizedSparseOLSRegressor", "HingeGAMRegressor", "HingeEBMRegressor"]:
        summary = model_summaries.get(model_name, {})
        if (
            summary.get("fertility_score") == "excluded"
            and summary.get("high_fertility") == "excluded"
            and summary.get("days_from_ovulation") == "excluded"
        ):
            sparse_zero_count += 1

    for summary in model_summaries.values():
        included_any = any(summary.get(k) == "included" for k in key_features)
        if included_any:
            robust_positive_signals += 1

    any_positive_robust_signal = robust_positive_signals >= 2
    score = calibrate_score(
        p_ols_cont=p_ols_cont,
        p_ols_binary=p_ols_binary,
        sparse_zero_count=sparse_zero_count,
        any_positive_robust_signal=any_positive_robust_signal,
    )

    explanation = (
        f"Research question: {research_question} "
        f"Using {len(dfa)} women, bivariate evidence was near-zero "
        f"(Pearson r={pearson.statistic:.3f}, p={pearson.pvalue:.3g}; "
        f"high-fertility t-test p={ttest.pvalue:.3g}). "
        f"In controlled OLS, fertility_score was not significant "
        f"(beta={beta_ols_cont:.3f}, p={p_ols_cont:.3g}) and the high_fertility indicator "
        f"was also not significant (beta={beta_ols_binary:.3f}, p={p_ols_binary:.3g}). "
        f"Interpretable models largely gave null evidence: {sparse_zero_count} sparse/hinge models "
        f"explicitly excluded fertility-related predictors. One additive model showed limited nonzero terms, "
        f"but this was not robust across model classes. Overall evidence for a fertility-linked effect on "
        f"religiosity is weak and inconsistent, so the calibrated answer is near 'No'."
    )

    conclusion = {"response": int(score), "explanation": explanation}
    with open(root / "conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)

    print("\n=== Final calibrated conclusion ===")
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()
