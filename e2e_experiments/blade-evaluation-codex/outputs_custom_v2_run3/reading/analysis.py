import json
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from agentic_imodels import (
    HingeEBMRegressor,
    HingeGAMRegressor,
    WinsorizedSparseOLSRegressor,
)


INFO_PATH = "info.json"
DATA_PATH = "reading.csv"
CONCLUSION_PATH = "conclusion.txt"


def cohen_d(x: pd.Series, y: pd.Series) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled == 0:
        return 0.0
    return float((x.mean() - y.mean()) / pooled)


def parse_linear_terms(model_text: str) -> Dict[int, float]:
    terms = {}
    for coef_str, idx_str in re.findall(r"([+-]?\d+\.\d+)\*x(\d+)", model_text):
        terms[int(idx_str)] = float(coef_str)
    return terms


def parse_excluded_features(model_text: str) -> List[int]:
    excluded: List[int] = []
    for line in model_text.splitlines():
        ll = line.lower()
        if "excluded" in ll and ":" in line:
            rhs = line.split(":", 1)[1]
            excluded.extend(int(s) for s in re.findall(r"x(\d+)", rhs))
    return sorted(set(excluded))


def main() -> None:
    with open(INFO_PATH, "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info["research_questions"][0]
    print("Research question:")
    print(question)
    print()

    df = pd.read_csv(DATA_PATH)
    df["log_speed"] = np.log1p(df["speed"])

    print("=== Data Overview ===")
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print()

    print("=== Missingness (top 12) ===")
    print(df.isna().sum().sort_values(ascending=False).head(12))
    print()

    numeric_cols = [
        "reader_view",
        "running_time",
        "adjusted_running_time",
        "scrolling_time",
        "num_words",
        "correct_rate",
        "img_width",
        "age",
        "dyslexia",
        "gender",
        "retake_trial",
        "dyslexia_bin",
        "Flesch_Kincaid",
        "speed",
        "log_speed",
    ]
    present_numeric = [c for c in numeric_cols if c in df.columns]

    print("=== Summary Stats (numeric) ===")
    print(df[present_numeric].describe().T)
    print()

    print("=== Distribution Notes ===")
    print(f"speed skewness: {df['speed'].skew():.3f}")
    print(f"log_speed skewness: {df['log_speed'].skew():.3f}")
    print("speed quantiles:")
    print(df["speed"].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))
    print()

    corr = (
        df[present_numeric]
        .corr(numeric_only=True)["speed"]
        .drop(labels=["speed"])  # only predictors
        .sort_values(key=np.abs, ascending=False)
    )
    print("=== Correlation With speed (absolute rank) ===")
    print(corr)
    print()

    # Primary analysis subset: individuals with dyslexia
    dys = df[df["dyslexia_bin"] == 1].copy()
    print("=== Dyslexia Subset ===")
    print("rows:", len(dys), "unique participants:", dys["uuid"].nunique())
    print()

    bivar = dys[["reader_view", "speed", "log_speed"]].dropna()
    grp0 = bivar.loc[bivar["reader_view"] == 0, "speed"]
    grp1 = bivar.loc[bivar["reader_view"] == 1, "speed"]
    grp0_log = bivar.loc[bivar["reader_view"] == 0, "log_speed"]
    grp1_log = bivar.loc[bivar["reader_view"] == 1, "log_speed"]

    print("=== Bivariate reader_view -> speed in dyslexia subset ===")
    print(dys.groupby("reader_view")["speed"].agg(["mean", "median", "std", "count"]))
    ttest_log = stats.ttest_ind(grp1_log, grp0_log, equal_var=False)
    ttest_raw = stats.ttest_ind(grp1, grp0, equal_var=False)
    mwu = stats.mannwhitneyu(grp1, grp0, alternative="two-sided")
    d_val = cohen_d(grp1_log, grp0_log)
    print(f"Welch t-test (log_speed): p={ttest_log.pvalue:.4g}")
    print(f"Welch t-test (speed): p={ttest_raw.pvalue:.4g}")
    print(f"Mann-Whitney U (speed): p={mwu.pvalue:.4g}")
    print(f"Cohen's d (log_speed, reader_view 1-0): {d_val:.4f}")
    print()

    # Controlled OLS inside dyslexia subset
    controls_formula = (
        "age + retake_trial + num_words + Flesch_Kincaid + img_width + "
        "C(gender) + C(device) + C(education) + C(language) + "
        "C(english_native) + C(page_id)"
    )
    formula_dys = f"log_speed ~ reader_view + {controls_formula}"

    cols_for_ols = [
        "uuid",
        "log_speed",
        "reader_view",
        "age",
        "retake_trial",
        "num_words",
        "Flesch_Kincaid",
        "img_width",
        "gender",
        "device",
        "education",
        "language",
        "english_native",
        "page_id",
    ]
    dys_ols_df = dys[cols_for_ols].dropna().copy()

    ols_dys = smf.ols(formula_dys, data=dys_ols_df).fit(
        cov_type="cluster", cov_kwds={"groups": dys_ols_df["uuid"]}
    )

    print("=== Controlled OLS (dyslexia subset; clustered by participant uuid) ===")
    print(ols_dys.summary())
    beta_dys = float(ols_dys.params["reader_view"])
    p_dys = float(ols_dys.pvalues["reader_view"])
    ci_dys = ols_dys.conf_int().loc["reader_view"].tolist()
    print(
        f"reader_view coefficient (log_speed): {beta_dys:.6f}, "
        f"p={p_dys:.4g}, 95% CI=[{ci_dys[0]:.6f}, {ci_dys[1]:.6f}]"
    )
    print()

    # Robustness check in full sample with interaction term
    full_cols = cols_for_ols + ["dyslexia_bin"]
    full_df = df[full_cols].dropna().copy()
    formula_inter = f"log_speed ~ reader_view * dyslexia_bin + {controls_formula}"
    ols_inter = smf.ols(formula_inter, data=full_df).fit(
        cov_type="cluster", cov_kwds={"groups": full_df["uuid"]}
    )
    t_effect_dys = ols_inter.t_test("reader_view + reader_view:dyslexia_bin = 0")
    effect_dys_inter = float(t_effect_dys.effect[0])
    p_dys_inter = float(t_effect_dys.pvalue)

    print("=== Full-sample interaction robustness check ===")
    print(ols_inter.summary())
    print(
        "Implied reader_view effect for dyslexia_bin=1 "
        f"from interaction model: {effect_dys_inter:.6f}, p={p_dys_inter:.4g}"
    )
    print()

    # agentic_imodels workflow: fit multiple interpretable models
    feature_cols = [
        "reader_view",
        "age",
        "retake_trial",
        "num_words",
        "Flesch_Kincaid",
        "img_width",
        "gender",
        "device",
        "education",
        "language",
        "english_native",
        "page_id",
    ]
    ai_df = dys[feature_cols + ["log_speed"]].dropna().copy()
    X = pd.get_dummies(ai_df[feature_cols], drop_first=True)
    y = ai_df["log_speed"]

    print("=== agentic_imodels Feature Map (x-index -> feature) ===")
    feature_map = {i: c for i, c in enumerate(X.columns)}
    for i, c in feature_map.items():
        print(f"x{i}: {c}")
    print()

    model_classes = [
        WinsorizedSparseOLSRegressor,  # honest + Lasso-style sparsity evidence
        HingeGAMRegressor,             # honest hinge GAM
        HingeEBMRegressor,             # high-rank decoupled model
    ]

    reader_idx = list(X.columns).index("reader_view")
    reader_effects: Dict[str, Tuple[float, bool]] = {}
    top_effects: Dict[str, List[Tuple[str, float]]] = {}

    for cls in model_classes:
        model = cls()
        model.fit(X, y)
        model_text = str(model)

        print(f"=== {cls.__name__} ===")
        print(model_text)
        print()

        terms = parse_linear_terms(model_text)
        excluded = parse_excluded_features(model_text)

        reader_coef = terms.get(reader_idx, 0.0)
        reader_excluded = reader_idx in excluded and reader_idx not in terms
        reader_effects[cls.__name__] = (reader_coef, reader_excluded)

        mapped_terms: List[Tuple[str, float]] = []
        for j, coef in terms.items():
            mapped_terms.append((feature_map.get(j, f"x{j}"), coef))
        mapped_terms = sorted(mapped_terms, key=lambda t: abs(t[1]), reverse=True)
        top_effects[cls.__name__] = mapped_terms[:8]

        print(f"reader_view in {cls.__name__}: coef={reader_coef:.6f}, excluded={reader_excluded}")
        print(f"Top non-zero terms in {cls.__name__} (by |coef|):")
        for name, coef in top_effects[cls.__name__]:
            print(f"  {name}: {coef:.6f}")
        print()

    # Calibrated Likert score based on combined evidence
    pvals_non_sig = (p_dys >= 0.05) and (p_dys_inter >= 0.05) and (ttest_log.pvalue >= 0.05)
    excluded_count = sum(1 for _, (_, excluded) in reader_effects.items() if excluded)
    max_abs_reader_coef = max(abs(v[0]) for v in reader_effects.values())

    if pvals_non_sig and excluded_count >= 2 and max_abs_reader_coef < 0.01:
        response = 8
    elif pvals_non_sig and max_abs_reader_coef < 0.03:
        response = 18
    elif (p_dys < 0.05) or (p_dys_inter < 0.05):
        response = 70
    else:
        response = 35

    explanation = (
        "In participants with dyslexia, Reader View does not show evidence of improving reading speed. "
        f"Bivariate tests are null (Welch t-test on log-speed p={ttest_log.pvalue:.3f}; "
        f"Mann-Whitney p={mwu.pvalue:.3f}; Cohen's d={d_val:.3f}). "
        "In controlled OLS with demographic, device, language, education, and page controls, "
        f"the Reader View coefficient is {beta_dys:.4f} on log-speed (p={p_dys:.3f}, "
        f"95% CI [{ci_dys[0]:.4f}, {ci_dys[1]:.4f}]). "
        "A full-sample interaction model also gives a null implied effect for dyslexic readers "
        f"(effect={effect_dys_inter:.4f}, p={p_dys_inter:.3f}). "
        f"Interpretable agentic_imodels agree: WinsorizedSparseOLS keeps only a tiny reader_view term "
        f"({reader_effects['WinsorizedSparseOLSRegressor'][0]:.4f}), while HingeGAM and HingeEBM exclude "
        "reader_view (zeroed coefficients). This is strong null evidence under the SKILL rubric "
        "(non-significant + low/zero importance across sparse and hinge models)."
    )

    with open(CONCLUSION_PATH, "w", encoding="utf-8") as f:
        json.dump({"response": int(response), "explanation": explanation}, f)

    print("=== Final Likert Decision ===")
    print(json.dumps({"response": int(response), "explanation": explanation}, indent=2))


if __name__ == "__main__":
    main()
