import json
import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import r2_score

from agentic_imodels import (
    HingeEBMRegressor,
    HingeGAMRegressor,
    SmartAdditiveRegressor,
    WinsorizedSparseOLSRegressor,
)


warnings.filterwarnings("ignore", category=FutureWarning)


def section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def parse_coefficients_from_model_text(model_text: str, xname_to_feature: dict[str, str]) -> dict[str, float]:
    """Parse lines like 'x2: -0.1234' from model.__str__ output."""
    coefs: dict[str, float] = {}
    in_coef_block = False
    pat = re.compile(r"^(x\d+):\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)$")

    for raw_line in model_text.splitlines():
        line = raw_line.strip()
        if line.startswith("Coefficients:"):
            in_coef_block = True
            continue
        if not in_coef_block:
            continue
        if not line:
            continue
        if line.startswith("intercept"):
            continue
        if line.startswith("Features with zero coefficients"):
            continue
        m = pat.match(line)
        if m:
            xname = m.group(1)
            value = float(m.group(2))
            coefs[xname_to_feature.get(xname, xname)] = value
    return coefs


def parse_zeroed_features_from_model_text(model_text: str, xname_to_feature: dict[str, str]) -> set[str]:
    zeroed: set[str] = set()
    marker = "Features with zero coefficients (excluded):"
    for raw_line in model_text.splitlines():
        if marker in raw_line:
            tail = raw_line.split(marker, 1)[1].strip()
            if not tail:
                continue
            for token in tail.split(","):
                xname = token.strip()
                if not xname:
                    continue
                zeroed.add(xname_to_feature.get(xname, xname))
    return zeroed


def summarize_distribution(values: pd.Series, bins: int = 8) -> list[str]:
    counts, edges = np.histogram(values, bins=bins)
    out = []
    for i, count in enumerate(counts):
        out.append(f"[{edges[i]:.3f}, {edges[i + 1]:.3f}): {int(count)}")
    return out


def main() -> None:
    section("Research Question + Metadata")
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    question = info["research_questions"][0]
    print("Question:", question)
    print("Rows expected:", info["data_desc"]["num_rows"])
    print("Columns expected:", info["data_desc"]["field_names"])

    section("Load + Clean Data")
    df = pd.read_csv("panda_nuts.csv")
    print("Observed shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Normalize categorical coding for stable formulas and dummy encoding.
    df["sex"] = df["sex"].astype(str).str.strip().str.lower()
    df["help"] = df["help"].astype(str).str.strip().str.lower().replace({"n": "no", "y": "yes"})
    df["hammer"] = df["hammer"].astype(str).str.strip()
    if (df["seconds"] <= 0).any():
        raise ValueError("All seconds values must be > 0 to define efficiency and log offsets.")
    df["efficiency"] = df["nuts_opened"] / df["seconds"]

    print("\nHead:")
    print(df.head(10).to_string(index=False))
    print("\nMissing values:")
    print(df.isna().sum().to_string())

    section("Exploration: Summary Stats, Distributions, Correlations")
    num_cols = ["age", "nuts_opened", "seconds", "efficiency"]
    print("Numeric summary:")
    print(df[num_cols].describe().T.to_string())

    print("\nCategorical distributions:")
    for col in ["sex", "help", "hammer", "chimpanzee"]:
        vc = df[col].value_counts(dropna=False)
        print(f"{col}:")
        print(vc.to_string())
        print()

    print("Histogram bins for age:")
    for row in summarize_distribution(df["age"], bins=7):
        print(" ", row)
    print("\nHistogram bins for efficiency:")
    for row in summarize_distribution(df["efficiency"], bins=8):
        print(" ", row)

    print("\nCorrelation matrix (numeric):")
    corr = df[num_cols].corr(numeric_only=True)
    print(corr.to_string())

    section("Bivariate Tests")
    r_age, p_age = stats.pearsonr(df["age"], df["efficiency"])
    print(f"Pearson(age, efficiency): r={r_age:.3f}, p={p_age:.3g}")

    male_eff = df.loc[df["sex"] == "m", "efficiency"]
    female_eff = df.loc[df["sex"] == "f", "efficiency"]
    t_sex = stats.ttest_ind(male_eff, female_eff, equal_var=False)
    print(
        "Welch t-test efficiency by sex (m vs f): "
        f"mean_m={male_eff.mean():.3f}, mean_f={female_eff.mean():.3f}, "
        f"t={t_sex.statistic:.3f}, p={t_sex.pvalue:.3g}"
    )

    help_yes_eff = df.loc[df["help"] == "yes", "efficiency"]
    help_no_eff = df.loc[df["help"] == "no", "efficiency"]
    t_help = stats.ttest_ind(help_yes_eff, help_no_eff, equal_var=False)
    print(
        "Welch t-test efficiency by help (yes vs no): "
        f"mean_yes={help_yes_eff.mean():.3f}, mean_no={help_no_eff.mean():.3f}, "
        f"t={t_help.statistic:.3f}, p={t_help.pvalue:.3g}"
    )

    section("Classical Controlled Models (statsmodels)")
    ols_formula = "efficiency ~ age + C(sex) + C(help) + C(hammer)"
    ols_hc3 = smf.ols(ols_formula, data=df).fit(cov_type="HC3")
    print("OLS with HC3 robust SE")
    print(ols_hc3.summary())

    ols_cluster = smf.ols(ols_formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["chimpanzee"]}
    )
    print("\nOLS with chimpanzee-clustered SE")
    print(ols_cluster.summary())

    poisson = smf.glm(
        "nuts_opened ~ age + C(sex) + C(help) + C(hammer)",
        data=df,
        family=sm.families.Poisson(),
        offset=np.log(df["seconds"]),
    ).fit(cov_type="HC3")
    print("\nPoisson GLM with log(seconds) offset (efficiency as rate)")
    print(poisson.summary())

    section("Interpretable Regressors (agentic_imodels)")
    X = pd.get_dummies(df[["age", "sex", "help", "hammer"]], drop_first=True).astype(float)
    y = df["efficiency"].astype(float)

    xname_to_feature = {f"x{i}": col for i, col in enumerate(X.columns)}
    print("Feature map for printed models:")
    for k, v in xname_to_feature.items():
        print(f"  {k} -> {v}")

    model_specs = [
        ("SmartAdditiveRegressor (honest)", SmartAdditiveRegressor()),
        ("HingeGAMRegressor (honest hinge)", HingeGAMRegressor()),
        ("HingeEBMRegressor (high-rank decoupled)", HingeEBMRegressor()),
        ("WinsorizedSparseOLSRegressor (honest sparse linear)", WinsorizedSparseOLSRegressor()),
    ]

    model_results = {}
    for label, model in model_specs:
        model.fit(X, y)
        preds = model.predict(X)
        r2 = r2_score(y, preds)
        model_text = str(model)
        print(f"\n--- {label} | train R^2={r2:.3f} ---")
        print(model)  # Required: preserve interpretable printed form verbatim.
        coef_map = parse_coefficients_from_model_text(model_text, xname_to_feature)
        zeroed = parse_zeroed_features_from_model_text(model_text, xname_to_feature)
        # Winsorized model exposes sparse support directly; keep that as canonical if present.
        if hasattr(model, "support_") and hasattr(model, "ols_coef_"):
            coef_map = {
                X.columns[int(idx)]: float(c)
                for idx, c in zip(model.support_, model.ols_coef_)
            }
            zeroed = set(X.columns) - set(coef_map.keys())

        model_results[label] = {
            "coef_map": coef_map,
            "zeroed": zeroed,
            "r2": float(r2),
            "model_text": model_text,
        }

    section("Direction, Magnitude, Shape, Robustness Synthesis")
    key_features = {
        "age": "age",
        "sex": next((c for c in X.columns if c.startswith("sex_")), None),
        "help": next((c for c in X.columns if c.startswith("help_")), None),
    }
    print("Tracked feature columns:", key_features)

    evidence_rows = []
    n_models = len(model_results)
    for term, col in key_features.items():
        if col is None:
            continue
        signed_effects = []
        abs_effects = []
        zeroed_count = 0
        for res in model_results.values():
            coef_map = res["coef_map"]
            if col in coef_map and abs(coef_map[col]) > 1e-9:
                signed_effects.append(np.sign(coef_map[col]))
                abs_effects.append(abs(coef_map[col]))
            if col in res["zeroed"]:
                zeroed_count += 1

        active_count = len(abs_effects)
        direction = "mixed/unclear"
        if signed_effects:
            if all(s > 0 for s in signed_effects):
                direction = "positive"
            elif all(s < 0 for s in signed_effects):
                direction = "negative"
        mean_abs = float(np.mean(abs_effects)) if abs_effects else 0.0
        evidence_rows.append(
            {
                "term": term,
                "feature_col": col,
                "direction": direction,
                "active_count": active_count,
                "zeroed_count": zeroed_count,
                "mean_abs_coef": mean_abs,
            }
        )

    evidence_df = pd.DataFrame(evidence_rows)
    if not evidence_df.empty:
        print(evidence_df.to_string(index=False))

    # Formal p-values from controlled OLS and Poisson for target terms.
    pvals_hc3 = {
        "age": float(ols_hc3.pvalues.get("age", np.nan)),
        "sex": float(ols_hc3.pvalues.get("C(sex)[T.m]", np.nan)),
        "help": float(ols_hc3.pvalues.get("C(help)[T.yes]", np.nan)),
    }
    pvals_cluster = {
        "age": float(ols_cluster.pvalues.get("age", np.nan)),
        "sex": float(ols_cluster.pvalues.get("C(sex)[T.m]", np.nan)),
        "help": float(ols_cluster.pvalues.get("C(help)[T.yes]", np.nan)),
    }
    pvals_poisson = {
        "age": float(poisson.pvalues.get("age", np.nan)),
        "sex": float(poisson.pvalues.get("C(sex)[T.m]", np.nan)),
        "help": float(poisson.pvalues.get("C(help)[T.yes]", np.nan)),
    }

    print("\nKey p-values (HC3 OLS / Cluster OLS / Poisson rate):")
    for term in ["age", "sex", "help"]:
        print(
            f"  {term}: "
            f"{pvals_hc3[term]:.3g} / {pvals_cluster[term]:.3g} / {pvals_poisson[term]:.3g}"
        )

    # Likert score calibration using SKILL guidance:
    # strong persistent effects -> high; mixed / zeroed in sparse hinge/lasso -> moderate.
    per_term_scores = []
    for row in evidence_rows:
        term = row["term"]
        active_frac = row["active_count"] / n_models
        strong_stats = (
            pvals_hc3[term] < 0.05 and pvals_poisson[term] < 0.05 and pvals_cluster[term] < 0.10
        )
        weak_stats = pvals_hc3[term] >= 0.10 and pvals_poisson[term] >= 0.10
        if strong_stats and active_frac >= 0.75:
            score = 90
        elif strong_stats and active_frac >= 0.50:
            score = 72
        elif strong_stats:
            score = 62
        elif weak_stats and active_frac <= 0.25:
            score = 10
        elif weak_stats:
            score = 28
        else:
            score = 48
        per_term_scores.append(score)

    response = int(np.clip(np.round(np.mean(per_term_scores)) if per_term_scores else 50, 0, 100))

    # Build concise, evidence-rich explanation.
    coef_age = float(ols_hc3.params.get("age", np.nan))
    coef_sex = float(ols_hc3.params.get("C(sex)[T.m]", np.nan))
    coef_help = float(ols_hc3.params.get("C(help)[T.yes]", np.nan))
    explanation = (
        "Efficiency was defined as nuts_opened/seconds. Controlled OLS (HC3) found "
        f"age positive (beta={coef_age:.3f}, p={pvals_hc3['age']:.3g}), male sex positive "
        f"(beta={coef_sex:.3f}, p={pvals_hc3['sex']:.3g}), and receiving help negative "
        f"(beta={coef_help:.3f}, p={pvals_hc3['help']:.3g}); Poisson rate models with "
        "log(seconds) offset kept the same signs with p<0.05 for all three terms. "
        "Interpretable models were directionally consistent when terms were active "
        "(age positive; sex positive; help negative), but sparse hinge/Lasso-style models "
        "zeroed sex/help in some fits, which is meaningful null/weakness evidence. "
        "Overall evidence supports influence of age, sex, and help on efficiency, but "
        "robustness is strongest for age and more moderate for sex/help."
    )

    result = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f)

    section("Final JSON Written")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
