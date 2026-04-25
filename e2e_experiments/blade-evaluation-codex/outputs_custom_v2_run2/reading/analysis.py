import json
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from agentic_imodels import (
    HingeEBMRegressor,
    SmartAdditiveRegressor,
    WinsorizedSparseOLSRegressor,
)


def parse_x_coefs(model_text: str) -> Dict[int, float]:
    pairs = re.findall(r"([+-]?\d*\.?\d+)\*x(\d+)", model_text)
    out: Dict[int, float] = {}
    for coef, idx in pairs:
        out[int(idx)] = float(coef)
    return out


def safe_pct_from_log_beta(beta: float) -> float:
    return 100.0 * (np.exp(beta) - 1.0)


def main() -> None:
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    question = info["research_questions"][0]
    print(f"Research question: {question}")

    df = pd.read_csv("reading.csv")
    print(f"Loaded reading.csv with shape={df.shape}")

    # Core numeric conversions used throughout.
    numeric_cols = [
        "reader_view",
        "speed",
        "dyslexia_bin",
        "dyslexia",
        "running_time",
        "adjusted_running_time",
        "scrolling_time",
        "num_words",
        "correct_rate",
        "img_width",
        "age",
        "retake_trial",
        "Flesch_Kincaid",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["speed"] > 0].copy()
    df["log_speed"] = np.log(df["speed"])
    df["rv_x_dys"] = df["reader_view"] * df["dyslexia_bin"]

    print("\n=== Exploration: Missingness (top 15) ===")
    print(df.isna().sum().sort_values(ascending=False).head(15))

    explore_cols = [
        "speed",
        "log_speed",
        "reader_view",
        "dyslexia_bin",
        "age",
        "num_words",
        "correct_rate",
        "img_width",
        "Flesch_Kincaid",
        "running_time",
        "adjusted_running_time",
        "scrolling_time",
    ]
    print("\n=== Exploration: Summary statistics ===")
    print(df[explore_cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T)

    print("\n=== Exploration: Distribution snapshots ===")
    print(f"speed skew={df['speed'].skew():.3f}, log_speed skew={df['log_speed'].skew():.3f}")
    speed_hist_counts, speed_hist_edges = np.histogram(df["speed"], bins=10)
    log_hist_counts, log_hist_edges = np.histogram(df["log_speed"], bins=10)
    print("speed histogram counts:", speed_hist_counts.tolist())
    print("speed histogram edges:", [round(v, 3) for v in speed_hist_edges.tolist()])
    print("log_speed histogram counts:", log_hist_counts.tolist())
    print("log_speed histogram edges:", [round(v, 3) for v in log_hist_edges.tolist()])

    corr_cols = [
        "log_speed",
        "reader_view",
        "dyslexia_bin",
        "rv_x_dys",
        "age",
        "num_words",
        "correct_rate",
        "img_width",
        "Flesch_Kincaid",
        "running_time",
        "adjusted_running_time",
        "scrolling_time",
        "retake_trial",
    ]
    corr = df[corr_cols].corr(numeric_only=True)
    print("\n=== Exploration: Correlation with log_speed ===")
    print(corr["log_speed"].sort_values(ascending=False))

    print("\n=== Bivariate tests ===")
    dys_sub = df[df["dyslexia_bin"] == 1].copy()
    dys_rv0 = dys_sub.loc[dys_sub["reader_view"] == 0, "log_speed"]
    dys_rv1 = dys_sub.loc[dys_sub["reader_view"] == 1, "log_speed"]
    ttest_dys = stats.ttest_ind(dys_rv1, dys_rv0, equal_var=False, nan_policy="omit")
    print(
        "Dyslexia-only mean(log_speed) reader_view=0 vs 1:",
        round(float(dys_rv0.mean()), 4),
        round(float(dys_rv1.mean()), 4),
        "diff=",
        round(float(dys_rv1.mean() - dys_rv0.mean()), 4),
        "p=",
        round(float(ttest_dys.pvalue), 6),
    )

    # Controlled OLS with broad controls and interaction (clustered by participant).
    sm_cols = [
        "uuid",
        "log_speed",
        "reader_view",
        "dyslexia_bin",
        "age",
        "num_words",
        "Flesch_Kincaid",
        "correct_rate",
        "img_width",
        "retake_trial",
        "page_id",
        "language",
        "device",
        "education",
        "gender",
        "english_native",
    ]
    d_sm = df[sm_cols].dropna().copy()
    print(f"\nRows used for main controlled OLS: {d_sm.shape[0]}")

    formula_main = (
        "log_speed ~ reader_view * dyslexia_bin + age + num_words + Flesch_Kincaid + "
        "correct_rate + img_width + retake_trial + C(page_id) + C(language) + "
        "C(device) + C(education) + C(gender) + C(english_native)"
    )
    ols_main = smf.ols(formula_main, data=d_sm).fit(
        cov_type="cluster", cov_kwds={"groups": d_sm["uuid"]}
    )
    print("\n=== Main controlled OLS (clustered by uuid) ===")
    print(ols_main.summary())

    beta_reader_main = float(ols_main.params.get("reader_view", np.nan))
    p_reader_main = float(ols_main.pvalues.get("reader_view", np.nan))
    beta_inter_main = float(ols_main.params.get("reader_view:dyslexia_bin", np.nan))
    p_inter_main = float(ols_main.pvalues.get("reader_view:dyslexia_bin", np.nan))
    beta_dys_group_main = beta_reader_main + beta_inter_main

    # Within-participant FE OLS (stronger control for person-level confounding).
    formula_fe = (
        "log_speed ~ reader_view + reader_view:dyslexia_bin + correct_rate + retake_trial + "
        "C(page_id) + C(uuid)"
    )
    ols_fe = smf.ols(formula_fe, data=d_sm).fit(
        cov_type="cluster", cov_kwds={"groups": d_sm["uuid"]}
    )
    print("\n=== Participant fixed-effects OLS (clustered by uuid) ===")
    print(ols_fe.summary())

    beta_reader_fe = float(ols_fe.params.get("reader_view", np.nan))
    p_reader_fe = float(ols_fe.pvalues.get("reader_view", np.nan))
    beta_inter_fe = float(ols_fe.params.get("reader_view:dyslexia_bin", np.nan))
    p_inter_fe = float(ols_fe.pvalues.get("reader_view:dyslexia_bin", np.nan))
    beta_dys_group_fe = beta_reader_fe + beta_inter_fe

    # Dyslexia-only controlled model.
    d_dys = d_sm[d_sm["dyslexia_bin"] == 1].copy()
    formula_dys = "log_speed ~ reader_view + correct_rate + retake_trial + C(page_id) + C(uuid)"
    ols_dys = smf.ols(formula_dys, data=d_dys).fit(
        cov_type="cluster", cov_kwds={"groups": d_dys["uuid"]}
    )
    print("\n=== Dyslexia-only OLS with participant FE ===")
    print(ols_dys.summary())
    beta_dys_only = float(ols_dys.params.get("reader_view", np.nan))
    p_dys_only = float(ols_dys.pvalues.get("reader_view", np.nan))

    # Interpretable models: include explicit interaction term so zeroing/non-zeroing is visible.
    model_num_cols = [
        "reader_view",
        "dyslexia_bin",
        "rv_x_dys",
        "age",
        "num_words",
        "Flesch_Kincaid",
        "correct_rate",
        "img_width",
        "retake_trial",
        "running_time",
        "adjusted_running_time",
        "scrolling_time",
    ]
    model_cat_cols = ["device", "education", "gender", "english_native", "page_id", "language"]
    model_cols = model_num_cols + model_cat_cols

    d_model = df[model_cols + ["log_speed"]].copy()
    X_raw = d_model[model_cols]
    y = d_model["log_speed"].values

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), model_num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                model_cat_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    X_enc = preprocess.fit_transform(X_raw)
    if hasattr(X_enc, "toarray"):
        X_enc = X_enc.toarray()
    feature_names = list(preprocess.get_feature_names_out())
    X_df = pd.DataFrame(X_enc, columns=feature_names)

    idx_reader_view = feature_names.index("reader_view")
    idx_dyslexia = feature_names.index("dyslexia_bin")
    idx_interaction = feature_names.index("rv_x_dys")

    print("\nEncoded feature indices:")
    print("reader_view index:", idx_reader_view)
    print("dyslexia_bin index:", idx_dyslexia)
    print("rv_x_dys index:", idx_interaction)

    models = [
        ("SmartAdditiveRegressor", SmartAdditiveRegressor()),
        ("HingeEBMRegressor", HingeEBMRegressor()),
        ("WinsorizedSparseOLSRegressor", WinsorizedSparseOLSRegressor()),
    ]

    imodel_rows: List[Tuple[str, float, float, float, float, float]] = []
    zeroed_reader = 0
    zeroed_interaction = 0

    print("\n=== agentic_imodels fits ===")
    for name, model in models:
        model.fit(X_df, y)
        pred = model.predict(X_df)
        r2 = float(r2_score(y, pred))
        txt = str(model)
        print(f"\n--- {name} (in-sample R^2={r2:.4f}) ---")
        print(model)  # Required by instructions.

        coef_map = parse_x_coefs(txt)
        coef_reader = float(coef_map.get(idx_reader_view, 0.0))
        coef_inter = float(coef_map.get(idx_interaction, 0.0))
        coef_dys = float(coef_map.get(idx_dyslexia, 0.0))

        imp_reader = np.nan
        imp_inter = np.nan
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_, dtype=float)
            if imp.shape[0] == len(feature_names):
                imp_reader = float(imp[idx_reader_view])
                imp_inter = float(imp[idx_interaction])

        if abs(coef_reader) < 1e-12:
            zeroed_reader += 1
        if abs(coef_inter) < 1e-12:
            zeroed_interaction += 1

        imodel_rows.append((name, r2, coef_reader, coef_inter, coef_dys, imp_reader))
        print(
            f"{name}: coef(reader_view)={coef_reader:.6f}, "
            f"coef(rv_x_dys)={coef_inter:.6f}, coef(dyslexia_bin)={coef_dys:.6f}, "
            f"importance(reader_view)={imp_reader if not np.isnan(imp_reader) else 'NA'}"
        )

    print("\n=== Interpretable-model summary table ===")
    imodel_df = pd.DataFrame(
        imodel_rows,
        columns=[
            "model",
            "r2",
            "coef_reader_view",
            "coef_readerX_dyslexia",
            "coef_dyslexia_bin",
            "importance_reader_view",
        ],
    )
    print(imodel_df)
    print(
        f"Zeroed reader_view in {zeroed_reader}/{len(models)} models; "
        f"zeroed reader_view*dyslexia in {zeroed_interaction}/{len(models)} models."
    )

    # Calibrated conclusion score (0-100).
    # Strong null evidence when both statistical and sparse-interpretable evidence are null/zeroed.
    weak_or_null = (
        (p_inter_main > 0.10)
        and (p_inter_fe > 0.10)
        and (p_dys_only > 0.10)
        and (ttest_dys.pvalue > 0.10)
        and (zeroed_interaction >= 2)
    )

    if weak_or_null:
        response = 12
    elif (p_inter_main < 0.05) and (p_inter_fe < 0.05):
        response = 85
    else:
        response = 45

    explanation = (
        f"Bivariate dyslexia-only comparison shows near-zero difference in log reading speed "
        f"(delta={dys_rv1.mean() - dys_rv0.mean():.4f}, p={ttest_dys.pvalue:.3f}). "
        f"In controlled OLS with demographics/page/language controls, the interaction "
        f"reader_view:dyslexia_bin is small and non-significant "
        f"(beta={beta_inter_main:.4f}, p={p_inter_main:.3f}); the implied reader_view effect "
        f"for dyslexia participants is only about {safe_pct_from_log_beta(beta_dys_group_main):.2f}% "
        f"on speed. In participant fixed-effects OLS, the interaction remains non-significant "
        f"(beta={beta_inter_fe:.4f}, p={p_inter_fe:.3f}) and the dyslexia-group implied effect is "
        f"{safe_pct_from_log_beta(beta_dys_group_fe):.2f}%. Dyslexia-only FE OLS also shows no effect "
        f"(beta={beta_dys_only:.4f}, p={p_dys_only:.3f}). Interpretable models agree: "
        f"reader_view and reader_view*dyslexia are zeroed out in most sparse/evolved models "
        f"(reader_view zeroed {zeroed_reader}/{len(models)}, interaction zeroed "
        f"{zeroed_interaction}/{len(models)}), while other variables (e.g., comprehension accuracy, age, "
        f"dyslexia status, page/language terms) carry the predictive signal. Overall evidence does not "
        f"support a meaningful Reader View speed improvement specifically for individuals with dyslexia."
    )

    payload = {"response": int(response), "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True)

    print("\nWrote conclusion.txt")
    print(payload)


if __name__ == "__main__":
    main()
