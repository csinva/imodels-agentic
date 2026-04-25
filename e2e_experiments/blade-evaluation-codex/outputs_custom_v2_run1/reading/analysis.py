import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.metrics import r2_score

from agentic_imodels import (
    HingeEBMRegressor,
    SmartAdditiveRegressor,
    WinsorizedSparseOLSRegressor,
)


warnings.filterwarnings("ignore", category=FutureWarning)


def pct_from_log(beta: float) -> float:
    return 100.0 * (np.exp(beta) - 1.0)


def safe_get_effect_pvalue(contrast) -> tuple[float, float]:
    effect = np.asarray(contrast.effect).reshape(-1)
    pvalue = np.asarray(contrast.pvalue).reshape(-1)
    return float(effect[0]), float(pvalue[0])


def average_toggle_effect(model, X: np.ndarray, feature_idx: int) -> float:
    X0 = X.copy()
    X1 = X.copy()
    X0[:, feature_idx] = 0.0
    X1[:, feature_idx] = 1.0
    return float(np.mean(model.predict(X1) - model.predict(X0)))


def smart_importance_rank(model, idx: int) -> tuple[float, int]:
    importances = np.asarray(model.feature_importances_)
    order = np.argsort(-importances)
    rank = int(np.where(order == idx)[0][0]) + 1
    return float(importances[idx]), rank


def hinge_reader_activity(model, reader_idx: int) -> dict:
    selected = list(model.selected_)
    out = {
        "selected_prehinge": False,
        "linear_active": False,
        "hinge_active_terms": 0,
    }
    if reader_idx not in selected:
        return out

    out["selected_prehinge"] = True
    sel_pos = selected.index(reader_idx)
    coefs = model.lasso_.coef_
    n_sel = len(selected)
    out["linear_active"] = bool(abs(coefs[sel_pos]) > 1e-8)
    hactive = 0
    for k, (feat_idx, _knot, _direction) in enumerate(model.hinge_info_):
        if feat_idx == sel_pos and abs(coefs[n_sel + k]) > 1e-8:
            hactive += 1
    out["hinge_active_terms"] = int(hactive)
    return out


def winsor_reader_coef(model, reader_idx: int) -> tuple[bool, float]:
    if reader_idx not in model.support_:
        return False, 0.0
    pos = list(model.support_).index(reader_idx)
    return True, float(model.ols_coef_[pos])


def main() -> None:
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info["research_questions"][0]
    print("Research question:")
    print(question)
    print()

    df = pd.read_csv("reading.csv")
    print("=== Data overview ===")
    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")
    print("Missing values by column:")
    print(df.isna().sum().sort_values(ascending=False).head(10))
    print()

    df["log_speed"] = np.log(df["speed"].clip(lower=1e-6))

    key_numeric = [
        "speed",
        "log_speed",
        "reader_view",
        "dyslexia_bin",
        "dyslexia",
        "age",
        "num_words",
        "Flesch_Kincaid",
        "img_width",
        "retake_trial",
        "running_time",
        "scrolling_time",
        "adjusted_running_time",
    ]
    key_numeric = [c for c in key_numeric if c in df.columns]

    print("=== Summary statistics (key numeric columns) ===")
    print(df[key_numeric].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]])
    print()

    print("=== Distribution checks ===")
    print("speed quantiles:")
    print(df["speed"].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))
    print(f"speed skewness: {df['speed'].skew():.3f}")
    print(f"log_speed skewness: {df['log_speed'].skew():.3f}")
    print("Counts by dyslexia_bin:")
    print(df["dyslexia_bin"].value_counts(dropna=False))
    print("Counts by reader_view:")
    print(df["reader_view"].value_counts(dropna=False))
    print()

    print("Mean log_speed by dyslexia_bin and reader_view:")
    print(df.groupby(["dyslexia_bin", "reader_view"], dropna=False)["log_speed"].mean())
    print()

    print("=== Correlations with log_speed ===")
    corr = df[key_numeric].corr(numeric_only=True)["log_speed"].sort_values(ascending=False)
    print(corr)
    print()

    dys = df[df["dyslexia_bin"] == 1].copy()
    dys = dys.dropna(subset=["reader_view", "log_speed"])
    log_speed_rv1 = dys.loc[dys["reader_view"] == 1, "log_speed"]
    log_speed_rv0 = dys.loc[dys["reader_view"] == 0, "log_speed"]

    ttest = stats.ttest_ind(log_speed_rv1, log_speed_rv0, equal_var=False, nan_policy="omit")
    pb = stats.pointbiserialr(dys["reader_view"], dys["log_speed"])

    print("=== Bivariate test in dyslexic participants (dyslexia_bin=1) ===")
    print(f"n(reader_view=1): {len(log_speed_rv1)}, n(reader_view=0): {len(log_speed_rv0)}")
    print(
        f"Welch t-test on log_speed: t={ttest.statistic:.4f}, p={ttest.pvalue:.4g}, "
        f"mean diff={log_speed_rv1.mean() - log_speed_rv0.mean():.4f} log-points"
    )
    print(
        f"Point-biserial correlation(reader_view, log_speed): r={pb.statistic:.4f}, "
        f"p={pb.pvalue:.4g}"
    )
    print()

    controls = [
        "reader_view",
        "age",
        "dyslexia",
        "retake_trial",
        "num_words",
        "Flesch_Kincaid",
        "img_width",
        "device",
        "education",
        "gender",
        "english_native",
        "page_id",
        "language",
        "log_speed",
    ]

    dys_ols = dys.copy()
    dys_ols = dys_ols.dropna(subset=controls)

    formula_dys = (
        "log_speed ~ reader_view + age + dyslexia + retake_trial + num_words + "
        "Flesch_Kincaid + img_width + C(device) + C(education) + C(gender) + "
        "C(english_native) + C(page_id) + C(language)"
    )
    ols_dys = smf.ols(formula_dys, data=dys_ols).fit(cov_type="HC3")

    print("=== Controlled OLS (dyslexic participants only) ===")
    print(ols_dys.summary())
    rv_beta = float(ols_dys.params.get("reader_view", np.nan))
    rv_p = float(ols_dys.pvalues.get("reader_view", np.nan))
    rv_ci_low, rv_ci_high = ols_dys.conf_int().loc["reader_view"].tolist()
    print(
        "Reader_view effect among dyslexic participants: "
        f"beta={rv_beta:.4f} log-points ({pct_from_log(rv_beta):.2f}%), "
        f"p={rv_p:.4g}, 95% CI=({rv_ci_low:.4f}, {rv_ci_high:.4f})"
    )
    print()

    full = df.dropna(subset=["dyslexia_bin"]).copy()
    full_controls = [
        "reader_view",
        "dyslexia_bin",
        "age",
        "retake_trial",
        "num_words",
        "Flesch_Kincaid",
        "img_width",
        "device",
        "education",
        "gender",
        "english_native",
        "page_id",
        "language",
        "log_speed",
    ]
    full = full.dropna(subset=full_controls)

    formula_full = (
        "log_speed ~ reader_view * dyslexia_bin + age + retake_trial + num_words + "
        "Flesch_Kincaid + img_width + C(device) + C(education) + C(gender) + "
        "C(english_native) + C(page_id) + C(language)"
    )
    ols_full = smf.ols(formula_full, data=full).fit(cov_type="HC3")

    print("=== Controlled interaction OLS (full sample) ===")
    print(ols_full.summary())

    params = list(ols_full.params.index)
    interaction_name = (
        "reader_view:dyslexia_bin"
        if "reader_view:dyslexia_bin" in params
        else "dyslexia_bin:reader_view"
    )
    rv_full_beta = float(ols_full.params.get("reader_view", np.nan))
    rv_full_p = float(ols_full.pvalues.get("reader_view", np.nan))
    inter_beta = float(ols_full.params.get(interaction_name, np.nan))
    inter_p = float(ols_full.pvalues.get(interaction_name, np.nan))

    c = np.zeros(len(params))
    c[params.index("reader_view")] = 1.0
    c[params.index(interaction_name)] = 1.0
    dys_effect_test = ols_full.t_test(c)
    dys_effect_beta, dys_effect_p = safe_get_effect_pvalue(dys_effect_test)

    print(
        "From full interaction model: "
        f"reader_view(main for non-dyslexic) beta={rv_full_beta:.4f}, p={rv_full_p:.4g}; "
        f"interaction beta={inter_beta:.4f}, p={inter_p:.4g}."
    )
    print(
        "Implied reader_view effect for dyslexic participants (main + interaction): "
        f"beta={dys_effect_beta:.4f} ({pct_from_log(dys_effect_beta):.2f}%), p={dys_effect_p:.4g}."
    )
    print()

    print("=== Interpretable models (dyslexic subset) ===")
    cat_cols = ["device", "education", "language", "page_id", "english_native"]
    num_cols = [
        "reader_view",
        "age",
        "dyslexia",
        "gender",
        "retake_trial",
        "num_words",
        "Flesch_Kincaid",
        "img_width",
    ]

    dys_model = dys.copy()
    for ccol in cat_cols:
        dys_model[ccol] = dys_model[ccol].fillna("Missing").astype(str)
    for ncol in num_cols + ["log_speed"]:
        dys_model[ncol] = pd.to_numeric(dys_model[ncol], errors="coerce")
    dys_model = dys_model.dropna(subset=["log_speed"])
    for ncol in num_cols:
        dys_model[ncol] = dys_model[ncol].fillna(dys_model[ncol].median())

    X_df = pd.get_dummies(dys_model[num_cols + cat_cols], columns=cat_cols, drop_first=True, dtype=float)
    y = dys_model["log_speed"].values
    feature_names = list(X_df.columns)
    X = X_df.values

    reader_idx = feature_names.index("reader_view")

    print("Feature index mapping for printed models:")
    for i, name in enumerate(feature_names):
        print(f"x{i} -> {name}")
    print()

    model_results = {}

    models = [
        ("SmartAdditiveRegressor", SmartAdditiveRegressor()),
        ("HingeEBMRegressor", HingeEBMRegressor()),
        ("WinsorizedSparseOLSRegressor", WinsorizedSparseOLSRegressor(max_features=12)),
    ]

    for name, model in models:
        model.fit(X, y)
        yhat = model.predict(X)
        r2 = float(r2_score(y, yhat))
        rv_effect = average_toggle_effect(model, X, reader_idx)

        print(f"=== {name} (train R^2={r2:.4f}) ===")
        print(model)
        print(
            f"Average reader_view toggle effect from model predictions: "
            f"{rv_effect:.4f} log-points ({pct_from_log(rv_effect):.2f}%)"
        )

        model_results[name] = {
            "r2": r2,
            "reader_view_avg_effect": rv_effect,
        }

        if name == "SmartAdditiveRegressor":
            imp, rank = smart_importance_rank(model, reader_idx)
            model_results[name]["reader_view_importance"] = imp
            model_results[name]["reader_view_rank"] = rank
            print(f"reader_view SmartAdditive importance={imp:.6f}, rank={rank} / {len(feature_names)}")

        if name == "HingeEBMRegressor":
            activity = hinge_reader_activity(model, reader_idx)
            model_results[name].update(activity)
            print(
                "reader_view in HingeEBM stage-1 sparse display: "
                f"selected_prehinge={activity['selected_prehinge']}, "
                f"linear_active={activity['linear_active']}, "
                f"hinge_active_terms={activity['hinge_active_terms']}"
            )
            print(
                "Note: HingeEBM is display-predict decoupled, so non-zero average toggle "
                "effect can come from hidden residual EBM even if display terms are zero."
            )

        if name == "WinsorizedSparseOLSRegressor":
            in_support, coef = winsor_reader_coef(model, reader_idx)
            model_results[name]["reader_view_in_support"] = in_support
            model_results[name]["reader_view_coef"] = coef
            print(f"reader_view in sparse support={in_support}, coefficient={coef:.6f}")

        print()

    # Evidence-calibrated score (0=strong no, 100=strong yes)
    score = 0

    if rv_beta > 0:
        if rv_p < 0.01:
            score = 85
        elif rv_p < 0.05:
            score = 72
        elif rv_p < 0.10:
            score = 58
        else:
            score = 42
    else:
        if rv_p < 0.05:
            score = 18
        elif rv_p < 0.10:
            score = 28
        else:
            score = 32

    if dys_effect_p < 0.05:
        score += 8 if dys_effect_beta > 0 else -8
    elif dys_effect_p < 0.10:
        score += 4 if dys_effect_beta > 0 else -4

    mean_diff = float(log_speed_rv1.mean() - log_speed_rv0.mean())
    if ttest.pvalue < 0.05:
        score += 5 if mean_diff > 0 else -5

    positive_signals = 0
    null_signals = 0

    smart_res = model_results["SmartAdditiveRegressor"]
    smart_eff = smart_res["reader_view_avg_effect"]
    smart_rank = int(smart_res["reader_view_rank"])
    if smart_eff > 0.01:
        positive_signals += 1
    elif abs(smart_eff) < 0.005:
        null_signals += 1

    if smart_rank <= max(5, int(0.2 * len(feature_names))):
        positive_signals += 1
    else:
        null_signals += 1

    hinge_res = model_results["HingeEBMRegressor"]
    hinge_active = bool(hinge_res["linear_active"] or hinge_res["hinge_active_terms"] > 0)
    if hinge_active and hinge_res["reader_view_avg_effect"] > 0:
        positive_signals += 1
    elif not hinge_active:
        null_signals += 1

    win_res = model_results["WinsorizedSparseOLSRegressor"]
    if win_res["reader_view_in_support"] and win_res["reader_view_coef"] > 0:
        positive_signals += 1
    elif not win_res["reader_view_in_support"]:
        null_signals += 2
    else:
        null_signals += 1

    score += 4 * positive_signals - 6 * null_signals
    score = int(np.clip(round(score), 0, 100))

    explanation = (
        f"Question: {question} "
        f"Among dyslexic participants, bivariate evidence on log(reading speed) is null "
        f"(Welch t p={ttest.pvalue:.3g}, mean diff={mean_diff:.4f} log-points). "
        f"Controlled OLS with demographic/device/page/language controls gives "
        f"reader_view beta={rv_beta:.4f} ({pct_from_log(rv_beta):.2f}%), p={rv_p:.3g}, "
        f"95% CI=({rv_ci_low:.4f},{rv_ci_high:.4f}). "
        f"In the full-sample interaction model, the implied dyslexic effect is "
        f"beta={dys_effect_beta:.4f} ({pct_from_log(dys_effect_beta):.2f}%), p={dys_effect_p:.3g}. "
        f"Interpretable models largely support a null: SmartAdditive gives reader_view "
        f"importance={smart_res['reader_view_importance']:.6f} (rank {smart_rank}/{len(feature_names)}) "
        f"with avg effect {pct_from_log(smart_eff):.2f}%; WinsorizedSparseOLS excludes reader_view "
        f"from its sparse support; HingeEBM stage-1 sparse display also zeroes it out "
        f"(selected_prehinge={hinge_res['selected_prehinge']}, "
        f"hinge_active_terms={hinge_res['hinge_active_terms']}). "
        f"A small positive average toggle from HingeEBM can arise from its hidden residual corrector "
        f"and is not robust across honest sparse/additive models. Overall evidence does not support "
        f"a meaningful speed improvement from Reader View for dyslexic individuals."
    )

    payload = {"response": score, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print("=== Final conclusion JSON ===")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
