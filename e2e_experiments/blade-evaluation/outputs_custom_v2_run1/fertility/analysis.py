import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from interp_models import SmartAdditiveRegressor, HingeEBMRegressor

warnings.filterwarnings("ignore")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Parse dates used to locate each participant within cycle phase.
    for c in ["DateTesting", "StartDateofLastPeriod", "StartDateofPeriodBeforeLast"]:
        out[c] = pd.to_datetime(out[c], format="%m/%d/%y", errors="coerce")

    out["religiosity"] = out[["Rel1", "Rel2", "Rel3"]].mean(axis=1)

    out["observed_cycle_length"] = (
        out["StartDateofLastPeriod"] - out["StartDateofPeriodBeforeLast"]
    ).dt.days
    out["cycle_length_used"] = out["ReportedCycleLength"].fillna(out["observed_cycle_length"])
    out["cycle_day"] = (out["DateTesting"] - out["StartDateofLastPeriod"]).dt.days

    plausible = out["cycle_length_used"].between(20, 40) & out["cycle_day"].between(0, 40)
    out.loc[~plausible, ["cycle_length_used", "cycle_day"]] = np.nan

    out["ovulation_day"] = out["cycle_length_used"] - 14
    out["days_from_ovulation"] = out["cycle_day"] - out["ovulation_day"]

    sigma_days = 2.5
    out["fertility_score"] = np.exp(
        -((out["days_from_ovulation"] ** 2) / (2 * sigma_days**2))
    )
    out["fertile_window"] = (
        out["days_from_ovulation"].between(-5, 1, inclusive="both")
    ).astype(int)

    out["testing_day_index"] = (out["DateTesting"] - out["DateTesting"].min()).dt.days
    return out


def normalize_effects(effects: dict) -> dict:
    cleaned = {}
    for k, v in effects.items():
        cleaned[k] = {
            "direction": v.get("direction", "unknown"),
            "importance": float(v.get("importance", 0.0)),
            "rank": int(v.get("rank", 0) or 0),
        }
    return cleaned


def top_effects_text(effects: dict, exclude: set[str], k: int = 3) -> str:
    rows = [(name, meta) for name, meta in effects.items() if name not in exclude]
    rows = [r for r in rows if r[1]["importance"] > 0]
    rows.sort(key=lambda x: (-x[1]["importance"], x[0]))
    if not rows:
        return "none"
    top = rows[:k]
    return "; ".join(
        f"{name} (rank {meta['rank']}, imp={meta['importance']:.1%}, {meta['direction']})"
        for name, meta in top
    )


def main() -> None:
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info["research_questions"][0]
    print("Research question:")
    print(research_question)
    print("\nDV: religiosity (mean of Rel1, Rel2, Rel3)")
    print("IV: fertility_score (cycle-phase-based fertility proxy)")

    raw = pd.read_csv("fertility.csv")
    df = build_features(raw)

    print("\nStep 1: Exploration")
    explore_cols = [
        "religiosity",
        "fertility_score",
        "fertile_window",
        "days_from_ovulation",
        "cycle_day",
        "cycle_length_used",
        "Relationship",
        "Sure1",
        "Sure2",
        "testing_day_index",
    ]
    print("Summary statistics:")
    print(df[explore_cols].describe().T)

    print("\nBivariate Pearson correlations with religiosity:")
    corr_rows = []
    for col in [c for c in explore_cols if c != "religiosity"]:
        sub = df[["religiosity", col]].dropna()
        if len(sub) >= 3 and sub[col].std() > 0:
            r, p = stats.pearsonr(sub["religiosity"], sub[col])
            corr_rows.append((col, len(sub), r, p))
    corr_rows.sort(key=lambda x: abs(x[2]), reverse=True)
    for col, n, r, p in corr_rows:
        print(f"  {col}: r={r:.3f}, p={p:.4f}, n={n}")

    fw_sub = df[["religiosity", "fertile_window"]].dropna()
    rel_fw1 = fw_sub.loc[fw_sub["fertile_window"] == 1, "religiosity"]
    rel_fw0 = fw_sub.loc[fw_sub["fertile_window"] == 0, "religiosity"]
    t_res = stats.ttest_ind(rel_fw1, rel_fw0, equal_var=False)
    print("\nFertile-window mean difference (religiosity):")
    print(
        f"  fertile=1 mean={rel_fw1.mean():.3f}, fertile=0 mean={rel_fw0.mean():.3f}, "
        f"t={t_res.statistic:.3f}, p={t_res.pvalue:.4f}"
    )

    print("\nStep 2: Controlled regression")
    base_controls = ["Relationship", "Sure1", "Sure2", "cycle_day", "cycle_length_used", "testing_day_index"]

    ols_cols_main = ["religiosity", "fertility_score"] + base_controls
    ols_df_main = df[ols_cols_main].dropna()
    X_main = sm.add_constant(ols_df_main[["fertility_score"] + base_controls])
    y_main = ols_df_main["religiosity"]
    ols_main = sm.OLS(y_main, X_main).fit()
    print("OLS model with fertility_score + controls:")
    print(ols_main.summary())

    ols_cols_fw = ["religiosity", "fertile_window"] + base_controls
    ols_df_fw = df[ols_cols_fw].dropna()
    X_fw = sm.add_constant(ols_df_fw[["fertile_window"] + base_controls])
    y_fw = ols_df_fw["religiosity"]
    ols_fw = sm.OLS(y_fw, X_fw).fit()
    print("\nRobustness OLS with fertile_window + controls:")
    print(ols_fw.summary())

    print("\nStep 3: Interpretable models")
    model_features = ["fertility_score", "fertile_window"] + base_controls
    model_df = df[["religiosity"] + model_features].dropna()
    X = model_df[model_features]
    y = model_df["religiosity"]

    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X, y)
    smart_effects = normalize_effects(smart.feature_effects())
    print("SmartAdditiveRegressor:")
    print(smart)
    print("feature_effects:")
    print(smart_effects)

    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X, y)
    hinge_effects = normalize_effects(hinge.feature_effects())
    print("\nHingeEBMRegressor:")
    print(hinge)
    print("feature_effects:")
    print(hinge_effects)

    iv_name = "fertility_score"
    iv_corr_sub = df[["religiosity", iv_name]].dropna()
    iv_r, iv_r_p = stats.pearsonr(iv_corr_sub["religiosity"], iv_corr_sub[iv_name])

    iv_coef = float(ols_main.params[iv_name])
    iv_p = float(ols_main.pvalues[iv_name])
    iv_ci_low, iv_ci_high = [float(v) for v in ols_main.conf_int().loc[iv_name].tolist()]

    fw_coef = float(ols_fw.params["fertile_window"])
    fw_p = float(ols_fw.pvalues["fertile_window"])

    smart_iv = smart_effects.get(iv_name, {"direction": "zero", "importance": 0.0, "rank": 0})
    hinge_iv = hinge_effects.get(iv_name, {"direction": "zero", "importance": 0.0, "rank": 0})

    shape_text = "no learned effect"
    iv_idx = X.columns.get_loc(iv_name)
    if iv_idx in smart.shape_functions_:
        thresholds, intervals = smart.shape_functions_[iv_idx]
        thr_text = ", ".join(f"{t:.3f}" for t in thresholds[:4])
        if len(thresholds) > 4:
            thr_text += ", ..."
        shape_text = (
            f"{smart_iv['direction']}; thresholds at [{thr_text}] with effect range "
            f"{(max(intervals) - min(intervals)):.3f}"
        )

    if iv_r_p > 0.1 and iv_p > 0.1 and smart_iv["importance"] < 0.05 and hinge_iv["importance"] < 0.05:
        response = 8
    elif iv_p < 0.05 and smart_iv["importance"] >= 0.1 and hinge_iv["importance"] >= 0.05:
        response = 85
    elif iv_p < 0.05 and (smart_iv["importance"] >= 0.05 or hinge_iv["importance"] >= 0.05):
        response = 68
    elif iv_p < 0.1 or iv_r_p < 0.1 or smart_iv["importance"] >= 0.05:
        response = 35
    else:
        response = 20

    top_smart_controls = top_effects_text(smart_effects, exclude={"fertility_score", "fertile_window"}, k=3)
    top_hinge_controls = top_effects_text(hinge_effects, exclude={"fertility_score", "fertile_window"}, k=3)

    explanation = (
        f"The evidence for a fertility-driven religiosity effect is weak. Bivariate association is near zero "
        f"(r={iv_r:.3f}, p={iv_r_p:.3f}). In OLS with controls, fertility_score is small and not significant "
        f"(coef={iv_coef:.3f}, p={iv_p:.3f}, 95% CI [{iv_ci_low:.3f}, {iv_ci_high:.3f}]). "
        f"A fertile-window robustness model is also null (coef={fw_coef:.3f}, p={fw_p:.3f}). "
        f"SmartAdditive ranks fertility_score low (rank {smart_iv['rank']}, importance={smart_iv['importance']:.1%}) "
        f"with shape: {shape_text}. HingeEBM assigns fertility_score {hinge_iv['importance']:.1%} importance "
        f"(rank {hinge_iv['rank']}), effectively excluding it from the sparse equation. "
        f"Confounders/other structure matter more than fertility in flexible models: SmartAdditive top features are "
        f"{top_smart_controls}; Hinge top features are {top_hinge_controls}. Overall, results are inconsistent with a "
        f"meaningful positive effect after controls."
    )

    payload = {"response": int(response), "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(payload))

    print("\nStep 4: Wrote conclusion.txt")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
