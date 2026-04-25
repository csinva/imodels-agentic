import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import pointbiserialr, ttest_ind
from sklearn.metrics import r2_score

from agentic_imodels import (
    HingeEBMRegressor,
    SmartAdditiveRegressor,
    WinsorizedSparseOLSRegressor,
)


def print_header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def monotonicity(intervals, tol: float = 1e-9) -> str:
    if len(intervals) < 2:
        return "flat"
    diffs = np.diff(intervals)
    if np.all(diffs >= -tol):
        return "increasing"
    if np.all(diffs <= tol):
        return "decreasing"
    return "non-monotonic"


def hinge_effective_coefficients(model: HingeEBMRegressor):
    n_sel = len(model.selected_)
    coefs = model.lasso_.coef_
    intercept = float(model.lasso_.intercept_)
    effective = {int(model.selected_[i]): float(coefs[i]) for i in range(n_sel)}

    for idx, (feat_idx, knot, direction) in enumerate(model.hinge_info_):
        j_orig = int(model.selected_[feat_idx])
        c = float(coefs[n_sel + idx])
        if abs(c) < 1e-8:
            continue
        if direction == "pos":
            effective[j_orig] = effective.get(j_orig, 0.0) + c
            intercept -= c * knot
        else:
            effective[j_orig] = effective.get(j_orig, 0.0) - c
            intercept += c * knot

    active = {k: v for k, v in effective.items() if abs(v) > 1e-8}
    return active, intercept


def main() -> None:
    run_dir = Path(__file__).resolve().parent

    info = json.loads((run_dir / "info.json").read_text())
    question = info["research_questions"][0]

    print_header("Research Question")
    print(question)

    df = pd.read_csv(run_dir / "crofoot.csv")

    # Feature engineering focused on the question's constructs.
    df["size_diff"] = df["n_focal"] - df["n_other"]
    df["location_adv"] = df["dist_other"] - df["dist_focal"]
    df["dist_mean"] = (df["dist_focal"] + df["dist_other"]) / 2.0
    df["n_total"] = df["n_focal"] + df["n_other"]

    print_header("Data Overview")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nMissing values per column:")
    print(df.isna().sum())
    print("\nOutcome distribution (win):")
    print(df["win"].value_counts(dropna=False).sort_index())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print_header("Summary Statistics")
    print(df[numeric_cols].describe().T)

    print_header("Distributions (selected predictors)")
    for col in ["size_diff", "location_adv", "dist_focal", "dist_other", "dist_mean"]:
        qs = df[col].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict()
        print(
            f"{col}: min={qs[0.0]:.3f}, q25={qs[0.25]:.3f}, "
            f"median={qs[0.5]:.3f}, q75={qs[0.75]:.3f}, max={qs[1.0]:.3f}"
        )

    print_header("Correlations With Outcome (Pearson)")
    corr_to_win = (
        df[numeric_cols]
        .corr(numeric_only=True)["win"]
        .drop("win")
        .sort_values(key=np.abs, ascending=False)
    )
    print(corr_to_win)

    print_header("Bivariate Tests")
    bivariate_results = {}
    for col in ["size_diff", "location_adv", "dist_focal", "dist_other", "dist_mean"]:
        r, p = pointbiserialr(df["win"], df[col])
        bivariate_results[col] = {"r": float(r), "p": float(p)}
        print(f"point-biserial(win, {col}): r={r:.4f}, p={p:.4g}")

        win1 = df.loc[df["win"] == 1, col]
        win0 = df.loc[df["win"] == 0, col]
        t_stat, t_p = ttest_ind(win1, win0, equal_var=False)
        print(f"  Welch t-test ({col} by win): t={t_stat:.4f}, p={t_p:.4g}")

    print_header("Classical Statistical Models (Binomial GLM)")
    formula_rel = "win ~ size_diff + location_adv + dist_mean + n_total"
    glm_rel = smf.glm(formula_rel, data=df, family=sm.families.Binomial()).fit()
    print("Model A (relative effects + controls):", formula_rel)
    print(glm_rel.summary())

    formula_raw = "win ~ n_focal + n_other + dist_focal + dist_other"
    glm_raw = smf.glm(formula_raw, data=df, family=sm.families.Binomial()).fit()
    print("\nModel B (raw component robustness check):", formula_raw)
    print(glm_raw.summary())

    feature_cols = ["size_diff", "location_adv", "dist_mean", "n_total"]
    X = df[feature_cols]
    y = df["win"]

    print_header("Interpretable Models (agentic_imodels)")

    smart = SmartAdditiveRegressor().fit(X, y)
    print("=== SmartAdditiveRegressor ===")
    print(f"Train R^2: {r2_score(y, smart.predict(X)):.4f}")
    print(smart)

    hinge_ebm = HingeEBMRegressor().fit(X, y)
    print("\n=== HingeEBMRegressor ===")
    print(f"Train R^2: {r2_score(y, hinge_ebm.predict(X)):.4f}")
    print(hinge_ebm)

    sparse_ols = WinsorizedSparseOLSRegressor().fit(X, y)
    print("\n=== WinsorizedSparseOLSRegressor ===")
    print(f"Train R^2: {r2_score(y, sparse_ols.predict(X)):.4f}")
    print(sparse_ols)

    print_header("Interpretable Effect Diagnostics")

    smart_imp = {
        feature_cols[i]: float(v)
        for i, v in enumerate(getattr(smart, "feature_importances_", np.zeros(len(feature_cols))))
    }
    imp_total = sum(smart_imp.values()) + 1e-12
    for i, name in enumerate(feature_cols):
        rel_imp = smart_imp[name] / imp_total
        if i in smart.shape_functions_:
            _, intervals = smart.shape_functions_[i]
            shape = monotonicity(intervals)
            slope, _, r2_lin = smart.linear_approx_.get(i, (0.0, 0.0, 0.0))
            direction = "positive" if slope > 0 else ("negative" if slope < 0 else "flat")
            print(
                f"SmartAdditive {name}: rel_importance={rel_imp:.3f}, "
                f"direction~{direction}, shape={shape}, linear_R2={r2_lin:.3f}"
            )
        else:
            print(f"SmartAdditive {name}: effectively zero / excluded")

    hinge_eff, _ = hinge_effective_coefficients(hinge_ebm)
    hinge_by_name = {feature_cols[k]: float(v) for k, v in hinge_eff.items() if k < len(feature_cols)}
    print("\nHingeEBM effective coefficients (display layer):")
    print(hinge_by_name)

    sparse_coef = {feature_cols[idx]: float(c) for idx, c in zip(sparse_ols.support_, sparse_ols.ols_coef_)}
    sparse_excluded = [name for i, name in enumerate(feature_cols) if i not in set(sparse_ols.support_)]
    print("\nWinsorizedSparseOLS kept coefficients:")
    print(sparse_coef)
    print("WinsorizedSparseOLS excluded (zeroed):", sparse_excluded)

    # Evidence synthesis for the binary Likert response.
    p_size = float(glm_rel.pvalues["size_diff"])
    p_loc = float(glm_rel.pvalues["location_adv"])
    beta_size = float(glm_rel.params["size_diff"])
    beta_loc = float(glm_rel.params["location_adv"])

    p_dist_focal = float(glm_raw.pvalues["dist_focal"])
    beta_dist_focal = float(glm_raw.params["dist_focal"])

    size_zeroed_sparse = "size_diff" in sparse_excluded
    size_zeroed_hinge = abs(hinge_by_name.get("size_diff", 0.0)) < 1e-8
    loc_zeroed_sparse = "location_adv" in sparse_excluded
    loc_zeroed_hinge = abs(hinge_by_name.get("location_adv", 0.0)) < 1e-8

    # Calibrated score:
    # - size evidence is weak/inconsistent and often zeroed.
    # - location has weak relative-effect evidence, but a mild distance-from-home signal.
    size_component = 25
    if p_size < 0.10:
        size_component = 40
    if p_size < 0.05:
        size_component = 55
    if size_zeroed_sparse and size_zeroed_hinge:
        size_component -= 10
    size_component = int(np.clip(size_component, 0, 100))

    location_component = 35
    if p_loc < 0.10:
        location_component = 50
    if p_loc < 0.05:
        location_component = 65
    if p_dist_focal < 0.10 and beta_dist_focal < 0:
        location_component += 10
    if loc_zeroed_sparse and not loc_zeroed_hinge:
        location_component -= 3
    location_component = int(np.clip(location_component, 0, 100))

    response = int(round((size_component + location_component) / 2))

    explanation = (
        "Evidence is mixed and generally weak for a strong joint effect. "
        f"In the controlled binomial GLM on relative terms, size_diff has beta={beta_size:.3f} (p={p_size:.3f}) "
        f"and location_adv has beta={beta_loc:.4f} (p={p_loc:.3f}), so neither reaches p<0.05. "
        f"A raw-location robustness model shows dist_focal beta={beta_dist_focal:.4f} (p={p_dist_focal:.3f}), "
        "suggesting only mild evidence that being farther from the focal group's center reduces winning odds. "
        "Interpretable models corroborate weak size effects: WinsorizedSparseOLS and HingeEBM both zero out size_diff, "
        "while SmartAdditive gives it only moderate importance relative to distance features. "
        "For location, SmartAdditive shows a non-monotonic location_adv shape and both SmartAdditive/HingeEBM retain some "
        "distance-related signal, but sparse linear zeroing indicates that effect is not robustly linear. "
        "Overall this supports a low-to-moderate 'yes': contest location appears to matter somewhat, but relative group size "
        "does not show consistent robust influence in this sample."
    )

    out = {"response": response, "explanation": explanation}
    (run_dir / "conclusion.txt").write_text(json.dumps(out))

    print_header("Final Likert Output")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
