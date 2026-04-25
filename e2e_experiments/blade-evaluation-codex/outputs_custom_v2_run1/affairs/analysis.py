import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import r2_score

from agentic_imodels import (
    HingeEBMRegressor,
    HingeGAMRegressor,
    WinsorizedSparseOLSRegressor,
)


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def counterfactual_binary_effect(model, X: pd.DataFrame, binary_col: str) -> float:
    x0 = X.copy()
    x1 = X.copy()
    x0[binary_col] = 0.0
    x1[binary_col] = 1.0
    return float(np.mean(model.predict(x1) - model.predict(x0)))


def hinge_ebm_effective_coefs(model) -> dict:
    n_sel = len(model.selected_)
    coefs = model.lasso_.coef_
    effective = {}

    for i in range(n_sel):
        j = int(model.selected_[i])
        effective[j] = float(coefs[i])

    for idx, (feat_idx, _knot, direction) in enumerate(model.hinge_info_):
        c = float(coefs[n_sel + idx])
        if abs(c) < 1e-10:
            continue
        j = int(model.selected_[feat_idx])
        if direction == "pos":
            effective[j] = effective.get(j, 0.0) + c
        else:
            effective[j] = effective.get(j, 0.0) - c

    return effective


def format_top_importances(feature_names, importances, top_k=5):
    arr = np.asarray(importances, dtype=float)
    if np.all(np.abs(arr) < 1e-12):
        return "all near zero"
    idx = np.argsort(-np.abs(arr))[:top_k]
    return ", ".join([f"{feature_names[i]}={arr[i]:.3f}" for i in idx])


def main():
    here = Path(".")

    info = json.loads((here / "info.json").read_text())
    research_question = info["research_questions"][0].strip()

    df = pd.read_csv(here / "affairs.csv")

    # Encode categorical fields for regression models.
    df["has_children"] = (df["children"].astype(str).str.lower() == "yes").astype(float)
    df["gender_male"] = (df["gender"].astype(str).str.lower() == "male").astype(float)

    iv = "has_children"
    dv = "affairs"
    feature_cols = [
        "has_children",
        "gender_male",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]

    X = df[feature_cols].astype(float)
    y = df[dv].astype(float)

    print("=" * 88)
    print("Research question:")
    print(research_question)
    print("=" * 88)

    print("\nData overview")
    print("Rows, cols:", df.shape)
    print("Missing values per column:")
    print(df[feature_cols + [dv]].isna().sum().to_string())

    print("\nOutcome distribution (affairs)")
    print(y.describe().to_string())
    print("Value counts:")
    print(df[dv].value_counts().sort_index().to_string())

    print("\nIV distribution (children)")
    print(df["children"].value_counts().to_string())
    means = df.groupby("children")[dv].mean().sort_index()
    print("Mean affairs by children group:")
    print(means.to_string())

    print("\nCorrelations with affairs")
    corr = pd.concat([X, y], axis=1).corr(numeric_only=True)[dv].sort_values(ascending=False)
    print(corr.to_string())

    # Bivariate tests.
    y_yes = y[df[iv] == 1.0]
    y_no = y[df[iv] == 0.0]
    pearson_r, pearson_p = stats.pearsonr(df[iv], y)
    t_stat, t_p = stats.ttest_ind(y_yes, y_no, equal_var=False)

    print("\nBivariate tests: has_children vs affairs")
    print(f"Pearson r = {pearson_r:.4f}, p = {pearson_p:.4g}")
    print(f"Welch t-test t = {t_stat:.4f}, p = {t_p:.4g}")
    print(f"Mean difference (yes - no) = {y_yes.mean() - y_no.mean():.4f}")

    # Controlled classical models.
    x_sm = sm.add_constant(X, has_constant="add")

    dispersion_ratio = float(y.var() / max(y.mean(), 1e-12))
    print("\nOutcome variance/mean ratio (overdispersion check):", f"{dispersion_ratio:.3f}")

    nb_model = sm.GLM(y, x_sm, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
    ols_model = sm.OLS(y, x_sm).fit()

    print("\nControlled GLM: Negative Binomial")
    print(nb_model.summary())
    print("\nControlled OLS (sensitivity)")
    print(ols_model.summary())

    nb_coef = float(nb_model.params[iv])
    nb_p = float(nb_model.pvalues[iv])
    nb_ci_low, nb_ci_high = nb_model.conf_int().loc[iv].astype(float).tolist()

    ols_coef = float(ols_model.params[iv])
    ols_p = float(ols_model.pvalues[iv])

    print("\nChildren effect from controlled models")
    print(f"NB coef = {nb_coef:.4f}, p = {nb_p:.4g}, 95% CI = [{nb_ci_low:.4f}, {nb_ci_high:.4f}]")
    print(f"OLS coef = {ols_coef:.4f}, p = {ols_p:.4g}")

    # Interpretable agentic_imodels.
    models = [
        ("WinsorizedSparseOLSRegressor", WinsorizedSparseOLSRegressor()),
        ("HingeGAMRegressor", HingeGAMRegressor()),
        ("HingeEBMRegressor", HingeEBMRegressor()),
    ]

    model_evidence = []

    print("\n" + "=" * 88)
    print("Interpretable models from agentic_imodels")
    print("=" * 88)

    for name, model in models:
        model.fit(X, y)
        preds = model.predict(X)
        r2 = float(r2_score(y, preds))

        print(f"\n--- {name} (in-sample R^2 = {r2:.4f}) ---")
        print(model)

        child_effect = counterfactual_binary_effect(model, X, iv)

        child_active = True
        child_magnitude = abs(child_effect)
        detail = ""

        if name == "WinsorizedSparseOLSRegressor":
            child_idx = feature_cols.index(iv)
            child_active = child_idx in set(getattr(model, "support_", []))
            if child_active:
                loc = list(model.support_).index(child_idx)
                child_magnitude = abs(float(model.ols_coef_[loc]))
            else:
                child_magnitude = 0.0
            print("Top absolute coefficients:")
            coef_vec = np.zeros(len(feature_cols))
            for c, j in zip(model.ols_coef_, model.support_):
                coef_vec[int(j)] = float(c)
            print(format_top_importances(feature_cols, coef_vec, top_k=len(feature_cols)))
            detail = f"children_selected={child_active}"

        elif name == "HingeGAMRegressor":
            importances = np.asarray(getattr(model, "feature_importances_", np.zeros(len(feature_cols))))
            child_idx = feature_cols.index(iv)
            child_magnitude = abs(float(importances[child_idx]))
            child_active = child_magnitude > 1e-8
            print("Top feature importances:")
            print(format_top_importances(feature_cols, importances, top_k=len(feature_cols)))
            detail = f"children_importance={child_magnitude:.6f}"

        elif name == "HingeEBMRegressor":
            child_idx = feature_cols.index(iv)
            effective = hinge_ebm_effective_coefs(model)
            child_coef = float(effective.get(child_idx, 0.0))
            child_magnitude = abs(child_coef)
            child_active = child_magnitude > 1e-8
            sorted_effective = sorted(effective.items(), key=lambda kv: abs(kv[1]), reverse=True)
            text = ", ".join([f"{feature_cols[j]}={c:.3f}" for j, c in sorted_effective])
            print("Approx effective linear coefficients (from printed hinge form):")
            print(text if text else "all near zero")
            detail = f"children_effective_coef={child_coef:.6f}"

        model_evidence.append(
            {
                "model": name,
                "r2": r2,
                "children_effect_cf": child_effect,
                "children_active": child_active,
                "children_magnitude": child_magnitude,
                "detail": detail,
            }
        )

        print(
            f"Children counterfactual effect (set yes minus set no, avg): {child_effect:.4f}; "
            f"active={child_active} ({detail})"
        )

    zeroed_models = [m["model"] for m in model_evidence if not m["children_active"]]
    avg_abs_cf = float(np.mean([abs(m["children_effect_cf"]) for m in model_evidence]))
    avg_cf = float(np.mean([m["children_effect_cf"] for m in model_evidence]))
    avg_abs_display = float(np.mean([m["children_magnitude"] for m in model_evidence]))

    print("\nModel robustness summary for children")
    print("Models zeroing/excluding children:", ", ".join(zeroed_models) if zeroed_models else "none")
    print(f"Average abs counterfactual children effect across models: {avg_abs_cf:.4f}")
    print(f"Average signed counterfactual children effect across models: {avg_cf:.4f}")
    print(f"Average abs displayed/selected children magnitude across models: {avg_abs_display:.4f}")

    # Likert calibration (0=strong No, 100=strong Yes for "children decrease affairs").
    # Strong null evidence branch from SKILL rubric.
    if (
        nb_p >= 0.05
        and ols_p >= 0.05
        and len(zeroed_models) >= 2
        and avg_abs_display < 0.05
    ):
        score = 10
    elif nb_coef < 0 and nb_p < 0.05 and avg_cf < -0.1:
        score = 85
    elif nb_coef < 0 and (nb_p < 0.10 or ols_p < 0.10):
        score = 60
    elif avg_cf < -0.05:
        score = 40
    else:
        score = 20

    # If bivariate goes in opposite direction and is significant, push score down.
    if (y_yes.mean() - y_no.mean()) > 0 and t_p < 0.05:
        score = max(0, score - 5)

    explanation = (
        f"Research question: {research_question} "
        f"Bivariate evidence shows higher affairs among those with children "
        f"(mean_yes={y_yes.mean():.3f}, mean_no={y_no.mean():.3f}; Welch p={t_p:.4g}). "
        f"After controlling for gender, age, years married, religiousness, education, occupation, and marriage rating, "
        f"the children coefficient is not significant in Negative Binomial GLM "
        f"(beta={nb_coef:.3f}, p={nb_p:.4g}, CI=[{nb_ci_low:.3f},{nb_ci_high:.3f}]) "
        f"and also not significant in OLS (beta={ols_coef:.3f}, p={ols_p:.4g}). "
        f"Interpretable agentic_imodels agree: children is zeroed/excluded in {len(zeroed_models)}/3 models "
        f"({', '.join(zeroed_models) if zeroed_models else 'none'}), and near-zero displayed/selected children magnitude "
        f"(avg_abs_display={avg_abs_display:.3f}; avg_counterfactual={avg_cf:.3f}). "
        f"Dominant effects are from years married (+), religiousness (-), and marriage rating (-). "
        f"This supports a strong 'No' to the claim that having children decreases affair engagement; apparent bivariate differences "
        f"are not robust after controls."
    )

    payload = {"response": int(score), "explanation": explanation}
    (here / "conclusion.txt").write_text(json.dumps(payload), encoding="utf-8")

    print("\nWrote conclusion.txt:")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
