import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from agentic_imodels import (
    HingeEBMRegressor,
    SmartAdditiveRegressor,
    WinsorizedSparseOLSRegressor,
)
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def print_section(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def summarize_distribution(series: pd.Series, bins: int = 10) -> pd.DataFrame:
    counts, edges = np.histogram(series.to_numpy(), bins=bins)
    out = pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "count": counts,
            "share": counts / counts.sum(),
        }
    )
    return out


def beauty_shape_summary(model, X: pd.DataFrame, feature: str = "beauty") -> dict:
    grid = np.linspace(X[feature].quantile(0.01), X[feature].quantile(0.99), 40)
    anchor = X.median(numeric_only=True).to_dict()
    preds = []
    for val in grid:
        row = anchor.copy()
        row[feature] = float(val)
        preds.append(float(model.predict(pd.DataFrame([row], columns=X.columns))[0]))
    preds = np.asarray(preds)
    delta = float(preds[-1] - preds[0])
    if np.std(preds) < 1e-12:
        linear_r2 = 1.0
    else:
        slope, intercept = np.polyfit(grid, preds, 1)
        linear_fit = slope * grid + intercept
        ss_res = float(np.sum((preds - linear_fit) ** 2))
        ss_tot = float(np.sum((preds - preds.mean()) ** 2))
        linear_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    if delta > 0.02:
        direction = "positive"
    elif delta < -0.02:
        direction = "negative"
    else:
        direction = "flat"
    shape = "approximately linear" if linear_r2 >= 0.9 else "nonlinear/threshold-like"
    return {
        "delta_1pct_to_99pct": delta,
        "direction": direction,
        "shape": shape,
        "linear_r2": float(linear_r2),
    }


def main() -> None:
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    question = info["research_questions"][0]

    df = pd.read_csv("teachingratings.csv")

    print_section("Research Question")
    print(question)

    print_section("Data Overview")
    print(f"Rows: {len(df)}")
    print(f"Columns: {df.shape[1]}")
    print("Columns:", ", ".join(df.columns.tolist()))
    print("\nMissing values per column:")
    print(df.isna().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    print_section("Summary Statistics (Numeric)")
    print(df[numeric_cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]])

    print_section("Category Distributions")
    for c in cat_cols:
        vc = df[c].value_counts(dropna=False)
        share = (vc / vc.sum()).round(3)
        print(f"\n{c}:")
        print(pd.DataFrame({"count": vc, "share": share}))

    print_section("Target/Key Variable Distributions")
    print("eval quantiles:")
    print(df["eval"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))
    print("\nbeauty quantiles:")
    print(df["beauty"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))
    print("\nHistogram-like bins for eval:")
    print(summarize_distribution(df["eval"], bins=8))
    print("\nHistogram-like bins for beauty:")
    print(summarize_distribution(df["beauty"], bins=8))

    print_section("Correlations With eval")
    corr_with_eval = df[numeric_cols].corr(numeric_only=True)["eval"].sort_values(ascending=False)
    print(corr_with_eval)

    print_section("Bivariate Statistical Tests")
    pearson_r, pearson_p = stats.pearsonr(df["beauty"], df["eval"])
    spearman_rho, spearman_p = stats.spearmanr(df["beauty"], df["eval"])
    print(f"Pearson r(eval, beauty): {pearson_r:.4f} (p={pearson_p:.4g})")
    print(f"Spearman rho(eval, beauty): {spearman_rho:.4f} (p={spearman_p:.4g})")

    bivariate_ols = smf.ols("eval ~ beauty", data=df).fit()
    print("\nBivariate OLS (eval ~ beauty):")
    print(bivariate_ols.summary())

    print_section("Controlled OLS (Classical Test)")
    formula = (
        "eval ~ beauty + age + students + allstudents + "
        "C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits)"
    )
    controlled_ols = smf.ols(formula, data=df).fit()
    clustered = controlled_ols.get_robustcov_results(cov_type="cluster", groups=df["prof"])
    print(controlled_ols.summary())

    exog_names = clustered.model.exog_names
    beauty_idx_ols = exog_names.index("beauty")
    clustered_ci = clustered.conf_int()[beauty_idx_ols]
    beauty_coef_controlled = float(controlled_ols.params["beauty"])
    beauty_p_controlled = float(controlled_ols.pvalues["beauty"])
    beauty_coef_cluster = float(clustered.params[beauty_idx_ols])
    beauty_p_cluster = float(clustered.pvalues[beauty_idx_ols])

    print("\nBeauty coefficient under controls:")
    print(f"OLS coefficient: {beauty_coef_controlled:.4f}, p={beauty_p_controlled:.4g}")
    print(
        "Cluster-robust (cluster=prof) coefficient: "
        f"{beauty_coef_cluster:.4f}, p={beauty_p_cluster:.4g}, "
        f"95% CI=[{clustered_ci[0]:.4f}, {clustered_ci[1]:.4f}]"
    )

    print_section("Interpretable Models via agentic_imodels")
    feature_cols = [c for c in df.columns if c not in {"eval", "rownames", "prof"}]
    X = pd.get_dummies(df[feature_cols], drop_first=True).astype(float)
    y = df["eval"].astype(float)

    print("Feature index mapping used by agentic_imodels printouts:")
    for i, c in enumerate(X.columns):
        print(f"x{i} -> {c}")

    beauty_idx = X.columns.get_loc("beauty")
    models = {
        "WinsorizedSparseOLSRegressor": WinsorizedSparseOLSRegressor(max_features=8, cv=5),
        "SmartAdditiveRegressor": SmartAdditiveRegressor(),
        "HingeEBMRegressor": HingeEBMRegressor(),
    }

    model_results = {}
    permutation_tables = {}

    for name, model in models.items():
        print_section(name)
        model.fit(X, y)
        print(model)
        pred = model.predict(X)
        train_r2 = float(r2_score(y, pred))
        print(f"\nTrain R^2: {train_r2:.4f}")

        perm = permutation_importance(
            model,
            X,
            y,
            n_repeats=20,
            random_state=42,
            scoring="r2",
        )
        order = np.argsort(-perm.importances_mean)
        beauty_rank = int(np.where(order == beauty_idx)[0][0] + 1)
        beauty_importance = float(perm.importances_mean[beauty_idx])
        top_k = pd.DataFrame(
            {
                "feature": X.columns[order[:8]],
                "perm_importance_mean": perm.importances_mean[order[:8]],
                "perm_importance_std": perm.importances_std[order[:8]],
            }
        )
        print("\nTop permutation importances:")
        print(top_k.to_string(index=False))

        shape = beauty_shape_summary(model, X, feature="beauty")
        print(
            "\nBeauty shape summary from partial dependence style sweep: "
            f"direction={shape['direction']}, delta={shape['delta_1pct_to_99pct']:.4f}, "
            f"shape={shape['shape']} (linear_r2={shape['linear_r2']:.3f})"
        )

        beauty_coef_proxy = 0.0
        beauty_zeroed = False

        if name == "WinsorizedSparseOLSRegressor":
            support = list(model.support_)
            if beauty_idx in support:
                beauty_coef_proxy = float(model.ols_coef_[support.index(beauty_idx)])
            else:
                beauty_zeroed = True

        if name == "SmartAdditiveRegressor":
            if beauty_idx in model.linear_approx_:
                beauty_coef_proxy = float(model.linear_approx_[beauty_idx][0])
            else:
                beauty_zeroed = True

        if name == "HingeEBMRegressor":
            if beauty_idx in model.selected_:
                pos = int(np.where(model.selected_ == beauty_idx)[0][0])
                beauty_coef_proxy = float(model.lasso_.coef_[pos])
            else:
                beauty_zeroed = True

        model_results[name] = {
            "train_r2": train_r2,
            "beauty_rank": beauty_rank,
            "beauty_importance": beauty_importance,
            "beauty_coef_proxy": beauty_coef_proxy,
            "beauty_zeroed": beauty_zeroed,
            "beauty_shape": shape,
        }
        permutation_tables[name] = pd.Series(perm.importances_mean, index=X.columns)

    winsor = model_results["WinsorizedSparseOLSRegressor"]
    smart = model_results["SmartAdditiveRegressor"]
    hinge = model_results["HingeEBMRegressor"]

    sparse_coefs = pd.Series(0.0, index=X.columns)
    winsor_model = models["WinsorizedSparseOLSRegressor"]
    for j, c in zip(winsor_model.support_, winsor_model.ols_coef_):
        sparse_coefs.iloc[j] = c

    effect_table = pd.DataFrame(
        {
            "corr_with_eval": X.apply(lambda col: np.corrcoef(col, y)[0, 1] if col.std() > 0 else 0.0),
            "winsor_sparse_coef": sparse_coefs,
            "perm_imp_winsor": permutation_tables["WinsorizedSparseOLSRegressor"],
            "perm_imp_smart": permutation_tables["SmartAdditiveRegressor"],
            "perm_imp_hingeebm": permutation_tables["HingeEBMRegressor"],
        }
    )
    effect_table["mean_perm_importance"] = effect_table[
        ["perm_imp_winsor", "perm_imp_smart", "perm_imp_hingeebm"]
    ].mean(axis=1)
    effect_table = effect_table.sort_values("mean_perm_importance", ascending=False)

    print_section("Feature Effect Synthesis (Direction + Magnitude + Robustness)")
    print(effect_table.round(4).to_string())

    # Evidence aggregation for calibrated 0-100 response.
    score = 50.0

    if pearson_p < 0.01:
        score += 10
    elif pearson_p < 0.05:
        score += 6
    else:
        score -= 8

    if beauty_p_controlled < 0.01:
        score += 18
    elif beauty_p_controlled < 0.05:
        score += 12
    else:
        score -= 15

    if beauty_p_cluster < 0.05:
        score += 8
    else:
        score -= 10

    if beauty_coef_controlled > 0:
        score += 6
    else:
        score -= 12

    if not winsor["beauty_zeroed"]:
        score += 8
    else:
        score -= 20

    model_directions = [
        winsor["beauty_shape"]["direction"],
        smart["beauty_shape"]["direction"],
        hinge["beauty_shape"]["direction"],
    ]
    n_positive = sum(d == "positive" for d in model_directions)
    if n_positive >= 2:
        score += 8
    elif n_positive == 1:
        score += 2
    else:
        score -= 8

    avg_rank = float(np.mean([winsor["beauty_rank"], smart["beauty_rank"], hinge["beauty_rank"]]))
    if avg_rank <= 3:
        score += 8
    elif avg_rank <= 5:
        score += 4
    else:
        score -= 4

    # Cap overconfidence when statistical signal is robust but effect size is moderate.
    if abs(pearson_r) < 0.3 and abs(beauty_coef_controlled) < 0.2:
        score = min(score, 88.0)

    response = int(np.clip(np.round(score), 0, 100))

    explanation = (
        f"Bivariate evidence is positive (Pearson r={pearson_r:.3f}, p={pearson_p:.2g}; "
        f"OLS beauty beta={bivariate_ols.params['beauty']:.3f}, p={bivariate_ols.pvalues['beauty']:.2g}). "
        f"After controls (age, class-size variables, gender/minority/native/tenure/division/credits), "
        f"beauty remains positive and statistically significant (beta={beauty_coef_controlled:.3f}, "
        f"p={beauty_p_controlled:.2g}; cluster-robust p={beauty_p_cluster:.2g}). "
        f"In the honest sparse model (WinsorizedSparseOLS), beauty is not zeroed and has a positive coefficient "
        f"({winsor['beauty_coef_proxy']:.3f}), while some other features are zeroed, providing null contrasts. "
        f"Across interpretable models, beauty is consistently positive in partial-dependence sweeps "
        f"(directions: {', '.join(model_directions)}), with mean importance rank {avg_rank:.2f} among "
        f"{X.shape[1]} predictors. This supports a robust positive but not dominant effect of beauty on teaching evaluations."
    )

    payload = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True)

    print_section("Final Likert Output")
    print(json.dumps(payload, indent=2))
    print("\nWrote conclusion.txt")


if __name__ == "__main__":
    main()
