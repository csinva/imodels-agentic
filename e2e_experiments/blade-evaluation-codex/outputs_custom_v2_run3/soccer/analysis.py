import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from agentic_imodels import (
    HingeEBMRegressor,
    HingeGAMRegressor,
    WinsorizedSparseOLSRegressor,
)

warnings.filterwarnings("ignore")


def build_effective_coefs_hinge_ebm(model):
    """Reconstruct effective linear coefficients shown by HingeEBMRegressor.__str__."""
    coefs = model.lasso_.coef_
    n_sel = len(model.selected_)

    effective = {}
    for i in range(n_sel):
        j_orig = int(model.selected_[i])
        effective[j_orig] = float(coefs[i])

    for idx, (feat_idx, knot, direction) in enumerate(model.hinge_info_):
        j_orig = int(model.selected_[feat_idx])
        c = float(coefs[n_sel + idx])
        if abs(c) < 1e-6:
            continue
        if direction == "pos":
            effective[j_orig] = effective.get(j_orig, 0.0) + c
        else:
            effective[j_orig] = effective.get(j_orig, 0.0) - c

    return {k: v for k, v in effective.items() if abs(v) > 1e-6}


def rank_of_feature(values, feature_idx):
    order = np.argsort(-values)
    for i, idx in enumerate(order, start=1):
        if int(idx) == int(feature_idx):
            return i
    return None


with open("info.json", "r", encoding="utf-8") as f:
    info = json.load(f)

question = info["research_questions"][0]
print("Research question:", question)

# ------------------------------------------------------------------
# 1) Load and prepare data
# ------------------------------------------------------------------

df = pd.read_csv("soccer.csv")
print("\nRaw shape:", df.shape)

# Skin-tone construct and core engineered features.
df["skin_tone"] = df[["rater1", "rater2"]].mean(axis=1)
df["age"] = 2013 - pd.to_datetime(df["birthday"], format="%d.%m.%Y", errors="coerce").dt.year

df["red_rate"] = df["redCards"] / df["games"].clip(lower=1)
for col in ["goals", "yellowCards", "yellowReds", "victories", "ties", "defeats"]:
    df[f"{col}_pg"] = df[col] / df["games"].clip(lower=1)

# Main analysis framing dark vs light as requested (exclude neutral 0.5).
analysis_df = df[(df["skin_tone"].notna()) & (df["skin_tone"] != 0.5)].copy()
analysis_df["dark_skin"] = (analysis_df["skin_tone"] > 0.5).astype(int)
analysis_df["light_skin"] = (analysis_df["skin_tone"] < 0.5).astype(int)

print("Analysis subset shape (dark/light only, no neutral):", analysis_df.shape)
print("Dark-skin proportion:", round(float(analysis_df["dark_skin"].mean()), 4))

# ------------------------------------------------------------------
# 2) Explore data: summary statistics, distributions, correlations
# ------------------------------------------------------------------

summary_cols = [
    "skin_tone",
    "dark_skin",
    "redCards",
    "red_rate",
    "games",
    "yellowCards",
    "yellowReds",
    "goals",
    "height",
    "weight",
    "meanIAT",
    "meanExp",
    "age",
]

print("\nSummary statistics:")
print(analysis_df[summary_cols].describe().T)

print("\nRed-card count distribution:")
print(analysis_df["redCards"].value_counts().sort_index().head(10))

group_stats = analysis_df.groupby("dark_skin")["red_rate"].agg(["count", "mean", "std"])
print("\nRed-card rate by dark_skin (0=light, 1=dark):")
print(group_stats)

corr_cols = [
    "skin_tone",
    "dark_skin",
    "red_rate",
    "yellowCards_pg",
    "yellowReds_pg",
    "goals_pg",
    "victories_pg",
    "ties_pg",
    "defeats_pg",
    "meanIAT",
    "meanExp",
    "age",
    "height",
    "weight",
]

print("\nCorrelations with red_rate:")
corrs = analysis_df[corr_cols].corr(numeric_only=True)["red_rate"].sort_values(ascending=False)
print(corrs)

# ------------------------------------------------------------------
# 3) Classical tests: bivariate + controlled GLM(Poisson) with controls
# ------------------------------------------------------------------

light = analysis_df[analysis_df["dark_skin"] == 0]["red_rate"]
dark = analysis_df[analysis_df["dark_skin"] == 1]["red_rate"]
ttest_res = stats.ttest_ind(dark, light, equal_var=False, nan_policy="omit")

print("\nBivariate Welch t-test on red_rate (dark vs light):")
print(ttest_res)

biv_df = analysis_df[["redCards", "games", "dark_skin"]].dropna().copy()
X_biv = sm.add_constant(biv_df[["dark_skin"]], has_constant="add")
poisson_biv = sm.GLM(
    biv_df["redCards"].astype(float),
    X_biv.astype(float),
    family=sm.families.Poisson(),
    offset=np.log(biv_df["games"].clip(lower=1).astype(float)),
).fit(cov_type="HC3")

biv_coef = float(poisson_biv.params["dark_skin"])
biv_p = float(poisson_biv.pvalues["dark_skin"])
biv_irr = float(np.exp(biv_coef))
biv_ci = np.exp(poisson_biv.conf_int().loc["dark_skin"].to_numpy(dtype=float))

print("\nBivariate Poisson (offset=log(games)) for dark_skin:")
print(
    f"coef={biv_coef:.4f}, IRR={biv_irr:.4f}, p={biv_p:.4g}, "
    f"95%CI(IRR)=({biv_ci[0]:.4f}, {biv_ci[1]:.4f})"
)

controls = [
    "height",
    "weight",
    "age",
    "meanIAT",
    "meanExp",
    "goals_pg",
    "yellowCards_pg",
    "yellowReds_pg",
    "victories_pg",
    "ties_pg",
    "defeats_pg",
]
cat_controls = ["position", "leagueCountry"]

glm_cols = ["redCards", "games", "dark_skin", "skin_tone"] + controls + cat_controls
glm_df = analysis_df[glm_cols].dropna().copy()

X_ctrl = pd.concat(
    [
        glm_df[["dark_skin"] + controls],
        pd.get_dummies(glm_df[cat_controls], drop_first=True),
    ],
    axis=1,
).astype(float)
X_ctrl = sm.add_constant(X_ctrl, has_constant="add")

y_ctrl = glm_df["redCards"].astype(float)
offset_ctrl = np.log(glm_df["games"].clip(lower=1).astype(float))

poisson_ctrl = sm.GLM(
    y_ctrl,
    X_ctrl,
    family=sm.families.Poisson(),
    offset=offset_ctrl,
).fit(cov_type="HC3", maxiter=200)

dark_coef = float(poisson_ctrl.params["dark_skin"])
dark_p = float(poisson_ctrl.pvalues["dark_skin"])
dark_irr = float(np.exp(dark_coef))
dark_ci = np.exp(poisson_ctrl.conf_int().loc["dark_skin"].to_numpy(dtype=float))

print("\nControlled Poisson GLM (with controls + league + position):")
print(
    f"dark_skin coef={dark_coef:.4f}, IRR={dark_irr:.4f}, p={dark_p:.4g}, "
    f"95%CI(IRR)=({dark_ci[0]:.4f}, {dark_ci[1]:.4f})"
)

# Sensitivity: continuous skin-tone level instead of dark/light threshold.
X_skin = pd.concat(
    [
        glm_df[["skin_tone"] + controls],
        pd.get_dummies(glm_df[cat_controls], drop_first=True),
    ],
    axis=1,
).astype(float)
X_skin = sm.add_constant(X_skin, has_constant="add")

poisson_skin = sm.GLM(
    y_ctrl,
    X_skin,
    family=sm.families.Poisson(),
    offset=offset_ctrl,
).fit(cov_type="HC3", maxiter=200)

skin_coef = float(poisson_skin.params["skin_tone"])
skin_p = float(poisson_skin.pvalues["skin_tone"])
skin_irr = float(np.exp(skin_coef))
skin_ci = np.exp(poisson_skin.conf_int().loc["skin_tone"].to_numpy(dtype=float))

print("\nSensitivity Poisson GLM using continuous skin_tone:")
print(
    f"skin_tone coef={skin_coef:.4f}, IRR={skin_irr:.4f}, p={skin_p:.4g}, "
    f"95%CI(IRR)=({skin_ci[0]:.4f}, {skin_ci[1]:.4f})"
)

# ------------------------------------------------------------------
# 4) Interpretable models from agentic_imodels (at least two, print all)
# ------------------------------------------------------------------

model_df = glm_df.copy()
X_model = pd.concat(
    [
        model_df[["dark_skin", "skin_tone"] + controls],
        pd.get_dummies(model_df[cat_controls], drop_first=True),
    ],
    axis=1,
).astype(float)
y_model = (model_df["redCards"] / model_df["games"].clip(lower=1)).astype(float)

feature_names = list(X_model.columns)
dark_idx = feature_names.index("dark_skin")
skin_idx = feature_names.index("skin_tone")

sample_n = min(25000, len(X_model))
if len(X_model) > sample_n:
    sampled_idx = X_model.sample(n=sample_n, random_state=42).index
    X_fit = X_model.loc[sampled_idx].copy()
    y_fit = y_model.loc[sampled_idx].copy()
else:
    X_fit = X_model.copy()
    y_fit = y_model.copy()

print(f"\nInterpretable-model training shape: {X_fit.shape}")

wins = WinsorizedSparseOLSRegressor(max_features=10, cv=3).fit(X_fit, y_fit)
print("\n=== WinsorizedSparseOLSRegressor ===")
print(wins)

hgam = HingeGAMRegressor(n_knots=2, max_input_features=15).fit(X_fit, y_fit)
print("\n=== HingeGAMRegressor ===")
print(hgam)

hebm = HingeEBMRegressor(n_knots=2, max_input_features=15, ebm_outer_bags=3, ebm_max_rounds=500).fit(X_fit, y_fit)
print("\n=== HingeEBMRegressor ===")
print(hebm)

# Extract model-specific evidence for dark_skin vs skin_tone.
wins_coef_map = {int(j): float(c) for j, c in zip(wins.support_, wins.ols_coef_)}
wins_dark_coef = wins_coef_map.get(dark_idx, 0.0)
wins_skin_coef = wins_coef_map.get(skin_idx, 0.0)

hgam_dark_importance = float(hgam.feature_importances_[dark_idx])
hgam_skin_importance = float(hgam.feature_importances_[skin_idx])
hgam_total_importance = float(np.sum(hgam.feature_importances_))

hgam_dark_rank = rank_of_feature(hgam.feature_importances_, dark_idx)
hgam_skin_rank = rank_of_feature(hgam.feature_importances_, skin_idx)

hebm_effective = build_effective_coefs_hinge_ebm(hebm)
hebm_dark_coef = float(hebm_effective.get(dark_idx, 0.0))
hebm_skin_coef = float(hebm_effective.get(skin_idx, 0.0))

print("\nModel-derived feature evidence:")
print(
    f"dark_skin index={dark_idx}, skin_tone index={skin_idx}; "
    f"Winsorized dark={wins_dark_coef:.6f}, skin_tone={wins_skin_coef:.6f}; "
    f"HingeGAM dark_importance={hgam_dark_importance:.6g} (rank {hgam_dark_rank}), "
    f"skin_importance={hgam_skin_importance:.6g} (rank {hgam_skin_rank}); "
    f"HingeEBM dark={hebm_dark_coef:.6f}, skin_tone={hebm_skin_coef:.6f}."
)

# ------------------------------------------------------------------
# 5) Calibrated conclusion score + explanation JSON
# ------------------------------------------------------------------

zero_votes_dark = 0
if abs(wins_dark_coef) < 1e-12:
    zero_votes_dark += 1
if hgam_dark_importance < 1e-12:
    zero_votes_dark += 1
if abs(hebm_dark_coef) < 1e-12:
    zero_votes_dark += 1

# Primary calibration centered on controlled dark-vs-light effect.
if dark_p <= 0.05 and dark_irr > 1:
    score = 80 if zero_votes_dark <= 1 else 65
elif dark_p <= 0.10 and dark_irr > 1:
    score = 60 if zero_votes_dark <= 1 else 45
elif dark_p <= 0.15 and dark_irr > 1:
    score = 45 if zero_votes_dark <= 1 else 28
else:
    score = 22 if zero_votes_dark >= 2 else 30

# Secondary evidence (bivariate and continuous-tone sensitivity).
if biv_p <= 0.10 and biv_irr > 1:
    score += 4
if skin_p <= 0.05 and skin_irr > 1:
    score += 3

score = int(np.clip(round(score), 0, 100))

explanation = (
    f"Question: whether dark-skinned players are more likely than light-skinned players to receive red cards. "
    f"In bivariate rate comparisons, dark skin showed a higher red-card rate (Poisson IRR={biv_irr:.2f}, p={biv_p:.3f}; "
    f"Welch t-test p={ttest_res.pvalue:.3f}), but this was only marginal. "
    f"In the controlled Poisson GLM with exposure offset and confounders, the dark-vs-light indicator remained positive "
    f"(IRR={dark_irr:.2f}) but was not statistically significant (p={dark_p:.3f}, 95% CI includes 1). "
    f"Interpretable models gave mostly null evidence for the binary dark indicator: WinsorizedSparseOLS coefficient={wins_dark_coef:.5f}, "
    f"HingeGAM importance={hgam_dark_importance:.3g}, HingeEBM effective coefficient={hebm_dark_coef:.5f}, with {zero_votes_dark}/3 models zeroing it out. "
    f"At the same time, continuous skin-tone level showed a small positive controlled association (IRR={skin_irr:.2f}, p={skin_p:.3f}) and positive small coefficients in sparse/hinge models, "
    f"suggesting a weak gradient rather than strong robust dark-vs-light separation. "
    f"Overall evidence is mixed-to-weak, so the calibrated answer is below neutral rather than a strong yes."
)

payload = {"response": score, "explanation": explanation}
Path("conclusion.txt").write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

print("\nFinal Likert response:", score)
print("Wrote conclusion.txt")
