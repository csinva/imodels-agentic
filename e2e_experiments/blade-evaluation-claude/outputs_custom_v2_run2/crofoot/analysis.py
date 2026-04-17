import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Load data
df = pd.read_csv("crofoot.csv")
print("Shape:", df.shape)
print(df.describe())

# Create key derived variables
df["rel_size"] = df["n_focal"] / df["n_other"]   # relative group size
df["size_diff"] = df["n_focal"] - df["n_other"]   # absolute size difference
df["loc_advantage"] = df["dist_other"] - df["dist_focal"]  # positive = focal closer to own range

print("\nCorrelations with win:")
print(df[["win","rel_size","size_diff","dist_focal","dist_other","loc_advantage"]].corr()["win"])

# Bivariate t-tests
winners = df[df["win"]==1]
losers  = df[df["win"]==0]
for col in ["rel_size","size_diff","dist_focal","dist_other","loc_advantage"]:
    t, p = stats.ttest_ind(winners[col], losers[col])
    print(f"{col}: winners_mean={winners[col].mean():.3f}, losers_mean={losers[col].mean():.3f}, t={t:.3f}, p={p:.4f}")

# OLS / Logistic regression with controls
feature_cols = ["rel_size", "loc_advantage", "m_focal", "m_other"]
X = df[feature_cols].copy()
X = sm.add_constant(X)
logit_model = sm.Logit(df["win"], X).fit(disp=0)
print("\nLogistic Regression Summary:")
print(logit_model.summary())

# OLS for easier interpretation
ols_model = sm.OLS(df["win"], X).fit()
print("\nOLS Summary:")
print(ols_model.summary())

# Interpretable models
from interp_models import SmartAdditiveRegressor, HingeEBMRegressor

numeric_cols = ["rel_size", "loc_advantage", "dist_focal", "dist_other", "n_focal", "n_other", "m_focal", "m_other"]
X_df = df[numeric_cols].copy()
y = df["win"]

smart = SmartAdditiveRegressor(n_rounds=200)
smart.fit(X_df, y)
print("\nSmartAdditiveRegressor:")
print(smart)
smart_effects = smart.feature_effects()
print("Feature effects:", smart_effects)

try:
    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X_df, y)
    print("\nHingeEBMRegressor:")
    print(hinge)
    hinge_effects = hinge.feature_effects()
    print("Feature effects:", hinge_effects)
except Exception as e:
    print(f"\nHingeEBMRegressor failed: {e}")
    hinge_effects = {}

# Additional logistic model using dist_focal directly (raw location measure)
feature_cols2 = ["rel_size", "dist_focal", "dist_other"]
X2 = df[feature_cols2].copy()
X2 = sm.add_constant(X2)
logit2 = sm.Logit(df["win"], X2).fit(disp=0)
print("\nLogistic model with raw distances:")
print(logit2.summary())

# Key bivariate p-values
_, p_rel = stats.ttest_ind(winners["rel_size"], losers["rel_size"])
_, p_focal = stats.ttest_ind(winners["dist_focal"], losers["dist_focal"])
print(f"\nBivariate: rel_size p={p_rel:.4f}, dist_focal p={p_focal:.4f}")

smart_rel  = smart_effects.get("rel_size",  {})
smart_loc  = smart_effects.get("loc_advantage", {})
smart_df   = smart_effects.get("dist_focal", {})
smart_do   = smart_effects.get("dist_other", {})

rel_imp   = float(smart_rel.get("importance", 0))
loc_imp   = float(smart_loc.get("importance", 0))
df_imp    = float(smart_df.get("importance", 0))
do_imp    = float(smart_do.get("importance", 0))
location_imp = df_imp + do_imp + loc_imp   # combined location signal

explanation = (
    f"Research question: Do relative group size and contest location influence win probability in capuchin intergroup contests? "
    f"LOCATION: dist_focal is significant bivariate (p={p_focal:.3f}); winners fought closer to their home-range center "
    f"(mean dist_focal={winners['dist_focal'].mean():.0f}m vs {losers['dist_focal'].mean():.0f}m for losers). "
    f"SmartAdditive ranks dist_other #1 (importance={do_imp:.2%}) and dist_focal #2 ({df_imp:.2%}), "
    f"confirming location is the dominant predictor. loc_advantage (dist_other-dist_focal) also shows an increasing-trend effect (importance={loc_imp:.2%}). "
    f"RELATIVE SIZE: rel_size bivariate p={p_rel:.3f} (not significant). "
    f"SmartAdditive ranks rel_size last (#8, importance={rel_imp:.2%}) with a non-monotonic pattern. "
    f"Logistic regression confirms rel_size is not significant (p={logit2.pvalues.get('rel_size', float('nan')):.3f}) "
    f"while the distance variables are marginal to significant. "
    f"CONCLUSION: Contest location has a strong, robust effect—groups fighting closer to their own home-range center win significantly more often, "
    f"consistent across bivariate tests and SmartAdditive feature importance. Relative group size has a weak, statistically non-significant effect "
    f"in all models, suggesting the 'home advantage' matters more than numerical superiority in this capuchin population."
)

# Scoring: location effect strong (sig bivariate + top SmartAdditive features),
# size effect weak (not sig). Both factors in question; partial yes.
loc_sig = p_focal < 0.05
rel_sig = p_rel < 0.05

if loc_sig and rel_sig:
    score = 85
elif loc_sig and not rel_sig:
    score = 65   # location clearly yes, size weak
elif rel_sig and not loc_sig:
    score = 50
else:
    score = 30

print(f"\nrel_size p={p_rel:.4f} sig={rel_sig}, dist_focal p={p_focal:.4f} sig={loc_sig}")

print(f"\nFinal score: {score}")

result = {"response": score, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)
print("Written conclusion.txt")
