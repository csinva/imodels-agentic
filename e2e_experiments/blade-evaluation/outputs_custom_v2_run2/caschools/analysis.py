import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

from interp_models import SmartAdditiveRegressor, HingeEBMRegressor


# -----------------------------
# Step 0: Load metadata and data
# -----------------------------
with open("info.json", "r", encoding="utf-8") as f:
    info = json.load(f)

research_question = info["research_questions"][0]
print("Research question:", research_question)

raw_df = pd.read_csv("caschools.csv")

# Construct key variables for the question
raw_df["str_ratio"] = raw_df["students"] / raw_df["teachers"]
raw_df["avg_score"] = (raw_df["read"] + raw_df["math"]) / 2.0
raw_df["grade_kk08"] = (raw_df["grades"] == "KK-08").astype(int)

# Dependent variable and focal independent variable
DV = "avg_score"
IV = "str_ratio"

# Keep substantive numeric predictors (exclude pure identifiers and DV components)
numeric_predictors = [
    "str_ratio",
    "students",
    "teachers",
    "calworks",
    "lunch",
    "computer",
    "expenditure",
    "income",
    "english",
    "grade_kk08",
]

analysis_cols = [DV] + numeric_predictors
df = raw_df[analysis_cols].copy()

print("\nRows used:", len(df))
print("Columns used:", analysis_cols)


# -----------------------------
# Step 1: Explore
# -----------------------------
print("\n=== Step 1: Summary statistics ===")
print(df.describe().T)

print("\n=== Step 1: Distribution diagnostics (skewness) ===")
print(df.skew(numeric_only=True).sort_values(ascending=False))

print("\n=== Step 1: Bivariate correlation with DV ===")
corr_with_dv = df.corr(numeric_only=True)[DV].sort_values(ascending=False)
print(corr_with_dv)

bivar_corr = float(df[[IV, DV]].corr().iloc[0, 1])
print(f"\nPearson corr({IV}, {DV}) = {bivar_corr:.4f}")


# -----------------------------
# Step 2: OLS with controls
# -----------------------------
print("\n=== Step 2: OLS models ===")
y = df[DV]

# Bivariate OLS
X_bi = sm.add_constant(df[[IV]])
model_bi = sm.OLS(y, X_bi).fit()
print("\n--- Bivariate OLS: avg_score ~ str_ratio ---")
print(model_bi.summary())

# Controlled OLS
X_full = sm.add_constant(df[numeric_predictors])
model_full = sm.OLS(y, X_full).fit()
print("\n--- Controlled OLS: avg_score ~ str_ratio + controls ---")
print(model_full.summary())

bi_coef = float(model_bi.params[IV])
bi_p = float(model_bi.pvalues[IV])
full_coef = float(model_full.params[IV])
full_p = float(model_full.pvalues[IV])


# -----------------------------
# Step 3: Interpretable models
# -----------------------------
print("\n=== Step 3: SmartAdditiveRegressor ===")
X_interp = df[numeric_predictors]

smart = SmartAdditiveRegressor(n_rounds=200)
smart.fit(X_interp, y)
print(smart)
smart_effects = smart.feature_effects()
print("SmartAdditive feature effects:")
print(json.dumps(smart_effects, indent=2))

print("\n=== Step 3: HingeEBMRegressor ===")
hinge = HingeEBMRegressor(n_knots=3, max_input_features=len(numeric_predictors))
hinge.fit(X_interp, y)
print(hinge)
hinge_effects = hinge.feature_effects()
print("HingeEBM feature effects:")
print(json.dumps(hinge_effects, indent=2))

smart_iv = smart_effects.get(IV, {"direction": "zero", "importance": 0.0, "rank": 0})
hinge_iv = hinge_effects.get(IV, {"direction": "zero", "importance": 0.0, "rank": 0})

# Pull threshold info from SmartAdditive for shape interpretation
smart_threshold_text = ""
if hasattr(smart, "shape_functions_"):
    iv_idx = numeric_predictors.index(IV)
    if iv_idx in smart.shape_functions_:
        thresholds = smart.shape_functions_[iv_idx][0]
        if len(thresholds) > 0:
            shown = ", ".join([f"{t:.2f}" for t in thresholds[:3]])
            extra = "" if len(thresholds) <= 3 else ", ..."
            smart_threshold_text = f"Threshold behavior appears around STR values near {shown}{extra}."


# -----------------------------
# Step 4: Build scored conclusion
# -----------------------------
def dir_supports_hypothesis(direction: str) -> bool:
    d = (direction or "").lower()
    return ("negative" in d) or ("decreasing" in d)


score = 50.0

# Controlled OLS gets highest weight; if effect vanishes with controls, penalize.
if full_coef < 0:
    if full_p < 0.001:
        score += 32
    elif full_p < 0.01:
        score += 28
    elif full_p < 0.05:
        score += 22
    elif full_p < 0.10:
        score += 10
    else:
        score -= 15
else:
    score -= 40 if full_p < 0.05 else 25

# Bivariate evidence gets secondary weight.
if bi_coef < 0:
    if bi_p < 0.001:
        score += 12
    elif bi_p < 0.01:
        score += 10
    elif bi_p < 0.05:
        score += 7
    elif bi_p < 0.10:
        score += 4
else:
    score -= 12 if bi_p < 0.05 else 6

# SmartAdditive evidence
smart_imp = float(smart_iv.get("importance", 0.0) or 0.0)
if dir_supports_hypothesis(smart_iv.get("direction", "")):
    score += min(8, smart_imp * 40)
else:
    score -= min(8, smart_imp * 40)

# Hinge evidence (sparse selection is strong evidence of relevance/irrelevance)
hinge_imp = float(hinge_iv.get("importance", 0.0) or 0.0)
if hinge_imp == 0:
    score -= 8
elif dir_supports_hypothesis(hinge_iv.get("direction", "")):
    score += min(10, hinge_imp * 60)
else:
    score -= min(10, hinge_imp * 60)

# Extra robustness bump/penalty
if smart_iv.get("rank", 0) and hinge_iv.get("rank", 0):
    if smart_iv["rank"] <= 3 and hinge_iv["rank"] <= 3:
        score += 5
elif smart_iv.get("importance", 0.0) < 0.05 and hinge_iv.get("rank", 0) == 0:
    score -= 6

score = int(np.clip(round(score), 0, 100))

# Identify meaningful OLS confounders
confounder_stats = []
for col in numeric_predictors:
    if col == IV:
        continue
    p = float(model_full.pvalues.get(col, np.nan))
    coef = float(model_full.params.get(col, np.nan))
    if np.isfinite(p) and p < 0.05:
        confounder_stats.append((col, coef, p))

confounder_stats.sort(key=lambda x: x[2])
if confounder_stats:
    conf_text = "; ".join([
        f"{c} (coef={b:.3f}, p={p:.3g})" for c, b, p in confounder_stats[:4]
    ])
else:
    conf_text = "No control variable reached p<0.05 in the controlled OLS."

# Top features from interpretable models
smart_top = sorted(
    [(k, v) for k, v in smart_effects.items() if v.get("importance", 0) > 0 and k != IV],
    key=lambda kv: kv[1]["importance"],
    reverse=True,
)[:3]
hinge_top = sorted(
    [(k, v) for k, v in hinge_effects.items() if v.get("importance", 0) > 0 and k != IV],
    key=lambda kv: kv[1]["importance"],
    reverse=True,
)[:3]

smart_top_text = ", ".join([
    f"{k} (rank={v.get('rank', 0)}, imp={100*v.get('importance', 0):.1f}%, dir={v.get('direction')})"
    for k, v in smart_top
]) if smart_top else "none"

hinge_top_text = ", ".join([
    f"{k} (rank={v.get('rank', 0)}, imp={100*v.get('importance', 0):.1f}%, dir={v.get('direction')})"
    for k, v in hinge_top
]) if hinge_top else "none"

explanation = (
    f"Research question: whether lower student-teacher ratio (STR) is associated with higher academic performance. "
    f"Using avg_score=(read+math)/2 as DV, bivariate evidence is corr={bivar_corr:.3f}, "
    f"OLS coef for STR={bi_coef:.3f} (p={bi_p:.3g}). "
    f"After controls, STR coef={full_coef:.3f} (p={full_p:.3g}), so the adjusted effect is "
    f"{'negative' if full_coef < 0 else 'positive'} and "
    f"{'statistically robust' if full_p < 0.05 else 'weak/marginal'}. "
    f"SmartAdditive ranks STR #{smart_iv.get('rank', 0)} with importance={100*smart_imp:.1f}% and direction={smart_iv.get('direction', 'NA')}; "
    f"HingeEBM ranks STR #{hinge_iv.get('rank', 0)} with importance={100*hinge_imp:.1f}% and direction={hinge_iv.get('direction', 'NA')}. "
    f"{smart_threshold_text} "
    f"Key confounders in controlled OLS: {conf_text}. "
    f"Most influential non-STR features were SmartAdditive: {smart_top_text}; HingeEBM: {hinge_top_text}. "
    f"Overall, evidence strength reflects consistency across bivariate, controlled OLS, and two interpretable models."
)

result = {
    "response": score,
    "explanation": explanation,
}

with open("conclusion.txt", "w", encoding="utf-8") as f:
    json.dump(result, f)

print("\nWrote conclusion.txt:")
print(json.dumps(result, indent=2))
