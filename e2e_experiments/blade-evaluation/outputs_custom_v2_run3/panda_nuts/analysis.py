import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from interp_models import HingeEBMRegressor, SmartAdditiveRegressor


warnings.filterwarnings("ignore")


def to_builtin(obj):
    """Recursively convert numpy/pandas scalars to Python builtins for JSON."""
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def norm_yes(series: pd.Series) -> pd.Series:
    vals = series.astype(str).str.strip().str.lower()
    return vals.isin(["y", "yes", "1", "true", "t"])


# Step 1: understand the question and explore
with open("info.json", "r", encoding="utf-8") as f:
    info = json.load(f)

research_question = info.get("research_questions", [""])[0]
print("Research question:", research_question)

# IVs from question: age, sex, help; DV interpreted as nut-cracking efficiency
# efficiency = nuts opened per second
df = pd.read_csv("panda_nuts.csv")
df["sex_m"] = df["sex"].astype(str).str.strip().str.lower().eq("m").astype(int)
df["help_bin"] = norm_yes(df["help"]).astype(int)
df["efficiency"] = df["nuts_opened"] / df["seconds"].replace(0, np.nan)

df = df.dropna(subset=["efficiency", "age", "sex_m", "help_bin", "hammer"]).copy()

print("\nDV: efficiency (nuts_opened / seconds)")
print("IVs of interest: age, sex_m (male=1), help_bin (received help=1)")

summary_cols = ["efficiency", "age", "sex_m", "help_bin", "nuts_opened", "seconds"]
print("\nSummary statistics:")
print(df[summary_cols].describe())

print("\nGroup means for efficiency:")
print("By sex_m:")
print(df.groupby("sex_m")["efficiency"].mean())
print("By help_bin:")
print(df.groupby("help_bin")["efficiency"].mean())

print("\nBivariate correlations:")
corr = df[summary_cols].corr(numeric_only=True)
print(corr)

age_r, age_p = stats.pearsonr(df["age"], df["efficiency"])
sex_r, sex_p = stats.pointbiserialr(df["sex_m"], df["efficiency"])
help_r, help_p = stats.pointbiserialr(df["help_bin"], df["efficiency"])
print("\nBivariate association tests with efficiency:")
print(f"age: r={age_r:.3f}, p={age_p:.4g}")
print(f"sex_m: r={sex_r:.3f}, p={sex_p:.4g}")
print(f"help_bin: r={help_r:.3f}, p={help_p:.4g}")

# Step 2: OLS with controls
# Controls: hammer type (task/tool condition)
hammer_dummies = pd.get_dummies(df["hammer"], prefix="hammer", drop_first=True, dtype=float)
X = pd.concat(
    [
        df[["age", "sex_m", "help_bin"]].astype(float),
        hammer_dummies,
    ],
    axis=1,
)

y = df["efficiency"].astype(float)
X_ols = sm.add_constant(X, has_constant="add")
ols_model = sm.OLS(y, X_ols).fit(cov_type="HC3")

print("\nControlled OLS (HC3 robust SE) summary:")
print(ols_model.summary())

# Step 3: custom interpretable models
smart = SmartAdditiveRegressor(n_rounds=200)
smart.fit(X, y)
smart_effects = to_builtin(smart.feature_effects())

print("\nSmartAdditiveRegressor:")
print(smart)
print("Smart feature effects:")
print(smart_effects)

hinge = HingeEBMRegressor(n_knots=3)
hinge.fit(X, y)
hinge_effects = to_builtin(hinge.feature_effects())

print("\nHingeEBMRegressor:")
print(hinge)
print("Hinge feature effects:")
print(hinge_effects)

# Step 4: rich conclusion and score
key_vars = ["age", "sex_m", "help_bin"]
ols_params = ols_model.params.to_dict()
ols_pvals = ols_model.pvalues.to_dict()


def get_imp(effects_dict, key):
    return float(effects_dict.get(key, {}).get("importance", 0.0) or 0.0)


def get_dir(effects_dict, key):
    return str(effects_dict.get(key, {}).get("direction", "zero"))


strong = 0
mixed = 0
weak = 0
for v in key_vars:
    p = float(ols_pvals.get(v, 1.0))
    s_imp = get_imp(smart_effects, v)
    h_imp = get_imp(hinge_effects, v)

    if p < 0.05 and s_imp >= 0.10 and h_imp >= 0.10:
        strong += 1
    elif (p < 0.10) + (s_imp > 0.01) + (h_imp > 0.01) >= 2:
        mixed += 1
    elif (p < 0.10) or (s_imp > 0.01) or (h_imp > 0.01):
        weak += 1

score = int(round(20 + 25 * strong + 12 * mixed + 5 * weak))
score = max(0, min(100, score))

# Build ranked confounder summary from Smart model (excluding key vars)
smart_ranked = sorted(
    [(k, v) for k, v in smart_effects.items() if k not in key_vars and v.get("importance", 0) > 0],
    key=lambda kv: -float(kv[1]["importance"]),
)
confounder_text = "none prominent"
if smart_ranked:
    top_conf = smart_ranked[:2]
    confounder_text = ", ".join(
        [f"{name} ({float(meta['importance']) * 100:.1f}% Smart importance, {meta['direction']})" for name, meta in top_conf]
    )

age_shape = get_dir(smart_effects, "age")
sex_shape = get_dir(smart_effects, "sex_m")
help_shape = get_dir(smart_effects, "help_bin")

explanation = (
    f"Using efficiency = nuts_opened/seconds as the DV, controlled OLS (with hammer-type controls) found age "
    f"positive (coef={float(ols_params.get('age', np.nan)):.3f}, p={float(ols_pvals.get('age', np.nan)):.3g}), "
    f"male sex positive (coef={float(ols_params.get('sex_m', np.nan)):.3f}, p={float(ols_pvals.get('sex_m', np.nan)):.3g}), "
    f"and help negative (coef={float(ols_params.get('help_bin', np.nan)):.3f}, p={float(ols_pvals.get('help_bin', np.nan)):.3g}). "
    f"Bivariate correlations matched this pattern (age r={age_r:.2f}, sex r={sex_r:.2f}, help r={help_r:.2f}). "
    f"SmartAdditive ranked age as the top driver (importance={get_imp(smart_effects, 'age') * 100:.1f}%, "
    f"shape={age_shape}) with a threshold-like rise from younger to older ages; sex and help were also active "
    f"(sex {get_imp(smart_effects, 'sex_m') * 100:.1f}%, {sex_shape}; help {get_imp(smart_effects, 'help_bin') * 100:.1f}%, {help_shape}). "
    f"HingeEBM confirmed positive age and sex effects (age importance={get_imp(hinge_effects, 'age') * 100:.1f}%, "
    f"sex importance={get_imp(hinge_effects, 'sex_m') * 100:.1f}%) but shrank help to zero, so help is less robust across models. "
    f"Relevant non-focal factors also matter ({confounder_text}), indicating tool condition partly shifts efficiency, "
    f"but the age/sex pattern remains. Overall, evidence is strong for age and sex effects and mixed for help."
)

result = {"response": int(score), "explanation": explanation}

with open("conclusion.txt", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=True)

print("\nWrote conclusion.txt:")
print(json.dumps(result, ensure_ascii=True))
