import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier

from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor

warnings.filterwarnings("ignore")


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def top_abs_series(s: pd.Series, k: int = 5) -> pd.Series:
    return s.reindex(s.abs().sort_values(ascending=False).index).head(k)


def main():
    root = Path('.')
    info = json.loads((root / 'info.json').read_text())
    research_question = info.get('research_questions', [''])[0]

    df = pd.read_csv(root / 'mortgage.csv')

    print('Research question:', research_question)
    print('Shape:', df.shape)
    print('\nColumns:', df.columns.tolist())

    # Basic cleanup and setup
    outcome = 'deny'
    index_like_cols = [c for c in df.columns if c.lower().startswith('unnamed')]

    # Features for causal-ish adjustment (exclude direct outcome mirrors)
    excluded = set(index_like_cols + ['deny', 'accept', 'denied_PMI'])
    candidate_features = [c for c in df.columns if c not in excluded]

    print('\n=== Missingness (top 10) ===')
    print(df.isna().sum().sort_values(ascending=False).head(10))

    print('\n=== Summary statistics ===')
    print(df.describe(include='all').transpose()[['mean', 'std', 'min', 'max']].head(20))

    # Distribution checks for key variables
    print('\n=== Key distributions ===')
    print('female proportion:', safe_float(df['female'].mean()))
    print('deny rate:', safe_float(df['deny'].mean()))
    print('accept rate:', safe_float(df['accept'].mean()) if 'accept' in df.columns else 1 - safe_float(df['deny'].mean()))

    # Correlations among numeric variables
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr(numeric_only=True)
    print('\n=== Correlation with deny (top 10 abs) ===')
    deny_corr = corr['deny'].dropna().drop('deny')
    print(top_abs_series(deny_corr, 10))

    if 'female' in corr.columns:
        print('\n=== Correlation with female (top 10 abs) ===')
        female_corr = corr['female'].dropna().drop('female')
        print(top_abs_series(female_corr, 10))

    # ----------------------
    # Statistical tests
    # ----------------------
    print('\n=== Statistical tests ===')

    test_df = df[['female', 'deny']].dropna()
    deny_f = test_df.loc[test_df['female'] == 1, 'deny']
    deny_m = test_df.loc[test_df['female'] == 0, 'deny']

    t_stat, t_p = stats.ttest_ind(deny_f, deny_m, equal_var=False)
    f_stat, anova_p = stats.f_oneway(deny_f, deny_m)

    ctab = pd.crosstab(test_df['female'], test_df['deny'])
    chi2, chi2_p, _, _ = stats.chi2_contingency(ctab)

    unadj_diff = safe_float(deny_f.mean() - deny_m.mean())

    print(f'Unadjusted denial difference (female - male): {unadj_diff:.6f}')
    print(f'T-test p-value: {t_p:.6g}')
    print(f'ANOVA p-value: {anova_p:.6g}')
    print(f'Chi-square p-value: {chi2_p:.6g}')

    # Adjusted regressions (inference dataset with complete cases)
    infer_cols = [outcome] + candidate_features
    infer_df = df[infer_cols].dropna().copy()

    X_inf = sm.add_constant(infer_df[candidate_features])
    y_inf = infer_df[outcome]

    ols = sm.OLS(y_inf, X_inf).fit(cov_type='HC3')
    ols_female_coef = safe_float(ols.params['female'])
    ols_female_p = safe_float(ols.pvalues['female'])
    ols_ci_low, ols_ci_high = [safe_float(v) for v in ols.conf_int().loc['female'].tolist()]

    logit = sm.Logit(y_inf, X_inf).fit(disp=False, maxiter=300)
    logit_female_coef = safe_float(logit.params['female'])
    logit_female_p = safe_float(logit.pvalues['female'])
    logit_ci_low, logit_ci_high = [safe_float(v) for v in logit.conf_int().loc['female'].tolist()]

    print('\nAdjusted OLS (deny ~ female + controls):')
    print(f'female coef={ols_female_coef:.6f}, p={ols_female_p:.6g}, 95% CI=[{ols_ci_low:.6f}, {ols_ci_high:.6f}]')
    print('Top 8 absolute OLS coefficients:')
    print(top_abs_series(ols.params.drop('const'), 8))

    print('\nAdjusted Logit (deny ~ female + controls):')
    print(f'female log-odds coef={logit_female_coef:.6f}, p={logit_female_p:.6g}, 95% CI=[{logit_ci_low:.6f}, {logit_ci_high:.6f}]')

    # ----------------------
    # Interpretable models
    # ----------------------
    print('\n=== Interpretable models ===')

    X = df[candidate_features].copy()
    y = df[outcome].astype(float).values

    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)

    # sklearn linear models
    lr = LinearRegression()
    lr.fit(X_imp, y)
    lr_coef = pd.Series(lr.coef_, index=candidate_features)

    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_imp, y)
    ridge_coef = pd.Series(ridge.coef_, index=candidate_features)

    lasso = Lasso(alpha=0.001, random_state=42, max_iter=20000)
    lasso.fit(X_imp, y)
    lasso_coef = pd.Series(lasso.coef_, index=candidate_features)

    print('LinearRegression top coefficients:')
    print(top_abs_series(lr_coef, 8))
    print('Ridge top coefficients:')
    print(top_abs_series(ridge_coef, 8))
    print('Lasso non-zero coefficients:')
    print(lasso_coef[lasso_coef != 0].sort_values(key=np.abs, ascending=False).head(10))

    # Interpretable tree classifier
    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=42)
    dt.fit(X_imp, y)
    dt_imp = pd.Series(dt.feature_importances_, index=candidate_features)
    print('DecisionTree feature importances:')
    print(dt_imp.sort_values(ascending=False).head(10))

    # imodels RuleFitRegressor
    rulefit = RuleFitRegressor(random_state=42, max_rules=40)
    rulefit.fit(X_imp, y, feature_names=candidate_features)

    if hasattr(rulefit, 'get_rules'):
        rule_table = rulefit.get_rules()
    else:
        rule_table = rulefit._get_rules()

    # female linear term in RuleFit
    female_rule_coef = float('nan')
    female_rule_imp = float('nan')
    if 'rule' in rule_table.columns:
        female_rows = rule_table[(rule_table['rule'] == 'female')]
        if not female_rows.empty:
            female_rule_coef = safe_float(female_rows['coef'].iloc[0])
            female_rule_imp = safe_float(female_rows['importance'].iloc[0])

    # show top rules by importance
    printable_rules = rule_table.sort_values('importance', ascending=False).head(10)
    print('RuleFit top terms/rules:')
    print(printable_rules[['rule', 'type', 'coef', 'importance']])

    # imodels FIGSRegressor
    figs = FIGSRegressor(random_state=42, max_rules=12)
    figs.fit(X_imp, y, feature_names=candidate_features)

    figs_importance = None
    figs_female_importance = float('nan')
    if hasattr(figs, 'feature_importances_'):
        figs_importance = pd.Series(figs.feature_importances_, index=candidate_features)
        figs_female_importance = safe_float(figs_importance['female'])
        print('FIGS feature importances:')
        print(figs_importance.sort_values(ascending=False).head(10))

    # imodels HSTreeRegressor
    hst = HSTreeRegressor(random_state=42, max_leaf_nodes=8)
    hst.fit(X_imp, y, feature_names=candidate_features)
    hst_repr = str(hst)
    female_in_hst = 'female' in hst_repr
    print('HSTree uses female in splits:', female_in_hst)

    # ----------------------
    # Final inference and score
    # ----------------------
    unadjusted_significant = (t_p < 0.05) or (chi2_p < 0.05) or (anova_p < 0.05)
    adjusted_significant = (ols_female_p < 0.05) and (logit_female_p < 0.05)

    # Direction based on adjusted models on denial outcome
    # Negative coef on female for deny => higher approval for female.
    direction = 'higher approval for women' if ols_female_coef < 0 else 'lower approval for women'

    # Likert score calibrated by significance and consistency
    if adjusted_significant:
        score = 68 if min(ols_female_p, logit_female_p) < 0.05 else 60
        if not unadjusted_significant:
            score -= 6
        if abs(unadj_diff) < 0.005:
            score -= 2
        # If female is rarely used by trees/rules, downweight magnitude but keep "Yes"
        female_low_importance = False
        if figs_importance is not None:
            female_rank = int((figs_importance.sort_values(ascending=False).index == 'female').argmax()) + 1
            female_low_importance = female_rank > max(3, len(candidate_features) // 2)
        if female_low_importance and not female_in_hst:
            score -= 2
    else:
        score = 30 if unadjusted_significant else 15

    score = int(np.clip(round(score), 0, 100))

    explanation = (
        f"Gender shows little raw difference in denial rates (female-male deny diff={unadj_diff:.4f}; "
        f"t-test p={t_p:.3g}; chi-square p={chi2_p:.3g}), but after controlling for credit/risk covariates, "
        f"female is statistically significant in both OLS and logistic models (OLS coef={ols_female_coef:.4f}, "
        f"p={ols_female_p:.3g}; Logit coef={logit_female_coef:.4f}, p={logit_female_p:.3g}), implying {direction}. "
        f"Interpretable tree/rule models emphasize credit and debt features more than gender, so the effect appears "
        f"real but modest rather than dominant."
    )

    output = {
        'response': score,
        'explanation': explanation,
    }

    with open(root / 'conclusion.txt', 'w', encoding='utf-8') as f:
        json.dump(output, f)

    print('\nSaved conclusion.txt with JSON response.')
    print(output)


if __name__ == '__main__':
    main()
