# Mar 30 Results Report

## 1. Interpretability Test Suite Overview

The interpretability evaluation (`src/interp_eval.py`) measures whether an LLM (GPT-4o) can extract meaningful, actionable information from a model's **string representation**. Each test:

1. Generates **synthetic data** with a known ground truth (e.g., `y = 10*x0 + noise`)
2. Fits the model to that data
3. Converts the fitted model to a **human-readable string** (e.g., tree text, coefficient table, GAM partial effects)
4. Asks GPT-4o a specific question about the model (e.g., "What does this model predict for x0=2.0?")
5. Checks whether the LLM's answer matches the ground truth within a tolerance

There are 4 test suites: **Standard** (8 tests), **Hard** (5 tests), **Insight** (6 tests), and **Discrim** (10 tests) = 29 tests total.

### Detailed Example: `test_point_prediction`

**Setup:** Generate synthetic data with `y = 5*x0 + noise` (3 features, 300 samples).

```python
X, y = _single_feature_data(n_features=3, true_feature=0, coef=5.0)
m = clone(model); m.fit(X, y)
true_pred = m.predict([[2.0, 0.0, 0.0]])[0]  # e.g., ~10.05
```

**Model string (example for OLS):**
```
OLS Linear Regression:  y = 9.9729*x0 + -0.0104*x1 + 0.0205*x2 + 0.0122
```

**LLM prompt:**
> Here is a trained regression model:
> [model string above]
> What does this model predict for the input x0=2.0, x1=0.0, x2=0.0? Answer with just a single number.

**LLM response:** `10.0600`

**Grading:** Pass if `|llm_answer - true_pred| < max(|true_pred| * 0.25, 1.5)`. Here `|10.06 - 10.06| < 2.5` --> **PASS**.

This test measures **simulatability**: can someone reading the model string compute a prediction? Linear models pass trivially; MLPs and Random Forests typically fail because the string representation doesn't expose enough information to trace a prediction.

---

## 2. Test-by-Test Results

Each row shows one interpretability test. Pass rate is computed over all 26 model variants evaluated. "Models Passed" and "Models Failed" list the model names.

| Test | Short Description | Detailed Description | Pass Rate | Models Passed | Models Failed |
|------|------------------|---------------------|-----------|---------------|---------------|
| `most_important_feature` | Identify dominant feature | Fit on `y=10*x0+noise` (5 features). Ask which single feature matters most. Check if "x0" appears in response. | 92% | All except EBM, TabPFN | EBM, TabPFN |
| `feature_ranking` | Rank top-3 features | Fit on `y=5*x0+3*x1+1.5*x2`. Ask for top-3 features ranked. Check x0 appears before x1. | 92% | All except EBM, TabPFN | EBM, TabPFN |
| `point_prediction` | Predict at a point | Fit on `y=5*x0+noise`. Ask prediction at x0=2.0. Check within 25% tolerance. | 72% | Most interpretable models | EBM, MLP, MiniGBM, PyGAM, RF, RuleFit, TabPFN |
| `direction_of_change` | Quantify feature effect | Fit on `y=8*x0+noise`. Ask how prediction changes when x0 goes 0->1. | 52% | FIGS_large, GBM, HSTreeCV, HSTree_large, HierShrinkTree, LassoCV, OLS, PyGAM, RidgeCV, RuleFit, SparseInteraction, SparseLinear, SplineGAM | 12 models including DTs, MLP, RF |
| `threshold_identification` | Find decision boundary | Fit on threshold data (y jumps at x0=0.5). Ask for the threshold value. | 58% | 15 models including DT_mini, FIGS, GBM, HSTree variants | DT_large, EBM, HingeGBM, LassoCV, MLP, PyGAM, RidgeCV, others |
| `irrelevant_features` | Identify unused features | Fit on `y=10*x0+noise` (5 features). Ask which features have no effect. Check >=2 of x1-x4 listed. | 69% | 18 models | DT_large, DT_mini, DecisionTreeSimple, EBM, FIGS_mini, HSTree_mini, HierShrinkTree, TabPFN |
| `sign_of_effect` | Detect negative coefficient | Fit on `y=5*x0-5*x1+noise`. Ask how prediction changes with x1+1. Check sign and magnitude. | 52% | DT_mini, FIGS_mini, HSTree_large, HierShrinkTree, LassoCV, LinearTreeHybrid, OLS, PyGAM, RidgeCV, RuleFit, SparseInteraction, SparseLinear, SplineGAM | 12 models |
| `counterfactual_prediction` | Predict at new point | Fit on `y=4*x0+noise`. Given pred at x0=1, ask pred at x0=3. | 68% | 17 models | EBM, LinearTreeHybrid, MiniGBM, PyGAM, RuleFit, SparseLinear, SplineGAM, TabPFN |
| `hard_all_features_active` | Multi-feature prediction | Fit on `y=3*x0+2*x1+x2`. Predict at x0=1.7, x1=0.8, x2=-0.5 (tighter 15% tol). | 56% | DTs, GBM, HSTree variants, OLS, RidgeCV, RuleFit, SparseLinear, SplineGAM | FIGS, LassoCV, MLP, PyGAM, RF, others |
| `hard_pairwise_anti_intuitive` | Compare two samples | Fit on `y=5*x0+3*x1`. Compare pred(x0=2,x1=0.1) vs pred(x0=0.5,x1=3.3). | 20% | DT_mini, GBM, HSTree_large, HSTree_mini, SparseInteraction | 20 models |
| `hard_quantitative_sensitivity` | Quantify sensitivity | Fit on `y=4*x0+noise`. Ask prediction change from x0=0.5 to x0=2.5 (15% tol). | 60% | 15 models | EBM, FIGS_mini, GBM, HSTreeCV, HierShrinkTree, MLP, MiniGBM, RF, RuleFit, TabPFN |
| `hard_mixed_sign_goes_negative` | Negative prediction | Fit on `y=3*x0-2*x1+x2`. Predict at x0=1,x1=2.5,x2=1 (expect ~negative). | 40% | DT_large, DT_mini, GBM, HSTreeCV, HSTree_large, HSTree_mini, MiniGBM, PyGAM, RuleFit, SparseLinear | 15 models |
| `hard_two_feature_perturbation` | Two-feature change | Fit on `y=3*x0+2*x1`. Given pred at origin, ask pred when x0->2, x1->1.5. | 36% | DT_large, DT_mini, FIGS_mini, GBM, HSTreeCV, HSTree_large, HierShrinkTree, OLS, RidgeCV | 16 models |
| `insight_simulatability` | Full simulation | Fit on `y=5*x0+3*x1`. Predict at x0=1,x1=2,x2=0.5,x3=-0.5 (15% tol). | 35% | DT_large, DT_mini, DecisionTreeSimple, HSTreeCV, HSTree_large, HSTree_mini, HierShrinkTree, RidgeCV, SplineGAM | 17 models |
| `insight_sparse_feature_set` | Identify active features | Fit on 10 features, only x0,x1 matter. Ask which features contribute. | 65% | 17 models | AdditiveTree, EBM, HSTreeCV, HierShrinkTree, LinearTreeHybrid, RuleFit, SparseInteraction, SparseLinear, TabPFN |
| `insight_nonlinear_threshold` | Hockey stick threshold | Fit on `y=3*max(0,x0)+noise`. Ask threshold below which x0 has no effect. | 62% | 16 models including DTs, FIGS, PyGAM, MLP, TabPFN | GBM, HSTree_large, HingeGBM, LassoCV, OLS, RF, RidgeCV, others |
| `insight_nonlinear_direction` | Nonlinear prediction | Fit on hockey-stick data. Predict at x0=2.0 (20% tol). | 58% | 15 models | EBM, FIGS_mini, HSTree_mini, HingeGBM, LinearTreeHybrid, MLP, MiniGBM, RF, RuleFit, SparseInteraction, TabPFN |
| `insight_counterfactual_target` | Inverse prediction | Fit on `y=4*x0+2*x1`. Given pred at x0=1, find x0 for pred+8 (hardest test). | 8% | OLS, SplineGAM | 24 models |
| `insight_decision_region` | Find prediction boundary | Fit on `y=4*x0+noise`. Find x0 where pred crosses 6.0. | 73% | 19 models | EBM, HingeGBM, LassoCV, LinearTreeHybrid, MLP, RF, TabPFN |
| `discrim_simulate_all_active` | Simulate 5-feature point | Fit on 5-feature linear data, predict at non-round values (20% tol). | 12% | LassoCV, MiniGBM, SparseLinear | 22 models |
| `discrim_compactness` | Model fits in 10 ops? | Ask LLM if model can be computed in <=10 operations. Yes/no. | 35% | FIGS_mini, HSTree_mini, HingeGBM, LassoCV, LinearTreeHybrid, OLS, PyGAM, RidgeCV, SplineGAM | 17 models |
| `discrim_dominant_feature_sample` | Per-sample feature attribution | Ask which feature contributes most to a specific sample (x0=2 dominates). | 100% | All 26 models | (none) |
| `discrim_unit_sensitivity` | Exact unit change | Ask exact prediction delta when x0 goes 0->1 (tight 10% tol). | 40% | FIGS_large, HSTree_large, LassoCV, LinearTreeHybrid, OLS, RidgeCV, RuleFit, SparseInteraction, SparseLinear, SplineGAM | 16 models |
| `discrim_predict_above_threshold` | Predict above step | Fit on step-function data (threshold=1.0). Predict at x0=2.0 (above). | 55% | 12 models | 10 models including RF, MLP, EBM, TabPFN |
| `discrim_predict_below_threshold` | Predict below step | Same step data. Predict at x0=-0.5 (below threshold). | 68% | 15 models | EBM, FIGS_mini, HSTree_large, LinearTreeHybrid, MLP, RF, TabPFN |
| `discrim_simulate_mixed_sign` | Simulate 6 mixed-sign features | Fit on 6-feature data with +/- coefficients. Predict at specific point. | 32% | DT_mini, DecisionTreeSimple, GBM, HSTreeCV, HSTree_large, HSTree_mini, OLS, RidgeCV | 17 models |
| `discrim_simulate_double_threshold` | Simulate two-step function | Fit on data with jumps at x0=0 and x0=1.5. Predict at x0=0.8. | 56% | 14 models | EBM, FIGS_mini, GBM, MLP, OLS, PyGAM, RF, RuleFit, SparseLinear, SplineGAM, TabPFN |
| `discrim_simulate_additive_nonlinear` | Simulate nonlinear additive | Fit on `y=3*max(0,x0)+2*sin(x1)+x2`. Predict at specific point. | 52% | 13 models (DTs, FIGS, HSTree, LassoCV, OLS, RidgeCV) | 12 models including GBM, MLP, RF, PyGAM |
| `discrim_simulate_interaction` | Simulate interaction model | Fit on `y=3*x0+2*x1+1.5*x0*x1`. Predict at x0=2,x1=1.5. | 48% | DTs, FIGS, HSTree variants, OLS, RidgeCV | AdditiveTree, EBM, GBM, LassoCV, MLP, RF, PyGAM, others |

---

## 3. Model String Visualizations

Below are the string representations of three different model types when fit to the first test's synthetic data (`y = 10*x0 + noise`, 5 features, 300 samples). These illustrate why some models are more interpretable than others.

### 3.1 Decision Tree (max_leaf_nodes=8)

```
Decision Tree Regressor (max_depth=None):
|--- x0 <= -0.07
|   |--- x0 <= -1.10
|   |   |--- x0 <= -1.87
|   |   |   |--- value: [-23.34]
|   |   |--- x0 >  -1.87
|   |   |   |--- value: [-14.23]
|   |--- x0 >  -1.10
|   |   |--- x0 <= -0.53
|   |   |   |--- value: [-7.96]
|   |   |--- x0 >  -0.53
|   |   |   |--- value: [-3.13]
|--- x0 >  -0.07
|   |--- x0 <= 1.16
|   |   |--- x0 <= 0.60
|   |   |   |--- value: [3.18]
|   |   |--- x0 >  0.60
|   |   |   |--- value: [8.33]
|   |--- x0 >  1.16
|   |   |--- x0 <= 1.80
|   |   |   |--- value: [15.12]
|   |   |--- x0 >  1.80
|   |   |   |--- value: [20.61]
```

**Discussion:** The decision tree's string representation is directly simulatable -- an LLM can trace through the if/else branches to arrive at a leaf value. The tree correctly identifies that **only x0 matters** (no other features appear in splits). The 8 leaf nodes create a piecewise-constant approximation of the linear relationship. For `x0=2.0`, you follow: `x0 > -0.07` -> `x0 > 1.16` -> `x0 > 1.80` -> predict **20.61**. This is easy for an LLM to execute, which is why DT_mini achieves a 79% interp pass rate. However, the piecewise-constant nature means predictions are coarse approximations, contributing to weaker performance rank (12.17 avg).

### 3.2 Linear Regression (OLS)

```
OLS Linear Regression:  y = 9.9876*x0 + 0.0490*x1 + 0.0311*x2 + 0.0406*x3 + -0.0084*x4 + 0.0122

Coefficients:
  x0: 9.9876
  x1: 0.0490
  x2: 0.0311
  x3: 0.0406
  x4: -0.0084
  intercept: 0.0122
```

**Discussion:** The linear model is the gold standard for interpretability on linear data -- the entire model is a single equation. For `x0=2.0, x1=0, x2=0`: `y = 9.9876*2 + 0.0122 = 19.99`. The LLM can compute this with basic arithmetic, making point predictions trivial. The near-zero coefficients on x1-x4 clearly show they are irrelevant. OLS achieves 72% interp pass rate and 9.85 avg rank. Its limitations show on nonlinear data (e.g., it fails `insight_nonlinear_threshold` since it cannot represent the hockey-stick shape) and on inverse queries (`insight_counterfactual_target` where it's one of only 2 models that pass, because solving `y = mx + b` for `x` is straightforward).

### 3.3 Random Forest (50 estimators, max_depth=5)

```
Random Forest Regressor -- Feature Importances (higher = more important):
  x0: 0.9987
  x2: 0.0005
  x4: 0.0004
  x1: 0.0003
  x3: 0.0001

First estimator tree (depth <= 3):
|--- x0 <= -0.07
|   |--- x0 <= -1.13
|   |   |--- x0 <= -2.03
|   |   |   |--- ...truncated...
|   |   |--- x0 >  -2.03
|   |   |   |--- ...truncated...
|   |--- x0 >  -1.13
|   |   |--- ...truncated...
|--- x0 >  -0.07
|   |--- x0 <= 0.95
|   |   |--- ...truncated...
|   |--- x0 >  0.95
|   |   |--- ...truncated...
```

**Discussion:** The Random Forest string exposes **feature importances** (correctly showing x0 dominates) and **only the first of 50 trees** (truncated at depth 3). This is fundamentally insufficient for simulation -- predicting a specific value requires averaging predictions across all 50 trees, each with up to 32 leaves. The LLM can answer high-level questions like "which feature is most important?" (pass) but cannot compute "what does this model predict for x0=2.0?" (fail). This explains the RF's 24% interp pass rate despite its strong 5.96 avg performance rank. The RF embodies the classic **accuracy-interpretability tradeoff**: excellent predictions, but the model internals are opaque.

---

## 4. Key Takeaways

The plot below (from `interpretability_vs_performance.png`) shows the Pareto frontier:

- **Black-box models** (TabPFN rank 3.8, EBM rank 4.5) dominate performance but score ~7% on interpretability
- **Tree-based interpretable models** (HSTree_large: rank 9.6, interp 83%) offer the best interpretability while maintaining reasonable performance
- **Linear models** (OLS, RidgeCV: rank ~9.5-9.9, interp 72%) are strong on linear synthetic tests but fail nonlinear ones
- **Evolved models** (SparseInteraction variants: rank 6.1-7.6, interp 48-79%) push the Pareto frontier, trading off between the two objectives
- **HingeGBM** (rank 4.4-5.4, interp 17-83%) represents the most aggressive evolved approach, blending an interpretable polynomial equation with a GBM residual correction

The hardest tests (`insight_counterfactual_target` at 8%, `discrim_simulate_all_active` at 12%) require the LLM to perform multi-step arithmetic or algebraic inversion -- tasks where even the clearest model strings push LLM capabilities.
