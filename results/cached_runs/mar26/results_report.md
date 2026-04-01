# Mar 26 Experiment Results Report

## 1. Interpretability Evaluation Tests

The interpretability evaluation (`src/interp_eval.py`) tests whether an LLM (GPT-4o) can answer questions about a model's behavior by reading its **string representation** alone. Each test:

1. Generates a synthetic dataset with known ground truth (e.g., `y = 10*x0 + noise`)
2. Fits the model to that data
3. Converts the fitted model to a human-readable string (e.g., decision tree rules, linear coefficients, GAM partial-effect tables)
4. Asks GPT-4o a specific question about the model's behavior (e.g., "Which feature is most important?")
5. Checks the LLM's response against ground truth with a tolerance

There are 4 test suites totaling 25 tests: **Standard** (8), **Hard** (5), **Insight** (6), and **Discrimination** (6).

### Detailed Example: `test_point_prediction`

**Setup:** Generate data with `y = 5.0 * x0 + noise` (3 features, 300 samples). Fit the model.

**Ground truth:** Compute the model's actual prediction for `x0=2.0, x1=0.0, x2=0.0`.

**Prompt sent to GPT-4o:**
```
Here is a trained regression model:

<model string representation>

What does this model predict for the input x0=2.0, x1=0.0, x2=0.0?
Answer with just a single number (e.g., '10.5').
```

**Grading:** Extract numbers from the LLM response. Pass if `|llm_answer - true_pred| < max(|true_pred| * 0.25, 1.5)`.

For example, an OLS model would produce the string `y = 4.9856*x0 + 0.0311*x1 + ... + 0.0122`, and GPT-4o can easily compute `4.9856*2.0 + 0.0122 = 10.06`. A decision tree provides explicit paths to trace. An MLP's weight matrices are much harder for the LLM to mentally multiply through.

---

## 2. Test Descriptions and Pass Rates

| Test | Suite | Short Description | Detailed Description | Pass Rate |
|------|-------|-------------------|----------------------|-----------|
| most_important_feature | Standard | Identify top feature | Data has one dominant feature (coef=10). Ask LLM which single feature matters most. | 58/60 (97%) |
| point_prediction | Standard | Predict at a point | Compute model output for x0=2.0, others=0. Tests if LLM can trace the model. | 50/60 (83%) |
| direction_of_change | Standard | Quantify change direction | How much does prediction change when x0 goes from 0 to 1? Tests marginal effect reading. | 48/60 (80%) |
| feature_ranking | Standard | Rank top-3 features | Data has coefs [5,3,1.5,0,0]. Ask LLM to rank features by importance. | 58/60 (97%) |
| threshold_identification | Standard | Find decision threshold | Step-function data with threshold at x0=0.5. Ask LLM to identify the threshold. | 50/60 (83%) |
| irrelevant_features | Standard | Spot zero-effect features | One active feature among 5. Ask LLM which features have no effect. | 52/60 (87%) |
| sign_of_effect | Standard | Quantify signed effect | Data has y=5*x0-5*x1. Ask LLM the effect of +1 unit change in x1. | 43/60 (72%) |
| counterfactual_prediction | Standard | Counterfactual prediction | Given pred at x0=1, ask what model predicts at x0=3. Tests mental simulation. | 17/60 (28%) |
| hard_all_features_active | Hard | Simulate multi-feature input | All 3 features active at non-zero values. Harder to trace than single-feature. | 12/60 (20%) |
| hard_pairwise_anti_intuitive | Hard | Compare two samples | Compare predictions where dominant feature is higher in A but B has higher x1. | 6/60 (10%) |
| hard_quantitative_sensitivity | Hard | Quantify sensitivity | How much does pred change when x0 goes from 0.5 to 2.5? Requires precise reading. | 51/60 (85%) |
| hard_mixed_sign_goes_negative | Hard | Predict negative output | Mixed-sign coefficients [3,-2,1]. Predict at point where result is negative. | 37/60 (62%) |
| hard_two_feature_perturbation | Hard | Two features change at once | Given baseline pred, compute new pred when both x0 and x1 change simultaneously. | 11/60 (18%) |
| insight_simulatability | Insight | Full 4-feature simulation | Predict at x0=1, x1=2, x2=0.5, x3=-0.5. Tests end-to-end simulatability. | 39/60 (65%) |
| insight_sparse_feature_set | Insight | Identify active features | 10 features, only x0 and x1 matter. Ask LLM to list meaningful features. | 50/60 (83%) |
| insight_nonlinear_threshold | Insight | Detect hockey-stick shape | y=3*max(0,x0). Ask LLM to identify the threshold below which x0 has no effect. | 43/60 (72%) |
| insight_nonlinear_direction | Insight | Predict on nonlinear data | Hockey-stick data, predict at x0=2.0. Tests if model captures nonlinearity. | 41/60 (68%) |
| insight_counterfactual_target | Insight | Inverse prediction (solve for x0) | Given pred=P at x0=1, find x0 that yields P+8. Tests algebraic reasoning from model string. | 37/60 (62%) |
| insight_decision_region | Insight | Find decision boundary | For what x0 does model predict above 6.0? Tests boundary identification. | 50/60 (83%) |
| discrim_simulate_all_active | Discrim | Simulate 5-feature sample | All 5 features active at non-round values. Designed to separate interpretable from black-box. | 35/60 (58%) |
| discrim_compactness | Discrim | Judge model compactness | Ask if model can be computed in <=10 operations. Sparse models pass; MLPs/ensembles fail. | 44/60 (73%) |
| discrim_dominant_feature_sample | Discrim | Identify dominant feature for sample | Which feature contributes most to a specific sample? Coef=7, value=2.0 for x0. | 60/60 (100%) |
| discrim_unit_sensitivity | Discrim | Exact unit sensitivity | Exact change when x0 goes 0->1. Tight tolerance (10%) rewards readable representations. | 43/60 (72%) |
| discrim_predict_above_threshold | Discrim | Predict above threshold | Predict at x0=2.0 on step-function data (threshold=1.0). Tests precise simulation. | 11/57 (19%) |
| discrim_predict_below_threshold | Discrim | Predict below threshold | Predict at x0=-0.5 on step-function data (threshold=1.0). Easier since output ~ 0. | 43/57 (75%) |

### Which Models Passed Each Test

| Test | Models That Passed |
|------|-------------------|
| most_important_feature | All except EBM, TabPFN |
| point_prediction | Most models; failed: PyGAM, RF, MLP, RuleFit, EBM, TabPFN, AdditiveTree, HingeAdaptive, HingeDebiasGBM, ModelTree |
| direction_of_change | Most Hinge variants + baselines; failed: DT_mini, DT_large, RF, MLP, FIGS_mini, HSTree_mini, DecisionTreeSimple, HingeAdaptive, ModelTree, HingeBlend, HingeTreeCombo |
| feature_ranking | All except EBM, TabPFN |
| threshold_identification | Most models; failed: PyGAM, DT_large, LassoCV, RidgeCV, MLP, EBM, TabPFN, BinnedAdditive, HingeLinearV2, HingeTreeCombo |
| irrelevant_features | Most models; failed: DT_mini, DT_large, FIGS_mini, HSTree_mini, EBM, TabPFN, DecisionTreeSimple, BinnedAdditive |
| sign_of_effect | Most Hinge variants + some baselines; failed: DT_large, RF, GBM, MLP, FIGS_large, HSTree_mini, EBM, TabPFN, DecisionTreeSimple, AdditiveTree, BinnedAdditive, ScoringReg, RuleLinear, HingeAdaptive, ModelTree, HingeTreeCombo, HingeGBMv3 |
| counterfactual_prediction | Only 17/60: DT_mini/large, OLS, LassoCV, RidgeCV, RF, GBM, MLP, FIGS_mini/large, HSTree_mini/large, DecisionTreeSimple, BinnedAdditive, RuleLinear, HingeGBMv3, HingeBagGBM |
| hard_all_features_active | Only 12/60: DT_mini/large, OLS, RidgeCV, GBM, RuleFit, HSTree_mini/large, DecisionTreeSimple, RuleLinear, HingePosOnly, HingeBlend |
| hard_pairwise_anti_intuitive | Only 6/60: DT_mini, GBM, HSTree_mini/large, BinnedAdditive, HingeAdaptive |
| hard_quantitative_sensitivity | 51/60; failed: RF, GBM, MLP, FIGS_mini, RuleFit, EBM, TabPFN, ModelTree, HingeAdaptive |
| hard_mixed_sign_goes_negative | 37/60; most Hinge-GBM variants pass, most non-Hinge baselines fail |
| hard_two_feature_perturbation | Only 11/60: DT_mini/large, OLS, RidgeCV, GBM, FIGS_mini, HSTree_large, ScoringReg, ModelTree, HingeBagGBM, HingeGBM2dp |
| insight_simulatability | 39/60; many Hinge variants + DT/HSTree pass; PyGAM, OLS, LassoCV, RF, MLP, FIGS_mini, EBM, TabPFN fail |
| insight_sparse_feature_set | 50/60; failed: RuleFit, EBM, TabPFN, DecisionTreeSimple (missing), BinnedAdditive (missing), various Hinge variants |
| insight_nonlinear_threshold | 43/60; OLS, LassoCV, RidgeCV, RF, GBM, RuleFit, HSTree_large fail (linear models can't capture hockey-stick) |
| insight_nonlinear_direction | 41/60; RF, MLP, FIGS_mini, HSTree_mini, EBM, TabPFN, various Hinge variants fail |
| insight_counterfactual_target | 37/60; primarily Hinge variants + OLS pass; DTs, PyGAM, RF, GBM, MLP, FIGS, EBM, TabPFN fail |
| insight_decision_region | 50/60; most pass; failed: LassoCV, RF, MLP, FIGS (some), EBM, TabPFN, HingeAdaptive, ModelTree, HingeBagGBM |
| discrim_simulate_all_active | 35/60; Hinge variants dominate; DTs, PyGAM, OLS, RidgeCV, RF, GBM, MLP, FIGS, HSTree, EBM, TabPFN fail |
| discrim_compactness | 44/60; DTs, RF, GBM, MLP, FIGS_large, RuleFit, HSTree_large, EBM, TabPFN, AdditiveTree, ModelTree, various fail |
| discrim_dominant_feature_sample | **100%** — all 60 models pass |
| discrim_unit_sensitivity | 43/60; DTs, PyGAM, RF, GBM, MLP, FIGS_mini, HSTree_mini, EBM, TabPFN fail |
| discrim_predict_above_threshold | Only 11/57: PyGAM, DT_mini, GBM, FIGS_mini/large, HSTree_mini/large, DecisionTreeSimple, BinnedAdditive, ModelTree, HingeBlend |
| discrim_predict_below_threshold | 43/57; most Hinge variants + baselines pass |

---

## 3. Model String Visualizations

Below are the string representations of three different model types when fit to the synthetic data from the `most_important_feature` test (`y = 10*x0 + noise`, 5 features, 300 samples).

### Decision Tree (max_depth=3)

```
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

**Discussion:** The shallow decision tree is **fully simulatable** -- an LLM can trace any input through the if/else branches to reach a leaf value. It clearly shows that only `x0` matters (no other features appear in splits). However, it approximates the linear relationship as a piecewise-constant function with only 8 bins, losing precision. The LLM can easily answer "what does the model predict for x0=2.0?" by following: `x0 > -0.07 -> x0 > 1.16 -> x0 > 1.80 -> value = 20.61`. This explains why DT_mini scores 76% on interpretability tests despite weaker predictive performance (rank 12.67).

### OLS Linear Regression

```
y = 9.9876*x0 + 0.0490*x1 + 0.0311*x2 + 0.0406*x3 + -0.0084*x4 + 0.0122

Coefficients:
  x0: 9.9876
  x1: 0.0490
  x2: 0.0311
  x3: 0.0406
  x4: -0.0084
  intercept: 0.0122
```

**Discussion:** The linear model provides the most **compact and algebraically manipulable** representation. The LLM can compute any prediction by substitution (e.g., `9.9876*2.0 + 0.0122 = 20.00`), identify irrelevant features (x1-x4 have near-zero coefficients), determine signs of effects, and solve inverse problems (what x0 gives target y?). This is why OLS scores 72% on interpretability despite being the simplest model. It fails on tests requiring nonlinear reasoning (hockey-stick threshold detection) since a linear model cannot represent `max(0, x0)`.

### Random Forest (100 trees, max_depth=5)

```
Feature Importances:
  x0: 0.9989
  x2: 0.0004
  x1: 0.0003
  x4: 0.0003
  x3: 0.0001

First estimator tree (depth <= 3):
|--- x0 <= -0.07
|   |--- x0 <= -1.13
|   |   |--- x0 <= -2.03
|   |   |   |--- x0 <= -2.33
|   |   |   |   |--- truncated branch of depth 2
|   |   |   |--- x0 >  -2.33
|   |   |   |   |--- value: [-20.72]
|   |   |--- x0 >  -2.03
|   |   |   |--- x0 <= -1.49
|   |   |   |   |--- truncated branch of depth 2
|   |   |   |--- x0 >  -1.49
|   |   |   |   |--- truncated branch of depth 2
    ... (99 more trees, many truncated branches)
```

**Discussion:** The Random Forest's string representation exposes **aggregate feature importances** (correctly showing x0 dominates at 0.999) but only one of 100 trees, truncated at depth 3. The LLM can identify the most important feature and rank features, but **cannot simulate a prediction** -- it would need to average predictions across all 100 trees, each with up to 32 leaves. This explains RF's very low 28% interpretability score despite strong predictive performance (rank 6.30). The model is effectively a black box from the string representation.

---

## 4. Key Takeaways

- **Easiest tests** (>90% pass rate): `most_important_feature` and `feature_ranking` -- even black-box models expose enough structure (feature importances) for these.
- **Hardest tests** (<20% pass rate): `hard_pairwise_anti_intuitive` and `hard_two_feature_perturbation` -- these require precise multi-step arithmetic that even clear model strings make difficult.
- **Best discriminating tests**: `discrim_simulate_all_active` and `insight_counterfactual_target` -- these cleanly separate interpretable models (Hinge variants, linear models) from black-box ones (RF, MLP, TabPFN, EBM).
- **Hinge-family models** consistently achieve 80% interpretability while maintaining competitive performance (ranks 4-8), representing the best interpretability-performance tradeoff.
- **EBM and TabPFN** score only 8% on interpretability despite being the top-2 performers (ranks 4.60 and 3.56), because their string representations are not human-simulatable.
