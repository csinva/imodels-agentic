# Background

The goal is to evaluate how well different classes in the `../result_libs/apr-9-main-result/interpretable_regressors_lib` folder perform on (1) various regression tasks and (2) on interpretability metrics. The original results are stored in the `original_results` folder, particularly the `original_results/overall_results.csv` file.

The original results were obtained by running the `../evolve/interpretable_regressor.py` script. That script makes calls to `../evolve/src/performance_eval.py` and `../evolve/src/interp_eval.py` to compute the performance and interpretability metrics, respectively.

# Your task

Your task is to evaluate all of the classes in the `../result_libs/apr-9-main-result/interpretable_regressors_lib` folder on **new** regression tasks and **new** interpretability metrics and save the results in a new folder called `new_results`. To do so, you should create a new script called `evaluate_new_generalization.py` that performs this evaluation. The new regression tasks and interpretability metrics should be different from those used in the original results. Also include all of the original models from the `../evolve/run_baselines.py` script in your evaluation.

To evaluate predictive performance, write a script to use the following OpenML ids to get datasets: [44065, 44066, 44068, 44069, 45048, 45041, 45043, 45047, 45045, 45046, 44055, 44056, 44059, 44061, 44062, 44063]. (These are from suite 335 in <https://github.com/LeoGrin/tabular-benchmark>, 1 dataset is removed for overlapping with training (abalone).)

To evaluate interpretability write a script that makes minor variations of every test in the original interpretability metrics, e.g. use slightly different synthetic inputs.

Run them all and save / visualize the results into the `new_results` folder, following the same format as the original results. Try to import and reuse functions from previous scripts whenever possible rather than replicating code. Use the `original_results/overall_results.csv` file as a template for how to format your new results.

## Followup

Carefully read the 6 description of the tests in `../paper-imodels-agentic/content.tex` methods section. The code for the prior 43 tests is available in `../evolve/src/interp_eval.py`. In this folder, you wrote and ran experiments to evaluate the generalization of the evolved models to minor variations of the original interpretability tests. Your task is to replace these minor variation tests with a new set of interpretability tests that are substantially different from the original ones, ensuring they still measure the same cognitive operations but with different synthetic inputs or problem formulations. You should implement 157 new tests, meeting the following criteria:

- Tests should all fall into the original 6 categories of cognitive operations: feature attribution, point simulation, sensitivity analysis, counterfactual reasoning, structural understanding, and complex function simulation
- The number of tests should be more balanced across these 6 categories
- No test should be exactly the same as the original tests, e.g. "most important feature"
- Most tests will be similar to tests from the original set
- No test should be biased against a particular form of a model, e.g. "nonlinear threshold identification" unfairly penalizes linear models that have no nonlinear threshold
- No test can have an answer that is unachievable for a model, e.g. "counterfactual target" must ensure that the counterfactual is achievable (by evaluating the model at a different input value)

Once you have written the tests, write and run a script to run the tests on all the baseline models and all the models in each folder of `../result_libs`. Save an `interpretability_results_test.csv` and a `overall_results_test.csv` file in each folder of `../result_libs`, and use them to save an `interpretability_vs_performance_test.png` into each folder.

Next, update the paper with these changes, especially Table A6 and Fig 4a. For Fig 4a, include models from multiple different runs.

Finally, delete any code and results related to the previous minor variations of tests you had run (do not delete code related to testing with a different evaluator, i.e. gpt-5.4).
