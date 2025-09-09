# LLMs Don't Know Their Own Decision Boundaries

This repo contains the code and data used in the paper *"LLMs Don't Know Their Own Decision Boundaries: The Unreliability of Self-Generated Counterfactual Explanations" (EMNLP 2025)*. The paper explores whether LLMs can generate counterfactual explanations that both flip their own answers (**validity**) while making the smallest possible change to the input (**minimality**).

## Repo layout

```
analysis/            Analysis notebooks and placeholders for additional experiments
data/                Local dataset caches
figures/             Notebooks for figures.
results/             Example results produced by the pipeline
src/                 Source code for generation, post‑processing and scoring
    distance_matrices/        Pre-computed distance matrices for all datasets
    helper_functions/         Utility modules
    models_datasets/          Model and dataset configuration files
    main.py                   Main code to run a model on a dataset and collect the counterfactuals
    postprocessing.py         Clean the results to add distances...etc
    scorer.py                 Compute aggregate validity/minimality metrics [Generally use scorer_notebook.ipynb for this]
    scorer_notebook.ipynb     Score the results
    run_main_results.sh       Example script to reproduce the main results (all models on all datasets)
requirements.txt      Python dependencies
```

## Running experiments
1. Install dependencies (Python 3.12):
   ```bash
   pip install -r requirements.txt
   ```
2. Generate model predictions and counterfactual:
   ```bash
   python src/main.py \
       --task_model meta-llama/Meta-Llama-3-8B-Instruct \
       --dataset income \
       --sample_size 2000 \
       --max_tokens 1000 \
       --tensor_parallel_size 1
   ```
   The script uses the model and dataset definitions in `src/models_datasets/` and writes a JSON file to `results/<dataset>/<model>.json`.
3. Post-process the results and compute distances:
   ```bash
   python src/postprocessing.py --filepath results/income/llama3_8B.json --dataset income
   ```
4. Score results using scorer_notebook.ipynb  
5. (Optional) Overwrite the results file with the aggregate scores (useful if storage constraints):
   ```bash
   python src/scorer.py --filepath results/income/llama3_8B.json
   ```

The `run_main_results.sh` script provides a larger set of commands for reproducing the experiments across several models and datasets.

## Notes
- The full results are not provided as they are large. These can be generated using the pipeline above.
- The details of the experiments in analysis will be provided in time.
