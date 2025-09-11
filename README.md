# LLMs Don't Know Their Own Decision Boundaries
*Harry Mayne, Ryan Othniel Kearns, Yushi Yang, Andrew M. Bean, Eoin Delaney, Chris Russell, Adam Mahdi*


This repo contains the code and data used in the paper *"LLMs Don't Know Their Own Decision Boundaries: The Unreliability of Self-Generated Counterfactual Explanations" (EMNLP 2025)*. The paper explores whether LLMs can generate counterfactual explanations that both flip their own answers (**validity**) while making the smallest possible change to the input (**minimality**).

We do this by evaluating models in tabular data, binary classification tasks. **(A)** First we elicit predictions across the whole input space. This forms a decision boundary.  **(B)** Next, we ask models to provide self-generated counterfactual explanations (SCEs) for their predictions. SCEs are *valid* when they cross the decision boundary (below, red to blue) and are *minimal* if they are close to the dashed instance at the decision boundary. We asked to provide counterfactual explanations, we find the SCEs are tyically valid but far from minimal. **(C)** In a separate continuation from the original predictions, we ask models to provide minimal counterfactual explanations. In the majority of cases, these SCEs fail to cross the decision boundary. There is a trade-off between validity and minimality.

<p align="center">
  <img src="figures/figure_1.png" alt="figure_1" width="600"/><br/>
</p>


## Repo layout

```
analysis/            Analysis notebooks and additional experiments (Section 4.4 and 4.5 of the paper)
data/                Local dataset caches
figures/             Notebooks for figures.
results/             Example results produced by the pipeline (only included Llama 3.3 70B for example)
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
1. Create venv and install dependencies (Python 3.12):
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` at the repo root with any keys you need:
   - `OPENAI_API_KEY=...` (for OpenAI models)
   - `ANTHROPIC_API_KEY=...` (for Anthropic models)
   If you only run local/vLLM models, you can skip this.
3. Generate model predictions and counterfactual:
   ```bash
   python src/main.py \
       --task_model meta-llama/Meta-Llama-3-8B-Instruct \
       --dataset income \
       --sample_size 2000 \
       --max_tokens 1000 \
       --tensor_parallel_size 1
   ```
   The script uses the model and dataset definitions in `src/models_datasets/` and writes a JSON file to `results/<dataset>/<model>.json`.
4. Post-process the results and compute distances:
   ```bash
   python src/postprocessing.py --filepath results/income/llama3_8B.json --dataset income
   ```
5. Score results using scorer_notebook.ipynb  
6. (Optional) Overwrite the results file with the aggregate scores (useful if storage constraints):
   ```bash
   python src/scorer.py --filepath results/income/llama3_8B.json
   ```

The `run_main_results.sh` script provides a larger set of commands for reproducing the experiments across several models and datasets.

## Notes
- The full results are not provided as they are large. These can be generated using the pipeline above.

## Citation
If you use this work, please cite:

```bibtex
@inproceedings{mayne2025llms,
title={{LLMs} Don't Know Their Own Decision Boundaries: The Unreliability of Self-Generated Counterfactual Explanations},
author={Harry Mayne and Ryan Othniel Kearns and Yushi Yang and Andrew M. Bean and Eoin D. Delaney and Chris Russell and Adam Mahdi},
booktitle={The 2025 Conference on Empirical Methods in Natural Language Processing},
year={2025},
url={https://openreview.net/forum?id=mhEjUNFZtU}
}
```
