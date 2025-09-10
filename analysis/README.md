# Analysis

This directory contains analysis code and results for the experiments referenced in the paper. Each subfolder corresponds to an experiment and typically includes a short README describing its contents and how to reproduce or score results.

#### Experiments in the main paper
- Distance function sensitivity (Section 4.4) — scoring results using alternative distance metrics.
- Prompt sensitivity (Section 4.4) — evaluating robustness to prompt perturbations.
- Decision boundary consistency (Section 4.5) — consistency of model decisions under small changes.
- Operationalising distance (Section 4.5) — end-to-end experiment operationalising our distance measures.
- Metacognitive prompting (Section 4.5) — brief self-prediction style experiment.

#### Appendix experiments
- Temperature 1 (Section 4.4, Appendix C.3) — results at temperature = 1 for several datasets.
- Metacognitive prompting (Appendix E) — additional results and analysis.

Notes
- We keep representative result files in the repo (e.g., 70B variants) to illustrate outputs; larger or redundant files can be regenerated via the scripts noted in each subfolder.
- The notebook `analysis/generate_datasets.ipynb` documents how prompts/datasets are generated and is included for reference.



