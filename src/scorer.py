#!/usr/bin/env python
# coding: utf-8
"""
scorer.py
=========

Not the default to use this.

Score a result file produced by our inference pipeline.

Usage
-----

If being run from the REPO_ROOT ->
python src/scorer.py --filepath results/income/llama3_8B.json

The script overwrites the existing score file in place, so avoid using it in
production runs where you need to keep the full per-example results.

Exit status is non-zero if the score computation fails.
"""

import argparse
import json
import os
import gc
import warnings
from typing import List, Dict, Any
import time
import torch
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from vllm import LLM, SamplingParams
from utils import postprocessing, scoring_function
import os

import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd

import os
import sys

parent_dir = os.path.abspath(os.path.join("", ".."))
sys.path.insert(0, parent_dir)

#from helper_functions.scoring_functions import scoring_function

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Unified offline/online inference")

# required
parser.add_argument("--filepath", required=True, type=str,
                    help="filepath to the data")

args = parser.parse_args()

# process
results = scoring_function(args.filepath, overwrite=True, verbose=0, detail=False)