#!/usr/bin/env python
# coding: utf-8
"""
postprocessing.py
=========

If being run from the REPO_ROOT ->

python3 src/postprocessing.py --filepath results/income/llama3_8B.json --dataset income

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
import utils as  useful_functions # project‑specific helpers
from utils import postprocessing
import os


########################################################################################################################
parser = argparse.ArgumentParser(description="Unified offline/online inference")

# required
parser.add_argument("--filepath", required=True, type=str,
                    help="filepath to the data")
parser.add_argument("--dataset", required=True, type=str,
                    help="dataset")

args = parser.parse_args()

# process
results_edited = postprocessing(args.filepath, explicit_dataset=args.dataset, force=True, distance_metric="gower", save=True)
print("Postprocessing done and saved")