#!/usr/bin/env python
# coding: utf-8
"""
Main script to generate sample of results with logprobs. Used for the decision boundary consistency experimemnt.



python main.py \
  --task_model meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset folktexts \
  --sample_size 1000 \
  --max_tokens 512 \
  --tensor_parallel_size 1
  --gpu_memory_utilization 

VLLM_DISABLE_COMPILE_CACHE=1

CUDA_VISIBLE_DEVICES="3" python main_log_probs.py   --task_model meta-llama/Llama-3.1-8B-Instruct   --dataset income   --sample_size 2000   --max_tokens 1000   --tensor_parallel_size 1 --log_probs 10

##### OpenAI
CUDA_VISIBLE_DEVICES="1" python main.py   --task_model gpt-4.1-nano-2025-04-14   --dataset income   --sample_size 20   --max_tokens 1000   --tensor_parallel_size 1 --max_concurrent 1 --wait 5

##### --> 4o rate limits are easy to hit! 
CUDA_VISIBLE_DEVICES="1" python main.py   --task_model gpt-4o   --dataset income   --sample_size 2000   --max_tokens 1000   --tensor_parallel_size 1 --max_concurrent 25 --wait 60

######################################################################################################################################################
### CUDA_VISIBLE_DEVICES="0,1,2,3" python main_log_probs.py   --task_model meta-llama/Llama-3.3-70B-Instruct   --dataset income   --sample_size 2000   --max_tokens 1000   --tensor_parallel_size 4 --log_probs 10
######################################################################################################################################################
"""

import pickle, gzip                        
import argparse
import json
import os
import gc
import warnings
from typing import List, Dict, Any
import torch
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from vllm import LLM, SamplingParams
import log_probs_useful_functions as useful_functions 
import time
from dotenv import load_dotenv
load_dotenv()
import sys
from collections import Counter
sys.path.insert(0, "../../src")
sys.path.insert(0, "../..")
from config import REPO_ROOT

# -------------------------------------------------------------------------------------
# CLI ---------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Unified offline/online inference")

# required
parser.add_argument("--task_model", required=True, type=str,
                    help="Model identifier (e.g. meta-llama/Meta-Llama-3-8B-Instruct)")
parser.add_argument("--dataset", required=True, type=str, help="Dataset identifier")
parser.add_argument("--sample_size", required=True, type=int, help="Number of samples")

# optional (kept for API stability)
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--max_tokens", type=int, default=1000, help="Max new tokens per turn")
parser.add_argument("--output_filepath", type=str, default="", help="Custom .json output path")
parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
parser.add_argument("--model_precision", type=str, default="bfloat16",
                    help="float32|float16|bfloat16|float64")
parser.add_argument("--tensor_parallel_size", type=int, default=1,
                    help="#GPUs to split the model across (vLLM tensor parallelism)")
parser.add_argument("--pipeline_parallelism", type=int, default=1,
                    help="#GPUs to split the model across (vLLM tensor parallelism)")
parser.add_argument("--max_concurrent", type=int, default=8,
                    help="Max in-flight OpenAI requests")
parser.add_argument("--wait", type=int, default=5,
                    help="Waiting when using openai")
parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                    help="GPU utalisation")
parser.add_argument("--log_probs",type=int,default=0, help="If > 0, request log-probs for every generated token and keep the top-k alternatives per token (k = this value).")


args = parser.parse_args()
useful_functions.set_seed(args.seed)


# -------------------------------------------------------------------------------------
# Model -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
print(args.wait*5)
with open("models_datasets/models.json", "r") as f:
    model_dict = json.load(f)

model_meta   = model_dict[args.task_model]
model_name   = model_meta["name"]
provider     = model_meta["provider"] 

print(f"\nLoading model via {provider} …")
load_start = time.time()
# llm = LLM(
#     model=model_name,
#     dtype=args.model_precision.lower(),
#     tensor_parallel_size=args.tensor_parallel_size,
#     trust_remote_code=True,
# )

llm = useful_functions.make_client(
    provider=provider,
    model_name=model_name,
    dtype=args.model_precision.lower(),
    tensor_parallel_size=args.tensor_parallel_size,
    pipeline_parallel_size = args.pipeline_parallelism,
    max_concurrent=args.max_concurrent,
    gpu_memory_utilization=args.gpu_memory_utilization,
    wait = args.wait,
)

overhead_time = time.time() - load_start
print(f"Model ready: {overhead_time}")

inference_start = time.time()
sampling_params = SamplingParams(
    temperature=float(args.temperature),
    max_tokens=args.max_tokens,
    seed=args.seed,
)

# ---------- helpers ---------------------------------------------------------

def save_logprobs(obj, path: str):
    """
    Binary-serialise *anything* (tensors, numpy, SDK classes…) using pickle 5+.
    """
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:                 # gzip ≈ 4-5× smaller
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# -------------------------------------------------------------------------------------
# Dataset -----------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

with open(REPO_ROOT / "arc/models_datasets/datasets.json", "r") as f:
    datasets_dict = json.load(f)

ds_meta = datasets_dict[args.dataset]
if ds_meta["local"]:
    dataset = load_from_disk(ds_meta["filepath"])
else:
    dataset = load_dataset(ds_meta["filepath"], name=ds_meta["name"], split=ds_meta["split"])

# subsample
dataset = useful_functions.sample(dataset, args.sample_size, args.seed)

# add prompt & input columns (idempotent)
std_prompts, input_key_data = useful_functions.standard_prompts(dataset, ds_meta)
for col_name, col_values in {
    "standard_prompts": std_prompts,
    "input_key_data": input_key_data,
}.items():
    if col_name not in dataset.column_names:
        dataset = dataset.add_column(col_name, col_values)
if "id" not in dataset.column_names:
    dataset = dataset.add_column("id", list(range(len(dataset))))

print(f"Dataset ready | size: {len(dataset)}\n")

# -------------------------------------------------------------------------------------
# 1. First‑pass generation ------------------------------------------------------------
# -------------------------------------------------------------------------------------

print("Generating first‑pass answers …")
convos: List[List[Dict[str, str]]] = [
    useful_functions.to_chat(p, model_dict[args.task_model]["system"])
    for p in dataset["standard_prompts"]
]

# store a textual version of each full prompt for logging
if provider == "vllm":
    tokenizer         = llm.llm.get_tokenizer()
    full_prompts_str  = [
        tokenizer.apply_chat_template(conv, tokenize=False,
                                      add_generation_prompt=True)
        for conv in convos
    ]
else:
    full_prompts_str  = [
        "\n".join(f"{m['role']}: {m['content']}" for m in conv)
        for conv in convos
    ]

# list of length of the dataset
first_pass = llm.chat(
    messages    = convos,
    temperature = args.temperature,
    max_tokens  = args.max_tokens,
    seed        = args.seed,
    log_probs   = args.log_probs,     
)

# unpack text & log-probs
first_responses, first_logprobs = zip(*first_pass)  

# save
save_logprobs(first_logprobs, REPO_ROOT / f"analysis/decision_boundary_consistentcy/{args.dataset}.logprobs.pkl")

