#!/usr/bin/env python
# coding: utf-8
"""
python src/main.py \
  --task_model meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset folktexts \
  --sample_size 1000 \
  --max_tokens 512 \
  --tensor_parallel_size 1
  --gpu_memory_utilization 

VLLM_DISABLE_COMPILE_CACHE=1

vllm
CUDA_VISIBLE_DEVICES="1" python src/main.py   --task_model meta-llama/Meta-Llama-3-8B-Instruct   --dataset income   --sample_size 2000   --max_tokens 1000   --tensor_parallel_size 1
CUDA_VISIBLE_DEVICES="0,1,2,3" python src/main.py   --task_model meta-llama/Llama-3.3-70B-Instruct   --dataset income   --sample_size 2000   --max_tokens 1000   --tensor_parallel_size 1 big_model  # what happened to the big_model flag???

OpenAI
CUDA_VISIBLE_DEVICES="1" python src/main.py   --task_model gpt-4.1-nano-2025-04-14   --dataset income   --sample_size 20   --max_tokens 1000   --tensor_parallel_size 1 --max_concurrent 1 --wait 5

GPT-4o rate limits are easy to hit! This failed!
CUDA_VISIBLE_DEVICES="1" python src/main.py   --task_model gpt-4o   --dataset income   --sample_size 2000   --max_tokens 1000   --tensor_parallel_size 1 --max_concurrent 25 --wait 60

Anthropic
CUDA_VISIBLE_DEVICES="1" python src/main.py   --task_model claude-3-7-sonnet-20250219   --dataset income   --sample_size 20   --max_tokens 1000   --tensor_parallel_size 1 --max_concurrent 1 --wait 5 --extended_thinking disabled

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
import utils
from config import REPO_ROOT
from dotenv import load_dotenv
load_dotenv()

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
parser.add_argument("--extended_thinking", type=str, default="disabled", help="Extended thinking for Anthropic models. Can be enabled or disabled")
parser.add_argument("--model_precision", type=str, default="bfloat16",
                    help="float32|float16|bfloat16|float64")
parser.add_argument("--tensor_parallel_size", type=int, default=1,
                    help="#GPUs to split the model across (vLLM tensor parallelism)") # this replaced the big_model flag. Just set this to 4 and vllm will handle everything else.
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
utils.set_seed(args.seed)


# -------------------------------------------------------------------------------------
# Model -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
print(args.wait*5)
with open(REPO_ROOT / "src/models_datasets/models.json", "r") as f:
    model_dict = json.load(f)

model_meta   = model_dict[args.task_model]
model_name   = model_meta["name"]
provider     = model_meta["provider"] 

print(f"\nLoading model via {provider} …")
load_start = time.time()

llm = utils.make_client(
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

# -------------------------------------------------------------------------------------
# Dataset -----------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

with open(REPO_ROOT / "src/models_datasets/datasets.json", "r") as f:
    datasets_dict = json.load(f)

ds_meta = datasets_dict[args.dataset]
if ds_meta["local"]:
    dataset = load_from_disk(REPO_ROOT / ds_meta["filepath"])
else:
    dataset = load_dataset(ds_meta["filepath"], name=ds_meta["name"], split=ds_meta["split"])

# subsample
dataset = utils.sample(dataset, args.sample_size, args.seed)

# add prompt & input columns (idempotent)
std_prompts, input_key_data = utils.standard_prompts(dataset, ds_meta)
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
    utils.to_chat(p, model_dict[args.task_model]["system"])
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
first_responses = llm.chat(
    messages     = convos,
    temperature  = args.temperature,
    max_tokens   = args.max_tokens,
    seed         = args.seed,
    extended_thinking  = args.extended_thinking,
)

# first_responses = [o.outputs[0].text for o in first_outputs]

# This probably fails too. Though the extract_dict_answer() function is different to the dict_keys_to_numbers() one
json_outputs = [utils.parse_json_from_text(x) for x in first_responses]
model_answers = [utils.extract_dict_answer(x) for x in json_outputs]
model_answers = [utils.nothing_2_NA(x) for x in model_answers]

# -------------------------------------------------------------------------------------
# 2. Counterfactual follow‑up ---------------------------------------------------------
# -------------------------------------------------------------------------------------

print("Generating counterfactual follow‑ups …")
answer_choices = ds_meta["options"]
complements = [utils.select_counterfactual(ans, answer_choices) for ans in model_answers]

followup_convos = [
    utils.to_chat(p, model_meta["system"]) +
    utils.to_chat_followup(r, c, ds_meta["followup_template"])
    for p, r, c in zip(dataset["standard_prompts"],
                       first_responses,
                       complements)
]

if provider == "vllm":
    second_full_prompts = [
        tokenizer.apply_chat_template(c, tokenize=False,
                                      add_generation_prompt=True)
        for c in followup_convos
    ]
else:
    second_full_prompts = [
        "\n".join(f"{m['role']}: {m['content']}" for m in c)
        for c in followup_convos
    ]

follow_responses = llm.chat(
    messages    = followup_convos,
    temperature = args.temperature,
    max_tokens  = args.max_tokens,
    seed        = args.seed,
    extended_thinking  = args.extended_thinking,
)

# this part is problematic (and the next part probably as fails when no JSON)
fu_json = [utils.parse_json_from_text(x) for x in follow_responses]  # extracts the JSON. returns NA if no JSON found
fu_numeric = [utils.dict_keys_to_numbers(x) for x in fu_json]        # 11th May: fixed to handle "NA" strings
fu_strings = [utils.fill_templates(x) for x in fu_numeric]

inference_time = time.time() - inference_start

# -------------------------------------------------------------------------------------
# 3. Assemble results -----------------------------------------------------------------
# -------------------------------------------------------------------------------------

print("Assembling results …")
var_names_map = {
    "age": "AGEP_CF",
    "class_of_worker": "COW_CF",
    "education": "SCHL_CF",
    "marital_status": "MAR_CF",
    "occupation": "OCCP_CF",
    "place_of_birth": "POBP_CF",
    "relationship_to_person": "RELP_CF",
    "hours_worked": "WKHP_CF",
    "sex": "SEX_CF",
    "race": "RAC1P_CF",
    "area": "area_CF",
    "bedrooms": "bedrooms_CF",
    "bathrooms": "bathrooms_CF",
    "floors": "floors_CF",
    "systolic_bp": "systolic_bp_CF",
    "total_cholesterol": "total_cholesterol_CF",
}

local_results: Dict[str, Any] = {}
answer_key = ds_meta.get("answer_key")

# everything seems to be the right shape here. 
for idx in tqdm(range(len(dataset)), desc="Collating"):
    local_results[str(idx)] = {
        "dataset_id": int(dataset["id"][idx]),
        "standard_prompt": dataset["standard_prompts"][idx],
        "original_input": dataset["input_key_data"][idx],
        "original_full_prompt": full_prompts_str[idx],
        "original_full_response": first_responses[idx],
        "original_answer": model_answers[idx],
        "selected_complement": complements[idx],
        "followup_full_prompt": second_full_prompts[idx], 
        "followup_full_response": follow_responses[idx],
        "followup_answer": fu_strings[idx],
    }
    # numeric counterfactuals -----------------------------------------------------------
    try:
        for k, v in fu_numeric[idx].items():
            local_results[str(idx)][var_names_map[k]] = v
    except Exception:
        pass

# -------------------------------------------------------------------------------------
# 4. Save -----------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

if args.output_filepath:
    assert args.output_filepath.endswith(".json"), "Output path must end with .json"
    results_path = args.output_filepath
else:
    results_path = REPO_ROOT / f"results/{args.dataset}/{model_dict[args.task_model]['short_name']}.json"
    if args.extended_thinking == "enabled":
        results_path = REPO_ROOT / f"results/{args.dataset}/{model_dict[args.task_model]['short_name']}_thinking.json"

os.makedirs(os.path.dirname(results_path), exist_ok=True)

with open(results_path, "w") as f:
    json.dump(local_results, f, indent=4)

print("-" * 100)
print("Processing complete ✔️")
print(f"Samples processed:\t{args.sample_size}")
print(f"Output file:\t\t{results_path}")
print(f"Overhead time:\t\t{overhead_time:.2f} s")
print(f"Inference time:\t\t{inference_time:.2f} s")
print("-" * 100)