"""
Gower's Distance Evaluation Script

This script evaluates whether language models can correctly compute Gower's distance 
between data points. It generates multiple-choice questions where models must identify 
the data point with the smallest Gower's distance to a starting point.

Example usage:
    python experiment.py
    
    # For different models:
    # Edit the model_name variable below to test different models:
    # - For vLLM models: "google/gemma-2-9b-it"
    # - For OpenAI models: "gpt-4o-mini" 
    # - For Anthropic models: "claude-3-haiku-20240307"

The script:
1. Loads a dataset (default: house_prices)
2. Generates random starting points and candidate options
3. Computes ground truth Gower's distances
4. Prompts the model to identify the closest candidate
5. Evaluates accuracy and saves results
"""

import json
import random
import numpy as np
from datasets import load_dataset, load_from_disk, Dataset
from vllm import LLM, SamplingParams
from typing import Any, Dict, Optional, Tuple, List, Union
import re
import os
import sys
from pathlib import Path
import argparse
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import parse_json_from_text, make_client

# Set CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Settings
print("\n\n**** Initializing Gower's Distance Evaluation Script ****\n\n")

# required
parser = argparse.ArgumentParser(description="Argpass")
parser.add_argument("--model", required=True, type=str, help="Model identifier (e.g. meta-llama/Meta-Llama-3-8B-Instruct)")
parser.add_argument("--length", required=True, type=int, help="Model identifier (e.g. meta-llama/Meta-Llama-3-8B-Instruct)")
parser.add_argument("--max_concurrent", type=int, default=8, help="Max in-flight OpenAI requests")
parser.add_argument("--wait", type=int, default=5, help="Waiting when using openai")
parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU utalisation")
parser.add_argument("--tensor_parallel_size", type=int, default=1, help="#GPUs to split the model across (vLLM tensor parallelism)")
parser.add_argument("--extended_thinking", type=str, default="disabled", help="Extended thinking for Anthropic models. Can be enabled or disabled")

args = parser.parse_args()

number_of_options = 4
len_dataset = args.length
model_name = args.model  
tensor_parallel_size = 4
gpu_memory_utilization = args.gpu_memory_utilization
extra_results_path = PROJECT_ROOT / "analysis/operationalising_distance/breakdown.json"

print("Settings:\n"
      f"  number_of_options = {number_of_options}\n"
      f"  len_dataset = {len_dataset}\n"
      f"  model_name = {model_name}\n"
      f"  tensor_parallel_size = {args.tensor_parallel_size}\n\n")

# Load dataset
with open(PROJECT_ROOT / "src/models_datasets/datasets.json", "r") as f:
    datasets_dict = json.load(f)

ds_meta = datasets_dict["house_prices"]
if ds_meta["local"]:
    dataset = load_from_disk(PROJECT_ROOT / ds_meta["filepath"])
else:
    dataset = load_dataset(ds_meta["filepath"], name=ds_meta["name"], split=ds_meta["split"])

distance_metric = "gower"
distance_matrix = np.load(PROJECT_ROOT / f'src/distance_matrices/house_prices/house_prices_{distance_metric}.npy')

# Helper functions
def get_ids(k):
    rng = random.Random()
    n = len(dataset)
    assert k < n, "k must be smaller than the dataset size"
    
    all_idxs = rng.sample(range(n), k)
    start_idx, candidate_idxs = all_idxs[0], all_idxs[1:]
    return start_idx, candidate_idxs

def starting_point_2_point(starting_point, candidate_points):
    distances = []
    for x in candidate_points:
        distances.append(distance_matrix[starting_point, x])
    return distances

def to_chat(x, system):
    chat = [{"role": "user", "content": x}]
    if system != "":
        chat = [{"role": "system", "content": system}] + chat
    return chat

def determine_provider(model_name):
    """Determine which provider to use based on model name."""
    openai_models = ["gpt-3.5", "gpt-4", "o1", "o3"]
    anthropic_models = ["claude"]
    
    for pattern in openai_models:
        if pattern in model_name.lower():
            return "openai"
    
    for pattern in anthropic_models:
        if pattern in model_name.lower():
            return "anthropic"
    
    return "vllm"

# Generate dataset
new_dataset = {}

for i in range(len_dataset):
    # Get random points
    start_idx, candidate_idxs = get_ids(int(number_of_options+1))

    # Calculate distances
    distances = starting_point_2_point(start_idx, candidate_idxs)

    # Check if distances are equal
    equal = len(set(distances)) < len(distances)

    # Map candidates to letters
    candidates = {}
    letters = ['A', 'B', 'C', 'D', 'E', 'F']
    for ix, x in enumerate(candidate_idxs):
        candidates.update({letters[ix]: x})
    
    # Find minimum distance
    min_pos = np.argmin(distances)
    min_letter = letters[min_pos]

    # Extract data for candidates
    data = {}
    for ix, x in enumerate(candidate_idxs):
        area = dataset[x]['area']
        bedrooms = int(dataset[x]['bedrooms'])
        bathrooms = int(dataset[x]['bathrooms'])
        floors = int(dataset[x]['floors'])
        data.update({letters[ix]: (area, bedrooms, bathrooms, floors)})

    # Create prompts
    prompt_initial = f"Area (sq ft): {dataset[start_idx]['area']}, Bedrooms: {dataset[start_idx]['bedrooms']}, Bathrooms: {dataset[start_idx]['bathrooms']}, Floors: {dataset[start_idx]['floors']}\n"
    
    prompt_candidate = ""
    for k, v in data.items():
        string = f"{k}. Area (sq ft): {v[0]}, Bedrooms: {v[1]}, Bathrooms: {v[2]}, Floors: {v[3]}\n"
        prompt_candidate = prompt_candidate + string

    prompt_complete = f"""You will be provided with a starting data point and {len(candidates)} candidate data points. Your task is to compute and compare the Gower's Distance between the starting point and each candidate. You must use the definition and range information provided below. Return the letter of the candidate with the smaller Gower's Distance to the starting point.


Starting data:\n{prompt_initial}

Candidate options:\n{prompt_candidate}

Gower's Distance: 
For numeric or ordinal fields, the per-field distance is the absolute difference between the values, divided by the full range of that variable. For categorical fields, the distance is 0 if the values match and 1 otherwise. The total Gower's Distance is the average of the per-field distances.

Ranges:
area: ['500', '1000', '1500', '2000', '2500', '3000', '3500', '4000', '4500', '5000', '5500', '6000', '6500', '7000', '7500', '8000', '8500', '9000', '9500', '10000'] (ordinal)
bedrooms: 1-5 inclusive (integer)
bathrooms: 1-4 inclusive (integer)
floors: 1-4 inclusive (integer)

Only use the information provided. Do not make any assumptions beyond the definitions and ranges above.

Only respond with JSON output. Do not include any additional words in your answer. Provide the letter of the closest candidate to the starting data.
{{"answer": ""}}
"""

    # Update main dataset
    new_dataset[i] = {
        "start_id": start_idx, 
        "start_ids": candidates, 
        "data": data, 
        "prompt_initial": prompt_initial, 
        "prompt_candidate": prompt_candidate, 
        "prompt_complete": prompt_complete, 
        "distances": distances, 
        "equal": equal, 
        "answer": min_letter
    }

# Remove cases with equal distances
new_dataset = {k: v for k, v in new_dataset.items() if not v["equal"]}

# Create conversations
prompts = [v['prompt_complete'] for k, v in new_dataset.items()]
conversations = [to_chat(x, system="") for x in prompts]

print(len(new_dataset))
print(len(conversations))

# Load model
provider = determine_provider(model_name)
print(f"Using provider: {provider}")

if provider == "vllm":
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=5000
    )
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization
    )
else:
    # Use the client factory from utils.py
    llm = make_client(
        provider=provider,
        model_name=model_name,
        max_concurrent=args.max_concurrent,
        wait=args.wait
    )

# Generate answers
print(f"Processing {len(conversations)} conversations")
if provider == "vllm":
    outputs = llm.chat(conversations, sampling_params, use_tqdm=True)
    outputs_short = [outputs[i].outputs[0].text for i in range(len(outputs))]
else:
    outputs_short = llm.chat(
        messages=conversations,
        temperature=0.0,
        max_tokens=5000,
        seed=42,
        extended_thinking=args.extended_thinking
    )
print(f"Generated {len(outputs_short)} responses")

# Extract answers
model_answers = []
for block in outputs_short:
    answer = "NA"
    try:
        # Find the first JSON object
        m = re.search(r'\{.*?\}', block, flags=re.S)
        if m:
            data = json.loads(m.group(0))
            answer = data.get("answer", "NA")
    except json.JSONDecodeError:
        # Fallback: extract letter directly
        m = re.search(r'"answer"\s*:\s*"?([A-Z])"?', block)
        if m:
            answer = m.group(1)
    model_answers.append(answer)

# Update dataset with model answers
for ix, (k, v) in enumerate(new_dataset.items()):
    v['model_answer'] = model_answers[ix]

# Calculate and print results
count = sum(1 for v in new_dataset.values() if v['model_answer'] == v['answer'])
count_NA = sum(1 for v in new_dataset.values() if v['model_answer'] == "NA")
try:
    accuracy = (count / (len(new_dataset) - count_NA)) * 100
except:
    accuracy = 0

print("\n\n**** Results ****")
print(f"Total instances: {len(new_dataset)}")
print(f"Correct predictions: {count}")
print(f"Accuracy: {accuracy:.2f}% ({count}/{len(new_dataset) - count_NA})")
print(f"Total NA: {count_NA}")
print("\n\n")

# Save results
results_path = PROJECT_ROOT / "analysis/operationalising_distance/results.json"

# Load existing results or create new
if results_path.exists():
    with open(results_path, "r") as f:
        results = json.load(f)
else:
    results = {}

dataset_name = "house_prices"
results.setdefault(dataset_name, {})
results[dataset_name].setdefault(model_name, {})

results[dataset_name][model_name][str(number_of_options)] = {
    "accuracy": accuracy,
    "correct": count,
    "total": len(new_dataset)
}

with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved results to {results_path}")

# Save detailed results
def make_jsonable(obj):
    if isinstance(obj, dict):
        return {k: make_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_jsonable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [make_jsonable(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

serializable_dataset = make_jsonable(new_dataset)

if extra_results_path:
    with open(extra_results_path, "w") as f:
        json.dump(serializable_dataset, f, indent=2)
    print(f"Also saved detailed results to {extra_results_path}")