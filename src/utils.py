import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
import os
import sys
from tqdm import tqdm
import json
from torch.utils.data import DataLoader
import time
import traceback
import numpy as np
import random
import re
import os
import copy
import gower
import nltk
from typing import Union
import string
import argparse
import gc
import ast
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import random
import numpy as np

import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
from config import REPO_ROOT


from collections import defaultdict

import json
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset, concatenate_datasets
import random
import pandas as pd


import warnings
warnings.filterwarnings("ignore")

import os
import re
import ast
import json
import time
import random
import string
import warnings
import traceback
import backoff
import torch
from tqdm.auto import tqdm  
import time


import numpy as np
import nltk
from tqdm import tqdm
from typing import Any, Dict, Optional, Tuple, List, Union
import matplotlib.pyplot as plt
from vllm import LLM, SamplingParams
import anthropic
#import google.generativeai as genai

import os, logging, concurrent.futures as cf
from tqdm.auto import tqdm
from openai import AsyncOpenAI
import asyncio
from itertools import islice

from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    wait_exponential_jitter,
    wait_exponential,
)


import warnings
warnings.filterwarnings("ignore")

################################################################################################################################################################################################
################################################################################################################################################################################################
### Helper functions for main.py
################################################################################################################################################################################################
################################################################################################################################################################################################


################################################################################################################################################################################################
# reproducability
################################################################################################################################################################################################

def set_seed(seed):
    """ set the random seed everywhere. confirmed this works as expected """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

################################################################################################################################################################################################
# dataset filtering and sampling
################################################################################################################################################################################################

def early_filter(dataset, condition, limit):
    """
    Iterates over the dataset and returns a new Dataset containing at most `limit`
    examples that satisfy the condition.
    """
    results = []
    for ex in dataset:
        if condition(ex):
            results.append(ex)
            if len(results) >= limit:
                break
    return Dataset.from_list(results)

def sample_and_balance(dataset_, sample_size_, options_, seed_):
    """ 
    Give a dataset and sample size, returns a sampled and balanced dataset
    Initiated based on whether the balance key is True
    """

    # options and number of options
    
    num_options = len(options_)
    samples_per_option = sample_size_ // num_options
    extra = sample_size_ % num_options

    # init
    sampled_subdatasets = []

    #
    for option in options_:

        # filter to sample_size_ items per category (not that efficient)
        option_ds = early_filter(dataset_, lambda ex: ex['answer'] == option, sample_size_) # filter using early_filtering
        
        # Calculate the number of examples to sample for this option.
        count = samples_per_option + (1 if extra > 0 else 0)
        if extra > 0:
            extra -= 1
        
        # Optionally, check if the option has enough examples.
        if count > len(option_ds):
            raise ValueError(f"Not enough examples for option {option}: requested {count}, available {len(option_ds)}")
        
        # Shuffle the option-specific dataset and select the count number of examples.
        option_sample = option_ds.shuffle(seed=seed_).select(range(count))
        sampled_subdatasets.append(option_sample)

    # Concatenate all the subdatasets into a single balanced dataset and shuffle the final result.
    balanced_sample = concatenate_datasets(sampled_subdatasets).shuffle(seed=seed_)

    return balanced_sample

def sample(dataset, sample_size, seed):

    if sample_size < len(dataset):
        rng = random.Random(seed)
        indices = random.sample(range(len(dataset)), sample_size)
        dataset = dataset.select(indices)

    return dataset

################################################################################################################################################################################################
# Prompt creation
################################################################################################################################################################################################

def standard_prompts(dataset, dataset_details):
    """ 
    Use the input_key and template in the dataset dict to make the filled_prompts.
    Return a list of filled prompts 
    """
    input_key = dataset_details['input_key']

    # init
    filled_prompts = []
    descriptions = [] # input key
    template = dataset_details['standard_prompt_template']

    for i in dataset:
        filled_prompts.append(template.format(**{input_key:i[input_key]}))
        descriptions.append(i[input_key])

    return filled_prompts, descriptions

def verification_prompts(dataset, dataset_details, followup_model_answer):
    """ 
    Use the input_key and template in the dataset dict to make the filled_prompts.
    Requires "followup_model_answer" which is a list of the counterfactual inputs
    Return a list of filled prompts 
    """
    input_key = dataset_details['input_key'] ## ah! this needs to change...

    # init
    filled_prompts = []
    template = dataset_details['standard_prompt_template']

    for text in followup_model_answer:
        filled_prompts.append(template.format(**{input_key:text}))
        # this has to be a dict due to input_key being a string

    return filled_prompts

def to_chat(x, system):

    # define chat template
    chat = [
        {"role": "user", "content": x}
    ]

    # add system if it is in the model dict
    if system!="":
        chat = [{"role": "system", "content": system}] + chat

    return chat

def to_chat_followup(response, complement, followup_template):

    # generate follow up template
    followup_template = followup_template.format(complement=complement)

    # define chat template
    chat = [
        {"role": "assistant", "content": response},
        {"role": "user", "content": followup_template}
    ]

    return chat

################################################################################################################################################################################################
# UNIVERSAL LLM CLIENT FACTORY
################################################################################################################################################################################################
def _ensure_batch(msgs):
    """Return (batched_msgs, was_batched_bool)."""
    return (msgs, True) if msgs and isinstance(msgs[0], list) else ([msgs], False)

def chunks(iterable, size):
    """Yield successive `size`-length chunks."""
    it = iter(iterable)
    while (batch := list(islice(it, size))):
        yield batch

# 1. vLLM ───────────────────────────────────────────────────────────────────
class VllmClient:
    def __init__(self, model_name: str, dtype: str, tensor_parallel_size: int, max_concurrent, gpu_memory_utilization, wait, pipeline_parallel_size, **_):
        self.llm = LLM(model=model_name,
                       dtype=dtype,
                       tensor_parallel_size=tensor_parallel_size,
                       pipeline_parallel_size=pipeline_parallel_size,
                       trust_remote_code=True,
                       gpu_memory_utilization= gpu_memory_utilization,
                       )

    def chat(self, messages: List[List[Dict[str, str]]], temperature: float,
             max_tokens: int, seed: int, **_):
        params  = SamplingParams(temperature=temperature,
                                 max_tokens=max_tokens,
                                 seed=seed)
        outputs = self.llm.chat(messages, sampling_params=params, use_tqdm=True)
        return [o.outputs[0].text for o in outputs]

# 2. OpenAI ───────────────────────────────────────────────────────────────────
class OpenAIClient:
    """Simple asynchronous OpenAI wrapper."""

    REASONING_MODELS = [
        "o3-2025-04-16",
    ]

    def __init__(self, model_name: str, max_concurrent: int = 8, wait: float = 0.0, **_):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model_name
        self.wait = wait
        self.batch_size = max_concurrent

    async def _single_call_async(self, conv, temperature, max_tokens, seed):
        for attempt in range(8):
            try:
                if self.model in self.REASONING_MODELS:  # reasoning models use different parameters
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=conv,
                        max_completion_tokens=max_tokens,
                        seed=seed,
                    )
                else:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=conv,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        seed=seed,
                    )
                return resp.choices[0].message.content
            except Exception as e:
                print(f"OpenAI error: {e}")
                logging.exception("OpenAI chat attempt failed")
                if attempt == 7:
                    logging.warning("OpenAI chat failed: %s", e)
                    return None
                await asyncio.sleep(2 ** attempt)

    async def _chat_async(self, messages, temperature=0.0, max_tokens=256, seed=0):
        batched, already = _ensure_batch(messages)
        outs = []
        pbar = tqdm(total=len(batched), desc=f"OpenAI {self.model}", unit="chat", leave=False)
        for batch in chunks(batched, self.batch_size):
            results = await asyncio.gather(
                *(self._single_call_async(conv, temperature, max_tokens, seed) for conv in batch),
                return_exceptions=True,
            )
            outs.extend(results)
            pbar.update(len(batch))
            if self.wait > 0:
                await asyncio.sleep(self.wait)
        pbar.close()
        return outs if already else outs[0]

    def chat(self, messages: List[List[Dict[str, str]]], temperature:float, max_tokens: int, seed: int, **_):
        return asyncio.run(self._chat_async(messages, temperature, max_tokens, seed))

# 3. Anthropic ─────────────────────────────────────────────────────────────---
class AnthropicClient:
    """Simple asynchronous Anthropic wrapper."""

    REASONING_MODELS = [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-3-7-sonnet-20250219",
    ]

    def __init__(self, model_name: str, max_concurrent: int = 8, wait: float = 0.0, extended_thinking: str = "disabled", **_):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model_name
        self.wait = wait
        self.batch_size = max_concurrent
        self.extended_thinking = extended_thinking

    async def _single_call_async(self, conv, temperature, max_tokens, seed, extended_thinking):
        for attempt in range(8):
            try:
                if self.model in self.REASONING_MODELS:  # reasoning models may support extended thinking
                    thinking_dict = {}
                    if extended_thinking=="enabled":
                        thinking_dict.update({"type":"enabled","budget_tokens": 10000})
                    else:
                        thinking_dict.update({"type":"disabled"})
                    resp = await self.client.messages.create(
                        model=self.model,
                        messages=conv,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        thinking=thinking_dict,
                    )
                    content = "".join(getattr(block, "text", "") for block in resp.content)
                    thinking = "".join(getattr(block, "thinking", "") for block in resp.content) # check this against API documentation
                else: # standard models e.g. Haiku
                    resp = await self.client.messages.create(
                        model=self.model,
                        messages=conv,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    content = "".join(getattr(block, "text", "") for block in resp.content)
                    thinking = ""
                return thinking + content # this might not be the best way to do this... would be nice to return them separately but this will do.
            except Exception as e:
                print(f"Anthropic error: {e}")
                logging.exception("Anthropic chat attempt failed")
                if attempt == 7:
                    logging.warning("Anthropic chat failed: %s", e)
                    return None
                await asyncio.sleep(2 ** attempt)

    async def _chat_async(self, messages, temperature=0.0, max_tokens=256, seed=0, extended_thinking="disabled"):
        batched, already = _ensure_batch(messages)
        outs = []
        pbar = tqdm(total=len(batched), desc=f"Anthropic {self.model}", unit="chat", leave=False)
        for batch in chunks(batched, self.batch_size):
            results = await asyncio.gather(
                *(self._single_call_async(conv, temperature, max_tokens, seed, extended_thinking) for conv in batch),
                return_exceptions=True,
            )
            outs.extend(results)
            pbar.update(len(batch))
            if self.wait > 0:
                await asyncio.sleep(self.wait)
        pbar.close()
        return outs if already else outs[0]

    def chat(self, messages: List[List[Dict[str, str]]], temperature:float, max_tokens: int, seed: int, extended_thinking: str, **_):
        return asyncio.run(self._chat_async(messages, temperature, max_tokens, seed, extended_thinking))

# 3. FACTORY ────────────────────────────────────────────────────────────────
_CLIENTS = {
    "vllm":      VllmClient,
    "openai":    OpenAIClient,
    "anthropic": AnthropicClient,
    #"google":    GeminiClient,
}

def make_client(provider: str, **kwargs):
    """Instantiate an LLM client implementing `.chat(...)`."""
    if provider not in _CLIENTS:
        raise ValueError(f"Unsupported provider '{provider}'")
    return _CLIENTS[provider](**kwargs)


################################################################################################################################################################################################
# Answer extraction -- Helper functions
################################################################################################################################################################################################

def parse_json_from_text(text: str) -> Any:
    """
    Robustly extract and parse JSON (object or array) from a text blob,
    correcting common LLM formatting errors.
    On failure it now returns the sentinel string "NA" instead of raising.
    """
    try:
        # ─────────────────────────── helpers ────────────────────────────
        def strip_code_fences(src: str) -> str:
            return re.sub(r"```(?:json)?\s*\n(.*?)```", r"\1", src, flags=re.DOTALL)

        def find_candidate(src: str) -> Tuple[str, int, Tuple[str, str]]:
            for opener, closer in (("{", "}"), ("[", "]")):
                for start in (m.start() for m in re.finditer(re.escape(opener), src)):
                    depth = 0
                    for i in range(start, len(src)):
                        if src[i] == opener:
                            depth += 1
                        elif src[i] == closer:
                            depth -= 1
                        if depth == 0:
                            return src[start:i + 1], 0, (opener, closer)
                    return src[start:], depth, (opener, closer)
            print("No JSON object or array found in text")

        def fix_single_quoted_values(src: str) -> str:
            out, i, n = [], 0, len(src)
            while i < n:
                ch = src[i]
                if ch == '"':                               # keep existing double-quoted strings
                    out.append(ch); i += 1
                    while i < n:
                        out.append(src[i])
                        if src[i] == '"' and src[i - 1] != '\\':
                            i += 1; break
                        i += 1
                elif ch == "'":                              # convert single-quoted *values*
                    i += 1; buf = []
                    while i < n:
                        c = src[i]
                        if c == "\\" and i + 1 < n:
                            buf.append(src[i:i + 2]); i += 2
                        elif c == "'":
                            i += 1; break
                        else:
                            buf.append(c); i += 1
                    out.append(json.dumps(''.join(buf)))
                else:
                    out.append(ch); i += 1
            return ''.join(out)

        def escape_interior_double_quotes(src: str) -> str:
            out, i, n = [], 0, len(src)
            while i < n:
                if src[i] == '"':
                    out.append('"'); i += 1; start = i
                    while i < n and not (src[i] == '"' and src[i - 1] != '\\'):
                        i += 1
                    content = src[start:i]
                    out.append(content.replace('"', '\\"'))
                    if i < n: out.append('"'); i += 1
                else:
                    out.append(src[i]); i += 1
            return ''.join(out)

        def regex_fixes(src: str) -> str:
            s = src
            s = s.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
            s = re.sub(r'`([^`]*)`', r'"\1"', s)                       # back-ticks → quotes
            s = re.sub(r'//.*?$|#.*?$', '', s, flags=re.MULTILINE)     # strip comments
            s = re.sub(r"[\x00-\x1f]+", "", s)                         # control chars
            # quote bare or single-quoted keys
            key_pat = r'(?P<prefix>[\{,\[])\s*(?P<key>[A-Za-z0-9_\-\' ]+?)\s*:'
            def _q(m):
                key = m.group("key").strip().strip("'")
                return f"{m.group('prefix')} \"{key}\":"
            s = re.sub(key_pat, _q, s)
            # add missing commas / remove trailing commas
            s = re.sub(r'([\]\}"\d])\s+([\{\["\w])', r"\1, \2", s)
            s = re.sub(r",\s*([\}\]])", r"\1", s)
            return s
        # ─────────────────────────── pipeline ───────────────────────────
        clean = strip_code_fences(text)
        candidate, unbalanced, (opener, closer) = find_candidate(clean)
        if unbalanced > 0:
            candidate += closer * unbalanced            # auto-close

        # ---------- “raw” parse attempts ----------
        for parser, fn in (("json.loads", json.loads),
                           ("ast.literal_eval", lambda s: json.loads(json.dumps(ast.literal_eval(s))))):
            try:
                return fn(candidate)
            except Exception:
                pass  # fall through to clean-up passes
        # ------------------------------------------------

        # heavy-duty fixes
        candidate = fix_single_quoted_values(candidate)
        candidate = escape_interior_double_quotes(candidate)

        try:
            return json.loads(candidate)
        except Exception:
            pass
        try:
            obj = ast.literal_eval(candidate)
            return json.loads(json.dumps(obj))
        except Exception:
            pass

        cleaned = escape_interior_double_quotes(regex_fixes(candidate))
        try:
            return json.loads(cleaned)
        except Exception:
            pass
        try:
            obj = ast.literal_eval(cleaned)
            return json.loads(json.dumps(obj))
        except Exception:
            pass

        # If we’re still here, every strategy failed
        return "NA"

    # Any helper-level or unforeseen error bubbles here
    except Exception:
        return "NA"

def nothing_2_NA(text):
    """ The JSON parsing code sometimes returns "" so this just changes it to NA"""
    if len(text)>0:
        return text
    else:
        return 'NA'

def extract_dict_answer(obj: Any, default: str = "NA") -> str:
    """
    Return the value associated with a key named 'answer' (case-insensitive),
    tolerating extra leading/trailing spaces in the key itself.

    Parameters
    ----------
    obj : Any
        A parsed-JSON structure (dict / list / mixed). If it's not a dict- or
        list-like object, `default` is returned.
    default : str, optional
        Fallback result when the key isn't found.  Defaults to "NA".

    Returns
    -------
    str
        The matched value, or `default`.
    """

    def _dfs(node: Any) -> Optional[Any]:
        if isinstance(node, dict):
            for k, v in node.items():
                # tolerate spaces and any capitalisation
                if isinstance(k, str) and k.strip().lower() == "answer":
                    return v
            for v in node.values():                     # search deeper
                found = _dfs(v)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = _dfs(item)
                if found is not None:
                    return found
        return "NA"                                     # no match here

    found = _dfs(obj)
    return default if found is None else found

################################################################################################################################################################################################
# Dictionaries
################################################################################################################################################################################################

education = {
        'N/A - no schooling completed': 1,
        'Nursery school / preschool': 2,
        'Kindergarten': 3,
        '1st grade only': 4,
        '2nd grade': 5,
        '3rd grade': 6,
        '4th grade': 7,
        '5th grade': 8,
        '6th grade': 9,
        '7th grade': 10,
        '8th grade': 11,
        '9th grade': 12,
        '10th grade': 13,
        '11th grade': 14,
        '12th grade, no diploma': 15,
        'Regular high school diploma': 16,
        'GED or alternative credential': 17,
        'Some college, less than 1 year': 18,
        'Some college, 1 or more years, no degree': 19,
        "Associate's degree": 20,
        "Bachelor's degree": 21,
        "Master's degree": 22,
        "Professional degree beyond a bachelor's degree": 23,
        'Doctorate degree': 24
    }

class_of_worker = {
    'Working for a for-profit private company or organization': 1,
    'Working for a non-profit organization': 2,
    'Working for the local government': 3,
    'Working for the state government': 4,
    'Working for the federal government': 5,
    'Owner of non-incorporated business, professional practice, or farm': 6,
    'Owner of incorporated business, professional practice, or farm': 7,
    'Working without pay in a for-profit family business or farm': 8,
}

marital_status = {
    'Married': 1,
    'Widowed': 2,
    'Divorced': 3,
    'Separated': 4,
    'Never married': 5,
}

sex = {
    'Male': 1,
    'Female': 2,
}

race = {
    'White': 1,
    'Black or African American': 2,
    'American Indian': 3,
    'Alaska Native': 4,
    'American Indian and Alaska Native tribes specified; or American Indian or Alaska Native, not specified and no other races': 5,
    'Asian': 6,
    'Native Hawaiian and Other Pacific Islander': 7,
    'Some other race alone (non-White)': 8,
    'Two or more races': 9,
}

# Collect them in one place for easy iteration
_LOOKUPS_RAW: Dict[str, Dict[str, int]] = {
    "education": education,
    "class_of_worker": class_of_worker,
    "marital_status": marital_status,
    "sex": sex,
    "race": race,
}

################################################################################################################################################################################################
# Revised data as JSON, i.e. take the dictionary and turn it into numerical data
################################################################################################################################################################################################

def _normalize(text: str) -> str:
    """
    Lower-case, remove apostrophes/punctuation (except alphanumerics),
    collapse whitespace. E.g. "Bachelor's degree" → "bachelors degree".
    """
    text = text.lower().replace("'", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())

# Build *normalised* lookup tables once
_LOOKUPS_NORM = {
    field: {_normalize(k): v for k, v in mapping.items()}
    for field, mapping in _LOOKUPS_RAW.items()
}

def dict_keys_to_numbers(record: Union[Dict[str, Any], str]) -> Union[Dict[str, Any], str]:
    """
    Return a **new** dict where every *listed* categorical variable
    (education, class_of_worker, marital_status, sex, race) is replaced
    by its numeric code.  
    Unrecognised or missing values become the string 'NA'.
    """
    if type(record)==str:
        return "NA"

    out = record.copy() # this fails when record is a string

    for field, lookup in _LOOKUPS_NORM.items():
        if field not in out:
            continue

        raw_val = str(out[field])
        norm_val = _normalize(raw_val)

        # ---------- Education: extra grade-number heuristic ----------
        if field == "education" and norm_val not in lookup:
            m = re.search(r"\b(\d{1,2})\b", norm_val)
            if m:
                g = int(m.group(1))
                if 1 <= g <= 12:
                    out[field] = g + 3          # grade n → code n+3
                    continue

        # ---------- Generic lookup ----------
        out[field] = lookup.get(norm_val, "NA")

    return out

################################################################################################################################################################################################
# Create a new descption from the data
################################################################################################################################################################################################

# -------------------------------------------------------------
# One-time at module top, just after education dict
_code2edu  = {v: k for k, v in education.items()}
_code2cow  = {v: k for k, v in class_of_worker.items()}
_code2mar  = {v: k for k, v in marital_status.items()}
_code2sex  = {v: k for k, v in sex.items()}
_code2race = {v: k for k, v in race.items()}
# -------------------------------------------------------------

_CODE_LOOKUP: Dict[str, Dict[int, str]] = {
    "education": _code2edu,
    "class_of_worker": _code2cow,
    "marital_status": _code2mar,
    "sex": _code2sex,
    "race": _code2race,
}

TEMPLATES = {
    'age': "- The age is: {} years old.",
    'class_of_worker': "- The class of worker is: {}.",
    'education': "- The highest educational attainment is: {}.",
    'marital_status': "- The marital status is: {}.",
    'occupation': "- The occupation is: {}.",
    'place_of_birth': "- The place of birth is: {}.",
    'relationship_to_person': "- The relationship to the reference person in the survey is: {}.",
    'hours_worked': "- The usual number of hours worked per week is: {} hours.",
    'sex': "- The sex is: {}.",
    'race': "- The race is: {}.",
    'area': "- The size of the house (sq ft) is: {}.", # house prices
    'bedrooms': "- The number of bedrooms is: {}.", # house prices
    'bathrooms': "- The number of bathrooms is: {}.", # house prices
    'floors': "- The number of floors is: {}.", # house prices
    "systolic_bp":"- The systolic blood pressure (mmHg) is: {}.", # heart disease
    "total_cholesterol":"- The total cholesterol (mg/dL) is: {}.", # heart disease
}

def _translate(field: str, value: Any) -> Any:
    """
    If `value` is an int (or int-like string) and `field` has a reverse
    dictionary, return the corresponding label; otherwise return `value`.
    Unknown codes become "NA".
    """
    lookup = _CODE_LOOKUP.get(field)
    if lookup is None:
        return value

    try:
        code = int(value)
    except (ValueError, TypeError):
        return value

    return lookup.get(code, "NA")

def fill_templates(data: Dict[str, Any],
                   templates: Dict[str, str] = TEMPLATES,
                   sep: str = "\n") -> str:
    """
    Build a human-readable description string.

    • For every key present in `templates`, if it’s in `data`, render it.
    • Numeric codes for the five enumerated variables are converted via
      their reverse dictionaries.
    • Unknown codes become "NA".
    • Keys missing from `data` are skipped.
    """
    fragments = []
    for key in templates:               # keep template order
        if key not in data:
            continue
        fragments.append(
            templates[key].format(_translate(key, data[key]))
        )
    return sep.join(fragments)

################################################################################################################################################################################################
# Other...
################################################################################################################################################################################################

def clean_key(key: str) -> str:
    """
    Normalizes a key by removing whitespace and punctuation, and converting to uppercase.
    This allows for robust matching even if there are formatting errors.
    """
    # Remove all whitespace and punctuation, then convert to uppercase
    return re.sub(r'[\s' + re.escape(string.punctuation) + ']', '', key).upper()

def escape_apostrophes(text: str) -> str:
    """
    Escapes apostrophes that are used as part of words (e.g. in "associate's" or "associates'").
    This function will insert a backslash before an apostrophe if it is between letters
    or if it follows a letter and is followed by whitespace or the end of the string.

    Important: We skip any apostrophe that comes right after "\n" (as in "\\n'")
    so that we avoid turning "\\n'\n" into "\\n\\'\n".
    --> This is fairly good!
    """
    # 1) Escape apostrophes between letters (e.g. "associate's"),
    #    but skip if preceded by "\n" --> use negative lookbehind `(?<!\\n)`.
    text = re.sub(
        r'(?<!\\n)(?<=[A-Za-z])\'(?=[A-Za-z])',
        r"\\'",
        text
    )
    
    # 2) Escape apostrophes after a letter if followed by whitespace or end-of-string (e.g. "associates'"),
    #    again skip if preceded by "\n".
    text = re.sub(
        r'(?<!\\n)(?<=[A-Za-z])\'(?=\s|$)',
        r"\\'",
        text
    )
    
    return text

def remove_template_dicts(dict_str: str) -> str:
    """
    Removes any dictionary fragments in dict_str that are template placeholders for 
    the keys 'ANSWER' and 'REVISED_DATA'. This includes:
      - An empty template (e.g. {'ANSWER':} or {'REVISED_DATA':})
      - A complete placeholder (e.g. {'ANSWER':\"\"\"string\"\"\"}, {'REVISED_DATA':\"\"\"string\"\"\"})
      - An incomplete placeholder (e.g. {'REVISED_DATA':\"\"\"string)
    """
    keys = ["ANSWER", "REVISED_DATA"]
    for key in keys:
        # Pattern for an empty template: e.g. {'KEY':}
        pattern_empty = re.compile(
            r"\{['\"]\s*" + re.escape(key) + r"\s*['\"]\s*:\s*\}",
            flags=re.DOTALL | re.IGNORECASE
        )
        # Pattern for a complete placeholder template: e.g. {'KEY':"""string"""}
        pattern_placeholder = re.compile(
            r"\{['\"]\s*" + re.escape(key) + r"\s*['\"]\s*:\s*\"\"\"string\"\"\"\}",
            flags=re.DOTALL | re.IGNORECASE
        )
        # Pattern for an incomplete placeholder: e.g. {'KEY':"""string
        pattern_incomplete = re.compile(
            r"\{['\"]\s*" + re.escape(key) + r"\s*['\"]\s*:\s*\"\"\"string.*?(?=\}|$)",
            flags=re.DOTALL | re.IGNORECASE
        )
        dict_str = pattern_empty.sub("", dict_str)
        dict_str = pattern_placeholder.sub("", dict_str)
        dict_str = pattern_incomplete.sub("", dict_str)
    return dict_str

################################################################################################################################################################################################
# Answer extraction -- actual functions
# The extract_answer() could be made more robust to formatting differences... but it seems to be okay empirically (no invalid answers when testing)
# Currently doesn't fallback to anything if the formatting of the model is wrong but it could do e.g. search for all options in the text
################################################################################################################################################################################################

def extract_answer(input_str: str) -> str:
    """
    Extracts the dictionary from the input string and returns the value associated with a key
    that normalizes to 'ANSWER'. If not found, returns "invalid_answer".
    """

    input_str = escape_apostrophes(input_str)
    input_str = remove_template_dicts(input_str)

    # Find the substring that looks like a dictionary
    start = input_str.find('{')
    end = input_str.rfind('}') + 1
    if start == -1 or end == -1:
        return "invalid_answer"
    
    dict_str = input_str[start:end]
    
    # Safely evaluate the dictionary string
    try:
        data_dict = ast.literal_eval(dict_str)
    except Exception:
        return "invalid_answer"
    
    # Define the normalized target key
    target = clean_key("ANSWER")
    
    # Iterate through the keys in the dictionary, normalize, and compare
    for key, value in data_dict.items():
        if clean_key(str(key)) == target:
            return str(value)
    
    return "invalid_answer"

def extract_revision(input_str: str) -> str:
    """
    Extracts the dictionary from the input string and returns the value associated with a key
    that normalizes to 'REVISED_DATA'. If not found, returns "dict_error".
    """

    # remove any anwer dictionaries from this string first!
    input_str = escape_apostrophes(input_str)
    input_str = remove_template_dicts(input_str)

    # Find the dictionary substring in the input string
    start = input_str.find('{')
    end = input_str.rfind('}') + 1
    if start == -1 or end == -1:
        return "dict_error"
    
    dict_str = input_str[start:end]
    
    # Safely evaluate the dictionary string using ast.literal_eval
    try:
        data_dict = ast.literal_eval(dict_str)
    except Exception:
        return "dict_error"
    
    # Define the normalized target key
    target = clean_key("REVISED_DATA")
    
    # Iterate through the keys, normalize, and compare
    for key, value in data_dict.items():
        if clean_key(str(key)) == target:
            return str(value)
    
    return "dict_error"

################################################################################################################################################################################################
# Counterfactual generation
################################################################################################################################################################################################

def select_counterfactual(answer, answer_choices):
    # Build a list of valid counterfactual options (i.e. all choices not equal to the answer)
    valid_options = [choice for choice in answer_choices if choice != answer]
    if len(valid_options) == 0:
        raise ValueError(f"No valid counterfactual options available for answer: {answer}")
    # If there are 2 or more options, randomly select one; otherwise, return the single option.
    return random.choice(valid_options) if len(valid_options) >= 2 else valid_options[0]

################################################################################################################################################################################################
# Edit distance functions
################################################################################################################################################################################################

def normalised_levenstein(text1: str, text2: str) -> float:
    """
    Compute the normalized levenstein distance between two texts (percentage)
    Lower is better (smaller edit)
    https://tedboy.github.io/nlps/generated/generated/nltk.edit_distance.html
    """
    return (nltk.edit_distance(text1, text2)*100) / max(len(text1), len(text2))


################################################################################################################################################################################################
################################################################################################################################################################################################
### Helper functions for postprocessing.py
################################################################################################################################################################################################
################################################################################################################################################################################################


############################################################################################################
# cleaning functions
############################################################################################################

def general_string2num(text: str) -> Union[float, str]:
    """
    Extracts the first number from a string and returns it as a float.
    Returns 'NA' if no number is found.
    """
    # return the float if it is already numeric
    if (type(text) == float) or (type(text) == int):
        return float(text)

    match = re.search(r'\d+', text)
    if match:
        return float(match.group())
    else:
        return "NA"


def edu2num(text: str) -> Union[int, str]:
    """
    Converts an education level description into an ordinal number.
    Returns pd.NA if the description is not found.
    """
    edu_dict = {
        'N/A - no schooling completed.': 1,
        'Nursery school / preschool.': 2,
        'Kindergarten.': 3,
        '1st grade only.': 4,
        '2nd grade.': 5,
        '3rd grade.': 6,
        '4th grade.': 7,
        '5th grade.': 8,
        '6th grade.': 9,
        '7th grade.': 10,
        '8th grade.': 11,
        '9th grade.': 12,
        '10th grade.': 13,
        '11th grade.': 14,
        '12th grade, no diploma.': 15,
        'Regular high school diploma.': 16,
        'GED or alternative credential.': 17,
        'Some college, less than 1 year.': 18,
        'Some college, 1 or more years, no degree.': 19,
        "Associate's degree.": 20,
        "Bachelor's degree.": 21,
        "Master's degree.": 22,
        "Professional degree beyond a bachelor's degree.": 23,
        'Doctorate degree.': 24
    }
    
    return edu_dict.get(text, None)

def set_column_dtypes(df: pd.DataFrame, dtype_map: dict, na_values: list = None) -> pd.DataFrame:
    """
    Safely set dtypes for specified columns in a pandas DataFrame while handling common missing value strings.

    This function converts columns to the desired data types using robust conversion methods.
    For numeric conversions ('int' and 'float'), it first replaces common missing value strings with pd.NA.

    Parameters:
    - df (pd.DataFrame): The DataFrame to modify.
    - dtype_map (dict): Dictionary mapping column names to desired dtypes.
                        Supported dtypes include 'category', 'float', 'int', 'bool', 'object', etc.
    - na_values (list, optional): List of strings to recognize as missing values.
                        Defaults to ["NA", "N/A", "null"].

    Returns:
    - pd.DataFrame: The updated DataFrame with converted dtypes.
    """
    # Set default missing value strings if none provided.
    if na_values is None:
        na_values = ["NA", "N/A", "null"]

    for col, dtype in dtype_map.items():
        if col in df.columns:
            try:
                # For numeric types, replace common missing value strings with pd.NA.
                if dtype in ['float', 'int']:
                    df[col] = df[col].replace(na_values, pd.NA)
                
                # Conversion using appropriate pandas functions for robust type casting.
                if dtype == 'float':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif dtype == 'int':
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')  # nullable integer type
                elif dtype == 'bool':
                    df[col] = df[col].astype('boolean')  # supports NA values
                else:
                    df[col] = df[col].astype(dtype)
            except Exception as e:
                print(f"Could not convert column '{col}' to {dtype}: {e}")
        else:
            print(f"Column '{col}' not found in DataFrame.")

    return df

def clean_data(entry):
    """ Remove any NA and change to None """
    return {k: (None if v == 'NA' else v) for k, v in entry.items()}

def make_grid(results_dict, variables):
    """
    Create an N-dimensional lookup grid for 'original_answer' values keyed by combinations
    of the specified variables.

    Args:
        results_dict (dict): Mapping identifiers to dicts that contain the given variables
                             and an 'original_answer' entry.
        variables (list of str): Names of the variables to index on.  The order of variables
                                 determines the nesting order in the returned dict.

    Returns:
        dict: A nested dictionary such that
              grid[val1][val2]...[valN] == original_answer,
              where val1 comes from variables[0], val2 from variables[1], etc.
    """
    if not variables:
        raise ValueError("`variables` must contain at least one variable name.")

    grid = {}
    for res in results_dict.values():
        # Walk/create the nested path for all but the last variable
        d = grid
        for var in variables[:-1]:
            key = res[var]
            d = d.setdefault(key, {})

        # At the final variable, assign the original_answer
        final_key = res[variables[-1]]
        d[final_key] = res['original_answer']

    return grid


######################################################################################################################################################

def income_postprocessing(results_, force=False):

    # check if results has already been processed
    results_dict = copy.deepcopy(results_)   
    results_dict = {int(k): v for k, v in results_dict.items()}
    if force==False:
        try:
            x = results_dict[0]['AGEP_CF']
            print("File already processed...")
            return results_dict
        except:
            x = ""

    # load the dataset dictionary (info about the dataset)
    with open(REPO_ROOT / 'src/models_datasets/datasets.json', 'r') as file:
        datasets_dict = json.load(file)
    ds_info = datasets_dict['income']
    dataset = load_from_disk(REPO_ROOT / ds_info['filepath'])#, name=ds_info['name'], split=ds_info['split'])
    variables = ['AGEP', 'SCHL']

    ########################################################################################################
    # for each result, do the data extraction
    for result in tqdm(results_dict.values()):

        # get the dataset id and update the results with the original numerical data for age and schooling. 
        dataset_id = result.get('dataset_id')
        ids = dataset["id"]
        id2row = {id_val: idx for idx, id_val in enumerate(ids)}

        result.update({'AGEP_OR':dataset[id2row[dataset_id]]['AGEP'], 'SCHL_OR':dataset[id2row[dataset_id]]['SCHL']})

        # Make sure everything is a float
        extract_floats = ['AGEP_OR', 'SCHL_OR', 'AGEP_CF', 'SCHL_CF']
        for variable in extract_floats:
            try:
                result[variable] = general_string2num(result[variable])
            except: 
                result[variable] = None

        # Ensure each counterfactual variable exists and is cast to float
        for var in variables:
            key = f"{var}_CF"
            if key not in result or result[key] is None:
                result[key] = None
            else:
                try:
                    result[key] = float(result[key])
                except ValueError:
                    result[key] = None

    ########################################################################################################
    # create look up grid of the decision boundary
    ########################################################################################################
    grid = make_grid(results_dict, ['AGEP_OR', 'SCHL_OR'])

    ########################################################################################################
    # fill some stats
    ########################################################################################################
    for key, value in results_dict.items():
        try:
            value['verification_answer'] = grid[value['AGEP_CF']][value['SCHL_CF']]
        except:
            value['verification_answer'] = 'NA' # could happen if grid not fully sampled
        
        if (value['original_answer'] == 'NA' or 
            value['followup_answer'] == 'NA' or 
            value['verification_answer'] == 'NA'):
            value['is_valid_counterfactual'] = "NA"
        else:
            value['is_valid_counterfactual'] = int(value['selected_complement'] == value['verification_answer'])

    return results_dict

######################################################################################################################################################

def income_distance(results_dict, distance_metric):

    # read in the relevant distance_matrix (ordered by the dataset)
    distance_matrix = np.load(REPO_ROOT / f'src/distance_matrices/income/income_{distance_metric}.npy') # src/

    # data <-> idx
    # idx  <-> data
    # idx <-> prediction
    data_2_idx = {}
    idx_2_prediction = {}
    for k,v in results_dict.items():
        idx = v['dataset_id']
        data = (v['AGEP_OR'], v['SCHL_OR'])
        prediction = v['original_answer']
        data_2_idx.update({data:idx})
        idx_2_prediction.update({idx:prediction})
    idx_2_data = {v:k for k,v in data_2_idx.items()}

    # Identify the minimum counterfactual and get the Gower's Distance to the minimal couterfactual
    above_keys = [k for k,v in idx_2_prediction.items() if v == "Above $50,000"]
    below_keys = [k for k,v in idx_2_prediction.items() if v == "Below $50,000"]

    # everything to just above or just below
    above_distance_matrix = distance_matrix[:, above_keys]
    below_distance_matrix = distance_matrix[:, below_keys]

    # mapping dicts
    above_idx_2_idx = {new_pos:orig_idx  for new_pos, orig_idx in enumerate(above_keys)}
    below_idx_2_idx = {new_pos:orig_idx  for new_pos, orig_idx in enumerate(below_keys)}
        
    # for each row
    for k,v in results_dict.items():

        # Get the Gower's Distance to the counterfactual --> Confirmed correct
        try:
            idx = v['dataset_id']
            counterfactual_idx = data_2_idx[(v['AGEP_CF'], v['SCHL_CF'])]
            v['gower_2_counterfactual'] = float(distance_matrix[idx, counterfactual_idx])
        except:
            v['gower_2_counterfactual'] = None

        try:
            idx = v['dataset_id']
            count_idx = data_2_idx[(v["AGEP_CF"], v["SCHL_CF"])]
            ##########################################################################################
            # Identify the minimum counterfactual (confirmed)
            # if original answer was above search from the below distances...
            if v['original_answer'] == "Above $50,000":
                min_counterfactual_idx = np.argmin(below_distance_matrix[idx,:])
                min_counterfactual_distance = below_distance_matrix[idx,min_counterfactual_idx]
                min_counterfactual_idx = below_idx_2_idx[min_counterfactual_idx]
                v['AGEP_CF_MIN'], v['SCHL_CF_MIN'] = idx_2_data[min_counterfactual_idx]
                v['gower_2_min_counterfactual'] = float(min_counterfactual_distance)

            # if original answer was below search from the above distances...
            if v['original_answer'] == "Below $50,000":
                min_counterfactual_idx = np.argmin(above_distance_matrix[idx,:])
                min_counterfactual_distance = above_distance_matrix[idx,min_counterfactual_idx]
                min_counterfactual_idx = above_idx_2_idx[min_counterfactual_idx]
                v['AGEP_CF_MIN'], v['SCHL_CF_MIN'] = idx_2_data[min_counterfactual_idx]
                v['gower_2_min_counterfactual'] = float(min_counterfactual_distance)

            # calculate RCF (confirmed)
            v['RCF'] = float(((v['gower_2_counterfactual'])/v['gower_2_min_counterfactual'])-1)
            v['RCF_A'] = float((v['gower_2_counterfactual']) - v['gower_2_min_counterfactual'])

            ##########################################################################################
            # counterfactual to decision boundary (confirmed)
            if v['verification_answer'] == "Above $50,000":
                v['counterfactual_2_decision_boundary'] = float(np.min(below_distance_matrix[count_idx,:]))
            if v['verification_answer'] == "Below $50,000":
                v['counterfactual_2_decision_boundary'] = float(np.min(above_distance_matrix[count_idx,:]))

            # ICF
            v['ICF'] = float(((v['counterfactual_2_decision_boundary'])/v['gower_2_min_counterfactual']))
            v['ICF_A'] = float(v['counterfactual_2_decision_boundary'])

            # Calcualte RCF_A if you want...

        except:
            v['AGEP_CF_MIN'], v['SCHL_CF_MIN'] = None, None
            v['gower_2_min_counterfactual'] = None
            v['RCF'] = None
            v['RCF'] = None
            v['counterfactual_2_decision_boundary'] = None
            v['ICF'] = None
            v['ICF_A'] = None

    return results_dict

############################################################################################################
# House prices
#############################################################################################################
def house_prices_postprocessing(results_, force=False):

    # check if results has already been processed
    results_dict = copy.deepcopy(results_)   
    results_dict = {int(k): v for k, v in results_dict.items()}
    if force==False:
        try:
            x = results_dict[0]['area_CF']
            print("File already processed...")
            return results_dict
        except:
            x = ""

    # load the dataset dictionary (info about the dataset)
    with open(REPO_ROOT / 'src/models_datasets/datasets.json', 'r') as file:
        datasets_dict = json.load(file)
    ds_info = datasets_dict['house_prices']
    dataset = load_from_disk(REPO_ROOT / ds_info['filepath'])#, name=ds_info['name'], split=ds_info['split'])
    variables = ['area', 'bedrooms', 'bathrooms', 'floors']

    ########################################################################################################
    # for each result, do the data extraction
    for result in tqdm(results_dict.values()):

        # get the dataset id and update the results with the original numerical data for age and schooling. 
        dataset_id = result.get('dataset_id')
        ids = dataset["id"]
        id2row = {id_val: idx for idx, id_val in enumerate(ids)}

        # Ensure each counterfactual variable exists and is cast to float
        for var in variables:
            key = f"{var}_CF"
            if key not in result or result[key] is None:
                result[key] = None
            else:
                try:
                    result[key] = float(result[key])
                except ValueError:
                    result[key] = None

        result.update({'area_OR':dataset[id2row[dataset_id]]['area'],
        'bedrooms_OR':dataset[id2row[dataset_id]]['bedrooms'],
        'bathrooms_OR':dataset[id2row[dataset_id]]['bathrooms'],
        'floors_OR':dataset[id2row[dataset_id]]['floors']})

        # Make sure everything is a float
        extract_floats = ['area_OR', 'bedrooms_OR', 'bathrooms_OR', 'floors_OR']
        for variable in extract_floats:
            try:
                result[variable] = general_string2num(result[variable])
            except: 
                result[variable] = None

    ########################################################################################################
    # create look up grid of the decision boundary
    ########################################################################################################
    grid = make_grid(results_dict, ['area_OR', 'bedrooms_OR', "bathrooms_OR", "floors_OR"])

    ########################################################################################################
    # fill some stats
    ########################################################################################################
    for key, value in results_dict.items():
        # add verification answer based on the counterfactual response
        # make sure you call it with the correct items!
        try:
            value['verification_answer'] = grid[value['area_CF']][value['bedrooms_CF']][value['bathrooms_CF']][value['floors_CF']]
        except:
            value['verification_answer'] = 'NA' # could happen if grid not fully sampled
        
        if (value['original_answer'] == 'NA' or 
            value['followup_answer'] == 'NA' or 
            value['verification_answer'] == 'NA'):
            value['is_valid_counterfactual'] = "NA"
            #value['prediction_matches_gt'] = "NA"
        else:
            value['is_valid_counterfactual'] = int(value['selected_complement'] == value['verification_answer'])
            #value['prediction_matches_gt'] = int(value['original_answer'] == value['ground_truth'])

    # count if any errors

    return results_dict

def house_prices_distance(results_dict, distance_metric):

    # read in the relevant distance_matrix (ordered by the dataset)
    distance_matrix = np.load(REPO_ROOT / f'src/distance_matrices/house_prices/house_prices_{distance_metric}.npy')

    # data <-> idx
    # idx  <-> data
    # idx <-> prediction
    data_2_idx = {}
    idx_2_prediction = {}
    for k,v in results_dict.items():
        idx = v['dataset_id']
        data = (v['area_OR'], v['bedrooms_OR'], v['bathrooms_OR'], v['floors_OR'])
        prediction = v['original_answer']
        data_2_idx.update({data:idx})
        idx_2_prediction.update({idx:prediction})
    idx_2_data = {v:k for k,v in data_2_idx.items()}

    # Identify the minimum counterfactual and get the Gower's Distance to the minimal couterfactual
    above_keys = [k for k,v in idx_2_prediction.items() if v == "Above $1,500,000"]
    below_keys = [k for k,v in idx_2_prediction.items() if v == "Below $1,500,000"]

    # everything to just above or just below
    above_distance_matrix = distance_matrix[:, above_keys]
    below_distance_matrix = distance_matrix[:, below_keys]

    # mapping dicts
    above_idx_2_idx = {new_pos:orig_idx  for new_pos, orig_idx in enumerate(above_keys)}
    below_idx_2_idx = {new_pos:orig_idx  for new_pos, orig_idx in enumerate(below_keys)}
        
    # for each row
    for k,v in results_dict.items():

        # Get the Gower's Distance to the counterfactual --> Confirmed correct
        try:
            idx = v['dataset_id']
            counterfactual_idx = data_2_idx[(v['area_CF'], v['bedrooms_CF'], v['bathrooms_CF'], v['floors_CF'])]
            v['gower_2_counterfactual'] = float(distance_matrix[idx, counterfactual_idx])
        except:
            v['gower_2_counterfactual'] = None

        try:
            idx = v['dataset_id']
            count_idx = data_2_idx[(v['area_CF'], v['bedrooms_CF'], v['bathrooms_CF'], v['floors_CF'])]
            ##########################################################################################
            # Identify the minimum counterfactual (confirmed)
            # if original answer was above search from the below distances...
            if v['original_answer'] == "Above $1,500,000":
                min_counterfactual_idx = np.argmin(below_distance_matrix[idx,:])
                min_counterfactual_distance = below_distance_matrix[idx,min_counterfactual_idx]
                min_counterfactual_idx = below_idx_2_idx[min_counterfactual_idx]
                v['area_CF_MIN'], v['bedrooms_CF_MIN'], v['bathrooms_CF_MIN'], v['floors_CF_MIN'] = idx_2_data[min_counterfactual_idx]
                v['gower_2_min_counterfactual'] = float(min_counterfactual_distance)

            # if original answer was below search from the above distances...
            if v['original_answer'] == "Below $1,500,000":
                min_counterfactual_idx = np.argmin(above_distance_matrix[idx,:])
                min_counterfactual_distance = above_distance_matrix[idx,min_counterfactual_idx]
                min_counterfactual_idx = above_idx_2_idx[min_counterfactual_idx]
                v['area_CF_MIN'], v['bedrooms_CF_MIN'], v['bathrooms_CF_MIN'], v['floors_CF_MIN'] = idx_2_data[min_counterfactual_idx]
                v['gower_2_min_counterfactual'] = float(min_counterfactual_distance)

            # calculate RCF (confirmed)
            v['RCF'] = float(((v['gower_2_counterfactual'])/v['gower_2_min_counterfactual'])-1)
            v['RCF_A'] = float((v['gower_2_counterfactual']) - v['gower_2_min_counterfactual'])

            ##########################################################################################
            # counterfactual to decision boundary (confirmed)
            if v['verification_answer'] == "Above $1,500,000":
                v['counterfactual_2_decision_boundary'] = float(np.min(below_distance_matrix[count_idx,:]))
            if v['verification_answer'] == "Below $1,500,000":
                v['counterfactual_2_decision_boundary'] = float(np.min(above_distance_matrix[count_idx,:]))

            # ICF
            v['ICF'] = float(((v['counterfactual_2_decision_boundary'])/v['gower_2_min_counterfactual']))
            v['ICF_A'] = float(v['counterfactual_2_decision_boundary'])

            # Calcualte RCF_A if you want...

        except:
            v['area_CF_MIN'], v['bedrooms_CF_MIN'], v['bathrooms_CF_MIN'], v['floors_CF_MIN'] = None, None, None, None
            v['gower_2_min_counterfactual'] = None
            v['RCF'] = None
            v['RCF'] = None
            v['counterfactual_2_decision_boundary'] = None
            v['ICF'] = None
            v['ICF_A'] = None

    return results_dict

############################################################################################################
# Heart disease
#############################################################################################################
def heart_disease_postprocessing(results_, force=False):

    # check if results has already been processed
    results_dict = copy.deepcopy(results_)   
    results_dict = {int(k): v for k, v in results_dict.items()}
    if force==False:
        try:
            x = results_dict[0]['total_cholesterol_CF']
            print("File already processed...")
            return results_dict
        except:
            x = ""

    # load the dataset dictionary (info about the dataset)
    with open(REPO_ROOT / 'src/models_datasets/datasets.json', 'r') as file:
        datasets_dict = json.load(file)
    ds_info = datasets_dict['heart_disease']
    dataset = load_from_disk(REPO_ROOT / ds_info['filepath'])#, name=ds_info['name'], split=ds_info['split'])
    variables = ['AGEP', 'SEX', 'systolic_bp', 'total_cholesterol']

    ########################################################################################################
    # for each result, do the data extraction
    for result in tqdm(results_dict.values()):

        # get the dataset id and update the results with the original numerical data for age and schooling. 
        dataset_id = result.get('dataset_id')
        ids = dataset["id"]
        id2row = {id_val: idx for idx, id_val in enumerate(ids)}

        # Ensure each counterfactual variable exists and is cast to float
        for var in variables:
            key = f"{var}_CF"
            if key not in result or result[key] is None:
                result[key] = None
            else:
                try:
                    result[key] = float(result[key])
                except ValueError:
                    result[key] = None

        result.update({'AGEP_OR':dataset[id2row[dataset_id]]['AGEP'],
        'SEX_OR':dataset[id2row[dataset_id]]['SEX'],
        'systolic_bp_OR':dataset[id2row[dataset_id]]['systolic_bp'],
        'total_cholesterol_OR':dataset[id2row[dataset_id]]['total_cholesterol']})

        # Make sure everything is a float
        extract_floats = ['AGEP_OR', 'systolic_bp_OR', 'total_cholesterol_OR']
        for variable in extract_floats:
            try:
                result[variable] = general_string2num(result[variable])
            except: 
                result[variable] = None

    ########################################################################################################
    # create look up grid of the decision boundary
    ########################################################################################################
    grid = make_grid(results_dict, ['AGEP_OR', 'SEX_OR', 'systolic_bp_OR', "total_cholesterol_OR"])

    ########################################################################################################
    # fill some stats
    ########################################################################################################
    for key, value in results_dict.items():
            # add verification answer based on the counterfactual response
        # make sure you call it with the correct items!
        try:
            value['verification_answer'] = grid[value['AGEP_CF']][value['SEX_CF']][value['systolic_bp_CF']][value['total_cholesterol_CF']]
        except:
            value['verification_answer'] = 'NA' # could happen if grid not fully sampled

        if (value['original_answer'] == 'NA' or 
            value['followup_answer'] == 'NA' or 
            value['verification_answer'] == 'NA'):
            value['is_valid_counterfactual'] = "NA"
            #value['prediction_matches_gt'] = "NA"
        else:
            value['is_valid_counterfactual'] = int(value['selected_complement'] == value['verification_answer'])
            #value['prediction_matches_gt'] = int(value['original_answer'] == value['ground_truth'])

    # count if any errors

    return results_dict

def heart_disease_distance(results_dict, distance_metric):

    # read in the relevant distance_matrix (ordered by the dataset)
    distance_matrix = np.load(REPO_ROOT / f'src/distance_matrices/heart_disease/heart_disease_{distance_metric}.npy')

    # data <-> idx
    # idx  <-> data
    # idx <-> prediction
    data_2_idx = {}
    idx_2_prediction = {}
    for k,v in results_dict.items():
        idx = v['dataset_id']
        data = (v['AGEP_OR'], v['SEX_OR'], v['systolic_bp_OR'], v['total_cholesterol_OR']) 
        prediction = v['original_answer']
        data_2_idx.update({data:idx})
        idx_2_prediction.update({idx:prediction})
    idx_2_data = {v:k for k,v in data_2_idx.items()}

    # Identify the minimum counterfactual and get the Gower's Distance to the minimal couterfactual
    above_keys = [k for k,v in idx_2_prediction.items() if v == "Heart disease"]
    below_keys = [k for k,v in idx_2_prediction.items() if v == "No heart disease"]

    # everything to just above or just below
    above_distance_matrix = distance_matrix[:, above_keys]
    below_distance_matrix = distance_matrix[:, below_keys]

    # mapping dicts
    above_idx_2_idx = {new_pos:orig_idx  for new_pos, orig_idx in enumerate(above_keys)}
    below_idx_2_idx = {new_pos:orig_idx  for new_pos, orig_idx in enumerate(below_keys)}
        
    # for each row
    for k,v in results_dict.items():

        # Get the Gower's Distance to the counterfactual --> Confirmed correct
        try:
            idx = v['dataset_id']
            counterfactual_idx = data_2_idx[(v['AGEP_CF'], v['SEX_CF'], v['systolic_bp_CF'], v['total_cholesterol_CF'])]
            v['gower_2_counterfactual'] = float(distance_matrix[idx, counterfactual_idx])
        except:
            v['gower_2_counterfactual'] = None

        try:
            idx = v['dataset_id']
            count_idx = data_2_idx[(v['AGEP_CF'], v['SEX_CF'], v['systolic_bp_CF'], v['total_cholesterol_CF'])]
            ##########################################################################################
            # Identify the minimum counterfactual (confirmed)
            # if original answer was above search from the below distances...
            if v['original_answer'] == "Heart disease":
                min_counterfactual_idx = np.argmin(below_distance_matrix[idx,:])
                min_counterfactual_distance = below_distance_matrix[idx,min_counterfactual_idx]
                min_counterfactual_idx = below_idx_2_idx[min_counterfactual_idx]
                v['AGEP_CF_MIN'], v['SEX_CF_MIN'], v['systolic_bp_CF_MIN'], v['total_cholesterol_CF_MIN'] = idx_2_data[min_counterfactual_idx]
                v['gower_2_min_counterfactual'] = float(min_counterfactual_distance)

            # if original answer was below search from the above distances...
            if v['original_answer'] == "No heart disease":
                min_counterfactual_idx = np.argmin(above_distance_matrix[idx,:])
                min_counterfactual_distance = above_distance_matrix[idx,min_counterfactual_idx]
                min_counterfactual_idx = above_idx_2_idx[min_counterfactual_idx]
                v['AGEP_CF_MIN'], v['SEX_CF_MIN'], v['systolic_bp_CF_MIN'], v['total_cholesterol_CF_MIN'] = idx_2_data[min_counterfactual_idx]
                v['gower_2_min_counterfactual'] = float(min_counterfactual_distance)

            # calculate RCF (confirmed)
            v['RCF'] = float(((v['gower_2_counterfactual'])/v['gower_2_min_counterfactual'])-1)
            v['RCF_A'] = float((v['gower_2_counterfactual']) - v['gower_2_min_counterfactual'])

            ##########################################################################################
            # counterfactual to decision boundary (confirmed)
            if v['verification_answer'] == "Heart disease":
                v['counterfactual_2_decision_boundary'] = float(np.min(below_distance_matrix[count_idx,:]))
            if v['verification_answer'] == "No heart disease":
                v['counterfactual_2_decision_boundary'] = float(np.min(above_distance_matrix[count_idx,:]))

            # ICF
            v['ICF'] = float(((v['counterfactual_2_decision_boundary'])/v['gower_2_min_counterfactual']))
            v['ICF_A'] = float(v['counterfactual_2_decision_boundary'])

            # Calcualte RCF_A if you want...

        except:
            v['AGEP_CF_MIN'], v['SEX_CF_MIN'], v['systolic_bp_CF_MIN'], v['total_cholesterol_CF_MIN'] = None, None, None, None
            v['gower_2_min_counterfactual'] = None
            v['RCF'] = None
            v['RCF_A'] = None
            v['counterfactual_2_decision_boundary'] = None
            v['ICF'] = None
            v['ICF_A'] = None

    return results_dict

################################################################################################################################################
################################################################################################################################################
################################################################################################################################################

############################################################################################################
# call the post-processing
############################################################################################################

def postprocessing(filepath, explicit_dataset, force=False, distance_metric="gower", save=True):
    """
    JSON -> JSON (just fills in data)

    This function performs the following steps:
      1. Loads JSON data from the specified filepath.
      2. Extracts the dataset name and model name from the filepath.
      3. Applies dataset-specific postprocessing
      4. Saves the updated JSON data back to the original filepath.

    Parameters:
    -----------
    filepath : str
        The path to the JSON file to be processed.

    Returns:
    --------
    None
    """
    # Load results from the file, extract the dataset name and model name from the filepath
    with open(filepath, 'r') as file:
        results = json.load(file)
    filepath = str(filepath)
    extracted_dataset_name = filepath.split("results/")[-1].split("/")[0]
    if len(extracted_dataset_name)==0:
        extracted_dataset_name = filepath.split("results_reasoning/")[-1].split("/")[0]
    if len(extracted_dataset_name)==0:
        extracted_dataset_name = filepath.split("results_temp_0/")[-1].split("/")[0]
    if len(extracted_dataset_name)==0:
        extracted_dataset_name = filepath.split("results_sensitivity/")[-1].split("/")[0]
    if len(extracted_dataset_name)==0:
        extracted_dataset_name = filepath.split("temperature_1/")[-1].split("/")[0]
    if len(extracted_dataset_name)==0:
        extracted_dataset_name = explicit_dataset 



    print(f"Extracted Dataset Name: {extracted_dataset_name}")
    if ("income" in extracted_dataset_name) or ("folktexts" in extracted_dataset_name):
        results = income_postprocessing(results, force=force) # initial data extraction
        results = income_distance(results, distance_metric) # calculate gower
        
    # house prices
    if "house_prices" in extracted_dataset_name:
        results = house_prices_postprocessing(results, force=force) # initial data extraction
        results = house_prices_distance(results, distance_metric) # calculate gower

    # heart disease
    if "heart_disease" in extracted_dataset_name:
        results = heart_disease_postprocessing(results, force=force) # initial data extraction
        results = heart_disease_distance(results, distance_metric) # calculate gower

    # Save the processed results back to the same filepath
    if save==True:
        with open(filepath, 'w') as file:
            json.dump(results, file, indent=4)
    
    return results


################################################################################################################################################################################################
################################################################################################################################################################################################
### Helper functions for scprer.py
################################################################################################################################################################################################
################################################################################################################################################################################################


def scoring_function(filepath, verbose=0, overwrite=False, detail=False):

    """
    BIG SCORING FUNCTION
    NOTE: This removes any incomplete generations prior to scoring
    Works fairly well but could do with being cleaned
    Also returns the exact match
    """

    # load model dict
    # models_path = os.path.join(os.path.dirname(__file__), "models_datasets", "models.json")
    with open(REPO_ROOT / 'src/models_datasets/models.json', 'r') as file:
        model_dict = json.load(file)
    with open(filepath, 'r') as file:
        results = json.load(file)

    ##########################################################################################
    # simple check to see if post-processing has been done
    ##########################################################################################
    try:
        sample_key = random.choice(list(results))
        results[sample_key]['is_valid_counterfactual']
    except KeyError:
        print("Error: Unable to score results. Please run the postprocessing step before scoring.")
        return "Error"
        

    ##########################################################################################
    # Record and drop invalid answers
    ##########################################################################################
    invalid_list = ["NA", None]
    invalid_original_answers = []
    invalid_counterfactual_generations = []
    invalid_verification_answers = []

    for key, value in results.items(): 
        invalid_original_answers.append(int(value['original_answer'] in  invalid_list))
        invalid_counterfactual_generations.append(int(value['followup_answer'] in  invalid_list))
        invalid_verification_answers.append(int(value['verification_answer'] in  invalid_list))

    # make overall error list
    overall_errors = np.maximum.reduce([
        invalid_original_answers,
        invalid_counterfactual_generations,
        invalid_verification_answers
    ]).tolist()

    # filter out generation errors
    filtered_results = {k: v for k, v, err in zip(results.keys(), results.values(), overall_errors) if err == 0}

    # short generation warning (no filtering)
    # trigger warning if counterfactual is less than half the length of the original input
    short_gens = []
    for key, value in results.items(): 
        short_gens.append(int(len(value['followup_answer']) < (len(value['original_input']))/2))

    short_gens_indices = [i for i, val in enumerate(short_gens) if val == 1]

    ##########################################################################################
    # score valid generations
    ##########################################################################################

    valid_l = []
    distance_2_counterfactuals = []
    counterfactual_2_decision_boundarys = []
    RCFs = []
    ICFs = []

    # absolute versions
    RCF_As = []
    ICF_As = []

    # exact hits
    exact_hit = []

    for key, value in filtered_results.items():
        valid_l.append(int(value['is_valid_counterfactual']))
        distance_2_counterfactuals.append(value['gower_2_counterfactual'])
        counterfactual_2_decision_boundarys.append(value['counterfactual_2_decision_boundary'])
        RCFs.append(value['RCF'])
        ICFs.append(value['ICF'])
        RCF_As.append(value['RCF_A'])
        ICF_As.append(value['ICF_A'])

        if value['RCF']==float(0) and int(value['is_valid_counterfactual'])==1:
            exact_hit.append(value['dataset_id'])

    # RCFs_valid and ICFs_invalid
    RCFs_clean = [value for flag, value in zip(valid_l, RCFs) if flag == 1]
    RCF_As_clean = [value for flag, value in zip(valid_l, RCF_As) if flag == 1]
    ICFs_clean = [value for flag, value in zip(valid_l, ICFs) if flag == 0]
    ICF_As_clean = [value for flag, value in zip(valid_l, ICF_As) if flag == 0]
    

    # FIXED!
    # minimum_CF_invalid = [value for flag, value in zip(valid_l, minimal_CF_distance) if (flag == 0) and (not pd.isna(value))]
    # counterfactual_to_decision_boundarys_invalid = [value for flag, value, indicator in zip(valid_l, counterfactual_to_decision_boundarys, minimal_CF_distance) if (flag == 0) and (not pd.isna(indicator))]
    # if verbose>0:
    #     print(minimum_CF_invalid)
    # try:
    #     ICF = [x/y for x,y in zip(counterfactual_to_decision_boundarys_invalid, minimum_CF_invalid)]
    # except:
    #     ICF = "NA"

    # validity
    validity = np.mean(valid_l).item() # accuracy given valid generations
    validity_percent = validity*100

    # Mean distance of interventions
    mean_distance = np.nanmean(np.array(distance_2_counterfactuals, dtype=float))

    # RCFs
    mean_RCF = np.nanmean(np.array(RCFs_clean, dtype=float))
    medican_RCF = np.nanmedian(np.array(RCFs_clean, dtype=float))

    # RCF_As
    mean_RCF_A = np.nanmean(np.array(RCF_As_clean, dtype=float))
    medican_RCF_A = np.nanmedian(np.array(RCF_As_clean, dtype=float))

    # ICFs
    mean_ICF = np.nanmean(np.array(ICFs_clean, dtype=float))
    medican_ICF = np.nanmedian(np.array(ICFs_clean, dtype=float))

    # ICF_As
    mean_ICF_A = np.nanmean(np.array(ICF_As_clean, dtype=float))
    medican_ICF_A = np.nanmedian(np.array(ICF_As_clean, dtype=float))

    # Total minimality [0 upwards]
    medican_ICF_nonan = np.nan_to_num(medican_ICF, nan=0.0)
    TM = validity*medican_RCF + (1-validity)*medican_ICF_nonan

    # absolute distance -- Use the mean for this...
    medican_ICF_A_nonan = np.nan_to_num(mean_ICF_A, nan=0.0)
    TM_A = validity*mean_RCF_A + (1-validity)*medican_ICF_A_nonan

    # get exact_hit stats
    hits = len(exact_hit)
    hits_percent = (hits*100)/len(filtered_results)

    # results in a dict
    scoring_results = {
        "validity_percent": validity_percent,
        "mean_distance":mean_distance,
        "mean_RCF":mean_RCF,
        "median_RCF":medican_RCF,
        "mean_RCF_A":mean_RCF_A,
        "median_RCF_A":medican_RCF_A,
        "mean_ICF":mean_ICF,
        "median_ICF":medican_ICF,
        "mean_ICF_A":mean_ICF_A,
        "median_ICF_A":medican_ICF_A,
        "total_minimality":TM,
        "total_minimality_A":TM_A,
        "exact_match":hits_percent,
        }
    
    # results in a dict
    scoring_results_detailed = {
        "validity_percent": validity_percent,
        "Distances":distance_2_counterfactuals,
        "RCFs":RCFs_clean,
        "ICFs":ICFs_clean}
    
    ################################
    if verbose>0:
        print("\n")
        print("-"*100)
        print("Generation Errors + Warnings:")
        print(f"Original Answer Errors:\t\t {invalid_original_answers.count(1)}")
        print(f"Counterfactual Errors:\t\t {invalid_counterfactual_generations.count(1)}")
        print(f"Validation Answer Errors:\t {invalid_verification_answers.count(1)}")
        print(f"Overall Errors (dropped):\t {overall_errors.count(1)}")
        print(f"Short counterfactuals:\t\t {short_gens.count(1)}. {short_gens_indices}")
        print("")
        print("Key data:")
        print(f"Filepath:\t\t\t\t {filepath}")
        print(f"Sample size:\t\t\t {len(valid_l)}")
        print(f"Validity:\t\t\t {validity_percent:0.2f}%") 
        print("")
        print("Distance and Minimality:")
        print(f"Mean intervention distance:\t {mean_distance:0.4f}")
        print(f"Mean RCF:\t\t\t {mean_RCF:0.4f}") # task accuracy
        print(f"Median RCF:\t\t\t {medican_RCF:0.4f}") # task accuracy
        print(f"Mean ICF:\t\t\t {mean_ICF:0.4f}") 
        print(f"Median ICF:\t\t\t {medican_ICF:0.4f}")
        print(f"Total Minimality:\t\t {TM:0.4f}")
        print("")
        print("Absolutes:")
        print(f"Mean RCF_A:\t\t\t {mean_RCF_A:0.4f}") # task accuracy
        print(f"Median RCF_A:\t\t\t {medican_RCF_A:0.4f}") # task accuracy
        print(f"Mean ICF_A:\t\t\t {mean_ICF_A:0.4f}") 
        print(f"Median ICF_A:\t\t\t {medican_ICF_A:0.4f}")
        print(f"Total Minimality_A:\t\t {TM_A:0.4f}")
        print("-"*100)
        print(f"Exact Match:\t\t\t {hits_percent:0.4f}")
        print("\n")
    ################################

    # write scoring results back to the filepath
    if overwrite==True:
        with open(filepath, "w") as f:
            json.dump(scoring_results, f, indent=2)
        
    if detail==True:
        return scoring_results_detailed

    # return the scoring results regardless
    return scoring_results