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
import google.generativeai as genai

import os, logging, concurrent.futures as cf
from tqdm.auto import tqdm
import openai
from openai import AsyncOpenAI
import asyncio

from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    wait_exponential_jitter,
    wait_exponential,
)

sys.path.insert(0, "../../src")
sys.path.insert(0, "../..")
from config import REPO_ROOT


import warnings
warnings.filterwarnings("ignore")

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

# 1. vLLM ───────────────────────────────────────────────────────────────────
class VllmClient:
    def __init__(self, model_name: str, dtype: str, tensor_parallel_size: int, max_concurrent, gpu_memory_utilization, wait, pipeline_parallel_size):
        self.llm = LLM(model=model_name,
                       dtype=dtype,
                       tensor_parallel_size=tensor_parallel_size,
                       pipeline_parallel_size=pipeline_parallel_size,
                       trust_remote_code=True,
                       gpu_memory_utilization= gpu_memory_utilization,
                       )

    def chat(self, messages: List[List[Dict[str, str]]], temperature: float,
             max_tokens: int, seed: int, log_probs):
        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            logprobs=log_probs or None,          # 👈 new
        )
        outs = self.llm.chat(messages, sampling_params=params, use_tqdm=True)
        return [
            (o.outputs[0].text, o.outputs[0].logprobs if log_probs else None)
            for o in outs
]

# 2. OPENAI ────────────────────────────────────────────────────────────────
class OpenAIClient:
    """
    Async OpenAI client for lists of conversations with concurrency limit and retries.
    """
    def __init__(self, model_name, max_concurrent=8, **_):
        """
        :param model_name: the OpenAI model to use (e.g., "gpt-4o").
        :param max_concurrent: maximum number of concurrent requests.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model_name
        self._sem = asyncio.Semaphore(max_concurrent)

    @backoff.on_exception(
        backoff.expo,
        (Exception,),  # retry on any Exception like rate limits or server errors
        max_tries=4,
        max_time=240,
    )
    async def _safe_call(self, conv, temperature, max_tokens):
        """Send a single conversation with retry and concurrency control."""
        async with self._sem:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=conv,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

    def chat(self, messages, temperature=0.0, max_tokens=256, **_):
        """
        Synchronously process a list of conversations by spinning up a temporary
        event loop around the existing async retry logic.
        """
        async def _gather_all():
            tasks = [self._safe_call(conv, temperature, max_tokens)
                     for conv in messages]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            out = []
            for r in results:
                if isinstance(r, Exception):
                    print(f"Error in conversation: {r}")
                    out.append(None)
                else:
                    out.append(r)
            return out
 
        # Run our async gather in its own loop, and return the List[str]
        return asyncio.run(_gather_all())


# 3. OPENAI ────────────────────────────────────────────────────────────────

@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=61, max=240),
)
def _safe_anthropic_call(client, model, conv, temperature, max_tokens):
    resp = client.messages.create(
        model       = model,
        messages    = conv,
        temperature = temperature,
        max_tokens  = max_tokens,
    )
    # adjust this if your SDK returns differently
    return getattr(resp, "completion", resp.content)

class AnthropicClient:
    """
    Thread-pooled Anthropic wrapper with the same interface & pacing logic
    as our OpenAIClient.
    """
    def __init__(self, model_name: str, max_concurrent: int = 8, wait: float = 5.0, **_):
        # set up Anthropic SDK
        self.client      = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model       = model_name
        self.max_workers = max_concurrent
        self.wait        = wait

    def chat(self, messages, temperature, max_tokens, **_):
        # support single vs batch
        batched, already = _ensure_batch(messages)
        n        = len(batched)
        outs     = [None] * n
        errs     = []
        completed = 0

        with cf.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            fut_map = {
                pool.submit(
                    _safe_anthropic_call,
                    self.client,
                    self.model,
                    conv,
                    temperature,
                    max_tokens
                ): idx
                for idx, conv in enumerate(batched)
            }

            for fut in tqdm(
                cf.as_completed(fut_map),
                total=n,
                desc=f"Anthropic {self.model}",
                unit="chat",
            ):
                idx = fut_map[fut]
                try:
                    outs[idx] = fut.result()
                except Exception as e:
                    logging.warning("Anthropic chat %d failed: %s", idx, e)
                    errs.append((idx, str(e)))
                finally:
                    completed += 1
                    # throttle after each batch of `max_workers`
                    if self.wait > 0 and completed % self.max_workers == 0:
                        time.sleep(self.wait)

        if errs:
            logging.info("Anthropic completed with %d errors", len(errs))
        return outs if already else outs[0]

# 4. GOOGLE GEMINI ────────────────────────────────────────────────────────────
# TO FIX --> MAKE IT THE SAME AS OPENAI
# class GeminiClient:
#     def __init__(self, model_name: str, **_):
#         genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#         self.model = genai.GenerativeModel(model_name)     # official SDK object :contentReference[oaicite:1]{index=1}

#     @staticmethod
#     def _to_gemini(conv):
#         """canonical role/content → Gemini contents list"""
#         out = []
#         for m in conv:
#             role = "user" if m["role"] in ("user", "system") else "model"
#             out.append({"role": role, "parts": [{"text": m["content"]}]})
#         return out                                        # structure from docs :contentReference[oaicite:2]{index=2}

#     def chat(self, messages, temperature, max_tokens, seed=0, **_):
#         batched, already = _ensure_batch(messages)
#         outs = []

#         for conv in tqdm(batched,
#                          desc=f"Gemini {self.model._model_name}",
#                          unit="chat",
#                          total=len(batched),
#                          leave=False):
#             rsp = self.model.generate_content(             # synchronous call :contentReference[oaicite:3]{index=3}
#                 self._to_gemini(conv),
#                 generation_config={
#                     "temperature":       temperature,
#                     "max_output_tokens": max_tokens,
#                     "seed":              seed,
#                 },
#             )
#             outs.append(rsp.text)
#         return outs if already else outs[0]

# 5. FACTORY ────────────────────────────────────────────────────────────────
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
                return f'{m.group("prefix")} "{m.group("key").strip().strip("\'")}":'
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