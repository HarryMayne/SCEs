#!/bin/bash

####################################################################################
# Run the main results of the paper
# Designed for 4 H100s
# bash run_main_results.sh
####################################################################################


# Configuration
export CUDA_VISIBLE_DEVICES="0,1,2,3"
NUM_PROCESSES=4
NUM_PROCESSES_70B=1

# GPT-4.1
####################################################################################

# CUDA_VISIBLE_DEVICES="1" python src/main.py   --task_model gpt-4.1-2025-04-14   --dataset income   --sample_size 2000 --temperature 1.0   --max_tokens 4000   --tensor_parallel_size 1 --max_concurrent 500 --wait 30 --output_filepath analysis/temperature_1/income/gpt4_1.json

# CUDA_VISIBLE_DEVICES="1" python src/main.py   --task_model gpt-4.1-2025-04-14   --dataset income_minimal   --sample_size 2000 --temperature 1.0   --max_tokens 4000   --tensor_parallel_size 1 --max_concurrent 500 --wait 30 --output_filepath analysis/temperature_1/income_minimal/gpt4_1.json

# CUDA_VISIBLE_DEVICES="1" python src/main.py   --task_model gpt-4.1-2025-04-14   --dataset house_prices   --sample_size 2000 --temperature 1.0   --max_tokens 4000   --tensor_parallel_size 1 --max_concurrent 500 --wait 30 --output_filepath analysis/temperature_1/house_prices/gpt4_1.json

# CUDA_VISIBLE_DEVICES="1" python src/main.py   --task_model gpt-4.1-2025-04-14   --dataset house_prices_minimal   --sample_size 2000 --temperature 1.0   --max_tokens 4000   --tensor_parallel_size 1 --max_concurrent 500 --wait 30 --output_filepath analysis/temperature_1/house_prices_minimal/gpt4_1.json

# CUDA_VISIBLE_DEVICES="1" python src/main.py   --task_model gpt-4.1-2025-04-14   --dataset heart_disease   --sample_size 2000 --temperature 1.0   --max_tokens 4000   --tensor_parallel_size 1 --max_concurrent 500 --wait 30 --output_filepath analysis/temperature_1/heart_disease/gpt4_1.json

# CUDA_VISIBLE_DEVICES="1" python src/main.py   --task_model gpt-4.1-2025-04-14   --dataset heart_disease_minimal   --sample_size 2000 --temperature 1.0   --max_tokens 4000   --tensor_parallel_size 1 --max_concurrent 500 --wait 30 --output_filepath analysis/temperature_1/heart_disease_minimal/gpt4_1.json

# Claude Sonnet 3.7
####################################################################################

CUDA_VISIBLE_DEVICES="1" python src/main.py   --task_model claude-3-7-sonnet-20250219   --dataset income                --sample_size 2000 --max_tokens 4000 --tensor_parallel_size 1 --max_concurrent 25 --wait 60 --extended_thinking disabled --output_filepath analysis/temperature_1/income/claude_3_7_sonnet.json --temperature 1.0

CUDA_VISIBLE_DEVICES="1" python src/main.py   --task_model claude-3-7-sonnet-20250219   --dataset income_minimal        --sample_size 2000 --max_tokens 4000 --tensor_parallel_size 1 --max_concurrent 25 --wait 60 --extended_thinking disabled --output_filepath analysis/temperature_1/income_minimal/claude_3_7_sonnet.json --temperature 1.0

CUDA_VISIBLE_DEVICES="1" python src/main.py   --task_model claude-3-7-sonnet-20250219   --dataset house_prices           --sample_size 2000 --max_tokens 4000 --tensor_parallel_size 1 --max_concurrent 25 --wait 60 --extended_thinking disabled --output_filepath analysis/temperature_1/house_prices/claude_3_7_sonnet.json --temperature 1.0

CUDA_VISIBLE_DEVICES="1" python src/main.py   --task_model claude-3-7-sonnet-20250219   --dataset house_prices_minimal   --sample_size 2000 --max_tokens 4000 --tensor_parallel_size 1 --max_concurrent 25 --wait 60 --extended_thinking disabled --output_filepath analysis/temperature_1/house_prices_minimal/claude_3_7_sonnet.json --temperature 1.0

CUDA_VISIBLE_DEVICES="1" python src/main.py   --task_model claude-3-7-sonnet-20250219   --dataset heart_disease          --sample_size 2000 --max_tokens 4000 --tensor_parallel_size 1 --max_concurrent 25 --wait 60 --extended_thinking disabled --output_filepath analysis/temperature_1/heart_disease/claude_3_7_sonnet.json --temperature 1.0

CUDA_VISIBLE_DEVICES="1" python src/main.py   --task_model claude-3-7-sonnet-20250219   --dataset heart_disease_minimal  --sample_size 2000 --max_tokens 4000 --tensor_parallel_size 1 --max_concurrent 25 --wait 60 --extended_thinking disabled --output_filepath analysis/temperature_1/heart_disease_minimal/claude_3_7_sonnet.json --temperature 1.0
