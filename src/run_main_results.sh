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


###############################################################################################################
# house prices
###############################################################################################################

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model microsoft/phi-4 \
  --dataset house_prices \
  --sample_size 5000 --batch_size 50 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model microsoft/phi-4 \
  --dataset house_prices_minimal \
  --sample_size 5000 --batch_size 50 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model google/gemma-2-2b-it \
  --dataset house_prices \
  --sample_size 5000 --batch_size 50 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model google/gemma-2-2b-it \
  --dataset house_prices_minimal \
  --sample_size 5000 --batch_size 50 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model google/gemma-2-9b-it \
  --dataset house_prices \
  --sample_size 5000 --batch_size 30 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model google/gemma-2-9b-it \
  --dataset house_prices_minimal \
  --sample_size 5000 --batch_size 30 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model google/gemma-2-27b-it \
  --dataset house_prices \
  --sample_size 5000 --batch_size 20 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model google/gemma-2-27b-it \
  --dataset house_prices_minimal \
  --sample_size 5000 --batch_size 20 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES_70B main_script.py \
  --task_model meta-llama/Llama-3.3-70B-Instruct \
  --dataset house_prices \
  --sample_size 5000 --batch_size 100 --max_tokens 1000 --temperature 0 --big_model

accelerate launch --num_processes=$NUM_PROCESSES_70B main_script.py \
  --task_model meta-llama/Llama-3.3-70B-Instruct \
  --dataset house_prices_minimal \
  --sample_size 5000 --batch_size 100 --max_tokens 1000 --temperature 0 --big_model

accelerate launch --num_processes=$NUM_PROCESSES_70B main_script.py \
  --task_model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --dataset house_prices \
  --sample_size 5000 --batch_size 20 --max_tokens 3000 --temperature 0 --big_model

accelerate launch --num_processes=$NUM_PROCESSES_70B main_script.py \
  --task_model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --dataset house_prices_minimal \
  --sample_size 5000 --batch_size 20 --max_tokens 3000 --temperature 0 --big_model


# ###############################################################################################################
# # heart disease
# ###############################################################################################################

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model microsoft/phi-4 \
  --dataset heart_disease \
  --sample_size 5000 --batch_size 50 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model microsoft/phi-4 \
  --dataset heart_disease_minimal \
  --sample_size 5000 --batch_size 50 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model google/gemma-2-2b-it \
  --dataset heart_disease \
  --sample_size 5000 --batch_size 50 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model google/gemma-2-2b-it \
  --dataset heart_disease_minimal \
  --sample_size 5000 --batch_size 50 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model google/gemma-2-9b-it \
  --dataset heart_disease \
  --sample_size 5000 --batch_size 30 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model google/gemma-2-9b-it \
  --dataset heart_disease_minimal \
  --sample_size 5000 --batch_size 30 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model google/gemma-2-27b-it \
  --dataset heart_disease \
  --sample_size 5000 --batch_size 20 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model google/gemma-2-27b-it \
  --dataset heart_disease_minimal \
  --sample_size 5000 --batch_size 20 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES_70B main_script.py \
  --task_model meta-llama/Llama-3.3-70B-Instruct \
  --dataset heart_disease \
  --sample_size 5000 --batch_size 100 --max_tokens 1000 --temperature 0 --big_model

accelerate launch --num_processes=$NUM_PROCESSES_70B main_script.py \
  --task_model meta-llama/Llama-3.3-70B-Instruct \
  --dataset heart_disease_minimal \
  --sample_size 5000 --batch_size 100 --max_tokens 1000 --temperature 0 --big_model

accelerate launch --num_processes=$NUM_PROCESSES_70B main_script.py \
  --task_model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --dataset heart_disease \
  --sample_size 5000 --batch_size 20 --max_tokens 3000 --temperature 0 --big_model

accelerate launch --num_processes=$NUM_PROCESSES_70B main_script.py \
  --task_model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --dataset heart_disease_minimal \
  --sample_size 5000 --batch_size 20 --max_tokens 3000 --temperature 0 --big_model

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model google/gemma-3-1b-it \
  --dataset income \
  --sample_size 5000 --batch_size 100 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model google/gemma-3-1b-it \
  --dataset income_minimal \
  --sample_size 5000 --batch_size 100 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model google/gemma-3-4b-it \
  --dataset income \
  --sample_size 5000 --batch_size 50 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model google/gemma-3-4b-it \
  --dataset income_minimal \
  --sample_size 5000 --batch_size 50 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model google/gemma-3-12b-it \
  --dataset income \
  --sample_size 5000 --batch_size 50 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model google/gemma-3-12b-it \
  --dataset income_minimal \
  --sample_size 5000 --batch_size 50 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model google/gemma-3-27b-it \
  --dataset income \
  --sample_size 5000 --batch_size 50 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model google/gemma-3-27b-it \
  --dataset income_minimal \
  --sample_size 5000 --batch_size 50 --max_tokens 1000 --temperature 0

# ################################################################################################################
# income
# ###############################################################################################################

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model google/gemma-2-2b-it \
  --dataset income \
  --sample_size 5000 --batch_size 50 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model google/gemma-2-2b-it \
  --dataset income_minimal \
  --sample_size 5000 --batch_size 50 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model google/gemma-2-9b-it \
  --dataset income \
  --sample_size 5000 --batch_size 30 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model google/gemma-2-9b-it \
  --dataset income_minimal \
  --sample_size 5000 --batch_size 30 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model google/gemma-2-27b-it \
  --dataset income \
  --sample_size 5000 --batch_size 20 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model google/gemma-2-27b-it \
  --dataset income_minimal \
  --sample_size 5000 --batch_size 20 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model meta-llama/Llama-3.2-1B-Instruct \
  --dataset income \
  --sample_size 5000 --batch_size 200 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model meta-llama/Llama-3.2-1B-Instruct \
  --dataset income_minimal \
  --sample_size 5000 --batch_size 200 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model meta-llama/Llama-3.2-3B-Instruct \
  --dataset income \
  --sample_size 5000 --batch_size 200 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model meta-llama/Llama-3.2-3B-Instruct \
  --dataset income_minimal \
  --sample_size 5000 --batch_size 200 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model meta-llama/Llama-3.1-8B-Instruct \
  --dataset income \
  --sample_size 5000 --batch_size 150 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES main_script.py \
  --task_model meta-llama/Llama-3.1-8B-Instruct \
  --dataset income_minimal \
  --sample_size 5000 --batch_size 150 --max_tokens 1000 --temperature 0

accelerate launch --num_processes=$NUM_PROCESSES_70B main_script.py \
  --task_model meta-llama/Llama-3.3-70B-Instruct \
  --dataset income \
  --sample_size 5000 --batch_size 100 --max_tokens 1000 --temperature 0 --big_model

accelerate launch --num_processes=$NUM_PROCESSES_70B main_script.py \
  --task_model meta-llama/Llama-3.3-70B-Instruct \
  --dataset income_minimal \
  --sample_size 5000 --batch_size 100 --max_tokens 1000 --temperature 0 --big_model

accelerate launch --num_processes=$NUM_PROCESSES_70B main_script.py \
  --task_model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --dataset income \
  --sample_size 5000 --batch_size 20 --max_tokens 3000 --temperature 0 --big_model

accelerate launch --num_processes=$NUM_PROCESSES_70B main_script.py \
  --task_model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --dataset income_minimal \
  --sample_size 5000 --batch_size 30 --max_tokens 3000 --temperature 0 --big_model
