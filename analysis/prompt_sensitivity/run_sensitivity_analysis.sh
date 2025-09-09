#!/bin/bash

# Configuration
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export VLLM_DISABLE_COMPILE_CACHE=1
RESULTS_DIR="analysis/sensitivity_analysis/results"

# Create results directory if it doesn't exist
mkdir -p $RESULTS_DIR

# Run experiments for house_prices datasets
for i in {0..19}
do
  DATASET_NAME="house_prices_$i"
  OUTPUT_FILE="${RESULTS_DIR}/${DATASET_NAME}.json"
  echo "Running experiment for $DATASET_NAME -> $OUTPUT_FILE"
  
  CUDA_VISIBLE_DEVICES="0,1,2,3" python src/main.py \
    --task_model meta-llama/Llama-3.3-70B-Instruct \
    --dataset $DATASET_NAME \
    --sample_size 5000 --max_tokens 1000 --temperature 0 \
    --tensor_parallel_size 4 --gpu_memory_utilization 0.90 \
    --output_filepath $OUTPUT_FILE
done

# Run experiments for house_prices_minimal datasets
for i in {0..19}
do
  DATASET_NAME="house_prices_minimal_$i"
  OUTPUT_FILE="${RESULTS_DIR}/${DATASET_NAME}.json"
  echo "Running experiment for $DATASET_NAME -> $OUTPUT_FILE"
  
  CUDA_VISIBLE_DEVICES="0,1,2,3" python src/main.py \
    --task_model meta-llama/Llama-3.3-70B-Instruct \
    --dataset $DATASET_NAME \
    --sample_size 5000 --max_tokens 1000 --temperature 0 \
    --tensor_parallel_size 4 --gpu_memory_utilization 0.90 \
    --output_filepath $OUTPUT_FILE
done
