#!/bin/bash


learning_rates=(3e-4)
lora_alphas=(24 )
lora_ranks=(16) 


BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
DATA_PATH='/root/autodl-tmp/Dislora/commonsense_170k.json'

DATASET_NAME=$(basename "${DATA_PATH}" .json)

BATCH_SIZE=36
MICRO_BATCH_SIZE=4
NUM_EPOCHS=1
CUTOFF_LEN=256
VAL_SET_SIZE=120
USE_GRADIENT_CHECKPOINTING=False
ADAPTER_NAME=mylora
TARGET_MODULES='[q_proj,v_proj,k_proj,o_proj]'
WANDB_PROJECT='llm-adapters-math'
WANDB_WATCH='gradients'
WANDB_LOG_MODEL='false'


for lr in "${learning_rates[@]}"; do
  for alpha in "${lora_alphas[@]}"; do
    for rank in "${lora_ranks[@]}"; do

      output_dir="./trained_models/qwen-lora-${DATASET_NAME}-lr${lr}-alpha${alpha}-rank${rank}-epoch3"
      wandb_run_name="qwen-lora-${DATASET_NAME}-lr${lr}-alpha${alpha}-rank${rank}"
      if [ -d "$output_dir" ]; then
        echo "Skip the existing training combinations: lr=${lr}, alpha=${alpha}, rank=${rank}"
        continue
      fi
      echo "Running with dataset=${DATASET_NAME}, lr=${lr}, lora_alpha=${alpha}, lora_r=${rank}"
      echo "Output directory: ${output_dir}"
      echo "Wandb run name: ${wandb_run_name}"

      deepspeed --num_gpus=3 finetune_MyLoRA.py \
        --base_model "${BASE_MODEL}" \
        --data_path "${DATA_PATH}" \
        --output_dir "${output_dir}" \
        --batch_size ${BATCH_SIZE} \
        --micro_batch_size ${MICRO_BATCH_SIZE} \
        --num_epochs ${NUM_EPOCHS} \
        --learning_rate ${lr} \
        --cutoff_len ${CUTOFF_LEN} \
        --val_set_size ${VAL_SET_SIZE} \
        --use_gradient_checkpointing ${USE_GRADIENT_CHECKPOINTING} \
        --adapter_name ${ADAPTER_NAME} \
        --lora_target_modules "${TARGET_MODULES}" \
        --lora_r ${rank} \
        --lora_alpha ${alpha} \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_run_name "${wandb_run_name}" \
        --wandb_watch "${WANDB_WATCH}" \
        --wandb_log_model "${WANDB_LOG_MODEL}"

      echo "Finished run with dataset=${DATASET_NAME}, lr=${lr}, lora_alpha=${alpha}, lora_r=${rank}"
      echo "-------------------------------------------------"
    done
  done
done

echo "All runs completed."