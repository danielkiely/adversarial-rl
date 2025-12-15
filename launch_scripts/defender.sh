#!/bin/bash

set -e

export WANDB_PROJECT="Adversarial-RL"

LR=1e-5
RUN_NAME=adv_rl_llama_3_1_8b_instruct_lora

ATTACKER_MODEL_NAME_OR_PATH=meta-llama/Llama-3.1-8B-Instruct
DEFENDER_MODEL_NAME_OR_PATH=meta-llama/Llama-3.1-8B-Instruct

echo "starting training"
export CUDA_VISIBLE_DEVICES=0,1,2,3
# no more accelerate :(
python train.py \
    --attacker_model_name_or_path ${ATTACKER_MODEL_NAME_OR_PATH} \
    --defender_model_name_or_path ${DEFENDER_MODEL_NAME_OR_PATH} \
    --reward_functions InjecAgentToolCallingReward \
    --dataset data/InjecAgent/dataset/train.json \
    --attn_implementation flash_attention_2 \
    --num_generations 4 \
    --num_iterations 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_steps 1 \
    --num_train_epochs 1 \
    --bf16 True \
    --beta 0.0 \
    --warmup_ratio 0.03 \
    --gradient_checkpointing True \
    --learning_rate ${LR} \
    --lr_scheduler_type constant_with_warmup \
    --use_peft True \
    --lora_r 128 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --logging_steps 1 \
    --save_strategy epoch \
    --save_only_model True \
    --output_dir checkpoints/${RUN_NAME} \
    --report_to wandb \
    --run_name ${RUN_NAME}
