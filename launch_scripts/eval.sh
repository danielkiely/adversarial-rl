set -e
export WANDB_PROJECT="Adversarial-RL"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

LR=1e-5
RUN_NAME=EVAL_rl_hammer_target_llama_3_1_8b_instruct_lora

ATTACKER_MODEL_NAME_OR_PATH=meta-llama/Llama-3.1-8B-Instruct
TARGET_MODEL_NAME_OR_PATH=meta-llama/Llama-3.1-8B-Instruct

# Eval all checkpoints
export CUDA_VISIBLE_DEVICES=0
for dir in adv_rl_checkpoints/attackers/*; do
    if [ -d "$dir" ]; then
        python injecagent_eval.py \
            --attacker_model_name_or_path ${dir} \
            --attacker_base_model_name_or_path ${ATTACKER_MODEL_NAME_OR_PATH} \
            --target_base_model_name_or_path ${TARGET_MODEL_NAME_OR_PATH} \
            --target_model_name_or_path adv_rl_checkpoints/defender_round_0 \
            --validation_data_path data/InjecAgent/dataset/eval.json \
            --enable_wandb True \
            --run_name eval_${RUN_NAME}_attack_llama_3_2_8b_instruct
    fi
done
