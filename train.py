# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from trl import TrlParser, ModelConfig, GRPOTrainer
from peft import LoraConfig
from transformers import AutoModelForCausalLM
import torch

# Custom imports
from config import LocalGRPOConfig
from reward_func import ALL_REWARD_FUNCS
from utils import (
    set_random_seed,
    InjecAgentDataset,
)


def main(grpo_config, model_config):
    # Set seed for reproducibility
    set_random_seed(grpo_config.seed)
    
    # Load dataset
    train_set = InjecAgentDataset(grpo_config.dataset)

    # Lora configuration
    if model_config.use_peft is True:
        peft_config = LoraConfig(
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "down_proj",
                "gate_proj",
            ],
            task_type="CAUSAL_LM",
            lora_dropout=model_config.lora_dropout,
        )
    else:
        peft_config = None

    grpo_config.gradient_checkpointing_kwargs = {"use_reentrant": False}
    grpo_config.ddp_find_unused_parameters = False
    grpo_config.model_init_kwargs = {}
    if model_config.torch_dtype is not None:
        grpo_config.model_init_kwargs["torch_dtype"] = model_config.torch_dtype
    else:
        grpo_config.model_init_kwargs["torch_dtype"] = "bfloat16"
    if model_config.attn_implementation is not None:
        grpo_config.model_init_kwargs["attn_implementation"] = (
            model_config.attn_implementation
        )
    else:
        grpo_config.model_init_kwargs["attn_implementation"] = "flash_attention_2"

    # load target model
    target_model = AutoModelForCausalLM.from_pretrained(
        grpo_config.target_model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    
    grpo_config.model_name_or_path = grpo_config.attacker_model_name_or_path
    grpo_config.model_init_kwargs["device_map"] = {"": 0}  # put attacker model on cuda:0

    attack_trainer = GRPOTrainer(
        args=grpo_config,
        model=grpo_config.attacker_model_name_or_path,
        peft_config=peft_config,
        reward_funcs=[ALL_REWARD_FUNCS["InjecAgentToolCallingReward"](grpo_config, target_model, "attacker")],
        train_dataset=train_set,
        use_vllm=True,
        vllm_mode="colocate",
    )
    
    defend_trainer = GRPOTrainer(
        args=grpo_config,
        model=grpo_config.defender_model_name_or_path,
        peft_config=peft_config,
        reward_funcs=[ALL_REWARD_FUNCS["InjecAgentToolCallingReward"](grpo_config, target_model, "defender")],
        train_dataset=train_set,
        use_vllm=True,
        vllm_mode="colocate"
    )
    
    rounds = 2
    attacker_checkpoint = grpo_config.attacker_model_name_or_path
    defender_checkpoint = grpo_config.defender_model_name_or_path
    
    for i in range(rounds):
        # TODO: tear down old models using nlp code
        # TODO: make sure wandb works for both models
        # load frozen opponent
        defender_frozen = AutoModelForCausalLM.from_pretrained(defender_checkpoint)
        defender_frozen.eval()
        defender_frozen.requires_grad_(False)

        # train attacker
        attack_trainer = GRPOTrainer(
            args=grpo_config,
            model=attacker_checkpoint,
            peft_config=peft_config,
            reward_funcs=[AttackerReward(defender_frozen)],
            train_dataset=train_set,
        )
        attack_trainer.train()

        # save attacker state
        attacker_checkpoint = f".../attacker_round_{i}"
        attack_trainer.save_model(attacker_checkpoint)

        # load frozen attacker
        attacker_frozen = AutoModelForCausalLM.from_pretrained(attacker_checkpoint)
        attacker_frozen.eval()
        attacker_frozen.requires_grad_(False)

        # train defender
        defend_trainer = GRPOTrainer(
            args=grpo_config,
            model=defender_checkpoint,
            peft_config=peft_config,
            reward_funcs=[DefenderReward(attacker_frozen)],
            train_dataset=train_set,
        )
        defend_trainer.train()

        defender_checkpoint = f".../defender_round_{i}"
        defend_trainer.save_model(defender_checkpoint)
    

if __name__ == "__main__":
    parser = TrlParser((LocalGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()
    main(grpo_config=grpo_config, model_config=model_config)
