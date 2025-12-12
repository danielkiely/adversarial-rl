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

    # set gpus for each device
    attack_device = "cuda:0"
    target_device = "cuda:1"
    
    # load target model
    target_model = AutoModelForCausalLM.from_pretrained(
        grpo_config.target_model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    target_model.to(target_device)
    
    # load attacker model
    attacker_model = AutoModelForCausalLM.from_pretrained(
        grpo_config.attacker_model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    attacker_model.to(attack_device)
    
    # Add reward functions - right now this is only the InjecAgentToolCallingReward
    reward_functions = [
        ALL_REWARD_FUNCS[curr_func](grpo_config, target_model)
        for curr_func in grpo_config.reward_functions
    ]
    
    # Initialize and run trainer
    trainer = GRPOTrainer(
        args=grpo_config,
        model=attacker_model,
        peft_config=peft_config,
        reward_funcs=reward_functions,
        train_dataset=train_set,
    )
    trainer.target_model = target_model

    trainer.train(resume_from_checkpoint=grpo_config.resume_from_checkpoint)
    
    # TODO: write own training loop so that we can call trainer.step for each model after computing the rewards together
    


if __name__ == "__main__":
    parser = TrlParser((LocalGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()
    main(grpo_config=grpo_config, model_config=model_config)
