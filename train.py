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
        
        
    # clean up old models
    def cleanup_model(*models):
        """
        Free GPU memory by deleting models and clearing cache.

        Usage: cleanup_model(trainer, model) or cleanup_model(shaped_trainer)
        """
        for m in models:
            if m is not None:
                try:
                    # If it's a trainer, get the model
                    if hasattr(m, 'model'):
                        del m.model
                    del m
                except:
                    pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("GPU memory cleared!")
    
    def set_device_map_grpo(model_name_or_path):
        grpo_config.model_name_or_path = model_name_or_path
        grpo_config.model_init_kwargs["device_map"] = "auto"
        grpo_config.model_init_kwargs["max_memory"] = {0: "24.0GB", 1: "24.0GB"}
    

    rounds = 2
    attacker_checkpoint = grpo_config.attacker_model_name_or_path
    defender_checkpoint = grpo_config.defender_model_name_or_path
    
    for i in range(rounds):
        # TODO: make sure wandb works for both models
        
        # shuffle the dataset at start of each round
        train_set.shuffle()
        
        # load frozen opponent and put on gpus 2, 3
        defender_frozen = AutoModelForCausalLM.from_pretrained(
            defender_checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory={2: "24.0GB", 3: "24.0GB"},
        )
        # frozen model only used for inference
        defender_frozen.eval()
        defender_frozen.requires_grad_(False)
        
        # put attacker model on gpus 0, 1
        set_device_map_grpo(attacker_checkpoint)

        # train attacker
        attack_trainer = GRPOTrainer(
            args=grpo_config,
            model=attacker_checkpoint,
            peft_config=peft_config,
            reward_funcs=[ALL_REWARD_FUNCS["InjecAgentToolCallingReward"](grpo_config, defender_frozen, "attacker")],
            train_dataset=train_set,
        )
        attack_trainer.train()
        
        # save attacker state
        # TODO: make the checkpoint directory
        attacker_checkpoint = f"adv_rl_checkpoints/attacker_round_{i}"
        attack_trainer.save_model(attacker_checkpoint)
        
        # cleanup models
        cleanup_model(attack_trainer, defender_frozen)

        # load frozen attacker on gpus 2, 3
        attacker_frozen = AutoModelForCausalLM.from_pretrained(
            attacker_checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory={2: "24.0GB", 3: "24.0GB"},
        )
        # frozen model only used for inference
        attacker_frozen.eval()
        attacker_frozen.requires_grad_(False)
        
        # put defender on gpus 0, 1
        set_device_map_grpo(defender_checkpoint)

        # train defender
        defend_trainer = GRPOTrainer(
            args=grpo_config,
            model=defender_checkpoint,
            peft_config=peft_config,
            reward_funcs=[ALL_REWARD_FUNCS["InjecAgentToolCallingReward"](grpo_config, attacker_frozen, "defender")],
            train_dataset=train_set,
        )
        defend_trainer.train()

        # save defender state
        defender_checkpoint = f"adv_rl_checkpoints/defender_round_{i}"
        defend_trainer.save_model(defender_checkpoint)
        
        # cleanup models
        cleanup_model(defend_trainer, attacker_frozen)
    

if __name__ == "__main__":
    parser = TrlParser((LocalGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()
    main(grpo_config=grpo_config, model_config=model_config)
