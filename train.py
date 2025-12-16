from trl import TrlParser, ModelConfig, GRPOTrainer
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader
from vllm import LLM
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)
from vllm.lora.request import LoRARequest
import torch
import json
from tqdm import tqdm
import gc
import time
import ray
import contextlib
import os
import wandb

# Custom imports
from config import LocalGRPOConfig
from reward_func import ALL_REWARD_FUNCS
from utils import (
    set_random_seed,
    InjecAgentDataset,
    ATTACKER_SYS_PROMPT,
    INJECAGENT_SYS_PROMPT,
    INJECAGENT_USER_PROMPT,
    injecagent_get_tool_dict,
    log_gpu_usage,
)
from reward_func import extract_attack_prompt


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
        grpo_config.model_init_kwargs["attn_implementation"] = (model_config.attn_implementation)
    else:
        grpo_config.model_init_kwargs["attn_implementation"] = "flash_attention_2"
        

    def cleanup_model(*models):
        """
        free GPU memory by deleting models and clearing cache
        """
        for m in models:
            if m is not None:
                try:
                    # If it's a trainer, delete each piece
                    if hasattr(m, 'model'):
                        m.model.to('cpu')
                        del m.model
                    if hasattr(m, "accelerator"):
                        m.accelerator.free_memory()
                    if hasattr(m, 'optimizer'):
                        m.optimizer.to('cpu')
                        del m.optimizer
                    if hasattr(m, 'lr_scheduler'):
                        m.lr_scheduler.to('cpu')
                        del m.lr_scheduler
                    if hasattr(m, 'trainer_state'):
                        m.trainer_state.to('cpu')
                        del m.trainer_state
                    m.to('cpu')
                    del m
                except:
                    pass
        gc.collect()
        if torch.cuda.is_available():
            with torch.no_grad():
                for j in range(torch.cuda.device_count()):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(j)
        print("GPU memory cleared!")
    
    def delete_vllm_model(model):
        destroy_model_parallel()
        destroy_distributed_environment()
        model.llm_engine.engine_core.shutdown()
        del model
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()
        ray.shutdown()
    
    def set_device_map_grpo(model_name_or_path):
        """Puts GRPO trainer on GPUs 0, 1

        Args:
            model_name_or_path (str): name or path of model to be trained via GRPO
        """
        grpo_config.model_name_or_path = model_name_or_path
        grpo_config.model_init_kwargs["device_map"] = "auto"
        grpo_config.model_init_kwargs["max_memory"] = {0: "23.5GB", 1: "23.5GB"}
    
    
    def create_defender_dataset(i: int) -> Dataset:
        """takes the current attack model and creates a dataset of adversarial prompts. transforms
            this dataset into a format that can be used to train the target model.

        Args:
            i (int): round of adversarial training. used for checkpoint loading/saving
        """
        # load training data as raw json, Dataset, and DataLoader
        train_raw = json.load(open(grpo_config.dataset, "r"))
        train_dataset = Dataset.from_list(train_raw)
        train_loader = DataLoader(
            train_dataset,
            batch_size=16, # hyperparameter
            shuffle=True,
        )
        # use vLLM to load attacker model from most recent checkpoint
        attacker_model = LLM(
            model="meta-llama/Llama-3.1-8B-Instruct",    
            dtype="bfloat16",
            trust_remote_code=True,
            enable_lora=True,
            max_lora_rank=128,
            max_model_len=8192,
            gpu_memory_utilization=0.85
        )
        lora_request = LoRARequest("attack_lora", 1, lora_path=f"adv_rl_checkpoints/attacker_round_{i}")
        attacker_tokenizer = AutoTokenizer.from_pretrained(
            f"adv_rl_checkpoints/attacker_round_{i}", trust_remote_code=True
        )
        # generate adversarial prompts
        adv_prompt_results = []
        for k, train_batch in tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc="Generating dataset of adversarial prompts",
        ):
            attacker_goals = train_batch["Attacker Instruction"]
            attacker_prompts = [
                ATTACKER_SYS_PROMPT.format(goal=attacker_goal)
                for attacker_goal in attacker_goals
            ]
            attacker_messages = [
                [{"role": "user", "content": attacker_prompt}]
                for attacker_prompt in attacker_prompts
            ]
            attacker_input_texts = attacker_tokenizer.apply_chat_template(
                attacker_messages, add_generation_prompt=True, tokenize=False
            )
            sampling_params = attacker_model.get_default_sampling_params()
            sampling_params.max_tokens = 1024
            attacker_outputs = attacker_model.generate(
                attacker_input_texts, sampling_params, lora_request=lora_request
            )
            attacker_output_texts = [
                output.outputs[0].text for output in attacker_outputs
            ]

            # extract the attack prompt from the output
            for j in range(len(train_batch["Attacker Instruction"])):
                attacker_output_text = attacker_output_texts[j]
                attacker_goal = train_batch["Attacker Instruction"][j]
                attacker_adv_prompt = extract_attack_prompt(attacker_output_text)

                adv_prompt_results.append(
                    {
                        "adv_goal": attacker_goal,
                        "attacker_output": attacker_output_text,
                        "attacker_adv_prompt": attacker_adv_prompt,
                        "attacker_tools": train_batch['Attacker Tools'][0][j],
                        "user_tool": train_batch['User Tool'][j],
                    }
                )

        # delete the attacker model to free up memory
        delete_vllm_model(attacker_model)
        del attacker_tokenizer

        # save the adversarial prompts to json file
        os.makedirs(f"adv_rl_saved_attacks/{i}", exist_ok=True)
        with open(f"adv_rl_saved_attacks/{i}/adv_prompt_results.json", "w") as f:
            json.dump(adv_prompt_results, f, indent=4)
        
        # add sample_id to each prompt
        for j in range(len(adv_prompt_results)):
            adv_prompt_results[j]["sample_id"] = j
        
        # make adv_prompt_results a dataset and dataloader
        adv_prompt_results_dataset = Dataset.from_list(adv_prompt_results)
        adv_prompt_results_dataloader = DataLoader(
                adv_prompt_results_dataset,
                batch_size=16,
                shuffle=False,
        )
        
        tool_dict = injecagent_get_tool_dict()
        
        # create dataset of user prompts to the defender model
        user_inputs = []
        for adv_prompt_batch in tqdm(adv_prompt_results_dataloader, desc="Attacking target model"):
            for j in range(len(adv_prompt_batch["adv_goal"])):
                curr_data_row = train_raw[adv_prompt_batch["sample_id"][j]]
                available_tool_names = [curr_data_row["User Tool"]] + curr_data_row[
                    "Attacker Tools"
                ]
                available_tool_descriptions = json.dumps(
                    [tool_dict[tool] for tool in available_tool_names], indent=True
                )
                obvervation = curr_data_row["Tool Response Template"].replace(
                    "<Attacker Instruction>", adv_prompt_batch["attacker_adv_prompt"][j]
                )
                agent_scratchpad = f"\n{curr_data_row['Thought']}\nAction: {curr_data_row['User Tool']}\nAction Input: {curr_data_row['Tool Parameters']}\nObservation: {obvervation}\n"

                user_prompt_filled = INJECAGENT_USER_PROMPT.format(
                    available_tool_descriptions=available_tool_descriptions,
                    input=curr_data_row["User Instruction"],
                    available_tool_names=available_tool_names,
                    agent_scratchpad=agent_scratchpad,
                )
                user_inputs.append(
                    {
                        "prompt": [
                            {"role": "system", "content": INJECAGENT_SYS_PROMPT},
                            {"role": "user", "content": user_prompt_filled},
                        ],
                        "sample_id": adv_prompt_batch["sample_id"][j],
                        "adv_goal": adv_prompt_batch["adv_goal"][j],
                        "attacker_adv_prompt": adv_prompt_batch["attacker_adv_prompt"][j],
                        "attacker_tools": adv_prompt_batch["attacker_tools"][j],
                        "user_tool": adv_prompt_batch["user_tool"][j],
                    }
                )
            
        return Dataset.from_list(user_inputs)

    def set_wandb_run(role, round_idx):
        grpo_config.run_name = f"{role}-round{round_idx}"
        
    
    rounds = 4
    
    attacker_base_string = grpo_config.attacker_model_name_or_path
    defender_base_string = grpo_config.defender_model_name_or_path
    
    attacker_checkpoint = "adv_rl_checkpoints/attacker_round_0"
    defender_checkpoint = "adv_rl_checkpoints/defender_round_0"
    
    gpu_log_file = "gpu.log"
    
    for i in range(1, rounds):
        # load frozen opponent and put on gpus 2, 3
        # if first round, load base. otherwise, load LoRA weights from checkpoint
        set_wandb_run("attacker", i)
        
        if i == 0:
            defender_base_model = None
            defender_frozen = AutoModelForCausalLM.from_pretrained(
                defender_base_string,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory={2: "23.5GB", 3: "23.5GB"},
                attn_implementation="flash_attention_2",
            )
        else:
            defender_base_model = AutoModelForCausalLM.from_pretrained(
                defender_base_string,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory={2: "23.5GB", 3: "23.5GB"},
                attn_implementation="flash_attention_2",
            )
            defender_frozen = PeftModel.from_pretrained(
                defender_base_model,
                defender_checkpoint,
            )
            
        # frozen model only used for inference
        defender_frozen.eval()
        defender_frozen.requires_grad_(False)
        
        # put attacker model on gpus 0, 1
        if i == 0:
            attacker_base_model = None
            attacker_train_model = AutoModelForCausalLM.from_pretrained(
                attacker_base_string,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory={0: "23.5GB", 1: "23.5GB"},
                attn_implementation="flash_attention_2",
            )
        else:
            attacker_base_model = AutoModelForCausalLM.from_pretrained(
                attacker_base_string,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory={0: "23.5GB", 1: "23.5GB"},
                attn_implementation="flash_attention_2",
            )
            attacker_train_model = PeftModel.from_pretrained(
                attacker_base_model,
                attacker_checkpoint,
                is_trainable=True,
            )
        
        attacker_train_model.train()
        attacker_train_model.config.use_cache = False
        attacker_train_model.gradient_checkpointing_enable()
        attacker_train_model.enable_input_require_grads()
        set_device_map_grpo(attacker_base_string)

        # train attacker
        if i == 0:
            attack_trainer = GRPOTrainer(
                args=grpo_config,
                model=attacker_train_model,
                peft_config=peft_config,
                reward_funcs=[ALL_REWARD_FUNCS["InjecAgentToolCallingReward"](grpo_config, defender_frozen, "attacker")],
                train_dataset=train_set,
            )
        else:
            attack_trainer = GRPOTrainer(
                args=grpo_config,
                model=attacker_train_model,
                reward_funcs=[ALL_REWARD_FUNCS["InjecAgentToolCallingReward"](grpo_config, defender_frozen, "attacker")],
                train_dataset=train_set,
            )
            
        attack_trainer.train()
        
        # save attacker state
        attacker_checkpoint = f"adv_rl_checkpoints/attacker_round_{i}"
        attack_trainer.save_model(attacker_checkpoint)
        
        log_gpu_usage(f"Attacker training round {i} completed.", gpu_log_file)
        
        # cleanup models
        cleanup_model(attack_trainer, defender_frozen, defender_base_model, attacker_train_model, attacker_base_model)
        attack_trainer = None
        defender_frozen = None
        defender_base_model = None
        attacker_train_model = None
        attacker_base_model = None
        del attack_trainer
        del defender_frozen
        del defender_base_model
        del attacker_train_model
        del attacker_base_model
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5)
        
        log_gpu_usage(f"Cleanup after attacker round {i} completed.", gpu_log_file)
        
        defender_dataset = create_defender_dataset(i)
        defender_dataset.to_json('defender_dataset.json')
        log_gpu_usage(f"Defender dataset created for round {i}.", gpu_log_file)

        set_wandb_run("defender", i)
        # load frozen attacker on gpus 2, 3
        if i == 0:
            attacker_base_model = None
            attacker_frozen = AutoModelForCausalLM.from_pretrained(
                attacker_base_string,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory={2: "23.5GB", 3: "23.5GB"},
                attn_implementation="flash_attention_2",
            )
        else:
            attacker_base_model = AutoModelForCausalLM.from_pretrained(
                attacker_base_string,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory={2: "23.5GB", 3: "23.5GB"},
                attn_implementation="flash_attention_2",
            )
            attacker_frozen = PeftModel.from_pretrained(
                attacker_base_model,
                attacker_checkpoint,
            )
            
        # frozen model only used for inference
        attacker_frozen.eval()
        attacker_frozen.requires_grad_(False)
        
        if i == 0:
            defender_base_model = None
            defender_train_model = AutoModelForCausalLM.from_pretrained(
                defender_base_string,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory={0: "23.5GB", 1: "23.5GB"},
                attn_implementation="flash_attention_2",
            )
        else:
            defender_base_model = AutoModelForCausalLM.from_pretrained(
                defender_base_string,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory={0: "23.5GB", 1: "23.5GB"},
                attn_implementation="flash_attention_2",
            )
            defender_train_model = PeftModel.from_pretrained(
                defender_base_model,
                defender_checkpoint,
                is_trainable=True,
            )
        
        defender_train_model.train()
        defender_train_model.config.use_cache = False
        defender_train_model.gradient_checkpointing_enable()
        defender_train_model.enable_input_require_grads()
        # put defender on gpus 0, 1
        set_device_map_grpo(defender_base_string)

        # train defender
        if i == 0:
            defend_trainer = GRPOTrainer(
                args=grpo_config,
                model=defender_train_model,
                peft_config=peft_config,
                reward_funcs=[ALL_REWARD_FUNCS["DefenderReward"](grpo_config)],
                train_dataset=defender_dataset,
            )
        else:
            defend_trainer = GRPOTrainer(
                args=grpo_config,
                model=defender_train_model,
                reward_funcs=[ALL_REWARD_FUNCS["DefenderReward"](grpo_config)],
                train_dataset=defender_dataset,
            )
        defend_trainer.train()
        
        log_gpu_usage(f"Defender training round {i} completed.", gpu_log_file)

        # save defender state
        defender_checkpoint = f"adv_rl_checkpoints/defender_round_{i}"
        defend_trainer.save_model(defender_checkpoint)
        
        # cleanup models
        cleanup_model(defend_trainer, attacker_frozen, attacker_base_model, defender_train_model, defender_base_model)
        
        defend_trainer = None
        attacker_frozen = None
        attacker_base_model = None
        defender_train_model = None
        defender_base_model = None
        del defend_trainer
        del attacker_frozen
        del attacker_base_model
        del defender_train_model
        del defender_base_model
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5)
        log_gpu_usage(f"Cleanup after defender round {i} completed.", gpu_log_file)
    
    print("made it!!")

if __name__ == "__main__":
    parser = TrlParser((LocalGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()
    main(grpo_config=grpo_config, model_config=model_config)
