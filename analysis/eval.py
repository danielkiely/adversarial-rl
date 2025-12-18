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
from injecagent_output_parsing import evaluate_output_prompted


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


def eval_model(attacker_checkpoint: str, defender_checkpoint: str) -> Dataset:
    # load eval data as raw json, Dataset, and DataLoader
    eval_raw = json.load(open("data/InjecAgent/dataset/eval.json", "r"))
    eval_dataset = Dataset.from_list(eval_raw)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=16, # hyperparameter
        shuffle=True,
    )
    # use vLLM to load attacker model from checkpoint
    attacker_model = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",    
        dtype="bfloat16",
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=128,
        max_model_len=8192,
        gpu_memory_utilization=0.85
    )
    lora_request = LoRARequest("attack_lora", 1, lora_path=attacker_checkpoint)
    attacker_tokenizer = AutoTokenizer.from_pretrained(
        attacker_checkpoint, trust_remote_code=True
    )
    # generate adversarial prompts
    adv_prompt_results = []
    for k, eval_batch in tqdm(
        enumerate(eval_loader),
        total=len(eval_loader),
        desc="Generating dataset of adversarial prompts",
    ):
        attacker_goals = eval_batch["Attacker Instruction"]
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
        for j in range(len(eval_batch["Attacker Instruction"])):
            attacker_output_text = attacker_output_texts[j]
            attacker_goal = eval_batch["Attacker Instruction"][j]
            attacker_adv_prompt = extract_attack_prompt(attacker_output_text)

            adv_prompt_results.append(
                {
                    "adv_goal": attacker_goal,
                    "attacker_output": attacker_output_text,
                    "attacker_adv_prompt": attacker_adv_prompt,
                    "attacker_tools": eval_batch['Attacker Tools'][0][j],
                    "user_tool": eval_batch['User Tool'][j],
                }
            )

    # delete the attacker model to free up memory
    delete_vllm_model(attacker_model)
    del attacker_tokenizer
    
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
    
    # load target model
    target_model = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",    
        dtype="bfloat16",
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=128,
        max_model_len=8192,
        gpu_memory_utilization=0.85
    )
    lora_request = LoRARequest("defend_lora", 1, lora_path=defender_checkpoint)
    target_tokenizer = AutoTokenizer.from_pretrained(
        defender_checkpoint, trust_remote_code=True
    )
    
    tool_dict = injecagent_get_tool_dict()
    target_model_results = []
    
    # create dataset of user prompts to the defender model
    user_inputs = []
    for adv_prompt_batch in tqdm(adv_prompt_results_dataloader, desc="Attacking target model"):
        for j in range(len(adv_prompt_batch["adv_goal"])):
            curr_data_row = eval_raw[adv_prompt_batch["sample_id"][j]]
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
            
            target_model_input_texts = target_tokenizer.apply_chat_template(
            target_model_messages, add_generation_prompt=True, tokenize=False
            )
            sampling_params = target_model.get_default_sampling_params()
            sampling_params.max_tokens = args.val_max_new_tokens
            target_model_outputs = target_model.generate(
                target_model_input_texts, sampling_params
            )
            target_model_output_texts = [
                output.outputs[0].text for output in target_model_outputs
            ]

            for i in range(len(adv_prompt_batch["adv_goal"])):
                target_model_results.append(
                    {
                        "adv_goal": adv_prompt_batch["adv_goal"][i],
                        "attacker_output": adv_prompt_batch["attacker_output"][i],
                        "attacker_adv_prompt": adv_prompt_batch["attacker_adv_prompt"][
                            i
                        ],
                        "target_model_output": target_model_output_texts[i],
                    }
                )
    # Judge
    attack_success_count = 0
    final_results = []
    for i in range(len(target_model_results)):
        curr_row = target_model_results[i]
        curr_data_row = eval_raw[i]

        eval_result = evaluate_output_prompted(
            curr_row["target_model_output"],
            curr_data_row["Attacker Tools"][0],
            curr_data_row["Attacker Tools"],
            curr_data_row["User Tool"],
            curr_row["attacker_adv_prompt"],
        )
        judge_output = eval_result["eval"]
        if_attack_success = judge_output == "succ"

        final_results.append(
            {
                "adv_goal": curr_row["adv_goal"],
                "attacker_output": curr_row["attacker_output"],
                "attacker_adv_prompt": curr_row["attacker_adv_prompt"],
                "target_model_output": curr_row["target_model_output"],
                "judge_output": judge_output,
                "if_attack_success": if_attack_success,
            }
        )

        if if_attack_success is True:
            attack_success_count += 1

        # Log the results
        if args.enable_wandb is True:
            wandb_table.add_data(
                curr_row["adv_goal"],
                curr_row["attacker_output"],
                curr_row["attacker_adv_prompt"],
                curr_row["target_model_output"],
                judge_output,
                if_attack_success,
            )

    attack_success_rate = attack_success_count / len(final_results)
    print(f"Validation completed. Attack success rate: {attack_success_rate:.2%}")
