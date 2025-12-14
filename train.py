from trl import TrlParser, ModelConfig, GRPOTrainer
from peft import LoraConfig
from transformers import AutoModelForCausalLM
from datasets import Dataset
from torch.utils.data import DataLoader
from vllm import LLM
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)
import torch
import json
from tqdm import tqdm
import gc

# Custom imports
from config import LocalGRPOConfig
from reward_func import ALL_REWARD_FUNCS
from utils import (
    set_random_seed,
    InjecAgentDataset,
    ATTACKER_SYS_PROMPT
)
from reward_func import evaluate_output_prompted, extract_attack_prompt

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
        grpo_config.model_name_or_path = model_name_or_path
        grpo_config.model_init_kwargs["device_map"] = "auto"
        grpo_config.model_init_kwargs["max_memory"] = {0: "24.0GB", 1: "24.0GB"}
    
    
    def create_attack_dataset(i):
        train_raw = json.load(open(grpo_config.dataset, "r"))
        train_dataset = Dataset.from_list(train_raw)
        train_loader = DataLoader(
            train_dataset,
            batch_size=16, # hyperparameter
            shuffle=True,
        )
        attacker_model = LLM(
            model=f"adv_rl_checkpoints/attacker_round_{i}",
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=8192,
        )
        lora_request = None
        attacker_tokenizer = AutoTokenizer.from_pretrained(
            f"adv_rl_checkpoints/attacker_round_{i}", trust_remote_code=True
        )
        adv_prompt_results = []
        for _, train_batch in tqdm(
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
            if args.temperature is not None:
                sampling_params.temperature = args.temperature
            sampling_params.max_tokens = args.max_new_tokens
            attacker_outputs = attacker_model.generate(
                attacker_input_texts, sampling_params, lora_request=lora_request
            )
            attacker_output_texts = [
                output.outputs[0].text for output in attacker_outputs
            ]

            # Extract the attack prompt from the output
            for i in range(len(train_batch["Attacker Instruction"])):
                attacker_output_text = attacker_output_texts[i]
                attacker_goal = train_batch["Attacker Instruction"][i]

                # Extract the attack prompt from the output
                attacker_adv_prompt = extract_attack_prompt(attacker_output_text) # TODO import this

                adv_prompt_results.append(
                    {
                        "adv_goal": attacker_goal,
                        "attacker_output": attacker_output_text,
                        "attacker_adv_prompt": attacker_adv_prompt,
                    }
                )

        # Delete the attacker model to free up memory
        delete_vllm_model(attacker_model)
        del attacker_tokenizer

        # Save the adversarial prompts
        # TODO: make this the right path
        os.makedirs(f"adv_rl_saved_attacks/{i}", exist_ok=True)
        with open(f"adv_rl_saved_attacks/{i}/adv_prompt_results.json", "w") as f:
            json.dump(adv_prompt_results, f, indent=4)
    

    rounds = 2
    attacker_checkpoint = grpo_config.attacker_model_name_or_path
    defender_checkpoint = grpo_config.defender_model_name_or_path
    
    for i in range(rounds):
        # TODO: make sure wandb works for both models
        
        # shuffle the dataset at start of each round
        # train_set.shuffle()
        
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
        attacker_checkpoint = f"adv_rl_checkpoints/attacker_round_{i}"
        attack_trainer.save_model(attacker_checkpoint)
        
        # cleanup models
        cleanup_model(attack_trainer, defender_frozen)
        
        # TODO: generate a datset of attacks to be used in defender training
        create_attack_dataset(i)
        print("we made it!!!!")
        break

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

        # TODO: switch this to be the attacks dataset
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
