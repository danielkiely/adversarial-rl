# pip install sentence-transformers umap-learn scikit-learn matplotlib

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt

from transformers import AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader
from vllm import LLM
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)
from vllm.lora.request import LoRARequest
import torch
from tqdm import tqdm
import gc
import ray
import contextlib
from typing import List
import json

from utils import ATTACKER_SYS_PROMPT
from reward_func import extract_attack_prompt

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

def visualize_text_embeddings(
    texts_list,
    group_labels=None,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    n_neighbors=30,
    min_dist=0.1,
    random_state=0,
):
    """
    texts_list: List[List[str]] - List of text lists, where each inner list shares the same label
    group_labels: Optional[List[str]] - Custom labels for each group. If None, will use 'Group 1', 'Group 2', etc.
    """
    # Flatten the list of lists and create corresponding labels
    texts = []
    labels = []
    for i, text_group in enumerate(texts_list):
        texts.extend(text_group)
        label = group_labels[i] if group_labels and i < len(group_labels) else f"Group {i+1}"
        labels.extend([label] * len(text_group))

    # 1. embed
    model = SentenceTransformer(model_name)
    X = model.encode(texts, normalize_embeddings=True)

    # 2. PCA -> UMAP
    X50 = PCA(n_components=min(50, X.shape[0] - 1), random_state=random_state).fit_transform(X)
    Z = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    ).fit_transform(X50)

    # 3. plot
    fig, ax = plt.subplots(figsize=(4.6, 3.2))


    for label in dict.fromkeys(labels):  # preserves order
        idx = [j for j, l in enumerate(labels) if l == label]
        ax.scatter(
            Z[idx, 0], Z[idx, 1],
            s=28,
            facecolors='none',
            linewidths=1.0,
            alpha=1.0,
            label=label,
            rasterized=True,
        )


    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    ax.legend(loc="best", frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig("analysis/plots/attack_prompt_embeddings.png", dpi=300)


def create_attack_list(attacker_checkpoint: str) -> List[str]:
        """takes an attack model and generates a list of adversarial prompts

        Args:
            attacker_checkpoint (str): path to attacker checkpoint
        """
        # load eval data as raw json, Dataset, and DataLoader
        eval_raw = json.load(open('data/InjecAgent/dataset/eval.json', "r"))
        eval_dataset = Dataset.from_list(eval_raw)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=16, # hyperparameter
            shuffle=True,
        )
        # use vLLM to load attacker model from the checkpoint
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
        res = []
        for _, eval_batch in tqdm(
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
                res.append(attacker_adv_prompt)

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
        
        # TODO: save them in a file
        return res

def main():
    checkpoints = []
    labels = []
    text_groups = []
    for i in range(4):
        checkpoints.append(f"adv_rl_checkpoints/attackers/attacker_round_{i}")
        labels.append(f"Round {i}")
        text_groups.append(create_attack_list(f"adv_rl_checkpoints/attackers/attacker_round_{i}"))

    visualize_text_embeddings(text_groups, labels)

if __name__=="__main__":
    main()
