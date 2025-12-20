# pip install sentence-transformers umap-learn scikit-learn matplotlib

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List
import json


def save_prompts(prompts, path):
    with open(path, "w") as f:
        json.dump(prompts, f)

def load_prompts(path):
    with open(path, "r") as f:
        return json.load(f)



# UMAP embeddings
def visualize_text_embeddings(
    texts_list,
    group_labels=None,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    n_neighbors=30,
    min_dist=0.1,
    random_state=0,
):
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA
    import umap
    import matplotlib.pyplot as plt
    import numpy as np
    
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

    # 3. plot - split view (each group highlighted separately)
    unique_labels = list(dict.fromkeys(labels))
    fig, axes = plt.subplots(2, 2, figsize=(6, 5))
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, (ax, label) in enumerate(zip(axes, unique_labels)):
        # Gray background
        ax.scatter(Z[:, 0], Z[:, 1], c='lightgray', s=15, alpha=0.3)
        
        # Highlight current group
        idx = [j for j, l in enumerate(labels) if l == label]
        ax.scatter(Z[idx, 0], Z[idx, 1], c=[colors[i]], s=50, alpha=0.7, 
                  edgecolors='white', linewidth=0.5)
        
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_title(f"{label}", fontweight='bold')

    print("total texts:", len(texts), "unique labels:", len(set(labels)))
    print("first 3 labels:", labels[:3])

    fig.tight_layout()
    fig.savefig("analysis/plots/attack_prompt_embeddings.png", dpi=300)
    plt.show()
    



def main():
    labels = []
    text_groups = []

    for i in range(4):
        path = f"analysis/attack_outputs/adv_rl_checkpoints_attackers_attacker_round_{i}/data_InjecAgent_dataset_eval/default.json"
        with open(path, "r") as f:
            data = json.load(f)  # list[dict]

        round_prompts = []
        for ex in data:
            adv_goal = ex.get("adv_goal", "") or ""
            adv_prompt = ex.get("attacker_adv_prompt", "") or ""

            if adv_goal and adv_goal in adv_prompt:
                adv_prompt = adv_prompt.replace(adv_goal, "")

            print(adv_prompt)
            ex["attacker_adv_prompt"] = adv_prompt
            round_prompts.append(adv_prompt)
        
        labels.append(f"Round {i+1}")
        text_groups.append(round_prompts)

    print(len(text_groups))
    print(len(text_groups[0]))
    visualize_text_embeddings(text_groups, labels)
    

if __name__=="__main__":
    main()
