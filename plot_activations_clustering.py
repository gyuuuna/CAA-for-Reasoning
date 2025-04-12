import json
import torch as t
import os
from matplotlib import pyplot as plt
import argparse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from behaviors import HUMAN_NAMES, get_activations_path, get_ab_data_path, get_analysis_dir, ALL_BEHAVIORS
from utils.helpers import get_model_path, set_plotting_settings
from tqdm import tqdm
import numpy as np
from collections import Counter

set_plotting_settings()

def cluster_negatives_and_local_pca(
    behavior, activations_pos, activations_neg, title, layer, activation_type, fname_prefix="pca_cluster", n_clusters=4
):
    import os
    import json
    import numpy as np
    import torch as t
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from behaviors import get_ab_data_path, get_analysis_dir

    act_type_str = activation_type if activation_type else "residual"
    save_dir = os.path.join(get_analysis_dir(behavior), f"{fname_prefix}-{act_type_str}")
    os.makedirs(save_dir, exist_ok=True)

    with open(get_ab_data_path(behavior), "r") as f:
        ab_data = json.load(f)

    assert len(ab_data) == len(activations_pos), (
        f"Mismatch: {len(ab_data)} items vs {len(activations_pos)} pos activations"
    )

    all_activations = t.cat([activations_pos, activations_neg], dim=0).cpu().numpy()
    pca = PCA(n_components=2)
    all_projected = pca.fit_transform(all_activations)

    n_pos = activations_pos.shape[0]
    projected_pos = all_projected[:n_pos]
    projected_neg = all_projected[n_pos:]

    cmap = plt.get_cmap("tab10")

    # Clustering only negative samples
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_ids_neg = kmeans.fit_predict(projected_neg)

    # overall vec
    mean_pos_proj = projected_pos.mean(axis=0)
    mean_neg_proj = projected_neg.mean(axis=0)
    overall_vec = mean_pos_proj - mean_neg_proj

    # ----------- Overall PCA Plot ------------
    plt.figure(figsize=(7, 7))
    for i, ((x_pos, y_pos), (x_neg, y_neg)) in enumerate(zip(projected_pos, projected_neg)):
        cid = cluster_ids_neg[i]
        color = cmap(cid % 10)
        plt.scatter(x_neg, y_neg, color=color, marker='x', alpha=0.5,
                    label=f"neg {cid}" if f"neg {cid}" not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.scatter(x_pos, y_pos, color=color, marker='o', alpha=0.3,
                    label=f"pos {cid}" if f"pos {cid}" not in plt.gca().get_legend_handles_labels()[1] else "")

    # overall vector
    plt.arrow(
        mean_neg_proj[0], mean_neg_proj[1], overall_vec[0], overall_vec[1],
        color="black", width=0.002, head_width=0.05, length_includes_head=True, label="overall shift"
    )

    # cluster vectors
    for cid in range(n_clusters):
        neg_indices = [i for i, c in enumerate(cluster_ids_neg) if c == cid]
        if len(neg_indices) < 2:
            continue
        pos_indices = neg_indices
        local_neg = projected_neg[neg_indices]
        local_pos = projected_pos[pos_indices]

        neg_mean = local_neg.mean(axis=0)
        pos_mean = local_pos.mean(axis=0)
        vec = pos_mean - neg_mean

        color = cmap(cid % 10)
        plt.arrow(
            neg_mean[0], neg_mean[1], vec[0], vec[1],
            color=color, width=0.0015, head_width=0.04, length_includes_head=True
        )

    plt.title(f"{title} - Overall PCA ({activation_type or 'residual'})")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{fname_prefix}_{behavior}_layer_{layer}_overall.png"))
    plt.close()

    # ----------- Cluster Subplots (2x2) ------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for cid in range(n_clusters):
        ax = axes[cid]
        color = cmap(cid % 10)

        # cluster indices
        neg_indices = [i for i, c in enumerate(cluster_ids_neg) if c == cid]
        pos_indices = neg_indices  # 1:1

        if len(neg_indices) < 2:
            print(f"Skipping cluster {cid} due to insufficient data.")
            continue

        local_neg = projected_neg[neg_indices]
        local_pos = projected_pos[pos_indices]

        for x, y in local_neg:
            ax.scatter(x, y, color=color, marker='x', alpha=0.5,
                       label="neg" if "neg" not in ax.get_legend_handles_labels()[1] else "")
        for x, y in local_pos:
            ax.scatter(x, y, color=color, marker='o', alpha=0.3,
                       label="pos" if "pos" not in ax.get_legend_handles_labels()[1] else "")

        # cluster vector
        neg_mean = local_neg.mean(axis=0)
        pos_mean = local_pos.mean(axis=0)
        vec = pos_mean - neg_mean
        ax.arrow(
            neg_mean[0], neg_mean[1], vec[0], vec[1],
            color=color, width=0.0015, head_width=0.04, length_includes_head=True
        )
        # overall vector
        ax.arrow(
            mean_neg_proj[0], mean_neg_proj[1], overall_vec[0], overall_vec[1],
            color="black", width=0.0015, head_width=0.04, length_includes_head=True
        )

        ax.set_title(f"Cluster {cid}")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.legend()

    plt.suptitle(f"{title} - PCA Cluster Views ({activation_type or 'residual'})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, f"{fname_prefix}_{behavior}_layer_{layer}_clusters.png"))
    plt.close()

    print(f"[✓] Saved PCA plots with overall and cluster mean arrows to {save_dir}")


def save_activation_projection_pca(behavior: str, layer: int, model_name_path: str, activation_type:str = None):
    title = f"{HUMAN_NAMES[behavior]}, layer {layer}"

    activations_pos = t.load(
        get_activations_path(behavior, layer, model_name_path, "pos", type=activation_type)
    )
    activations_neg = t.load(
        get_activations_path(behavior, layer, model_name_path, "neg", type=activation_type)
    )

    # if "choice" in behavior or "multiwozdst" in behavior or "paradetox" in behavior:
    cluster_negatives_and_local_pca(
        behavior,
        activations_pos,
        activations_neg,
        title,
        layer,
        activation_type=activation_type,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--behaviors", nargs="+", type=str, default=ALL_BEHAVIORS)
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--activation_type", type=str, choices=["none", "attn", "mlp", "layer"], default="none")
    parser.add_argument("--model_size", type=str, choices=["7b", "13b"], default="7b")
    args = parser.parse_args()

    activation_type = None if args.activation_type == "none" else args.activation_type
    model_name_path = get_model_path(args.model_size, args.use_base_model)

    for behavior in args.behaviors:
        print(f"Plotting {behavior} activations PCA")
        for layer in tqdm(args.layers):
            save_activation_projection_pca(
                behavior,
                layer,
                model_name_path,
                activation_type
            )