"""
Script to plot PCA of constrastive activations

Usage:
python plot_activations.py --behaviors sycophancy --layers 9 10 11 --use_base_model --model_size 7b
"""

import json
import torch as t
import itertools
import os
from matplotlib import pyplot as plt
import argparse
from sklearn.decomposition import PCA
from behaviors import HUMAN_NAMES, get_activations_path, get_ab_data_path, get_analysis_dir, ALL_BEHAVIORS # get_mcp_data_path, MCP_BEHAVIOR_OPTIONS, get_mcp_label_tokens
from behaviors import *
from utils.helpers import get_model_path, set_plotting_settings
from tqdm import tqdm
from sklearn.cluster import KMeans
import pandas as pd


DATASET_FILE = os.path.join("preprocessed_data", "generate_dataset.json")

set_plotting_settings()

def save_activation_projection_pca(behavior: str, layer: int, model_name_path: str, activation_type:str = None):
    title = f"{HUMAN_NAMES[behavior]}, layer {layer}"
    fname = f"pca_{behavior}_layer_{layer}.png"
    save_dir = os.path.join(get_analysis_dir(behavior), f"pca-{activation_type}" if activation_type else "pca")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Loading activations
    activations_pos = t.load(
        get_activations_path(behavior, layer, model_name_path, "pos", type=activation_type)
    )
    activations_neg = t.load(
        get_activations_path(behavior, layer, model_name_path, "neg", type=activation_type)
    )

    # Getting letters
    if "choice" in behavior:
        cluster_and_save_pca_projection(behavior, activations_pos, activations_neg, title, save_dir, fname, activation_type)
    if "multiwozdst" in behavior or "paradetox" in behavior:
        cluster_and_save_pca_projection(behavior, activations_pos, activations_neg, title, save_dir, fname, activation_type)

def cluster_and_save_pca_projection(behavior, activations_pos, activations_neg, title, save_dir, fname, activation_type, n_clusters=4):
    """
    Clusters the PCA projections of activations and saves the cluster-labeled data points
    grouped by cluster ID, including the original item metadata.
    """
    # Load AB data (with full item info)
    with open(get_ab_data_path(behavior), "r") as f:
        ab_data = json.load(f)

    assert len(ab_data) == len(activations_pos), \
        f"Mismatch: {len(ab_data)} items vs {len(activations_pos)} pos activations"

    # Combine and PCA
    activations = t.cat([activations_pos, activations_neg], dim=0)
    activations_np = activations.cpu().numpy()

    pca = PCA(n_components=2)
    projected_activations = pca.fit_transform(activations_np)

    # Apply clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_ids = kmeans.fit_predict(projected_activations)

    # Split pos/neg
    n_pos = activations_pos.shape[0]
    projected_pos = projected_activations[:n_pos]
    projected_neg = projected_activations[n_pos:]
    cluster_pos = cluster_ids[:n_pos]
    cluster_neg = cluster_ids[n_pos:]

    # Group by cluster ID
    clusters_dict = {i: [] for i in range(n_clusters)}

    for i, (x, y) in enumerate(projected_pos):
        clusters_dict[int(cluster_pos[i])].append({
            "behavior": behavior,
            "label": "pos",
            "PC1": float(x),
            "PC2": float(y),
            "item": ab_data[i]
        })

    for i, (x, y) in enumerate(projected_neg):
        clusters_dict[int(cluster_neg[i])].append({
            "behavior": behavior,
            "label": "neg",
            "PC1": float(x),
            "PC2": float(y),
            "item": ab_data[i]  # neg도 같은 item 순서 기준
        })

    # Save as JSON
    save_path = os.path.join(save_dir, fname.replace(".png", "_clusters.json"))
    with open(save_path, "w") as f:
        json.dump(clusters_dict, f, indent=2)

    print(f"Saved clustered PCA data (with item metadata) to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--behaviors",
        nargs="+",
        type=str,
        default=ALL_BEHAVIORS,
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        required=True,
    )
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--activation_type", type=str, choices=["none", "attn", "mlp", "layer"], default="none")
    parser.add_argument("--model_size", type=str, choices=["7b", "13b"], default="7b")
    args = parser.parse_args()
    activation_type = None if args.activation_type == "none" else args.activation_type
    model_name_path = get_model_path(args.model_size, args.use_base_model)
    args = parser.parse_args()
    for behavior in args.behaviors:
        print(f"plotting {behavior} activations PCA")
        for layer in tqdm(args.layers):
            save_activation_projection_pca(
                behavior,
                layer,
                model_name_path,
                activation_type
            )
