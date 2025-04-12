import json
import torch as t
import os
from matplotlib import pyplot as plt
import argparse
from behaviors import HUMAN_NAMES, get_activations_path, get_analysis_dir, ALL_BEHAVIORS
from utils.helpers import get_model_path, set_plotting_settings
from tqdm import tqdm
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

set_plotting_settings()

def compute_projection_metrics(
        pos_acts: t.Tensor, 
        neg_acts: t.Tensor, 
        compute_method="MD" # "PCA"
    ):
    if compute_method == "MD":
        # 1-1. Compute MD vector
        pos_mean = pos_acts.mean(dim=0)
        neg_mean = neg_acts.mean(dim=0)
        direction = (pos_mean - neg_mean)
        direction = direction / direction.norm()
    elif compute_method == "PCA":
        # 1-2. Compute the first principal component of differences
        diffs = pos_acts - neg_acts  # assumes pos_acts and neg_acts are aligned (same shape)
        diffs_np = diffs.cpu().numpy()
        pca = PCA(n_components=1)
        pca.fit(diffs_np)
        direction_np = pca.components_[0]  # shape: (hidden_dim,)
        direction = t.tensor(direction_np, dtype=pos_acts.dtype, device=pos_acts.device)
    
    direction = direction / direction.norm()  # normalize

    # 2. Project activations
    pos_proj = (pos_acts @ direction).cpu().numpy()
    neg_proj = (neg_acts @ direction).cpu().numpy()

    # 3. Fit Gaussians and compute overlap
    pos_mu, pos_std = pos_proj.mean(), pos_proj.std()
    neg_mu, neg_std = neg_proj.mean(), neg_proj.std()

    def gaussian_overlap(mu1, std1, mu2, std2):
        d = abs(mu1 - mu2)
        std = (std1**2 + std2**2)**0.5
        return 2 * norm.cdf(-d / (2 * std))

    overlap = gaussian_overlap(pos_mu, pos_std, neg_mu, neg_std)

    # 4. ROC AUC
    y_true = [1] * len(pos_proj) + [0] * len(neg_proj)
    y_scores = list(pos_proj) + list(neg_proj)
    auc = roc_auc_score(y_true, y_scores)

    return pos_proj, neg_proj, overlap, auc, direction

def plot_projection_histogram(pos_proj, neg_proj, behavior, layer, save_dir, activation_type, compute_method = "MD"):
    plt.figure(figsize=(8, 5))
    plt.hist(pos_proj, bins=50, alpha=0.5, label='pos', color='blue', density=True)
    plt.hist(neg_proj, bins=50, alpha=0.5, label='neg', color='red', density=True)
    plt.title(f"1D {compute_method} Projection Histogram - {HUMAN_NAMES[behavior]} (Layer {layer})" + (f" - {activation_type} activation" if activation_type else " - residual stream activation"))
    plt.xlabel("Projection onto mean shift vector")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{compute_method}_projection_hist_layer_{layer}.png"))
    plt.close()

def plot_gausian_overlap(layers, overlaps, save_dir, activation_type, compute_method):
    # Plot overlap vs. layer
    plt.figure(figsize=(8, 5))
    plt.plot(layers, overlaps, marker='o')
    plt.title(f"Projection Overlap for {HUMAN_NAMES[behavior]} {compute_method}" + (f" - {activation_type} activation" if activation_type else " - residual stream activation"))
    plt.xlabel("Layer")
    plt.ylabel("Gaussian Overlap (lower is better)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{compute_method}_overlap_plot.png"))
    plt.close()

def plot_roc_auc_overlap(layers, overlaps, save_dir, activation_type, compute_method):
    # Plot AUC vs. layer
    plt.figure(figsize=(8, 5))
    plt.plot(layers, overlaps, marker='o', color='green')
    plt.title(f"Projection ROC AUC for {HUMAN_NAMES[behavior]} {compute_method}" + (f" - {activation_type} activation" if activation_type else " - residual stream activation"))
    plt.xlabel("Layer")
    plt.ylabel("ROC AUC (higher is better)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{compute_method}_auc_plot.png"))
    plt.close()

def plot_pos_neg_sim(layers, similarities, save_dir, activation_type):
    plt.figure(figsize=(8, 5))
    plt.plot(layers, similarities, marker='o', label="Pos-Neg Diff Cosine Similarity")
    plt.title(f"Pos-Neg Contrastive Similarity for {HUMAN_NAMES[behavior]}" + (f" - {activation_type} activation" if activation_type else " - residual stream activation"))
    plt.xlabel("Layer")
    plt.ylabel("Cosine Similarity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"contrastive_similarity_plot.png"))
    plt.close()

def plot_direction_sim(layers, similarities_with_direction, save_dir, activation_type, compute_method):
    plt.figure(figsize=(8, 5))
    plt.plot(layers, similarities_with_direction, marker='o', label="Diff and Compute Method Cosine Similarity")
    plt.title(f"Diff and {compute_method} Similarity for {HUMAN_NAMES[behavior]}" + (f" - {activation_type} activation" if activation_type else " - residual stream activation"))
    plt.xlabel("Layer")
    plt.ylabel("Cosine Similarity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{compute_method}_similarity_plot.png"))
    plt.close()

def compute_and_plot_overlap(behavior: str, layers: list, model_name_path: str, activation_type:str = None, compute_method: str = "MD"):
    overlaps = []
    aucs = []
    save_dir_overlap = os.path.join(get_analysis_dir(behavior), f"projection_overlap-{activation_type}" if activation_type else "projection_overlap")
    os.makedirs(save_dir_overlap, exist_ok=True)

    similarities = []
    similarities_with_direction = []
    save_dir_sim = os.path.join(get_analysis_dir(behavior), f"similarity-{activation_type}" if activation_type else "similarity")
    os.makedirs(save_dir_sim, exist_ok=True)

    for layer in layers:
        activations_pos = t.load(get_activations_path(behavior, layer, model_name_path, "pos", type=activation_type))
        activations_neg = t.load(get_activations_path(behavior, layer, model_name_path, "neg", type=activation_type))

        # compute overlaps
        pos_proj, neg_proj, overlap, auc, direction = compute_projection_metrics(activations_pos, activations_neg, compute_method=compute_method)
        overlaps.append(overlap)
        aucs.append(auc)

        # plot projection to the direction
        plot_projection_histogram(pos_proj, neg_proj, behavior, layer, save_dir_overlap, activation_type, compute_method)

        # compute similarities
        # 1. Cosine similarity between Pos-Neg difference vectors
        pos_diff = activations_pos - activations_neg
        neg_diff = activations_neg - activations_pos
        cosine_sim_pos_neg_diff = cosine_similarity(pos_diff.cpu().numpy(), neg_diff.cpu().numpy()).mean()

        # 2. Cosine similarity between difference vectors and compute_method vector (direction)
        cosine_sim_diff_direction = cosine_similarity(pos_diff.cpu().numpy(), direction.cpu().numpy().reshape(1, -1)).mean()
        
        similarities.append(cosine_sim_pos_neg_diff)
        similarities_with_direction.append(cosine_sim_diff_direction)

    # plot overlaps
    plot_gausian_overlap(layers=layers, overlaps=overlaps, save_dir=save_dir_overlap, activation_type=activation_type, compute_method=compute_method)
    plot_roc_auc_overlap(layers=layers, overlaps=aucs, save_dir=save_dir_overlap, activation_type=activation_type, compute_method=compute_method)

    # plot similarities
    plot_pos_neg_sim(layers, similarities, save_dir_sim, activation_type=activation_type)
    plot_direction_sim(layers, similarities_with_direction, save_dir_sim, activation_type=activation_type, compute_method=compute_method)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--behaviors", nargs="+", type=str, default=ALL_BEHAVIORS)
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--model_size", type=str, choices=["7b", "13b"], default="7b")
    parser.add_argument("--activation_type", type=str, choices=["none", "attn", "mlp", "layer"], default="none")
    parser.add_argument("--compute_method", type=str, choices=["MD", "PCA", "All"], default="All")
    args = parser.parse_args()

    model_name_path = get_model_path(args.model_size, args.use_base_model)

    for behavior in args.behaviors:
        print(f"Computing projection overlap for {behavior} - Methods: {args.compute_method}")
        activation_type = None if args.activation_type == "none" else args.activation_type
        if args.compute_method == "All":
            compute_and_plot_overlap(behavior, args.layers, model_name_path, activation_type, "MD")
            compute_and_plot_overlap(behavior, args.layers, model_name_path, activation_type, "PCA")
        else:
            compute_and_plot_overlap(behavior, args.layers, model_name_path, activation_type, args.compute_method)
