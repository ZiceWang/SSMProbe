import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def parse_layer_wise_results(root_dir: str):
    root_path = Path(root_dir)
    backbones = ["cub_vit-mae", "cub_dinov2", "cub_beit", "cub_vit"]
    layers = ["layer_3", "layer_6", "layer_9", "layer_12"]
    heads = ["gap", "cls", "sinkhorn"]
    seeds = ["seed_42", "seed_43", "seed_44", "seed_45", "seed_46"]

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for bb in backbones:
        for layer in layers:
            for head in heads:
                for seed in seeds:
                    json_path = root_path / bb / seed / layer / head / "eval_metrics.json"
                    if json_path.exists():
                        try:
                            with open(json_path, "r") as f:
                                data = json.load(f)
                                if len(data) > 0:
                                    best_acc = max(item["accuracy"] for item in data) * 100
                                    results[bb][layer][head].append(best_acc)
                        except Exception as e:
                            print(f"Error reading {json_path}: {e}")
                            
    return results, backbones, layers

def plot_layer_wise_evolution(results, backbones, layers, output_path="layer_wise_evolution.pdf"):
    # Set plotting style
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'font.family': 'serif'
    })
    
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), constrained_layout=True)
    
    backbone_titles = {
        "cub_vit-mae": "ViT-MAE (Masked Recon.)",
        "cub_dinov2": "DINOv2 (Self-Distillation)",
        "cub_beit": "BEiT (Masked Image Model)",
        "cub_vit": "ViT (Supervised)"
    }
    
    colors = {"cls": "#d62728", "gap": "#1f77b4", "sinkhorn": "#2ca02c"}
    labels = {"cls": "CLS Token", "gap": "Global Avg Pool", "sinkhorn": "Sinkhorn (Ours)"}
    markers = {"cls": "s", "gap": "^", "sinkhorn": "o"}
    
    x_pos = [3, 6, 9, 12]
    
    for i, bb in enumerate(backbones):
        ax = axes[i]
        
        for head in ["gap", "cls", "sinkhorn"]:
            means = []
            stds = []
            for layer in layers:
                vals = results[bb][layer][head]
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
                    
            means = np.array(means)
            stds = np.array(stds)
            
            # Plot line with error bars
            ax.plot(x_pos, means, marker=markers[head], color=colors[head], 
                    linewidth=2.5, markersize=8, label=labels[head])
            ax.fill_between(x_pos, means - stds, means + stds, alpha=0.15, color=colors[head])
            
        ax.set_title(backbone_titles.get(bb, bb), fontweight='bold', fontsize=15)
        ax.set_xlabel("Network Layer")
        if i == 0:
            ax.set_ylabel("Linear Probe Accuracy (%)")
            
        ax.set_xticks(x_pos)
        ax.set_xticklabels(["Layer 3", "Layer 6", "Layer 9", "Layer 12"])
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.7)
        
    # Legend on the first subplot
    axes[0].legend(loc='upper left', frameon=True, fancybox=True, shadow=False)
        
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    results, backbones, layers = parse_layer_wise_results(repo_root / "results" / "layer_wise_ablation")
    plot_layer_wise_evolution(results, backbones, layers, repo_root / "SSMProbe" / "paper" / "nips2026" / "figs" / "layer_wise_evolution.pdf")
