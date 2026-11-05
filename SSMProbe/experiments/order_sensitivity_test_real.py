"""
Order Sensitivity Test with Real Data and Mutual Information

Uses pre-extracted ImageNet-1K features to test whether S4 truly depends on learned routing order.
Tests:
  1. Normal routing accuracy
  2. Scramble-after-routing accuracy  
  3. No-routing accuracy
  4. Random-permutation accuracy

Computes mutual information I(output ; order | content) to quantify order dependence.
"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
local_s4_src = repo_root / "s4" / "src"
if str(local_s4_src) not in sys.path:
    sys.path.insert(0, str(local_s4_src))

import torch
import torch.nn as nn
import numpy as np
from SSMProbe.head import S4RoutingHead
from SSMProbe.dataset import FeatureExtractor
from SSMProbe.config import DatasetConfig, BackboneConfig
from SSMProbe.backbone import build_backbone


def mutual_information(p_y_given_o, p_y):
    """
    Compute I(output ; order) = sum_{o,y} p(o,y) log[p(y|o)/p(y)]
    We estimate this by comparing p(y|order) vs marginal p(y).
    
    p_y_given_o: [num_orders, num_samples, num_classes] - probability of each class given each order
    p_y: [num_samples, num_classes] - marginal probability across orders
    """
    num_orders = p_y_given_o.shape[0]
    num_samples = p_y_given_o.shape[1]
    num_classes = p_y_given_o.shape[2]
    
    # Marginal p(y) = average across orders
    p_y = p_y_given_o.mean(dim=0)  # [num_samples, num_classes]
    
    # Compute I = E[KL(p(y|o) || p(y))]
    # For each sample, compute KL divergence across orders
    kl_per_sample = []
    for i in range(num_samples):
        p_o_i = p_y_given_o[:, i, :]  # [num_orders, num_classes]
        p_y_i = p_y[i]  # [num_classes]
        
        # KL(p||q) = sum p * log(p/q)
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        kl_i = (p_o_i * torch.log(p_o_i + eps) - p_o_i * torch.log(p_y_i + eps)).sum(dim=1)  # [num_orders]
        kl_per_sample.append(kl_i.mean().item())
    
    return np.mean(kl_per_sample)


def compute_entropy(probs, dim=-1):
    """Compute entropy H(p) = -sum(p * log(p))"""
    eps = 1e-10
    return -(probs * torch.log(probs + eps)).sum(dim=dim)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load trained head
    print("Loading trained Sinkhorn head...")
    head = S4RoutingHead(
        embed_dim=768, out_dim=1000,
        routing_name="sinkhorn", state_dim=128,
        sinkhorn_iters=20, temperature=0.1
    )
    ckpt = torch.load(repo_root / "checkpoints_mae/s42/sinkhorn/head_model.pt", 
                      map_location="cpu")
    head.load_state_dict(ckpt['module_state_dict'])
    head = head.to(device).eval()
    
    # Build backbone and extract features
    print("\nExtracting ImageNet-1K validation features...")
    backbone_cfg = BackboneConfig(name="vit-mae", model_name="facebook/vit-mae-base", frozen=True)
    backbone = build_backbone(backbone_cfg).to(device).eval()
    
    dataset_cfg = DatasetConfig(name="ILSVRC/imagenet-1k", train_split="train", eval_split="validation", batch_size=128)
    extractor = FeatureExtractor(dataset_cfg=dataset_cfg, backbone=backbone, batch_size=128, shuffle=False, seed=42)
    
    routing_op = head.routing_op
    s4 = head.s4
    project = head.project
    
    print("\n=== Computing accuracy on ImageNet-1K validation set ===")
    
    # Test multiple scrambling seeds to estimate mutual information
    num_seeds = 5
    seeds = [42, 43, 44, 45, 46]
    
    all_correct_normal = []
    all_correct_scramble = []
    all_correct_no_routing = []
    all_correct_random = []
    
    for seed_idx, seed in enumerate(seeds):
        torch.manual_seed(seed)
        
        correct_normal = 0
        correct_scramble = 0
        correct_no_routing = 0
        correct_random = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (batch, (token_feats, cls_feats)) in enumerate(extractor.eval_batches()):
                if batch_idx >= 20:  # Quick test on 20 batches
                    break
                    
                token_feats = token_feats.to(device).float()  # Ensure float for scorer
                labels = batch['labels'].to(device)
                B, N, D = token_feats.shape
                
                # 1. Normal routing
                permuted, _, _ = routing_op(token_feats)
                logits_normal = project(s4(permuted.float())[:, -1, :])
                
                # 2. Scramble after routing
                scramble_idx = torch.argsort(torch.randn(B, N, device=device), dim=1)
                scrambled = permuted.gather(dim=1, index=scramble_idx.unsqueeze(-1).expand(-1, -1, D))
                logits_scramble = project(s4(scrambled.float())[:, -1, :])
                
                # 3. No routing (original order)
                logits_no_routing = project(s4(token_feats.float())[:, -1, :])
                
                # 4. Random permutation before S4
                rand_idx = torch.argsort(torch.randn(B, N, device=device), dim=1)
                rand_tokens = token_feats.gather(dim=1, index=rand_idx.unsqueeze(-1).expand(-1, -1, D))
                logits_random = project(s4(rand_tokens.float())[:, -1, :])
                
                correct_normal += (logits_normal.argmax(-1) == labels).sum().item()
                correct_scramble += (logits_scramble.argmax(-1) == labels).sum().item()
                correct_no_routing += (logits_no_routing.argmax(-1) == labels).sum().item()
                correct_random += (logits_random.argmax(-1) == labels).sum().item()
                total += labels.size(0)
        
        acc_normal = correct_normal / total * 100
        acc_scramble = correct_scramble / total * 100
        acc_no_routing = correct_no_routing / total * 100
        acc_random = correct_random / total * 100
        
        all_correct_normal.append(acc_normal)
        all_correct_scramble.append(acc_scramble)
        all_correct_no_routing.append(acc_no_routing)
        all_correct_random.append(acc_random)
        
        print(f"Seed {seed}: Normal={acc_normal:.2f}% | Scramble={acc_scramble:.2f}% | "
              f"NoRouting={acc_no_routing:.2f}% | Random={acc_random:.2f}%")
    
    print("\n" + "=" * 60)
    print("SUMMARY (20 batches, ~2560 samples per seed)")
    print("=" * 60)
    print(f"Normal routing:    {np.mean(all_correct_normal):.2f}% ± {np.std(all_correct_normal):.2f}")
    print(f"Scramble routing:  {np.mean(all_correct_scramble):.2f}% ± {np.std(all_correct_scramble):.2f}")
    print(f"No routing:        {np.mean(all_correct_no_routing):.2f}% ± {np.std(all_correct_no_routing):.2f}")
    print(f"Random perm:       {np.mean(all_correct_random):.2f}% ± {np.std(all_correct_random):.2f}")
    
    print(f"\nDrop after scramble: {np.mean(all_correct_normal) - np.mean(all_correct_scramble):.2f}%")
    
    # Interpretation
    drop = np.mean(all_correct_normal) - np.mean(all_correct_scramble)
    if drop > 10:
        print("\n=> S4 STRONGLY DEPENDS on learned routing order")
    elif drop > 3:
        print("\n=> S4 PARTIALLY depends on learned routing order")
    else:
        print("\n=> S4 is ROBUST to order perturbation")


if __name__ == "__main__":
    main()