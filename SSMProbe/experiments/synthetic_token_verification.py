"""
Synthetic Token Verification (E1 + M1/M2/M3/M5).

Validates that three diagnostic scores independently track three orthogonal
ground-truth parameters.

Design:
  π* = optimal permutation (first N//4 positions carry the label signal)
  train head under π*, freeze α_k + classifier
  eval under: a_spatial (partially matches π*), random, π* (optimal), reverse(π*)

Parameters:
  topo_strength: fraction of positions where spatial order matches π*
  het_strength:  concentration of α_label (higher -> fewer positions matter)
  ord_strength:  asymmetry of π* (higher -> reverse differs more from π*)

Label: sign(Σ α_label_k · embed(s_k[π*(k)]) · w_label)
  where α_label is concentrated on first N*het_strength positions.

Usage:
  python -m SSMProbe.experiments.synthetic_token_verification --grid-size 8
"""
from __future__ import annotations
import argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────
# Token generation
# ──────────────────────────────────────────────────────────────────────

def generate_tokens(N=64, d_model=32, topo_strength=0.5, het_strength=0.5,
                    ord_strength=0.0, noise_std=0.1, num_samples=800, seed=42):
    """Generate tokens and labels with controllable diagnostic parameters."""
    rng = np.random.RandomState(seed)

    # ── Build s_k with spatial autocorrelation ──
    # A smooth 2D random field gives realistic s_k
    gs = int(np.sqrt(N))
    s_2d = rng.randn(gs, gs).astype(np.float32)
    # Gaussian blur for spatial smoothness
    from scipy.ndimage import gaussian_filter
    s_2d = gaussian_filter(s_2d, sigma=0.8)
    s_k = torch.from_numpy(s_2d.flatten())
    s_k = (s_k - s_k.mean()) / (s_k.std() + 1e-6)

    # Ground-truth metrics
    cn = torch.arange(gs, dtype=torch.float32).repeat_interleave(gs) / max(gs-1, 1)
    gt_ord = abs(float(np.corrcoef(s_k.numpy(), cn.numpy())[0, 1]))

    adj_corrs = []
    for i in range(gs):
        for j in range(gs):
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i+di, j+dj
                if 0<=ni<gs and 0<=nj<gs:
                    k, nk = i*gs+j, ni*gs+nj
                    adj_corrs.append(float((s_k[k]-s_k.mean())*(s_k[nk]-s_k.mean())))
    gt_topo = np.mean(adj_corrs)/(s_k.std().item()**2+1e-9) if adj_corrs else 0.0
    gt_het = float(s_k.std())

    # ── π*: optimal permutation ──
    # First N*ord_strength tokens are "special" (sorted by s_k), rest are mixed
    n_special = max(1, int(N * abs(ord_strength)))
    if ord_strength > 0:
        # High-s tokens at front
        pi_star = torch.cat([
            torch.topk(s_k, n_special).indices,
            torch.argsort(s_k, descending=True)[n_special:]  # rest sorted
        ])
    elif ord_strength < 0:
        # Low-s tokens at front (reverse pattern)
        pi_star = torch.cat([
            torch.topk(-s_k, n_special).indices,
            torch.argsort(s_k)[n_special:]  # rest sorted ascending
        ])
    else:
        # Random permutation (no structure)
        pi_star = torch.randperm(N, generator=torch.Generator().manual_seed(seed))

    # ── Spatial permutation: agrees with π* on topo_strength fraction ──
    n_spatial_match = int(N * topo_strength)
    spatial_perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed+999))
    if n_spatial_match > 0:
        spatial_perm[:n_spatial_match] = pi_star[:n_spatial_match]

    # ── α_label: concentrated with scale het_strength ──
    front_len = max(1, int(N * max(0.05, het_strength)))
    alpha_label = torch.zeros(N)
    alpha_label[:front_len] = torch.exp(-torch.arange(front_len, dtype=torch.float32) /
                                         (front_len/3.0 + 1e-6))
    alpha_label /= alpha_label.sum()

    # ── Build embeddings ──
    proj_dir = torch.from_numpy(rng.randn(d_model).astype(np.float32))
    proj_dir /= proj_dir.norm()
    w_label = torch.from_numpy(rng.randn(d_model).astype(np.float32))
    w_label /= w_label.norm()

    all_tokens, all_labels = [], []
    for si in range(num_samples):
        s_noisy = s_k + torch.from_numpy(rng.randn(N).astype(np.float32)) * noise_std * 0.3
        token = s_noisy.unsqueeze(1) * proj_dir.unsqueeze(0)
        n_noise = torch.from_numpy(rng.randn(N, d_model).astype(np.float32))
        n_noise -= (n_noise @ proj_dir).unsqueeze(1) * proj_dir.unsqueeze(0)
        token = token + n_noise * noise_std * 0.2

        ordered = token[pi_star]
        z = (ordered * alpha_label.unsqueeze(1)).sum(dim=0)
        label = 1 if (z * w_label).sum() > 0 else 0

        all_tokens.append(token); all_labels.append(label)

    tokens, labels = torch.stack(all_tokens), torch.tensor(all_labels, dtype=torch.long)
    meta = dict(N=N, topo_strength=topo_strength, het_strength=het_strength,
                ord_strength=ord_strength, gt_topo=float(gt_topo), gt_het=gt_het,
                gt_ord=gt_ord, pi_star=pi_star, spatial_perm=spatial_perm,
                alpha_label=alpha_label)
    return tokens, labels, meta


# ──────────────────────────────────────────────────────────────────────
# Trainable S4 head
# ──────────────────────────────────────────────────────────────────────

class TrainableS4Head(nn.Module):
    def __init__(self, N, d_model, num_classes=2):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(N))
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, tokens, perm_indices):
        B, N, d = tokens.shape
        alpha = F.softmax(self.log_alpha, dim=0)
        ordered = tokens.gather(1, perm_indices.unsqueeze(-1).expand(-1,-1,d))
        z = (ordered * alpha[None,:,None]).sum(dim=1)
        return self.classifier(z)


# ──────────────────────────────────────────────────────────────────────
# Core protocol
# ──────────────────────────────────────────────────────────────────────

def train_and_freeze(head, tokens, labels, perm, epochs=150, lr=0.01, bs=128):
    head.train()
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    n = tokens.shape[0]
    for ep in range(epochs):
        idx = torch.randperm(n)
        for s in range(0, n, bs):
            bi = idx[s:s+bs]
            pb = perm.unsqueeze(0).expand(bi.shape[0], -1)
            loss = F.cross_entropy(head(tokens[bi], pb), labels[bi])
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head


def eval_frozen(head, tokens, labels, perm, bs=256):
    correct, total = 0, 0
    for s in range(0, tokens.shape[0], bs):
        idx = slice(s, s+bs)
        pb = perm.unsqueeze(0).expand(tokens[idx].shape[0], -1)
        correct += (head(tokens[idx], pb).argmax(1) == labels[idx]).sum().item()
        total += tokens[idx].shape[0]
    return correct / total


def compute_scores(args, N, d_model, tokens, labels, pi_star, spatial_perm, device):
    tokens_d, labels_d = tokens.to(device), labels.to(device)
    n_tr = int(0.7 * tokens.shape[0])
    tr_t, te_t = tokens_d[:n_tr], tokens_d[n_tr:]
    tr_l, te_l = labels_d[:n_tr], labels_d[n_tr:]

    pst = pi_star.to(device)

    # Train ONE head under π* (optimal), freeze
    head = TrainableS4Head(N, d_model, 2).to(device)
    frozen = train_and_freeze(head, tr_t, tr_l, pst, args.epochs, args.lr, args.batch_size)

    with torch.no_grad():
        # Spatial (partially matches π*)
        acc_sp = eval_frozen(frozen, te_t, te_l, spatial_perm.to(device))

        # Random (average over multiple)
        acc_rnd = float(np.mean([
            eval_frozen(frozen, te_t, te_l,
                        torch.randperm(N, generator=torch.Generator().manual_seed(100+i)).to(device))
            for i in range(args.num_random_perms)
        ]))

        # Optimal (π*)
        acc_opt = eval_frozen(frozen, te_t, te_l, pst)

        # Reversed π*
        acc_rev = eval_frozen(frozen, te_t, te_l, pst.flip(0))

    return dict(
        acc_spatial=acc_sp, acc_random=acc_rnd, acc_optimal=acc_opt, acc_reversed=acc_rev,
        delta_topo=acc_sp - acc_rnd,
        delta_het=acc_opt - max(acc_sp, acc_rnd),
        delta_ord=acc_opt - acc_rev
    )


# ──────────────────────────────────────────────────────────────────────
# Main sweep
# ──────────────────────────────────────────────────────────────────────

def run_sweep(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    N = args.grid_size * args.grid_size
    print(f"Device: {device},  N={N},  d={args.d_model}")
    print(f"Samples: {args.num_samples},  Epochs: {args.epochs},  LR: {args.lr}")
    print(f"Repeats: {args.num_repeats},  Random perms: {args.num_random_perms}")
    print()

    def _sweep(vals, pname, ct, ch, co):
        print("━" * 95)
        print(f"SWEEP: Vary {pname.upper()} (topo={ct}, het={ch}, ord={co})")
        hdr = (f"  {pname:>6s}  {'Δ_topo':>8s}  {'Δ_het':>8s}  "
               f"{'Δ_ord':>8s}  {'gt':>8s}  {'sp':>6s}  {'rnd':>6s}  "
               f"{'opt':>6s}  {'rev':>6s}")
        print(hdr); print("-" * len(hdr))
        results = []
        gk = {"topo": "gt_topo", "het": "gt_het", "ord": "gt_ord"}[pname]
        for pv in vals:
            slist = []
            for rep in range(args.num_repeats):
                kw = dict(N=N, d_model=args.d_model, noise_std=args.noise_std,
                          num_samples=args.num_samples, seed=1000+rep)
                kw["topo_strength"] = pv if pname == "topo" else ct
                kw["het_strength"] = pv if pname == "het" else ch
                kw["ord_strength"] = pv if pname == "ord" else co
                tokens, labels, meta = generate_tokens(**kw)
                sc = compute_scores(args, N, args.d_model, tokens, labels,
                                    meta["pi_star"], meta["spatial_perm"], device)
                slist.append(sc)
            avg = {k: float(np.mean([s[k] for s in slist])) for k in slist[0]}
            results.append((pv, avg, meta[gk]))
            print(f"  {pv:6.2f}  {avg['delta_topo']:8.4f}  {avg['delta_het']:8.4f}  "
                  f"{avg['delta_ord']:8.4f}  {meta[gk]:8.4f}  "
                  f"{avg['acc_spatial']:6.4f}  {avg['acc_random']:6.4f}  "
                  f"{avg['acc_optimal']:6.4f}  {avg['acc_reversed']:6.4f}")

        va = np.array([r[0] for r in results])
        dt = np.array([r[1]["delta_topo"] for r in results])
        dh = np.array([r[1]["delta_het"] for r in results])
        do = np.array([r[1]["delta_ord"] for r in results])

        def sc(x, y):
            if x.std() < 1e-9 or y.std() < 1e-9: return 0.0
            c = np.corrcoef(x, y)[0, 1]
            return float(c) if not np.isnan(c) else 0.0

        ct_, ch_, co_ = sc(va, dt), sc(va, dh), sc(va, do)
        idx = {"topo": 0, "het": 1, "ord": 2}[pname]
        cors = [ct_, ch_, co_]
        tgt = abs(cors[idx])
        off = sum(abs(cors[j]) for j in range(3) if j != idx)
        sel = tgt / (off + 0.01)
        print(f"  Corr: Δ_topo={ct_:+.3f} Δ_het={ch_:+.3f} Δ_ord={co_:+.3f} -> sel={sel:.1f}×")
        return sel

    s1 = _sweep([0.0, 0.33, 0.67, 1.0], "topo", None, 0.5, 0.3)
    s2 = _sweep([0.05, 0.2, 0.5, 0.8, 1.0], "het", 0.5, None, 0.3)
    s3 = _sweep([-0.5, -0.25, 0.0, 0.25, 0.5], "ord", 0.5, 0.5, None)

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    p = 0
    for lb, sel in [("Topology", s1), ("Heterogeneity", s2), ("Order-dep.", s3)]:
        st = "✓ PASS" if sel > 5 else ("⚠ MARGINAL" if sel > 2 else "✗ FAIL")
        if sel > 5: p += 1
        print(f"  {lb}: {st}  (selectivity={sel:.1f}×)")
    print(f"\n  {p}/3 sweeps selective.")
    if p == 3:
        print("  -> Three-score decomposition VALIDATED.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid-size", type=int, default=8)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--num-samples", type=int, default=800)
    p.add_argument("--noise-std", type=float, default=0.1)
    p.add_argument("--num-random-perms", type=int, default=10)
    p.add_argument("--num-repeats", type=int, default=3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--cpu", action="store_true")
    run_sweep(p.parse_args())


if __name__ == "__main__":
    main()
