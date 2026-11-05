"""Plot token sensitivity results — publication-quality PDFs for the paper.

Produces two figures:
  1. Drop curves (5 representative heads) -> combo_drop_curves.pdf
  2. Permutation delta bar chart -> perm_delta_bar.pdf

Usage: run from repo root:
    python SSMProbe/plotting/plot_token_sensitivity.py
"""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

DISPLAY = {
    'cls':                     'CLS',
    'gap':                     'GAP',
    'raster':                  'Raster',
    'snake_4dir':              'Snake',
    'vmamba_4dir':             'VMamba',
    'diag_4dir':               'Diagonal',
    'sinkhorn':                'Sinkhorn',
    'big_transformer': 'Transformer',
    'bidirectional_gru':       'Bi-GRU',
    'deep_sets':               'DeepSets',
    'attention_pool':          'AttnPool',
    'content_weighted_pool':   'ContentPool',
    'sinkhorn_1dconv':         'Sinkhorn+Conv',
    'topk_pool':               'TopK',
    'random_fixed_perm':       'Rand-Fixed',
    'random_dynamic_perm':     'Rand-Dynamic',
}

# Heads to show in the drop-curve grid (6 representative tiers, 2×3 grid)
DROP_HEADS = ['gap', 'raster', 'snake_4dir', 'sinkhorn', 'bidirectional_gru', 'big_transformer']

# Color palette for methods
METHOD_COLORS = {
    'cls_top':    '#d62728',
    'cls_bottom': '#2ca02c',
    'random':     '#7f7f7f',
}
METHOD_LABELS = {
    'cls_top':    'Drop top-$k$ (by CLS sim.)',
    'cls_bottom': 'Drop bottom-$k$',
    'random':     'Drop random $k$',
}

PERM_ORDER = ['shuffle', 'reverse', 'cls_desc', 'cls_asc']
PERM_LABELS = {
    'shuffle':   'Shuffle',
    'reverse':   'Reverse',
    'cls_desc':  'CLS-desc',
    'cls_asc':   'CLS-asc',
}


def main():
    repo_root = Path(__file__).resolve().parents[2]
    results_json = repo_root / 'results' / 'token_sensitivity' / 'mae_imagenet_seed42.json'
    out_dir = repo_root / 'SSMProbe' / 'paper' / 'nips2026' / 'figures' / 'token_sensitivity'
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results_json.exists():
        print('Results not found:', results_json)
        return

    data = json.loads(results_json.read_text())
    df = pd.DataFrame(data)

    sns.set_style('whitegrid')
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'legend.fontsize': 8,
        'figure.dpi': 150,
    })

    # ── Figure 1: Drop curves (selected heads) ────────────────────────────────
    n = len(DROP_HEADS)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows),
                             squeeze=False)

    for idx, head in enumerate(DROP_HEADS):
        ax = axes[idx // cols][idx % cols]
        sub = df[(df['head'] == head) & (df['intervention'] == 'drop')]
        baseline = df[(df['head'] == head) & (df['intervention'] == 'baseline')]['accuracy'].mean()

        for method in ['cls_top', 'cls_bottom', 'random']:
            msub = sub[sub['method'] == method].sort_values('param')
            if msub.empty:
                continue
            ax.plot(msub['param'], msub['accuracy'] * 100,
                    marker='o', markersize=4, linewidth=1.5,
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method])

        ax.axhline(baseline * 100, color='black', linestyle='--',
                    linewidth=0.8, alpha=0.6, label='Baseline')
        ax.set_title(DISPLAY.get(head, head), fontweight='bold')
        ax.set_xlabel('Dropped tokens $k$')
        ax.set_ylabel('Accuracy (%)')
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Hide unused subplot
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    # Single legend at bottom
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4,
               fontsize=8, frameon=True, bbox_to_anchor=(0.52, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out1 = out_dir / 'combo_drop_curves.pdf'
    fig.savefig(out1, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print('Saved', out1)

    # ── Figure 2: Permutation delta bar chart ─────────────────────────────────
    perm_df = df[df['intervention'] == 'permute'].copy()
    # Only heads that have permutation data (skip order-invariant baselines
    # that are visually boring; include all for completeness)
    perm_heads = sorted(perm_df['head'].unique(),
                        key=lambda h: -perm_df[perm_df['head'] == h]['delta'].abs().max())

    fig2, ax2 = plt.subplots(figsize=(6, 3.2))
    x_pos = 0
    group_gap = 0.6
    bar_width = 0.18
    colors_perm = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd']
    x_ticks, x_labels = [], []

    for head in perm_heads:
        hsub = perm_df[perm_df['head'] == head]
        for j, perm_name in enumerate(PERM_ORDER):
            row = hsub[hsub['method'] == perm_name]
            if row.empty:
                x_pos += bar_width
                continue
            delta = row['delta'].values[0] * 100
            ax2.bar(x_pos, delta, width=bar_width, color=colors_perm[j],
                    edgecolor='white', linewidth=0.3)
            x_pos += bar_width
        x_ticks.append(x_pos - bar_width * 2)
        x_labels.append(DISPLAY.get(head, head))
        x_pos += group_gap

    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_labels, rotation=35, ha='right', fontsize=8)
    ax2.set_ylabel('$\\Delta$ Accuracy (%)')
    ax2.axhline(0, color='black', linewidth=0.6)
    ax2.legend([PERM_LABELS[p] for p in PERM_ORDER], ncol=4, fontsize=8,
               loc='lower right', frameon=True)
    fig2.tight_layout()

    out2 = out_dir / 'perm_delta_bar.pdf'
    fig2.savefig(out2, format='pdf', bbox_inches='tight')
    plt.close(fig2)
    print('Saved', out2)


if __name__ == '__main__':
    main()
