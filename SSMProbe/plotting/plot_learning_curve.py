import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

sns.set_theme(style="whitegrid")

base_dir = "checkpoints_mae/s42"
paths = {
    "CLS": "cls",
    "GAP": "gap",
    "Attention Pool": "attention_pool",
    "Content Weighted Pool": "content_weighted_pool",
    "Top-K Pool": "topk_pool",
    "Raster": "raster",
    "Diag 4-Dir": "diag_4dir",
    "Snake 4-Dir": "snake_4dir",
    "VMamba 4-Dir": "vmamba_4dir",
    "Sinkhorn": "sinkhorn",
}

plt.figure(figsize=(10, 6))

for label, folder in paths.items():
    csv_path = os.path.join(base_dir, folder, "loss_log.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        train_df = df[df['stage'] == 'train'].copy()
        if not train_df.empty:
            plt.plot(train_df['step'], train_df['loss'], label=label, linewidth=2)

plt.xlabel("Training Steps")
plt.ylabel("Training Loss")
plt.legend()
plt.tight_layout()
plt.savefig("SSMProbe/paper/nips2026/figures/train_loss_s42.pdf")
print("Saved train_loss_s42.pdf")
