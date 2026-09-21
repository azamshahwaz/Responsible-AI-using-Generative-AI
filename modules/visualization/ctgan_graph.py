import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from modules.visualization.graph_paper_utils import apply_graph_paper_grid


# =========================================================
# CTGAN REAL vs SYNTHETIC COMPARISON
# =========================================================

def plot_ctgan_real_vs_synthetic(real_df, synthetic_df, graph_dir):

    print("\n========== CTGAN REAL vs SYNTHETIC ==========")

    os.makedirs(graph_dir, exist_ok=True)

    numeric_cols = real_df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    numeric_cols = [
        col for col in numeric_cols
        if col.lower() not in ["id", "passengerid", "student_id"]
    ]

    if len(numeric_cols) == 0:
        print("\nNo Numeric Columns Found")
        return

    top_features = numeric_cols[:4]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()

    for ax, col in zip(axes, top_features):

        sns.kdeplot(
            real_df[col], ax=ax,
            label="Real Data", fill=True, linewidth=2
        )

        sns.kdeplot(
            synthetic_df[col], ax=ax,
            label="Synthetic Data", fill=True, linewidth=2
        )

        ax.set_title(col, fontsize=13, fontweight="bold")
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=9)

    # Remove empty subplots
    if len(top_features) < 4:
        for i in range(len(top_features), 4):
            fig.delaxes(axes[i])

    plt.suptitle(
        "CTGAN Real vs Synthetic Distribution Comparison",
        fontsize=18, fontweight="bold", y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.canvas.draw()

    # Apply square grid to each active subplot
    for ax, col in zip(axes, top_features):
        apply_graph_paper_grid(fig, ax)

    output_path = os.path.join(graph_dir, "ctgan_real_vs_synthetic.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nGraph Saved:\n{output_path}")
