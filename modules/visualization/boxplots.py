import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from modules.visualization.graph_paper_utils import apply_graph_paper_grid


# =========================================================
# BEFORE vs AFTER BOXPLOTS
# =========================================================

def plot_boxplots(df_before, df_after, save_dir, max_cols=4):

    print("\n========== BEFORE vs AFTER BOXPLOTS ==========")

    os.makedirs(save_dir, exist_ok=True)

    num_before = df_before.select_dtypes(
        include=["int64", "float64"]
    ).columns

    num_after = df_after.select_dtypes(
        include=["int64", "float64"]
    ).columns

    valid_cols = [col for col in num_before if col in num_after]

    for col in valid_cols[:max_cols]:

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # =====================================================
        # BEFORE
        # =====================================================

        sns.boxplot(y=df_before[col], ax=axes[0], width=0.4)

        axes[0].set_title(
            f"Before: {col}",
            fontsize=14, fontweight="bold"
        )
        axes[0].set_ylabel(col, fontsize=11)
        axes[0].set_xlabel("")

        # =====================================================
        # AFTER
        # =====================================================

        sns.boxplot(y=df_after[col], ax=axes[1], width=0.4)

        axes[1].set_title(
            f"After: {col}",
            fontsize=14, fontweight="bold"
        )
        axes[1].set_ylabel(col, fontsize=11)
        axes[1].set_xlabel("")

        # =====================================================
        # MAIN TITLE + LAYOUT
        # =====================================================

        fig.suptitle(
            f"Boxplot Comparison: {col}",
            fontsize=18, fontweight="bold"
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.canvas.draw()

        # Apply square grid to both subplots
        apply_graph_paper_grid(fig, axes[0])
        apply_graph_paper_grid(fig, axes[1])

        output_path = os.path.join(save_dir, f"boxplot_{col}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"\nGraph Saved:\n{output_path}")
