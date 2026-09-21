import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from modules.visualization.graph_paper_utils import apply_graph_paper_grid


def plot_before_after_counts(df_before, df_after, save_dir, max_cols=4):

    import os
    os.makedirs(save_dir, exist_ok=True)

    print("\n========== BEFORE vs AFTER COUNT PLOTS ==========")

    common_cols   = [col for col in df_before.columns if col in df_after.columns]
    selected_cols = common_cols[:max_cols]

    for col in selected_cols:

        if df_after[col].nunique() <= 10:

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # BEFORE
            sns.countplot(x=df_before[col], ax=ax1)
            ax1.set_title(f"Before: {col}", fontweight="bold")

            # AFTER
            sns.countplot(x=df_after[col], ax=ax2)
            ax2.set_title(f"After: {col}", fontweight="bold")

            plt.tight_layout()
            fig.canvas.draw()

            apply_graph_paper_grid(fig, ax1)
            apply_graph_paper_grid(fig, ax2)

            out = os.path.join(save_dir, f"before_after_{col}.png")
            plt.savefig(out, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"\nGraph Saved:\n{out}")


def plot_edqs_comparison(
    before_score,
    after_score,
    graph_dir,
    task_type="classification"
):

    import os
    os.makedirs(graph_dir, exist_ok=True)

    print("\n========== EDQS COMPARISON ==========")

    labels = ["EDQS Before", "EDQS After"]
    values = [before_score, after_score]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(labels, values, width=0.55, zorder=5)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Ethical Data Quality Score (%)", fontsize=11)

    title = (
        "Classification EDQS Before vs After"
        if task_type == "classification"
        else "Regression EDQS Before vs After"
    )
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    fig.canvas.draw()

    apply_graph_paper_grid(fig, ax)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.2f}%",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
            zorder=6
        )

    out = os.path.join(graph_dir, "edqs_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nGraph Saved:\n{out}")
