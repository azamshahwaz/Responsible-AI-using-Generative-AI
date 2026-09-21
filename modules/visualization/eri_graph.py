import os
import numpy as np
import matplotlib.pyplot as plt

from modules.visualization.graph_paper_utils import apply_graph_paper_grid


def plot_eri_comparison(eri_before, eri_after, graph_dir):

    fig, ax = plt.subplots(figsize=(10, 6))

    labels = ["Before", "After"]
    values = [eri_before, eri_after]

    bars = ax.bar(labels, values, width=0.55, zorder=5)

    ax.set_title(
        "Ethical Risk Index (ERI) Comparison",
        fontsize=16, fontweight="bold", pad=15
    )
    ax.set_xlabel("Dataset State",  fontsize=12, fontweight="bold")
    ax.set_ylabel("ERI Score",      fontsize=12, fontweight="bold")

    ymax = max(values)
    ax.set_ylim(0, ymax * 1.20)

    plt.tight_layout()
    fig.canvas.draw()

    apply_graph_paper_grid(fig, ax)

    # Value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + (ymax * 0.02),
            f"{height:.4f}",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
            zorder=6
        )

    # ERI reduction
    improvement = (
        (eri_before - eri_after) / max(eri_before, 0.0001)
    ) * 100

    plt.figtext(
        0.5, 0.02,
        f"Risk Reduction: {improvement:.2f}%",
        ha="center", fontsize=11, fontweight="bold"
    )

    os.makedirs(graph_dir, exist_ok=True)
    output_path = os.path.join(graph_dir, "eri_comparison.png")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nGraph Saved:\n{output_path}")
