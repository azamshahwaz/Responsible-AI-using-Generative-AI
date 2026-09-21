import os
import numpy as np
import matplotlib.pyplot as plt

from modules.visualization.graph_paper_utils import apply_graph_paper_grid


def plot_edqs_comparison(
    before_score,
    after_score,
    graph_dir,
    task_type="classification"
):

    print(
        "\n========== EDQS COMPARISON =========="
    )

    labels = [
        "EDQS Before",
        "EDQS After"
    ]

    values = [
        before_score,
        after_score
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    bars = ax.bar(
        labels,
        values,
        width=0.55,
        zorder=5
    )

    # =====================================================
    # AXIS SETTINGS
    # =====================================================

    ax.set_ylim(
        0,
        100
    )

    ax.set_ylabel(
        "Ethical Data Quality Score (%)",
        fontsize=11
    )

    if task_type == "classification":

        ax.set_title(
            "Classification EDQS Before vs After",
            fontsize=14,
            fontweight="bold"
        )

    else:

        ax.set_title(
            "Regression EDQS Before vs After",
            fontsize=14,
            fontweight="bold"
        )

    # =====================================================
    # GRAPH PAPER GRID  (perfect squares)
    # =====================================================

    plt.tight_layout()
    fig.canvas.draw()

    apply_graph_paper_grid(fig, ax)

    # =====================================================
    # VALUE LABELS  (after grid so zorder=6 stays on top)
    # =====================================================

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            zorder=6
        )

    # =====================================================
    # IMPROVEMENT TEXT
    # =====================================================

    improvement = (
        (after_score - before_score)
        / max(before_score, 0.0001)
    ) * 100

    plt.figtext(
        0.5,
        0.02,
        f"EDQS Improvement: {improvement:.2f}%",
        ha="center",
        fontsize=10,
        fontweight="bold"
    )

    # =====================================================
    # SAVE
    # =====================================================

    os.makedirs(
        graph_dir,
        exist_ok=True
    )

    output_path = os.path.join(
        graph_dir,
        "edqs_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nGraph Saved:\n{output_path}"
    )
