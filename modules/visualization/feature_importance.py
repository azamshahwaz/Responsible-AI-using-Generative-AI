import os
import pandas as pd
import matplotlib.pyplot as plt

from modules.visualization.graph_paper_utils import apply_graph_paper_grid


def plot_feature_importance(
    model,
    feature_names,
    graph_dir
):

    print(
        "\n========== FEATURE IMPORTANCE =========="
    )

    importance = None

    if hasattr(model, "feature_importances_"):

        try:
            importance = model.feature_importances_

        except Exception:
            pass

    if importance is None:
        importance = getattr(
            model,
            "_fi_cache",
            None
        )

    if importance is None:

        print(
            "\nModel does not support feature importance"
        )

        return

    importance_df = (
        pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        })
        .sort_values(
            by="Importance",
            ascending=False
        )
        .head(15)
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    bars = ax.barh(
        importance_df["Feature"],
        importance_df["Importance"],
        zorder=5
    )

    ax.set_title(
        "Top 15 Feature Importance",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel(
        "Importance Score",
        fontsize=12
    )

    ax.set_ylabel(
        "Features",
        fontsize=12
    )

    ax.invert_yaxis()

    plt.tight_layout()

    fig.canvas.draw()

    apply_graph_paper_grid(
        fig,
        ax
    )

    xmax = (
        importance_df["Importance"]
        .max()
    )

    for bar in bars:

        width = bar.get_width()

        ax.text(
            width + (xmax * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            va="center",
            fontsize=9,
            fontweight="bold",
            zorder=6
        )

    os.makedirs(
        graph_dir,
        exist_ok=True
    )

    output_path = os.path.join(
        graph_dir,
        "feature_importance.png"
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