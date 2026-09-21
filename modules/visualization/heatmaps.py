import os
import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_heatmap(
    df,
    graph_dir
):

    print(
        "\n========== CORRELATION HEATMAP =========="
    )

    numeric_df = df.select_dtypes(
        include=[
            "int64",
            "float64"
        ]
    )

    if len(numeric_df.columns) < 2:

        print(
            "\nCorrelation Heatmap Skipped "
            "(insufficient numeric columns)"
        )

        return

    plt.figure(
        figsize=(10, 8)
    )

    correlation_matrix = (
        numeric_df.corr()
    )

    sns.heatmap(
        correlation_matrix,
        annot=False,
        cmap="coolwarm",
        linewidths=0.5
    )

    plt.title(
        "Feature Correlation Heatmap"
    )

    output_path = os.path.join(
        graph_dir,
        "correlation_heatmap.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nCorrelation Heatmap Saved:"
        f"\n{output_path}"
    )