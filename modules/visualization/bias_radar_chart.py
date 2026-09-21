import os
import numpy as np
import matplotlib.pyplot as plt


def bias_radar_chart(
    bias_before,
    bias_after,
    graph_dir
):

    labels = bias_before[
        "Bias Type"
    ].tolist()

    before_values = bias_before[
        "Probability"
    ].tolist()

    after_values = bias_after[
        "Probability"
    ].tolist()

    # circle close karne ke liye
    before_values += before_values[:1]
    after_values += after_values[:1]

    angles = np.linspace(
        0,
        2 * np.pi,
        len(labels),
        endpoint=False
    ).tolist()

    angles += angles[:1]

    fig = plt.figure(
        figsize=(8, 8)
    )

    ax = plt.subplot(
        111,
        polar=True
    )

    ax.plot(
        angles,
        before_values,
        linewidth=2,
        label="Before"
    )

    ax.fill(
        angles,
        before_values,
        alpha=0.25
    )

    ax.plot(
        angles,
        after_values,
        linewidth=2,
        label="After"
    )

    ax.fill(
        angles,
        after_values,
        alpha=0.25
    )

    ax.set_xticks(
        angles[:-1]
    )

    ax.set_xticklabels(
        labels
    )

    plt.title(
        "Bias Radar Chart"
    )

    plt.legend(
        loc="upper right"
    )

    output_path = os.path.join(
        graph_dir,
        "bias_radar_chart.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nBias Radar Chart Saved:"
        f"\n{output_path}"
    )