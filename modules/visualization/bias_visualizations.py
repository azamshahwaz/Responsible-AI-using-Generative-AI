# =========================================================
# BIAS VISUALIZATION MODULE
# =========================================================

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from modules.visualization.graph_paper_utils import apply_graph_paper_grid


# =========================================================
# CREATE SAVE DIRECTORY
# =========================================================

def create_folder(path):
    os.makedirs(path, exist_ok=True)


# =========================================================
# BIAS BAR GRAPH
# =========================================================

def bias_bar_graph(bias_results, save_dir):

    try:

        create_folder(save_dir)

        fig, ax = plt.subplots(figsize=(11, 6))

        bars = ax.bar(
            bias_results["Bias Type"],
            bias_results["Probability"],
            zorder=5
        )

        ax.set_title(
            "Bias Probability Graph",
            fontsize=16, fontweight="bold"
        )
        ax.set_xlabel("Bias Type",    fontsize=12)
        ax.set_ylabel("Probability",  fontsize=12)

        plt.xticks(rotation=20)

        plt.tight_layout()
        fig.canvas.draw()

        apply_graph_paper_grid(fig, ax)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold",
                zorder=6
            )

        save_path = os.path.join(save_dir, "bias_bar_graph.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\nBias Bar Graph Saved:\n{save_path}")

    except Exception as e:
        print(f"\nBias Bar Graph Error: {e}")


# =========================================================
# BEFORE vs AFTER BIAS COMPARISON GRAPH
# =========================================================

def bias_before_after_graph(before_df, after_df, save_dir):

    try:

        create_folder(save_dir)

        merged = pd.merge(
            before_df[["Bias Type", "Probability"]],
            after_df[["Bias Type", "Probability"]],
            on="Bias Type",
            suffixes=("_Before", "_After")
        )

        x     = np.arange(len(merged))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))

        bars_before = ax.bar(
            x - width / 2,
            merged["Probability_Before"],
            width, label="Before", zorder=5
        )

        bars_after = ax.bar(
            x + width / 2,
            merged["Probability_After"],
            width, label="After", zorder=5
        )

        ax.set_xticks(x)
        ax.set_xticklabels(merged["Bias Type"], rotation=30)
        ax.set_ylabel("Bias Probability", fontsize=12)
        ax.set_title(
            "Bias Before vs After Comparison",
            fontsize=16, fontweight="bold"
        )
        ax.legend()

        plt.tight_layout()
        fig.canvas.draw()

        apply_graph_paper_grid(fig, ax)

        for bars in [bars_before, bars_after]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.2f}",
                    ha="center", va="bottom",
                    fontsize=8, zorder=6
                )

        save_path = os.path.join(save_dir, "bias_before_after.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\nBias Comparison Saved:\n{save_path}")

    except Exception as e:
        print(f"\nBias Comparison Error: {e}")


# =========================================================
# BIAS HEATMAP  (no grid needed — seaborn handles it)
# =========================================================

def bias_heatmap(df, save_dir):

    try:

        create_folder(save_dir)

        numeric_df = df.select_dtypes(include=np.number)
        corr       = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(14, 8))

        sns.heatmap(
            corr, annot=True, cmap="coolwarm",
            fmt=".2f", linewidths=0.5, ax=ax
        )

        ax.set_title(
            "Bias Correlation Heatmap",
            fontsize=16, fontweight="bold"
        )

        plt.tight_layout()
        save_path = os.path.join(save_dir, "bias_heatmap.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\nBias Heatmap Saved:\n{save_path}")

    except Exception as e:
        print(f"\nBias Heatmap Error: {e}")


# =========================================================
# FAIRNESS COMPARISON GRAPH
# =========================================================

def fairness_comparison_chart(before_score, after_score, save_dir):

    try:

        create_folder(save_dir)

        labels = ["Before Fix", "After Fix"]
        scores = [before_score, after_score]

        fig, ax = plt.subplots(figsize=(8, 5))

        bars = ax.bar(labels, scores, zorder=5)

        ax.set_ylim(0, 1)
        ax.set_ylabel("Fairness Score", fontsize=12)
        ax.set_title(
            "Fairness Comparison",
            fontsize=16, fontweight="bold"
        )

        plt.tight_layout()
        fig.canvas.draw()

        apply_graph_paper_grid(fig, ax)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                zorder=6
            )

        improvement = (
            (after_score - before_score) / max(before_score, 0.0001)
        ) * 100

        plt.figtext(
            0.5, 0.02,
            f"Fairness Improvement: {improvement:.2f}%",
            ha="center", fontsize=10, fontweight="bold"
        )

        save_path = os.path.join(save_dir, "fairness_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\nFairness Comparison Saved:\n{save_path}")

    except Exception as e:
        print(f"\nFairness Comparison Error: {e}")
