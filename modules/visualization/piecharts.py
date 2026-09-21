import os
import matplotlib.pyplot as plt


def plot_piecharts(
    df_before,
    df_after,
    cat_cols,
    save_dir,
    max_cols=2
):

    print("\n========== PIE CHART COMPARISON ==========")

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    valid_cols = [

        col for col in cat_cols

        if (
            col in df_before.columns
            and col in df_after.columns
        )
    ]

    for col in valid_cols[:max_cols]:

        if df_after[col].nunique() <= 10:

            plt.figure(figsize=(8, 4))

            # BEFORE
            plt.subplot(1, 2, 1)

            df_before[col].value_counts().plot.pie(
                autopct='%1.1f%%'
            )

            plt.title(
                f"Before: {col}"
            )

            plt.ylabel('')

            # AFTER
            plt.subplot(1, 2, 2)

            df_after[col].value_counts().plot.pie(
                autopct='%1.1f%%'
            )

            plt.title(
                f"After: {col}"
            )

            plt.ylabel('')

            plt.tight_layout()

            output_path = os.path.join(
                save_dir,
                f"piechart_{col}.png"
            )

            plt.savefig(
                output_path,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            print(
                f"Pie Chart Saved:\n{output_path}"
            )