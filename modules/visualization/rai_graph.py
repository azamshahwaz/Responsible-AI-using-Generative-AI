import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator


def plot_rai_comparison(
    rai_before,
    rai_after,
    graph_dir
):

    DPI        = 600
    CM_PER_IN  = 2.54

    # Physical size of ONE major square = 1 cm
    MAJOR_CM   = 1.0
    MAJOR_IN   = MAJOR_CM / CM_PER_IN   # inches

    # Both axes share same data range: 0..110
    DATA_RANGE = 110.0          # same on x and y
    DY_MAJOR   = 10.0
    DY_MINOR   =  1.0
    DX_MAJOR   = DY_MAJOR       # same data units → same physical size
    DX_MINOR   = DY_MINOR

    # Number of major squares
    N_MAJOR    = int(DATA_RANGE / DY_MAJOR)   # 11

    # Axes physical size (equal on both → squares)
    AX_IN      = N_MAJOR * MAJOR_IN           # 11 cm

    # =====================================================
    # FIGURE SIZE  (add margins)
    # =====================================================

    L, R, T, B = 1.1, 0.5, 0.9, 1.0   # inches
    fig_w      = L + AX_IN + R
    fig_h      = T + AX_IN + B

    fig        = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)

    ax = fig.add_axes([
        L / fig_w,
        B / fig_h,
        AX_IN / fig_w,
        AX_IN / fig_h
    ])

    # =====================================================
    # BAR POSITIONS in 0..110 x-space
    # Place bars at x=28 and x=83 (spread evenly)
    # bar width in data units = 8
    # =====================================================

    ymax_val    = max(rai_before, rai_after)
    upper_limit = max(110, np.ceil(ymax_val / 10) * 10 + 10)

    BAR_W       = 8.0
    X_BEFORE    = 28.0
    X_AFTER     = 83.0

    bar_data = [
        (X_BEFORE, rai_before, "#E07B39", "RAI Before"),
        (X_AFTER,  rai_after,  "#3A7FD5", "RAI After"),
    ]

    bars = []
    for xc, val, col, lbl in bar_data:
        b = ax.bar(
            xc, val,
            width     = BAR_W,
            color     = col,
            edgecolor = "#2a2a2a",
            linewidth = 1.2,
            zorder    = 5
        )
        bars.append((b[0], val, lbl))

    # =====================================================
    # AXIS LIMITS — square: both 0..upper_limit
    # =====================================================

    ax.set_xlim(0, upper_limit)
    ax.set_ylim(0, upper_limit)

    # =====================================================
    # TICK LOCATORS
    # =====================================================

    ax.set_facecolor("#ffffff")
    ax.set_axisbelow(True)

    major_ticks = np.arange(0, upper_limit + DX_MAJOR, DX_MAJOR)
    minor_ticks = np.arange(0, upper_limit + DX_MINOR, DX_MINOR)

    ax.xaxis.set_major_locator(FixedLocator(major_ticks))
    ax.xaxis.set_minor_locator(FixedLocator(minor_ticks))
    ax.yaxis.set_major_locator(FixedLocator(major_ticks))
    ax.yaxis.set_minor_locator(FixedLocator(minor_ticks))

    # =====================================================
    # GRID  (real graph paper colours)
    # =====================================================

    ax.grid(which="minor", axis="both",
            linestyle="-", linewidth=0.35,
            color="#a8c8e8", alpha=1.0)

    ax.grid(which="major", axis="both",
            linestyle="-", linewidth=1.1,
            color="#3a6fa8", alpha=1.0)

    # =====================================================
    # BORDER
    # =====================================================

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color("#222222")

    # =====================================================
    # Y TICK LABELS  (numbers 0,10,20…)
    # =====================================================

    ax.tick_params(axis="y", which="major", labelsize=10)
    ax.tick_params(axis="y", which="minor", length=0, labelsize=0)

    # =====================================================
    # X TICK LABELS
    # Hide numeric ticks; add category labels at bar centres
    # =====================================================

    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    for _, val, lbl in bars:
        xc = X_BEFORE if "Before" in lbl else X_AFTER
        ax.text(
            xc, -upper_limit * 0.04,
            lbl,
            ha="center", va="top",
            fontsize=11, fontweight="bold",
            transform=ax.transData,
            clip_on=False
        )

    # =====================================================
    # VALUE LABELS ON BARS
    # =====================================================

    for bar_obj, val, lbl in bars:
        ax.text(
            bar_obj.get_x() + bar_obj.get_width() / 2,
            val + upper_limit * 0.012,
            f"{val:.2f}",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold",
            color="#111111"
        )

    # =====================================================
    # TITLE & AXIS LABELS
    # =====================================================

    ax.set_title(
        "Responsible AI Index (RAI) Before vs After",
        fontsize=14, fontweight="bold", pad=14
    )
    ax.set_ylabel("RAI Score (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Dataset State",  fontsize=11, fontweight="bold",
                  labelpad=28)

    # =====================================================
    # IMPROVEMENT TEXT
    # =====================================================

    improvement = (
        (rai_after - rai_before) / max(rai_before, 0.0001)
    ) * 100

    fig.text(
        0.5, 0.01,
        f"RAI Improvement: {improvement:.2f}%",
        ha="center", fontsize=11, fontweight="bold"
    )

    # =====================================================
    # SAVE
    # =====================================================

    os.makedirs(graph_dir, exist_ok=True)
    output_path = os.path.join(graph_dir, "rai_comparison.png")
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"\nGraph Saved:\n{output_path}")
