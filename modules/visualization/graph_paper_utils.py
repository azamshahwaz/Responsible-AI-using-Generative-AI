import numpy as np
from matplotlib.ticker import FixedLocator


def apply_graph_paper_grid(fig, ax, n_major=10):

    # --------------------------------------------------
    # 1. Measure physical axes size
    # --------------------------------------------------

    bbox         = ax.get_position()
    fig_w, fig_h = fig.get_size_inches()
    ax_w         = bbox.width  * fig_w   # inches
    ax_h         = bbox.height * fig_h   # inches

    # --------------------------------------------------
    # 2. Data ranges
    # --------------------------------------------------

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    x_range = xmax - xmin
    y_range = ymax - ymin

    if x_range <= 0 or y_range <= 0 or ax_w <= 0 or ax_h <= 0:
        return

    # --------------------------------------------------
    # 3. Inch-per-data-unit on each axis
    # --------------------------------------------------

    ipu_x = ax_w / x_range
    ipu_y = ax_h / y_range

    # --------------------------------------------------
    # 4. Y spacings  (divide Y range into n_major blocks)
    #    Major: 1 block   Minor: 1/10 block
    # --------------------------------------------------

    dy_major = y_range / n_major
    dy_minor = dy_major / 10

    # --------------------------------------------------
    # 5. X spacings derived so physical size == Y
    #    dx * ipu_x == dy * ipu_y
    #    => dx = dy * ipu_y / ipu_x
    # --------------------------------------------------

    dx_major = dy_major * ipu_y / ipu_x
    dx_minor = dy_minor * ipu_y / ipu_x

    # --------------------------------------------------
    # 6. Build tick arrays
    # --------------------------------------------------

    y_maj = np.arange(ymin, ymax + dy_major * 0.5, dy_major)
    y_min = np.arange(ymin, ymax + dy_minor * 0.5, dy_minor)

    x_maj = np.arange(xmin, xmax + dx_major * 0.5, dx_major)
    x_min = np.arange(xmin, xmax + dx_minor * 0.5, dx_minor)

    # --------------------------------------------------
    # 7. Apply locators
    # --------------------------------------------------

    ax.set_axisbelow(True)
    ax.set_facecolor("#f7fff7")

    ax.yaxis.set_major_locator(FixedLocator(y_maj))
    ax.yaxis.set_minor_locator(FixedLocator(y_min))
    ax.xaxis.set_major_locator(FixedLocator(x_maj))
    ax.xaxis.set_minor_locator(FixedLocator(x_min))

    # --------------------------------------------------
    # 8. Grid lines  (green graph-paper style)
    # --------------------------------------------------

    ax.grid(
        which="minor", axis="both",
        linestyle="-", linewidth=0.45,
        color="#81c784", alpha=0.90
    )

    ax.grid(
        which="major", axis="both",
        linestyle="-", linewidth=1.2,
        color="#2e7d32", alpha=0.95
    )

    # --------------------------------------------------
    # 9. Spine style
    # --------------------------------------------------

    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="both", labelsize=10)
