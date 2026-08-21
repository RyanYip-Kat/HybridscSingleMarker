"""Heatmap visualization for gene-list evidence matrices.
matplotlib is imported lazily inside :func:`plot_gene_scores` so that importing
this module (and the package) stays cheap — importing matplotlib costs ~0.5 s.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_gene_scores(
    matrix: pd.DataFrame,
    *,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    cmap: str = "viridis",
    celltype_label: str = "cell type",
    gene_label: str = "gene",
    top_n: int = 20,
    save_path: str | os.PathLike[str] | None = None,
) -> Figure:
    """Plot an evidence matrix + Score row as a two-panel figure.
    Cell types (columns) are sorted by the Score row (the DataFrame's *last*
    row) **descending**, left to right, and limited to the top ``top_n`` (default
    20) when there are more — so wide scopes stay readable.
    Top: ``imshow`` heatmap of the integer evidence rows (rows = genes, columns
    = cell types) colored by ``cmap`` (a sequential colormap for magnitude), with
    a colorbar labeled **"support evidence"** — the legend covers the evidence
    integer range. Bottom: the Score row as a heatmap row sharing the cell-type 
    x-axis, colored by a purple sequential colormap.
    ``save_path`` optionally writes the figure (``bbox_inches="tight"``,
    dpi=150). Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if len(matrix) < 2:
        raise ValueError("matrix must have evidence rows plus a final 'Score' row")

    # Sort cell types by the Score row descending; keep the top ``top_n``.
    score_row = matrix.iloc[-1]
    matrix = matrix.loc[:, score_row.sort_values(ascending=False).index]
    if top_n is not None and matrix.shape[1] > top_n:
        matrix = matrix.iloc[:, :top_n]

    n_rows = len(matrix) - 1
    n_cols = matrix.shape[1]
    if n_cols == 0:
        raise ValueError("matrix has no cell-type columns")

    data = matrix.iloc[:-1].to_numpy(dtype=np.float64)
    scores = matrix.iloc[-1].to_numpy(dtype=np.float64)
    row_labels = [str(x) for x in matrix.index[:-1]]
    col_labels = [str(x) for x in matrix.columns]

    if figsize is None:
        figsize = (
            max(5.0, 0.30 * n_cols + 1.0),
            max(4.0, 0.24 * n_rows + 2.2),
        )

    # Shrink tick fonts as the matrix grows so labels stay legible.
    yfont = max(6.0, min(12.0, 120.0 / max(1, n_rows)))
    xfont = max(6.0, min(12.0, 90.0 / max(1, n_cols)))

    # 自定义热图配色：蓝→紫→红渐变，匹配参考图风格；若用户传入cmap则优先使用用户配置
    if cmap == "viridis":
        heat_cmap = LinearSegmentedColormap.from_list(
            "blue_purple_red",
            ["#000066", "#1a53ff", "#aa33cc", "#ff44aa", "#ff1a1a"],
            N=256
        )
    else:
        heat_cmap = cmap
    score_cmap = "Purples_r"  # Score行：高分浅紫，低分深紫，匹配参考图视觉

    fig, (ax_heat, ax_score) = plt.subplots(
        2, 1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [max(1, n_rows), 1]},
    )

    # ========== 深色背景设置 ==========
    fig.patch.set_facecolor("black")
    ax_heat.set_facecolor("black")
    ax_score.set_facecolor("black")

    # ========== 上方证据热图 ==========
    im = ax_heat.imshow(data, aspect="auto", cmap=heat_cmap, interpolation="nearest")

    # 单元格添加白色数值文本
    for i in range(n_rows):
        for j in range(n_cols):
            val = int(data[i, j])
            ax_heat.text(
                j, i, str(val),
                ha="center", va="center",
                color="white", fontsize=yfont * 0.9
            )

    ax_heat.set_yticks(range(n_rows))
    ax_heat.set_yticklabels(row_labels, fontsize=yfont, color="white")
    ax_heat.set_xticks(range(n_cols))
    ax_heat.set_xticklabels([])  # 细胞类型标签统一放在底部Score轴
    ax_heat.set_ylabel(gene_label, color="white")
    if title is not None:
        ax_heat.set_title(title, color="white")

    # 右侧证据色条
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.02)
    cbar.set_label("support evidence", color="white")
    cbar.ax.tick_params(colors="white")
    vmin, vmax = int(data.min()), int(data.max())
    if vmax - vmin <= 12:
        cbar.set_ticks(np.arange(vmin, vmax + 1))
    # 色条两端标注 Low/High，匹配参考图样式
    cbar.ax.text(
        1.6, 0, "Low (Support)",
        va="center", ha="left", color="white",
        rotation=270, transform=cbar.ax.transAxes
    )
    cbar.ax.text(
        1.6, 1, "High",
        va="center", ha="left", color="white",
        rotation=270, transform=cbar.ax.transAxes
    )

    # ========== 底部Score热图行 ==========
    score_data = scores.reshape(1, -1)
    im_score = ax_score.imshow(
        score_data, aspect="auto",
        cmap=score_cmap, interpolation="nearest"
    )

    # Score行Y轴标签（紫色文字）
    ax_score.set_yticks([0])
    ax_score.set_yticklabels(
        ["Score (Confidence)"],
        fontsize=yfont, color="#cc99ff"
    )
    ax_score.tick_params(axis="y", length=0)  # 隐藏Y轴刻度线

    # 底部细胞类型标签
    ax_score.set_xticks(range(n_cols))
    ax_score.set_xticklabels(
        col_labels, rotation=45, ha="right",
        fontsize=xfont, color="white"
    )
    ax_score.set_xlabel(celltype_label, color="white")

    # 底部Score色条
    cbar_score = fig.colorbar(
        im_score, ax=ax_score,
        orientation="horizontal",
        fraction=0.06, pad=0.12
    )
    cbar_score.set_label("Score Max", color="#cc99ff")
    cbar_score.ax.tick_params(colors="#cc99ff")
    s_min, s_max = float(scores.min()), float(scores.max())
    cbar_score.set_ticks([s_min, s_max])
    cbar_score.set_ticklabels(["0", f"{int(round(s_max, 0))}"])

    fig.tight_layout()

    if save_path is not None:
        # 保存时保留黑色背景
        fig.savefig(save_path, bbox_inches="tight", dpi=150, facecolor="black")

    return fig

