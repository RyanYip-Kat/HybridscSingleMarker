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
    integer range. Bottom: the Score row as bars sharing the cell-type x-axis,
    each annotated with its numeric value and colored by its Score on the same
    ``cmap`` — larger Score → lighter color. The Score scale differs from the
    evidence, so it is kept on its own panel rather than colored by the evidence
    colormap.

    ``save_path`` optionally writes the figure (``bbox_inches="tight"``,
    dpi=150). Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt

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

    fig, (ax_heat, ax_score) = plt.subplots(
        2, 1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [max(1, n_rows), 1]},
    )

    im = ax_heat.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest")
    ax_heat.set_yticks(range(n_rows))
    ax_heat.set_yticklabels(row_labels, fontsize=yfont)
    ax_heat.set_xticks(range(n_cols))
    ax_heat.set_xticklabels([])  # cell-type labels shown on the score panel
    ax_heat.set_ylabel(gene_label)
    if title is not None:
        ax_heat.set_title(title)

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.02)
    cbar.set_label("support evidence")
    vmin, vmax = int(data.min()), int(data.max())
    if vmax - vmin <= 12:  # integer ticks while the evidence range is sparse
        cbar.set_ticks(np.arange(vmin, vmax + 1))

    xpos = np.arange(n_cols)
    # Bars colored by their Score on the same cmap: larger Score -> lighter color.
    score_norm = plt.Normalize(vmin=float(scores.min()), vmax=float(scores.max()))
    bar_colors = plt.colormaps[cmap](score_norm(scores))
    ax_score.bar(xpos, scores, color=bar_colors, alpha=0.85,
                 edgecolor="black", linewidth=0.5)
    for xi, s in zip(xpos, scores):
        ax_score.text(xi, s, f"{s:.3g}", ha="center", va="bottom", fontsize=xfont)
    ax_score.set_xticks(xpos)
    ax_score.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=xfont)
    ax_score.set_xlabel(celltype_label)
    ax_score.set_ylabel("Score")
    ax_score.margins(y=0.25)
    ax_score.set_axisbelow(True)
    ax_score.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
