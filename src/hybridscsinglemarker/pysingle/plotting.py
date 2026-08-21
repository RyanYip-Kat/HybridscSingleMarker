"""SingleR 风格可视化（对应 R 包 ``SingleR.Plotting.R``）。

配色与样式尽量与 R 版一致：

- ``SINGLER_COLORS`` 复刻 R ``singler.colors``：RColorBrewer 全部
  qualitative 调色板按字母序拼接、剔除 4/27 位后重复三次；
- 得分热图使用 ``pheatmap`` 默认的 ``rev(RdYlBu)`` 配色；
- 散点/箱线图采用透明背景、黑坐标轴的 ``theme_classic`` 风格。

matplotlib 采用惰性导入：核心算法包可在未安装 matplotlib 时正常使用。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# RColorBrewer qualitative 调色板（顺序与 R 的 brewer.pal.info 字母序一致）
# ---------------------------------------------------------------------------

_BREWER_QUAL: list[tuple[str, list[str]]] = [
    ("Accent", ["#7FC97F", "#BEAED4", "#FDC086", "#FFFF99", "#386CB0",
                "#F0027F", "#BF5B17", "#666666"]),
    ("Dark2", ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E",
               "#E6AB02", "#A6761D", "#666666"]),
    ("Paired", ["#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", "#FB9A99",
                "#E31A1C", "#FDBF6F", "#FF7F00", "#CAB2D6", "#6A3D9A",
                "#FFFF99", "#B15928"]),
    ("Pastel1", ["#FBB4AE", "#B3CDE3", "#CCEBC5", "#DECBE4", "#FED9A6",
                 "#FFFFCC", "#E5D8BD", "#FDDAEC", "#F2F2F2"]),
    ("Pastel2", ["#B3E2CD", "#FDCDAC", "#CBD5E8", "#F4CAE4", "#E6F5C9",
                 "#FFF2AE", "#F1E2CC", "#CCCCCC"]),
    ("Set1", ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
              "#FFFF33", "#A65628", "#F781BF", "#999999"]),
    ("Set2", ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854",
              "#FFD92F", "#E5C494", "#B3B3B3"]),
    ("Set3", ["#8DD3C7", "#FFFFB3", "#BEBADA", "#FB8072", "#80B1D3",
              "#FDB462", "#B3DE69", "#FCCDE5", "#D9D9D9", "#BC80BD",
              "#CCEBC5", "#FFED6F"]),
]


def _build_singler_colors() -> list[str]:
    """复刻 R: ``unlist(mapply(brewer.pal, ...))[c(-4,-27)]`` 重复三次。"""
    colors = [c for _, pal in _BREWER_QUAL for c in pal]
    drop_1based = {4, 27}                      # R: singler.colors[c(-4,-27)]
    colors = [c for i, c in enumerate(colors) if (i + 1) not in drop_1based]
    return colors * 3                          # R: c(colors, colors, colors)


SINGLER_COLORS: list[str] = _build_singler_colors()


def _import_matplotlib():
    """惰性导入 matplotlib，未安装时给出明确提示。"""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError as exc:                 # pragma: no cover
        raise ImportError(
            "可视化需要 matplotlib，请先安装：uv add --group dev matplotlib"
        ) from exc
    matplotlib.rcParams["axes.facecolor"] = "white"
    return plt


# ---------------------------------------------------------------------------
# 得分热图（对应 R: SingleR.DrawHeatmap）
# ---------------------------------------------------------------------------

def plot_score_heatmap(
    result: dict[str, Any],
    top_n: int = 40,
    cells: list | None = None,
    *,
    figsize: tuple[float, float] | None = None,
    show_cell_labels: bool = False,
) -> Any:
    """得分热图：行 = 细胞（按预测类型聚集），列 = 细胞类型（得分）。

    对应 R ``SingleR.DrawHeatmap``（pheatmap 风格），并针对"细胞标签过多
    重叠成黑团"做了优化：

    - **同种预测类型的细胞聚集为连续块**（组内 ward 聚类，块间按类型字母序）；
    - 左侧一条颜色条（``SINGLER_COLORS``）标注每细胞的预测类型，替代逐细胞
      标签；y 轴每个类型块中心仅显示一个类型名；
    - 图例映射 颜色 -> 预测细胞类型。

    其余细节与 R 一致：逐细胞对类型 z-score 取 top_n 类型、逐细胞归一化后
    取立方、ward.D2 聚类、``pheatmap`` 默认 ``rev(RdYlBu)`` 配色
    （低值蓝、高值红）。

    Parameters:
        result: ``singleR_annotate`` 的返回字典。
        top_n: 保留的细胞类型数量（逐细胞 z-score 后按类型最大值排序），默认 40。
        cells: 仅展示这些细胞（R 的 ``cells.use``），默认全部。
        figsize: 图像尺寸，默认 ``(8, 10)``。
        show_cell_labels: 是否显示每个细胞的刻度标签（默认不显示，避免黑团）。

    Returns:
        ``matplotlib.figure.Figure``。
    """
    plt = _import_matplotlib()
    from scipy.cluster.hierarchy import leaves_list, linkage
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedFormatter, FixedLocator

    scores = result["scores"]
    labels = result["labels"]
    if cells is not None:
        scores = scores.loc[cells]
        labels = labels.loc[cells]

    # R: m = apply(t(scale(t(scores))), 2, max)
    #     scale 逐"细胞"对类型做 z-score（ddof=1，同 R scale 的样本 sd），
    #     再对每类型取跨细胞最大值，按降序保留 top.n 个类型。
    #     注意用 .sub/.div(axis=0) 按"行"对齐（Series 索引为细胞，与列不同名）
    z = scores.sub(scores.mean(axis=1), axis=0).div(scores.std(axis=1, ddof=1), axis=0)
    z = z.fillna(0.0)                                 # 常数列 sd==0 时不参与选择
    keep = z.max(axis=0).sort_values(ascending=False).index[:top_n]
    scores = scores[keep]

    # R: (x - mmin)/(mmax - mmin) 后 ^3（按细胞即行归一化）
    data = scores.sub(scores.min(axis=1), axis=0).div(
        scores.max(axis=1) - scores.min(axis=1), axis=0
    )
    data = np.nan_to_num(np.power(data.to_numpy(dtype=float), 3), nan=0.0)

    # ---- 细胞按预测类型聚集（同种细胞聚为连续块，组内 ward 聚类）----
    label_arr = labels.to_numpy()
    type_order = sorted(pd.unique(label_arr))          # 字母序（与调色板映射一致）
    ordered: list[int] = []
    for t in type_order:
        grp = np.where(label_arr == t)[0]
        if grp.size > 1:
            link = linkage(data[grp], method="ward")
            grp = grp[leaves_list(link)]
        ordered.extend(grp.tolist())
    ordered = np.array(ordered, dtype=int)
    data = data[ordered]

    # 类型列（x 轴）ward 聚类（对应 pheatmap clustering_method='ward.D2'）
    col_link = linkage(data.T, method="ward")
    data = data[:, leaves_list(col_link)]
    type_cols = scores.columns.to_numpy()[leaves_list(col_link)]

    # ---- 绘制：左侧类型颜色条 + 主热图 ----
    if figsize is None:
        figsize = (8, 10)
    fig, (ax_bar, ax) = plt.subplots(
        1, 2, figsize=(figsize[0] + 0.6, figsize[1]),
        gridspec_kw={"width_ratios": [0.06, 1], "wspace": 0.03},
        constrained_layout=True,
    )

    # 左侧颜色条：SINGLER_COLORS 标注每细胞预测类型
    # 注意：不能 sharey（会把 tick formatter 路由到第一个轴导致热图标签失效），
    #       改用手动对齐 ylim。
    cats = type_order
    color_map = {c: SINGLER_COLORS[i % len(SINGLER_COLORS)] for i, c in enumerate(cats)}
    if "X" in color_map:
        color_map["X"] = "black"
    bar_colors = np.array(
        [to_rgb(color_map[c]) for c in label_arr[ordered]]   # hex -> (r,g,b)
    )
    ax_bar.imshow(bar_colors[:, None, :], aspect="auto")
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])

    # 主热图
    im = ax.imshow(
        data,
        aspect="auto",
        cmap="RdYlBu_r",                        # pheatmap 默认 rev(RdYlBu)
        interpolation="nearest",
    )
    ax.set_xticks(range(len(type_cols)))
    ax.set_xticklabels(type_cols, rotation=90, fontsize=6)
    if show_cell_labels:
        ax.yaxis.set_major_locator(FixedLocator(range(len(ordered))))
        ax.yaxis.set_major_formatter(FixedFormatter(
            scores.index.to_numpy()[ordered]))
        ax.tick_params(axis="y", labelsize=5)
    else:
        # y 轴：每个类型块中心显示一个类型名（避免逐细胞标签成黑团）
        boundaries = [0]
        for t in type_order:
            boundaries.append(boundaries[-1] + int((label_arr[ordered] == t).sum()))
        centers = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(len(type_order))]
        ax.yaxis.set_major_locator(FixedLocator(centers))
        ax.yaxis.set_major_formatter(FixedFormatter(type_order))
        ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax_bar.set_ylim(ax.get_ylim())                    # 手动对齐颜色条与热图
    ax.set_ylabel("Cells (grouped by predicted type)")
    ax.set_xlabel("Cell type (score)")

    # 图例：颜色 -> 预测类型
    legend = [Patch(facecolor=color_map[c], edgecolor="none", label=c) for c in cats]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=min(len(cats), 6), fontsize=8, frameon=False,
              title="predicted type")

    fig.colorbar(im, ax=ax, shrink=0.7, label="score (normalized^3)")
    return fig


# ---------------------------------------------------------------------------
# 注释散点图（对应 R: SingleR.PlotTsne）
# ---------------------------------------------------------------------------

def plot_annotation_scatter(
    result: dict[str, Any],
    coords: np.ndarray,
    labels: pd.Series | None = None,
    *,
    dot_size: float = 10.0,
    alpha: float = 0.5,
    title: str = "",
    figsize: tuple[float, float] = (6, 5),
) -> Any:
    """按注释标签着色的 t-SNE / UMAP 散点图。

    对应 R ``SingleR.PlotTsne``：使用 ``SINGLER_COLORS`` 按类型字母序着色
    （``X`` 标签固定为黑色），透明背景、无网格、黑色坐标轴。

    Parameters:
        result: ``singleR_annotate`` 的返回字典。
        coords: 每个细胞的二维坐标，形状 ``(n_cells, 2)``，顺序与
            ``result["labels"]`` 一致。
        labels: 可选，覆盖默认标签（``result["labels"]``）。
        dot_size: 点的大小。
        alpha: 点的透明度。
        title: 图标题。
        figsize: 图像尺寸。

    Returns:
        ``matplotlib.figure.Figure``。
    """
    plt = _import_matplotlib()

    labels = pd.Series(labels if labels is not None else result["labels"])
    labels = labels.reindex(result["labels"].index)
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords 必须为 (n_cells, 2) 的二维数组")

    cats = sorted(labels.dropna().unique().tolist())     # R: factor levels 字母序
    palette = {c: SINGLER_COLORS[i % len(SINGLER_COLORS)] for i, c in enumerate(cats)}
    if "X" in palette:
        palette["X"] = "black"                            # R: cols['X']='black'

    fig, ax = plt.subplots(figsize=figsize)
    for cat in cats:
        mask = (labels == cat).to_numpy()
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=dot_size, alpha=alpha, c=palette[cat],
            edgecolors="none", label=cat,
        )

    # R PlotTsne: 无网格、透明背景、黑色坐标轴
    ax.set_axisbelow(True)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
    ax.tick_params(color="black")
    ax.set_xlabel("tSNE 1")
    ax.set_ylabel("tSNE 2")
    ax.set_title(title)

    num_levels = len(cats)
    if num_levels > 35:                                   # R: 图例放底部
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
                  ncol=min(num_levels, 9), fontsize=6)
    else:
        font_size = max(5, min(10, 250 / max(num_levels, 1)))
        ax.legend(fontsize=font_size)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 单细胞相关分布箱线图（对应 R: SingleR.DrawBoxPlot）
# ---------------------------------------------------------------------------

def plot_cell_boxplot(
    query_expr,
    ref_expr,
    ref_labels: pd.Series,
    cell_id: int | str,
    *,
    top_n: int = 50,
    figsize: tuple[float, float] = (10, 6),
    quantile_order: float = 0.8,
) -> Any:
    """展示单个查询细胞与各参考类型细胞的 Spearman 相关分布箱线图。

    对应 R ``SingleR.DrawBoxPlot``：类型按 ``quantile_order``（默认 0.8）
    分位排序取 top_n，箱线图叠加抖动散点，全部使用黑色（R 中
    ``scale_color_manual(rep('black',8))``），x 轴标签旋转 45°。

    Parameters:
        query_expr: 查询表达矩阵（``pd.DataFrame``）。
        ref_expr: 参考表达矩阵（``pd.DataFrame``）。
        ref_labels: 参考细胞类型标签，索引与参考细胞 ID 对齐。
        cell_id: 待展示的查询细胞 ID。
        top_n: 展示的细胞类型数量，默认 50。
        quantile_order: 类型排序所用的相关分位数，默认 0.8。

    Returns:
        ``matplotlib.figure.Figure``。
    """
    plt = _import_matplotlib()
    from .core import _expr_to_dense, _intersect_and_filter, _spearman_corr

    ref_dense, ref_genes, ref_cells = _expr_to_dense(ref_expr, "ref_expr")
    query_dense, query_genes, query_cells = _expr_to_dense(query_expr, "query_expr")

    # 定位目标细胞列
    if query_cells is not None:
        if isinstance(cell_id, str) and cell_id not in query_cells:
            raise ValueError(f"cell_id {cell_id!r} 不在查询矩阵中")
        ci = list(query_cells).index(cell_id)
    else:
        ci = int(cell_id)

    # 基因交集 + 质量过滤（与 singleR_annotate 保持一致）
    if ref_genes is None or query_genes is None:
        if ref_dense.shape[0] != query_dense.shape[0]:
            raise ValueError("缺少基因名时，参考与查询必须行数一致")
        ref_sub, query_sub = ref_dense, query_dense
    else:
        ref_sub, query_sub, _ = _intersect_and_filter(
            ref_dense, ref_genes, query_dense, query_genes
        )

    r = _spearman_corr(query_sub[:, [ci]], ref_sub)[0]    # 单个细胞 vs 全部参考细胞
    lab = ref_labels.reindex(ref_cells).astype(str).to_numpy() if ref_cells is not None \
        else np.asarray(ref_labels, dtype=str)

    df = pd.DataFrame({"corr": r, "type": lab})
    order = (
        df.groupby("type")["corr"]
        .quantile(quantile_order)
        .sort_values(ascending=False)
        .index[:top_n]
    )
    df = df[df["type"].isin(order)]

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(
        [df.loc[df["type"] == t, "corr"].to_numpy() for t in order],
        patch_artist=True, showfliers=False,
    )
    # R: geom_boxplot(alpha=0.2) + scale_color_manual(rep('black',8))
    for patch in bp["boxes"]:
        patch.set_facecolor("grey")                        # 浅灰箱体（alpha 0.2 效果）
        patch.set_alpha(0.2)
        patch.set_edgecolor("black")
    for whisker, cap, median in zip(bp["whiskers"], bp["caps"], bp["medians"]):
        whisker.set_color("black"); cap.set_color("black"); median.set_color("black")

    rng = np.random.default_rng(0)
    for j, t in enumerate(order):
        y = df.loc[df["type"] == t, "corr"].to_numpy()
        x = j + 1 + rng.uniform(-0.2, 0.2, size=y.size)   # R: position='jitter'
        ax.scatter(x, y, s=8, alpha=0.4, color="black", edgecolors="none")

    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Spearman correlation")
    ax.set_title(str(cell_id))
    ax.grid(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 单细胞 vs 参考样本散点（对应 R: SingleR.DrawScatter）
# ---------------------------------------------------------------------------

def plot_cell_vs_ref_scatter(
    query_expr,
    ref_expr,
    cell_id: int | str,
    sample_id: int | str,
    *,
    figsize: tuple[float, float] = (6, 5),
) -> Any:
    """单细胞与单个参考样本的表达值散点 + 线性拟合。

    对应 R ``SingleR.DrawScatter``：取共有基因，横轴为单个查询细胞、
    纵轴为单个参考样本的表达值；蓝色散点 + 红色 lm 拟合线，
    标题标注两者 Spearman 相关 ``R``，``theme_classic`` 风格。

    Parameters:
        query_expr: 查询表达矩阵（``pd.DataFrame`` / ``AnnData`` 等）。
        ref_expr: 参考表达矩阵，格式同 ``query_expr``。
        cell_id: 待展示的查询细胞 ID。
        sample_id: 待展示的参考样本（细胞）ID。
        figsize: 图像尺寸。

    Returns:
        ``matplotlib.figure.Figure``。
    """
    plt = _import_matplotlib()
    from .core import _expr_to_mat, _intersect_and_filter

    q_mat, q_genes, q_cells = _expr_to_mat(query_expr, "query_expr")
    r_mat, r_genes, r_cells = _expr_to_mat(ref_expr, "ref_expr")
    if q_genes is not None and r_genes is not None:
        q_sub, r_sub, _ = _intersect_and_filter(q_mat, q_genes, r_mat, r_genes)
    else:
        q_sub, r_sub = q_mat, r_mat

    if q_cells is not None:
        ci = list(q_cells).index(cell_id)
    else:
        ci = int(cell_id)
    if r_cells is not None:
        si = list(r_cells).index(sample_id)
    else:
        si = int(sample_id)

    def _col(mat, j):                                # 取第 j 列并稠密化
        col = np.asarray(mat[:, j].toarray() if hasattr(mat, "toarray") else mat[:, j])
        return col.ravel().astype(float)

    x, y = _col(q_sub, ci), _col(r_sub, si)

    fig, ax = plt.subplots(figsize=figsize)
    # R: geom_point(size=0.5, alpha=0.5, color='blue')
    ax.scatter(x, y, s=3, alpha=0.5, color="blue", edgecolors="none")
    # R: geom_smooth(method='lm', color='red')
    slope, intercept = np.polyfit(x, y, 1)
    x_range = np.linspace(x.min(), x.max(), 50)
    ax.plot(x_range, slope * x_range + intercept, color="red")

    spearman = stats.spearmanr(x, y).statistic        # R: cor(..., method='spearman')
    ax.set_xlabel("Single cell")
    ax.set_ylabel("Reference sample")
    ax.set_title(f"R = {spearman:.3f}")
    ax.grid(False)                                    # theme_classic：无网格、黑坐标轴
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
    fig.tight_layout()
    return fig
