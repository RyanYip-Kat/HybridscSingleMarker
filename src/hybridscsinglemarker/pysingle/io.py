"""数据读写与高层注释 API。

- ``_extract_expr_and_labels``：从 ``anndata.AnnData`` 提取表达矩阵与标签；
- ``annotate``：面向用户的高层入口，自动完成 加载 → 提取 → 注释 → 写回；
- ``read_h5ad`` / ``save_h5ad`` / ``load_reference``：基础 h5ad 读写。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .core import singleR_annotate, singleR_annotate_multi
from .seurat_method import seurat_annotate
from .utils import validate_anndata


def _load_adata(data, name: str) -> ad.AnnData:
    """统一加载 h5ad 路径 或 已加载的 ``AnnData`` 对象。"""
    if isinstance(data, (str, Path)):
        return ad.read_h5ad(str(data))
    if isinstance(data, ad.AnnData):
        return data
    raise TypeError(
        f"{name} 须为 h5ad 文件路径（str/Path）或 anndata.AnnData 对象，"
        f"实际为 {type(data).__name__}"
    )


def _extract_expr_and_labels(
    adata: ad.AnnData,
    celltype_col: str | None = None,
    layer: str = "X",
) -> tuple[pd.DataFrame, pd.Series | None]:
    """从 ``AnnData`` 提取表达矩阵与细胞类型标签。

    Parameters:
        adata: ``AnnData`` 对象（X 为 细胞×基因，可为 scipy.sparse）。
        celltype_col: ``adata.obs`` 中的细胞类型标签列名；``None`` 表示不提取
            （查询数据通常无标签）。
        layer: 使用 ``adata.X``（默认 ``"X"``）还是 ``adata.layers`` 中的
            自定义层。

    Returns:
        ``(expr, labels)``：``expr`` 为 **基因×细胞** 的稠密 ``pd.DataFrame``
        （稀疏输入自动稠密化并转置）；``labels`` 为按 ``obs_names`` 对齐的
        标签 ``pd.Series``，``celltype_col`` 为 ``None`` 时为 ``None``。

    Raises:
        KeyError: 指定的 ``celltype_col`` 或 ``layer`` 不存在时。
    """
    validate_anndata(adata)

    # 先校验标签列，缺失时给出可用列，避免先做昂贵稠密化再报错
    if celltype_col is not None:
        if celltype_col not in adata.obs.columns:
            raise KeyError(
                f"参考数据 adata.obs 中缺少细胞类型列 {celltype_col!r}；"
                f"可用列: {list(adata.obs.columns)}"
            )

    # 提取表达矩阵（默认 X 或自定义 layer），保持稀疏以省内存
    if layer == "X" or layer is None:
        X = adata.X
    else:
        if layer not in adata.layers:
            raise KeyError(
                f"adata.layers 中不存在层 {layer!r}；可用层: {list(adata.layers)}"
            )
        X = adata.layers[layer]
    if X is None:
        raise ValueError("AnnData 缺少表达矩阵（X 与所选 layer 均为空）")

    # 转置为 基因×细胞 的稠密 DataFrame（Spearman 需稠密，入口处统一转换）
    if sp.issparse(X):
        dense = X.T.toarray()                    # (细胞, 基因) -> (基因, 细胞)
    else:
        dense = np.asarray(X, dtype=float).T
    expr = pd.DataFrame(
        dense,
        index=pd.Index(adata.var_names, name="gene"),
        columns=pd.Index(adata.obs_names, name="cell"),
    )

    labels: pd.Series | None = None
    if celltype_col is not None:
        labels = adata.obs[celltype_col].astype(str)
        labels.index = adata.obs_names            # 确保与 obs_names 对齐
        labels.name = celltype_col
    return expr, labels


def annotate(
    ref,
    query,
    celltype_col="celltype",
    method: str = "singler",
    layer: str = "X",
    fine_tune: bool = True,
    top_n: int = 5,
    combine_method: str = "max",
    **kwargs,
) -> ad.AnnData:
    """单细胞注释高层入口。

    自动完成：加载参考/查询数据 → 提取表达矩阵与标签 → 调用
    :func:`pysingle.core.singleR_annotate`（单参考）或
    :func:`pysingle.core.singleR_annotate_multi`（多参考）注释 → 将结果
    写回 ``query.obs``。

    支持**多参考（多数据库）注释**（对应现代 ``SingleR(test, ref=list(...),
    labels=list(...))``）：``ref`` 传入列表时，对每个参考分别注释后跨参考
    合并为共识标签（``combine_method``，默认 ``"max"``）。

    支持 **Seurat 参考映射标签转移**（``method="seurat"``，对应
    ``Seurat::FindTransferAnchors + TransferData``）。

    Parameters:
        ref: 参考数据。单参考：h5ad 文件路径或 ``AnnData`` 对象；多参考：
            列表（每个元素为 h5ad 路径或 ``AnnData``）。
        query: 查询数据，格式同 ``ref``（单参考情形）。
        celltype_col: 参考 ``adata.obs`` 中的细胞类型标签列名，默认
            ``"celltype"``。多参考时可为列表，逐参考指定（长度与 ``ref`` 一致）。
        method: 注释方法，``"singler"``（默认，SingleR）或 ``"seurat"``
            （Seurat 参考映射标签转移）。``"seurat"`` 仅支持单参考。
        combine_method: 多参考合并方式，``"max"``（默认）或 ``"mean"``。
        layer: 使用 ``adata.X``（默认）或 ``adata.layers`` 中的自定义层，
            参考与查询共用同一参数。
        fine_tune: 是否启用 fine-tuning 微调，默认 ``True``（仅 singler）。
        top_n: 每个细胞类型取相关性最高的前 ``top_n`` 个参考细胞计算得分，默认 5。
        **kwargs: 透传给 :func:`singleR_annotate`（``method="singler"``）或
            :func:`pysingle.seurat_method.seurat_annotate`（``method="seurat"``）
            的高级参数（如 ``gene_selection``、``max_genes``、``n_jobs``；
            Seurat 的 ``reduction``、``n_dims``、``k_anchor`` 等）。

    Returns:
        写回注释结果的 ``query`` ``AnnData`` 对象（原地修改并返回）：
        - ``query.obs["pysingle_celltype"]``：每个细胞的预测细胞类型；
        - ``query.obs["pysingle_score"]``：对应预测类型的得分。

    Raises:
        TypeError: 输入不是 h5ad 路径或 ``AnnData`` 对象时。
        KeyError: 参考数据缺少 ``celltype_col`` 列，或 ``layer`` 不存在时。

    .. note::
        Spearman 相关性需要对表达矩阵稠密化，超大参考集（数十万细胞 ×
        数万基因）会占用大量内存。建议先用 ``sc.pp.subsample`` 或
        ``adata[...]`` 对参考细胞抽样后再注释，或在注释前过滤基因。
    """
    query_adata = _load_adata(query, "query")
    if query_adata.is_view:                       # 视图写入 obs 会触发隐式拷贝警告
        query_adata = query_adata.copy()
    query_expr, _ = _extract_expr_and_labels(query_adata, None, layer)

    if method == "seurat":
        # ---- Seurat 参考映射标签转移（单参考）----
        ref_adata = _load_adata(ref, "ref")
        ref_expr, ref_labels = _extract_expr_and_labels(ref_adata, celltype_col, layer)
        result = seurat_annotate(
            ref_expr, ref_labels, query_expr, **kwargs,
        )
    elif isinstance(ref, (list, tuple)):
        # ---- 多参考（多数据库）注释 ----
        if isinstance(celltype_col, (list, tuple)):
            label_keys = list(celltype_col)
        else:
            label_keys = [celltype_col] * len(ref)
        if len(label_keys) != len(ref):
            raise ValueError("celltype_col 列表长度须与 ref 列表一致")
        references = []
        for r, key in zip(ref, label_keys):
            adata = _load_adata(r, "ref")
            expr, labels = _extract_expr_and_labels(adata, key, layer)
            references.append((expr, labels))
        result = singleR_annotate_multi(
            references, query_expr,
            combine_method=combine_method,
            fine_tune=fine_tune, top_n=top_n, **kwargs,
        )
    else:
        # ---- 单参考（SingleR）----
        ref_adata = _load_adata(ref, "ref")
        ref_expr, ref_labels = _extract_expr_and_labels(ref_adata, celltype_col, layer)
        result = singleR_annotate(
            ref_expr, ref_labels, query_expr,
            fine_tune=fine_tune, top_n=top_n, **kwargs,
        )

    # 每个细胞的预测类型及其对应得分
    labels = result["labels"]
    scores = result["scores"]
    col_pos = np.asarray([scores.columns.get_loc(lbl) for lbl in labels.to_numpy()])
    pred_scores = scores.to_numpy()[np.arange(len(labels)), col_pos]

    # 写回 query.obs
    query_adata.obs["pysingle_celltype"] = labels.to_numpy()
    query_adata.obs["pysingle_score"] = pred_scores
    return query_adata


def subset_cells(
    adata: ad.AnnData,
    n_cells: int | None,
    *,
    random: bool = False,
    seed: int = 0,
) -> ad.AnnData:
    """取细胞子集，用于大数据场景下先做小规模验证。

    Parameters:
        adata: ``AnnData`` 对象。
        n_cells: 保留的细胞数；``None`` 或 ``>= adata.n_obs`` 时直接返回原对象。
        random: 是否随机抽样；默认 ``False`` 按位置取前 ``n_cells`` 个细胞。
        seed: 随机抽样使用的随机种子，默认 0（保证可复现）。

    Returns:
        子集 ``AnnData``（返回新对象；未抽样时返回原对象）。
    """
    validate_anndata(adata)
    if n_cells is None or n_cells >= adata.n_obs:
        return adata
    if random:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(adata.n_obs, n_cells, replace=False))
        return adata[idx].copy()
    return adata[:n_cells].copy()


def read_h5ad(path: str | Path, *, backed: bool = False) -> ad.AnnData:
    """从 h5ad 文件加载 :class:`anndata.AnnData` 对象。

    Parameters:
        path: h5ad 文件路径。
        backed: 是否以 backed 模式读取（惰性加载，节省内存）。

    Returns:
        加载得到的 ``AnnData`` 对象，``X`` 为 细胞 × 基因 的矩阵。
    """
    return ad.read_h5ad(path, backed=backed)


def load_reference(adata: ad.AnnData, label_key: str) -> tuple[ad.AnnData, pd.Series]:
    """从参考 ``AnnData`` 提取细胞类型标签。

    Parameters:
        adata: 参考 ``AnnData`` 对象（X 为 细胞×基因）。
        label_key: ``adata.obs`` 中存放细胞类型标签的列名。

    Returns:
        ``(adata, labels)`` 元组，其中 ``labels`` 为按 ``obs_names`` 对齐的
        ``pd.Series``，可直接作为 ``singleR_annotate`` 的 ``ref_labels``。
    """
    validate_anndata(adata)
    if label_key not in adata.obs.columns:
        raise KeyError(f"adata.obs 中不存在列 {label_key!r}；可用列: {list(adata.obs.columns)}")
    labels = adata.obs[label_key].astype(str)
    return adata, labels


def save_h5ad(
    adata: ad.AnnData, path: str | Path, *, compression: str | None = "gzip"
) -> None:
    """将 ``AnnData`` 对象写出为 h5ad 文件。

    Parameters:
        adata: 待写出的 ``AnnData`` 对象。
        path: 输出文件路径。
        compression: 压缩方式，默认 ``"gzip"``；``None`` 表示不压缩。
    """
    validate_anndata(adata)
    adata.write_h5ad(path, compression=compression)
