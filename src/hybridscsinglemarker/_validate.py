"""逐细胞 DB marker 验证：V[cell] ∈ [0,1]。

公式（设计文档 §4）:
    V = (# c* 特征基因 ∩ DB范围 且在该细胞表达>阈值) / (# c* 特征基因 ∩ DB范围)

实现（全向量化，无逐细胞循环）:
    特征基因 ∩ DB范围 ∩ 查询基因 → 受限基因列表 ``restricted``；
    标签→基因 归属矩阵 M（n_labels × n_restricted，小矩阵稠密）；
    查询表达二值矩阵 E（cells × n_restricted，稀疏，>threshold）；
    一次稀疏 matmul ``counts = E @ M.T``（cells × n_labels，BLAS/C）得每个
    细胞对每个标签的"表达的特征基因数"，再 fancy-index 按每细胞初始标签 c*
    取列，除以该标签受限基因总数。

约定:
    - 表达阈值默认 1.0（log 归一化数据，与 cellmarkerannot 一致）；
    - 分母为 0（c* 特征基因全部不在 DB 范围 / 查询基因集）→ V = 0。
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import sparse as sp

from hybridscsinglemarker.cellmarkerannot.annotation import _resolve_x


def _ordered_union(genes_by_label: dict[str, Sequence[str]]) -> list[str]:
    return list(dict.fromkeys(g for genes in genes_by_label.values() for g in genes))


def validate_cells(
    query,
    genes_by_label: dict[str, Sequence[str]],
    cstar_labels,
    db_range_genes: set[str],
    *,
    layer: str | None = None,
    threshold: float = 1.0,
) -> np.ndarray:
    """批量逐细胞 DB 验证 → V[cell] 数组（长度 n_cells）。

    query: AnnData（X 或 layer）。
    genes_by_label: {标签: 特征基因列表}。
    cstar_labels: 每细胞初始标签 c*（hybrid.py 步骤④ 产物）。
    db_range_genes: (species, tissue) DB 范围 marker 基因集合。
    """
    if not genes_by_label:
        return np.zeros(len(query), dtype=np.float64)

    var_names = list(query.var_names)
    var_pos = {g: i for i, g in enumerate(var_names)}
    label_order = list(genes_by_label.keys())
    lab2i = {lab: i for i, lab in enumerate(label_order)}

    # 受限基因：在 DB 范围 且 在查询中 且 属于某个标签的特征基因
    union = _ordered_union(genes_by_label)
    restricted = [g for g in union if g in db_range_genes and g in var_pos]
    if not restricted:
        return np.zeros(len(query), dtype=np.float64)
    pos = np.array([var_pos[g] for g in restricted], dtype=np.int64)

    # 标签 → 基因归属矩阵（小；按基因→标签映射建立，避免 labels×genes 全扫描）
    gene_labels: dict[str, list[int]] = {}
    for i, lab in enumerate(label_order):
        for g in genes_by_label[lab]:
            if g in db_range_genes and g in var_pos:
                gene_labels.setdefault(g, []).append(i)
    M = np.zeros((len(label_order), len(restricted)), dtype=np.int8)
    for j, g in enumerate(restricted):
        for i in gene_labels.get(g, ()):
            M[i, j] = 1
    if M.sum() == 0:
        return np.zeros(len(query), dtype=np.float64)

    x = _resolve_x(query, layer)                          # CSR (cells × genes)
    xsub = x[:, pos].tocsr()
    E = (xsub > threshold).astype(np.int32)               # 稀疏表达二值
    counts = E @ M.T                                      # (cells × n_labels)
    if sp.issparse(counts):                               # 稀疏 @ 稀疏 → 稀疏
        counts = counts.toarray()
    counts = np.asarray(counts, dtype=np.float64)

    idx = pd.Series(np.asarray(cstar_labels, dtype=object)).map(lab2i).to_numpy()
    valid = ~np.isnan(idx)
    idx_v = idx[valid].astype(np.int64)
    denom = M[idx_v].sum(axis=1)                          # 每细胞 c* 受限基因数
    denom = np.asarray(denom, dtype=np.float64)

    V = np.zeros(len(query), dtype=np.float64)
    nz = np.nonzero(valid)[0]
    d = denom.copy()
    ok = d > 0
    V[nz[ok]] = counts[nz[ok], idx_v[ok]] / d[ok]
    return V


def validate_cell(
    expr_row,
    feature_genes: Sequence[str],
    db_range_genes: set[str],
    threshold: float = 1.0,
) -> float:
    """单细胞验证（供参考/调试）: 返回 V ∈ [0,1]；分母为 0 → 0。"""
    expr = {g: v for g, v in expr_row.items()} if hasattr(expr_row, "items") else None
    denom = 0
    numer = 0
    for g in feature_genes:
        if g not in db_range_genes:
            continue
        denom += 1
        v = expr.get(g, 0.0) if expr is not None else expr_row
        if v > threshold:
            numer += 1
    return numer / denom if denom > 0 else 0.0
