"""通用工具函数。

包含数据标准化、表达谱聚合与输入校验等辅助逻辑。
"""

from __future__ import annotations

import anndata as ad
import numpy as np


def validate_anndata(adata: ad.AnnData, *, require_x: bool = True) -> None:
    """校验 ``AnnData`` 对象的基本结构与完整性。

    Parameters:
        adata: 待校验的 ``AnnData`` 对象。
        require_x: 是否要求存在表达矩阵 ``X``。

    Raises:
        TypeError: ``adata`` 不是 ``AnnData`` 实例时。
        ValueError: 缺失 ``X`` 或形状非法时。
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError(
            f"期望 anndata.AnnData 实例，实际为 {type(adata).__name__}"
        )
    if require_x and adata.X is None:
        raise ValueError("AnnData 缺少表达矩阵 X")
    if adata.X is not None:
        n_obs, n_var = adata.X.shape
        if n_obs != len(adata.obs_names) or n_var != len(adata.var_names):
            raise ValueError("AnnData 的 X 维度与 obs_names/var_names 不一致")


def log1p_normalize(counts: np.ndarray) -> np.ndarray:
    """执行 ``log1p`` 标准化，对应 ``Seurat::LogNormalize``。

    先将每个细胞的计数归一化到总读数（median library size），
    再取 ``log1p``。适用于 ``numpy`` 数组或稀疏矩阵。

    Parameters:
        counts: 原始计数矩阵，形状 ``(n_cells, n_genes)``。

    Returns:
        标准化后的表达矩阵，形状与 ``counts`` 一致。
    """
    raise NotImplementedError


def aggregate_profiles(
    adata: ad.AnnData,
    label_key: str,
    *,
    method: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """按细胞类型聚合参考表达谱。

    对应 R 的 ``SingleR::trainSingleR`` 中参考特征表达谱的构建：
    对每种细胞类型，聚合其内部细胞的表达得到特征向量。

    Parameters:
        adata: 参考 ``AnnData`` 对象。
        label_key: ``adata.obs`` 中存放细胞类型标签的列名。
        method: 聚合方式，``"mean"`` 或 ``"median"``。

    Returns:
        ``(profiles, labels)`` 元组，其中 ``profiles`` 形状为
        ``(n_labels, n_genes)``，``labels`` 为对应的类型标签数组。
    """
    raise NotImplementedError
