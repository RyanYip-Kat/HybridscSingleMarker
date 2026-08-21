"""pysingle — R 包 SingleR 的 Python 移植版。

基于 :class:`anndata.AnnData`（h5ad 文件格式）实现单细胞转录组数据的
细胞类型自动注释，算法语义与 R 包 ``SingleR`` 保持一致。

典型用法（高层入口）::

    import pysingle

    query = pysingle.annotate(ref="ref.h5ad", query="query.h5ad",
                              celltype_col="celltype")
    query.obs["pysingle_celltype"]   # 每个细胞的预测细胞类型
    query.obs["pysingle_score"]      # 对应得分

也可直接调用底层核心算法 ``singleR_annotate`` 获得完整得分矩阵。
"""

from __future__ import annotations

from . import plotting
from .core import select_hvg, singleR_annotate, singleR_annotate_multi
from .io import annotate, load_reference, read_h5ad
from .seurat_method import seurat_annotate

__version__ = "0.1.0"

__all__ = [
    # 高层入口
    "annotate",
    # 核心算法
    "singleR_annotate",
    "singleR_annotate_multi",
    "seurat_annotate",
    "select_hvg",
    # 数据 IO
    "read_h5ad",
    "load_reference",
    # 可视化（SingleR 风格）
    "plotting",
    # 元信息
    "__version__",
]
