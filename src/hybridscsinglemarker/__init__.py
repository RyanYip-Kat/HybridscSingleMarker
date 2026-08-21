"""hybridscsinglemarker — 融合 pysingle 与 cellmarkerannot 的混合细胞类型注释包。

算法架构与数据流见:
  docs/superpowers/specs/2026-08-19-hybrid-annotation-design.md

包结构（集中管理：两套底层包内嵌为子包）:
  hybridscsinglemarker
  ├── cellmarkerannot/          CellMarker 3.0 DB 注释（子包）
  └── pysingle/                 SingleR/Seurat 参考注释（子包）

公共 API:
  hybrid_annotate(query, ref=None, ...)  ->  AnnData（obs 写回 hybrid_celltype /
                                              hybrid_confidence / hybrid_status）
  cellmarkerannot                         底层 DB 注释子包（原 cellmarkerannot）
  pysingle                                底层参考注释子包（原 pysingle）
"""
from __future__ import annotations

from hybridscsinglemarker.__version__ import __version__, __version_info__
from hybridscsinglemarker import cellmarkerannot, pysingle
from hybridscsinglemarker.hybrid import hybrid_annotate

__all__ = [
    "hybrid_annotate",
    "cellmarkerannot",
    "pysingle",
    "__version__",
    "__version_info__",
]
