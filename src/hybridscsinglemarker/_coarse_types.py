"""粗分类族关键词映射：参考标签 / DB cell_name → 粗分类族（T/B/NK/Mono/DC…）。

基线: ``scripts/pipeline.py::_predicted_broad``（本模块将其改写为可配置的
精确匹配 + 关键词子串字典）。支持用户字典覆盖（``hybrid_annotate(coarse_map=...)``）。

约定:
    - 精确匹配（DEFAULT_EXACT_MAP，小写）优先，其次用户 ``coarse_map`` 关键词，
      最后内置关键词（DEFAULT_COARSE_MAP，有序子串匹配）；
    - 未命中 → ``"Other"``（与 pipeline 基线一致）；
    - 输入为 NaN / None（无预测）→ 返回 ``None``（参与状态判定时按"无粗族"处理）。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

# 精确匹配（常见细粒度参考标签 → 粗族）。键为小写。
DEFAULT_EXACT_MAP: dict[str, str] = {
    # pbmc50k / pbmc3k 参考标签
    "memory bc": "B", "naive bc": "B", "asc": "B",
    "memory": "B", "naive": "B",        # 短写：naive/memory B
    "nk1": "NK", "nk2": "NK", "nk3": "NK",
    "cd14": "Mono", "cd16": "Mono",
    "cdc": "DC", "pdc": "DC", "cdc1": "DC", "cdc2": "DC",
    "asdc": "DC",                       # AXL+SIGLEC6+ 树突状细胞
    "abc": "B",                         # antibody-secreting cell（浆细胞）
    "t-mito": "T", "treg": "T", "mait": "T", "gdt": "T",
    "meg": "Other", "rbc": "Other",
    # 常见 DB cell_name
    "t cell": "T", "b cell": "B", "nk cell": "NK",
    "monocyte": "Mono", "dendritic cell": "DC",
    "megakaryocyte": "Other", "erythrocyte": "Other", "erythroid cell": "Other",
    "platelet": "Other",
}

# 关键词子串 → 粗族；有序，先命中先返回（小写子串匹配）。
DEFAULT_COARSE_MAP0: tuple[tuple[str, str], ...] = (
    # T
    ("cd4", "T"), ("cd8", "T"), ("cd3", "T"),
    ("t cell", "T"), ("t-cell", "T"), ("tcell", "T"),
    ("cytotoxic", "T"), ("regulatory t", "T"), ("helper t", "T"),
    # B / plasma
    ("b cell", "B"), ("b-cell", "B"),
    ("b naive", "B"), ("b intermediate", "B"), ("b memory", "B"),
    ("plasma", "B"), ("plasmablast", "B"), ("antibody", "B"),
    ("cd19", "B"), ("cd20", "B"),
    # NK
    ("natural killer", "NK"), ("nk", "NK"), ("cd56", "NK"),
    # Mono / Myeloid
    ("monocyte", "Mono"), ("mono", "Mono"), ("myeloid", "Mono"),
    ("macrophage", "Mono"),
    # DC
    ("dendritic", "DC"), ("cdc", "DC"), ("pdc", "DC"), ("langerhans", "DC"),
    # 其他
    ("erythro", "Other"), ("megakaryo", "Other"), ("platelet", "Other"),
)

# 关键词子串 → 粗族；有序，先命中先返回（小写子串匹配）
# 大类顺序：淋巴系（T/B/NK）→ 专职APC（DC）→ 泛髓系（Mono）→ 其他
DEFAULT_COARSE_MAP: tuple[tuple[str, str], ...] = (
    # ========== T 细胞谱系（淋巴系） ==========
    # 1. 长特异功能/亚群短语
    ("regulatory t cell", "T"), ("helper t cell", "T"), ("cytotoxic t cell", "T"),
    ("memory t cell", "T"), ("naive t cell", "T"), ("effector t cell", "T"),
    ("gamma delta t cell", "T"), ("natural killer t cell", "T"),
    ("t follicular helper", "T"), ("central memory t", "T"), ("effector memory t", "T"),
    # 2. 中短亚群描述
    ("regulatory t", "T"), ("helper t", "T"), ("t helper", "T"), ("cytotoxic t", "T"),
    ("memory t", "T"), ("naive t", "T"), ("effector t", "T"),
    # 3. 亚群缩写
    ("treg", "T"), ("th1", "T"), ("th2", "T"), ("th17", "T"), ("tfh", "T"), ("nkt", "T"),
    # 4. 通用名称变体
    ("t cell", "T"), ("t-cell", "T"), ("tcell", "T"),
    ("t lymphocyte", "T"), ("t lymphoid", "T"),
    # 5. 特异性Marker
    ("cd45ro", "T"), ("cd45ra", "T"),
    ("cd4", "T"), ("cd8", "T"), ("cd3", "T"),

    # ========== B 细胞 / 浆细胞谱系（淋巴系） ==========
    # 1. 长特异亚群短语
    ("germinal center b cell", "B"), ("memory b cell", "B"), ("naive b cell", "B"),
    ("plasma cell", "B"), ("plasmablast cell", "B"),
    # 2. 中短亚群描述
    ("b naive", "B"), ("b intermediate", "B"), ("b memory", "B"),
    ("memory b", "B"), ("naive b", "B"), ("activated b", "B"),
    # 3. 通用名称变体
    ("b cell", "B"), ("b-cell", "B"), ("bcell", "B"),
    ("b lymphocyte", "B"), ("b lymphoid", "B"),
    # 4. 浆细胞与功能分子
    ("plasmablast", "B"), ("plasmocyte", "B"), ("antibody", "B"),
    ("bcr", "B"), ("immunoglobulin", "B"),
    # 5. 特异性Marker
    ("cd19", "B"), ("cd20", "B"), ("cd27", "B"), ("cd38", "B"), ("cd138", "B"), ("cd79a", "B"),
    ("igm", "B"), ("igd", "B"),

    # ========== NK 细胞（淋巴系） ==========
    # 1. 长特异短语
    ("natural killer cell", "NK"), ("nk cell", "NK"), ("nk-cell", "NK"), ("nkcell", "NK"),
    ("nk cytotoxic cell", "NK"),
    # 2. 通用名称
    ("natural killer", "NK"), ("nk", "NK"),
    # 3. 功能/受体相关
    ("nkp", "NK"), ("nkg2d", "NK"),
    # 4. 特异性Marker
    ("cd56", "NK"), ("cd16", "NK"), ("cd57", "NK"),

    # ========== 树突状细胞 DC（专职抗原呈递细胞） ==========
    # 1. 长特异亚群短语
    ("conventional dendritic cell", "DC"), ("plasmacytoid dendritic cell", "DC"),
    ("langerhans cell", "DC"), ("interdigitating dendritic cell", "DC"),
    # 2. 中短亚群描述
    ("conventional dc", "DC"), ("plasmacytoid dc", "DC"),
    ("dendritic cell", "DC"), ("dc cell", "DC"),
    # 3. 亚群缩写
    ("cdc1", "DC"), ("cdc2", "DC"), ("pdc", "DC"), ("cdc", "DC"),
    # 4. 通用名称
    ("dendritic", "DC"), ("langerhans", "DC"),
    # 5. 特异性Marker
    ("cd11c", "DC"), ("cd123", "DC"), ("bdca1", "DC"), ("bdca2", "DC"), ("cd207", "DC"),

    # ========== 单核 / 泛髓系细胞 ==========
    # 1. 长特异亚群短语
    ("classical monocyte", "Mono"), ("non-classical monocyte", "Mono"), ("intermediate monocyte", "Mono"),
    ("m1 macrophage", "Mono"), ("m2 macrophage", "Mono"), ("tissue macrophage", "Mono"),
    ("alveolar macrophage", "Mono"), ("kupffer cell", "Mono"),
    # 2. 通用细胞名称
    ("monocyte", "Mono"), ("mono", "Mono"),
    ("macrophage", "Mono"), ("mononuclear phagocyte", "Mono"),
    # 3. 其他髓系细胞（粒细胞、肥大细胞等）
    ("neutrophil", "Mono"), ("granulocyte", "Mono"), ("eosinophil", "Mono"), ("basophil", "Mono"),
    ("mast cell", "Mono"), ("myeloid cell", "Mono"),
    # 4. 泛髓系名称
    ("myeloid", "Mono"),
    # 5. 特异性Marker
    ("cd14", "Mono"), ("cd68", "Mono"), ("cd33", "Mono"), ("cd11b", "Mono"),

    # ========== 其他细胞（红系、巨核、干祖、非免疫细胞） ==========
    # 1. 长特异短语
    ("hematopoietic stem cell", "Other"), ("hematopoietic progenitor", "Other"),
    ("red blood cell", "Other"), ("erythrocyte", "Other"),
    ("megakaryocyte", "Other"), ("thrombocyte", "Other"),
    ("stromal cell", "Other"), ("fibroblast", "Other"),
    ("epithelial cell", "Other"), ("endothelial cell", "Other"),
    # 2. 中短名称
    ("erythroid", "Other"), ("megakaryo", "Other"), ("platelet", "Other"),
    ("stem cell", "Other"), ("progenitor cell", "Other"),
    # 3. 缩写与短子串
    ("erythro", "Other"), ("rbc", "Other"), ("hsc", "Other"),
    ("cd41", "Other"), ("cd61", "Other"), ("hemoglobin", "Other"),
)


def _is_missing(label: Any) -> bool:
    if label is None:
        return True
    if isinstance(label, float) and math.isnan(label):
        return True
    try:
        return label is pd.NA or (label is not label)  # NaN 自不等
    except Exception:
        return False


def coarse_type(
    label: Any,
    coarse_map: dict[str, str] | None = None,
    fallback: Any = None,
) -> str | None:
    """把单个标签映射到粗分类族。

    label: 参考标签或 DB cell_name；None / NaN → None（无粗族）。
    coarse_map: 用户关键词覆盖字典（{关键词: 粗族}，先于内置匹配）。
    fallback: 关键词未命中时回退尝试的次选标签（如 DB cell_name_class 字段），
        递归用同一规则映射；仍未命中 → "Other"。
    """
    if _is_missing(label):
        return None
    s = str(label).lower()

    if s in DEFAULT_EXACT_MAP:
        return DEFAULT_EXACT_MAP[s]

    if coarse_map:
        for kw, fam in coarse_map.items():
            if kw.lower() in s:
                return fam

    for kw, fam in DEFAULT_COARSE_MAP:
        if kw in s:
            return fam

    if fallback is not None and not _is_missing(fallback):
        fb = coarse_type(fallback, coarse_map=coarse_map)
        if fb is not None:
            return fb
    return "Other"


def coarse_type_series(
    labels,
    coarse_map: dict[str, str] | None = None,
    fallback_series=None,
) -> np.ndarray:
    """批量映射（先对唯一标签去重匹配，再 pandas ``map`` 回填，避免逐细胞循环）。

    返回 object 数组，长度与输入一致；无预测的标签为 ``None``。
    """
    s = pd.Series(labels)
    uniq = s.dropna().unique()
    cache: dict = {}
    if fallback_series is not None:
        fb = pd.Series(fallback_series).reindex(uniq) if fallback_series is not None else None
    else:
        fb = None
    for u in uniq:
        f = fb.get(u) if fb is not None else None
        cache[u] = coarse_type(u, coarse_map, fallback=f)
    mapped = s.map(cache)                       # 缺失标签 → NaN（map 无 default）
    out = mapped.to_numpy(dtype=object)
    out[pd.isna(mapped)] = None                 # NaN / 未映射 → None
    return out
