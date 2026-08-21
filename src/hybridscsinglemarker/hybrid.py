"""混合注释主入口：融合 pysingle 与 cellmarkerannot。

算法架构与数据流见
``docs/superpowers/specs/2026-08-19-hybrid-annotation-design.md``（§3.1 有参考链路、
§3.2 无参考降级链路）。本模块只做编排，全部数值核（融合 / 先验 / 验证 / 分层）
位于 ``_fusion.py`` / ``_coarse_types.py`` / ``_validate.py`` / ``_layer.py``。

有参考链路（ref 非空）::

    S0     ← pysingle 首轮得分（singleR_annotate / seurat_annotate）
    P[cell,c] ← 每细胞 DB 证据先验（c 的特征基因中 DB marker 且该细胞表达的
             比例，逐行 max 归一化；:func:`per_cell_db_prior`）
    S_corr ← S0 × (1 + λ·P)               （逐细胞乘法融合，作用于首轮得分）
    c*     ← pysingle fine-tuned 标签（在融合后的首轮得分上精修；多参考为
             合并得分 argmax，与 pysingle 多参考语义一致）
    V      ← c* 特征基因在该细胞的 DB 支持比例（逐细胞验证）
    conf   ← clip(S_corr[c*], 0, 1) × (0.5 + 0.5·V)
    status ← 粗分类一致性分层（c* 粗族 vs DB 预测粗族）

无参考链路（ref is None）: 自动降级 ``cellmarkerannot.annotate_cells``，
``hybrid_status = "db_only"``。

输出约定: ``obs`` 新增 ``hybrid_celltype`` / ``hybrid_confidence`` /
``hybrid_status``；完整得分矩阵与中间量写入 ``uns["hybridsc"]``。
"""
from __future__ import annotations

import os
from typing import Any, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse as sp

from hybridscsinglemarker.__version__ import __version__
from hybridscsinglemarker._coarse_types import coarse_type_series
from hybridscsinglemarker._fusion import (
    _common_genes,
    _label_feature_genes,
    fuse_family_posterior,
    fuse_scores_per_cell,
    label_evidence_matrices,
    per_cell_db_prior,
    softmax_rows,
)
from hybridscsinglemarker._layer import (
    DB_ONLY,
    classify_status_bulk,
)
from hybridscsinglemarker._validate import validate_cells
from hybridscsinglemarker.cellmarkerannot import CellMarkerDB, annotate_cells
from hybridscsinglemarker.cellmarkerannot._config import MARKER_SOURCES
from hybridscsinglemarker.cellmarkerannot._scoring import _scope_rows
from hybridscsinglemarker.cellmarkerannot.annotation import _resolve_data_sources
from hybridscsinglemarker.pysingle import seurat_annotate, singleR_annotate
from hybridscsinglemarker.pysingle.core import _combine_scores


def _load_ann(obj: Any, what: str) -> AnnData:
    """把 h5ad 路径或 AnnData 统一为 AnnData。"""
    if isinstance(obj, (str, os.PathLike)):
        return ad.read_h5ad(str(obj))
    if isinstance(obj, AnnData):
        return obj
    raise TypeError(
        f"{what} 须为 h5ad 路径或 anndata.AnnData，实际为 {type(obj).__name__}"
    )


def _pysingle_snapshot(out: dict) -> dict:
    """把 pysingle 结果整理成 h5ad 可序列化的精简快照。

    ``out`` 为单个参考或合并后的结果字典；仅保留标签与得分标量数组，
    丢弃嵌套的 DataFrame / 每参考明细（已由 ``S_corr`` / ``F`` 承载）。
    """
    snap: dict = {}
    if "labels" in out:
        snap["labels"] = np.asarray(out["labels"])
    for key in ("pval", "prediction_score"):
        if key in out and out[key] is not None:
            snap[key] = np.asarray(out[key])
    return snap


def _label_family_map(
    labels: Sequence[str],
    coarse_map: dict[str, str] | None = None,
) -> dict[str, str | None]:
    """把标签列表映射为 ``{标签: 粗分类族}`` 字典（供族级融合/状态判定）。

    通过 :func:`coarse_type_series` 批量去重匹配（关键词字典），未命中的标签
    映射为 ``None``（``fuse_family_posterior`` 中 ``None`` 族不参与族级再排序，
    等价于无 DB 族证据）。
    """
    fams = coarse_type_series(list(labels), coarse_map=coarse_map)
    return {
        str(lab): (fam if fam is not None else None)
        for lab, fam in zip(labels, fams)
    }


def _layer_view(query: AnnData, layer: str | None) -> AnnData | Any:
    """查询层表达式视图：``layer=None`` 用 X（零拷贝）；否则把 layer 提升为 X。

    pysingle 只读 ``adata.X``（不感知 layers）；DB 打分走 ``_resolve_x``（默认
    也读 X）。两者需在同一份表达数据上打分，故显式给定 ``layer`` 时构造一个
    轻量视图给 pysingle（稀疏矩阵引用共享，不深拷贝数据）。
    """
    if layer is None:
        return query
    if layer not in query.layers:
        raise KeyError(f"query 没有 layer {layer!r}")
    return AnnData(
        X=query.layers[layer],
        obs=query.obs.copy(deep=False),
        var=query.var.copy(deep=False),
    )


def hybrid_annotate(
    query,
    ref=None,
    celltype_col: str | Sequence[str] = "celltype",
    species: str = "Human",
    tissue: str = "Blood",
    method: str = "singler",
    lambda_: float = 0.3,
    coarse_map: dict[str, str] | None = None,
    confidence_threshold: float = 0.3,
    data_source: str | Sequence[str] = "experiment",
    marker_sources: Sequence[str] | None = None,
    db_dataset: str = "all_cell_marker",
    db_method: str = "weighted",
    db: CellMarkerDB | None = None,
    layer: str | None = None,
    expr_threshold: float = 1.0,
    feature_gene_top_n: int = 200,
    feature_genes: dict[str, Sequence[str]] | None = None,
    prior_breadth_threshold: float | None = None,
    lambda_margin_gate: float | None = 2.0,
    family_boost_only: bool = True,
    center_prior: bool = False,
    n_jobs: int = 1,
    scoring: str = "cells",
    top_n: int = 5,
    fine_tune: bool = True,
    gene_selection: str = "hvg",
    max_genes: int = 5000,
    combine_method: str = "max",
    max_cells_per_type: int | None = None,
    ref_cap_seed: int = 0,
    inplace: bool = False,
    reuse_singler: list[dict] | None = None,
    family_posterior: bool = False,
    include_label_evidence: bool = False,
    **kwargs,
) -> AnnData:
    """混合细胞类型注释主入口（pysingle 参考相关 + cellmarkerannot DB 证据融合）。

    Parameters:
        query: 查询数据，h5ad 路径或 ``AnnData``（X 为 细胞×基因）。
        ref: 参考数据。可为一个或多个 ``AnnData`` / h5ad 路径
            （``None`` = 无参考，自动降级为纯 DB 注释）。也接受
            ``(ref, labels)`` 元组列表以直接给定标签（跳过 ``celltype_col``）。
        celltype_col: 参考的标签列名。字符串对所有参考生效；多参考时可给列表
            按位置一一映射（如 ``["celltype", "cell_type"]``，ref1 用
            ``celltype``、ref2 用 ``cell_type``）。
        species / tissue: CellMarker DB 打分范围（传给 cellmarkerannot）。
        method: 参考相关方法，``"singler"``（SingleR 移植）或 ``"seurat"``
            （Seurat 标签转移）。
        lambda_: 融合强度。先验为**逐细胞** DB 证据（:func:`per_cell_db_prior`，
            每细胞自身表达），融合作用于 pysingle **首轮**得分、fine-tuning
            在其上精修；默认 0.3。融合需与默认双门控（``lambda_margin_gate=2.0``
            + ``family_boost_only=True``）配套使用，单独固定 λ 全标签施加在
            大规模上净为负（2026-08-20 实证：1ref-50000 Δacc −5.8%，门控后
            ≤10k 与多参考 +0.8~1.2% rel、50k 残余 −0.8%）。``0`` 退化为纯
            pysingle 相关（内置 sanity check，逐位一致）。
        coarse_map: 用户粗分类关键词字典覆盖（见 :mod:`_coarse_types`）。
        confidence_threshold: 分层阈值（``consistent`` vs ``low_confidence``）。
        data_source: DB 数据来源范围，``"all"`` / ``"experiment"`` /
            ``"single_cell"`` / ``"method"`` / ``"review"`` / ``"company"``
            或其组合（与 cellmarkerannot 一致）。
        marker_sources: DB ``marker_source`` 精确值集合；默认由 ``data_source``
            解析，保证 DB 预测与先验使用同一范围。
        db_dataset: 默认 DB 数据集名（当 ``db=None``）。
        db_method: DB 打分方法（``"weighted"`` / ``"overlap"`` / ``"ssgsea"``）。
        db: 显式 ``CellMarkerDB``；默认加载 ``db_dataset``。
        layer: 查询表达层（默认 X）。DB 打分与 pysingle 均在其上运行。
        expr_threshold: 逐细胞验证的表达阈值（与 cellmarkerannot 一致，默认 1.0）。
        feature_gene_top_n: 自动特征基因数（DE 风格 top-N），默认 200。
        feature_genes: 用户自定义 ``{标签: 基因列表}``；给定后跳过自动计算。
        prior_breadth_threshold: 不为 ``None`` 时，计算逐细胞先验前剔除在查询中
            表达广度（>expr_threshold 的细胞比例）超过该值的 DB 重叠基因。广表达
            基因（管家/泛白细胞 marker）会让逐细胞先验退化为近似静态先验（对每个
            细胞抬高同一批标签）；剔除后先验才保留细胞特异性。默认 ``None``（不
            过滤，保持原行为）。
        lambda_margin_gate: 不为 ``None`` 时，按 pysingle 首轮得分的 winner margin
            （top1−top2）逐细胞门控融合强度：``λ_eff = λ × clip(1 − margin /
            (gate·median_margin), 0, 1)``，即 margin 达中位数的 ``gate`` 倍时 λ 衰减
            到 0，margin 最小的细胞保留全强度。高置信细胞不被融合触碰（实证：消除
            大规模下"改错已正确细胞"的退化）。默认 ``2.0``。
        family_boost_only: ``True``（默认）时，先验先按粗族聚合（族内取最大值），
            同族细标签共用同一乘数——族内乘数一致 ⇒ 族内细标签排序与 pysingle
            完全一致，DB 只改变族间平衡、不改族内细标签（实证：细粒度 ARI/acc
            缺口来自族内乱翻；全标签施加时 T 族内 P 差异 0.91 vs 0.7 会把
            细标签搅动 1/3）。设为 ``False`` 恢复逐标签施加。
        center_prior: ``True`` 时把逐细胞先验中心化（``P − rowmean P``），使先验既
            能抬升也能压低标签（默认 ``False``：boost-only，只升不降）。
        n_jobs: pysingle 并行进程数。
        scoring / top_n / fine_tune / gene_selection / max_genes: 透传给
            ``singleR_annotate``（``method="singler"``）。
        combine_method: 多参考合并方式，``"max"``（默认）或 ``"mean"``。
        max_cells_per_type: 可选，每个参考类型最多保留的细胞数（None=全量）。
            大参考提速（逐类型确定性封顶，见 pysingle.singleR_annotate），
            cells 模式语义保留；多参考时每个参考独立封顶。
        ref_cap_seed: 封顶抽样种子，默认 0。仅 ``method="singler"`` 生效；
            ``method="seurat"`` 路径忽略。
        inplace: ``False``（默认）在 query 副本上写回并返回新对象；``True``
            直接修改传入的 ``AnnData``。
        reuse_singler: 可选，``method="singler"`` 且 ``lambda_=0`` 时传入
            已算好的 ``singleR_annotate`` 结果列表（每参考一个，顺序与
            ``ref`` 一致），跳过重复的 pysingle 计算。λ=0 时先验变换为恒等，
            复用结果与重新计算逐位等价（骨架设计 §5.3 运行时闸门：全量档
            单参考 λ=0 时 hybrid 场景直接复用其 pysingle 孪生）。λ>0 时
            传入会报 ``ValueError``（融合必须在 pysingle 内部作用首轮得分）。
        family_posterior: ``True`` 时启用 **Variant B 族级后验再排序**（设计
            §6.3）：DB 证据只做粗族决策（``chosen_family``），族内细标签永远
            由 pysingle 决定（fine-tuned 标签若在该族内则原样保留，否则取该族
            内 pysingle 得分最高者）。``F[cell,c] = S_final × (1 + λ_eff·P_fam)``，
            族内同乘数 ⇒ 族内顺序不变；margin 门控与 λ 语义与 v_full 一致，
            λ=0 时逐位等价纯 pysingle。默认 ``False``（v_full 乘法融合）。
        include_label_evidence: ``True`` 时在 ``uns["hybridsc"]["label_evidence"]``
            写入每参考标签一张 ``score_gene_list`` 风格的注释矩阵（特征基因 ×
            DB 细胞类型的支持证据 + Score 行，批量向量化构建），供逐标签
            可视化（管线保存 CSV 与热图）。默认 ``False``。
        **kwargs: 其余参数透传给 pysingle 入口（``singleR_annotate`` 的
            ``fine_tune_thres`` / ``min_genes`` / ``n_de_scale`` / ``chunk_size``，
            ``seurat_annotate`` 的 ``reduction`` / ``n_dims`` / ``k_anchor`` 等）。

    Returns:
        写回后的 ``AnnData``：
        - ``obs["hybrid_celltype"]``  每细胞最终预测标签（c*）；
        - ``obs["hybrid_confidence"]`` 融合置信度 ∈ [0,1]；
        - ``obs["hybrid_status"]``    ``consistent`` / ``low_confidence`` /
          ``unknown`` / ``db_only``；
        - ``uns["hybridsc"]``         完整得分矩阵（``S_corr`` / ``F`` /
          ``softmax_prob`` / ``P_prior`` / ``V`` / DB 预测与粗族中间量）。

    Raises:
        ValueError: 范围为空、参考缺标签列、celltype_col 长度不匹配等。
        TypeError: 输入类型不合法。
    """
    # ------------------------------------------------------------------ #
    # 0. 输入归一化                                                       #
    # ------------------------------------------------------------------ #
    result = _load_ann(query, "query")
    if not inplace:
        result = result.copy()

    db = db if db is not None else CellMarkerDB(dataset=db_dataset)

    if ref is None:
        return _annotate_db_only(
            result, db,
            species=species, tissue=tissue, method=db_method, layer=layer,
            marker_sources=_resolve_marker_sources(data_source, marker_sources),
            expr_threshold=expr_threshold,
        )

    if reuse_singler is not None and lambda_ != 0.0:
        raise ValueError(
            "reuse_singler 仅允许 lambda_=0（λ>0 时先验必须在 pysingle 内作用）"
        )

    refs, labels_list = _normalize_refs(ref, celltype_col)
    if not refs:
        raise ValueError("ref 为空")

    # 查询层视图：让 pysingle 与 DB 打分在同一份表达数据上
    query_py = _layer_view(result, layer)

    # ------------------------------------------------------------------ #
    # 1. DB 范围：一次解析，DB 预测 / 逐细胞先验 / 验证共用同一范围           #
    # ------------------------------------------------------------------ #
    ms = _resolve_marker_sources(data_source, marker_sources)
    in_scope = _scope_rows(db, species, tissue, ms)
    db_range_genes = set(in_scope["marker"])

    # ------------------------------------------------------------------ #
    # 1b. 每参考标签特征基因（先于 pysingle：逐细胞先验依赖）                #
    # ------------------------------------------------------------------ #
    if feature_genes is not None:
        genes_by_label = {str(k): list(v) for k, v in dict(feature_genes).items()}
    else:
        ref_data = list(zip(refs, labels_list))
        common = _common_genes(ref_data, result)
        genes_by_label = _label_feature_genes(
            ref_data, common, top_n=feature_gene_top_n,
        )

    # 可选：剔除广表达 DB 重叠基因（否则逐细胞先验退化为静态先验）
    if prior_breadth_threshold is not None and genes_by_label:
        genes_by_label = _filter_broad_genes(
            genes_by_label, query_py, db_range_genes,
            threshold=prior_breadth_threshold,
        )

    # 1c. 逐细胞 DB 证据先验 P[cell,c]（每细胞自身表达，方向性正确）         #
    P_cell = per_cell_db_prior(
        query_py, genes_by_label, db_range_genes, expr_threshold=expr_threshold,
    )

    # ------------------------------------------------------------------ #
    # 1. pysingle: S_corr[细胞 × 参考标签]（多参考按 combine_method 合并）   #
    #    逐细胞先验通过 score_transform 作用在**首轮**得分上，fine-tuning    #
    #    在其上精修（λ=0 → 变换为恒等，逐位等价纯 pysingle）。              #
    # ------------------------------------------------------------------ #
    if method == "singler":
        # Variant B（family_posterior）：DB 在首轮得分上做族级决策并 mask 掉
        # 非选定族的标签，fine-tuning 只在族内精修；否则用 v_full 逐标签乘法。
        # 每个参考的标签集不同 ⇒ transform 必须逐参考构建。
        fp_state: dict | None = None
        family_trs: list[tuple[Any, dict]] | None = None
        if family_posterior:
            fam_map = _label_family_map(list(P_cell.columns), coarse_map)
            family_trs = [
                _make_family_transform(
                    P_cell, fam_map, lab, lambda_,
                    margin_gate=lambda_margin_gate,
                )
                for lab in labels_list
            ]
            if len(refs) == 1:
                fp_state = family_trs[0][1]   # 单参考：捕获实际运行的 transform 状态
        if reuse_singler is not None:
            if len(reuse_singler) != len(refs):
                raise ValueError(
                    f"reuse_singler 长度 {len(reuse_singler)} 与 ref 数 "
                    f"{len(refs)} 不一致"
                )
            per = reuse_singler
        else:
            per = [
                singleR_annotate(
                    r, lab, query_py,
                    fine_tune=fine_tune, top_n=top_n,
                    gene_selection=gene_selection,
                    max_genes=max_genes, scoring=scoring, n_jobs=n_jobs,
                    max_cells_per_type=max_cells_per_type,
                    ref_cap_seed=ref_cap_seed,
                    **kwargs,
                    score_transform=(
                        family_trs[i][0]
                        if family_posterior
                        else _make_score_transform(
                            P_cell, lab, lambda_,
                            margin_gate=lambda_margin_gate,
                            center_prior=center_prior,
                            family_boost_only=family_boost_only,
                        )
                    ),
                )
                for i, (r, lab) in enumerate(zip(refs, labels_list))
            ]
    elif method == "seurat":
        per = [
            seurat_annotate(
                r, lab, query_py, max_genes=max_genes, n_jobs=n_jobs, **kwargs,
            )
            for r, lab in zip(refs, labels_list)
        ]
    else:
        raise ValueError(f"method 须为 'singler' / 'seurat'，实际为 {method!r}")

    if len(per) == 1:
        S_corr = per[0]["scores"]
        pysingle_out = per[0]
    else:
        S_corr = _combine_scores([p["scores"] for p in per], combine_method)
        pysingle_out = {"per_reference": per, "scores": S_corr,
                        "labels": S_corr.idxmax(axis=1)}

    # seurat 入口无 score_transform 钩子：融合后置于合并得分矩阵上
    if method == "seurat" and lambda_ != 0.0 and len(P_cell.columns):
        S_corr = fuse_scores_per_cell(S_corr, P_cell, lambda_)

    if len(S_corr) != result.n_obs:
        raise ValueError("pysingle 得分行数与查询细胞数不一致（可能有重复 obs_names）")

    # ------------------------------------------------------------------ #
    # 2. DB 预测（逐细胞原始预测，max_cell_types=0 关闭 "Other" 合并）        #
    # ------------------------------------------------------------------ #
    db_ann = annotate_cells(
        result, db,
        method=db_method, species=species, tissue=tissue, layer=layer,
        marker_sources=ms, expr_threshold=expr_threshold,
        max_cell_types=0, min_cells=0, inplace=False,
    )
    db_celltype = np.asarray(db_ann["celltype_predicted"], dtype=object)
    db_confidence = db_ann["confidence"].to_numpy(dtype=float)

    # ------------------------------------------------------------------ #
    # 3. 融合矩阵（先验已作用在 pysingle 内部）+ 置信度 + 验证 + 分层        #
    # ------------------------------------------------------------------ #
    labels = np.asarray(list(S_corr.columns), dtype=object)

    # pysingle 标签：单参考 = fine-tuned 标签（融合影响其首轮得分与候选集），
    # 多参考 = 合并得分 argmax（与 pysingle 多参考语义一致）。
    if len(per) == 1:
        labels_py = np.asarray(pysingle_out["labels"], dtype=object)
    else:
        labels_py = labels[np.asarray(S_corr.to_numpy(dtype=float)).argmax(axis=1)]

    # ---- 融合矩阵（v_full 与 Variant B 均已在 pysingle 内部作用首轮得分）----
    F = S_corr
    cstar = labels_py
    pos = {lab: i for i, lab in enumerate(labels)}
    cstar_idx = pd.Series(cstar).map(pos).to_numpy(dtype=float)
    fallback = np.asarray(F.to_numpy(dtype=float)).argmax(axis=1)
    cstar_idx = np.where(
        np.isnan(cstar_idx), fallback, cstar_idx.astype(np.int64),
    )
    # 诊断量：Variant B 单参考时取 transform 状态；否则由 cstar 粗族回填
    if family_posterior and len(per) == 1 and fp_state is not None:
        chosen_family = fp_state.get("chosen_family")
        lam_eff = fp_state.get("lambda_eff")
    else:
        chosen_family = None
        lam_eff = None

    prob = softmax_rows(F)                                  # 每行 softmax 概率（仅存 uns 供参考）
    # 置信度：winner 原始融合得分（与 pysingle 多参考 winner score 同量纲，
    # 截断负数），乘 DB 验证折扣。不用 softmax——多标签下 softmax 概率被分母
    # 摊薄到 ≈1/N（25-46 标签时 <0.05），使 consistent 层结构性不可达。
    score_winner = np.asarray(F.to_numpy(dtype=float))[
        np.arange(F.shape[0]), cstar_idx
    ]
    score_winner = np.clip(score_winner, 0.0, 1.0)   # 融合分可 >1（λ>0 乘数）

    V = validate_cells(
        result, genes_by_label, cstar, db_range_genes,
        layer=layer, threshold=expr_threshold,
    )
    conf = score_winner * (0.5 + 0.5 * V)

    # 逐标签 score_gene_list 风格注释矩阵（可选；管线据此保存 CSV + 热图）
    label_evidence = None
    if include_label_evidence and genes_by_label:
        label_evidence = label_evidence_matrices(
            genes_by_label, db, species=species, tissue=tissue,
            marker_sources=ms, in_scope=in_scope,
        )

    coarse_ref = coarse_type_series(cstar, coarse_map=coarse_map)
    coarse_db = coarse_type_series(db_celltype, coarse_map=coarse_map)
    status = classify_status_bulk(
        coarse_ref, coarse_db, conf, threshold=confidence_threshold,
    )

    # ------------------------------------------------------------------ #
    # 4. 写回 obs + uns                                                    #
    # ------------------------------------------------------------------ #
    result.obs["hybrid_celltype"] = cstar
    result.obs["hybrid_confidence"] = conf
    result.obs["hybrid_status"] = status
    result.uns["hybridsc"] = {
        "version": __version__,
        "params": {
            "method": method, "lambda": lambda_,
            "species": species, "tissue": tissue,
            "data_source": data_source, "marker_sources": sorted(ms),
            "db_method": db_method, "confidence_threshold": confidence_threshold,
            "feature_gene_top_n": feature_gene_top_n,
            "max_cells_per_type": max_cells_per_type,
            "ref_cap_seed": ref_cap_seed,
        },
        "S_corr": S_corr,
        "F": F,
        "softmax_prob": prob,
        # 逐标签先验摘要（max over cells；保留 dict 约定，供可视化/向后兼容），
        # 完整逐细胞矩阵见 P_cell。
        "P_prior": {lab: float(P_cell[lab].max()) for lab in P_cell.columns},
        "P_cell": P_cell,
        "V": V,
        "db_celltype": db_celltype,
        "db_confidence": db_confidence,
        "coarse_ref": coarse_ref,
        "coarse_db": coarse_db,
        "family_posterior": family_posterior,
        "chosen_family": chosen_family,
        "lambda_eff_cell": lam_eff,
        "genes_by_label": genes_by_label,
        "label_evidence": label_evidence,
        "pysingle": _pysingle_snapshot(pysingle_out),
    }
    return result


def _normalize_refs(ref, celltype_col) -> tuple[list[AnnData], list[pd.Series]]:
    """归一化参考输入 → ``(refs, labels_list)``，标签与各参考细胞一一对齐。

    ``ref`` 支持：单个 AnnData/路径、其序列，或 ``(obj, labels)`` 元组序列
    （直接给定标签，跳过 ``celltype_col``）。
    """
    if isinstance(ref, (str, os.PathLike, AnnData)):
        items: list = [ref]
    else:
        items = list(ref)
    if not items:
        raise ValueError("ref 为空")

    refs: list[AnnData] = []
    overrides: list[pd.Series | None] = []
    for it in items:
        if isinstance(it, tuple) and len(it) == 2:
            obj, labels = it
            refs.append(_load_ann(obj, "ref"))
            overrides.append(labels)
        else:
            refs.append(_load_ann(it, "ref"))
            overrides.append(None)

    if isinstance(celltype_col, str):
        cols = [celltype_col] * len(refs)
    else:
        cols = list(celltype_col)
        if len(cols) != len(refs):
            raise ValueError(
                f"celltype_col 列表长度 {len(cols)} 与参考数量 {len(refs)} 不匹配"
            )

    labels_list: list[pd.Series] = []
    for r, ov, c in zip(refs, overrides, cols):
        if ov is not None:
            labels_list.append(pd.Series(ov, index=r.obs_names))
            continue
        if c not in r.obs.columns:
            raise ValueError(f"参考缺少标签列 {c!r}")
        labels_list.append(r.obs[c])
    return refs, labels_list


def _resolve_marker_sources(
    data_source: str | Sequence[str],
    marker_sources: Sequence[str] | None,
) -> frozenset[str]:
    """解析 DB ``marker_source`` 集合（DB 预测 / 先验 / 验证统一范围）。

    ``data_source="all"`` 解析为 ``None``（= 所有来源）；此处显式展开为
    ``MARKER_SOURCES`` 全集，避免 ``annotate_cells(marker_sources=None)``
    的"默认受限来源"语义与先验的"全来源"语义不一致。
    """
    if marker_sources is not None:
        return frozenset(marker_sources)
    resolved = _resolve_data_sources(data_source)
    return frozenset(MARKER_SOURCES) if resolved is None else resolved


def _make_score_transform(
    P_cell: pd.DataFrame,
    ref_labels,
    lambda_: float,
    margin_gate: float | None = None,
    center_prior: bool = False,
    family_boost_only: bool = True,
) -> Any | None:
    """构造作用于 pysingle **首轮**得分矩阵的逐细胞融合变换。

    返回 ``S -> S × mult``；``mult = 1 + λ_eff·Psub``，``Psub`` 为该参考标签列
    （按 pysingle 内部类型字母序对齐）的逐细胞先验。λ=0 / 无先验匹配时返回
    ``None``（= 恒等变换，逐位等价纯 pysingle）。

    ``center_prior=True`` 时 ``Psub`` 先中心化（``P − rowmean P``），先验可抬可压；
    ``margin_gate`` 非空时按首轮 winner margin 逐细胞缩放 λ（详见
    :func:`hybrid_annotate` 参数说明）。

    ``family_boost_only=True`` 时 ``Psub`` 先按粗族聚合（同族标签列共用族内
    最大值），同族乘数一致 ⇒ 族内细标签排序逐位等价 pysingle 首轮，DB 只改变
    族间平衡——DB marker 是粗族级证据，在细粒度上施力只会族内乱翻。
    """
    if lambda_ == 0.0 or P_cell is None or P_cell.shape[0] == 0:
        return None
    cols = sorted(set(pd.Series(ref_labels).astype(str)))
    if not cols or not set(cols).issubset(P_cell.columns):
        return None
    Psub = P_cell[cols].to_numpy(dtype=np.float64)
    if center_prior:
        Psub = Psub - Psub.mean(axis=1, keepdims=True)

    if family_boost_only:
        # 族级先验：同族细标签共用**同一**乘数（族内取 P 最大值），族内乘数一致 ⇒
        # 族内细标签排序与 pysingle 首轮完全一致，DB 只改变族间平衡、不改族内
        # 细标签（实证：把同族乘数置 1 的遮罩版会失去对正确 argmax 的保护，翻转
        # 精度跌到 0.18，远坏于现状；族级取 max 同时恢复保护并保留 T↔NK 修复）。
        fam = np.asarray(
            [f if f is not None else "Other" for f in coarse_type_series(cols)],
            dtype=object,
        )
        _, fam_col = np.unique(fam, return_inverse=True)
        n_fam = int(fam_col.max()) + 1
        Pfam = np.empty((Psub.shape[0], n_fam), dtype=np.float64)
        for i in range(n_fam):
            Pfam[:, i] = Psub[:, fam_col == i].max(axis=1)
        Psub = Pfam[:, fam_col]

    if margin_gate is not None:
        def _tr(S):
            s = np.asarray(S, dtype=np.float64)
            # 单类型（或零类型）时没有 top2，margin 定义为 0（无门控衰减）
            if s.shape[1] >= 2:
                margin = s.max(axis=1) - np.partition(s, -2, axis=1)[:, -2]
            else:
                margin = np.zeros(s.shape[0])
            med = float(np.median(margin))
            if med > 0:
                gate = np.clip(1.0 - margin / (margin_gate * med), 0.0, 1.0)
            else:
                gate = np.ones(s.shape[0])
            return s * (1.0 + (lambda_ * gate)[:, None] * Psub)
        return _tr

    def _tr(S):
        s = np.asarray(S, dtype=np.float64)
        return s * (1.0 + lambda_ * Psub)

    return _tr


def _make_family_transform(
    P_cell: pd.DataFrame,
    fam_map: dict[str, str | None],
    ref_labels,
    lambda_: float,
    margin_gate: float | None = 2.0,
) -> tuple[Any, dict]:
    """Variant B 族级后验 transform：作用于 pysingle **首轮**得分。

    与 :func:`_make_score_transform`（v_full，逐标签乘法融合）的区别：DB 先验
    只做**粗族决策**，族内细标签永远由 pysingle 决定。算法（全向量化）:

        1. 族级聚合:  S_fam[cell,fam] = max_{c∈fam} S0[cell,c]
                      P_fam[cell,fam] = max_{c∈fam} P_cell[cell,c]
        2. 族级 margin 门控: λ_eff = λ × clip(1 − margin/(gate·median), 0, 1)，
           其中 margin = 族级得分 top1−top2（高置信细胞 λ_eff=0 ⇒ 不改判）。
        3. G[cell,fam] = S_fam × (1 + λ_eff·P_fam) → chosen_family = argmax G
        4. 施加: F[cell,c] = S0[cell,c] × (1 + λ_eff·P_fam[family(c)])；
           λ_eff>0 的细胞，非 chosen_family 的标签置 -1e9（mask）——后续
           fine-tuning 的候选集只在 chosen_family 内 ⇒ DB 只改粗族，族内
           细标签与 fine-tuning 精修完全归 pysingle。

    λ=0 或 margin 门控把 λ_eff 压到 0 的细胞 → F=S0（mask 不生效），逐位
    等价纯 pysingle。``state`` 记录每细胞的 chosen_family 与 λ_eff（诊断用）。
    """
    state: dict = {"chosen_family": None, "lambda_eff": None}
    # pysingle 内部类型列按字母序排序（与 singleR_annotate 的 types 一致），
    # 故 transform 收到的 S0 列顺序固定为 sorted(ref_labels)
    cols = sorted(set(pd.Series(ref_labels).astype(str)))
    if not cols:
        return lambda S0: np.asarray(S0, dtype=np.float64), state
    P = P_cell.reindex(columns=cols).fillna(0.0).to_numpy(dtype=np.float64)
    fam_members: dict[str, list[int]] = {}
    for i, lab in enumerate(cols):
        f = fam_map.get(lab)
        if f is not None:
            fam_members.setdefault(f, []).append(i)
    fams = [f for f in fam_members if fam_members[f]]
    fam_idx = {f: np.asarray(fam_members[f], dtype=np.int64) for f in fams}

    def _tr(S0):
        s = np.asarray(S0, dtype=np.float64)
        n_cells = s.shape[0]
        if not fams:
            return s
        S_fam = np.column_stack([s[:, c].max(axis=1) for c in fam_idx.values()])
        P_fam = np.column_stack([P[:, c].max(axis=1) for c in fam_idx.values()])

        # 族级 margin 门控（自校准；无族或无 top2 时不衰减）
        if margin_gate is not None and S_fam.shape[1] >= 2:
            m = np.sort(S_fam, axis=1)
            margin = m[:, -1] - m[:, -2]
        else:
            margin = np.zeros(n_cells)
        med = float(np.median(margin))
        if margin_gate is not None and med > 0:
            lam_eff = lambda_ * np.clip(
                1.0 - margin / (margin_gate * med), 0.0, 1.0
            )
        else:
            lam_eff = np.full(n_cells, lambda_)

        G = S_fam * (1.0 + lam_eff[:, None] * P_fam)
        chosen = G.argmax(axis=1)
        chosen_family = np.asarray(fams, dtype=object)[chosen]
        state["chosen_family"] = chosen_family
        state["lambda_eff"] = lam_eff

        # 施加：族内同乘数（相对顺序不变）；非 chosen_family 的标签 mask 掉，
        # 使 fine-tuning 只在 DB 选定的族内精修（-1e9 而非 -inf，避免 inf 运算）
        F = s.copy()
        mult = np.ones((n_cells, s.shape[1]))
        for j, f in enumerate(fams):
            cols = fam_idx[f]
            mult[:, cols] = (1.0 + lam_eff * P_fam[:, j])[:, None]
        F = s * mult
        active = lam_eff > 0
        for j, f in enumerate(fams):
            mask = active & (chosen_family != f)
            if mask.any():
                F[np.ix_(mask, fam_idx[f])] = -1e9
        return F

    return _tr, state


def _filter_broad_genes(
    genes_by_label: dict[str, Sequence[str]],
    query,
    db_genes: set[str],
    threshold: float = 0.7,
) -> dict[str, list[str]]:
    """剔除在查询中表达广度（>expr_threshold 的细胞比例）超过 ``threshold`` 的
    DB 重叠基因，返回过滤后的 ``{标签: 基因列表}``。

    广表达基因（管家 / 泛白细胞 marker）会让逐细胞先验对每个细胞都抬高同一批
    标签，退化为近似静态先验；剔除后先验才保留细胞特异性。只统计 DB ∩ 查询
    空间内的基因（其余由 :func:`per_cell_db_prior` 自行丢弃），与先验共用
    ``expr_threshold=1.0`` 语义。
    """
    x = query.X
    if not sp.issparse(x):
        x = sp.csr_matrix(x)
    q_genes = np.asarray(query.var_names, dtype=object)
    gpos = {g: i for i, g in enumerate(q_genes)}
    db_q = [g for g in q_genes if g in db_genes]
    out = {k: list(v) for k, v in genes_by_label.items()}
    if not db_q:
        return out
    cols = np.array([gpos[g] for g in db_q], dtype=np.int64)
    sub = x[:, cols].tocsc()
    breadth = np.asarray((sub > 1.0).sum(axis=0)).ravel() / float(x.shape[0])
    brd = dict(zip(db_q, breadth.tolist()))
    for lab, gs in out.items():
        out[lab] = [g for g in gs if g not in brd or brd[g] <= threshold]
    return out


def _annotate_db_only(
    query: AnnData,
    db: CellMarkerDB,
    *,
    species: str,
    tissue: str,
    method: str,
    layer: str | None,
    marker_sources: frozenset[str],
    expr_threshold: float,
) -> AnnData:
    """无参考降级链路：纯 ``cellmarkerannot.annotate_cells``，status = db_only。"""
    db_ann = annotate_cells(
        query, db,
        method=method, species=species, tissue=tissue, layer=layer,
        marker_sources=marker_sources, expr_threshold=expr_threshold,
        inplace=False,
    )
    query.obs["hybrid_celltype"] = db_ann["celltype_predicted"].to_numpy()
    query.obs["hybrid_confidence"] = db_ann["confidence"].to_numpy(dtype=float)
    query.obs["hybrid_status"] = np.repeat(DB_ONLY, query.n_obs)
    query.uns["hybridsc"] = {
        "version": __version__,
        "params": {
            "method": "db_only", "species": species, "tissue": tissue,
            "marker_sources": sorted(marker_sources), "db_method": method,
        },
        "db_celltype": np.asarray(db_ann["celltype_predicted"], dtype=object),
        "db_confidence": db_ann["confidence"].to_numpy(dtype=float),
        "P_prior": {},
        "V": None,
    }
    return query
