#!/usr/bin/env python
"""hybrid_pipeline.py — HybridscSingleMarker 端到端测试脚本。
在真实 PBMC 数据上跑全部 5 个注释场景，横向对比注释结果并落盘图件与矩阵:
    场景                      组合方式
    pysingle_1ref             单参考 + 无 cellmarkerannot（纯 SingleR）
    hybrid_1ref               单参考 + cellmarkerannot（融合）
    pysingle_2ref             多参考 + 无 cellmarkerannot（SingleR 合并）
    hybrid_2ref               多参考 + cellmarkerannot（融合）
    cellmarkerannot_only       无参考，纯 cellmarkerannot（db_only 降级链路）
数据（全部人外周血，data_source="Experiment"，骨架设计 §4）:
    ref1   testdata/pbmc50k_refdata.h5ad   标签列 celltype
    ref2   testdata/pbmc3k_refdata.h5ad    标签列 celltype.l2
    query  testdata/vkhQ8_querydata.h5ad   真值列 celltype
对比口径（骨架设计 §6.3 质量闸门）:
    - 细粒度预测标签经 :func:`coarse_type_series` 映射到粗分类族（T/B/NK/Mono/DC/Other），
      与真值粗族逐细胞比对（排除真值 = "Other" 的细胞）得到 agreement；
    - 细粒度亚型指标：ARI / NMI / 稀有细胞 Macro-F1；
    - 每个 hybrid 场景与其 λ=0 孪生（对应 pysingle 场景）做翻转统计
      （good/bad/both_ok/both_bad + 翻转精度），量化"加入 cellmarkerannot
      是否有净价值"；
    - 融合强度按规模策略：单参考 ≥50k 细胞默认 λ=0（纯 pysingle），
      ``--lambda`` 可覆盖（设计 §4.1）；
    - 子集档可选跑 data_source="all" vs "Experiment" 的 A/B（设计 §6.3）。
    - 并展示各场景粗族分布 / UMAP 并列 / 状态分层 / 置信度 / DB 先验 P_c /
      CellMarker 证据热图（plot_gene_scores）。
输出:
    results/predictions.csv        每查询细胞 × 各场景 标签/粗族/置信度/状态
    results/summary.json           参数 + 各场景 agreement / 细粒度指标 / 状态计数 / 耗时
    results/scenario_<名>_*.csv    各场景得分矩阵 / 融合矩阵 / softmax / 先验 / V
    results/db_evidence.csv        CellMarker 证据矩阵
    results/figures/*.png          01 分布条 / 02 UMAP 并列 / 03 agreement /
                                   04 状态 / 05 置信度 / 06 先验 / 07 证据热图
用法:
    python scripts/hybrid_pipeline.py --ref1-cells 2000 --ref2-cells 2000 \
        --query-cells 1500 --n-jobs 4 --output-dir results
大参考用 backed 读入 + 分层抽样（默认 ref ~2000 / query ~1500 细胞），
保持内存与单场景耗时可控；抽样不改变场景语义（各标签取 top-N 相关样本均值）。
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    f1_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import matplotlib
matplotlib.use("Agg")                      # 无头环境，先于 pyplot/scanpy 导入
import matplotlib.pyplot as plt
import anndata as ad
from hybridscsinglemarker import hybrid_annotate
from hybridscsinglemarker._coarse_types import coarse_type_series
from hybridscsinglemarker.cellmarkerannot import CellMarkerDB, score_gene_list
from hybridscsinglemarker.cellmarkerannot.plotting import plot_gene_scores
from hybridscsinglemarker.pysingle import singleR_annotate, singleR_annotate_multi
import pipeline_utils as pu

# --------------------------------------------------------------------------- #
# 常量                                                                         #
# --------------------------------------------------------------------------- #
SPECIES = "Human"
TISSUE = "Peripheral Blood"
DATA_SOURCE = "Experiment"
# 粗分类族：展示顺序 + 统一配色（6 族 + Other + None）
FAMILY_ORDER = ["T", "B", "NK", "Mono", "DC"]
FAMILY_COLORS = {
    "T": "#4C72B0", "B": "#DD8452", "NK": "#55A868",
    "Mono": "#C44E52", "DC": "#8172B3", "Other": "#999999",
}
LEGEND_CATEGORIES = FAMILY_ORDER + ["Other", "None"]
# 与 scripts/pipeline.py 一致的 PBMC 标志基因（用于 DB 证据热图）
DEFAULT_PBMC_GENES = (
    "CD3D", "CD4", "CD8A", "MS4A1", "CD19", "NKG7",
    "GNLY", "NCAM1", "CD14", "FCGR3A", "FCER1A", "CLEC9A",
)
STATUS_ORDER = ["consistent", "low_confidence", "unknown", "db_only"]
# 场景展示顺序
SCENARIO_ORDER = [
    "pysingle_1ref", "hybrid_1ref",
    "pysingle_2ref", "hybrid_2ref", "cellmarkerannot_only",
]
SCENARIO_LABEL = {
    "pysingle_1ref": "pysingle 1ref",
    "hybrid_1ref": "hybrid 1ref",
    "pysingle_2ref": "pysingle 2ref",
    "hybrid_2ref": "hybrid 2ref",
    "cellmarkerannot_only": "cellmarkerannot only",
}
# 稀有细胞判定阈值（真值中占比低于该值视为稀有亚型）
RARE_CELL_THRESHOLD = 0.05

# --------------------------------------------------------------------------- #
# 数据加载                                                                     #
# --------------------------------------------------------------------------- #
def _stratified_subset(path: str, label_key: str, n_total: int, seed: int) -> ad.AnnData:
    """backed 读入 + 每标签分层（round-robin）抽样，返回内存中的 AnnData。
    每个标签至多取 ``n_total // n_unique`` 个细胞（不足全取），抽样总量
    接近 ``n_total``；抽样后随机洗牌。用于大参考内存安全子采样。
    """
    adata = ad.read_h5ad(path, backed="r")
    # 0 或 -1 均视为全量返回
    if n_total <= 0:
        return adata.to_memory()
    # 1. 利用 pandas groupby 替代 python 循环，大幅提升多标签场景性能
    obs_df = adata.obs[[label_key]].reset_index()
    idx_col = obs_df.columns[0]  # reset_index 会将原索引置于第一列
    obs_df.rename(columns={idx_col: "_idx"}, inplace=True)
    n_uniq = obs_df[label_key].nunique(dropna=False)
    per = max(1, n_total // n_uniq)
    # 2. 每组取前 per 个（observed=False 忽略未出现的分类值）
    sampled_df = obs_df.groupby(label_key, observed=False).head(per)
    keep_idx = sampled_df["_idx"].values
    rng = np.random.default_rng(seed)
    # 3. 【关键修复】当标签数 > n_total 时，原逻辑会抽出 n_uniq 个细胞，
    #    此处强制降采样至目标总数，确保“接近 n_total”
    if len(keep_idx) > n_total:
        keep_idx = rng.choice(keep_idx, n_total, replace=False)
    else:
        keep_idx = rng.permutation(keep_idx)
    sub = adata[keep_idx].to_memory()
    print(
        f"[load] {Path(path).name}: {n_uniq} 标签, 抽样 {len(keep_idx)} 细胞 "
        f"(每标签 ≤ {per})"
    )
    return sub


# 大参考自动封顶规则：(n_ref 下限, 每类型封顶数)，按 n_ref 从大到小匹配
# 注意：与 scripts/run_cellmarker_experiment.py 中的副本保持一致（改动需同步）
REF_CAP_RULES = ((120_000, 300), (50_000, 500))


def auto_cap_ref(n_ref: int) -> int | None:
    """按参考规模自动封顶：≥120k → 300，≥50k → 500，否则不封顶（None）。"""
    for lo, cap in REF_CAP_RULES:
        if n_ref >= lo:
            return cap
    return None

# --------------------------------------------------------------------------- #
# 对比口径                                                                     #
# --------------------------------------------------------------------------- #
def _family_agreement(pred_coarse, truth_coarse) -> float:
    """预测粗族与真值粗族逐细胞一致率（排除真值 = "Other" 与无预测细胞）。"""
    pred = np.asarray(pred_coarse, dtype=object)
    truth = np.asarray(truth_coarse, dtype=object)
    has_pred = np.array([x is not None for x in pred])
    ok = (truth != "Other") & has_pred
    if ok.sum() == 0:
        return float("nan")
    return float((pred[ok] == truth[ok]).mean())


def _compute_fine_metrics(pred_labels, truth_labels, rare_threshold: float = RARE_CELL_THRESHOLD) -> dict:
    """计算细粒度亚型指标：ARI / NMI / 稀有细胞 Macro-F1。

    Args:
        pred_labels: 预测细粒度标签数组
        truth_labels: 真值细粒度标签数组
        rare_threshold: 真值中细胞占比低于该阈值的类型视为稀有细胞，默认 5%
    Returns:
        包含各指标值的字典
    """
    pred = np.asarray(pred_labels, dtype=object)
    truth = np.asarray(truth_labels, dtype=object)

    # 过滤无效标签（真值缺失 或 预测缺失——DB-only 路径对无表达 marker 的
    # 细胞输出 NaN 预测，属设计行为，指标计算时跳过）
    valid_mask = ~pd.isna(truth) & ~pd.isna(pred) & (truth != None)
    truth_valid = truth[valid_mask]
    pred_valid = pred[valid_mask]

    if len(np.unique(truth_valid)) < 2:
        return {
            "fine_ari": float("nan"),
            "fine_nmi": float("nan"),
            "rare_macro_f1": float("nan"),
            "n_rare_types": 0,
        }

    # 基础细粒度聚类指标
    fine_ari = float(adjusted_rand_score(truth_valid, pred_valid))
    fine_nmi = float(normalized_mutual_info_score(truth_valid, pred_valid))

    # 稀有细胞 F1：筛选真值中低频类型
    type_freq = pd.Series(truth_valid).value_counts(normalize=True)
    rare_types = type_freq[type_freq < rare_threshold].index.tolist()
    n_rare = len(rare_types)

    if n_rare == 0:
        rare_f1 = float("nan")
    else:
        rare_f1 = float(
            f1_score(
                truth_valid,
                pred_valid,
                labels=rare_types,
                average="macro",
                zero_division=0,
            )
        )

    return {
        "fine_ari": fine_ari,
        "fine_nmi": fine_nmi,
        "rare_macro_f1": rare_f1,
        "n_rare_types": n_rare,
    }


def _confkind_note(scn) -> str:
    """置信度列取值说明：pysingle-only 场景存 pval（低 = 置信），展示时取 1-pval。"""
    return "1 - pval" if scn["conf_kind"] == "pval" else "confidence"

# --------------------------------------------------------------------------- #
# 场景运行                                                                     #
# --------------------------------------------------------------------------- #
def _run_pysingle(ref, labels, query, args) -> dict:
    """纯 SingleR（单参考）。"""
    t0 = time.time()
    out = singleR_annotate(
        ref, labels, query,
        fine_tune=args.fine_tune, top_n=args.top_n,
        gene_selection="hvg", max_genes=args.max_genes,
        scoring=args.scoring, chunk_size=args.chunk_size, n_jobs=args.n_jobs,
        max_cells_per_type=args.max_cells_per_type, ref_cap_seed=args.ref_cap_seed,
    )
    return {
        "labels": out["labels"].to_numpy(dtype=object),
        "confidence": out["pval"].to_numpy(dtype=float),
        "conf_kind": "pval",
        "status": None,
        "elapsed": time.time() - t0,
        "matrices": {"scores": out["scores"]},
        "singler_out": out,
        "P_prior": {},
        "V": None,
    }


def _run_pysingle_multi(refs, labels_list, query, args) -> dict:
    """纯 SingleR（多参考，跨参考合并）。"""
    t0 = time.time()
    out = singleR_annotate_multi(
        list(zip(refs, labels_list)), query,
        combine_method="max",
        fine_tune=args.fine_tune, top_n=args.top_n,
        gene_selection="hvg", max_genes=args.max_genes,
        chunk_size=args.chunk_size, n_jobs=args.n_jobs,
        scoring=args.scoring,
        max_cells_per_type=args.max_cells_per_type, ref_cap_seed=args.ref_cap_seed,
    )
    return {
        "labels": out["labels"].to_numpy(dtype=object),
        "confidence": out["scores"].max(axis=1).to_numpy(dtype=float),
        "conf_kind": "winner_score",
        "status": None,
        "elapsed": time.time() - t0,
        "matrices": {"scores": out["scores"]},
        "P_prior": {},
        "V": None,
    }


def _run_hybrid(query, refs, celltype_col, args, db,
                data_source: str = DATA_SOURCE, lambda_: float | None = None,
                reuse_singler: list | None = None,
                fusion_mode: str = "v_full") -> dict:
    """融合链路（单/多参考 + cellmarkerannot）。

    ``lambda_=None`` 时用 args.lambda_；``reuse_singler`` 在 λ=0 时复用
    已算好的 pysingle 结果（跳过重复计算，设计 §5.3 运行时闸门）。
    """
    t0 = time.time()
    lambda_ = args.lambda_ if lambda_ is None else lambda_
    out = hybrid_annotate(
        query, refs, celltype_col=celltype_col,
        species=SPECIES, tissue=TISSUE, method="singler",
        lambda_=lambda_, confidence_threshold=args.confidence_threshold,
        data_source=data_source, db_method=args.db_method, db=db,
        family_posterior=(fusion_mode == "family_posterior"),
        include_label_evidence=bool(args.label_evidence),
        feature_gene_top_n=args.feature_gene_top_n,
        n_jobs=args.n_jobs, scoring=args.scoring, top_n=args.top_n,
        fine_tune=args.fine_tune, gene_selection="hvg",
        max_genes=args.max_genes, combine_method="max",
        max_cells_per_type=args.max_cells_per_type, ref_cap_seed=args.ref_cap_seed,
        reuse_singler=reuse_singler,
    )
    hs = out.uns["hybridsc"]
    softmax_cols = list(hs["F"].columns)
    return {
        "labels": out.obs["hybrid_celltype"].to_numpy(dtype=object),
        "confidence": out.obs["hybrid_confidence"].to_numpy(dtype=float),
        "conf_kind": "hybrid",
        "status": out.obs["hybrid_status"].to_numpy(dtype=object),
        "elapsed": time.time() - t0,
        "matrices": {
            "S_corr": hs["S_corr"],
            "F": hs["F"],
            "softmax": pd.DataFrame(hs["softmax_prob"],
                                    index=query.obs_names, columns=softmax_cols),
        },
        "P_prior": hs["P_prior"],
        "V": hs["V"],
        "db_confidence": hs["db_confidence"],
        "lambda_eff": lambda_,
        "data_source": data_source,
        "fusion_mode": fusion_mode,
        "label_evidence": hs.get("label_evidence"),
        "chosen_family": hs.get("chosen_family"),
    }


def _run_db_only(query, args, db) -> dict:
    """无参考降级链路：纯 cellmarkerannot。"""
    t0 = time.time()
    out = hybrid_annotate(
        query, ref=None,
        species=SPECIES, tissue=TISSUE,
        data_source=DATA_SOURCE, db_method=args.db_method, db=db,
        confidence_threshold=args.confidence_threshold,
    )
    hs = out.uns["hybridsc"]
    return {
        "labels": out.obs["hybrid_celltype"].to_numpy(dtype=object),
        "confidence": out.obs["hybrid_confidence"].to_numpy(dtype=float),
        "conf_kind": "db",
        "status": out.obs["hybrid_status"].to_numpy(dtype=object),
        "elapsed": time.time() - t0,
        "matrices": {},
        "P_prior": hs["P_prior"],
        "V": hs["V"],
        "db_confidence": hs["db_confidence"],
    }


def run_scenarios(query, ref1, ref2, args, db) -> dict[str, dict]:
    """按固定顺序跑 5 个场景，返回 name -> result。

    融合强度按设计 §4.1 规模策略：``args.lambda_ is None`` 时，
    单参考 ≥50k 细胞 → 0（纯 pysingle），其余 → 0.3。
    """
    truth_labels = query.obs[args.query_label].astype(str)
    lambda_1ref = (
        args.lambda_
        if args.lambda_ is not None
        else pu.default_lambda(n_ref=ref1.n_obs, n_refs=1, method="singler")
    )
    lambda_2ref = (
        args.lambda_
        if args.lambda_ is not None
        else pu.default_lambda(n_ref=ref1.n_obs + ref2.n_obs, n_refs=2,
                               method="singler")
    )
    print(
        f"[fusion] 融合强度: hybrid_1ref λ={lambda_1ref}  "
        f"hybrid_2ref λ={lambda_2ref}  "
        f"(args.lambda_={args.lambda_})  模式: {args.fusion_mode}"
    )
    results: dict[str, dict] = {}
    # 1. 单参考 + 无 cellmarkerannot
    results["pysingle_1ref"] = _run_pysingle(
        ref1, ref1.obs[args.ref1_label], query, args)
    # 2. 单参考 + cellmarkerannot
    reuse_1ref = None
    if lambda_1ref == 0.0 and "singler_out" in results["pysingle_1ref"]:
        # λ=0 时 hybrid ≡ pysingle 孪生：复用其 singleR 结果，只补 DB 证据层
        reuse_1ref = [results["pysingle_1ref"]["singler_out"]]
        print("[fusion] hybrid_1ref λ=0 → 复用 pysingle_1ref 孪生，跳过重复 pysingle")
    results["hybrid_1ref"] = _run_hybrid(
        query, ref1, args.ref1_label, args, db, lambda_=lambda_1ref,
        reuse_singler=reuse_1ref, fusion_mode=args.fusion_mode)
    # 3. 多参考 + 无 cellmarkerannot
    results["pysingle_2ref"] = _run_pysingle_multi(
        [ref1, ref2], [ref1.obs[args.ref1_label],
                       ref2.obs[args.ref2_label]], query, args)
    # 4. 多参考 + cellmarkerannot
    results["hybrid_2ref"] = _run_hybrid(
        query, [ref1, ref2], [args.ref1_label, args.ref2_label], args, db,
        lambda_=lambda_2ref, fusion_mode=args.fusion_mode)
    # 5. cellmarkerannot 单独
    results["cellmarkerannot_only"] = _run_db_only(query, args, db)

    # 统一：粗分类族 + agreement
    for scn in results.values():
        scn["coarse"] = coarse_type_series(scn["labels"])
    truth_coarse = coarse_type_series(truth_labels.to_numpy())
    for name, scn in results.items():
        scn["agreement"] = _family_agreement(scn["coarse"], truth_coarse)

    # 细粒度亚型指标
    truth_fine = truth_labels.to_numpy()
    for name, scn in results.items():
        scn["fine_metrics"] = _compute_fine_metrics(scn["labels"], truth_fine)

    # 翻转质量闸门（设计 §6.3）：hybrid 场景 vs 其 λ=0 孪生（pysingle 场景）。
    # 孪生参数与 hybrid 完全一致（同数据、同 max_cells_per_type/max_genes/
    # scoring/fine_tune/n_jobs），即 run_scenarios 中的 pysingle_* 场景。
    truth_c = np.asarray(truth_coarse, dtype=object)
    truth_ok = (truth_c != "Other")
    for hy, twin in (("hybrid_1ref", "pysingle_1ref"),
                     ("hybrid_2ref", "pysingle_2ref")):
        res, base = results[hy], results[twin]
        mask = ~pd.isna(truth_fine)
        st_fine = pu.flip_stats(base["labels"][mask], res["labels"][mask],
                                truth_fine[mask])
        ok = truth_ok & np.array([x is not None for x in base["coarse"]]) \
            & np.array([x is not None for x in res["coarse"]])
        st_coarse = pu.flip_stats(base["coarse"][ok], res["coarse"][ok],
                                  truth_c[ok])
        res["flip_fine"] = st_fine
        res["flip_coarse"] = st_coarse
        res["n_flipped"] = int((res["labels"] != base["labels"]).sum())
        res["twin_scenario"] = twin

    return results, truth_coarse

# --------------------------------------------------------------------------- #
# 落盘：CSV 矩阵                                                               #
# --------------------------------------------------------------------------- #
def save_matrices(results: dict[str, dict], results_dir: Path) -> None:
    for name, scn in results.items():
        for mname, mat in scn["matrices"].items():
            pd.DataFrame(mat).to_csv(results_dir / f"scenario_{name}_{mname}.csv")
        if scn["P_prior"]:
            pd.DataFrame(
                sorted(scn["P_prior"].items(), key=lambda kv: -kv[1]),
                columns=["label", "P"],
            ).to_csv(results_dir / f"scenario_{name}_prior.csv", index=False)
        if scn["V"] is not None:
            pd.DataFrame({"V": scn["V"]}).to_csv(
                results_dir / f"scenario_{name}_V.csv", index=False)


def save_predictions(query, results: dict[str, dict], truth_coarse,
                     results_dir: Path, query_label: str) -> None:
    df = pd.DataFrame(
        {
            "barcode": query.obs_names,
            "truth": query.obs[query_label].astype(str),
            "truth_coarse": [str(x) for x in truth_coarse],
        }
    )
    for name in SCENARIO_ORDER:
        scn = results[name]
        df[f"{name}_label"] = [str(x) if x is not None else "" for x in scn["labels"]]
        df[f"{name}_coarse"] = [str(x) if x is not None else "" for x in scn["coarse"]]
        df[f"{name}_conf"] = scn["confidence"] if scn["confidence"] is not None else np.nan
        df[f"{name}_status"] = (
            scn["status"] if scn["status"] is not None else np.full(len(df), "")
        )
    df.to_csv(results_dir / "predictions.csv", index=False)


def save_summary(results: dict[str, dict], args, results_dir: Path,
                 total_elapsed: float | None = None) -> dict:
    scn_summary: dict = {}
    for name in SCENARIO_ORDER:
        scn = results[name]
        status_counts = (
            dict(Counter(scn["status"])) if scn["status"] is not None else {}
        )
        conf = scn["confidence"]
        fm = scn["fine_metrics"]
        scn_summary[name] = {
            "n_cells": len(scn["labels"]),
            "agreement_vs_truth": scn["agreement"],
            "fine_ari": fm["fine_ari"],
            "fine_nmi": fm["fine_nmi"],
            "rare_macro_f1": fm["rare_macro_f1"],
            "n_rare_cell_types": fm["n_rare_types"],
            "confidence_mean": float(np.nanmean(conf)) if conf is not None else None,
            "confidence_kind": _confkind_note(scn),
            "status_counts": status_counts,
            "elapsed_s": round(scn["elapsed"], 2),
            "top_labels": dict(Counter(scn["labels"]).most_common(10)),
        }
        if scn.get("flip_fine"):
            scn_summary[name]["flip_fine"] = scn["flip_fine"]
            scn_summary[name]["flip_coarse"] = scn["flip_coarse"]
            scn_summary[name]["n_flipped"] = scn["n_flipped"]
            scn_summary[name]["twin_scenario"] = scn["twin_scenario"]
        if scn.get("lambda_eff") is not None:
            scn_summary[name]["lambda_eff"] = scn["lambda_eff"]
            scn_summary[name]["data_source"] = scn["data_source"]
        if scn.get("fusion_mode"):
            scn_summary[name]["fusion_mode"] = scn["fusion_mode"]
        if scn.get("label_evidence"):
            scn_summary[name]["n_label_evidence"] = len(scn["label_evidence"])
    summary = {
        "version": 1,
        "params": {
            "species": SPECIES, "tissue": TISSUE, "data_source": DATA_SOURCE,
            "ref1": Path(args.ref1).name, "ref1_label": args.ref1_label,
            "ref2": Path(args.ref2).name, "ref2_label": args.ref2_label,
            "query": Path(args.query).name, "query_label": args.query_label,
            "ref1_cells": args.ref1_cells, "ref2_cells": args.ref2_cells,
            "query_cells": args.query_cells, "seed": args.seed,
            "lambda": args.lambda_, "confidence_threshold": args.confidence_threshold,
            "db_method": args.db_method, "feature_gene_top_n": args.feature_gene_top_n,
            "n_jobs": args.n_jobs, "scoring": args.scoring, "top_n": args.top_n,
            "fine_tune": args.fine_tune, "max_genes": args.max_genes,
            "max_cells_per_type": args.max_cells_per_type,
            "ref_cap_seed": args.ref_cap_seed,
            "rare_cell_threshold": RARE_CELL_THRESHOLD,
            "fusion_mode": args.fusion_mode,
            "label_evidence": args.label_evidence,
        },
        "scenarios": scn_summary,
    }
    if total_elapsed is not None:
        summary["total_elapsed_s"] = round(total_elapsed, 2)
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def run_ab_experiment(query, ref1, args, db) -> dict:
    """子集档 A/B：data_source="all" vs "Experiment" 的融合对比（设计 §6.3）。

    其余参数与 ``hybrid_1ref`` 场景完全一致，仅切换 DB 证据范围；
    返回 ``{metrics, flip, agreement}`` 供落盘与图件。
    """
    truth_labels = query.obs[args.query_label].astype(str)
    truth_coarse = coarse_type_series(truth_labels.to_numpy())
    truth_c = np.asarray(truth_coarse, dtype=object)
    truth_fine = truth_labels.to_numpy()
    eff_lambda = args.lambda_
    if eff_lambda is None:
        eff_lambda = pu.default_lambda(n_ref=ref1.n_obs, n_refs=1,
                                       method="singler")
    exp = _run_hybrid(query, ref1, args.ref1_label, args, db,
                      data_source=DATA_SOURCE, lambda_=eff_lambda,
                      fusion_mode=args.fusion_mode)
    alt = _run_hybrid(query, ref1, args.ref1_label, args, db,
                      data_source="all", lambda_=eff_lambda,
                      fusion_mode=args.fusion_mode)
    exp["coarse"] = coarse_type_series(exp["labels"])
    alt["coarse"] = coarse_type_series(alt["labels"])
    exp["agreement"] = _family_agreement(exp["coarse"], truth_coarse)
    alt["agreement"] = _family_agreement(alt["coarse"], truth_coarse)
    mask = ~pd.isna(truth_fine)
    ok = (truth_c != "Other") & np.array([x is not None for x in exp["coarse"]]) \
        & np.array([x is not None for x in alt["coarse"]])
    flip = pu.flip_stats(exp["labels"][mask], alt["labels"][mask],
                         truth_fine[mask])
    flip_coarse = pu.flip_stats(exp["coarse"][ok], alt["coarse"][ok],
                                truth_c[ok])
    return {
        "experiment": {
            "agreement": exp["agreement"],
            "elapsed_s": round(exp["elapsed"], 2),
            "lambda_eff": exp["lambda_eff"],
        },
        "all": {
            "agreement": alt["agreement"],
            "elapsed_s": round(alt["elapsed"], 2),
            "lambda_eff": alt["lambda_eff"],
        },
        "delta_agreement": alt["agreement"] - exp["agreement"],
        "flip_fine": flip,
        "flip_coarse": flip_coarse,
        "n_flipped": int((exp["labels"] != alt["labels"]).sum()),
    }


def write_report(results, args, summary, ab, results_dir: Path) -> Path:
    """写 ``report.md``：参数、各场景指标、翻转闸门、A/B、总耗时。"""
    def _n_cells(v: int) -> str:
        return "全量" if v <= 0 else f"{v} 细胞"

    lines = [
        "# HybridscSingleMarker 端到端验证报告",
        "",
        f"- 日期: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 物种/组织: {SPECIES} / {TISSUE}  data_source: {DATA_SOURCE}",
        f"- ref1: {Path(args.ref1).name} ({args.ref1_label}, {_n_cells(args.ref1_cells)})",
        f"- ref2: {Path(args.ref2).name} ({args.ref2_label}, {_n_cells(args.ref2_cells)})",
        f"- query: {Path(args.query).name} ({args.query_label}, {_n_cells(args.query_cells)})",
        f"- 融合强度: hybrid_1ref λ={results['hybrid_1ref'].get('lambda_eff')}, "
        f"hybrid_2ref λ={results['hybrid_2ref'].get('lambda_eff')}",
        f"- 融合模式: {args.fusion_mode}",
        f"- 每标签注释矩阵: {args.label_evidence}",
        f"- 总耗时: {summary.get('total_elapsed_s', 'n/a')} s",
        "",
        "## 各场景指标（粗族一致率 / 细粒度 ARI·NMI·稀有 Macro-F1）",
        "",
        "| 场景 | agreement | fine ARI | fine NMI | rare F1 | 耗时(s) |",
        "|---|---|---|---|---|---|",
    ]
    for n in SCENARIO_ORDER:
        s = summary["scenarios"][n]
        lines.append(
            f"| {SCENARIO_LABEL[n]} | {s['agreement_vs_truth']:.4f} | "
            f"{s['fine_ari']:.4f} | {s['fine_nmi']:.4f} | "
            f"{s['rare_macro_f1']:.4f} | {s['elapsed_s']} |"
        )
    lines += ["", "## 翻转质量闸门（hybrid vs λ=0 孪生）", "",
              "| 场景 | 翻转数 | good | bad | both_ok | both_bad | 翻转精度(细) | 翻转精度(粗) |",
              "|---|---|---|---|---|---|---|---|"]
    for hy in ("hybrid_1ref", "hybrid_2ref"):
        s = summary["scenarios"][hy]
        if "flip_fine" not in s:
            continue
        f, c = s["flip_fine"], s["flip_coarse"]
        lines.append(
            f"| {SCENARIO_LABEL[hy]} | {s['n_flipped']} | {f['n_good']} | "
            f"{f['n_bad']} | {f['n_both_ok']} | {f['n_both_bad']} | "
            f"{f['flip_precision']:.3f} | {c['flip_precision']:.3f} |"
        )
    lines.append(
        "\n> 翻转精度 = good/(good+bad)；< 0.5 表示该规模下融合无净价值"
        "（设计 §6.3）。\n"
    )
    if ab:
        lines += [
            "## A/B: data_source='all' vs 'Experiment'（子集档）", "",
            f"- Experiment 一致率: {ab['experiment']['agreement']:.4f}",
            f"- all 一致率: {ab['all']['agreement']:.4f}",
            f"- Δ一致率 (all − Experiment): {ab['delta_agreement']:+.4f}",
            f"- 翻转精度 (fine): {ab['flip_fine']['flip_precision']:.3f}",
            f"- 翻转数: {ab['n_flipped']}",
            "",
        ]
    n_evidence = sum(
        summary["scenarios"][n].get("n_label_evidence", 0)
        for n in SCENARIO_ORDER
    )
    if n_evidence:
        lines += [
            "## 每标签注释矩阵与热图",
            "",
            f"- 共保存 {n_evidence} 张每标签 score_gene_list 风格注释矩阵"
            "（CSV）及对应热图，见 `results/label_evidence_<scn>/` 与 "
            "`results/figures/label_evidence_<scn>/`、"
            "`09_label_evidence_score_<scn>.png`。",
            "",
        ]
    path = results_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def plot_flip_precision(results, ab, fig_dir: Path) -> None:
    """翻转精度图（设计 §6.3）：各 hybrid 场景细/粗族精度 + A/B 对照。"""
    labels, coarse, fine = [], [], []
    for hy in ("hybrid_1ref", "hybrid_2ref"):
        scn = results[hy]
        if "flip_fine" not in scn:
            continue
        labels.append(SCENARIO_LABEL[hy] + " (coarse)")
        labels.append(SCENARIO_LABEL[hy] + " (fine)")
        coarse.append(scn["flip_coarse"]["flip_precision"])
        fine.append(scn["flip_fine"]["flip_precision"])
    if not labels:
        return
    x = np.arange(len(labels))
    vals = np.array([v for pair in zip(coarse, fine) for v in pair])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(0.5, color="grey", ls="--", lw=1, label="net-value threshold (0.5)")
    ax.bar(x, vals, color=["#55A868" if v >= 0.5 else "#C44E52" for v in vals])
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    if ab is not None:
        xab = len(labels) + 0.5
        ax.bar([xab], [ab["flip_fine"]["flip_precision"]],
               color="#8172B3", label="A/B all vs Experiment (fine)")
        ax.text(xab, ab["flip_fine"]["flip_precision"] + 0.02,
                f"{ab['flip_fine']['flip_precision']:.2f}", ha="center", fontsize=8)
        labels.append("A/B (all→Experiment)")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("flip precision (good / (good + bad))")
    ax.set_title("Fusion flip precision vs λ=0 twin (design §6.3 gate)")
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "08_flip_precision.png", dpi=150)
    plt.close(fig)


def save_label_evidence(results, results_dir, fig_dir, args) -> None:
    """保存每标签 ``score_gene_list`` 风格注释矩阵 + 热图（Variant B 可视化）。

    对每个 hybrid 场景（含 ``uns["hybridsc"]["label_evidence"]``）:

    - CSV: ``results/label_evidence_<scn>/<label>.csv`` —— 每参考标签一张
      注释矩阵（特征基因 × DB 细胞类型的支持证据 + 最后一行 ``Score``，
      格式与 ``cellmarkerannot.score_gene_list`` 一致）；
    - 热图: ``figures/label_evidence_<scn>/<label>.png`` —— 用
      ``plot_gene_scores`` 绘制（证据热图 + Score 行），仅 top-K 标签；
    - 概览: ``figures/09_label_evidence_score_<scn>.png`` —— 标签 × DB
      细胞类型的富集 Score 热图（行=参考标签，列=DB 细胞类型）。
    """
    n_csv = 0
    for name in SCENARIO_ORDER:
        scn = results.get(name)
        if not scn or not scn.get("label_evidence"):
            continue
        mats = scn["label_evidence"]
        csv_dir = results_dir / f"label_evidence_{name}"
        fig_sub = fig_dir / f"label_evidence_{name}"
        csv_dir.mkdir(parents=True, exist_ok=True)
        fig_sub.mkdir(parents=True, exist_ok=True)

        # 按该场景预测频次取 top-K 标签做逐标签热图（CSV 全量保存）
        counts = Counter(scn["labels"])
        ranked = sorted(mats, key=lambda lab: (-counts.get(lab, 0), lab))
        topk = set(ranked[: max(1, args.label_heatmap_k)])
        for lab, m in mats.items():
            if m is None or len(m) == 0:
                continue
            m.to_csv(csv_dir / f"{lab}.csv")
            n_csv += 1
            if lab not in topk:
                continue
            try:
                fig = plot_gene_scores(
                    m,
                    title=f"{lab} — CellMarker evidence "
                          f"({SPECIES}/{TISSUE}/{DATA_SOURCE})",
                    save_path=fig_sub / f"{lab}.png",
                )
                plt.close(fig)
            except ValueError as exc:
                print(f"[warn] 标签 {lab} 热图跳过: {exc}")

        # 概览：标签 × DB 细胞类型 Score 热图（行=参考标签，列=DB 细胞类型）
        col_union = sorted({
            c for m in mats.values() if m is not None and len(m)
            for c in m.columns
        })
        if not col_union:
            continue
        rows, row_names = [], []
        for lab in ranked:
            m = mats[lab]
            if m is None or len(m) == 0:
                continue
            rows.append(
                m.loc["Score"].reindex(col_union, fill_value=0.0).to_numpy(dtype=float)
            )
            row_names.append(lab)
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(
            max(8.0, 0.22 * len(col_union)),
            max(4.0, 0.30 * len(rows)),
        ))
        im = ax.imshow(np.vstack(rows), aspect="auto", cmap="Purples_r")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(row_names, fontsize=8)
        ax.set_xticks(range(len(col_union)))
        ax.set_xticklabels(col_union, rotation=90, fontsize=6)
        ax.set_title(
            f"Label × DB cell-type enrichment Score "
            f"({SPECIES}/{TISSUE}/{DATA_SOURCE})"
        )
        fig.colorbar(im, ax=ax, label="Score")
        fig.tight_layout()
        fig.savefig(fig_dir / f"09_label_evidence_score_{name}.png", dpi=150)
        plt.close(fig)
    print(f"[label_evidence] 保存 {n_csv} 张每标签注释矩阵 CSV + 热图")

# --------------------------------------------------------------------------- #
# 图件                                                                         #
# --------------------------------------------------------------------------- #
def _legend_str(values) -> np.ndarray:
    return np.array([v if v is not None else "None" for v in values], dtype=object)


def _obs_coarse_column(obs, coarse, name: str) -> pd.Series:
    """把粗分类数组写入 obs 列（索引对齐 obs_names，修复全灰点）。

    旧实现 ``pd.Series(coarse)`` 使用默认 RangeIndex，赋值给以 barcode 为索引
    的 ``obs`` 时按索引对齐全部落空 → 列全 NaN → scanpy 把所有点画成灰色。
    这里显式传入 ``index=obs.index`` 并对齐类别（``None`` → ``"None"``）。
    """
    cats = pd.CategoricalDtype(categories=LEGEND_CATEGORIES)
    s = pd.Series(_legend_str(coarse), index=obs.index).astype(cats)
    s.name = name
    return s


def _prior_layout(n_labels: int) -> tuple[float, float, int]:
    """按标签数自适应先验图布局：``(每面板高度, y 轴字体, 刻度抽稀步长)``。

    标签很多时固定 3.4 英寸高度 + 默认字体会让 y 轴标签重叠成黑团；
    这里让面板随标签数变高、字体自适应缩小、超过 40 个标签时只显示
    每隔一个的刻度（抽稀，bars 仍全量绘制）。
    """
    height = min(max(3.4, 0.22 * n_labels), 12.0)
    fontsize = float(np.clip(130.0 / max(1, n_labels), 5.0, 10.0))
    step = 1 if n_labels <= 40 else 2
    return height, fontsize, step


def plot_family_distribution(results, truth_coarse, fig_dir: Path) -> None:
    cats = LEGEND_CATEGORIES
    names = ["truth"] + [SCENARIO_LABEL[n] for n in SCENARIO_ORDER]
    def fracs(coarse) -> np.ndarray:
        arr = _legend_str(coarse)
        u, c = np.unique(arr, return_counts=True)
        cnt = dict(zip(u, c))
        total = len(arr)
        return np.array([cnt.get(c, 0) / total for c in cats])
    M = np.vstack([fracs(truth_coarse)] +
                  [fracs(results[n]["coarse"]) for n in SCENARIO_ORDER])
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    bottom = np.zeros(len(names))
    colors = [FAMILY_COLORS.get(c, "#cccccc") for c in cats]
    for i, c in enumerate(cats):
        ax.bar(names, M[:, i], bottom=bottom, label=c, color=colors[i], width=0.62)
        bottom += M[:, i]
    ax.set_ylabel("cell fraction")
    ax.set_title("Coarse-family distribution (ref / query truth vs scenarios)")
    ax.legend(ncol=len(cats), fontsize=8, loc="upper center", frameon=False)
    ax.set_ylim(0, 1.06)
    fig.tight_layout()
    fig.savefig(fig_dir / "01_family_distribution.png", dpi=150)
    plt.close(fig)


def plot_umap_grid(query, results, truth_coarse, fig_dir: Path) -> None:
    """5 场景 + 真值 的 UMAP 并列（粗族着色，统一图例）。"""
    import scanpy as sc
    if "X_umap" not in query.obsm:
        print("[warn] 查询数据无 X_umap，跳过 UMAP 对比图")
        return
    q = query.copy()
    q.obs["coarse_truth"] = _obs_coarse_column(q.obs, truth_coarse, "coarse_truth")
    panels = [("truth", "coarse_truth")]
    for name in SCENARIO_ORDER:
        col = f"sc_{name}"
        q.obs[col] = _obs_coarse_column(q.obs, results[name]["coarse"], col)
        panels.append((SCENARIO_LABEL[name], col))
    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.3 * nrows))
    axes = np.asarray(axes).reshape(-1)
    palette = {c: FAMILY_COLORS.get(c, "#cccccc") for c in LEGEND_CATEGORIES}
    for ax, (title, col) in zip(axes, panels):
        sc.pl.umap(q, color=col, ax=ax, show=False, palette=palette,
                   legend_loc="right margin", frameon=True)
        ax.set_title(title, fontsize=12)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"UMAP coarse-family comparison ({SPECIES} / {TISSUE})",
                 y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_dir / "02_umap_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_agreement(results, fig_dir: Path) -> None:
    names = [SCENARIO_LABEL[n] for n in SCENARIO_ORDER]
    vals = [results[n]["agreement"] for n in SCENARIO_ORDER]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = np.asarray(ax.barh(names, vals, color="#4C72B0"), dtype=object)
    for bar in bars[np.isnan(vals)]:
        bar.set_alpha(0.3)
    for i, v in enumerate(vals):
        if not np.isnan(v):
            ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("coarse-family agreement vs query truth (truth ≠ Other)")
    ax.set_title("Annotation agreement across scenarios")
    fig.tight_layout()
    fig.savefig(fig_dir / "03_agreement.png", dpi=150)
    plt.close(fig)


def plot_status_stack(results, fig_dir: Path) -> None:
    names = [SCENARIO_LABEL[n] for n in SCENARIO_ORDER if results[n]["status"] is not None]
    order = [n for n in SCENARIO_ORDER if results[n]["status"] is not None]
    M = np.zeros((len(order), len(STATUS_ORDER)))
    for i, n in enumerate(order):
        arr = results[n]["status"]
        for j, st in enumerate(STATUS_ORDER):
            M[i, j] = float((arr == st).mean())
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bottom = np.zeros(len(names))
    colors = ["#55A868", "#DD8452", "#C44E52", "#4C72B0"]
    for j, st in enumerate(STATUS_ORDER):
        ax.bar(names, M[:, j], bottom=bottom, label=st, color=colors[j], width=0.5)
        bottom += M[:, j]
    ax.set_ylabel("cell fraction")
    ax.set_title("hybrid_status distribution (layering)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.06)
    fig.tight_layout()
    fig.savefig(fig_dir / "04_status_stacking.png", dpi=150)
    plt.close(fig)


def plot_confidence(results, fig_dir: Path) -> None:
    names, data = [], []
    for n in SCENARIO_ORDER:
        scn = results[n]
        if scn["confidence"] is None:
            continue
        names.append(SCENARIO_LABEL[n])
        c = scn["confidence"].astype(float)
        data.append((1.0 - c) if scn["conf_kind"] == "pval" else c)
    fig, ax = plt.subplots(figsize=(8, 4))
    bp = ax.boxplot(data, tick_labels=names, showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", lw=1.2))
    for patch in bp["boxes"]:
        patch.set_facecolor("#4C72B0")
    ax.set_ylabel("confidence (pysingle 1ref: 1 − pval; multi-ref & hybrid: winner score)")
    ax.set_title("Per-cell confidence / score distribution")
    fig.tight_layout()
    fig.savefig(fig_dir / "05_confidence.png", dpi=150)
    plt.close(fig)


def plot_prior(results, fig_dir: Path) -> None:
    hybrid = {SCENARIO_LABEL[n]: results[n] for n in SCENARIO_ORDER
              if results[n]["P_prior"]}
    if not hybrid:
        return
    layouts = [_prior_layout(len(scn["P_prior"])) for scn in hybrid.values()]
    heights = [h for h, _, _ in layouts]
    fig, axes = plt.subplots(len(hybrid), 1, figsize=(9, sum(heights)))
    axes = np.atleast_1d(axes)
    for ax, (name, scn), (height, fontsize, step) in zip(
        axes, hybrid.items(), layouts
    ):
        P = scn["P_prior"]
        labs = sorted(P, key=lambda k: -P[k])
        ax.barh(labs, [P[k] for k in labs], color="#DD8452")
        ax.set_title(f"{name}: DB prior P_c per ref label (max-normalized)",
                     fontsize=11)
        ax.set_xlim(0, 1.05)
        # 标签多时抽稀刻度（bars 全量保留），并自适应字体避免重叠
        tick_idx = list(range(0, len(labs), step))
        ax.set_yticks(tick_idx)
        ax.set_yticklabels([labs[i] for i in tick_idx], fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=8)
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(fig_dir / "06_prior.png", dpi=150)
    plt.close(fig)


def plot_evidence(db, fig_dir: Path) -> None:
    """CellMarker 证据热图：PBMC 标志基因 × 人外周血细胞类型（plot_gene_scores）。"""
    ev = score_gene_list(DEFAULT_PBMC_GENES, db, species=SPECIES,
                         tissue=TISSUE, data_source=DATA_SOURCE)
    ev.to_csv(fig_dir.parent / "db_evidence.csv")
    plot_gene_scores(
        ev,
        title=f"CellMarker evidence: PBMC marker genes ({SPECIES} / {TISSUE} / {DATA_SOURCE})",
        save_path=fig_dir / "07_db_evidence.png",
    )

# --------------------------------------------------------------------------- #
# 主流程                                                                       #
# --------------------------------------------------------------------------- #
def main(args) -> None:
    results_dir = Path(args.output_dir)
    fig_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # 全量档：按设计 §5.3 强制全量加载 + 降基因数到 3000 + 关闭 A/B（省时）
    if args.full:
        args.ref1_cells = args.ref2_cells = args.query_cells = 0
        if args.max_genes >= 5000:
            args.max_genes = 3000
        if args.ab is None:
            args.ab = False
        print("[full] 全量档: ref/query 全量加载, max_genes=3000, "
              "A/B 关闭（可 --ab 显式开启）")
    elif args.ab is None:
        args.ab = True
    print("=" * 70)
    print("HybridscSingleMarker 端到端测试")
    print(f"  ref1={args.ref1} ({args.ref1_label}), ref2={args.ref2} "
          f"({args.ref2_label})")
    print(f"  query={args.query} ({args.query_label}), species={SPECIES}, "
          f"tissue={TISSUE}, data_source={DATA_SOURCE}")
    print("=" * 70)
    db = CellMarkerDB()
    # 1. 数据加载（backed + 分层抽样）
    ref1 = _stratified_subset(args.ref1, args.ref1_label, args.ref1_cells, args.seed)
    ref2 = _stratified_subset(args.ref2, args.ref2_label, args.ref2_cells, args.seed + 1)
    query = _stratified_subset(args.query, args.query_label,
                               args.query_cells, args.seed + 2)
    # 参考封顶：显式参数优先，否则按自动规则（REF_CAP_RULES；n_ref 取实际加载规模
    # 之和，与 Task 5 的 run_cellmarker_experiment.py 口径一致；若用请求值，
    # --ref1-cells 0/-1 全量加载会绕过封顶）
    cap = args.max_cells_per_type
    if cap is None:
        cap = auto_cap_ref(ref1.n_obs + ref2.n_obs)
    if cap is not None and cap < 1:
        cap = None  # 与 --ref*-cells 0/-1=全量的约定一致：0 视为不封顶
    args.max_cells_per_type = cap  # 就地改写为生效值，供 summary.json 记录
    print(f"[cap] 每参考类型封顶细胞数: {cap if cap is not None else '不封顶'}")
    # 2. 跑 5 个场景
    results, truth_coarse = run_scenarios(query, ref1, ref2, args, db)
    # 3. 打印逐场景摘要
    print("\n[summary] 各场景粗族一致率 / 细粒度指标 + 耗时:")
    for n in SCENARIO_ORDER:
        scn = results[n]
        fm = scn["fine_metrics"]
        print(f"  {SCENARIO_LABEL[n]:<22} agreement={scn['agreement']:.3f}  "
              f"ARI={fm['fine_ari']:.3f}  NMI={fm['fine_nmi']:.3f}  "
              f"rare_F1={fm['rare_macro_f1']:.3f}  ({scn['elapsed']:.1f}s)")
    # 4. 落盘
    save_matrices(results, results_dir)
    save_predictions(query, results, truth_coarse, results_dir, args.query_label)
    summary = save_summary(results, args, results_dir, total_elapsed=time.time() - t0)
    print(f"\n[status] 分层计数（hybrid / db_only 场景）:")
    for n in SCENARIO_ORDER:
        if results[n]["status"] is not None:
            print(f"  {SCENARIO_LABEL[n]:<22} "
                  f"{dict(Counter(results[n]['status']))}")
    # 5. 图件
    plot_family_distribution(results, truth_coarse, fig_dir)
    plot_umap_grid(query, results, truth_coarse, fig_dir)
    plot_agreement(results, fig_dir)
    plot_status_stack(results, fig_dir)
    plot_confidence(results, fig_dir)
    plot_prior(results, fig_dir)
    plot_evidence(db, fig_dir)
    # 6. 翻转闸门图 + A/B（子集档）
    ab = None
    if args.ab:
        print("\n[ab] data_source='all' vs 'Experiment' A/B（hybrid_1ref, 子集档）...")
        ab = run_ab_experiment(query, ref1, args, db)
        (results_dir / "ab_experiment.json").write_text(
            json.dumps({k: (v if not isinstance(v, dict) else
                            {kk: round(vv, 4) if isinstance(vv, float) else vv
                             for kk, vv in v.items()})
                        for k, v in ab.items()}, indent=2, ensure_ascii=False))
        print(f"  Experiment agreement={ab['experiment']['agreement']:.4f}  "
              f"all agreement={ab['all']['agreement']:.4f}  "
              f"Δ={ab['delta_agreement']:+.4f}  "
              f"flip_precision={ab['flip_fine']['flip_precision']:.3f}")
    plot_flip_precision(results, ab, fig_dir)
    # 6b. 每标签注释矩阵 + 热图（Variant B 可视化交付物）
    save_label_evidence(results, results_dir, fig_dir, args)
    # 7. 报告
    report = write_report(results, args, summary, ab, results_dir)
    print(f"\n[figures] 写入 {fig_dir}:")
    for p in sorted(fig_dir.glob("*.png")):
        print(f"  - {p.name}")
    print(f"\n[results] 写入 {results_dir}（predictions.csv / summary.json / "
          f"report.md / ab_experiment.json / scenario_* 矩阵）")
    print(f"总耗时 {time.time() - t0:.1f}s")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref1", default=str(REPO_ROOT / "testdata" / "pbmc50k_refdata.h5ad"))
    p.add_argument("--ref1-label", default="celltype")
    p.add_argument("--ref2", default=str(REPO_ROOT / "testdata" / "pbmc3k_refdata.h5ad"))
    p.add_argument("--ref2-label", default="celltype.l2")
    p.add_argument("--query", default=str(REPO_ROOT / "testdata" / "vkhQ8_querydata.h5ad"))
    p.add_argument("--query-label", default="celltype")
    p.add_argument("--ref1-cells", type=int, default=2000)
    p.add_argument("--ref2-cells", type=int, default=2000)
    p.add_argument("--query-cells", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lambda", dest="lambda_", type=float, default=None,
                   help="融合强度 F = S_corr × (1 + λ·P[cell,c])；默认 None=自动"
                        "（单参考 ≥50k → 0，否则 0.3，见设计 §4.1）")
    p.add_argument("--confidence-threshold", type=float, default=0.3)
    p.add_argument("--db-method", default="weighted")
    p.add_argument("--feature-gene-top-n", type=int, default=200)
    p.add_argument("--n-jobs", type=int, default=16)
    p.add_argument("--scoring", default="cells", choices=["cells", "profile"])
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--max-genes", type=int, default=5000)
    p.add_argument("--chunk-size", type=int, default=5000)
    p.add_argument("--fine-tune", dest="fine_tune", action="store_true", default=True)
    p.add_argument("--no-fine-tune", dest="fine_tune", action="store_false")
    p.add_argument("--max-cells-per-type", type=int, default=None,
                   help="每参考类型最多细胞数（None=自动，规则见 REF_CAP_RULES）")
    p.add_argument("--ref-cap-seed", type=int, default=0,
                   help="封顶抽样种子（默认 0，逐位可复现）")
    p.add_argument("--output-dir", default=str(REPO_ROOT / "results"))
    p.add_argument("--full", action="store_true",
                   help="全量档：ref/query 全量加载（封顶自动、max_genes=3000），"
                        "目标 ≤30-45 分钟")
    p.add_argument("--ab", dest="ab", action="store_true", default=None,
                   help="显式开启 all vs Experiment A/B（子集档默认开启，全量档关闭）")
    p.add_argument("--no-ab", dest="ab", action="store_false",
                   help="关闭 A/B")
    p.add_argument(
        "--fusion-mode", default="v_full", choices=["v_full", "family_posterior"],
        help="融合模式：v_full=乘法融合+粗族先验（默认）；"
             "family_posterior=Variant B 族级后验再排序（DB 只改粗族，"
             "族内细标签归 pysingle，见设计 §6.3）",
    )
    p.add_argument("--label-evidence", dest="label_evidence",
                   action="store_true", default=True,
                   help="hybrid 场景保存每标签 score_gene_list 风格注释矩阵"
                        "（CSV + 热图）")
    p.add_argument("--no-label-evidence", dest="label_evidence",
                   action="store_false",
                   help="关闭每标签注释矩阵/热图输出")
    p.add_argument("--label-heatmap-k", type=int, default=12,
                   help="每场景生成逐标签热图的标签数（按预测频次 top-K，"
                        "默认 12）")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(args)
