#!/usr/bin/env python
"""CellMarker 增益实验：统一 ``hybrid_annotate`` 下 with vs without CellMarker。

三大研究问题:
  1. 引入 CellMarker 能否提升注释性能？
  2. 提升幅度多大？
  3. 用哪些指标与图作为证据？

实验设计（与骨架设计/验证报告一致，2026-08-21 更新）:
  - **统一管线**：所有 run 都用 ``hybrid_annotate``（species=Human,
    tissue=Peripheral Blood, data_source=Experiment, scoring=auto,
    fine_tune=True）。
  - **scoring=auto**（新默认）：查询规模 ≤10k 用 ``cells``（精度优先，实测
    5k: 0.937 vs profile 0.910）；>10k 用 ``profile``（速度优先，实测 14× 提速，
    全量 160k ref / 58k query 7.5 分钟，见验证报告 §3）。
  - **有 CellMarker**：``lambda_=0.3``（默认融合强度，逐细胞 DB 证据先验
    ``S_corr = S0 × (1 + λ·P[cell,c])`` 作用在首轮得分上）；``--scale-aware-lambda``
    默认开启：**单参考 ≥50k 细胞 → λ=0**（实测该规模融合净为负、翻转精度 0.253，
    见骨架设计 §4.1）。
  - **无 CellMarker**：``lambda_=0`` —— hybrid_annotate 内置退化开关，得分变换为
    恒等，逐位等价纯 pysingle（设计文档 §4 的 sanity check）。同一份
    (ref, query) 分层子集上跑两种条件（identical dataset）。
  - **规模梯度**（分层子采样，min(请求, 可用)）:
      双参考 (ref1=pbmc50k/celltype, ref2=pbmc3k/celltype.l2, query=vkhQ8/celltype):
          (2000,2000,2000) (5000,5000,5000) (10000,10000,10000) (50000,50000,50000)
      单参考 (ref=pbmc50k/celltype, query=vkhQ8/celltype):
          (2000,2000) (5000,5000) (10000,10000) (50000,50000)

指标（粗分类族层面，truth != "Other" 子集，与前序一致的可比粒度）:
  Accuracy / Macro-F1 / ARI / NMI / 混淆矩阵。
  ref 与 query 标签词汇不一致，故用粗族映射；ARI/NMI 对词汇不敏感，另在
  细粒度标签上各算一份作参考。

统计:
  - 每配置: McNemar 配对检验（逐细胞正确/错误，二项精确检验）。
  - 每配置: **翻转质量闸门**（with vs without 逐细胞对照：
    good/bad/both_ok/both_bad + 翻转精度 good/(good+bad)，<0.5 表示该规模
    融合无净价值，见骨架设计 §6.3）。
  - 跨配置: Wilcoxon 符号秩检验（8 个规模点上的 Δ）+ 提升/持平/退化计数。

输出: ``results/experiment/`` 下 summary.json / metrics.csv / deltas.csv /
aggregate.csv / confusion_*.csv / report.md / figures/。

用法:
  python scripts/run_cellmarker_experiment.py [--n-jobs 16] [--sizes 2000,5000,10000,50000]
  python scripts/run_cellmarker_experiment.py --max-configs 2   # 只跑前两个配置（冒烟）
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# CJK 字体：图中标题含中文，DejaVu 缺字形会渲染成方块
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import anndata as ad
import scanpy as sc
from scipy import stats
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from hybridscsinglemarker import hybrid_annotate
from hybridscsinglemarker._coarse_types import coarse_type_series
import pipeline_utils as pu

# --------------------------------------------------------------------------- #
# 常量                                                                          #
# --------------------------------------------------------------------------- #
SPECIES = "Human"
TISSUE = "Peripheral Blood"
DATA_SOURCE = "Experiment"
SEED = 0
LAMBDA_WITH = 0.3          # 有 CellMarker（默认融合强度）
LAMBDA_WITHOUT = 0.0       # 无 CellMarker（退化为纯 pysingle）

REF1 = ("testdata/pbmc50k_refdata.h5ad", "celltype")      # 25 标签
REF2 = ("testdata/pbmc3k_refdata.h5ad", "celltype.l2")    # 26 标签
QUERY = ("testdata/vkhQ8_querydata.h5ad", "celltype")     # 21 标签

SIZES = [2000, 5000, 10000, 50000]
# 大参考自动封顶规则：(n_ref 下限, 每类型封顶数)，按 n_ref 从大到小匹配
# 注意：与 scripts/hybrid_pipeline2.py 中的副本保持一致（改动需同步）
REF_CAP_RULES = ((120_000, 300), (50_000, 500))
# auto scoring 切换阈值：查询规模 > 该值时用 profile（实测 14× 提速）
PROFILE_AT_N = 10_000
FAMILY_ORDER = ["T", "B", "NK", "Mono", "DC", "Other"]
FAMILY_PALETTE = {
    "T": "#4C72B0", "B": "#DD8452", "NK": "#55A868",
    "Mono": "#C44E52", "DC": "#8172B3", "Other": "#CCB974",
}
METRICS = ["accuracy", "macro_f1", "ari", "nmi"]

# --------------------------------------------------------------------------- #
# 数据加载：全量读入内存一次，各规模分层子集按需切片，跨条件/跨配置复用           #
# --------------------------------------------------------------------------- #
_DATA = {}


def _load(path: str) -> ad.AnnData:
    if path not in _DATA:
        print(f"[load] {Path(path).name}: 读入内存 ...", flush=True)
        t0 = time.time()
        _DATA[path] = ad.read_h5ad(path)
        print(f"       {_DATA[path].shape}  {time.time()-t0:.1f}s", flush=True)
    return _DATA[path]


def strat_subset(adata: ad.AnnData, key: str, n: int, seed: int = SEED):
    """按标签 round-robin 分层抽 n 个细胞（返回视图，不拷贝 X）。

    给定 (adata, key, n, seed) 确定唯一细胞集合（seed 只影响内部洗牌顺序），
    保证"同一数据集两种条件/两种配置"用同一批细胞。
    """
    if n is None or n >= adata.n_obs:
        return adata
    groups = (adata.obs[[key]]
              .groupby(key, sort=False)
              .apply(lambda g: list(g.index)).to_dict())
    lab_order = sorted(groups, key=lambda l: -len(groups[l]))
    rng = random.Random(seed)
    for lab in lab_order:
        rng.shuffle(groups[lab])
    li = np.tile(lab_order, (n // len(lab_order) + 1))[:n]
    taken = {l: 0 for l in lab_order}
    out = []
    for lab in li:
        if taken[lab] < len(groups[lab]):
            out.append(groups[lab][taken[lab]])
            taken[lab] += 1
    return adata[out[:n]]


def _subset_cached(path: str, key: str, n: int) -> ad.AnnData:
    """按 (path, key, n) 缓存分层子集（同规模单/双参考共用同一子集）。"""
    cache = getattr(_subset_cached, "_cache", None)
    if cache is None:
        cache = _subset_cached._cache = {}
    tag = (path, key, n)
    if tag not in cache:
        cache[tag] = strat_subset(_load(path), key, n)
    return cache[tag]


def auto_cap_ref(n_ref: int) -> int | None:
    """按参考规模自动封顶：≥120k → 300，≥50k → 500，否则不封顶（None）。"""
    for lo, cap in REF_CAP_RULES:
        if n_ref >= lo:
            return cap
    return None


def resolve_scoring(scoring: str, query_n: int) -> str:
    """解析 scoring：显式值（cells/profile）透传；``"auto"`` 时按查询规模选择。

    依据验证报告 §3：cells+FT 在 5k 精度 0.937（116.6s），profile+FT 0.910
    （8.3s，14×）；全量（58k query）profile 实测 0.913。小规模精度优先用
    cells，大规模速度优先用 profile。
    """
    if scoring != "auto":
        return scoring
    return "profile" if query_n > PROFILE_AT_N else "cells"


def effective_lambda_with(
    n_ref: int,
    *,
    n_refs: int,
    scale_aware: bool,
    lambda_with: float,
) -> float:
    """"with CellMarker" 条件的实际 λ（规模感知策略，骨架设计 §4.1）。

    ``scale_aware=True`` 且单参考 ≥50k 细胞 → 0（实测该规模融合净为负，
    翻转精度 <0.5）；否则保持 ``lambda_with``（默认 0.3）。多参考不受影响。
    """
    if not scale_aware:
        return lambda_with
    return pu.default_lambda(n_ref=n_ref, n_refs=n_refs, method="singler")


def flatten_flip(flip: dict) -> dict:
    """把 flip_stats 结果展平进 delta：``n_good`` → ``flip_n_good``，
    ``flip_precision`` → ``flip_precision``（避免双前缀 ``flip_flip_precision``）。"""
    out = {}
    for k, v in flip.items():
        if k.startswith("flip_"):
            out[k] = v
        else:
            out[f"flip_{k}"] = v
    return out


# --------------------------------------------------------------------------- #
# 指标                                                                          #
# --------------------------------------------------------------------------- #
def _clean_coarse(labels) -> pd.Series:
    """粗族数组 → Series；未映射(None)记为 "Other"。"""
    return pd.Series(np.asarray(labels, dtype=object)).fillna("Other")


def compute_metrics(pred_coarse, truth_coarse, pred_fine, truth_fine) -> dict:
    """粗族层面指标（truth != "Other" 子集）+ 细粒度 ARI/NMI 参考值。"""
    yt_all = _clean_coarse(truth_coarse)
    yp_all = _clean_coarse(pred_coarse)
    m = yt_all != "Other"
    yt = yt_all[m].to_numpy()
    yp = yp_all[m].to_numpy()
    n = int(m.sum())
    if n == 0 or len(np.unique(yt)) < 2:
        return {"n_cells": n}
    labels = sorted(set(yt.tolist()) | set(yp.tolist()))
    return {
        "n_cells": n,
        "accuracy": float((yp == yt).mean()),
        "macro_f1": float(f1_score(yt, yp, average="macro", labels=labels,
                                   zero_division=0)),
        "ari": float(adjusted_rand_score(yt, yp)),
        "nmi": float(normalized_mutual_info_score(yt, yp)),
        "ari_fine": float(adjusted_rand_score(
            np.asarray(truth_fine, dtype=object),
            np.asarray(pred_fine, dtype=object))),
        "nmi_fine": float(normalized_mutual_info_score(
            np.asarray(truth_fine, dtype=object),
            np.asarray(pred_fine, dtype=object))),
        "confusion": confusion_matrix(yt, yp, labels=labels).tolist(),
        "confusion_labels": labels,
    }


def mcnemar(correct_with: np.ndarray, correct_without: np.ndarray) -> float:
    """McNemar 配对检验（逐细胞正确/错误）的精确二项 p 值。"""
    b = int(((correct_with) & (~correct_without)).sum())   # with 对、without 错
    c = int(((~correct_with) & (correct_without)).sum())   # with 错、without 对
    n_disc = b + c
    if n_disc == 0:
        return 1.0
    return float(stats.binomtest(min(b, c), n_disc, 0.5).pvalue)


# --------------------------------------------------------------------------- #
# 单配置运行                                                                      #
# --------------------------------------------------------------------------- #
def run_config(query_sub, refs, celltype_cols, args, tag: str,
               scoring: str, lambda_with_eff: float) -> dict:
    """同一 query/ref 子集上跑 with(λ_eff) 与 without(λ=0)，返回结果+指标。

    ``scoring`` / ``lambda_with_eff`` 由 main 按配置解析（auto scoring +
    规模感知 λ 策略）；with 条件的有效 λ 记录在返回字典供报告/摘要。
    """
    truth_fine = query_sub.obs[QUERY[1]].to_numpy(dtype=object)
    truth_coarse = coarse_type_series(truth_fine)

    cap = args.max_cells_per_type
    if cap is None:
        cap = auto_cap_ref(sum(r.n_obs for r in refs))

    outputs = {}
    for lam, cond in ((lambda_with_eff, "with"), (LAMBDA_WITHOUT, "without")):
        t0 = time.time()
        out = hybrid_annotate(
            query_sub, refs, celltype_col=celltype_cols,
            species=SPECIES, tissue=TISSUE, data_source=DATA_SOURCE,
            method="singler", lambda_=lam, confidence_threshold=0.3,
            n_jobs=args.n_jobs, scoring=scoring,
            fine_tune=True, top_n=5, gene_selection="hvg",
            max_genes=args.max_genes, combine_method="max",
            max_cells_per_type=cap, ref_cap_seed=args.ref_cap_seed,
        )
        pred_fine = out.obs["hybrid_celltype"].to_numpy(dtype=object)
        pred_coarse = coarse_type_series(pred_fine)
        outputs[cond] = {
            "fine": pred_fine, "coarse": pred_coarse,
            "elapsed": time.time() - t0,
        }

    metrics = {
        cond: compute_metrics(
            outputs[cond]["coarse"], truth_coarse,
            outputs[cond]["fine"], truth_fine,
        )
        for cond in ("with", "without")
    }
    delta = {}
    for met in METRICS:
        w, wout = metrics["with"].get(met), metrics["without"].get(met)
        if w is None or wout is None:
            delta[met] = None
            continue
        delta[met] = float(w - wout)
        delta[f"{met}_rel_pct"] = (
            float((w - wout) / wout * 100.0) if wout != 0 else None)

    # McNemar（truth != "Other" 子集上的逐细胞正确性配对）
    yt_all = _clean_coarse(truth_coarse)
    m = yt_all != "Other"
    cw = (_clean_coarse(outputs["with"]["coarse"])[m].to_numpy()
          == yt_all[m].to_numpy())
    cwo = (_clean_coarse(outputs["without"]["coarse"])[m].to_numpy()
           == yt_all[m].to_numpy())
    delta["mcnemar_p"] = mcnemar(cw, cwo)
    delta["n_cells_evaluated"] = int(m.sum())

    # 翻转质量闸门（骨架设计 §6.3）：with vs without（λ=0 孪生）对照。
    # 粗族层面在 truth != "Other" 且两条件均有粗族的细胞上统计。
    mask_fine = ~pd.isna(truth_fine) & ~pd.isna(outputs["without"]["fine"])
    delta["flip_fine"] = pu.flip_stats(
        outputs["without"]["fine"][mask_fine],
        outputs["with"]["fine"][mask_fine],
        truth_fine[mask_fine],
    )
    c_wo = np.asarray(outputs["without"]["coarse"], dtype=object)
    c_w = np.asarray(outputs["with"]["coarse"], dtype=object)
    mask_coarse = m & np.array([x is not None for x in c_wo]) \
        & np.array([x is not None for x in c_w])
    delta["flip_coarse"] = pu.flip_stats(
        c_wo[mask_coarse], c_w[mask_coarse],
        yt_all[mask_coarse].to_numpy(),
    )
    delta["n_flipped"] = int(
        (outputs["with"]["fine"] != outputs["without"]["fine"]).sum()
    )
    # 便于 deltas.csv / report 直接使用（与 dict 形式并存）
    delta.update(flatten_flip(delta["flip_coarse"]))

    return {
        "tag": tag,
        "lambda_with": lambda_with_eff, "lambda_without": LAMBDA_WITHOUT,
        "scoring": scoring,
        "elapsed_s": {c: outputs[c]["elapsed"] for c in ("with", "without")},
        "metrics": metrics,
        "delta": delta,
        # 供可视化与持久化的中间量（main 画完图后再清理）
        "_truth_coarse": truth_coarse,
        "_truth_fine": truth_fine,
        "_pred_coarse": {c: outputs[c]["coarse"] for c in ("with", "without")},
        "_pred_fine": {c: outputs[c]["fine"] for c in ("with", "without")},
        "_confusion": {c: metrics[c]["confusion"]
                       for c in ("with", "without")},
        "_confusion_labels": metrics["with"].get("confusion_labels", []),
    }


# --------------------------------------------------------------------------- #
# 可视化                                                                          #
# --------------------------------------------------------------------------- #
def _umap_embedding(query_sub: ad.AnnData) -> np.ndarray:
    """对查询子集计算 PCA+邻居+UMAP（按规模缓存，单/双参考共用）。"""
    cache = getattr(_umap_embedding, "_cache", None)
    if cache is None:
        cache = _umap_embedding._cache = {}
    key = (query_sub.n_obs, tuple(query_sub.var_names[:5]))
    if key not in cache:
        q = query_sub.copy()
        sc.pp.normalize_total(q, target_sum=1e4)
        sc.pp.log1p(q)
        sc.pp.highly_variable_genes(q, n_top_genes=2000, flavor="seurat")
        sc.pp.pca(q, n_comps=30, use_highly_variable=True, svd_solver="arpack")
        sc.pp.neighbors(q, n_neighbors=15, n_pcs=30)
        sc.tl.umap(q, random_state=SEED)
        cache[key] = q.obsm["X_umap"].copy()
    return cache[key]


def _scatter(ax, umap, colors, title, legend=True):
    xs, ys = umap[:, 0], umap[:, 1]
    c = _clean_coarse(colors)
    for fam in FAMILY_ORDER:
        mask = (c == fam).to_numpy()
        if mask.any():
            ax.scatter(xs[mask], ys[mask], s=6, c=FAMILY_PALETTE[fam],
                       label=fam, alpha=0.8, linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if legend:
        ax.legend(fontsize=7, markerscale=1.5, loc="center left",
                  bbox_to_anchor=(1.02, 0.5), frameon=False)


def plot_umap_side_by_side(query_sub, res, fig_dir: Path, tag: str) -> Path:
    """同一数据集 3 面板对照：truth / without-CM / with-CM（粗族着色）。"""
    umap = _umap_embedding(query_sub)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    _scatter(axes[0], umap, res["_truth_coarse"], "Truth")
    _scatter(axes[1], umap, res["_pred_coarse"]["without"],
             "hybrid_annotate λ=0 (no CellMarker)")
    _scatter(axes[2], umap, res["_pred_coarse"]["with"],
             "hybrid_annotate λ=0.3 (with CellMarker)")
    fig.suptitle(f"{tag} — 粗族注释对照", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = fig_dir / f"01_umap_{tag}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_metric_bars(results, fig_dir: Path) -> list[Path]:
    """每种指标：8 个配置 × (with/without) 分组柱状图。"""
    out = []
    configs = [r["tag"] for r in results]
    for met in METRICS:
        w = [r["metrics"]["with"].get(met) for r in results]
        wout = [r["metrics"]["without"].get(met) for r in results]
        if any(v is None for v in w + wout):
            continue
        x = np.arange(len(configs))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(x - 0.2, wout, 0.4, label="without CellMarker (λ=0)",
               color="#9BB6D8")
        ax.bar(x + 0.2, w, 0.4, label="with CellMarker (λ=0.3)",
               color="#4C72B0")
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(met)
        ax.set_title(f"{met} — with vs without CellMarker across dataset sizes")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p = fig_dir / f"02_metric_{met}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out.append(p)
    return out


def plot_delta(results, fig_dir: Path) -> Path:
    """各配置 accuracy 相对变化 %（with − without），按正负着色。"""
    configs = [r["tag"] for r in results]
    deltas = [r["delta"]["accuracy_rel_pct"] or 0.0 for r in results]
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#C44E52" if d >= 0 else "#55A868" for d in deltas]
    ax.bar(np.arange(len(configs)), deltas, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(configs)))
    ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("relative Δ accuracy (%)")
    ax.set_title("Accuracy 相对变化：with CellMarker vs without (正=提升, 负=退化)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = fig_dir / "03_delta_accuracy.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _size_of(tag: str) -> int:
    return int(next(s for s in tag.split("-") if s.isdigit()))


def plot_size_trend(results, fig_dir: Path) -> Path:
    """单/双参考各自的 accuracy 随规模变化折线（with vs without）。"""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for mode in ("1ref", "2ref"):
        rs = sorted([r for r in results if r["tag"].startswith(mode)],
                    key=lambda r: _size_of(r["tag"]))
        sizes = [_size_of(r["tag"]) for r in rs]
        ax.plot(sizes, [r["metrics"]["with"]["accuracy"] for r in rs],
                marker="o", label=f"{mode} with CellMarker")
        ax.plot(sizes, [r["metrics"]["without"]["accuracy"] for r in rs],
                marker="o", ls="--", label=f"{mode} without CellMarker")
    ax.set_xscale("log")
    ax.set_xticks(SIZES)
    ax.set_xticklabels(SIZES)
    ax.set_xlabel("dataset size (ref = query)")
    ax.set_ylabel("coarse accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = fig_dir / "05_accuracy_size_trend.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# 汇总与报告                                                                      #
# --------------------------------------------------------------------------- #
def aggregate(results) -> dict:
    """跨 8 个配置：逐指标 Δ 的均值/符号一致性 + Wilcoxon 符号秩检验。"""
    agg = {}
    for met in METRICS:
        vals = [r["delta"][met] for r in results if r["delta"].get(met) is not None]
        if not vals:
            continue
        n_imp = sum(v > 0 for v in vals)
        n_eq = sum(v == 0 for v in vals)
        n_deg = sum(v < 0 for v in vals)
        p = 1.0
        if len(vals) > 1 and np.any(np.asarray(vals) != 0):
            p = float(stats.wilcoxon(vals).pvalue)
        agg[met] = {
            "mean_delta": float(np.mean(vals)),
            "n_configs": len(vals),
            "n_improved": n_imp, "n_unchanged": n_eq, "n_degraded": n_deg,
            "wilcoxon_p": p,
        }
    pvals = [r["delta"]["mcnemar_p"] for r in results]
    agg["accuracy_mcnemar_sig_improved"] = sum(
        p < 0.05 and r["delta"]["accuracy"] > 0 for p, r in zip(pvals, results))
    agg["accuracy_mcnemar_sig_degraded"] = sum(
        p < 0.05 and r["delta"]["accuracy"] < 0 for p, r in zip(pvals, results))
    return agg


def _conclusion(acc: dict) -> str:
    mean_d = acc["mean_delta"]
    n_imp, n_deg = acc["n_improved"], acc["n_degraded"]
    p = acc["wilcoxon_p"]
    if mean_d > 0.005 and n_imp >= 6 and p < 0.05:
        return (f"**应当引入 CellMarker**：{n_imp}/8 个规模点提升、退化 {n_deg}/8，"
                f"平均 Δaccuracy = {mean_d:+.4f}，Wilcoxon p = {p:.4f}，方向一致且统计显著。")
    if mean_d > 0.005 and n_imp >= 6:
        return (f"**倾向于引入 CellMarker**：{n_imp}/8 个规模点提升，"
                f"平均 Δaccuracy = {mean_d:+.4f}，但 Wilcoxon p = {p:.4f} 未达显著，"
                f"建议结合混淆矩阵与 UMAP 判断。")
    if n_deg >= n_imp and mean_d < -0.005:
        return (f"**不建议强制引入 CellMarker**：{n_deg}/8 个规模点退化，"
                f"平均 Δaccuracy = {mean_d:+.4f}。")
    return (f"**CellMarker 增益有限/混合**：提升 {n_imp}/8、退化 {n_deg}/8，"
            f"平均 Δaccuracy = {mean_d:+.4f}，Wilcoxon p = {p:.4f}，"
            f"建议按场景选择性使用。")


def write_report(results, agg, out_dir: Path) -> Path:
    """四大分析交付物 → report.md。"""
    L = []
    L.append("# CellMarker 增益实验报告\n")
    L.append("- 日期：2026-08-21")
    L.append("- 统一管线：`hybrid_annotate`（Human / Peripheral Blood / "
             "data_source=Experiment / scoring=auto / fine_tune=True）")
    L.append("- 有 CellMarker：λ=0.3（逐细胞 DB 证据先验融合；规模感知：单参考 "
             "≥50k 自动 λ=0，见骨架设计 §4.1）；"
             "无 CellMarker：λ=0（内置退化，纯 pysingle）")
    L.append("- 翻转精度闸门：good/(good+bad)，<0.5 表示该规模融合无净价值"
             "（骨架设计 §6.3）\n")
    L.append("- 指标粒度：粗分类族（truth != \"Other\" 子集）\n")

    L.append("## 1. 各数据集规模下的相对变化（with vs without CellMarker）\n")
    L.append("| 配置 | scoring | λ_with | Accuracy (with/without) | Δ acc (rel%) | "
             "Macro-F1 Δ | ARI Δ | NMI Δ | 翻转精度 | McNemar p | 判定 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        d = r["delta"]
        a_w = r["metrics"]["with"]["accuracy"]
        a_wo = r["metrics"]["without"]["accuracy"]
        verdict = ("提升" if d["accuracy"] > 0 else
                   "持平" if d["accuracy"] == 0 else "退化")
        rel = f"{d['accuracy_rel_pct']:+.1f}%" if d["accuracy_rel_pct"] is not None else "—"
        fp = d.get("flip_precision")
        fp_s = (
            f"{fp:.3f}"
            if isinstance(fp, (int, float)) and fp == fp
            else "—"
        )
        L.append(f"| {r['tag']} | {r['scoring']} | {r['lambda_with']} | "
                 f"{a_w:.3f} / {a_wo:.3f} | {rel} | "
                 f"{d['macro_f1']:+.3f} | {d['ari']:+.3f} | {d['nmi']:+.3f} | "
                 f"{fp_s} | {d['mcnemar_p']:.4f} | {verdict} |")
    L.append("")

    L.append("## 2. 跨规模聚合结果\n")
    for met in METRICS:
        a = agg.get(met)
        if not a:
            continue
        L.append(f"- **{met}**：平均 Δ = {a['mean_delta']:+.4f}；提升 "
                 f"{a['n_improved']}/{a['n_configs']}、持平 {a['n_unchanged']}、"
                 f"退化 {a['n_degraded']}；Wilcoxon p = {a['wilcoxon_p']:.4f}")
    L.append("- **Accuracy McNemar 检验**（p<0.05）：显著提升 "
             f"{agg['accuracy_mcnemar_sig_improved']} 个配置，显著退化 "
             f"{agg['accuracy_mcnemar_sig_degraded']} 个配置")
    low_fp = [r["tag"] for r in results
              if r["delta"].get("flip_precision") is not None
              and r["delta"]["flip_precision"] < 0.5]
    if low_fp:
        L.append(f"- **翻转精度 <0.5（融合无净价值）的配置**：{', '.join(low_fp)}"
                 "——建议该规模 `--lambda-with 0` 或按需禁用融合")
    L.append("")

    L.append("## 3. UMAP 与混淆矩阵证据\n")
    L.append("- 侧面对照图：`figures/01_umap_<配置>.png`（truth / without-CM / "
             "with-CM 三面板，同一数据集）")
    L.append("- 指标对比：`figures/02_metric_<指标>.png`、`03_delta_accuracy.png`、"
             "`05_accuracy_size_trend.png`")
    L.append("- 混淆矩阵：`figures/04_confusion_<配置>.png`（without / with / "
             "Δcounts 三热图）\n")

    L.append("## 4. 结论\n")
    acc = agg.get("accuracy")
    L.append(_conclusion(acc) if acc else "数据不足，无法下结论。")
    L.append("")
    L.append("*（结论由脚本按聚合统计自动生成，供评审参考。）*")

    p = out_dir / "report.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# 主流程                                                                          #
# --------------------------------------------------------------------------- #
def build_configs(sizes):
    """生成 8 个配置：单/双参考 × 各规模。

    参考以 AnnData 列表传入，标签列名由 ``celltype_col`` 列表指定
    （``(obj, key)`` 元组会被 hybrid_annotate 当作"标签覆盖"而非列名）。
    """
    ref2_max = _load(REF2[0]).n_obs
    cfgs = []
    for n in sizes:
        r2 = min(n, ref2_max)
        cfgs.append((f"2ref-{n}", (n, r2, n),
                     [_subset_cached(*REF1, n), _subset_cached(*REF2, r2)],
                     [REF1[1], REF2[1]]))
        cfgs.append((f"1ref-{n}", (n, n),
                     [_subset_cached(*REF1, n)],
                     [REF1[1]]))
    return cfgs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=str, default="2000,5000,10000,50000")
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--scoring", type=str, default="auto",
                    choices=["auto", "cells", "profile"],
                    help="auto=≤10k 用 cells、>10k 用 profile（默认，见验证报告 §3）；"
                         "可显式指定 cells/profile")
    ap.add_argument("--lambda-with", type=float, default=LAMBDA_WITH,
                    help="with CellMarker 融合强度（默认 0.3）；scale-aware 时"
                         "单参考 ≥50k 会被覆盖为 0")
    ap.add_argument("--no-scale-aware-lambda", dest="scale_aware_lambda",
                    action="store_false", default=True,
                    help="关闭规模感知 λ 策略（单参考 ≥50k 也保持 λ=0.3）")
    ap.add_argument("--max-genes", type=int, default=5000)
    ap.add_argument("--max-cells-per-type", type=int, default=None,
                    help="每参考类型最多细胞数（None=自动，规则见 REF_CAP_RULES）")
    ap.add_argument("--ref-cap-seed", type=int, default=0,
                    help="封顶抽样种子（默认 0，逐位可复现）")
    ap.add_argument("--output-dir", type=str, default="results/experiment")
    ap.add_argument("--no-umap", action="store_true", help="跳过 UMAP 图（省时）")
    ap.add_argument("--max-configs", type=int, default=None,
                    help="只跑前 N 个配置（冒烟测试用，如 --max-configs 2）")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cfgs = build_configs(sizes)
    if args.max_configs is not None:
        cfgs = cfgs[:max(1, args.max_configs)]
        print(f"[limit] 仅跑前 {len(cfgs)} 个配置（--max-configs {args.max_configs}）",
              flush=True)
    print(f"[config] {len(cfgs)} 个配置: "
          + ", ".join(f"{tag}({','.join(map(str, dims))})"
                      for tag, dims, *_ in cfgs), flush=True)

    results, summary = [], {}
    t_start = time.time()
    for tag, dims, refs, celltype_cols in cfgs:
        query_n = dims[0]
        query_sub = _subset_cached(*QUERY, query_n)
        scoring = resolve_scoring(args.scoring, query_n)
        lam_with = effective_lambda_with(
            sum(r.n_obs for r in refs), n_refs=len(refs),
            scale_aware=args.scale_aware_lambda, lambda_with=args.lambda_with,
        )
        print(f"\n[run] {tag}: ref={dims[0]}, query={query_n} "
              f"scoring={scoring} λ_with={lam_with} "
              f"(t+{(time.time()-t_start)/60:.1f}min)", flush=True)
        t0 = time.time()
        res = run_config(query_sub, refs, celltype_cols, args, tag,
                         scoring=scoring, lambda_with_eff=lam_with)
        a_w = res["metrics"]["with"]["accuracy"]
        a_wo = res["metrics"]["without"]["accuracy"]
        rel = res["delta"]["accuracy_rel_pct"]
        fp = res["delta"].get("flip_precision")
        if fp is None:
            print(f"  [warn] {tag}: delta 缺少 flip_precision "
                  f"(keys={sorted(res['delta'])})", flush=True)
        fp_s = (
            f"{fp:.3f}"
            if isinstance(fp, (int, float)) and fp == fp
            else "n/a"
        )
        print(f"  with={a_w:.3f} without={a_wo:.3f} "
              f"Δ={rel:+.1f}% 翻转精度={fp_s} "
              f"翻转={res['delta']['n_flipped']} (t={time.time()-t0:.0f}s)",
              flush=True)

        # 图（需要 res 的私有中间量）
        plot_confusion_heatmaps([res], fig_dir)
        if not args.no_umap:
            try:
                plot_umap_side_by_side(query_sub, res, fig_dir, tag)
            except Exception as e:  # noqa: BLE001 —— UMAP 失败不中断实验
                print(f"  [warn] umap 失败: {e}", flush=True)
        # 混淆矩阵 CSV
        labels = res["_confusion_labels"]
        for cond in ("with", "without"):
            pd.DataFrame(res["_confusion"][cond], index=labels, columns=labels
                         ).to_csv(out_dir / f"confusion_{tag}_{cond}.csv")

        # 逐细胞预测持久化（供图再生成与后续分析；None 粗族写空串）
        pd.DataFrame({
            "barcode": query_sub.obs_names,
            "truth_fine": np.asarray(res["_truth_fine"], dtype=object),
            "truth_coarse": np.asarray(res["_truth_coarse"], dtype=object),
            "pred_fine_with": np.asarray(res["_pred_fine"]["with"], dtype=object),
            "pred_fine_without": np.asarray(res["_pred_fine"]["without"], dtype=object),
            "pred_coarse_with": np.asarray(res["_pred_coarse"]["with"], dtype=object),
            "pred_coarse_without": np.asarray(res["_pred_coarse"]["without"], dtype=object),
        }).to_csv(out_dir / f"predictions_{tag}.csv", index=False)

        # 清理私有字段后持久化
        res_clean = {k: v for k, v in res.items() if not k.startswith("_")}
        results.append(res_clean)
        summary[tag] = {
            "mode": tag.split("-")[0], "dims": list(dims),
            "scoring": res_clean["scoring"],
            "lambda_with_eff": res_clean["lambda_with"],
            "metrics": res_clean["metrics"], "delta": res_clean["delta"],
            "elapsed_s": res_clean["elapsed_s"],
        }
        (out_dir / "summary.json").write_text(
            json.dumps({"params": vars(args), "scenarios": summary},
                       indent=1, ensure_ascii=False), encoding="utf-8")

    # 跨配置聚合 + 指标对比图 + 报告
    agg = aggregate(results)

    metrics_rows = []
    for r in results:
        for cond in ("with", "without"):
            row = {"config": r["tag"], "condition": cond, **r["metrics"][cond]}
            row.pop("confusion", None); row.pop("confusion_labels", None)
            metrics_rows.append(row)
    pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics.csv", index=False)

    delta_rows = [{"config": r["tag"], **r["delta"]} for r in results]
    pd.DataFrame(delta_rows).to_csv(out_dir / "deltas.csv", index=False)

    agg_rows = [{"metric": k, **v} for k, v in agg.items() if isinstance(v, dict)]
    pd.DataFrame(agg_rows).to_csv(out_dir / "aggregate.csv", index=False)

    plot_metric_bars(results, fig_dir)
    plot_delta(results, fig_dir)
    plot_size_trend(results, fig_dir)
    report = write_report(results, agg, out_dir)

    (out_dir / "summary.json").write_text(json.dumps(
        {"params": vars(args), "scenarios": summary, "aggregate": agg},
        indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"\n[aggregate] " + "; ".join(
        f"{k}: Δ={v['mean_delta']:+.4f} ({v['n_improved']}↑/{v['n_unchanged']}=/"
        f"{v['n_degraded']}↓, Wilcoxon p={v['wilcoxon_p']:.3f})"
        for k, v in agg.items() if isinstance(v, dict)))
    print(f"[report] {report}")
    print(f"[figures] {fig_dir}")
    print(f"[done] 总耗时 {(time.time()-t_start)/60:.1f} min")


def plot_confusion_heatmaps(results, fig_dir: Path) -> list[Path]:
    """每配置 3 热图：without / with / Δcounts（with − without）。"""
    out = []
    for r in results:
        labels = r["_confusion_labels"]
        w = np.asarray(r["_confusion"]["with"], dtype=float)
        wo = np.asarray(r["_confusion"]["without"], dtype=float)
        d = w - wo
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        titles = ["without CellMarker", "with CellMarker", "Δ counts (with − without)"]
        for ax, mat, title in zip(axes, (wo, w, d), titles):
            if title.startswith("Δ"):
                vmax = max(abs(d.min()), abs(d.max()), 1e-9)
                im = ax.imshow(mat, cmap="RdBu_r",
                               norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
            else:
                im = ax.imshow(mat, cmap="Blues", aspect="auto")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlabel("predicted"); ax.set_ylabel("truth")
            ax.set_title(f"{title}", fontsize=9)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, int(mat[i, j]), ha="center", va="center",
                            fontsize=6,
                            color="white" if abs(mat[i, j]) > 0.6 * abs(mat).max()
                            else "black")
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"{r['tag']} — 粗族混淆矩阵", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        p = fig_dir / f"04_confusion_{r['tag']}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out.append(p)
    return out


if __name__ == "__main__":
    main()
