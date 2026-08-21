#!/usr/bin/env python3
"""基于 testdata 的注释功能测试：运行完整注释流程并保存结果。

说明：使用核心函数 ``pysingle.singleR_annotate`` 一次完成注释以同时获得
完整得分矩阵，并复刻 ``pysingle.annotate`` 的 obs 写回行为（新增
``pysingle_celltype`` / ``pysingle_score`` 两列）。``annotate`` 即该流程的
一行封装（加载 + 提取 + 注释 + 写回），此处直接使用核心以获得完整结果。

保存内容（默认根目录下 ``results/``）:
  results/annotation_summary.txt          文本摘要（耗时 / 分布 / 得分统计）
  results/tables/labels.csv               每细胞预测标签与得分
  results/tables/scores.csv               每细胞×类型最终得分矩阵
  results/tables/all_scores.csv           首轮全部类型原始得分矩阵
  results/tables/distribution.csv         预测类型分布（value_counts）
  results/figures/score_heatmap.png       得分热图（pheatmap 风格）
  results/figures/annotation_scatter.png  注释散点（PCA 坐标，PlotTsne 风格）
  results/figures/cell_boxplot.png        单细胞相关分布箱线图（DrawBoxPlot 风格）

用法:
  python scripts/test_pysingle_annotation.py                  # 默认 query/ref 前 500
  python scripts/test_pysingle_annotation.py --query-max 200 --ref-max 300
  python scripts/test_pysingle_annotation.py --ref-random     # 参考随机抽样
  python scripts/test_pysingle_annotation.py --gene-selection all
  python scripts/test_pysingle_annotation.py --out-dir /tmp/results
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # 无头模式

import anndata as ad
import numpy as np
import pandas as pd

import hybridscsinglemarker.pysingle as pysingle

ROOT = Path(__file__).resolve().parent.parent
REF_PATH = ROOT / "testdata" / "pbmc50k_refdata.h5ad"
QUERY_PATH = ROOT / "testdata" / "vkhQ8_querydata.h5ad"
QUERY_ANNOTAED_PATH = ROOT / "testdata" / "vkhQ8_querydata_annotated.h5ad"
CELLTYPE_COL = "celltype"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="testdata 注释功能测试：运行完整注释流程并保存结果。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--query-max", type=int, default=500, help="查询细胞数（前 N 个）")
    p.add_argument("--ref-max", type=int, default=500, help="参考细胞数（前 N 个）")
    p.add_argument("--ref-random", action="store_true", help="参考随机抽样（seed=0）")
    p.add_argument("--gene-selection", default="hvg",
                   choices=["hvg", "de", "sd", "all"], help="首轮基因选择")
    p.add_argument("--max-genes", type=int, default=5000, help="HVG 基因数")
    p.add_argument("--scoring", default="cells", choices=["cells", "profile"],
                   help="打分方式：cells=逐细胞相关+top-N；profile=类型中位谱相关(大参考快)")
    p.add_argument("--top-n", type=int, default=5, help="每类型 top-N 相关样本均值")
    p.add_argument("--n-jobs", type=int, default=min(os.cpu_count() or 1, 8),
                   help="并行进程数（首轮分块打分/参考中位数/fine-tuning 组）")
    p.add_argument("--out-dir", default=str(ROOT / "results"), help="结果输出目录")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    for path, name in ((REF_PATH, "参考"), (QUERY_PATH, "查询")):
        if not path.exists():
            print(f"[错误] 找不到{name}数据: {path}")
            return 1

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 加载 + 子集采样 ----
    t0 = time.time()
    print("[1/4] 加载参考/查询数据 ...", flush=True)
    ref = ad.read_h5ad(REF_PATH)
    query = ad.read_h5ad(QUERY_PATH)
    if args.ref_max < ref.shape[0]:
        ref = pysingle.io.subset_cells(ref, args.ref_max, random=args.ref_random)
    if args.query_max < query.shape[0]:
        query = pysingle.io.subset_cells(query, args.query_max)
    t_load = time.time() - t0
    print(f"      ref {ref.shape} / query {query.shape}（加载 {t_load:.1f}s）", flush=True)

    # ---- 2. 注释（核心一次调用，复刻 annotate 的 obs 写回） ----
    print(f"[2/4] 执行注释（gene_selection={args.gene_selection}, "
          f"top_n={args.top_n}）...", flush=True)
    t1 = time.time()
    result = pysingle.singleR_annotate(
        ref, ref.obs[CELLTYPE_COL], query,
        gene_selection=args.gene_selection, max_genes=args.max_genes,
        scoring=args.scoring,
        top_n=args.top_n, n_jobs=args.n_jobs,
    )
    # 等价于 pysingle.annotate(...) 的 obs 写回行为
    query.obs["pysingle_celltype"] = result["labels"].to_numpy()
    scores = result["scores"]
    col_pos = np.asarray([scores.columns.get_loc(lbl) for lbl in result["labels"]])
    query.obs["pysingle_score"] = scores.to_numpy()[np.arange(len(result["labels"])), col_pos]
    query.write_h5ad(QUERY_ANNOTAED_PATH, compression="gzip")
    t_annotate = time.time() - t1
    elapsed = time.time() - t0

    # ---- 3. 保存表格 ----
    print("[3/4] 保存结果表格 ...", flush=True)
    query.obs[["pysingle_celltype", "pysingle_score"]].to_csv(tab_dir / "labels.csv")
    result["scores"].to_csv(tab_dir / "scores.csv")
    result["all_scores"].to_csv(tab_dir / "all_scores.csv")
    dist = query.obs["pysingle_celltype"].value_counts().rename_axis("celltype")
    dist.to_csv(tab_dir / "distribution.csv", header=["n_cells"])

    # ---- 4. 生成图形 ----
    print("[4/4] 生成图形 ...", flush=True)
    from hybridscsinglemarker.pysingle import plotting as pl

    # PCA 坐标（用于注释散点）
    hvg_idx = pysingle.select_hvg(ref.X.T.tocsr(), args.max_genes)
    q_expr = np.asarray(query.X[:, hvg_idx].toarray(), dtype=float)
    from sklearn.decomposition import PCA
    xy = PCA(n_components=2).fit_transform(q_expr)

    pl.plot_score_heatmap(result).savefig(fig_dir / "score_heatmap.png",
                                          dpi=150, bbox_inches="tight")
    pl.plot_annotation_scatter(result, xy, dot_size=8).savefig(
        fig_dir / "annotation_scatter.png", dpi=150, bbox_inches="tight")
    sample_cell = query.obs_names[0]
    pl.plot_cell_boxplot(query, ref, ref.obs[CELLTYPE_COL], cell_id=sample_cell).savefig(
        fig_dir / "cell_boxplot.png", dpi=150, bbox_inches="tight")

    # ---- 摘要 ----
    lines = [
        "pysingle 注释测试结果摘要",
        "=" * 54,
        f"数据: 参考 {ref.shape} / 查询 {query.shape}",
        f"参数: gene_selection={args.gene_selection}, max_genes={args.max_genes}, "
        f"top_n={args.top_n}",
        f"运行耗时: 加载 {t_load:.1f}s | 注释 {t_annotate:.1f}s | 总耗时 {elapsed:.1f}s",
        f"得分统计: min={query.obs['pysingle_score'].min():.3f}  "
        f"max={query.obs['pysingle_score'].max():.3f}  "
        f"mean={query.obs['pysingle_score'].mean():.3f}",
        "",
        "预测细胞类型分布 (value_counts):",
        dist.to_string(),
        "",
        f"表格目录: {tab_dir}",
        f"图形目录: {fig_dir}",
    ]
    summary = "\n".join(lines)
    (out_dir / "annotation_summary.txt").write_text(summary, encoding="utf-8")

    print("\n" + summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
