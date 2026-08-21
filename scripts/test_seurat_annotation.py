#!/usr/bin/env python3
"""基于 testdata 的 Seurat 参考映射标签转移注释功能测试：运行并保存结果。

方法对应 Seurat::FindTransferAnchors + TransferData（vignette:
integration_mapping.Rmd），pysingle 中以纯 Python（numpy/scipy/sklearn）实现
（``pysingle.seurat_method.seurat_annotate``）。

参考数据集:
  testdata/pbmc50k_refdata.h5ad  细胞类型列 celltype
查询数据集:
  testdata/vkhQ8_querydata.h5ad

保存内容（默认根目录下 ``results/``）:
  results/seurat_annotation_summary.txt  文本摘要（耗时 / 分布 / 得分统计）
  results/tables/seurat_labels.csv       每细胞预测标签与得分
  results/tables/seurat_scores.csv       每细胞×标签预测得分矩阵
  results/tables/seurat_distribution.csv 预测类型分布
  results/figures/seurat_annotation_scatter.png  注释散点（PCA 坐标）

用法:
  python scripts/test_seurat_annotation.py                  # 默认 query/ref 前 500
  python scripts/test_seurat_annotation.py --query-max 1000 --ref-max 1000
  python scripts/test_seurat_annotation.py --reduction pcaproject
  python scripts/test_seurat_annotation.py --n-dims 50 --k-anchor 10
  python scripts/test_seurat_annotation.py --out-dir /tmp/results
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

import hybridscsinglemarker.pysingle as pysingle

ROOT = Path(__file__).resolve().parent.parent
REF_PATH = ROOT / "testdata" / "pbmc50k_refdata.h5ad"
QUERY_PATH = ROOT / "testdata" / "vkhQ8_querydata.h5ad"
CELLTYPE_COL = "celltype"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="testdata Seurat 标签转移注释功能测试：运行并保存结果。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--query-max", type=int, default=500, help="查询细胞数（前 N 个）")
    p.add_argument("--ref-max", type=int, default=500, help="参考细胞数（前 N 个）")
    p.add_argument("--ref-random", action="store_true", help="参考随机抽样（seed=0）")
    p.add_argument("--reduction", default="cca", choices=["cca", "pcaproject"],
                   help="低维嵌入：cca（经典）/ pcaproject（参考 PCA 投影）")
    p.add_argument("--max-genes", type=int, default=2000, help="共享可变基因数（HVG）")
    p.add_argument("--n-dims", type=int, default=30, help="嵌入维度数（dims）")
    p.add_argument("--k-anchor", type=int, default=5, help="MNN anchors 近邻数（k.anchor）")
    p.add_argument("--k-weight", type=int, default=50, help="权重近邻数（k.weight）")
    p.add_argument("--n-jobs", type=int, default=min(os.cpu_count() or 1, 8),
                   help="并行进程数（kNN 等）")
    p.add_argument("--out-dir", default=str(ROOT / "results"), help="结果输出目录")
    p.add_argument("--out-h5ad", type=str, default=None,
                   help="可选：将注释结果（含 obs 写回）另存为 h5ad 文件")
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
    print("[1/3] 加载参考/查询数据 ...", flush=True)
    ref = ad.read_h5ad(REF_PATH)
    query = ad.read_h5ad(QUERY_PATH)
    ref = pysingle.io.subset_cells(ref, args.ref_max, random=args.ref_random)
    query = pysingle.io.subset_cells(query, args.query_max)
    t_load = time.time() - t0
    print(f"      ref {ref.shape} / query {query.shape}（加载 {t_load:.1f}s）", flush=True)

    # ---- 2. Seurat 标签转移注释 ----
    print(f"[2/3] Seurat 注释（reduction={args.reduction}, n_dims={args.n_dims}, "
          f"k_anchor={args.k_anchor}）...", flush=True)
    t1 = time.time()
    result = pysingle.seurat_annotate(
        ref, ref.obs[CELLTYPE_COL], query,
        reduction=args.reduction, max_genes=args.max_genes,
        n_dims=args.n_dims, k_anchor=args.k_anchor, k_weight=args.k_weight,
        n_jobs=args.n_jobs,
    )
    # 等价于 pysingle.annotate(ref, query, method="seurat", ...) 的 obs 写回行为
    query.obs["pysingle_celltype"] = result["labels"].to_numpy()
    scores = result["scores"]
    col_pos = np.asarray([scores.columns.get_loc(lbl) for lbl in result["labels"]])
    query.obs["pysingle_score"] = scores.to_numpy()[np.arange(len(result["labels"])), col_pos]
    t_annotate = time.time() - t1
    elapsed = time.time() - t0

    # ---- 3. 保存表格 + 图形 + 摘要 ----
    print("[3/3] 保存结果 ...", flush=True)
    query.obs[["pysingle_celltype", "pysingle_score"]].to_csv(tab_dir / "seurat_labels.csv")
    result["scores"].to_csv(tab_dir / "seurat_scores.csv")
    dist = query.obs["pysingle_celltype"].value_counts().rename_axis("celltype")
    dist.to_csv(tab_dir / "seurat_distribution.csv", header=["n_cells"])

    from hybridscsinglemarker.pysingle import plotting as pl

    hvg_idx = pysingle.select_hvg(ref.X.T.tocsr(), args.max_genes)
    q_expr = np.asarray(query.X[:, hvg_idx].toarray(), dtype=float)
    from sklearn.decomposition import PCA
    xy = PCA(n_components=2).fit_transform(q_expr)
    pl.plot_annotation_scatter(result, xy, dot_size=8).savefig(
        fig_dir / "seurat_annotation_scatter.png", dpi=150, bbox_inches="tight")

    lines = [
        "pysingle Seurat 标签转移注释测试结果摘要",
        "=" * 58,
        f"数据: 参考 {ref.shape} / 查询 {query.shape}",
        f"参数: reduction={args.reduction}, max_genes={args.max_genes}, "
        f"n_dims={args.n_dims}, k_anchor={args.k_anchor}, k_weight={args.k_weight}",
        f"运行耗时: 加载 {t_load:.1f}s | 注释 {t_annotate:.1f}s | 总耗时 {elapsed:.1f}s",
        f"找到 anchors: {len(result['anchors'])}",
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
    (out_dir / "seurat_annotation_summary.txt").write_text(summary, encoding="utf-8")

    if args.out_h5ad:
        query.write_h5ad(args.out_h5ad, compression="gzip")
        print(f"注释结果已保存: {args.out_h5ad}")

    print("\n" + summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
