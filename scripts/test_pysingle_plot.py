#!/usr/bin/env python3
"""基于 testdata 的可视化功能测试：生成全部 SingleR 风格图形并保存到 ``./figures/``。

对应 R 包 ``SingleR.Plotting.R`` 的绘图函数映射:
  SingleR.DrawHeatmap  -> figures/score_heatmap.png      得分热图（pheatmap 风格）
  SingleR.PlotTsne     -> figures/annotation_scatter.png 注释散点（SINGLER_COLORS 配色）
  SingleR.DrawBoxPlot  -> figures/cell_boxplot.png       单细胞相关分布箱线图
  SingleR.DrawScatter  -> figures/cell_vs_ref_scatter.png 单细胞 vs 参考样本散点

配色与风格对齐说明:
  - SINGLER_COLORS 复刻 R singler.colors（RColorBrewer qual 拼接，剔 4/27，重复三次）；
  - 热图使用 pheatmap 默认 rev(RdYlBu)（低蓝高红）+ ward.D2 聚类 + 逐细胞归一化立方；
  - 箱线图黑色箱体/散点 + 0.8 分位排序；散点图无网格、黑坐标轴。

用法:
  python scripts/test_pysingle_plot.py                    # 默认 query/ref 前 500
  python scripts/test_pysingle_plot.py --query-max 200 --ref-max 300
  python scripts/test_pysingle_plot.py --fig-dir /tmp/figures
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
        description="testdata 可视化功能测试：生成全部 SingleR 风格图形并保存。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--query-max", type=int, default=500, help="查询细胞数（前 N 个）")
    p.add_argument("--ref-max", type=int, default=500, help="参考细胞数（前 N 个）")
    p.add_argument("--scoring", default="cells", choices=["cells", "profile"],
                   help="打分方式：cells=逐细胞相关+top-N；profile=类型中位谱相关")
    p.add_argument("--top-n", type=int, default=5, help="每类型 top-N 相关样本均值")
    p.add_argument("--n-jobs", type=int, default=min(os.cpu_count() or 1, 8),
                   help="并行进程数（首轮分块打分/参考中位数/fine-tuning 组）")
    p.add_argument("--fig-dir", default=str(ROOT / "figures"), help="图形输出目录")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    for path, name in ((REF_PATH, "参考"), (QUERY_PATH, "查询")):
        if not path.exists():
            print(f"[错误] 找不到{name}数据: {path}")
            return 1

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 加载 + 子集采样 ----
    t0 = time.time()
    print("[1/3] 加载参考/查询数据 ...", flush=True)
    ref = ad.read_h5ad(REF_PATH)[: args.ref_max]
    query = ad.read_h5ad(QUERY_PATH)[: args.query_max]
    print(f"      ref {ref.shape} / query {query.shape}（{time.time() - t0:.1f}s）", flush=True)

    # ---- 2. 注释 ----
    print("[2/3] 执行注释 ...", flush=True)
    t1 = time.time()
    result = pysingle.singleR_annotate(
        ref, ref.obs[CELLTYPE_COL], query,
        scoring=args.scoring, top_n=args.top_n, n_jobs=args.n_jobs,
    )
    print(f"      注释完成（{time.time() - t1:.1f}s）", flush=True)

    # ---- 3. 生成全部图形 ----
    print("[3/3] 生成图形 ...", flush=True)
    from hybridscsinglemarker.pysingle import plotting as pl

    # PCA 坐标（用于注释散点，PlotTsne 的 tSNE/UMAP 坐标用 PCA 代替）
    hvg_idx = pysingle.select_hvg(ref.X.T.tocsr(), 5000)
    q_expr = np.asarray(query.X[:, hvg_idx].toarray(), dtype=float)
    from sklearn.decomposition import PCA
    xy = PCA(n_components=2).fit_transform(q_expr)

    sample_cell = query.obs_names[0]
    sample_ref = ref.obs_names[0]

    outputs = {
        "score_heatmap.png": pl.plot_score_heatmap(result),
        "annotation_scatter.png": pl.plot_annotation_scatter(result, xy, dot_size=8),
        "cell_boxplot.png": pl.plot_cell_boxplot(
            query, ref, ref.obs[CELLTYPE_COL], cell_id=sample_cell),
        "cell_vs_ref_scatter.png": pl.plot_cell_vs_ref_scatter(
            query, ref, cell_id=sample_cell, sample_id=sample_ref),
    }

    mapping = [
        ("score_heatmap.png", "SingleR.DrawHeatmap"),
        ("annotation_scatter.png", "SingleR.PlotTsne"),
        ("cell_boxplot.png", "SingleR.DrawBoxPlot"),
        ("cell_vs_ref_scatter.png", "SingleR.DrawScatter"),
    ]
    print("\n" + "=" * 58)
    print(f"可视化测试完成（{time.time() - t0:.1f}s），输出目录: {fig_dir}")
    print("=" * 58)
    for fname, r_func in mapping:
        fig = outputs[fname]
        fig.savefig(fig_dir / fname, dpi=150, bbox_inches="tight")
        print(f"  {fname:<28}  <-  R: {r_func}")
    print("=" * 58)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
