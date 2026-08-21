#!/usr/bin/env python
"""inference.py — 基于 hybridscsinglemarker 的推理脚本（查询数据无需真值标签）。

与 ``scripts/hybrid_pipeline2.py`` 使用同一套测试数据与参数（配置在文件顶部
的常量区，**不接受命令行输入**），对无标签查询数据执行融合注释，并输出：

1. **预测 h5ad**：``outputs/predicted.h5ad``（obs 新增 ``hybrid_celltype`` /
   ``hybrid_confidence`` / ``hybrid_status``，uns 含完整中间量）；
2. **证据表**：``outputs/evidence_table.csv``（每参考标签：匹配基因数、DB 细胞
   类型数、top DB 细胞类型与富集 Score）+ 每标签 `score_gene_list` 风格矩阵
   （``outputs/evidence/<label>.csv``）+ 标志基因面板证据
   （``outputs/db_evidence.csv``）；
3. **证据图**：``outputs/figures/db_evidence_heatmap.png``（标志基因 × DB 细胞
   类型支持证据热图）、``outputs/figures/label_evidence_score.png``（标签 ×
   DB 细胞类型富集 Score 概览）、top-K 标签逐标签热图；
4. **预测 UMAP 图**：``outputs/figures/predicted_umap.png``（粗分类族 + 细粒度
   细胞类型两面板，基于查询自带的 X_umap）；
5. **关键指标**：``outputs/metrics.json`` —— 预测覆盖率 / 标签分布 / 置信度
   统计 / 状态分层 / hybrid-DB 粗族一致性（内部一致性，无需真值）/ 平均融合
   强度 / 耗时。

用法::

    python scripts/inference.py          # 从仓库根目录运行，输出到 ./outputs
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # 无头环境，先于 pyplot 导入
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import anndata as ad
from hybridscsinglemarker import hybrid_annotate
from hybridscsinglemarker._coarse_types import coarse_type_series
from hybridscsinglemarker.cellmarkerannot import CellMarkerDB, score_gene_list
from hybridscsinglemarker.cellmarkerannot.plotting import plot_gene_scores
import pipeline_utils as pu

# --------------------------------------------------------------------------- #
# 配置（与 scripts/hybrid_pipeline2.py 一致的参数；在此预先设定，无 CLI）       #
# --------------------------------------------------------------------------- #
SPECIES = "Human"
TISSUE = "Peripheral Blood"               # 与 hybrid_pipeline2.py 一致
DATA_SOURCE = "Experiment"                # 仅 Experiment 来源证据

# 测试数据（与 hybrid_pipeline2.py 相同）
REF1 = (REPO_ROOT / "testdata" / "pbmc50k_refdata.h5ad", "celltype")
REF2 = (REPO_ROOT / "testdata" / "pbmc3k_refdata.h5ad", "celltype.l2")
QUERY = REPO_ROOT / "testdata" / "vkhQ8_querydata.h5ad"

# 融合 / 打分参数
METHOD = "singler"                        # "singler" / "seurat"
LAMBDA_ = 0.3                             # 融合强度（多参考场景）
FUSION_MODE = "v_full"                    # "v_full" / "family_posterior"
SCORING = "auto"                          # auto: ≤10k → cells，>10k → profile
MAX_GENES = 5000
MAX_CELLS_PER_TYPE: int | None = None     # None=自动（≥120k→300，≥50k→500）
N_JOBS = 32
FINE_TUNE = True
COMBINE_METHOD = "max"
CONFIDENCE_THRESHOLD = 0.3
EXPR_THRESHOLD = 1.0
FEATURE_GENE_TOP_N = 200
INCLUDE_LABEL_EVIDENCE = True
TOP_K_LABEL_HEATMAPS = 12                 # 逐标签热图数量（按预测频次 top-K）

# 输出目录（默认 ./outputs，即从仓库根目录运行时的 <repo>/outputs）
OUTPUT_DIR = Path("outputs")

# 与 hybrid_pipeline2.py 一致的 PBMC 标志基因面板（用于 DB 证据热图）
DEFAULT_PBMC_GENES = (
    "CD3D", "CD4", "CD8A", "MS4A1", "CD19", "NKG7",
    "GNLY", "NCAM1", "CD14", "FCGR3A", "FCER1A", "CLEC9A",
)
FAMILY_ORDER = ["T", "B", "NK", "Mono", "DC", "Other"]
FAMILY_COLORS = {
    "T": "#4C72B0", "B": "#DD8452", "NK": "#55A868",
    "Mono": "#C44E52", "DC": "#8172B3", "Other": "#999999",
}
STATUS_ORDER = ["consistent", "low_confidence", "unknown", "db_only"]


# --------------------------------------------------------------------------- #
# 纯函数（可单测）                                                              #
# --------------------------------------------------------------------------- #


def evidence_table_from_mats(label_evidence: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """把每标签证据矩阵汇总为一张表（标签 × 关键统计）。

    每标签矩阵为 ``score_gene_list`` 风格：行 = 特征基因 + 最后一行 ``Score``，
    列 = DB 细胞类型。汇总列：匹配基因数、DB 细胞类型数、top 得分对应的
    DB 细胞类型、top/mean Score；按 top_score 降序。
    """
    rows = []
    for lab, mat in label_evidence.items():
        if mat is None or len(mat) == 0:
            continue
        srow = mat.loc["Score"]
        rows.append({
            "label": lab,
            "n_matched_genes": int(len(mat) - 1),
            "n_db_cell_types": int(mat.shape[1]),
            "top_db_celltype": str(srow.idxmax()),
            "top_score": float(srow.max()),
            "mean_score": float(srow.mean()),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("top_score", ascending=False).reset_index(drop=True)
    return out


def inference_metrics(adata: ad.AnnData,
                      confidence_threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    """无真值标签的推理指标（全部来自注释输出本身）。

    - 覆盖率 / 预测标签分布 / 置信度统计 / 状态分层；
    - ``db_consistency_agreement``：hybrid 预测粗族 vs DB 预测粗族的**内部
      一致性**（非真值指标，用于交叉校验）；
    - ``lambda_eff_mean``：实际施加的平均融合强度（uns 诊断量）。
    """
    obs = adata.obs
    n = int(len(obs))
    pred = obs["hybrid_celltype"]
    valid = ~pred.isna()
    conf = obs["hybrid_confidence"].to_numpy(dtype=float)
    hs = adata.uns.get("hybridsc", {})

    cr = np.asarray(hs.get("coarse_ref"), dtype=object) if "coarse_ref" in hs else None
    cd = np.asarray(hs.get("coarse_db"), dtype=object) if "coarse_db" in hs else None
    cons = None
    if cr is not None and cd is not None and len(cr) == n and len(cd) == n:
        ok = np.array([x is not None for x in cr]) \
            & np.array([x is not None for x in cd])
        if int(ok.sum()):
            cons = float((cr[ok] == cd[ok]).mean())

    lam = hs.get("lambda_eff_cell")
    return {
        "n_cells": n,
        "n_predicted": int(valid.sum()),
        "coverage": float(valid.mean()),
        "top_predictions": {
            str(k): int(v) for k, v in pred.value_counts().head(10).items()
        },
        "confidence_mean": float(np.nanmean(conf)) if n else None,
        "confidence_median": float(np.nanmedian(conf)) if n else None,
        "confidence_ge_threshold": float((conf >= confidence_threshold).mean()),
        "status_counts": obs["hybrid_status"].value_counts().to_dict(),
        "db_consistency_agreement": cons,
        "lambda_eff_mean": (
            float(np.mean(lam)) if lam is not None and len(lam) else None
        ),
    }


def h5ad_safe_hybridsc(hs: dict) -> dict:
    """把 ``uns["hybridsc"]`` 转成 h5ad 可序列化形式。

    DB-only 路径（无表达 marker 的细胞）会在 ``db_celltype`` 等 object 数组里
    留下 ``None`` / ``NaN``，h5py 写 vlen 字符串时会报
    "Can't implicitly convert non-string objects to strings"。这里把
    ``None`` / ``NaN`` 归一化为 ``""``，并剔除值为 ``None`` 的诊断键。
    """
    out = dict(hs)
    for key in ("db_celltype", "coarse_ref", "coarse_db", "chosen_family"):
        arr = out.get(key)
        if arr is None:
            out.pop(key, None)
            continue
        out[key] = np.array(
            [
                "" if x is None or (isinstance(x, float) and np.isnan(x))
                else str(x)
                for x in np.asarray(arr, dtype=object)
            ],
            dtype=object,
        )
    for key in ("lambda_eff_cell",):
        if out.get(key) is None:
            out.pop(key, None)
    return out


# --------------------------------------------------------------------------- #
# 可视化                                                                       #
# --------------------------------------------------------------------------- #


def plot_predicted_umap(adata: ad.AnnData, save_path: Path) -> None:
    """预测 UMAP：粗分类族 + 细粒度细胞类型两面板（用查询自带 X_umap）。"""
    if "X_umap" not in adata.obsm:
        print("[warn] 查询数据无 X_umap，跳过预测 UMAP 图")
        return
    umap = np.asarray(adata.obsm["X_umap"])
    pred = adata.obs["hybrid_celltype"].astype(str).to_numpy(dtype=object)
    coarse = np.asarray(coarse_type_series(pred), dtype=object)

    # 细粒度：按预测频次 top-15 上色，其余归 "Other"
    counts = pd.Series(pred).value_counts()
    top15 = counts.head(15).index.tolist()
    fine = np.array([x if x in top15 else "Other" for x in pred], dtype=object)
    fine_palette = {
        lab: plt.cm.tab20(i % 20) for i, lab in enumerate(sorted(set(top15)))
    }
    fine_palette["Other"] = "#cccccc"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, values, palette, title in (
        (axes[0], coarse, FAMILY_COLORS, "predicted coarse family"),
        (axes[1], fine, fine_palette, "predicted cell type (top-15)"),
    ):
        seen = []
        for fam in dict.fromkeys(values):
            mask = values == fam
            color = palette.get(fam, "#cccccc")
            ax.scatter(umap[mask, 0], umap[mask, 1], s=6, color=color,
                       label=str(fam), alpha=0.8, linewidths=0)
            seen.append(fam)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.legend(fontsize=6, markerscale=1.5, loc="center left",
                  bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle(f"Predicted annotation UMAP ({SPECIES} / {TISSUE} / "
                 f"{DATA_SOURCE})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_label_evidence(label_evidence: dict[str, pd.DataFrame],
                        out_dir: Path, fig_dir: Path,
                        pred_counts, top_k: int) -> None:
    """保存每标签证据矩阵 CSV + 逐标签热图 + 标签 × DB 细胞类型 Score 概览。"""
    ev_dir = out_dir / "evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)
    fig_sub = fig_dir / "label_evidence"
    fig_sub.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        label_evidence,
        key=lambda lab: (-pred_counts.get(lab, 0), lab),
    )
    topk = set(ranked[:top_k])
    for lab, mat in label_evidence.items():
        if mat is None or len(mat) == 0:
            continue
        mat.to_csv(ev_dir / f"{lab}.csv")
        if lab not in topk:
            continue
        try:
            fig = plot_gene_scores(
                mat,
                title=f"{lab} — CellMarker evidence "
                      f"({SPECIES}/{TISSUE}/{DATA_SOURCE})",
                save_path=fig_sub / f"{lab}.png",
            )
            plt.close(fig)
        except ValueError as exc:
            print(f"[warn] 标签 {lab} 热图跳过: {exc}")

    # 概览：标签 × DB 细胞类型富集 Score
    col_union = sorted({
        c for m in label_evidence.values()
        if m is not None and len(m)
        for c in m.columns
    })
    if not col_union:
        return
    rows, row_names = [], []
    for lab in ranked:
        m = label_evidence[lab]
        if m is None or len(m) == 0:
            continue
        rows.append(m.loc["Score"].reindex(col_union,
                                           fill_value=0.0).to_numpy(dtype=float))
        row_names.append(lab)
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(
        max(8.0, 0.22 * len(col_union)), max(4.0, 0.30 * len(rows)),
    ))
    im = ax.imshow(np.vstack(rows), aspect="auto", cmap="Purples_r")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(row_names, fontsize=8)
    ax.set_xticks(range(len(col_union)))
    ax.set_xticklabels(col_union, rotation=90, fontsize=6)
    ax.set_title(f"Label × DB cell-type enrichment Score "
                 f"({SPECIES}/{TISSUE}/{DATA_SOURCE})")
    fig.colorbar(im, ax=ax, label="Score")
    fig.tight_layout()
    fig.savefig(fig_dir / "label_evidence_score.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 主流程                                                                       #
# --------------------------------------------------------------------------- #


def main() -> None:
    t0 = time.time()
    out_dir = OUTPUT_DIR
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HybridscSingleMarker 推理（查询数据无需真值标签）")
    print(f"  query={QUERY.name}  refs={[r[0].name for r in (REF1, REF2)]}")
    print(f"  species={SPECIES}  tissue={TISSUE}  data_source={DATA_SOURCE}")
    print(f"  融合模式={FUSION_MODE}  λ={LAMBDA_}  scoring={SCORING}")
    print("=" * 70)

    # 查询/参考规模（backed 元数据读取，内存友好）→ 解析 auto 参数
    query_n = ad.read_h5ad(QUERY, backed="r").n_obs
    ref_n = sum(ad.read_h5ad(p, backed="r").n_obs for p, _ in (REF1, REF2))
    scoring = "profile" if SCORING == "auto" and query_n > 10_000 else SCORING
    cap = MAX_CELLS_PER_TYPE if MAX_CELLS_PER_TYPE is not None \
        else pu.auto_cap_for_ref(ref_n)
    print(f"[config] query={query_n} cells, ref={ref_n} cells, "
          f"scoring={scoring}, cap={cap}, n_jobs={N_JOBS}")

    # 1. 融合注释（多参考；无任何真值标签参与）
    adata = hybrid_annotate(
        QUERY,
        ref=[REF1[0], REF2[0]],
        celltype_col=[REF1[1], REF2[1]],
        species=SPECIES, tissue=TISSUE, method=METHOD,
        lambda_=LAMBDA_, data_source=DATA_SOURCE,
        family_posterior=(FUSION_MODE == "family_posterior"),
        include_label_evidence=INCLUDE_LABEL_EVIDENCE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        expr_threshold=EXPR_THRESHOLD,
        feature_gene_top_n=FEATURE_GENE_TOP_N,
        n_jobs=N_JOBS, scoring=scoring, top_n=5,
        fine_tune=FINE_TUNE, gene_selection="hvg",
        max_genes=MAX_GENES, combine_method=COMBINE_METHOD,
        max_cells_per_type=cap, ref_cap_seed=0,
    )
    print(f"[annotate] 完成（{time.time() - t0:.0f}s）")

    # 2. 关键指标（无真值；先于 uns 清洗计算，保留 None 语义）
    metrics = inference_metrics(adata)
    metrics["elapsed_s"] = round(time.time() - t0, 2)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[metrics] 覆盖率={metrics['coverage']:.3f}  "
          f"置信度均值={metrics['confidence_mean']:.3f}  "
          f"DB一致性={metrics['db_consistency_agreement']:.3f}")
    print("  状态: " + ", ".join(
        f"{k}={v}" for k, v in metrics["status_counts"].items()))

    # 3. 保存预测 h5ad（uns 先做 h5ad 安全化）
    adata.uns["hybridsc"] = h5ad_safe_hybridsc(adata.uns["hybridsc"])
    h5_path = out_dir / "predicted.h5ad"
    adata.write_h5ad(h5_path, compression="gzip")
    print(f"[h5ad] 已保存 -> {h5_path}")

    # 4. 证据表 + 证据图
    hs = adata.uns["hybridsc"]
    label_evidence = hs.get("label_evidence") or {}
    ev_table = evidence_table_from_mats(label_evidence)
    if len(ev_table):
        ev_table.to_csv(out_dir / "evidence_table.csv", index=False)
        print(f"[evidence] 证据表 {len(ev_table)} 个标签 -> evidence_table.csv")
    pred_counts = adata.obs["hybrid_celltype"].value_counts()
    plot_label_evidence(label_evidence, out_dir, fig_dir, pred_counts,
                        TOP_K_LABEL_HEATMAPS)

    db = CellMarkerDB(dataset="all_cell_marker")
    ev = score_gene_list(DEFAULT_PBMC_GENES, db, species=SPECIES,
                         tissue=TISSUE, data_source=DATA_SOURCE)
    ev.to_csv(out_dir / "db_evidence.csv")
    fig = plot_gene_scores(
        ev,
        title=f"CellMarker evidence: PBMC marker genes "
              f"({SPECIES}/{TISSUE}/{DATA_SOURCE})",
        save_path=fig_dir / "db_evidence_heatmap.png",
    )
    plt.close(fig)

    # 5. 预测 UMAP
    plot_predicted_umap(adata, fig_dir / "predicted_umap.png")

    print(f"\n[figures] {fig_dir}")
    for p in sorted(fig_dir.glob("*.png")):
        print(f"  - {p.name}")
    print(f"\n[outputs] {out_dir}")
    for p in sorted(out_dir.glob("*.csv")) + sorted(out_dir.glob("*.json")) \
            + sorted(out_dir.glob("*.h5ad")):
        print(f"  - {p.name} ({p.stat().st_size / 1e6:.1f} MB)")
    print(f"总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
