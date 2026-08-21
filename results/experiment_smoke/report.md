# CellMarker 增益实验报告

- 日期：2026-08-21
- 统一管线：`hybrid_annotate`（Human / Peripheral Blood / data_source=Experiment / scoring=auto / fine_tune=True）
- 有 CellMarker：λ=0.3（逐细胞 DB 证据先验融合；规模感知：单参考 ≥50k 自动 λ=0，见骨架设计 §4.1）；无 CellMarker：λ=0（内置退化，纯 pysingle）
- 翻转精度闸门：good/(good+bad)，<0.5 表示该规模融合无净价值（骨架设计 §6.3）

- 指标粒度：粗分类族（truth != "Other" 子集）

## 1. 各数据集规模下的相对变化（with vs without CellMarker）

| 配置 | scoring | λ_with | Accuracy (with/without) | Δ acc (rel%) | Macro-F1 Δ | ARI Δ | NMI Δ | 翻转精度 | McNemar p | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2ref-2000 | cells | 0.3 | 0.876 / 0.854 | +2.6% | +0.010 | +0.059 | +0.033 | 0.637 | 0.0006 | 提升 |
| 1ref-2000 | cells | 0.3 | 0.949 / 0.941 | +0.8% | +0.006 | +0.014 | +0.014 | 0.750 | 0.0043 | 提升 |

## 2. 跨规模聚合结果

- **accuracy**：平均 Δ = +0.0149；提升 2/2、持平 0、退化 0；Wilcoxon p = 0.5000
- **macro_f1**：平均 Δ = +0.0080；提升 2/2、持平 0、退化 0；Wilcoxon p = 0.5000
- **ari**：平均 Δ = +0.0363；提升 2/2、持平 0、退化 0；Wilcoxon p = 0.5000
- **nmi**：平均 Δ = +0.0236；提升 2/2、持平 0、退化 0；Wilcoxon p = 0.5000
- **Accuracy McNemar 检验**（p<0.05）：显著提升 2 个配置，显著退化 0 个配置

## 3. UMAP 与混淆矩阵证据

- 侧面对照图：`figures/01_umap_<配置>.png`（truth / without-CM / with-CM 三面板，同一数据集）
- 指标对比：`figures/02_metric_<指标>.png`、`03_delta_accuracy.png`、`05_accuracy_size_trend.png`
- 混淆矩阵：`figures/04_confusion_<配置>.png`（without / with / Δcounts 三热图）

## 4. 结论

**CellMarker 增益有限/混合**：提升 2/8、退化 0/8，平均 Δaccuracy = +0.0149，Wilcoxon p = 0.5000，建议按场景选择性使用。

*（结论由脚本按聚合统计自动生成，供评审参考。）*