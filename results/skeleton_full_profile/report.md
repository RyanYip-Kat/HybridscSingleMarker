# HybridscSingleMarker 端到端验证报告

- 日期: 2026-08-21 15:00:50
- 物种/组织: Human / Peripheral Blood  data_source: Experiment
- ref1: pbmc50k_refdata.h5ad (celltype, 全量)
- ref2: pbmc3k_refdata.h5ad (celltype.l2, 全量)
- query: vkhQ8_querydata.h5ad (celltype, 全量)
- 融合强度: hybrid_1ref λ=0.0, hybrid_2ref λ=0.3
- 融合模式: v_full
- 每标签注释矩阵: True
- 总耗时: 464.12 s

## 各场景指标（粗族一致率 / 细粒度 ARI·NMI·稀有 Macro-F1）

| 场景 | agreement | fine ARI | fine NMI | rare F1 | 耗时(s) |
|---|---|---|---|---|---|
| pysingle 1ref | 0.9129 | 0.5851 | 0.6574 | 0.2941 | 31.27 |
| hybrid 1ref | 0.9129 | 0.5851 | 0.6574 | 0.2941 | 27.91 |
| pysingle 2ref | 0.9008 | 0.5104 | 0.6001 | 0.2585 | 127.06 |
| hybrid 2ref | 0.9044 | 0.4596 | 0.5947 | 0.2485 | 179.43 |
| cellmarkerannot only | 0.4327 | 0.4571 | 0.4791 | 0.0000 | 5.54 |

## 翻转质量闸门（hybrid vs λ=0 孪生）

| 场景 | 翻转数 | good | bad | both_ok | both_bad | 翻转精度(细) | 翻转精度(粗) |
|---|---|---|---|---|---|---|---|
| hybrid 1ref | 0 | 0 | 0 | 31729 | 26948 | nan | nan |
| hybrid 2ref | 8352 | 1900 | 6452 | 26597 | 23728 | 0.227 | 0.490 |

> 翻转精度 = good/(good+bad)；< 0.5 表示该规模下融合无净价值（设计 §6.3）。

## 每标签注释矩阵与热图

- 共保存 71 张每标签 score_gene_list 风格注释矩阵（CSV）及对应热图，见 `results/label_evidence_<scn>/` 与 `results/figures/label_evidence_<scn>/`、`09_label_evidence_score_<scn>.png`。
