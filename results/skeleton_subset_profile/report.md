# HybridscSingleMarker 端到端验证报告

- 日期: 2026-08-21 14:03:49
- 物种/组织: Human / Peripheral Blood  data_source: Experiment
- ref1: pbmc50k_refdata.h5ad (celltype, 2000 细胞)
- ref2: pbmc3k_refdata.h5ad (celltype.l2, 2000 细胞)
- query: vkhQ8_querydata.h5ad (celltype, 1500 细胞)
- 融合强度: hybrid_1ref λ=0.3, hybrid_2ref λ=0.3
- 融合模式: v_full
- 每标签注释矩阵: True
- 总耗时: 92.47 s

## 各场景指标（粗族一致率 / 细粒度 ARI·NMI·稀有 Macro-F1）

| 场景 | agreement | fine ARI | fine NMI | rare F1 | 耗时(s) |
|---|---|---|---|---|---|
| pysingle 1ref | 0.8933 | 0.2576 | 0.5990 | 0.2891 | 3.91 |
| hybrid 1ref | 0.8918 | 0.2394 | 0.5965 | 0.2815 | 16.87 |
| pysingle 2ref | 0.8184 | 0.2541 | 0.5486 | 0.2589 | 9.81 |
| hybrid 2ref | 0.8221 | 0.2483 | 0.5443 | 0.2577 | 21.43 |
| cellmarkerannot only | 0.2246 | 0.1022 | 0.3322 | 0.0000 | 2.43 |

## 翻转质量闸门（hybrid vs λ=0 孪生）

| 场景 | 翻转数 | good | bad | both_ok | both_bad | 翻转精度(细) | 翻转精度(粗) |
|---|---|---|---|---|---|---|---|
| hybrid 1ref | 126 | 15 | 111 | 399 | 966 | 0.119 | 0.422 |
| hybrid 2ref | 280 | 54 | 226 | 331 | 880 | 0.193 | 0.440 |

> 翻转精度 = good/(good+bad)；< 0.5 表示该规模下融合无净价值（设计 §6.3）。

## A/B: data_source='all' vs 'Experiment'（子集档）

- Experiment 一致率: 0.8918
- all 一致率: 0.8621
- Δ一致率 (all − Experiment): -0.0297
- 翻转精度 (fine): 0.045
- 翻转数: 88

## 每标签注释矩阵与热图

- 共保存 71 张每标签 score_gene_list 风格注释矩阵（CSV）及对应热图，见 `results/label_evidence_<scn>/` 与 `results/figures/label_evidence_<scn>/`、`09_label_evidence_score_<scn>.png`。
