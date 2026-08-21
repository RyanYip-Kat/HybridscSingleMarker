# HybridscSingleMarker 方法与算法说明（Methods）

HybridscSingleMarker 是一个**混合单细胞类型注释**框架：把两条互补的证据链——
带标签参考数据的**表达相关证据**（pysingle：SingleR / Seurat 标签转移）与
CellMarker 3.0 数据库的**文献/实验 marker 证据**（cellmarkerannot）——在
统一的 `hybrid_annotate()` 入口内融合，输出逐细胞注释、置信度与一致性状态。
本文件说明其逻辑架构、核心算法（含伪代码与流程图）与创新点。

## 1. 总体架构

```mermaid
flowchart TD
    Q[query: AnnData / h5ad<br/>细胞 × 基因] --> H{hybrid_annotate}
    R[ref: AnnData 列表或 None<br/>带标签参考] --> H
    H -->|ref 为空| DB[cellmarkerannot<br/>纯 DB 注释<br/>status=db_only]
    H -->|ref 非空| PY[pysingle 首轮得分 S0<br/>cells 或 profile 打分]
    PY --> FU[DB 逐细胞先验 P[cell,c]<br/>乘法融合 F = S0 × (1 + λ_eff·P̂)]
    FU --> FT[fine-tuning 微调<br/>（在融合后的首轮得分上精修）]
    FT --> VA[逐细胞 DB 验证 V<br/>+ 粗族一致性分层]
    VA --> OUT[obs: hybrid_celltype<br/>hybrid_confidence<br/>hybrid_status]
    DB --> OUT
```

### 1.1 模块职责

| 模块 | 职责 |
|---|---|
| `hybrid.py` | `hybrid_annotate()` 主入口：输入归一化、无参考降级路由、融合编排、输出写回 |
| `pysingle/` | SingleR 移植（相关打分 + fine-tuning + 多参考合并）与 Seurat 标签转移 |
| `cellmarkerannot/` | CellMarker 3.0 数据库解析、三种 marker 打分、基因列表富集 |
| `_fusion.py` | 每参考标签特征基因、逐细胞 DB 先验、乘法融合、Variant B 族级后验、每标签证据矩阵 |
| `_coarse_types.py` | 参考标签 / DB 细胞类型 → 粗分类族（T/B/NK/Mono/DC…）关键词映射 |
| `_validate.py` | 逐细胞 DB marker 验证 `V[cell] ∈ [0,1]` |
| `_layer.py` | `consistent` / `low_confidence` / `unknown` / `db_only` 状态判定 |

## 2. 底层引擎算法

### 2.1 pysingle（SingleR 语义）

对应 R 包 SingleR 的主流程：

1. **基因交集与质量过滤**：基因名转小写、取交集（按参考顺序）；剔除参考或
   查询含 NaN、或**参考**全零表达的基因行；
2. **基因选择**：`hvg`（Seurat vst 风格 top-5000，稀疏友好）/ `de`（两两类型
   差异基因，`n = round(500·(2/3)^log2(K))` 每对）/ `sd` / `all`；
3. **打分（cells 模式）**：逐细胞 Spearman 相关
   `ρ(q, r) = corr(rank(q), rank(r))`，每类型取 top-`top_n` 相关样本均值；
   **profile 模式**：与每类型中位表达谱相关（`O(Q×K×G)`，14× 提速）；
4. **fine-tuning**：候选类型集 = 首轮得分 ≥ 最高分 − `fine_tune_thres` 的类型，
   迭代丢弃最差类型直到收敛；差异基因不足 `min_genes` 时提前终止；
5. **置信度**：卡方离群检验
   `p = 1 − CDF(χ²₁, ((max−mean)/sd)²)`；
6. **多参考**：各参考独立注释后按标签并集 `max` / `mean` 合并得分，argmax 得
   共识标签。

**Seurat 标签转移**（`method="seurat"`）：共享 HVG → 逐基因 z-score →
CCA（`LinearOperator` + `svds`，不物化 cells×cells 交叉协方差）或参考 PCA
投影 → L2 归一化 → kNN（`cKDTree`）→ MNN anchors → `ScoreAnchors`（共享邻居
数分位归一化）→ `TransferData` 权重
`w = 1 − exp(−proximity·anchor_score/(2/sd)²)` → 标签预测。

### 2.2 cellmarkerannot（CellMarker 3.0 数据库）

1. **范围限定**：`species` + `tissue_type`（精确、大小写不敏感）+ 默认
   `marker_source ∈ {Experiment, Single-cell sequencing}`（或仅 Experiment）；
2. **矩阵构建**：成员矩阵 `A[g,c] ∈ {0,1}`；证据权重
   `W[g,c] = max_s evidence(s) × idf(g)`，其中
   `idf(g) = 1/log1p(#细胞类型含 g)`；
3. **表达过滤**：只统计 `X > expr_threshold`（默认 1.0）的 marker；
4. **三种打分**：

   | 方法 | 公式 | 语义 |
   |---|---|---|
   | overlap | `(X>t) @ A` | 表达的 marker 数 |
   | weighted | `(X_sub @ W) / Σ_c W` | 特异性加权平均表达（消除集合规模偏差） |
   | ssgsea | Barbie 式随机游走 ES | 单样本基因集富集 |

5. **置信度**：预测类型 = argmax；softmax 概率为置信度；
6. **标签合并**：`min_cells` 下限 + 膝盖/覆盖率自动选 k（其余归 `Other`）；
7. **基因列表富集** `score_gene_list`：`基因 × 细胞类型` 支持证据矩阵 +
   超几何富集 Score `−log10(Hypergeom.sf(k_c−1, N, K_c, n_q)) × log1p(Σev)`。

## 3. 融合层算法（核心）

### 3.1 数据流

```mermaid
flowchart LR
    subgraph 证据链 1
        R1[参考数据] --> FE[每标签特征基因 G_c<br/>DE 风格 top-N]
        FE --> S0[pysingle 首轮得分 S0]
    end
    subgraph 证据链 2
        DB[CellMarker 3.0] --> SC[范围内 marker]
        FE --> P1[逐细胞先验 P[cell,c]<br/>G_c∩DB 中该细胞表达比例]
        SC --> P1
    end
    S0 --> FUS
    P1 --> FUS[F = S0 × 1 + λ_eff·P̂<br/>margin 门控 + 粗族先验]
    FUS --> FT[fine-tuning]
    FT --> CSTAR[c* 标签]
    CSTAR --> V[逐细胞验证 V<br/>c* 特征基因的 DB 支持比例]
    V --> CONF[hybrid_confidence<br/>= clip(F[c*],0,1)·(0.5+0.5V)]
    CSTAR --> STATUS[hybrid_status<br/>c* 粗族 vs DB 预测粗族]
```

### 3.2 逐细胞 DB 先验

对每个参考标签 `c`，取其与 DB 范围重叠的特征基因 `G_c = genes[c] ∩ DB`，
细胞在其中的表达比例构成原始支持，再**逐行（细胞）max 归一化**：

```
P[cell,c] = ( Σ_{g∈G_c} 1[X[cell,g] > t] ) / |G_c|
Pn[cell,c] = P[cell,c] / max_c P[cell,c]        # 整行无支持 → 0
```

关键点：先验是**逐细胞**的（用细胞自身表达计证据），而不是静态的标签偏好向量
——静态先验会把同一批 T 标签在全部细胞上抬高（实测跨细胞相关 ≈0.81），是早期
融合退化的首因。

### 3.3 v_full 乘法融合（默认）

```
输入: S0（首轮得分，cells×labels）、P_cell、粗族映射 family(c)

1. 族级聚合:   P̂[cell, family(c)] = max_{c'∈family(c)} Pn[cell,c']
              # 同族细标签共用同一乘数 ⇒ 族内细标签排序与 pysingle 逐位等价
2. winner margin 门控（自校准）:
              margin[cell] = top1(S0) − top2(S0)
              λ_eff[cell] = λ × clip(1 − margin/(gate·median(margin)), 0, 1)
              # 高置信细胞 λ→0，融合不触碰已正确细胞；默认 gate=2.0, λ=0.3
3. 融合:      F[cell,c] = S0[cell,c] × (1 + λ_eff[cell]·P̂[cell,family(c)])
4. 标签:      c* = pysingle fine-tuned 标签（在 F 上精修，多参考为合并得分 argmax）
```

该公式作用在**首轮**得分上（fine-tuning 在其上精修），而不是微调后得分——
微调标签显著优于 `argmax(微调得分)`，对微调后得分做乘法会破坏 DE 基因精修。

### 3.4 可选 Variant B：族级后验再排序

`family_posterior=True` 时，DB 只做**粗族决策**，族内细标签永远归 pysingle：

```
1. 族级聚合:   S_fam[cell,fam] = max_{c∈fam} S0[cell,c]
               P_fam[cell,fam] = max_{c∈fam} Pn[cell,c]
2. 族级 margin 门控 λ_eff（同 3.3）
3. G[cell,fam] = S_fam × (1 + λ_eff·P_fam)  →  chosen_family = argmax G
4. 施加:       非 chosen_family 的标签置 -1e9（mask）
               ⇒ fine-tuning 只在 DB 选定的族内精修
```

实测（子集档）该硬 mask 方案会放大 DB 的粗族错误（翻转精度 0.27 vs v_full
0.78），故默认 **v_full**；Variant B 作为可选模式保留。

### 3.5 置信度、验证与状态分层

```
V[cell]   = ( # c* 特征基因 ∩ DB 且该细胞表达 > t ) / ( # c* 特征基因 ∩ DB )
conf      = clip(F[c*], 0, 1) × (0.5 + 0.5·V)
```

| 状态 | 条件 |
|---|---|
| `consistent` | c* 粗族 == DB 预测粗族，且 `conf ≥ threshold`（默认 0.3） |
| `low_confidence` | 粗族一致但 `conf < threshold` |
| `unknown` | 粗族冲突，或无 DB 证据 |
| `db_only` | 无参考降级链路 |

### 3.6 规模感知策略与翻转闸门

- **单参考 ≥50k 细胞默认 λ=0**（实测该规模融合净为负：翻转精度 0.253，
  Δacc −0.8%）；`--lambda` 可覆盖；
- 每个 hybrid 场景都计算 λ=0 孪生对照，输出
  `flip_precision = good/(good+bad)`（good=融合改对，bad=融合改错）；
  **< 0.5 表示该规模融合无净价值**，报告自动标注。

## 4. 性能工程

全量（160,068 参考 × 58,677 查询）从最初的 **7–8 小时降到 ≈6–8 分钟**：

| 手段 | 说明 |
|---|---|
| `scoring="profile"` | 中位表达谱打分，`O(Q×K×G)` 替代 `O(Q×R×G)`，14× 提速 |
| 逐类型参考封顶 | `max_cells_per_type` 确定性抽样，成本与 n_ref 解耦 |
| fine-tune 组拆块并行 | 巨型候选组（T/NK）按 `fine_tune_chunk` 拆分，逐位不变 |
| `n_jobs` 多进程 | Linux fork + 写时复制共享大数组，每进程 BLAS 限 1 线程 |
| λ=0 复用孪生 | 单参考 λ=0 时复用 pysingle 结果，跳过重复计算 |
| 稀疏保持 | 全程保持稀疏、按块稠密化，相关矩阵不常驻内存 |

## 5. 算法创新点

1. **逐细胞 DB 证据先验**（而非静态标签偏好）：用细胞自身对"标签特征基因 ∩
   DB marker"的表达比例计证据，修复静态先验"对全体细胞抬高同一批 T 标签"
   的病理（静态先验 λ=1 时粗族一致率 0.92→0.60，逐细胞恢复到 ≥0.90）。
2. **winner-margin 自校准门控**：`λ_eff` 随首轮得分 top1−top2 逐细胞衰减，
   高置信细胞完全不被融合触碰，治愈"改错已正确细胞"的规模依赖退化。
3. **粗族级先验（family boost）**：DB 证据按粗族取 max 聚合，族内细标签
   排序与 pysingle 逐位等价——DB 只改变族间平衡，不搅动族内细标签
   （细粒度缺口从 −0.02 收敛到 ≈0）。
4. **翻转精度质量闸门 + 规模感知 λ**：用 good/(good+bad) 客观回答"融合在该
   规模是否有净价值"，并据此对 ≥50k 单参考自动关闭融合——把"是否融合"变成
   数据驱动决策而非固定参数。
5. **可解释的每标签证据矩阵**：每个参考标签输出 `score_gene_list` 风格矩阵
   （特征基因 × DB 细胞类型支持证据 + 超几何富集 Score）与热图，融合结果可
   逐标签人工核对。
6. **证据高效的工程实现**：批量向量化（一次稀疏 groupby/pivot + 全矩阵
   hypergeom 广播）、profile 打分 + 逐类型封顶 + 微调分块并行，使全量注释
   从小时级进入分钟级，同时保持 cells 语义可逐位复现。

## 6. 关键参数默认值

| 参数 | 默认 | 说明 |
|---|---|---|
| `species` / `tissue` | `Human` / `Blood` | DB 打分范围（管线显式传 `Peripheral Blood`） |
| `data_source` | `experiment` | 仅用 Experiment 来源的 DB 证据 |
| `lambda_` | `0.3` | 融合强度（0 = 纯 pysingle） |
| `lambda_margin_gate` | `2.0` | margin 门控 |
| `family_boost_only` | `True` | 粗族级先验 |
| `scoring` | `auto` | ≤10k → cells，>10k → profile |
| `max_cells_per_type` | 自动 | ≥120k → 300，≥50k → 500，否则不封顶 |
| `confidence_threshold` | `0.3` | 状态分层阈值 |

## 7. 参考

- Aran et al., *Nat Immunol* 2019（SingleR）
- Stuart et al., *Cell* 2019（Seurat integration / label transfer）
- CellMarker 3.0（https://bio-bigdata.hrbmu.edu.cn/CellMarker）
- 本仓库 `docs/superpowers/specs/2026-08-21-hybridscsinglemarker-skeleton-design.md`
  与 `2026-08-21-hybridscsinglemarker-verification-report.md`
