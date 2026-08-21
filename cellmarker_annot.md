# CellMarker 注释算法说明（methods）

本文档说明 `cellmarkerannot` 基于 CellMarker 3.0 数据库的细胞类型注释算法：数据准备、marker 矩阵构建、三种打分方法（overlap / weighted / ssgsea）、置信度与标签合并，以及基因列表富集打分。为便于理解，每种算法附带简化的伪代码（与实现一致，但省略工程细节）。

## 1. 总体流程

```
输入: AnnData (细胞×基因 表达矩阵) + CellMarker 数据库
  │
  ├─ 1. 范围限定: 物种 + 组织 + 数据来源 → 筛选相关 DB 行
  ├─ 2. 构建 marker 矩阵: 成员矩阵 A、权重矩阵 W
  ├─ 3. 表达过滤: 只保留显著表达的 marker 基因
  ├─ 4. 打分: 对每个细胞 × 每个候选细胞类型计算得分
  │        (overlap / weighted / ssgsea 三选一)
  ├─ 5. 预测: 每细胞取得分 argmax 的细胞类型
  ├─ 6. 置信度: softmax 概率（0-1）
  └─ 7. 标签合并: 最多保留 top-25 常见类型，其余归为 Other
```

## 2. 数据库与预处理

- 数据源：CellMarker 3.0 的 5 个数据集（`all` / `human` / `mouse` / `single_cell` / `method`），22 列 schema（species、tissue_class、tissue_type、cell_name、marker、marker_source 等）。
- 清洗：空值→`""`，数字 ID（`gene_id`/`pmid`/`year`）去除 `.0` 尾缀，去除首尾空白。
- 内置化：TSV 转 zstd Parquet 随包分发，运行时从包内数据加载。

## 3. 打分范围与 marker 矩阵构建

### 3.1 范围限定

只有"范围内"的 DB 行参与打分：

```
scope_rows = {row | row.species == species
              AND row.tissue_type == tissue                    # 仅 tissue_type，大小写不敏感
              AND row.marker_source ∈ 所选来源}
```

- **物种**：Human / Mouse。
- **组织**：仅匹配 `tissue_type`（精确、大小写不敏感）。按 `tissue_type` 而非 `tissue_class OR tissue_type` 可收窄范围、提升精确度——否则一个宽泛的 `tissue_class`（如 `Blood`）会把全部血液相关组织（aorta、artery、cord blood 等）都拉入范围。真实 Human/Blood 范围从 **636 类 / 20,936 marker** 收窄到 **297 类 / 10,946 marker**。
- **数据来源**（`marker_source`）：默认只使用实验验证来源 `Experiment` + `Single-cell sequencing`；可选 `method` / `review` / `company` 或全部。计算型（`Method`）与综述型（`Review`/`Company`）marker 会让宽泛细胞类型携带 ~2000 个高度重叠 marker，从而主导得分，故默认排除。

### 3.2 基因宇宙与成员矩阵

```
U       = 范围内 distinct marker 基因集合           # N_total = |U|
C       = 范围内细胞类型集合
A[g, c] = 1  如果基因 g 是细胞类型 c 的 marker，否则 0    # 成员矩阵 (|U| × |C|)
```

### 3.3 特异性权重（用于 weighted 方法）

每个 (基因, 细胞类型) 对有一个权重 `W[g, c]`，由**来源证据权重 × 逆细胞类型频率**组成：

```
evidence_weight(source) = { Experiment: 1.0, Method: 1.0, Single-cell sequencing: 0.8,
                            Review: 0.6, Company: 0.4 }
# 同一 (g, c) 对在多个来源出现时取 max

idf(g) = 1 / log1p( # 范围内包含基因 g 的细胞类型数 )     # 越少细胞类型使用 → 越特异

W[g, c] = max(evidence_weight(source)) × idf(g)
```

### 3.4 表达过滤

只统计**显著表达**的 marker（默认 `expr_threshold = 1.0`，适用于 log 归一化数据；原始 counts 数据可调低）：

```
X_sub[cell, g] = X[cell, g]   if X[cell, g] > threshold
                  0           otherwise
```

## 4. 三种细胞注释打分方法

### 4.1 overlap —— 简单计数匹配的 marker 数

```
overlap_scores(X_sub, A):
    Xb = (X_sub > 0).astype(float32)        # 二值表达矩阵
    score = Xb @ A                          # 稀疏矩阵乘法
    # score[cell, c] = 该细胞表达的、属于细胞类型 c 的 marker 个数
```

### 4.2 weighted —— 表达量 × 特异性加权（按集合规模归一化）

```
weighted_scores(X_sub, W, A):
    raw      = X_sub @ W                    # Σ_{g 表达} X[cell,g] × W[g,c]
    total_W  = Σ_g W[g,c]                   # 细胞类型 c 的全部 marker 权重之和
    score    = raw / total_W                # 特异性加权平均表达（均值而非求和）
```

> 除以 `total_W` 消除 **marker 集合规模偏差**：若不归一化，携带数千 marker 的宽泛细胞类型会因求和而主导得分（实测中典型 CD8 T 细胞会被 597-marker 的异常宽泛类型压过）。归一化后得分在不同规模集合间可比。

### 4.3 ssgsea —— 单样本基因集富集分析（Barbie 式）

对每个细胞独立计算富集得分，仅用成员矩阵 `A`（其加权来自表达秩，而非 DB 特异性）：

```
ssgsea_scores(X_sub, A, p=0.25):
    for 每个细胞:
        # 基因宇宙 = 该细胞表达的 marker 基因（秩 1..k，按表达升序，稳定排序）
        expressed = [g in marker genes where expr > 0]
        rank_g    = 1..k                    # 表达越低秩越小
        for 每个细胞类型 c:
            hits = expressed ∩ markers(c)   # c 在该细胞中命中的 marker
            N_h  = |hits|
            Z_c  = Σ_{g∈hits} rank_g^p      # 命中基因秩的 p 次方和
            # 沿表达降序做随机游走；ES 只在命中位置达到峰值
            run = 0; es = 0; t = 0
            for pos = 1..k:                 # 表达降序的位置
                g = 第 pos 个表达的基因
                if g ∈ c:
                    t += 1; run += rank_g^p
                    miss = (pos - t) / (k - N_h)
                    es = max(es, run / Z_c - miss)
            score[cell, c] = es             # 全命中 → 1.0，无命中 → 0.0
```

- 实现用 numba `@njit(parallel=True)` 对细胞并行，内部两遍扫描（先算 `Z_c`/`N_h`，再走游走），避免 Python 级每细胞循环。

### 4.4 三种方法的输出

```
score_matrix = DataFrame[cell, cell_type] = 原始得分   # 不归一化
```

## 5. 预测、置信度与标签合并

### 5.1 预测与 softmax 置信度

```
for 每个细胞:
    pred[cell] = argmax_c score_matrix[cell, c]        # 得分最高的细胞类型
    if max_c score_matrix[cell, c] == 0:               # 无任何表达 marker
        pred[cell] = NaN; confidence[cell] = 0
    else:
        probs = softmax(score_matrix[cell, :])         # 数值稳定: exp(x - max) 归一化
        confidence[cell] = probs[pred]                 # 预测类型的 softmax 概率
```

> `confidence` 反映预测类型相对其他类型的**优势程度**：清晰胜出 → 接近 1.0；前几名接近 → 较低。不再使用 min-max（min-max 会把最高分映射到 1.0，导致预测类型的置信度恒为 1.0）。

### 5.2 标签合并（限制细胞类型种类数）

范围内常含数百个细胞类型，可视化不便。`annotate_cells` 的 `max_cell_types` 控制保留的标签类型数：

- `None`（默认）—— **自动模式**：少于 `min_cells`（默认 100 barcode）的类型直接归 `Other`；对剩余类型（按预测频次降序）取 `k = max(膝盖, 覆盖率)`——**膝盖/肘部检测**（scree 最大垂直距离，收益递减点）与**覆盖率**（覆盖 `coverage`，默认 0.9，即 90% 预测细胞所需类型数）的较大者。覆盖率下限保证 `Other` 占比受控（≤1−coverage），膝盖保证分布陡峭时保留信息量大的类型。真实 PBMC（58k 细胞）：`coverage=0.9` → **17 类 + Other 12%**（纯膝盖仅 7 类、Other 29%）。
- 正整数 —— **固定上限**：同样先应用 `min_cells` 过滤，再保留频次最高的 `max_cell_types` 个。
- `0` —— **不合并**：保留全部标签。

```
_merge_labels(pred, max_cell_types, min_cells, coverage=0.9):
    if max_cell_types == 0: return pred              # 不合并
    freq = 各预测标签的细胞数（降序）
    candidates = freq[freq >= min_cells]             # 少于 min_cells 的剔除
    if candidates 为空: candidates = freq            # 小数据集放宽下限
    if max_cell_types is None:                       # 自动：膝盖 + 覆盖率取大
        k = max(knee_index(candidates),              # 最大垂直距离肘部
                coverage_index(candidates, coverage)) # 覆盖 coverage 比例所需
    else:
        k = min(max_cell_types, len(candidates))
    pred = "Other"  where pred ∉ candidates[:k]
```

> `min_cells` 在 auto 与固定上限两种模式都应用；小数据集（如几百细胞）需相应调低。`coverage` 越低 → 保留类型越少、`Other` 越多；越高 → 保留越多。

## 6. 基因列表富集打分（score_gene_list）

对**给定的基因列表**而非表达矩阵，计算每个细胞类型的支持证据与富集得分，输出 `基因 × 细胞类型` 矩阵 + 最后一行为 Score：

```
score_gene_list(genes, db, species, tissue, data_source="all"):
    rows = scope_rows(species, tissue) 且 marker_source 按 data_source 过滤
    evidence[g, c] = # rows where marker==g and cell_name==c   # 整数支持证据
    # 矩阵: 行 = 匹配基因(输入顺序, 无证据剔除), 列 = 有匹配的细胞类型,
    #       值 = evidence, 最后一行 = "Score"

    N_total = |U|                          # 范围内 distinct marker 数
    for 每个细胞类型 c:
        K_c    = # c 的 distinct marker 数（范围内）
        n_query= # 匹配的输入基因数
        k_c    = # 匹配基因中属于 c 的个数
        P_c    = Hypergeom.sf(k_c - 1, N_total, K_c, n_query)   # P(X ≥ k_c)
        evsum  = Σ_g evidence[g, c]        # 匹配基因的支持证据总和
        Score[c] = -log10(P_c) × log1p(evsum)
```

- **超几何检验**：从 `N_total` 个 marker 中抽 `n_query` 个，命中 `≥ k_c` 个 c 的 marker 的概率。`-log10(P)` 越小 p 值→ 富集越显著，得分越高。
- **支持证据加权**：`× log1p(Σ evidence)` 用匹配 marker 的文献/实验支持量加权，反映标记本身的可靠性。

## 7. 参数汇总

| 参数 | 默认 | 说明 |
|---|---|---|
| `species` / `tissue` | 必填 | 打分范围（`tissue` 仅匹配 `tissue_type`） |
| `marker_sources` | Experiment + Single-cell sequencing | 参与打分的 DB 数据来源 |
| `expr_threshold` | 1.0 | 只统计表达高于该值的 marker（log 归一化尺度） |
| `normalize`（weighted） | True | 除以该细胞类型总 marker 权重，消除集合规模偏差 |
| `method` | weighted | `overlap` / `weighted` / `ssgsea` |
| `max_cell_types` | None（自动） | 保留的预测标签类型数：`None`=自动（膝盖+覆盖率取大）、正整数=固定上限、`0`=不合并 |
| `min_cells` | 100 | 预测细胞数少于该值的类型归入 `Other`（auto 与固定上限模式均应用） |
| `coverage` | 0.9 | auto 模式的覆盖率下限（覆盖该比例预测细胞所需的类型数） |
| `data_source`（score_gene_list） | all | 支持证据计数的来源过滤（all/method/experiment/review/company/single_cell） |
