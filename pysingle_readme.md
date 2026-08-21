# pysingle

**R 包 [SingleR](https://www.bioconductor.org/packages/release/bioc/html/SingleR.html) 的 Python 移植版**，基于 [AnnData](https://anndata.readthedocs.io/)（h5ad 文件格式）实现单细胞转录组的细胞类型自动注释。

算法通过计算测试细胞表达谱与参考细胞类型特征表达谱之间的相关性打分，为每个细胞分配最匹配的参考标签，并提供可选的低置信度剪枝（pruning）与迭代细化（iterative refinement）。

## 安装

```bash
# 使用 uv
uv venv --python 3.12
uv sync

# 或使用 pip
pip install -e .
```

要求 Python >= 3.12。

## 快速开始

### 高层入口 `annotate`（推荐）

```python
import pysingle

# ref / query 均可传 h5ad 文件路径 或 已加载的 anndata.AnnData 对象
query = pysingle.annotate(
    ref="testdata/pbmc50k_refdata.h5ad",       # 参考数据（obs 需含 celltype 列）
    query="testdata/vkhQ8_querydata.h5ad",     # 查询数据
    celltype_col="celltype",                   # 参考细胞类型标签列名
    gene_selection="hvg", max_genes=5000,      # 默认取 top 5000 高变基因
    top_n=5, fine_tune=True,
)

query.obs["pysingle_celltype"]   # 每个细胞的预测细胞类型
query.obs["pysingle_score"]      # 对应得分
```

### 打分方式（`scoring`）

经典 SingleR 逐细胞相关是 **O(n_query × n_ref × n_genes)**，面对数万细胞的大
参考（如 `pbmc50k_refdata` 的 5.9 万参考细胞）会非常慢——R 本身同样如此。
提供两种打分方式：

| `scoring` | 语义 | 复杂度 |
| --------- | ---- | ------ |
| `"cells"`（默认） | 逐细胞 Spearman + 每类型 top-N 相关样本均值（经典 SingleR） | O(Q×R×G) |
| `"profile"` | 按类型**中位表达谱**相关性打分（SingleR 2.0 聚合思路） | O(Q×n_types×G) |

```python
# 大参考（数万细胞）快速路径：全量 DE 数分钟内完成
query = pysingle.annotate(ref, query, gene_selection="de", scoring="profile",
                          n_jobs=24)
```

配套性能优化（已内置）：
- fine-tuning 的候选 DE 基因改为**所有类型对预计算一次**（`_pairwise_de_genes`），
  候选集只做字典查找 + 并集——消除大参考下逐候选组 argsort 全部基因的瓶颈
  （实测 profile 3000ref/2000q：87s → **7.4s**）；
- 首轮分块打分、参考列排名、参考中位数矩阵、fine-tuning 候选组均可多进程并行
  （`n_jobs`，fork + 写时复制共享大数组，每进程 BLAS 限 1 线程）；
- 内存有界：fine-tuning 切片只稠密化 (候选细胞 × DE 基因)，避免全基因稠密化。

### 基因选择（`gene_selection`）

参考/查询表达矩阵中大量基因无信息量（零方差/低表达），默认使用
**高变基因（HVG，Seurat vst 风格）top 5000** 作为首轮计算输入，显著降低
耗时与内存且不影响注释精度。可选方法：

| `gene_selection` | 说明 |
| ---------------- | ---- |
| `"hvg"`（默认）  | 高变基因 top `max_genes`（默认 5000，稀疏友好） |
| `"de"`           | 两两细胞类型差异基因（对应 R `genes="de"`） |
| `"sd"`           | 中位表达谱标准差阈值法（对应 R `genes="sd"`） |
| `"all"`          | 全部共有基因 |

```python
query = pysingle.annotate(ref, query, gene_selection="all")      # 全部基因
query = pysingle.annotate(ref, query, gene_selection="hvg", max_genes=3000)
```

> 注：Spearman 相关性需对表达矩阵稠密化，超大参考集（数十万细胞 × 数万基因）
> 内存占用高、耗时大，建议先用 `sc.pp.subsample` / `adata[...]` 抽样参考细胞。

### Seurat 标签转移注释（`method="seurat"`）

对应 `Seurat::FindTransferAnchors + TransferData`（vignette:
integration_mapping.Rmd），纯 Python 实现（`pysingle/seurat_method.py`）：

1. 共享可变基因（HVG）→ 逐基因 z-score（`ScaleData`）
2. 低维嵌入：CCA（`RunCCA`，`crossprod`+SVD）或参考 PCA 投影（`pcaproject`）
3. 嵌入 L2 归一化 → kNN（`FindNN`）→ MNN anchors（`FindAnchorPairs`）
4. `ScoreAnchors`：共享邻居数 → 分位归一化得分
5. `TransferData`：`weight = 1-exp(-proximity·anchor_score/(2/sd)²)` → `scores = weightsᵀ%*%onehot(标签)`

**大样本性能优化**（对照 Seurat 的 C++ 实现）：
- **CCA 内存**：`M = crossprod(ref, query)` 为 (cells1×cells2) 大矩阵，改用
  `scipy LinearOperator`（`M@v = d1ᵀ@(d2@v)`）+ `svds`，**不物化交叉协方差矩阵**，
  内存从 O(cells1×cells2) 降到 O(genes×(cells1+cells2))（实测 ref5000×query3000
  ×2000 基因峰值内存仅 +0.13GB）；
- **kNN**：`scipy cKDTree`（C 实现）替代暴力法；
- **权重热循环**：编译 C 扩展 `pysingle/_fastseurat.c`（对应
  `seurat/src/integration.cpp` 的 `FindWeightsC`），无 C 编译器时自动回退 numpy。

```bash
uv sync   # 自动编译 C 扩展（需 gcc；无编译器则回退纯 Python）
```

```python
out = pysingle.annotate(ref, query, method="seurat", celltype_col="celltype",
                        reduction="cca", n_dims=30, k_anchor=5)
# 或直接调用核心
result = pysingle.seurat_annotate(ref, ref.obs["celltype"], query)
```

示例脚本：`python scripts/test_seurat_annotation.py`

### 多参考（多数据库）注释

对应现代 `SingleR(test, ref=list(...), labels=list(...))`：对每个参考分别
注释后跨参考合并为共识标签（`combineResults`，`max`/`mean`）。

```python
# 两个参考可具有不同的基因空间（自动分别与查询取交集）
out = pysingle.annotate(
    ref=["ref_A.h5ad", "ref_B.h5ad"],
    query="query.h5ad",
    celltype_col=["celltype_A", "cell_type"],   # 逐参考指定标签列
    combine_method="max",                       # 或 "mean"
)
```

底层函数：`pysingle.singleR_annotate_multi([(ref_A, labels_A), (ref_B, labels_B)], query)`，
返回 `labels`（共识标签）、`scores`/`all_scores`（跨参考合并）、`per_reference`（各参考完整结果）。

### 底层核心 `singleR_annotate`（完整得分矩阵）

```python
result = pysingle.singleR_annotate(ref_adata, ref_labels, query_adata,
                                   fine_tune=True, top_n=5)

result["labels"]       # 每个查询细胞的最终预测类型
result["scores"]       # 每个细胞对应各类型的最终得分
result["all_scores"]   # 首轮全部类型的原始得分
result["pval"]         # 卡方离群检验置信度 p 值
```

### 可视化（SingleR 风格，需 matplotlib）

```python
import pysingle.plotting as pl

pl.plot_score_heatmap(result)                 # 得分热图（pheatmap 风格）
pl.plot_annotation_scatter(result, xy)        # 注释散点图（tSNE/UMAP）
pl.plot_cell_boxplot(query_adata, ref_adata, ref_labels, cell_id="cell_1")
```

> 表达矩阵支持 4 种输入：`anndata.AnnData`（X 为 细胞×基因，自动转置）、
> `pd.DataFrame`（行=基因，列=细胞）、`numpy.ndarray` 与 `scipy.sparse` 矩阵
> （后两者按位置对齐，无基因名）。基因名自动转小写取交集，参考与查询行自动对齐。

## 开发

```bash
uv sync            # 安装依赖（含 dev 组的 pytest / matplotlib）
uv run pytest      # 快速测试套件（慢速集成测试默认排除）
```

## 一键验证（testdata 真实数据）

```bash
# 快速验证：子集采样（默认 query 前 500 / ref 前 500 细胞，约 6 分钟）
python run_test.py

# 更快：100 query / 200 ref，约 1 分钟
python run_test.py --quick

# 自定义规模 / 关闭微调
python run_test.py --query-max 200 --ref-max 300 --no-fine-tune

# 全量运行（内存需求极高，需显式确认）
python run_test.py --full --force
```

大数据处理策略：
- 核心算法对**稀疏输入保持稀疏**、按块稠密化，首轮打分使用分块计算
  （``chunk_size``，默认 5000），(n_query × n_ref) 相关矩阵不常驻内存，
  可支撑 10w+ 查询细胞；分块不改变计算结果；
- **多进程并行**（``n_jobs``，默认脚本为 `min(cpu, 8)`）：首轮分块打分、
  参考列排名、参考中位数矩阵、fine-tuning 候选组均按进程并行
  （Linux fork + 写时复制共享大数组，每进程 BLAS 限 1 线程避免核数争抢），
  实测 DE 模式 600/400 串行 94s → 8 进程 23s（4 倍）；
- 重计算（排序/矩阵乘）由 numpy（OpenBLAS）与 scipy 的 C 实现完成，
  与 Cython/C++ 改写后的同算法速度相当，避免了额外构建依赖；
- fine-tuning 的中位表达谱**预计算一次**复用（对应 R 的
  ``medianMatrix`` + ``mean_mat[, topLabels]``），避免每个候选组重复
  ``tocsc()`` 与中位数计算（这是此前 DE 全量跑十几个小时的根因）；
- 全量真实数据（~5.9万×3.3万 基因，DE 模式）仍需较大内存/耗时，
  建议 ``--n-jobs 16~32``，预计数十分钟内完成；`run_test.py` 默认仍先做
  子集采样验证算法，确认无误后再考虑全量（`--full --force`）。

真实数据集成测试（基于 `testdata/`）默认不随 `uv run pytest` 运行，
需显式指定，并可用环境变量调整规模：

```bash
uv run pytest -m slow tests/test_basic_annotation.py            # 默认 500/500，约 6 分钟
PYSUBSET_QUERY_MAX=200 PYSUBSET_REF_MAX=300 \
    uv run pytest -m slow tests/test_basic_annotation.py        # 缩小规模
```

## 测试脚本（`scripts/`）

基于 testdata 的端到端测试，可直接运行：

```bash
# 1. 注释功能测试：结果存 results/，图存 results/figures/，表存 results/tables/
python scripts/test_pysingle_annotation.py          # 默认 500/500 子集

# 2. 可视化功能测试：生成全部 SingleR 风格图形到 figures/
python scripts/test_pysingle_plot.py                # 默认 500/500 子集
```

| 脚本 | 保存内容 |
| ---- | -------- |
| `test_pysingle_annotation.py` | `results/annotation_summary.txt`、`results/tables/`（labels/scores/all_scores/distribution）、`results/figures/`（热图/散点/箱线图） |
| `test_pysingle_plot.py` | `figures/score_heatmap.png`、`annotation_scatter.png`、`cell_boxplot.png`、`cell_vs_ref_scatter.png` |

## 算法与 R → Python 映射

核心流程复刻 R 包 SingleR：

| 算法步骤                       | R (SingleR)                       | Python (pysingle)                     |
| ------------------------------ | --------------------------------- | ------------------------------------- |
| 高层注释入口（AnnData 适配）   | `SingleR.CreateObject()`          | `annotate()`                          |
| 顶层注释核心                   | `SingleR()`                       | `singleR_annotate()`                  |
| 基因交集与质量过滤             | `tolower`/`intersect`/NA-零行过滤  | `core._intersect_and_filter()`        |
| Spearman 相关矩阵（向量化）     | `cor.stable(method='spearman')`   | `core._spearman_corr()`               |
| 每类型得分（top-N 均值）        | `quantileMatrix`（分位数聚合）     | `core._top_n_scores()`（按需求改 top-N） |
| fine-tuning 微调               | `SingleR.FineTune`/`fineTuningRound` | `core._fine_tune()`                 |
| 置信度（卡方离群检验）          | `SingleR.ConfidenceTest`          | `core._chisq_outlier_pvalue()`        |
| 参考中位表达谱 / DE 基因        | `medianMatrix` / `genes="de"`     | `core._median_matrix()` / `_de_genes()` |
| 得分热图                       | `SingleR.DrawHeatmap`             | `plotting.plot_score_heatmap()`       |
| 注释散点图（tSNE/UMAP）        | `SingleR.PlotTsne`                | `plotting.plot_annotation_scatter()`  |
| 单细胞相关箱线图               | `SingleR.DrawBoxPlot`             | `plotting.plot_cell_boxplot()`        |
| 单细胞 vs 参考样本散点          | `SingleR.DrawScatter`             | `plotting.plot_cell_vs_ref_scatter()` |

## 目录结构

```
pysingle/
├── __init__.py   # 包入口与公共 API 导出
├── core.py       # 核心注释算法（singleR_annotate 及内部实现）
├── io.py         # h5ad 数据读写与参考集构建
├── plotting.py   # SingleR 风格可视化（SINGLER_COLORS + 绘图）
└── utils.py      # 标准化、表达谱聚合、输入校验
tests/            # 测试脚本
```
