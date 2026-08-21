# CellMarker 3.0 数据库数据结构说明

> 本说明文档基于 `./README.txt` 与 `./database/` 目录下全部文件的实际分析生成，所有关键数值均已通过磁盘文件直接验证。

## 1. 概述

CellMarker 3.0（https://bio-bigdata.hrbmu.edu.cn/CellMarker）是一个细胞类型 Marker 基因数据库。本项目中，`database/` 目录存放全部 Marker 数据，共 **5 个文件**，均为**制表符分隔的纯文本 TSV**，共享同一套 **22 列**结构，总数据量约 2.5 GB。

`./README.txt` 对 5 个文件的说明：

| 文件 | 说明（译） |
|---|---|
| `all_cell_marker` | 人（human）与鼠（mouse）不同组织、不同细胞类型的全部 Marker |
| `human_cell_marker` | 人的不同组织、不同细胞类型的 Marker |
| `mouse_cell_marker` | 鼠的不同组织、不同细胞类型的 Marker |
| `single_cell_marker` | 来源于单细胞测序研究的人与鼠 Marker |
| `method_cell_marker` | 使用多种机器学习算法计算得出的 Marker |

## 2. 文件清单

| 文件 | 大小 | 数据行数（不含表头） | 列数 | 物种 |
|---|---:|---:|---:|---|
| `all_cell_marker.txt` | 811,638,251 B（≈774 MB） | 2,537,570 | 22 | Human、Mouse |
| `human_cell_marker.txt` | 565,102,276 B（≈539 MB） | 1,779,944 | 22 | Human |
| `mouse_cell_marker.txt` | 246,536,199 B（≈235 MB） | 757,626 | 22 | Mouse |
| `single_cell_marker.txt` | 147,883,354 B（≈141 MB） | 418,933 | 22 | Human、Mouse |
| `method_cell_marker.txt` | 737,173,883 B（≈703 MB） | 2,318,927 | 22 | Human、Mouse |

## 3. 文件格式

- **编码**：UTF-8 / ASCII 纯文本，制表符 `\t` 分隔（TSV）。
- **表头**：每个文件第一行为表头行，含 22 个列名；其后每行一条数据记录。
- **转义**：无引号包裹、无转义符。标题等含空格的字段内部**不含制表符**，因此可直接按 `\t` 安全拆分。
- **记录粒度**：一行 = 一条"（组织 + 细胞类型 + 疾病条件下的）Marker 基因"注释记录，并与来源文献关联。
- **五个文件表头完全相同**，便于统一解析。

## 4. 字段结构（22 列）

统一表头（列位置 0 基）：

```
species  tissue_class  tissue_type  uberon_id  disease  cell_name_class  cell_name  cellontology_id
marker  symbol  gene_id  gene_type  gene_name  uniprot_id  technology_seq  marker_source
pmid  title  journal  year  series_id  method_details
```

### 4.1 物种与组织

| # | 列名 | 类型 | 说明 | 样例 | 数据质量提示 |
|---|---|---|---|---|---|
| 0 | `species` | string | 物种 | `Mouse`、`Human` | 取值仅 Human / Mouse 两种 |
| 1 | `tissue_class` | string | 组织的粗分类（层级上层） | `Adipose tissue`、`Muscle`、`Pancreas` | 共 162 个取值 |
| 2 | `tissue_type` | string | 组织的细分类（层级下层） | `Adipose tissue`、`Diaphragm`、`Endocrine pancreas` | 共 642 个取值；大小写/拼写未统一 |
| 3 | `uberon_id` | string | 组织对应的 UBERON 本体 ID | `UBERON_0001013` | 部分行为空 |

### 4.2 疾病

| # | 列名 | 类型 | 说明 | 样例 | 数据质量提示 |
|---|---|---|---|---|---|
| 4 | `disease` | string | 疾病/生理状态 | `Normal`、`Abdominal aortic aneurysm (AAA)` | 共 1,406 个取值；`Normal` 表示健康状态，其余为疾病名称 |

### 4.3 细胞类型

| # | 列名 | 类型 | 说明 | 样例 | 数据质量提示 |
|---|---|---|---|---|---|
| 5 | `cell_name_class` | string | 细胞类型的粗分类 | `Endothelial cell` | 共 331 个取值 |
| 6 | `cell_name` | string | 细胞类型具体名称 | `Endothelial cell`、`T cell` | 共 4,451 个取值 |
| 7 | `cellontology_id` | string | 细胞类型对应的 Cell Ontology（CL）本体 ID | `CL_0000115` | 共 859 个取值；14,091 行为空 |

### 4.4 基因 / Marker

| # | 列名 | 类型 | 说明 | 样例 | 数据质量提示 |
|---|---|---|---|---|---|
| 8 | `marker` | string | Marker 基因符号（本库核心字段） | `Egfl7`、`ESAM` | 全部文件共约 60,000+ 个不同 marker |
| 9 | `symbol` | string | 基因符号（NCBI 官方 Symbol） | `Egfl7` | 与 `gene_id` 同时为空的有 193,628 行 |
| 10 | `gene_id` | string | NCBI Entrez Gene ID | `353156.0` | **以浮点字符串形式存储**，如 `353156.0`，需去除 `.0` 再转整型 |
| 11 | `gene_type` | string | 基因生物类型 | `protein_coding` | 取值如 protein_coding、lncRNA 等 |
| 12 | `gene_name` | string | 基因全名 | `EGF-like domain 7` | — |
| 13 | `uniprot_id` | string | UniProt 蛋白 ID | `Q9QXT5` | 373,052 行为空 |

### 4.5 技术与来源

| # | 列名 | 类型 | 说明 | 样例 | 数据质量提示 |
|---|---|---|---|---|---|
| 14 | `technology_seq` | string | 测序/实验技术平台 | `FACS`、`scRNA-seq`、`10x Genomics scRNA-seq`、`Smart-seq2` | 200+ 个取值，**未规范化**（如 `Single-cell sequencing` 与 `Single-cell RNA sequencing` 并存）；32,731 行为空 |
| 15 | `marker_source` | string | Marker 来源类型 | `Method`、`Experiment`、`Review`、`Single-cell sequencing`、`Company` | 5 类取值，见 §5.4 分布 |
| 21 | `method_details` | string | 计算来源：机器学习算法名 | `COSG`、`CelliD`、`Cepo`、`FindAllMarker`、`SEMITONES`、`Spapros` | 6 种算法取值；绝大多数出现在 `marker_source='Method'` 行，但另有 10,335 行 `marker_source='Single-cell sequencing'` 也带算法名（数据不一致，见 §7 第 4 条） |

### 4.6 文献溯源

| # | 列名 | 类型 | 说明 | 样例 | 数据质量提示 |
|---|---|---|---|---|---|
| 16 | `pmid` | string | PubMed 文献 ID | `40670619.0` | **以浮点字符串形式存储** |
| 17 | `title` | string | 文献标题 | `Cluster-independent multiscale marker identification...` | 字段内含空格，但无制表符 |
| 18 | `journal` | string | 发表期刊 | `communications biology` | 大小写/拼写未统一 |
| 19 | `year` | string | 发表年份 | `2025.0` | **以浮点字符串形式存储**；范围 1978–2026 |
| 20 | `series_id` | string | 数据系列 / 数据集来源 ID | `Figshare_40670619`、GEO 系列号等 | 用作同源记录分组键 |

## 5. 数据组织方式

### 5.1 物种划分

- `all_cell_marker.txt` 内含 **Human 1,779,944 行 + Mouse 757,626 行**，两物种记录按数据来源批次**交错排布**（约 16,378 个连续区块），并非简单"先人后鼠"。
- `human_cell_marker.txt` 与 `mouse_cell_marker.txt` 分别为单物种文件。

### 5.2 组织层级（tissue_class → tissue_type）

`tissue_class` 是 `tissue_type` 的粗粒度归组，例如：

| tissue_class | tissue_type |
|---|---|
| Muscle | Diaphragm |
| Pancreas | Endocrine pancreas |
| Breast | Mammary Gland |
| Intestine | Fetal intestine |
| Lymph | Lymph node |
| Artery | Aortic valve |
| Uterus | Cervix |
| Head and neck | Head |

- `tissue_class`：162 个不同值；`tissue_type`：642 个不同值。
- `uberon_id` 提供组织对应的 UBERON 本体 ID，用于跨库对照。

### 5.3 细胞类型层级（cell_name_class → cell_name）

同理，`cell_name_class` 为粗分类、`cell_name` 为具体细胞类型：

- `cell_name_class`：331 个不同值；`cell_name`：4,451 个不同值。
- `cellontology_id` 提供细胞类型对应的 Cell Ontology（CL）本体 ID（859 个不同值）。

### 5.4 Marker 来源分布（`all_cell_marker` 统计）

`marker_source` 5 类取值及行数：

| marker_source | 行数 |
|---|---:|
| Method | 2,318,927 |
| Single-cell sequencing | 144,347 |
| Experiment | 63,036 |
| Review | 7,156 |
| Company | 4,104 |

`method_details`（计算算法）6 类取值及行数：

| method_details | 行数 |
|---|---:|
| COSG | 553,145 |
| CelliD | 535,640 |
| FindAllMarker | 518,250 |
| Cepo | 367,562 |
| SEMITONES | 346,313 |
| Spapros | 8,352 |

> 注：6 种算法行数合计 **2,329,262**，其中 2,318,927 行 `marker_source='Method'`，另有 **10,335 行 `marker_source='Single-cell sequencing'` 也带算法名**（来源标记与算法字段不一致，见 §7 第 4 条）。

年份范围：**1978–2026**。

### 5.5 各文件的范围差异

| 文件 | 特点 |
|---|---|
| `human_cell_marker` | 仅 Human；138 个 tissue_class、1,100 个 disease、3,188 个 cell_name、60,160 个不同 marker |
| `mouse_cell_marker` | 仅 Mouse；121 个 tissue_class、646 个 disease、2,265 个 cell_name、47,413 个不同 marker |
| `single_cell_marker` | `technology_seq` 全部为单细胞平台（scRNA-seq、10x Chromium、Smart-seq2 等）；`marker_source` **不含 Company**；年份 2010–2026 |
| `method_cell_marker` | `marker_source` 全部为 Method；`method_details` 恰为 6 种算法；63 个 tissue_class；年份 2016–2025 |

## 6. 文件间关系

五个文件共享同一 22 列 schema，且 `all_cell_marker.txt` 为**超集主文件**，其余文件均为它的过滤子集（已用多列键逐行程序化验证）：

```
                all_cell_marker.txt（主文件，2,537,570 行）
                ├── 按物种过滤 ──────────────→ human_cell_marker.txt（1,779,944 行）
                │                              mouse_cell_marker.txt（757,626 行）
                │                              （两者相加 = 2,537,570，键完全匹配 100%）
                ├── 按 marker_source='Method' ─→ method_cell_marker.txt（2,318,927 行）
                └── 按单细胞技术切片 ──────────→ single_cell_marker.txt（418,933 行）
```

- `all_cell_marker = human ∪ mouse`：行数精确相加，且用 `(species, tissue_type, cell_name, marker, pmid, title)` 键核对，两单物种文件的每一行都存在于主文件。
- `method_cell_marker = filter(all, marker_source='Method')`：2,318,927 / 2,318,927 行命中。
- `single_cell_marker`：`all` 中 `technology_seq` 属于单细胞平台的切片（与 `marker_source` 无关）。
- `single_cell_marker ∩ method_cell_marker` = 321,404 行（既为单细胞来源、又由算法计算的行）。
- **注意**：虽然内容上是子集，但各文件并非主文件的简单顺序切片，行序经过重新组装（如主文件以 Mouse 行开头），解析时不要假设顺序一致。

## 7. 数据质量注意事项

解析本库时需特别处理以下问题：

1. **数值 ID 存为浮点字符串**：`gene_id`、`pmid`、`year` 均形如 `"353156.0"`、`"40670619.0"`、`"2025.0"`。使用前应去掉尾部 `.0` 再转换为整数，否则直接按字符串比较会出现 `"40670619.0" != "40670619"` 等不一致。
2. **空字段普遍存在**（`all_cell_marker` 统计）：
   - `symbol` 与 `gene_id` 同时为空：193,628 行；
   - `uniprot_id` 为空：373,052 行；
   - `cellontology_id` 为空：14,091 行；
   - `technology_seq` 为空：32,731 行。
3. **取值未规范化**：`technology_seq` 同时存在 `Single-cell sequencing` 与 `Single-cell RNA sequencing` 等近义不同写法；`tissue_type`、`journal` 等存在大小写与拼写差异。五个文件中均**未发现尾部空格**，字段去重/比较前仍建议 `trim` 以防其他空白差异。
4. **来源标记与算法字段不一致**：`method_details` 有算法名的行（2,329,262）比 `marker_source='Method'` 的行（2,318,927）多 **10,335 行**——即存在 10,335 行 `marker_source='Single-cell sequencing'` 却带算法名的记录（Cepo 1,700 / COSG 2,200 / CelliD 2,300 / SEMITONES 1,900 / FindAllMarker 2,200 / Spapros 35）。反向（Method 但算法为空）为 0 行。解析时如按 `marker_source` 过滤计算来源，会漏掉这 10,335 行。
5. **极少数字段含前导空格**：仅 `title` 列发现 6 处值以空格开头。

## 附录：配套测试数据（testdata/）

`testdata/pbmc5k_querydata.h5ad`（≈802 MB）为单文件 HDF5 格式的 AnnData，用于数据库的查询/验证测试：

- 矩阵规模：**58,677 cells × 33,421 genes**，`X` 为 CSR 稀疏 float64 矩阵。
- `obs`：`orig.ident`、`nCount_RNA`、`nFeature_RNA`、`orig.Cluster`、`ident`、`celltype`、`RNA_snn_res.0.8`、`seurat_clusters`、`source`。
- `var`：`highly_variable`；`layers`：`counts`、`data`；`obsm`：`X_pca`、`X_umap`；`uns`：`seurat2anndata`。
- 细胞条码形如 `AAACCTGAGACACTAA-1`，基因名形如 `MIR1302-2HG`、`FAM138A`（PBMC 5k 数据集）。
