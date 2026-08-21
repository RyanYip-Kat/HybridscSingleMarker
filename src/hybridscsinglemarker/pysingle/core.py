"""SingleR 细胞类型注释核心算法（Python 移植版）。

对应 R 包 SingleR 的主流程 ``SingleR()``，输入/输出约定为：

- 表达矩阵：``pd.DataFrame``（行=基因，列=细胞），或 ``anndata.AnnData``
  （X 为 细胞×基因，自动转置）、``numpy.ndarray``、``scipy.sparse``；
- 参考标签：``pd.Series``，索引与参考细胞 ID 对齐。

各算法节点与 R 源码的对应关系：

1. 基因交集与质量过滤       -> R: ``SingleR()`` 前段（tolower / intersect / NA 零行过滤）
2. Spearman 相关性矩阵      -> R: ``cor.stable(..., method='spearman')``（HelperFunctions.R）
3. 每细胞类型得分(top-N 均值) -> R: ``quantileMatrix``（本项目按需求改为 top-N 相关样本均值）
4. fine-tuning 微调迭代     -> R: ``SingleR.FineTune`` / ``fineTuningRound``（SingleR.R）
5. 标签分配与置信度检验      -> R: ``max.col`` / ``SingleR.ConfidenceTest``

大样本性能（10w+ 细胞）：
- 稀疏输入在整个流程中保持稀疏，仅按块/按类型稠密化；
- 首轮得分使用 :func:`_score_by_chunks` 分块计算，相关矩阵不常驻内存；
- 相关性/排序等重计算由 numpy（OpenBLAS）与 scipy 的 C 实现完成，
  与 Cython/C++ 改写后的同算法在速度上相当，避免了额外构建依赖。
"""

from __future__ import annotations

import math
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

# ---------------------------------------------------------------------------
# 并行基础设施（Linux fork + 写时复制共享大数组，worker 各自 BLAS 限 1 线程）
# ---------------------------------------------------------------------------

_FORK_CTX = multiprocessing.get_context("fork")     # Linux 默认，共享大数组免拷贝
_MEDIAN_WORKER: tuple | None = None                  # 并行中位数矩阵的共享状态
_SCORE_WORKER: tuple | None = None                   # 并行分块打分的共享状态
_FT_WORKER: tuple | None = None                      # 并行 fine-tuning 组的共享状态
_RANK_WORKER: np.ndarray | None = None               # 并行列排名的共享状态


def _rank_columns_worker(spec: tuple[int, int]) -> tuple[int, np.ndarray]:
    """并行列排名 worker：对 ``spec=(start, stop)`` 的列块做 rankdata。"""
    with _limit_blas_threads():
        start, stop = spec
        return start, stats.rankdata(_RANK_WORKER[:, start:stop], axis=0)


def _rank_columns_parallel(mat: np.ndarray, n_jobs: int) -> np.ndarray:
    """按列并行计算 rankdata（参考矩阵列数大时加速，结果与串行一致）。"""
    global _RANK_WORKER
    n = mat.shape[1]
    per = max(1, math.ceil(n / n_jobs))
    specs = [(s, min(s + per, n)) for s in range(0, n, per)]
    _RANK_WORKER = mat
    try:
        with ProcessPoolExecutor(
            max_workers=min(n_jobs, len(specs)), mp_context=_FORK_CTX
        ) as ex:
            parts = list(ex.map(_rank_columns_worker, specs))
        parts.sort(key=lambda x: x[0])
        return np.concatenate([p for _, p in parts], axis=1)
    finally:
        _RANK_WORKER = None


def _limit_blas_threads():
    """将当前进程的 BLAS 线程限制为 1（避免多进程争抢核数）。"""
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=1, user_api="blas")
    except Exception:
        from contextlib import nullcontext

        return nullcontext()


def _median_type_worker(group: tuple[int, int]) -> tuple[int, np.ndarray]:
    """并行中位数 worker：计算一组类型（列）的中位表达谱。"""
    with _limit_blas_threads():
        ref_csc, col_sets = _MEDIAN_WORKER
        start, stop = group
        med = np.empty((ref_csc.shape[0], stop - start), dtype=float)
        for j in range(stop - start):
            cols = col_sets[start + j]
            block = ref_csc[:, cols].toarray() if sp.issparse(ref_csc) \
                else np.asarray(ref_csc)[:, cols]
            if cols.size == 1:
                med[:, j] = block[:, 0]
            else:
                med[:, j] = np.nanmedian(block, axis=1)
        return start, med


def _median_matrix_parallel(ref_csc, col_sets: list, n_jobs: int) -> np.ndarray:
    """按类型并行计算中位表达谱（父进程转 csc 一次，worker 分块处理类型）。"""
    global _MEDIAN_WORKER
    _MEDIAN_WORKER = (ref_csc, col_sets)
    try:
        k = len(col_sets)
        per = max(1, math.ceil(k / n_jobs))
        groups = [(g, min(g + per, k)) for g in range(0, k, per)]
        with ProcessPoolExecutor(
            max_workers=min(n_jobs, len(groups)), mp_context=_FORK_CTX
        ) as ex:
            parts = list(ex.map(_median_type_worker, groups))
        parts.sort(key=lambda x: x[0])
        return np.concatenate([med for _, med in parts], axis=1)
    finally:
        _MEDIAN_WORKER = None


# ---------------------------------------------------------------------------
# 输入归一化与基因对齐
# ---------------------------------------------------------------------------

def _densify_frame(expr_df: pd.DataFrame) -> np.ndarray:
    """将 DataFrame 统一转为稠密 float 数组。

    兼容三种存放形式：普通稠密列、pandas ``SparseDtype`` 稀疏列、
    以及 scipy.sparse 稀疏矩阵（此时 DataFrame 为 object dtype）。
    """
    if any(isinstance(dt, pd.SparseDtype) for dt in expr_df.dtypes):
        try:
            return expr_df.sparse.to_dense().to_numpy(dtype=float)
        except Exception:
            pass
    vals = expr_df.to_numpy()
    if sp.issparse(vals):
        return vals.toarray()
    if vals.dtype == object:
        try:
            return np.column_stack(
                [
                    np.concatenate(
                        [v.toarray().ravel() if sp.issparse(v) else np.asarray(v).ravel()
                         for v in expr_df[col]]
                    )
                    for col in expr_df.columns
                ]
            )
        except Exception:
            pass
    return np.asarray(vals, dtype=float)


def _is_anndata(obj) -> bool:
    """判断对象是否为 ``anndata.AnnData``（鸭子类型，避免强依赖导入）。"""
    return all(hasattr(obj, a) for a in ("X", "obs_names", "var_names", "to_df"))


def _expr_to_mat(expr, name: str) -> tuple[Any, list | None, list | None]:
    """统一表达式输入为 (基因 × 细胞) 矩阵，**保持稀疏**（不强制稠密化）。

    返回 ``(mat, gene_names, cell_names)``；``mat`` 可能是 ``np.ndarray``
    或 ``scipy.sparse`` 矩阵，供大样本场景按块/按类型稠密化。

    支持四种输入：
    - ``anndata.AnnData``（X 为 细胞×基因，自动转置为 基因×细胞 稀疏）；
    - ``pd.DataFrame``（行=基因，列=细胞，转稠密）；
    - ``scipy.sparse``（按 (基因×细胞) 约定，转 csr）；
    - ``numpy.ndarray``（按 (基因×细胞) 约定）。
    """
    if _is_anndata(expr):
        X = expr.X
        if sp.issparse(X):
            return X.T.tocsr(), list(expr.var_names), list(expr.obs_names)
        return np.asarray(X, dtype=float).T, list(expr.var_names), list(expr.obs_names)
    if isinstance(expr, pd.DataFrame):
        return _densify_frame(expr), list(expr.index), list(expr.columns)
    if sp.issparse(expr):
        return expr.tocsr(), None, None
    if isinstance(expr, np.ndarray):
        if expr.ndim != 2:
            raise ValueError(f"{name} 必须是二维数组，实际为 {expr.ndim} 维")
        return np.asarray(expr, dtype=float), None, None
    raise TypeError(
        f"{name} 须为 pd.DataFrame / anndata.AnnData / np.ndarray / scipy.sparse 矩阵"
    )


def _expr_to_dense(expr, name: str) -> tuple[np.ndarray, list | None, list | None]:
    """统一表达式输入为 (基因 × 细胞) 稠密数组（``_expr_to_mat`` 的稠密版）。"""
    mat, genes, cells = _expr_to_mat(expr, name)
    if sp.issparse(mat):
        mat = mat.toarray()
    return np.asarray(mat, dtype=float), genes, cells


def _row_index(mat, idx) -> Any:
    """按行索引（int 数组 / 布尔掩码 / 切片）取子矩阵，保持稀疏或稠密。"""
    if sp.issparse(mat):
        return mat[idx]
    return np.asarray(mat)[idx]


def _sparse_nan_rows(mat) -> np.ndarray:
    """稀疏矩阵中含 NaN 的行掩码（稀疏数据通常无 NaN，该分支极少触发）。"""
    nan_rows = np.zeros(mat.shape[0], dtype=bool)
    if np.isnan(mat.data).any():
        coo = mat.tocoo()
        rows = coo.row[np.isnan(coo.data)]
        nan_rows[rows] = True
    return nan_rows


def _nan_rows(mat) -> np.ndarray:
    """含 NaN 的基因行掩码（稀疏数据通常无 NaN，该分支极少触发）。"""
    if sp.issparse(mat):
        return _sparse_nan_rows(mat)
    return np.isnan(np.asarray(mat)).any(axis=1)


def _gene_filter_mask(ref_mat, query_mat) -> np.ndarray:
    """对应 R 的基因过滤：``not.use = rowSums(is.na(ref))>0 | rowSums(is.na(sc))>0 | rowSums(ref)==0``。

    注意：仅过滤**参考**全零行（R 语义），查询全零行不剔除。
    """
    if sp.issparse(ref_mat):
        ref_nan = _sparse_nan_rows(ref_mat)
        ref_zero = np.asarray(ref_mat.sum(axis=1)).ravel() == 0
    else:
        ref_arr = np.asarray(ref_mat, dtype=float)
        ref_nan = np.isnan(ref_arr).any(axis=1)
        ref_zero = ref_arr.sum(axis=1) == 0
    return ref_nan | _nan_rows(query_mat) | ref_zero


def _coerce_ref_labels(ref_labels, ref_cells) -> np.ndarray:
    """将参考标签对齐到参考细胞列，返回 str 数组。

    标签索引与参考细胞 ID 一致时直接使用；否则按参考细胞 ID 重索引，
    重索引后出现缺失即报错（避免静默错位）。
    """
    labels = pd.Series(ref_labels)
    if ref_cells is not None:
        idx = pd.Index(ref_cells)
        if not labels.index.equals(idx):
            labels = labels.reindex(idx)
        if labels.isna().any():
            raise ValueError(
                "ref_labels 无法对齐到 ref_expr 的细胞列（存在缺失标签）"
            )
    return labels.astype(str).to_numpy()


def _cap_ref_cells(ref_labels: np.ndarray, cap: int, seed: int = 0) -> np.ndarray:
    """逐类型确定性抽样：每类型保留 ≤ ``cap`` 个参考细胞，返回列索引子集。

    类型集合不变（稀有类型原样保留）；同 seed 结果逐位可复现。供大参考
    提速使用（`singleR_annotate(max_cells_per_type=...)`）：首轮打分与
    fine-tuning 共用封顶后的参考池，cells 模式语义完全保留。
    """
    if cap < 1:
        raise ValueError("cap 必须 >= 1")
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for lab in dict.fromkeys(ref_labels.tolist()):  # 保序去重（按首次出现）
        idx = np.flatnonzero(ref_labels == lab)
        if idx.size > cap:
            idx = rng.permutation(idx)[:cap]
        keep.append(idx)
    return np.concatenate(keep)


def _intersect_and_filter(
    ref_mat, ref_genes: list, query_mat, query_genes: list,
) -> tuple[Any, Any, list]:
    """基因交集与质量过滤。

    对应 R ``SingleR()`` 前段：
    ``rownames(ref_data)=tolower(...); A=intersect(rownames(ref_data),rownames(sc_data))``，
    以及 ``not.use = rowSums(is.na(ref))>0 | rowSums(is.na(sc))>0 | rowSums(ref)==0``。

    返回 ``(ref_sub, query_sub, common_genes)``，行按参考基因顺序对齐；
    稀疏输入保持稀疏。
    """
    ref_low = [g.lower() if isinstance(g, str) else str(g) for g in ref_genes]
    query_low = [g.lower() if isinstance(g, str) else str(g) for g in query_genes]

    ref_map: dict[str, int] = {}
    for i, g in enumerate(ref_low):
        ref_map.setdefault(g, i)
    query_map: dict[str, int] = {}
    for i, g in enumerate(query_low):
        query_map.setdefault(g, i)
    common = [g for g in dict.fromkeys(ref_low) if g in query_map]
    if not common:
        raise ValueError("参考与查询之间不存在共同基因")

    ref_idx = np.array([ref_map[g] for g in common])
    query_idx = np.array([query_map[g] for g in common])
    ref_sub = _row_index(ref_mat, ref_idx)
    query_sub = _row_index(query_mat, query_idx)

    # R: 任一行含 NA 或参考行全为 0 则剔除该基因（参考全零、查询/参考 NaN）
    not_use = _gene_filter_mask(ref_sub, query_sub)
    ref_sub = _row_index(ref_sub, ~not_use)
    query_sub = _row_index(query_sub, ~not_use)
    common = [g for g, keep in zip(common, ~not_use) if keep]
    return ref_sub, query_sub, common


# ---------------------------------------------------------------------------
# 参考表达谱聚合与 DE 可变基因选择
# ---------------------------------------------------------------------------

def _median_matrix(
    ref_expr, ref_labels: np.ndarray, types: list | None = None, *, n_jobs: int = 1
) -> np.ndarray:
    """按细胞类型取行中位数，得到 (基因 × 类型) 中位表达谱。

    对应 R ``medianMatrix``：``rowMedians(mat[,A], na.rm=T)``；
    某类型仅一个细胞时直接取该细胞（``if(sum(A)==1) mat[,A]``）。

    稀疏输入按类型稠密化（内存上界 = 最大类型细胞数 × 基因数）；
    ``n_jobs > 1`` 时按类型并行计算（结果与串行逐位一致）。
    """
    if types is None:
        types = sorted(set(ref_labels.tolist()))
    is_sparse = sp.issparse(ref_expr)
    ref_csc = ref_expr.tocsc() if is_sparse else ref_expr
    col_sets = [np.where(ref_labels == t)[0] for t in types]
    if n_jobs > 1 and len(types) > 1:
        return _median_matrix_parallel(ref_csc, col_sets, n_jobs)
    med = np.empty((ref_expr.shape[0], len(types)), dtype=float)
    for k in range(len(types)):
        cols = col_sets[k]
        block = ref_csc[:, cols].toarray() if is_sparse else np.asarray(ref_csc)[:, cols]
        if cols.size == 1:
            med[:, k] = block[:, 0]
        else:
            med[:, k] = np.nanmedian(block, axis=1)
    return med


def _de_genes(
    median_mat: np.ndarray, gene_idx: np.ndarray, n_de_scale: int = 500
) -> np.ndarray:
    """两两细胞类型差异基因（genes == "de"）选择。

    对应 R ``SingleR()`` 与 ``fineTuningRound`` 中的 DE 分支：
    ``n = round(500*(2/3)^(log2(ncol(mat))))``；对每个有序类型对 (j, i) 取
    ``mat[,j]-mat[,i]`` 为正的前 n 个基因，全部取并集；最后丢弃并集首元素
    （对应 R ``unique(unlist(...))[-1]``）。

    ``gene_idx`` 为行索引数组，返回按出现顺序去重后的基因行索引。
    """
    k = median_mat.shape[1]
    n = round(n_de_scale * (2 / 3) ** np.log2(k))
    ordered: list[Any] = []
    for j in range(k):
        for i in range(k):
            diff = median_mat[:, j] - median_mat[:, i]
            order = np.argsort(-diff)                 # 降序排列
            pos = diff[order] > 0
            n_pos = int(pos.sum())
            if n_pos == 0:                            # 自对比等无正差异
                continue
            take = min(n, n_pos)
            ordered.extend(gene_idx[order[pos][:take]])
    genes_uniq = list(dict.fromkeys(ordered))         # 保序去重
    if len(genes_uniq) > 1:
        genes_uniq = genes_uniq[1:]                   # R: unique(...)[-1]
    return np.asarray(genes_uniq, dtype=int)


def select_hvg(expr, max_genes: int = 5000, n_bins: int = 20) -> np.ndarray:
    """高变基因（highly variable genes）选择，返回所选基因的行索引。

    对应 Seurat::FindVariableFeatures(method="vst") / scanpy 的
    ``highly_variable_genes(flavor="seurat")``：按基因平均表达分箱，
    箱内对 dispersion（方差/均值）做 z-score 标准化，取 top-N 基因。

    稀疏友好：均值/方差通过稀疏矩阵的求和与平方和计算，无需稠密化。

    Parameters:
        expr: 基因×细胞 表达矩阵（稠密 ``np.ndarray`` 或 ``scipy.sparse``）。
        max_genes: 保留的高变基因数，默认 5000；不足时保留全部有效基因。
        n_bins: 按均值分箱数（Seurat 默认 20）。

    Returns:
        所选基因的行索引（升序排列的 ``np.ndarray``）。
    """
    n_cells = expr.shape[1]
    if sp.issparse(expr):
        mean = np.asarray(expr.mean(axis=1)).ravel()
        sq = np.asarray(expr.power(2).mean(axis=1)).ravel()   # E[X^2]
    else:
        arr = np.asarray(expr, dtype=float)
        mean = arr.mean(axis=1)
        sq = (arr * arr).mean(axis=1)
    var = np.clip(sq - mean ** 2, 0.0, None)

    # dispersion = var / mean（Seurat vst），mean == 0 置 NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        dispersion = var / mean

    valid = np.isfinite(dispersion) & (mean > 0) & (var > 0)
    gene_idx = np.arange(expr.shape[0])
    valid_idx = gene_idx[valid]
    if valid_idx.size == 0:
        return gene_idx                                    # 无有效基因回退全部
    if valid_idx.size <= max_genes:
        return valid_idx                                   # 有效基因不足则全部保留

    # 按平均表达分箱
    valid_mean = mean[valid]
    edges = np.quantile(valid_mean, np.linspace(0, 1, n_bins + 1))
    bins = np.searchsorted(edges[1:-1], valid_mean, side="left")
    per_bin = max(1, int(np.ceil(max_genes / n_bins)))

    selected: list[int] = []
    for b in range(n_bins):
        in_bin = np.where(bins == b)[0]
        if in_bin.size == 0:
            continue
        d = dispersion[valid][in_bin]
        sd = d.std()
        z = (d - d.mean()) / sd if sd > 0 else np.zeros(in_bin.size)
        take = in_bin[np.argsort(-z)[:per_bin]]
        selected.extend(valid_idx[take].tolist())

    sel = np.array(sorted(set(selected)))                  # 去重并排序
    if sel.size < max_genes:                               # 不足则按 dispersion 补足
        leftover = np.setdiff1d(valid_idx, sel)
        order = leftover[np.argsort(-dispersion[leftover])]
        sel = np.concatenate([sel, order[: max_genes - sel.size]])
    return sel[:max_genes]


# ---------------------------------------------------------------------------
# Spearman 相关性（向量化）
# ---------------------------------------------------------------------------

def _spearman_corr(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """向量化 Spearman 相关矩阵。

    ``query`` (G, Q)、``ref`` (G, R)，返回 (Q, R)。对应 R
    ``cor(x, y, method='spearman')``：Spearman 即对列内基因排序后求 Pearson。

    零方差列（如全 0 表达）的相关性置 0，对应 ``cor.stable`` 的省略处理。
    """
    q_rank = stats.rankdata(query, axis=0)            # (G, Q) 基因内排序
    r_rank = stats.rankdata(ref, axis=0)              # (G, R)
    q_centered = q_rank - q_rank.mean(axis=0, keepdims=True)
    r_centered = r_rank - r_rank.mean(axis=0, keepdims=True)
    q_norm = np.sqrt(np.einsum("gq,gq->q", q_centered, q_centered))
    r_norm = np.sqrt(np.einsum("gr,gr->r", r_centered, r_centered))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = (q_centered.T @ r_centered) / np.outer(q_norm, r_norm)
    corr[q_norm == 0, :] = 0.0                        # 零方差查询细胞
    corr[:, r_norm == 0] = 0.0                        # 零方差参考细胞
    return corr


# ---------------------------------------------------------------------------
# 每细胞类型得分：top-N 相关样本均值
# ---------------------------------------------------------------------------

def _top_n_scores(
    corr: np.ndarray, ref_labels: np.ndarray, types: list, top_n: int = 5
) -> np.ndarray:
    """每个细胞类型取相关性最高的前 ``top_n`` 个参考细胞求均值作为得分。

    对应 R ``quantileMatrix``（R 用分位数聚合），本项目按需求改为
    top-N 均值聚合；类型内细胞数不足 top_n 时取全部细胞均值。
    返回 (Q, len(types))。
    """
    scores = np.empty((corr.shape[0], len(types)), dtype=float)
    for k, t in enumerate(types):
        cols = np.where(ref_labels == t)[0]
        if cols.size == 0:
            scores[:, k] = np.nan
            continue
        sub = corr[:, cols]
        n_keep = min(top_n, sub.shape[1])
        if n_keep == sub.shape[1]:
            scores[:, k] = sub.mean(axis=1)
        else:
            top = np.partition(sub, -n_keep, axis=1)[:, -n_keep:]
            scores[:, k] = top.mean(axis=1)
    return scores


def _score_profiles(
    query_mat,
    median_all: np.ndarray,
    types: list,
    *,
    chunk_size: int = 5000,
    n_jobs: int = 1,
) -> np.ndarray:
    """按类型中位表达谱相关性打分（SingleR 2.0 风格，大参考快速路径）。

    每个查询细胞只与各类型的**中位表达谱**求 Spearman 相关，复杂度
    O(n_query × n_types × n_genes)——远快于逐细胞相关的
    O(n_query × n_ref × n_genes)。``median_all`` 为 (基因 × 类型) 中位谱。

    对应现代 Bioconductor SingleR 的参考聚合打分思路；``n_jobs>1`` 时
    查询分块按进程并行。结果与分块/并行无关。
    """
    n_query = query_mat.shape[1]
    n_types = median_all.shape[1]
    scores = np.empty((n_query, n_types), dtype=float)

    # 中位谱一次排名并中心化
    med_rank = stats.rankdata(np.asarray(median_all, dtype=float), axis=0)
    med_centered = med_rank - med_rank.mean(axis=0, keepdims=True)
    med_norm = np.sqrt(np.einsum("gt,gt->t", med_centered, med_centered))
    del med_rank

    q_is_sparse = sp.issparse(query_mat)
    q_csc = query_mat.tocsc() if q_is_sparse else query_mat

    specs = [(start, min(start + chunk_size, n_query))
             for start in range(0, n_query, chunk_size)]
    if n_jobs > 1 and len(specs) > 1:
        global _SCORE_WORKER
        # 复用 _score_chunk 的 worker：把"类型列"当"参考列"即可
        _SCORE_WORKER = (med_centered, med_norm,
                         {t: np.array([i]) for i, t in enumerate(types)},
                         types, 1, q_csc, q_is_sparse)
        try:
            with ProcessPoolExecutor(
                max_workers=min(n_jobs, len(specs)), mp_context=_FORK_CTX
            ) as ex:
                for start, chunk_scores in ex.map(_score_chunk_worker, specs):
                    scores[start:start + chunk_scores.shape[0]] = chunk_scores
        finally:
            _SCORE_WORKER = None
        return scores

    with np.errstate(divide="ignore", invalid="ignore"):
        for start, stop in specs:
            q_block = q_csc[:, start:stop]
            q_block = np.asarray(q_block.toarray() if q_is_sparse else q_block, dtype=float)
            scores[start:stop] = _score_chunk(
                q_block, med_centered, med_norm,
                {t: np.array([i]) for i, t in enumerate(types)}, types, 1)
    return scores


def _score_chunk(
    q_block: np.ndarray,
    ref_centered: np.ndarray,
    ref_norm: np.ndarray,
    type_cols: dict,
    types: list,
    top_n: int,
) -> np.ndarray:
    """计算一个查询分块的每类型 top-N 得分（供串行循环与并行 worker 共用）。"""
    q_block = np.asarray(q_block, dtype=float)
    q_rank = stats.rankdata(q_block, axis=0)
    q_centered = q_rank - q_rank.mean(axis=0, keepdims=True)
    q_norm = np.sqrt(np.einsum("gq,gq->q", q_centered, q_centered))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = (q_centered.T @ ref_centered) / np.outer(q_norm, ref_norm)
    corr[q_norm == 0, :] = 0.0                        # 零方差查询细胞
    corr[:, ref_norm == 0] = 0.0                      # 零方差参考细胞
    scores = np.empty((q_block.shape[1], len(types)), dtype=float)
    for k, t in enumerate(types):
        sub = corr[:, type_cols[t]]
        n_keep = min(top_n, sub.shape[1])
        if n_keep == sub.shape[1]:
            scores[:, k] = sub.mean(axis=1)
        else:
            scores[:, k] = np.partition(sub, -n_keep, axis=1)[:, -n_keep:].mean(axis=1)
    return scores


def _score_chunk_worker(spec: tuple[int, int]) -> tuple[int, np.ndarray]:
    """并行分块打分 worker：``spec = (start, stop)``。"""
    with _limit_blas_threads():
        ref_centered, ref_norm, type_cols, types, top_n, q_csc, q_is_sparse = _SCORE_WORKER
        start, stop = spec
        q_block = q_csc[:, start:stop]
        q_block = q_block.toarray() if q_is_sparse else np.asarray(q_block)
        return start, _score_chunk(q_block, ref_centered, ref_norm, type_cols, types, top_n)


def _score_by_chunks(
    query_mat,
    ref_mat,
    ref_labels: np.ndarray,
    types: list,
    top_n: int = 5,
    chunk_size: int = 5000,
    n_jobs: int = 1,
) -> np.ndarray:
    """分块计算每细胞类型 top-N 得分，避免 (n_query × n_ref) 相关矩阵常驻内存。

    对应 R ``SingleR.ScoreData`` 的 ``step`` 分块思想：查询细胞按
    ``chunk_size`` 分块，逐块计算 Spearman 相关与 top-N 得分后丢弃相关矩阵。

    稀疏输入保持稀疏、按块稠密化；参考矩阵排名一次完成（参考规模通常适中，
    超大参考建议先 ``subset_cells`` 抽样）。

    ``n_jobs > 1`` 时查询分块按进程并行（fork 写时复制共享参考排名，worker
    各自 BLAS 限 1 线程）。分块与并行均不改变计算结果（逐列排序与逐块矩阵
    乘在数学上等价）。
    """
    n_query = query_mat.shape[1]
    scores = np.empty((n_query, len(types)), dtype=float)
    ref_is_sparse = sp.issparse(ref_mat)
    q_is_sparse = sp.issparse(query_mat)

    # 参考：一次性排名并中心化（对应 R 中对参考列求 rank）
    ref_dense = np.asarray(ref_mat.toarray() if ref_is_sparse else ref_mat, dtype=float)
    if n_jobs > 1 and ref_dense.shape[1] > 1:
        ref_rank = _rank_columns_parallel(ref_dense, n_jobs)
    else:
        ref_rank = stats.rankdata(ref_dense, axis=0)
    ref_centered = ref_rank - ref_rank.mean(axis=0, keepdims=True)
    ref_norm = np.sqrt(np.einsum("gr,gr->r", ref_centered, ref_centered))
    del ref_rank

    type_cols = {t: np.where(ref_labels == t)[0] for t in types}
    q_csc = query_mat.tocsc() if q_is_sparse else query_mat

    specs = [(start, min(start + chunk_size, n_query))
             for start in range(0, n_query, chunk_size)]
    if n_jobs > 1 and len(specs) > 1:
        global _SCORE_WORKER
        _SCORE_WORKER = (ref_centered, ref_norm, type_cols, types, top_n,
                         q_csc, q_is_sparse)
        try:
            with ProcessPoolExecutor(
                max_workers=min(n_jobs, len(specs)), mp_context=_FORK_CTX
            ) as ex:
                for start, chunk_scores in ex.map(_score_chunk_worker, specs):
                    scores[start:start + chunk_scores.shape[0]] = chunk_scores
        finally:
            _SCORE_WORKER = None
        return scores

    with np.errstate(divide="ignore", invalid="ignore"):
        for start, stop in specs:
            q_block = q_csc[:, start:stop]
            q_block = np.asarray(q_block.toarray() if q_is_sparse else q_block, dtype=float)
            scores[start:stop] = _score_chunk(
                q_block, ref_centered, ref_norm, type_cols, types, top_n)
    return scores


# ---------------------------------------------------------------------------
# 置信度：卡方离群检验
# ---------------------------------------------------------------------------

def _chisq_outlier_pvalue(scores: np.ndarray) -> np.ndarray:
    """每细胞对得分向量做卡方离群检验，返回 p 值向量。

    对应 R ``SingleR.ConfidenceTest`` -> ``outliers::chisq.out.test``：
    ``p = 1 - pchisq((max(x) - mean(x))^2 / var(x), 1)``。
    得分无差异（sd == 0）时 p 取 1。
    """
    x = np.asarray(scores, dtype=float)
    if x.shape[1] < 2:                                # 单类型得分无离群可检验
        return np.ones(x.shape[0])
    mx = x.max(axis=1)
    mu = x.mean(axis=1)
    sd = x.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = ((mx - mu) / sd) ** 2
    p = stats.chi2.sf(np.nan_to_num(stat, nan=0.0), 1)
    p[(sd == 0) | np.isnan(sd)] = 1.0                 # 无差异时 p 取 1
    return p


# ---------------------------------------------------------------------------
# fine-tuning 微调
# ---------------------------------------------------------------------------

def _slice_cell_genes(cells_mat, cell_rows, gene_cols) -> np.ndarray:
    """从 (细胞 × 基因) 矩阵取子矩阵并稠密化，返回 (基因 × 细胞)。

    ``cells_mat`` 应为已转换为 (细胞×基因) 的 csr（或稠密数组），避免每次
    调用重复转置。内存只稠密化 ``(n_cells × n_genes)`` 小矩阵——对候选类型
    细胞 × DE 基因，而非全部基因行（修复大参考下 fine-tuning 的 OOM）。
    """
    if sp.issparse(cells_mat):
        block = cells_mat[cell_rows][:, gene_cols].toarray()
    else:
        block = np.asarray(cells_mat)[cell_rows][:, gene_cols]
    return np.asarray(block, dtype=float).T


def _pairwise_de_genes(median_mat: np.ndarray, n_de_scale: int = 500) -> dict:
    """预计算所有有序类型对的 DE 基因行索引（一次 argsort 每对）。

    返回 ``{(j, i): np.ndarray}``，对应 R ``genes="de"`` 的逐对 top-n 正差异
    基因。后续任一候选类型集的 DE 基因 = 该集合内所有有序对的并集（保序去重
    去首），无需对每个候选集重复对全部基因 argsort（大参考 fine-tuning 的
    主要开销来源）。
    """
    k = median_mat.shape[1]
    n = round(n_de_scale * (2 / 3) ** np.log2(k))
    pairs: dict[tuple[int, int], np.ndarray] = {}
    for j in range(k):
        for i in range(k):
            diff = median_mat[:, j] - median_mat[:, i]
            order = np.argsort(-diff)                 # 降序
            pos = diff[order] > 0
            n_pos = int(pos.sum())
            if n_pos == 0:
                pairs[(j, i)] = np.array([], dtype=int)
            else:
                take = min(n, n_pos)
                pairs[(j, i)] = order[pos][:take]
    return pairs


def _de_genes_from_pairs(pairs: dict, cand_cols: list) -> np.ndarray:
    """从预计算的有序对 DE 基因取候选集并集（保序去重 + 去首，对应 R）。"""
    ordered: list[int] = []
    for j in cand_cols:
        for i in cand_cols:
            ordered.extend(pairs[(j, i)].tolist())
    genes_uniq = list(dict.fromkeys(ordered))         # 保序去重
    if len(genes_uniq) > 1:
        genes_uniq = genes_uniq[1:]                   # R: unique(...)[-1]
    return np.asarray(genes_uniq, dtype=int)


def _process_fine_tune_group(
    cand: tuple,
    cells: np.ndarray,
    ref_csr,
    query_csr,
    ref_labels: np.ndarray,
    median_all: np.ndarray,
    type_col: dict,
    top_n: int,
    fine_tune_thres: float,
    min_genes: int,
    n_de_scale: int,
    pairwise_de: dict | None = None,
) -> tuple[tuple, np.ndarray, dict, np.ndarray | None]:
    """处理一个候选类型组（一轮微调），供串行循环与并行 worker 共用。

    返回 ``(cand, cells, result, cand_scores)``：``result`` 为 {细胞 ->
    ("label", 类型) 或 ("continue", 新候选元组)}；``cand_scores`` 为
    (n_cells, len(cand)) 的该轮得分（差异基因不足时返回 ``None``）。
    """
    k = len(cand)
    ref_mask = np.isin(ref_labels, list(cand))
    if pairwise_de is not None:
        de = _de_genes_from_pairs(pairwise_de, [type_col[t] for t in cand])
    else:
        med = median_all[:, [type_col[t] for t in cand]]      # R: mean_mat[,topLabels]
        de = _de_genes(med, np.arange(median_all.shape[0]), n_de_scale)
    result: dict[int, tuple] = {}
    if de.size < min_genes:                                   # R: length(genes.filtered)<20
        for c in cells:
            result[int(c)] = ("label", cand[0])
        return cand, cells, result, None
    q_sub = _slice_cell_genes(query_csr, cells, de)           # (n_de, n_cells)
    r_sub = _slice_cell_genes(ref_csr, np.where(ref_mask)[0], de)  # (n_de, n_ref_cand)
    corr = _spearman_corr(q_sub, r_sub)
    cand_scores = _top_n_scores(corr, ref_labels[ref_mask], list(cand), top_n)
    q_sd = q_sub.std(axis=0, ddof=1)
    for ci, c in enumerate(cells):
        if q_sd[ci] == 0:                                     # R: sd(sc_data.filtered)==0
            result[int(c)] = ("label", cand[0])
            continue
        s = cand_scores[ci]
        order = np.argsort(-s)                                # R: sort(agg_scores, decreasing=T)
        s_keep = s[order][:-1]                                # R: agg_scores[-length(...)]
        keep = s_keep >= (s_keep[0] - fine_tune_thres)
        new_cand = [cand[order[j]] for j in range(k - 1) if keep[j]]
        if len(new_cand) <= 1:
            result[int(c)] = ("label", new_cand[0] if new_cand else cand[int(order[0])])
        else:
            result[int(c)] = ("continue", tuple(new_cand))
    return cand, cells, result, cand_scores


def _fine_tune_group_worker(spec) -> tuple:
    """并行 fine-tuning 组 worker：``spec = (cand, cells)``。"""
    with _limit_blas_threads():
        cand, cells = spec
        (ref_csr, query_csr, ref_labels, median_all, type_col,
         top_n, fine_tune_thres, min_genes, n_de_scale, pairwise_de) = _FT_WORKER
        return _process_fine_tune_group(
            cand, cells, ref_csr, query_csr, ref_labels, median_all, type_col,
            top_n, fine_tune_thres, min_genes, n_de_scale, pairwise_de)


def _fine_tune(
    ref_expr,
    ref_labels: np.ndarray,
    query_expr,
    first_scores: np.ndarray,
    types: list,
    *,
    median_all: np.ndarray | None = None,
    pairwise_de: dict | None = None,
    top_n: int = 5,
    fine_tune_thres: float = 0.05,
    min_genes: int = 20,
    n_de_scale: int = 500,
    n_jobs: int = 1,
    fine_tune_chunk: int = 2500,
) -> tuple[np.ndarray, np.ndarray]:
    """fine-tuning 迭代（对应 R ``SingleR.FineTune`` + ``fineTuningRound``）。

    流程（与 R 逐细胞执行等价，此处按候选类型分组批处理）：
    1. 初始候选集 = 得分 >= 最高分 - ``fine_tune_thres`` 的类型；
    2. 迭代直至每细胞仅剩 1 个候选类型：
       - 以候选类型内的参考细胞构建中位表达谱，挑选两两差异基因；
       - 若差异基因 < ``min_genes``（R 为 20）则直接取候选集首类型；
       - 计算该细胞与候选参考细胞的 Spearman 相关并按 top-N 重新打分；
       - 排序后丢弃最差类型，保留与最高分差距 <= ``fine_tune_thres`` 的类型。

    稀疏输入按分组稠密化，内存开销受限于候选类型细胞数。

    性能说明：``median_all``（全部类型的中位表达谱）由上层**预计算一次**
    传入，候选组内只做列切片——避免为每个候选组重复 ``tocsc()`` 与
    中位数计算（R 原版正是如此：``SingleR()`` 先算 ``medianMatrix`` 一次，
    ``fineTuningRound`` 用 ``mean_mat[, topLabels]`` 切片）。

    fine_tune_chunk: 微调并行时单个候选组的查询细胞分块上限，默认 2500。
        巨型候选组（T/NK 等）拆块后多 worker 并行；分块不改变计算结果，
        仅当 n_jobs>1 时生效。

    返回 ``(labels, final_scores)``：labels 为最终标签（object 数组），
    final_scores 中参与微调的细胞其候选类型得分更新为最后一轮得分。
    """
    if fine_tune_chunk < 1:
        raise ValueError("fine_tune_chunk 必须 >= 1")
    n_query = query_expr.shape[1]
    types_arr = np.asarray(types, dtype=object)
    type_col = {t: i for i, t in enumerate(types)}
    labels = np.empty(n_query, dtype=object)
    final_scores = np.array(first_scores, copy=True)

    # 稀疏输入转 (细胞×基因) csr 一次（细胞行访问快），供所有候选组切片复用
    ref_csr = ref_expr.tocsc().T.tocsr() if sp.issparse(ref_expr) else ref_expr.T
    query_csr = query_expr.tocsc().T.tocsr() if sp.issparse(query_expr) else query_expr.T
    # 全部类型的中位表达谱：预计算一次，候选组内按列切片（R: mean_mat[,topLabels]）
    if median_all is None:
        median_all = _median_matrix(ref_expr, ref_labels, types)

    # R: topLabels = names(scores[i, scores[i,] >= max_score - fine.tune.thres])
    max_scores = first_scores.max(axis=1)
    cand_mask = first_scores >= (max_scores[:, None] - fine_tune_thres)
    n_cand = cand_mask.sum(axis=1)
    for i in range(n_query):
        if n_cand[i] <= 1:
            if n_cand[i] == 1:
                labels[i] = types_arr[int(np.flatnonzero(cand_mask[i])[0])]
            else:                                     # R: length(topLabels)==0 -> argmax
                labels[i] = types_arr[int(np.argmax(first_scores[i]))]

    active = np.where(n_cand > 1)[0]
    active_cands = [tuple(types_arr[np.flatnonzero(cand_mask[i])]) for i in active]

    while active.size:
        groups: dict[tuple, np.ndarray] = defaultdict(list)
        for idx, cand in zip(active, active_cands):
            groups[cand].append(int(idx))
        next_active: list[int] = []
        next_cands: list[tuple] = []

        # 组间相互独立：n_jobs>1 时并行处理（各 worker BLAS 限 1 线程）。
        # 巨型组（单组数万查询细胞）按 fine_tune_chunk 拆块后进 worker 池，
        # 结果逐位不变——分块只影响调度粒度（len(groups)==1 时也可并行）。
        if n_jobs > 1:
            global _FT_WORKER
            _FT_WORKER = (ref_csr, query_csr, ref_labels, median_all, type_col,
                          top_n, fine_tune_thres, min_genes, n_de_scale,
                          pairwise_de)
            try:
                specs = [
                    (cand, chunk)
                    for cand, cells in groups.items()
                    for chunk in (
                        np.asarray(cells, dtype=np.int64)[i:i + fine_tune_chunk]
                        for i in range(0, len(cells), fine_tune_chunk)
                    )
                ]
                with ProcessPoolExecutor(
                    max_workers=min(n_jobs, len(specs)), mp_context=_FORK_CTX
                ) as ex:
                    outcomes = list(ex.map(_fine_tune_group_worker, specs))
            finally:
                _FT_WORKER = None
        else:
            outcomes = [
                _process_fine_tune_group(
                    cand, np.asarray(cells, dtype=np.int64),
                    ref_csr, query_csr, ref_labels, median_all, type_col,
                    top_n, fine_tune_thres, min_genes, n_de_scale, pairwise_de)
                for cand, cells in groups.items()
            ]

        # 汇总各组的标签/候选更新与得分
        for cand, cells, result, cand_scores in outcomes:
            if cand_scores is not None:
                cand_cols = [type_col[t] for t in cand]
                final_scores[np.ix_(cells, cand_cols)] = cand_scores
            for c in cells:
                action, val = result[int(c)]
                if action == "label":
                    labels[c] = val
                else:
                    next_active.append(c)
                    next_cands.append(val)

        active = np.asarray(next_active, dtype=np.int64)
        active_cands = next_cands

    return labels, final_scores


def _fine_tune_profiles(
    query_expr,
    median_all: np.ndarray,
    first_scores: np.ndarray,
    types: list,
    *,
    pairwise_de: dict | None = None,
    fine_tune_thres: float = 0.05,
    min_genes: int = 20,
    n_de_scale: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """profile 模式的微调：候选类型间用 DE 基因 + 中位表达谱重新打分。

    与 :func:`_fine_tune` 同为"候选类型迭代精炼"（初始候选 -> DE 基因 ->
    重打分 -> 丢弃最差 -> 阈值收敛），但相关性对象是**各类型的聚合中位谱**
    而非逐细胞，复杂度 O(n_query × n_types × n_genes)，适配数万细胞的大参考
    （对应 SingleR 2.0 的聚合打分思路）。用于 ``scoring="profile"``。

    返回 ``(labels, final_scores)``。
    """
    n_query = query_expr.shape[1]
    types_arr = np.asarray(types, dtype=object)
    type_col = {t: i for i, t in enumerate(types)}
    labels = np.empty(n_query, dtype=object)
    final_scores = np.array(first_scores, copy=True)

    max_scores = first_scores.max(axis=1)
    cand_mask = first_scores >= (max_scores[:, None] - fine_tune_thres)
    n_cand = cand_mask.sum(axis=1)
    for i in range(n_query):
        if n_cand[i] <= 1:
            if n_cand[i] == 1:
                labels[i] = types_arr[int(np.flatnonzero(cand_mask[i])[0])]
            else:
                labels[i] = types_arr[int(np.argmax(first_scores[i]))]

    active = np.where(n_cand > 1)[0]
    active_cands = [tuple(types_arr[np.flatnonzero(cand_mask[i])]) for i in active]
    # (细胞×基因) csr：_slice_cell_genes 期望行=细胞
    q_csr = query_expr.tocsc().T.tocsr() if sp.issparse(query_expr) else query_expr.T

    while active.size:
        groups: dict[tuple, np.ndarray] = defaultdict(list)
        for idx, cand in zip(active, active_cands):
            groups[cand].append(int(idx))
        next_active: list[int] = []
        next_cands: list[tuple] = []

        for cand, cells in groups.items():
            cells = np.asarray(cells, dtype=np.int64)
            cand = tuple(cand)
            k = len(cand)
            if pairwise_de is not None:
                # 预计算的有序对 DE 基因 -> 候选集并集（O(K^2) 查找，非 argsort）
                de = _de_genes_from_pairs(pairwise_de, [type_col[t] for t in cand])
            else:
                med_cand = median_all[:, [type_col[t] for t in cand]]
                de = _de_genes(med_cand, np.arange(median_all.shape[0]), n_de_scale)
            if de.size < min_genes:                       # R: length(genes.filtered)<20
                for c in cells:
                    labels[c] = cand[0]
                continue
            q_sub = _slice_cell_genes(q_csr, cells, de)   # (n_de, n_cells)
            if pairwise_de is not None:
                med_de = median_all[de][:, [type_col[t] for t in cand]]  # (n_de, K)
            else:
                med_de = np.asarray(med_cand)[de]         # (n_de, K)
            corr = _spearman_corr(q_sub, med_de)          # (n_cells, K) 聚合谱相关
            final_scores[np.ix_(cells, [type_col[t] for t in cand])] = corr
            for ci, c in enumerate(cells):
                s = corr[ci]
                order = np.argsort(-s)                    # 降序
                s_keep = s[order][:-1]                    # 丢弃最差
                keep = s_keep >= (s_keep[0] - fine_tune_thres)
                new_cand = [cand[order[j]] for j in range(k - 1) if keep[j]]
                if len(new_cand) <= 1:
                    labels[c] = new_cand[0] if new_cand else cand[int(order[0])]
                else:
                    next_active.append(c)
                    next_cands.append(tuple(new_cand))
        active = np.asarray(next_active, dtype=np.int64)
        active_cands = next_cands
    return labels, final_scores


# ---------------------------------------------------------------------------
# 首轮基因选择
# ---------------------------------------------------------------------------

def _select_genes(
    ref_sub,
    ref_label_arr: np.ndarray,
    types: list,
    gene_selection: str,
    max_genes: int,
    min_genes: int,
    n_de_scale: int,
    median_all: np.ndarray | None = None,
) -> np.ndarray:
    """首轮打分所用的基因选择。

    - ``"hvg"``（默认）：高变基因 top ``max_genes``（稀疏友好，默认 5000）；
    - ``"de"``：两两细胞类型差异基因（对应 R ``genes="de"``）；
    - ``"sd"``：中位表达谱标准差阈值法（对应 R ``genes="sd"``）；
    - ``"all"``：全部共有基因。

    ``median_all`` 为预计算的全类型中位表达谱（避免重复计算）。
    所选基因数不足 ``min_genes`` 时回退到全部基因，避免病态输入。
    """
    if gene_selection == "all":
        return np.arange(ref_sub.shape[0])
    if gene_selection == "hvg":
        idx = select_hvg(ref_sub, max_genes)
        return idx if idx.size >= min_genes else np.arange(ref_sub.shape[0])
    if median_all is None:
        median_all = _median_matrix(ref_sub, ref_label_arr, types)
    if gene_selection == "de":
        idx = _de_genes(median_all, np.arange(ref_sub.shape[0]), n_de_scale)
    elif gene_selection == "sd":
        idx = np.where(median_all.std(axis=1) > 1.0)[0]   # R: sd.thres=1
    else:
        raise ValueError(
            f"未知 gene_selection: {gene_selection!r}，可选 'hvg'/'de'/'sd'/'all'"
        )
    return idx if idx.size >= min_genes else np.arange(ref_sub.shape[0])


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def singleR_annotate(
    ref_expr,
    ref_labels,
    query_expr,
    fine_tune: bool = True,
    top_n: int = 5,
    *,
    gene_selection: str = "hvg",
    max_genes: int = 5000,
    use_de_genes: bool | None = None,
    scoring: str = "cells",
    fine_tune_thres: float = 0.05,
    min_genes: int = 20,
    n_de_scale: int = 500,
    chunk_size: int = 5000,
    n_jobs: int = 1,
    score_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    max_cells_per_type: int | None = None,
    ref_cap_seed: int = 0,
) -> dict[str, Any]:
    """单细胞细胞类型自动注释（SingleR 核心算法）。

    Parameters:
        ref_expr: 参考表达矩阵，``pd.DataFrame``（行=基因名，列=细胞 ID）、
            ``anndata.AnnData``（X 为 细胞×基因，自动转置）、``np.ndarray``
            或 ``scipy.sparse``（后两者按 (基因×细胞) 约定）。
        ref_labels: 参考细胞类型标签，``pd.Series``，索引与参考细胞 ID 对齐。
        query_expr: 查询表达矩阵，格式同 ``ref_expr``。
        fine_tune: 是否启用 fine-tuning 微调迭代，默认 ``True``。
            仅当参考类型数 > 2 时才执行（与 R 一致）。
        top_n: 每个细胞类型取相关性最高的前 ``top_n`` 个参考细胞计算得分，默认 5。
        gene_selection: 首轮打分所用的基因选择方法：
            - ``"hvg"``（默认）：高变基因 top ``max_genes``（Seurat vst 风格，
              稀疏友好，过滤参考/查询中的低信息量基因）；
            - ``"de"``：两两细胞类型差异基因（对应 R ``genes="de"``）；
            - ``"sd"``：中位表达谱标准差阈值法（对应 R ``genes="sd"``）；
            - ``"all"``：全部共有基因。
        max_genes: ``gene_selection="hvg"`` 时保留的高变基因数，默认 5000。
        use_de_genes: 兼容旧参数；显式给定时代替 ``gene_selection``
            （``True`` -> ``"de"``，``False`` -> ``"all"``）。
        fine_tune_thres: 微调中"候选类型"判定阈值（与最高分差距），
            对应 R 的 ``fine.tune.thres``，默认 0.05。
        min_genes: 微调轮中差异基因数下限，不足则提前终止该细胞，
            对应 R ``length(genes.filtered) < 20``，默认 20。
        n_de_scale: DE 基因数量基准常数，对应 R ``500*(2/3)^(log2(K))``，默认 500。
        chunk_size: 首轮打分时的查询细胞分块大小，默认 5000。分块使
            (n_query × n_ref) 相关矩阵不常驻内存，支撑 10w+ 查询细胞；
            分块不改变计算结果。
        scoring: 首轮打分方式：
            - ``"cells"``（默认）：逐细胞 Spearman 相关 + 每类型 top-N 相关
              样本均值（经典 SingleR，O(n_query × n_ref × n_genes)）；
            - ``"profile"``：按类型中位表达谱相关性打分（SingleR 2.0 风格，
              O(n_query × n_types × n_genes)，**大参考（数万细胞）下显著更快**）。
        n_jobs: 并行进程数，默认 1（串行）。>1 时首轮分块打分与参考中位
            表达谱按进程并行（Linux fork + 写时复制共享大数组，每进程 BLAS
            限 1 线程，避免核数争抢），加速大样本注释；不改变计算结果。
        score_transform: 可选，作用于**首轮得分矩阵**（(Q × 类型) ndarray，
            列按类型字母序）的变换函数，返回同形状数组。在置信度检验与
            fine-tuning 之前应用，用于把先验/证据融合进打分（如
            ``lambda S: S * (1 + lam * prior[:, None])``）。传 ``None`` 时
            行为与旧版完全一致（含 ``all_scores`` / ``pval`` 语义）。
        max_cells_per_type: 可选，每参考类型最多保留的细胞数（None=全量）。
            大参考提速：首轮打分与 fine-tuning 共用封顶后的参考池，cells 模式
            语义保留，成本上界 O(n_query × n_types × cap × n_genes)。类型集合不变，稀有
            类型原样保留。
        ref_cap_seed: 封顶抽样的随机种子，默认 0（同种子结果逐位可复现）。

    Returns:
        字典：
        - ``labels``: ``pd.Series``，每个查询细胞的最终预测细胞类型；
        - ``scores``: ``pd.DataFrame``，每个细胞对应各类型的最终得分；
        - ``all_scores``: ``pd.DataFrame``，首轮全部细胞类型的原始得分矩阵；
        - ``pval``: ``pd.Series``，首轮得分的卡方离群检验 p 值（置信度，
          可据此将 ``pval > 阈值`` 的细胞标记为低置信）；
        - ``labels1``: ``pd.Series``，微调前（首轮）的预测标签。
    """
    if top_n < 1:
        raise ValueError("top_n 必须 >= 1")
    if fine_tune_thres < 0:
        raise ValueError("fine_tune_thres 必须 >= 0")
    if chunk_size < 1:
        raise ValueError("chunk_size 必须 >= 1")
    if max_cells_per_type is not None and max_cells_per_type < 1:
        raise ValueError("max_cells_per_type 必须 >= 1")

    # 0. 输入统一为 (基因 × 细胞) 矩阵（稀疏保持稀疏）
    ref_mat, ref_genes, ref_cells = _expr_to_mat(ref_expr, "ref_expr")
    query_mat, query_genes, query_cells = _expr_to_mat(query_expr, "query_expr")
    if ref_mat.shape[0] == 0 or ref_mat.shape[1] == 0:
        raise ValueError("参考表达矩阵为空")
    if query_mat.shape[1] == 0:
        raise ValueError("查询表达矩阵无细胞")
    ref_label_arr = _coerce_ref_labels(ref_labels, ref_cells)

    # 0b. 逐类型参考封顶（可选）：每类型确定性抽样至 ≤ max_cells_per_type，
    #     一次成型，首轮打分与 fine-tuning 共用；类型集合不变
    if max_cells_per_type is not None:
        keep = _cap_ref_cells(ref_label_arr, max_cells_per_type, ref_cap_seed)
        if keep.size < ref_mat.shape[1]:
            ref_mat = ref_mat[:, keep]
            if ref_cells is not None:
                ref_cells = [ref_cells[i] for i in keep]
            ref_label_arr = ref_label_arr[keep]

    # 1. 基因交集与质量过滤（R: tolower + intersect + NA/零行过滤）
    if ref_genes is not None and query_genes is not None:
        ref_sub, query_sub, _ = _intersect_and_filter(
            ref_mat, ref_genes, query_mat, query_genes
        )
    else:
        # 无基因名的 ndarray/稀疏输入：要求行数一致，按位置对齐
        if ref_mat.shape[0] != query_mat.shape[0]:
            raise ValueError("缺少基因名时，参考与查询必须行数一致")
        not_use = _gene_filter_mask(ref_mat, query_mat)
        ref_sub = _row_index(ref_mat, ~not_use)
        query_sub = _row_index(query_mat, ~not_use)
    if ref_sub.shape[0] == 0:
        raise ValueError("基因交集过滤后无可用基因")

    # 2. 类型列表按字母序（对应 R: levels(factor(types))）
    types = sorted(set(ref_label_arr.tolist()))
    if len(types) < 1:
        raise ValueError("参考数据不包含任何细胞类型")

    # 3. 首轮基因选择与分块打分（R: genes.filtered -> SingleR.ScoreData）
    if use_de_genes is not None:                                  # 兼容旧参数
        gene_selection = "de" if use_de_genes else "all"

    # 全部类型中位表达谱：预计算一次，供 DE 基因选择与 fine-tuning 复用
    # （对应 R: mat = medianMatrix(ref_data, types)，fineTuningRound 只做切片）
    median_all = _median_matrix(ref_sub, ref_label_arr, types, n_jobs=n_jobs)

    # 预计算所有类型对的 DE 基因（一次 argsort 每对），供 round-1 与
    # fine-tuning 复用（大参考下避免每个候选组重复 argsort）
    pairwise_de = _pairwise_de_genes(median_all, n_de_scale)

    gene_idx = _select_genes(
        ref_sub, ref_label_arr, types,
        gene_selection=gene_selection, max_genes=max_genes,
        min_genes=min_genes, n_de_scale=n_de_scale,
        median_all=median_all,
    )

    if scoring == "cells":
        all_scores = _score_by_chunks(
            _row_index(query_sub, gene_idx), _row_index(ref_sub, gene_idx),
            ref_label_arr, types, top_n, chunk_size, n_jobs=n_jobs,
        )
    elif scoring == "profile":
        all_scores = _score_profiles(
            _row_index(query_sub, gene_idx),
            median_all[gene_idx],
            types, chunk_size=chunk_size, n_jobs=n_jobs,
        )
    else:
        raise ValueError(f"未知 scoring: {scoring!r}，可选 'cells'/'profile'")

    # 3b. 可选先验变换：把 DB 证据融合进首轮得分，再进入置信度与 fine-tuning
    if score_transform is not None:
        all_scores = np.asarray(score_transform(all_scores), dtype=np.float64)
        if all_scores.shape != (query_mat.shape[1], len(types)):
            raise ValueError("score_transform 必须返回同形状得分矩阵")

    # 4. 置信度 p 值（R: SingleR.ConfidenceTest，针对首轮得分）
    pval = _chisq_outlier_pvalue(all_scores)

    # 5. 首轮标签 + fine-tuning 微调（R: SingleR.FineTune，仅类型数 > 2 时）
    labels1 = np.asarray(types, dtype=object)[np.argmax(all_scores, axis=1)]
    if fine_tune and len(types) > 2:
        if scoring == "profile":
            # 聚合谱微调：全程无逐细胞相关，O(Q×n_types×G)，大参考快速
            final_labels, final_scores = _fine_tune_profiles(
                query_sub, median_all, all_scores, types,
                pairwise_de=pairwise_de,
                fine_tune_thres=fine_tune_thres,
                min_genes=min_genes, n_de_scale=n_de_scale,
            )
        else:
            final_labels, final_scores = _fine_tune(
                ref_sub, ref_label_arr, query_sub, all_scores, types,
                median_all=median_all, pairwise_de=pairwise_de,
                top_n=top_n, fine_tune_thres=fine_tune_thres,
                min_genes=min_genes, n_de_scale=n_de_scale, n_jobs=n_jobs,
            )
    else:
        final_labels = labels1.copy()
        final_scores = np.array(all_scores, copy=True)

    # 6. 组装输出
    if query_cells is None:
        query_cells = list(range(query_mat.shape[1]))
    index = pd.Index(query_cells)
    return {
        "labels": pd.Series(final_labels, index=index, name="label"),
        "scores": pd.DataFrame(final_scores, index=index, columns=types),
        "all_scores": pd.DataFrame(all_scores, index=index, columns=types),
        "pval": pd.Series(pval, index=index, name="pval"),
        "labels1": pd.Series(labels1, index=index, name="label1"),
    }


# ---------------------------------------------------------------------------
# 多参考（多数据库）注释
# ---------------------------------------------------------------------------

def _combine_scores(
    scores_list: list[pd.DataFrame], method: str = "max"
) -> pd.DataFrame:
    """合并多参考得分（对应 SingleR 2.0 ``combineResults``）。

    对每个细胞类型标签（所有参考标签的并集）：
    - ``method="max"``（默认）：跨参考取最大得分；缺少该标签的参考贡献
      ``-inf``，被忽略；
    - ``method="mean"``：跨参考取平均得分；缺少该标签的参考贡献 ``NaN``，
      被忽略。

    返回 (细胞 × 标签并集) 的得分 ``DataFrame``。
    """
    if len(scores_list) == 1:
        return scores_list[0].copy()
    if method not in ("max", "mean"):
        raise ValueError(f"未知 combine_method: {method!r}，可选 'max'/'mean'")
    all_labels = sorted(set().union(*[set(s.columns) for s in scores_list]))
    n_cells = len(scores_list[0])
    combined = pd.DataFrame(
        index=scores_list[0].index, columns=all_labels, dtype=float
    )
    for lab in all_labels:
        cols = []
        for s in scores_list:
            if lab in s.columns:
                cols.append(s[lab].to_numpy())
            else:
                cols.append(np.full(
                    n_cells, np.nan if method == "mean" else -np.inf))
        arr = np.column_stack(cols)
        if method == "max":
            combined[lab] = np.nanmax(arr, axis=1)
        else:
            combined[lab] = np.nanmean(arr, axis=1)
    return combined


def singleR_annotate_multi(
    references,
    query_expr,
    *,
    combine_method: str = "max",
    fine_tune: bool = True,
    top_n: int = 5,
    gene_selection: str = "hvg",
    max_genes: int = 5000,
    chunk_size: int = 5000,
    n_jobs: int = 1,
    **kwargs,
) -> dict[str, Any]:
    """多参考（多数据库）细胞类型注释。

    对每个参考数据**分别**执行 :func:`singleR_annotate`，再按
    ``combine_method`` 跨参考合并为共识标签。对应现代 Bioconductor
    ``SingleR(test, ref=list(...), labels=list(...))`` 的行为。

    Parameters:
        references: 参考列表，每个元素为 ``(ref_expr, ref_labels)`` 元组，
            格式与 :func:`singleR_annotate` 的 ``ref_expr``/``ref_labels``
            一致。各参考可有**不同的基因空间**（自动分别与查询取交集）。
        combine_method: 跨参考合并方式，``"max"``（默认）或 ``"mean"``。
        fine_tune: 每个参考是否启用 fine-tuning，默认 ``True``。
        top_n: 每个参考每类型取相关性最高的前 ``top_n`` 个参考细胞，默认 5。
        gene_selection / max_genes / chunk_size / n_jobs: 透传给每个参考。
        **kwargs: 透传给每个参考的 :func:`singleR_annotate` 的其余参数。

    Returns:
        字典：
        - ``labels``: 合并后的共识预测标签（``pd.Series``）；
        - ``scores``: 跨参考合并的最终得分（``pd.DataFrame``，列=标签并集）；
        - ``all_scores``: 跨参考合并的首轮得分（``pd.DataFrame``）；
        - ``per_reference``: 每个参考的完整 ``singleR_annotate`` 结果列表。
    """
    if not references:
        raise ValueError("references 不能为空")
    per_ref = [
        singleR_annotate(
            ref_expr, ref_labels, query_expr,
            fine_tune=fine_tune, top_n=top_n,
            gene_selection=gene_selection, max_genes=max_genes,
            chunk_size=chunk_size, n_jobs=n_jobs, **kwargs,
        )
        for ref_expr, ref_labels in references
    ]
    scores = _combine_scores([r["scores"] for r in per_ref], combine_method)
    all_scores = _combine_scores([r["all_scores"] for r in per_ref], combine_method)
    labels = scores.idxmax(axis=1)
    labels.name = "label"
    return {
        "labels": labels,
        "scores": scores,
        "all_scores": all_scores,
        "per_reference": per_ref,
    }
