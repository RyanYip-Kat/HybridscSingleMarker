"""Seurat 参考映射标签转移注释（对应 Seurat::FindTransferAnchors + TransferData）。

基于 anndata 的纯 Python 实现，算法流程（逐节点对照 Seurat 源码）：

1. 共享可变基因（HVG）选择        -> 对应 ``FindVariableFeatures`` 的 vst 思路
2. 逐基因 z-score 标准化          -> 对应 ``ScaleData``
3. 低维嵌入：CCA（经典）或参考 PCA 投影
   - CCA（默认）                  -> 对应 ``RunCCA.default``：
     ``M = crossprod(scale(ref), scale(query))`` 后 SVD，取规范相关向量
     ``[u; v]`` 作为嵌入，符号校正使每维首元素为正；
   - ``pcaproject``               -> 对应 ``ProjectCellEmbeddings``：
     参考 PCA + 查询投影到参考 PCA 空间；
4. 嵌入 L2 归一化（每细胞）       -> 对应 ``L2Dim``（cca.l2 / pcaproject.l2）
5. kNN 图（ref→query, query→ref, 组内） -> 对应 ``FindNN``（k=30）
6. MNN anchors                    -> 对应 ``FindAnchorPairs``（k.anchor=5）
7. anchor 得分                    -> 对应 ``ScoreAnchors``（k.score=30）：
   共享邻居数经 0.9/0.01 分位归一化到 [0,1]
8. 权重 + 标签预测                -> 对应 ``TransferData`` / ``FindWeights`` /
   ``GetTransferPredictions``：
   ``weight = 1 - exp(-proximity * anchor_score / (2/sd)^2)`` 列归一化，
   ``scores = weights^T %*% onehot(ref_labels)``，argmax 得预测标签。

参考：Stuart, Butler, et al. Cell 2019 (https://doi.org/10.1016/j.cell.2019.05.031)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.sparse.linalg import LinearOperator, svds
from scipy.spatial import cKDTree

from .core import _coerce_ref_labels, _expr_to_mat, _intersect_and_filter, select_hvg

# 可选的 C++ 加速扩展（对应 Seurat src/integration.cpp FindWeightsC）；
# 未编译时回退到 numpy 实现
try:
    from ._fastseurat import find_weights_c as _find_weights_c_fast
except ImportError:                                   # pragma: no cover
    _find_weights_c_fast = None


# ---------------------------------------------------------------------------
# 预处理：基因选择 / 标准化
# ---------------------------------------------------------------------------

def _scale_genes(mat: np.ndarray, max_value: float = 10.0) -> np.ndarray:
    """逐基因 z-score（对应 Seurat ``ScaleData``）。``mat``: (基因, 细胞)。"""
    mean = mat.mean(axis=1, keepdims=True)
    sd = mat.std(axis=1, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0                                  # 零方差基因保持 0
    return np.clip((mat - mean) / sd, -max_value, max_value)


def _standardize_cells(mat: np.ndarray) -> np.ndarray:
    """逐细胞 z-score（对应 Seurat C++ ``Standardize``，用于 CCA 前处理）。"""
    mean = mat.mean(axis=0, keepdims=True)
    sd = mat.std(axis=0, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (mat - mean) / sd


# ---------------------------------------------------------------------------
# 低维嵌入：CCA / pcaproject
# ---------------------------------------------------------------------------

def _compute_cca(data1: np.ndarray, data2: np.ndarray, num_cc: int) -> np.ndarray:
    """规范相关分析（对应 ``RunCCA.default``）。

    ``data1/data2``: (基因, 细胞)。流程：逐细胞 z-score -> ``M = data1^T %*% data2``
    -> SVD 取前 ``num_cc`` 个规范相关向量 -> 嵌入 ``[u; v]``（所有细胞），
    符号校正使每维首元素为正。

    **内存优化**：``M = d1^T %*% d2`` 是 (cells1 × cells2) 矩阵，大样本下
    内存/计算量巨大。改用 ``LinearOperator`` 表达 ``M``（``M@v = d1^T@(d2@v)``），
    ``svds`` 只做矩阵-向量乘，**不物化 (cells1 × cells2) 交叉协方差矩阵**——
    内存从 O(cells1×cells2) 降到 O(genes×(cells1+cells2))，计算量约降一个量级。
    """
    d1 = _standardize_cells(data1)
    d2 = _standardize_cells(data2)
    n1, n2 = d1.shape[1], d2.shape[1]
    k = min(num_cc, n1, n2)
    if k == 0:
        raise ValueError("参考或查询为空")

    class _CrossCov(LinearOperator):
        """M = d1^T @ d2（(n1, n2)），以算子形式提供，避免物化大矩阵。"""

        def __init__(self) -> None:
            super().__init__(dtype=np.float64, shape=(n1, n2))

        def _matvec(self, v):
            return d1.T @ (d2 @ v)                       # (n1,)

        def _rmatvec(self, v):
            return d2.T @ (d1 @ v)                       # (n2,)

    if k >= min(n1, n2):
        # 维度不足：直接完整 SVD（矩阵很小）
        u, s, vt = np.linalg.svd(d1.T @ d2, full_matrices=False)
        u = np.pad(u, ((0, 0), (0, num_cc - k)))
        vt = np.pad(vt, ((0, num_cc - k), (0, 0)))
    else:
        u, s, vt = svds(_CrossCov(), k=k, which="LM")
        u, s, vt = u[:, ::-1], s[::-1], vt[::-1, :]      # svds 升序 -> 降序（同 irlba）
    v = vt.T[:, :k]
    # 符号校正：每维首元素为负则翻转（对应 R: if(sign(x[1])==-1) x <- -x）
    sign = np.sign(u[0, :])
    sign[sign == 0] = 1.0
    u = u * sign
    v = v * sign
    return np.vstack([u, v])                            # (cells1+cells2, k)


def _compute_pcaproject(
    ref_scaled: np.ndarray, query_scaled: np.ndarray, n_pcs: int
) -> np.ndarray:
    """参考 PCA 投影（对应 ``ProjectCellEmbeddings`` + pcaproject 组合嵌入）。

    参考 PCA（对 scale.data 做 SVD），查询投影到参考 PCA 空间，
    返回 ``[ref_embeddings; query_projected]`` 组合嵌入。
    """
    # 参考 PCA：对 细胞×基因 做 SVD（对应 Seurat RunPCA: irlba(t(data), nv)）
    # embeddings = U * s（细胞×k），loadings = V（基因×k）
    n_cells = ref_scaled.shape[1]
    k = min(n_pcs, n_cells, ref_scaled.shape[0])
    u, s, vt = np.linalg.svd(ref_scaled.T, full_matrices=False)
    loadings = vt[:k, :].T                              # (genes, k)
    ref_emb = u[:, :k] * s[:k]                          # (ref_cells, k)
    query_emb = query_scaled.T @ loadings                # (query_cells, k)
    return np.vstack([ref_emb, query_emb])


def _l2_norm(embeddings: np.ndarray) -> np.ndarray:
    """逐细胞 L2 归一化（对应 Seurat ``L2Dim``）。"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


# ---------------------------------------------------------------------------
# kNN / MNN anchors / 打分
# ---------------------------------------------------------------------------

def _knn_graph(data: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """kNN：返回 ``(indices, distances)``，均为 0 基；组内查询含自身（index 0）。

    用 scipy ``cKDTree``（C 实现，比 sklearn 暴力法快很多），大样本下显著提速。
    """
    k = min(k, len(data))
    distances, indices = cKDTree(data).query(query, k=k)
    if k == 1:
        indices = indices.reshape(-1, 1)
        distances = distances.reshape(-1, 1)
    return np.asarray(indices), np.asarray(distances)


def _find_anchor_pairs(
    nn_ab: np.ndarray, nn_ba: np.ndarray, k_anchor: int
) -> np.ndarray:
    """MNN anchors（对应 ``FindAnchorPairs``）。

    ``nn_ab``: 参考细胞 -> 查询细胞 的 kNN 索引（0 基，行=参考）。
    ``nn_ba``: 查询细胞 -> 参考细胞 的 kNN 索引（行=查询）。
    返回 (n_anchors, 3) 数组 ``[ref_idx, query_idx, score]``。
    """
    anchors: list[list[int]] = []
    for cell in range(nn_ab.shape[0]):
        neighbors_ab = nn_ab[cell, :k_anchor]
        for i, q in enumerate(neighbors_ab):
            if cell in nn_ba[q, :k_anchor]:            # 互近邻
                anchors.append([cell, int(q), 1])
    return np.asarray(anchors, dtype=float).reshape(-1, 3)


def _score_anchors(
    nn_aa: np.ndarray, nn_bb: np.ndarray,
    nn_ab: np.ndarray, nn_ba: np.ndarray,
    anchors: np.ndarray, k_score: int,
) -> np.ndarray:
    """anchor 得分（对应 ``ScoreAnchors``）：共享邻居数 -> 0.9/0.01 分位归一化。"""
    offset = nn_aa.shape[0]                             # 参考细胞数
    nbrset_a = [set(nn_aa[x, :k_score]) | set(nn_ab[x, :k_score] + offset)
                for x in anchors[:, 0].astype(int)]
    nbrset_b = [set(nn_ba[y, :k_score]) | set(nn_bb[y, :k_score] + offset)
                for y in anchors[:, 1].astype(int)]
    scores = np.array([len(a & b) for a, b in zip(nbrset_a, nbrset_b)], dtype=float)
    max_s = float(np.quantile(scores, 0.9))
    min_s = float(np.quantile(scores, 0.01))
    if max_s > min_s:
        scores = (scores - min_s) / (max_s - min_s)
    return np.clip(scores, 0.0, 1.0)


# ---------------------------------------------------------------------------
# TransferData：权重 + 标签预测
# ---------------------------------------------------------------------------

def _transfer_weights(
    query_emb: np.ndarray, anchors: np.ndarray, k_weight: int, sd_weight: float
) -> sp.csr_matrix:
    """anchor 权重矩阵 (n_anchors × n_query)。

    对应 ``FindWeights`` + C++ ``FindWeightsC``：每个查询细胞取其 k.weight 个
    最近 anchor 查询细胞，权重 = ``1 - exp(-proximity * anchor_score / (2/sd)^2)``，
    再按列归一化。热循环优先使用 C++ 扩展 ``_fastseurat``（对应 Seurat 的
    ``src/integration.cpp``），未编译时回退到 numpy 实现。
    """
    n_query = query_emb.shape[0]
    anchor_query_cells = np.unique(anchors[:, 1].astype(int))
    anchor_scores = anchors[:, 2].astype(np.float64)
    # 查询细胞 -> 该细胞参与的 anchor 行
    row_map: dict[int, list[int]] = {}
    for r in range(anchors.shape[0]):
        row_map.setdefault(int(anchors[r, 1]), []).append(r)
    # 展平为 offsets + rows（按 anchor_query_cells 顺序）
    offsets = [0]
    rows_flat: list[int] = []
    for ac in anchor_query_cells:
        rows_flat.extend(sorted(row_map.get(int(ac), [])))
        offsets.append(len(rows_flat))
    offsets = np.asarray(offsets, dtype=np.intp)
    rows_flat = np.asarray(rows_flat, dtype=np.int32)

    anchor_emb = query_emb[anchor_query_cells]
    neighbor_idx, dist = _knn_graph(anchor_emb, query_emb, k_weight)
    neighbor_idx = neighbor_idx.astype(np.int32)         # 索引指向 anchor_query_cells
    # 1 - dist/dist[:, kth]（对应 R: distances <- 1 - (distances / distances[, ncol])）
    kth = dist[:, [-1]].copy()
    kth[kth == 0] = 1.0
    dist_norm = (1 - dist / kth).astype(np.float64)

    if _find_weights_c_fast is not None:                 # C++ 加速路径
        rows, cols, vals = _find_weights_c_fast(
            dist_norm, neighbor_idx, offsets, rows_flat, anchor_scores, sd_weight)
    else:                                                # numpy 兜底
        rows, cols, vals = [], [], []
        for q in range(n_query):
            added = 0
            for j in range(dist_norm.shape[1]):
                if added >= k_weight:
                    break
                acell = anchor_query_cells[neighbor_idx[q, j]]
                for arow in row_map.get(int(acell), []):
                    if added >= k_weight:
                        break
                    w = 1 - np.exp(-dist_norm[q, j] * anchor_scores[arow]
                                   / (2 / sd_weight) ** 2)
                    rows.append(arow); cols.append(q); vals.append(w)
                    added += 1
        rows = np.asarray(rows, dtype=np.int32)
        cols = np.asarray(cols, dtype=np.int32)
        vals = np.asarray(vals, dtype=np.float64)

    weights = sp.coo_matrix((vals, (rows, cols)), shape=(anchors.shape[0], n_query))
    col_sums = np.asarray(weights.sum(axis=0)).ravel()
    col_sums[col_sums == 0] = 1.0
    weights = weights.multiply(1.0 / col_sums).tocsr()
    return weights


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def seurat_annotate(
    ref_expr,
    ref_labels,
    query_expr,
    *,
    reduction: str = "cca",
    max_genes: int = 2000,
    n_dims: int = 30,
    k_anchor: int = 5,
    k_score: int = 30,
    k_weight: int = 50,
    sd_weight: float = 1.0,
    n_jobs: int = 1,
    random_state: int = 0,
) -> dict[str, Any]:
    """Seurat 参考映射标签转移注释（对应 ``FindTransferAnchors`` + ``TransferData``）。

    Parameters:
        ref_expr: 参考表达矩阵（``pd.DataFrame`` / ``AnnData`` / sparse / ndarray，
            行=基因，列=细胞；AnnData 的 X 为 细胞×基因，自动转置）。
        ref_labels: 参考细胞类型标签（``pd.Series``，索引与参考细胞对齐）。
        query_expr: 查询表达矩阵，格式同 ``ref_expr``。
        reduction: 低维嵌入方式，``"cca"``（默认，经典）或 ``"pcaproject"``
            （参考 PCA 投影，对应 vignette ``reference.reduction="pca"``）。
        max_genes: 共享可变基因数（HVG，vst 风格），默认 2000。
        n_dims: 使用的嵌入维度数（对应 Seurat ``dims=1:n_dims``），默认 30。
        k_anchor: MNN anchors 近邻数（``k.anchor``），默认 5。
        k_score: anchor 打分近邻数（``k.score``），默认 30。
        k_weight: 权重近邻数（``k.weight``），默认 50。
        sd_weight: 高斯核带宽（``sd.weight``），默认 1。
        n_jobs: 保留（sklearn kNN 并行），默认 1。
        random_state: 随机种子。

    Returns:
        字典：
        - ``labels``: 每个查询细胞的预测标签（``pd.Series``）；
        - ``scores``: 每个细胞对各标签的预测得分（``pd.DataFrame``）；
        - ``all_scores``: 同 ``scores``（Seurat 转移无首轮/微调区分）；
        - ``prediction_score``: 每细胞最高预测得分（``pd.Series``）；
        - ``anchors``: 找到的 anchors 数组 (ref_idx, query_idx, score)。
    """
    # 1. 输入统一为 (基因 × 细胞) 矩阵 + 基因名
    ref_mat, ref_genes, ref_cells = _expr_to_mat(ref_expr, "ref_expr")
    query_mat, query_genes, query_cells = _expr_to_mat(query_expr, "query_expr")
    ref_label_arr = _coerce_ref_labels(ref_labels, ref_cells)

    # 2. 共享基因交集（对应 Seurat: intersect features）
    if ref_genes is not None and query_genes is not None:
        ref_sub, query_sub, common = _intersect_and_filter(
            ref_mat, ref_genes, query_mat, query_genes
        )
        common = np.asarray(common)
    else:
        if ref_mat.shape[0] != query_mat.shape[0]:
            raise ValueError("缺少基因名时，参考与查询必须行数一致")
        ref_sub, query_sub = ref_mat, query_mat
        common = np.arange(ref_mat.shape[0])

    def _dense(mat):
        return np.asarray(mat.toarray() if sp.issparse(mat) else mat, dtype=float)

    ref_dense = _dense(ref_sub)
    query_dense = _dense(query_sub)

    # 3. 可变基因（HVG，在参考上计算，与查询求交集）-> 用于后续嵌入
    n_genes = min(max_genes, ref_dense.shape[0])
    hvg_idx = select_hvg(ref_dense, max_genes=n_genes)
    features = hvg_idx
    ref_use = ref_dense[features]
    query_use = query_dense[features]

    # 4. 逐基因 z-score（ScaleData）
    ref_scaled = _scale_genes(ref_use)
    query_scaled = _scale_genes(query_use)

    # 5. 低维嵌入 + L2 归一化
    if reduction == "cca":
        embeddings = _compute_cca(ref_scaled, query_scaled, n_dims)
    elif reduction == "pcaproject":
        embeddings = _compute_pcaproject(ref_scaled, query_scaled, n_dims)
    else:
        raise ValueError(f"未知 reduction: {reduction!r}，可选 'cca'/'pcaproject'")
    embeddings_l2 = _l2_norm(embeddings)
    n_ref = ref_scaled.shape[1]
    ref_emb = embeddings_l2[:n_ref]
    query_emb = embeddings_l2[n_ref:]

    # 6. kNN 图（FindNN，k = max(k.anchor, k.score)）
    k_nn = max(k_anchor, k_score)
    nn_aa = _knn_graph(ref_emb, ref_emb, k_nn)[0]        # 参考 -> 参考（组内）
    nn_bb = _knn_graph(query_emb, query_emb, k_nn)[0]    # 查询 -> 查询（组内）
    nn_ab = _knn_graph(query_emb, ref_emb, k_nn)[0]      # 参考 -> 查询
    nn_ba = _knn_graph(ref_emb, query_emb, k_nn)[0]      # 查询 -> 参考

    # 7. MNN anchors
    anchors = _find_anchor_pairs(nn_ab, nn_ba, k_anchor)
    if anchors.shape[0] == 0:
        raise ValueError("未找到任何 anchors：参考与查询在嵌入空间过于分离")

    # 8. anchor 打分（ScoreAnchors）
    scores_anchor = _score_anchors(nn_aa, nn_bb, nn_ab, nn_ba, anchors, k_score)
    anchors[:, 2] = scores_anchor

    # 9. TransferData：权重 + 标签预测
    weights = _transfer_weights(query_emb, anchors, k_weight, sd_weight)
    possible_ids = sorted(set(ref_label_arr.tolist()))
    label_map = {lab: i for i, lab in enumerate(possible_ids)}
    pred_mat = np.zeros((anchors.shape[0], len(possible_ids)))
    for r in range(anchors.shape[0]):
        pred_mat[r, label_map[ref_label_arr[int(anchors[r, 0])]]] = 1.0
    pred_scores = (weights.T @ pred_mat)                 # (n_query, n_labels)
    pred_ids = np.asarray(possible_ids)[np.argmax(pred_scores, axis=1)]
    max_score = pred_scores.max(axis=1)

    # 10. 组装输出
    if query_cells is None:
        query_cells = list(range(query_dense.shape[1]))
    index = pd.Index(query_cells)
    scores_df = pd.DataFrame(pred_scores, index=index, columns=possible_ids)
    return {
        "labels": pd.Series(pred_ids, index=index, name="label"),
        "scores": scores_df,
        "all_scores": scores_df.copy(),
        "prediction_score": pd.Series(max_score, index=index, name="prediction_score"),
        "anchors": anchors,
    }
