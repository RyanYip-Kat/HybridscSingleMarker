"""得分融合：DB 先验 P_c 计算 + 乘法融合 + 得分归一化。

核心公式（设计文档 §3.1 / §4）:
    F[cell,c] = S_corr[cell,c] × (1 + λ·P_c)

约定:
    - S_corr 来自 pysingle（scoring="cells" 或 "profile"；多参考时用合并得分）。
    - P_c ∈ [0,1]：每参考标签的特征基因相对 CellMarker DB 的富集强度，跨标签
      min-max 归一化。DB 无匹配 → P_c = 0；全部标签等强（退化）→ 0.5。
    - λ=0 时 F ≡ S_corr（退化为纯相关，内置 sanity check）。

性能（相对逐标签调用 ``cellmarkerannot.score_gene_list``）:
    P_c 采用**批量向量化**：所有标签的特征基因并集一次性构建 evidence 矩阵
    （一次稀疏 groupby/pivot），再对 (标签 × DB 细胞类型) 全矩阵用
    ``scipy.stats.hypergeom.sf`` 向量化广播求富集 p 值 —— 无逐标签 Python 循环，
    富集核为 C/BLAS；每个标签只取一次特征基因（不随细胞数放大）。
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy import stats

from hybridscsinglemarker.cellmarkerannot._scoring import build_evidence_matrix
from hybridscsinglemarker.cellmarkerannot.annotation import _resolve_data_sources, _scope_rows


def _label_feature_genes(
    ref_data: Sequence[tuple[Any, pd.Series]],
    common_genes: Sequence[str],
    top_n: int = 200,
) -> dict[str, list[str]]:
    """每参考标签 → 特征基因（DE 风格：该标签均值表达减去其它标签最大均值的 top-N）。

    实现: 各参考对齐到 ``common_genes``（稀疏 COO 重排）→ 纵向拼接 → 标签指示
    稀疏矩阵一次 matmul 得每标签基因和 → 均值矩阵 → 逐标签 DE 排序（循环仅
    n_labels 次）。特征基因用于 DB 富集先验 P_c 与逐细胞验证 V。

    选择策略: 优先取该标签 DE 正基因（``mean[c] > 其它标签最大均值``）中
    top-N。**不因 DE 正基因太少就稀释回退**——把其它标签的 marker / 管家基因
    混入会同时冲淡先验富集与验证（分母虚增）；少而特异优于多而混杂。仅当该
    标签**没有任何** DE 正基因（完全被其它标签覆盖）时，才回退到该标签高表达
    top-N 作为最后手段。
    """
    common_pos = {g: i for i, g in enumerate(common_genes)}
    n_common = len(common_genes)
    pooled_rows: list[sp.csr_matrix] = []
    pooled_labels: list[np.ndarray] = []
    for ref, labels in ref_data:
        x = ref.X
        if not sp.issparse(x):
            x = sp.csr_matrix(x)
        x = x.tocsr()
        if len(ref.var_names) != x.shape[1]:
            raise ValueError("ref.var_names 必须与 ref.X 列数一致")
        col_map = np.array([common_pos.get(g, -1) for g in ref.var_names], dtype=np.int64)
        coo = x.tocoo()
        keep = col_map[coo.col] >= 0
        rows = coo.row[keep]
        cols = col_map[coo.col[keep]]
        data = coo.data[keep]
        aligned = sp.coo_matrix(
            (data, (rows, cols)), shape=(x.shape[0], n_common)
        ).tocsr()
        pooled_rows.append(aligned)
        pooled_labels.append(np.asarray(labels, dtype=object))

    pooled = sp.vstack(pooled_rows, format="csr")
    all_labels = np.concatenate(pooled_labels)
    cats = pd.Categorical(all_labels)
    codes = cats.codes                       # -1 = NaN 标签
    mask = codes >= 0
    n_use = int(mask.sum())
    if n_use == 0 or n_common == 0:
        return {}

    ind = sp.csr_matrix(
        (np.ones(n_use), (codes[mask], np.arange(n_use))),
        shape=(len(cats.categories), n_use),
    )
    sub = pooled[mask]
    sums = (ind @ sub).toarray().astype(np.float64)          # (n_labels × n_common)
    counts = np.bincount(codes[mask], minlength=len(cats.categories))
    mean_mat = sums / np.where(counts > 0, counts, 1.0)[:, None]

    feature: dict[str, list[str]] = {}
    for c, lab in enumerate(cats.categories):
        others = np.ones(len(cats.categories), dtype=bool)
        others[c] = False
        other_max = mean_mat[others].max(axis=0) if others.any() else np.zeros(n_common)
        diff = mean_mat[c] - other_max
        order = np.argsort(-diff)
        pos = diff[order] > 0
        n_pos = int(pos.sum())
        if n_pos > 0:                                       # DE 正基因（特异，优先）
            take = min(top_n, n_pos)
            genes = [str(common_genes[i]) for i in order[pos][:take]]
        else:                                               # 无任何正 DE：最后手段高表达
            genes = [str(common_genes[i]) for i in np.argsort(-mean_mat[c])[:top_n]]
        feature[lab] = genes
    return feature


def _common_genes(ref_data, query_adata, *, max_common: int = 20000) -> list[str]:
    """所有参考与查询共有基因，按查询 var_names 顺序（限定规模控制内存）。"""
    query_set = set(query_adata.var_names)
    common = query_set
    for ref, _ in ref_data:
        common &= set(ref.var_names)
    ordered = [g for g in query_adata.var_names if g in common]
    if len(ordered) > max_common:
        ordered = ordered[:max_common]
    return ordered


def db_prior_for_labels(
    genes_by_label: dict[str, Sequence[str]],
    db,
    *,
    species: str,
    tissue: str,
    data_source: str | Sequence[str] = "all",
    marker_sources: Sequence[str] | None = None,
    in_scope: pd.DataFrame | None = None,
) -> dict[str, float]:
    """每参考标签 → DB 先验 P_c ∈ [0,1]（批量向量化富集）。

    对全部标签的特征基因并集一次性构建 evidence 矩阵
    （:func:`cellmarkerannot.build_evidence_matrix`），再对 (标签 × DB 细胞类型)
    全矩阵向量化 hypergeom 富集::

        score[c,t] = -log10(Hypergeom.sf(k-1, N, K_t, n_c)) * log1p(ev_c,t)

    其中 ``n_c`` = 标签 c 匹配到的基因数，``k`` = 其中是 DB 细胞类型 t 的 marker
    数，``K_t`` = t 的 marker 总数，``N`` = 范围内 distinct marker 数。
    ``P_c`` = 标签 c 的最佳富集强度 ``max_t score[c,t]`` 除以全部标签的最大值
    （**max 归一化**，∈ (0,1]）。0 仅当标签无任何 DB 富集——与设计"DB 无匹配
    → P_c = 0"一致（min-max 会把最弱但有真实富集的标签压成 0，误导融合权重）。

    退化情况:
        - 范围为空 / 无任何基因匹配（``build_evidence_matrix`` 抛 ValueError）
          → 全部 P_c = 0；
        - 所有标签等强 → 全部 0.5（对 argmax 无区分作用）。
    """
    if not genes_by_label:
        return {}
    if marker_sources is None:
        marker_sources = _resolve_data_sources(data_source)
    if in_scope is None:
        in_scope = _scope_rows(db, species, tissue, marker_sources)
    if in_scope is None or len(in_scope) == 0:
        return {lab: 0.0 for lab in genes_by_label}

    label_order = list(genes_by_label.keys())
    union = list(dict.fromkeys(g for genes in genes_by_label.values() for g in genes))
    try:
        evidence = build_evidence_matrix(
            db, species, tissue, marker_sources, union, in_scope=in_scope
        )
    except ValueError:
        return {lab: 0.0 for lab in genes_by_label}

    matched = list(evidence.index)
    cell_types = list(evidence.columns)
    E = evidence.to_numpy(dtype=np.int64)                     # (matched × celltypes)
    matched_pos = {g: i for i, g in enumerate(matched)}

    # 标签 → 基因 归属矩阵（小矩阵，直接稠密）
    G = np.zeros((len(label_order), len(matched)), dtype=np.int8)
    for i, lab in enumerate(label_order):
        for g in genes_by_label[lab]:
            j = matched_pos.get(g)
            if j is not None:
                G[i, j] = 1
    if G.sum() == 0:
        return {lab: 0.0 for lab in genes_by_label}

    nq = G.sum(axis=1).astype(np.int64)                       # 每标签匹配基因数
    k = G @ (E > 0)                                           # (labels × celltypes)
    ev = G @ E                                                # 证据和 (labels × celltypes)
    n_total = int(in_scope["marker"].nunique())
    k_arr = (
        in_scope.groupby("cell_name")["marker"]
        .nunique()
        .reindex(cell_types)
        .fillna(0)
        .astype(np.int64)
        .to_numpy()
    )

    pos = (k > 0) & (nq[:, None] > 0)
    kk = np.where(pos, k, 0)
    nqq = np.where(nq > 0, nq, 1)
    pvals = stats.hypergeom.sf(kk - 1, n_total, k_arr[None, :], nqq[:, None])
    pvals = np.where(pos, pvals, 1.0)
    logp = np.where(pvals > 0, -np.log10(pvals), 0.0)
    score = logp * np.log1p(ev)
    m = score.max(axis=1)

    hi = float(m.max())
    if hi <= 0.0:                                  # 全部标签无任何富集 → 0
        return {lab: 0.0 for lab in label_order}
    lo = float(m.min())
    if hi - lo < 1e-12:                            # 退化：所有标签等强 → 0.5
        return {lab: 0.5 for lab in label_order}
    return {lab: float(m[i] / hi) for i, lab in enumerate(label_order)}


def per_cell_db_prior(
    query,
    genes_by_label: dict[str, Sequence[str]],
    db_genes: set[str],
    *,
    expr_threshold: float = 1.0,
) -> pd.DataFrame:
    """逐细胞 DB 证据先验 ``P[cell,c] ∈ [0,1]``（每细胞每参考标签）。

    对每个参考标签 c，取它与 DB 范围重叠的特征基因 ``G_c = genes[c] ∩ db_genes``；
    细胞在 ``G_c`` 中表达（``> expr_threshold``）的比例为该细胞对 c 的原始支持
    分，再对**每行（细胞）**做 max 归一化：

        P[cell,c] = raw[cell,c] / max_c raw[cell,c]      （整行无支持 → 全 0）

    与 :func:`db_prior_for_labels` 的**静态**逐标签先验不同，这里用每个细胞
    自身的表达计证据：T 细胞只在 T 标记上高支持，不会被 DB 中标记更丰富的
    pDC/CD16 等标签系统性拉偏——修复了静态先验"按 DB marker 丰富度全局加偏"
    的方向性问题（真实数据上静态先验 λ=1 会把粗族一致率从 0.92 拉到 0.60，
    逐细胞先验恢复到 ≥0.90 并可在 λ≈0.3 时提升到 0.95）。

    实现: 仅保留 DB 范围 ∩ 查询基因空间的基因列，稀疏 ``E``（细胞×该子集）
    布尔化后一次 matmul 得每标签支持计数（``E @ M.T``，M 为标签×基因布尔
    归属矩阵），全程无逐细胞循环；返回 ``(n_cells × n_labels)`` DataFrame。

    Parameters:
        query: ``AnnData``（X 为 细胞×基因，可含 layer 视图）。
        genes_by_label: ``{参考标签: 特征基因列表}``（键的顺序即列顺序）。
        db_genes: DB 范围内 distinct marker 集合（``set(in_scope["marker"])``）。
        expr_threshold: 表达阈值（与 cellmarkerannot 一致，默认 1.0）。
    """
    if not genes_by_label:
        return pd.DataFrame(index=query.obs_names)
    x = query.X
    if not sp.issparse(x):
        x = sp.csr_matrix(x)
    q_genes = np.asarray(query.var_names, dtype=object)
    db_genes_q = [g for g in q_genes if g in db_genes]
    labels = list(genes_by_label)
    if not db_genes_q:
        return pd.DataFrame(0.0, index=query.obs_names, columns=labels)

    gpos = {g: i for i, g in enumerate(q_genes)}
    db_cols = np.array([gpos[g] for g in db_genes_q], dtype=np.int64)
    E = x[:, db_cols].tocsr()
    E = (E > expr_threshold).astype(np.float64)            # 稀疏 (cells × n_db)

    M = np.zeros((len(labels), len(db_genes_q)), dtype=np.float64)
    for i, lab in enumerate(labels):
        feats = [g for g in genes_by_label[lab] if g in db_genes_q]
        if not feats:
            continue
        M[i, np.isin(db_genes_q, feats)] = 1.0
    raw = np.asarray(E @ M.T)                               # (cells × labels)
    denom = M.sum(axis=1)                                   # 每标签匹配基因数 (labels,)
    raw = np.divide(raw, np.where(denom > 0, denom, 1.0))
    rowmax = raw.max(axis=1, keepdims=True)
    Pn = np.where(rowmax > 0, raw / np.where(rowmax > 0, rowmax, 1.0), 0.0)
    return pd.DataFrame(Pn, index=query.obs_names, columns=labels)


def fuse_scores_per_cell(
    S_corr: pd.DataFrame,
    P_cell: pd.DataFrame,
    lambda_: float = 0.3,
) -> pd.DataFrame:
    """逐细胞乘法融合: ``F[cell,c] = S_corr[cell,c] × (1 + λ·P_cell[cell,c])``。

    ``P_cell`` 按行/列对齐到 ``S_corr``（缺失标签按 0 处理）；λ=0 时
    ``F ≡ S_corr``。纯 numpy 缩放（BLAS/C），无逐细胞循环。
    """
    S = S_corr.to_numpy(dtype=np.float64)
    P = P_cell.reindex(
        index=S_corr.index, columns=S_corr.columns,
    ).fillna(0.0).to_numpy(dtype=np.float64)
    F = S * (1.0 + lambda_ * P)
    return pd.DataFrame(F, index=S_corr.index, columns=S_corr.columns)


def normalize_prior(raw_scores) -> np.ndarray:
    """将一维得分向量 max 归一化到 (0,1]（与 :func:`db_prior_for_labels` 一致）。

    0 仅当全部得分为 0（无富集）；退化（全等且非零）→ 0.5。
    """
    a = np.asarray(raw_scores, dtype=np.float64)
    hi = float(a.max())
    if hi <= 0.0:
        return np.zeros_like(a)
    lo = float(a.min())
    if hi - lo < 1e-12:
        return np.full_like(a, 0.5)
    return a / hi


def fuse_scores(
    S_corr: pd.DataFrame,
    P: dict[str, float],
    lambda_: float = 1.0,
) -> pd.DataFrame:
    """乘法融合: F[cell,c] = S_corr[cell,c] × (1 + λ·P_c)。

    P 缺失的标签按 0 处理；λ=0 时 F ≡ S_corr（sanity check）。
    纯 numpy 列缩放（BLAS/C），无逐细胞循环。
    """
    labels = list(S_corr.columns)
    P_arr = np.array([P.get(l, 0.0) for l in labels], dtype=np.float64)
    mult = 1.0 + lambda_ * P_arr
    F = S_corr.to_numpy(dtype=np.float64) * mult[None, :]
    return pd.DataFrame(F, index=S_corr.index, columns=labels)


def softmax_rows(F) -> np.ndarray:
    """对得分矩阵逐行 softmax（数值稳定）→ 每细胞对标签的概率分布。"""
    X = np.asarray(F, dtype=np.float64)
    X = X - X.max(axis=1, keepdims=True)
    e = np.exp(X)
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Variant B — 族级后验再排序融合（design §6.3）                                #
# --------------------------------------------------------------------------- #


def fuse_family_posterior(
    S_final: pd.DataFrame,
    labels_final: pd.Series | np.ndarray,
    P_cell: pd.DataFrame,
    family_of: dict[str, str],
    *,
    lambda_: float = 0.3,
    margin_gate: float | None = 2.0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """族级后验再排序融合（Variant B）。

    设计思想：DB 证据只负责**粗族决策**，族内细标签永远由 pysingle 决定——
    消除 v_full 乘法融合对族内细标签的扰动，同时保留 DB 对粗族边界的修复
    （T↔NK/B/Mono 等，实测 DB 唯一有净价值的方向）。

    算法（全向量化，无逐细胞 Python 循环）::

        1. 每细胞 winner margin（S_final 的 top1−top2），按 margin 门控:
              λ_eff[cell] = λ × clip(1 − margin/(gate·median_margin), 0, 1)
           高置信细胞 λ_eff=0 ⇒ 完全保持 pysingle 标签（不改已正确细胞）。
        2. 族级聚合:
              S_fam[cell,fam] = max_{c∈fam} S_final[cell,c]   # pysingle 族支持
              P_fam[cell,fam] = max_{c∈fam} P_cell[cell,c]    # DB 族证据
        3. 族级后验得分:
              G[cell,fam] = S_fam × (1 + λ_eff·P_fam)
            chosen_family = argmax_fam G
        4. 最终标签:
              λ_eff=0 或 fine-tuned 标签的粗族 == chosen_family → 保留
              pysingle 标签；否则取 chosen_family 内 S_final 的 argmax
            （DB 只改族，不改族内细标签）。
        5. 融合得分矩阵:
              F[cell,c] = S_final[cell,c] × (1 + λ_eff·P_fam[cell,family(c)])
            （族内乘数一致 ⇒ 族内相对顺序不变；供置信度/uns 输出）。

    Returns:
        ``(F, final_labels, chosen_family, lambda_eff)``：F 为融合得分矩阵
        （cells × labels），final_labels 为最终标签数组（object），
        chosen_family 为每细胞的选定粗族，lambda_eff 为每细胞实际融合强度。
    """
    S = S_final.to_numpy(dtype=np.float64)
    n_cells, n_labels = S.shape
    labels_idx = list(S_final.columns)
    pos = {lab: i for i, lab in enumerate(labels_idx)}
    labels = np.asarray(labels_final, dtype=object)
    P = (
        P_cell.reindex(index=S_final.index, columns=S_final.columns)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )

    # ---- 1. winner-margin 门控（与 v_full 一致，自校准）----
    if n_labels >= 2:
        sorted_S = np.sort(S, axis=1)
        margin = sorted_S[:, -1] - sorted_S[:, -2]
    else:
        margin = np.zeros(n_cells)
    med_margin = float(np.median(margin))
    if margin_gate is not None and med_margin > 0:
        lam_eff = lambda_ * np.clip(
            1.0 - margin / (margin_gate * med_margin), 0.0, 1.0
        )
    else:
        lam_eff = np.full(n_cells, lambda_)

    # ---- 2. 族级聚合（pysingle 支持 + DB 证据均取族内 max）----
    families = sorted({f for f in family_of.values() if f is not None})
    fam_members: dict[str, list[int]] = {
        f: [i for i, lab in enumerate(labels_idx) if family_of.get(lab) == f]
        for f in families
    }
    fam_members = {f: c for f, c in fam_members.items() if c}  # 剔除空族
    families = list(fam_members)
    fam_pos = {f: j for j, f in enumerate(families)}
    n_fams = len(families)
    S_fam = np.zeros((n_cells, n_fams))
    P_fam = np.zeros((n_cells, n_fams))
    for j, f in enumerate(families):
        cols = fam_members[f]
        S_fam[:, j] = S[:, cols].max(axis=1)
        P_fam[:, j] = P[:, cols].max(axis=1)

    # ---- 3. 族级后验得分 + chosen_family ----
    G = S_fam * (1.0 + lam_eff[:, None] * P_fam)
    chosen = G.argmax(axis=1)
    chosen_family = np.asarray(families, dtype=object)[chosen]

    # ---- 4. 最终标签：族不变则保留 pysingle，族变则取新族内 S 最大 ----
    out = labels.copy()
    lab_pos = np.array([pos.get(str(lab), -1) for lab in labels], dtype=np.int64)
    lab_fam = np.array(
        [family_of.get(str(lab)) if p >= 0 else None
         for lab, p in zip(labels, lab_pos)],
        dtype=object,
    )
    flip = (lam_eff > 0) & (lab_fam != chosen_family)
    labels_arr = np.asarray(labels_idx, dtype=object)
    for j, f in enumerate(families):
        mask = flip & (chosen_family == f)
        if not mask.any():
            continue
        cols = np.asarray(fam_members[f], dtype=np.int64)
        sub = S[mask][:, cols]
        out[mask] = labels_arr[cols[sub.argmax(axis=1)]]

    # ---- 5. 融合得分矩阵：族内同乘数 ⇒ 族内顺序不变 ----
    mult = np.ones_like(S)
    for j, f in enumerate(families):
        cols = np.asarray(fam_members[f], dtype=np.int64)
        # 注意：必须取 (n_cells,1) 列向量再广播到该族各列，否则 numpy 会把
        # (n,) 向量按外积广播，导致第 i 个细胞的乘数错误引用第 k 个细胞的先验。
        mult[:, cols] = (1.0 + lam_eff * P_fam[:, j])[:, None]
    F = S * mult
    return (
        pd.DataFrame(F, index=S_final.index, columns=S_final.columns),
        out,
        chosen_family,
        lam_eff,
    )


def label_evidence_matrices(
    genes_by_label: dict[str, Sequence[str]],
    db,
    *,
    species: str,
    tissue: str,
    data_source: str | Sequence[str] = "experiment",
    marker_sources: Sequence[str] | None = None,
    in_scope: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """每参考标签一张 ``score_gene_list`` 风格的注释矩阵（批量向量化）。

    输出 ``{label: DataFrame}``，格式与 ``cellmarkerannot.score_gene_list``
    一致，便于逐标签可视化：

    - 行 = 该标签在范围内**有证据**的特征基因（输入顺序，首现去重）；
    - 列 = 范围内 ≥1 个命中基因的 DB 细胞类型（排序）；
    - 值 = 整数支持证据（范围内引用该 (基因, 细胞类型) 对的记录行数）；
    - 最后一行 ``"Score"`` = 该标签对每个 DB 细胞类型的富集得分
      ``-log10(Hypergeom.sf(k−1, N, K_t, n_c)) × log1p(ev)``
      （与 :func:`db_prior_for_labels` / ``score_gene_list`` 同一公式）。

    性能：全部标签的特征基因并集**只构建一次** evidence 矩阵（一次稀疏
    groupby/pivot），再对 (标签 × DB 细胞类型) 全矩阵向量化 hypergeom 富集
    （C/BLAS 核）；每标签只取一次特征基因，不随细胞数放大。
    """
    if not genes_by_label:
        return {}
    if marker_sources is None:
        marker_sources = _resolve_data_sources(data_source)
    if in_scope is None:
        in_scope = _scope_rows(db, species, tissue, marker_sources)
    if in_scope is None or len(in_scope) == 0:
        return {lab: pd.DataFrame() for lab in genes_by_label}

    label_order = list(genes_by_label.keys())
    union = list(dict.fromkeys(g for genes in genes_by_label.values() for g in genes))
    try:
        evidence = build_evidence_matrix(
            db, species, tissue, marker_sources, union, in_scope=in_scope
        )
    except ValueError:
        return {lab: pd.DataFrame() for lab in label_order}

    matched = list(evidence.index)
    cell_types = list(evidence.columns)
    E = evidence.to_numpy(dtype=np.int64)                 # (matched × celltypes)
    matched_pos = {g: i for i, g in enumerate(matched)}

    # 标签 → 基因 归属矩阵（小矩阵，直接稠密）
    G = np.zeros((len(label_order), len(matched)), dtype=np.int8)
    for i, lab in enumerate(label_order):
        for g in genes_by_label[lab]:
            j = matched_pos.get(g)
            if j is not None:
                G[i, j] = 1

    nq = G.sum(axis=1).astype(np.int64)                   # 每标签匹配基因数
    k = G @ (E > 0)                                       # (labels × celltypes)
    ev = G @ E                                            # 证据和
    n_total = int(in_scope["marker"].nunique())
    k_arr = (
        in_scope.groupby("cell_name")["marker"]
        .nunique()
        .reindex(cell_types)
        .fillna(0)
        .astype(np.int64)
        .to_numpy()
    )
    pos = (k > 0) & (nq[:, None] > 0)
    kk = np.where(pos, k, 0)
    nqq = np.where(nq > 0, nq, 1)
    pvals = stats.hypergeom.sf(kk - 1, n_total, k_arr[None, :], nqq[:, None])
    pvals = np.where(pos, pvals, 1.0)
    logp = np.where(pvals > 0, -np.log10(pvals), 0.0)
    score = logp * np.log1p(ev)                           # (labels × celltypes)

    out: dict[str, pd.DataFrame] = {}
    for i, lab in enumerate(label_order):
        row_idx = np.flatnonzero(G[i] > 0)                # 该标签命中的基因
        if row_idx.size == 0:
            out[lab] = pd.DataFrame()
            continue
        gene_order = [matched[j] for j in row_idx]        # 输入顺序（union 保序）
        col_idx = np.flatnonzero(E[row_idx].sum(axis=0) > 0)
        if col_idx.size == 0:
            out[lab] = pd.DataFrame()
            continue
        block = E[np.ix_(row_idx, col_idx)].astype(np.float64)
        score_row = score[i, col_idx]
        df = pd.DataFrame(block, index=gene_order, columns=[cell_types[j] for j in col_idx])
        df.loc["Score"] = score_row
        df.index.name = None
        df.columns.name = None
        out[lab] = df
    return out
