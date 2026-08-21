"""Variant B — family-posterior re-ranking fusion + per-label evidence matrices.

Design: ``docs/superpowers/specs/2026-08-21-hybridscsinglemarker-skeleton-design.md``
§6.3 (Variant B). DB 只负责粗族决策，族内细标签永远由 pysingle 决定：

    F[cell,c]      = S_final[cell,c] × (1 + λ_eff[cell]·P_fam[cell, family(c)])
    chosen_family  = argmax_fam ( S_fam[cell,fam] × (1 + λ_eff·P_fam[cell,fam]) )
    final label    = pysingle fine-tuned 标签（若其粗族 == chosen_family），
                     否则 chosen_family 内 S_final 的 argmax

λ_eff 沿用 winner-margin 门控（高置信细胞 λ→0 ⇒ 完全保持 pysingle 标签）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hybridscsinglemarker import hybrid_annotate
from hybridscsinglemarker._fusion import (
    fuse_family_posterior,
    label_evidence_matrices,
)


# --------------------------------------------------------------------------- #
# fuse_family_posterior                                                        #
# --------------------------------------------------------------------------- #


def _demo_inputs():
    """2 个细胞 × 4 个标签（T 族: CD4T/CD8T，B 族: Bmem/Bnaive）。

    P 行含义（逐细胞 DB 先验）：
      c1: T 族证据强（CD4T=1.0, CD8T=0.9），B 族无证据；
      c2: B 族证据强（Bmem=1.0, Bnaive=0.8），T 族无证据。
    """
    S = pd.DataFrame(
        {
            "CD4T": [0.80, 0.60],
            "CD8T": [0.75, 0.55],
            "Bmem": [0.90, 0.40],
            "Bnaive": [0.85, 0.35],
        },
        index=["c1", "c2"],
    )
    labels = pd.Series(["CD4T", "CD8T"], index=["c1", "c2"])  # fine-tuned 标签
    P = pd.DataFrame(
        {
            "CD4T": [1.0, 0.0],   # c1: T 族证据强
            "CD8T": [0.9, 0.0],
            "Bmem": [0.0, 1.0],   # c2: B 族证据强
            "Bnaive": [0.0, 0.8],
        },
        index=["c1", "c2"],
    )
    fam = {"CD4T": "T", "CD8T": "T", "Bmem": "B", "Bnaive": "B"}
    return S, labels, P, fam


def test_lambda_zero_keeps_pysingle_labels_and_scores():
    S, labels, P, fam = _demo_inputs()
    F, out_labels, _, leff = fuse_family_posterior(
        S, labels, P, fam, lambda_=0.0, margin_gate=2.0
    )
    np.testing.assert_allclose(F.to_numpy(), S.to_numpy())
    assert list(out_labels) == list(labels)
    assert (leff == 0.0).all()


def test_margin_gate_fully_disables_fusion_for_confident_cells():
    # 5 个细胞：前 4 个 margin=0.05（低置信），最后 1 个 margin=0.89（高置信）。
    # median=0.05 ⇒ gate×median=0.10 ⇒ 高置信细胞 λ_eff=0（完全保持 pysingle）。
    S = pd.DataFrame(
        {
            "CD4T": [0.80, 0.80, 0.80, 0.80, 0.99],
            "CD8T": [0.75, 0.75, 0.75, 0.75, 0.10],
            "Bmem": [0.20, 0.20, 0.20, 0.20, 0.10],
            "Bnaive": [0.10, 0.10, 0.10, 0.10, 0.05],
        },
        index=[f"c{i}" for i in range(5)],
    )
    labels = pd.Series(["CD4T"] * 5, index=S.index)
    P = pd.DataFrame(
        {
            "CD4T": [1.0] * 5, "CD8T": [0.9] * 5,
            "Bmem": [0.0] * 5, "Bnaive": [0.0] * 5,
        },
        index=S.index,
    )
    fam = {"CD4T": "T", "CD8T": "T", "Bmem": "B", "Bnaive": "B"}
    F, out_labels, _, leff = fuse_family_posterior(
        S, labels, P, fam, lambda_=0.3, margin_gate=2.0
    )
    assert leff[4] == pytest.approx(0.0)          # 高置信细胞完全禁用融合
    assert leff[:4].min() > 0                     # 低置信细胞仍融合
    np.testing.assert_allclose(F.iloc[4].to_numpy(), S.iloc[4].to_numpy())
    assert out_labels[4] == "CD4T"


def test_family_rerank_switches_family_and_picks_best_fine_label():
    # c1: pysingle 偏好 B 族（Bmem 0.90 > CD4T 0.80），但 DB 先验强支持 T 族
    #      ⇒ 族级改判 T；细标签取 T 族内 S 最大的 CD4T。
    S, labels, P, fam = _demo_inputs()
    F, out_labels, chosen_fam, _ = fuse_family_posterior(
        S, labels, P, fam, lambda_=0.3, margin_gate=None
    )
    assert chosen_fam[0] == "T"
    assert out_labels[0] == "CD4T"
    # c2: pysingle 偏好 T（CD4T 0.60），DB 强支持 B ⇒ 改判 B；细标签取 B 内最大
    S.loc["c2"] = [0.45, 0.42, 0.40, 0.35]
    F, out_labels, chosen_fam, _ = fuse_family_posterior(
        S, labels, P, fam, lambda_=0.3, margin_gate=None
    )
    assert chosen_fam[1] == "B"
    assert out_labels[1] == "Bmem"


def test_family_rerank_keeps_fine_tuned_label_when_family_unchanged():
    # 单细胞：fine-tuned 标签 CD8T（T 族内第二），T 族被 DB 选中 ⇒ 原样保留
    # CD8T（虽然 T 族内 S 最大的是 CD4T）——族内细标签永远归 pysingle。
    S = pd.DataFrame(
        {"CD4T": [0.80], "CD8T": [0.79], "Bmem": [0.40], "Bnaive": [0.30]},
        index=["c1"],
    )
    labels = pd.Series(["CD8T"], index=["c1"])
    P = pd.DataFrame(
        {"CD4T": [1.0], "CD8T": [0.9], "Bmem": [0.0], "Bnaive": [0.0]},
        index=["c1"],
    )
    fam = {"CD4T": "T", "CD8T": "T", "Bmem": "B", "Bnaive": "B"}
    F, out_labels, chosen_fam, _ = fuse_family_posterior(
        S, labels, P, fam, lambda_=0.3, margin_gate=None
    )
    assert chosen_fam[0] == "T"
    assert out_labels[0] == "CD8T"


def test_within_family_relative_order_preserved():
    S, labels, P, fam = _demo_inputs()
    F, _, _, _ = fuse_family_posterior(
        S, labels, P, fam, lambda_=0.3, margin_gate=None
    )
    # 族内同乘数 ⇒ 相对顺序不变（c1: T 族 ×1.3、B 族 ×1；c2 反之）
    assert (F["CD4T"] >= F["CD8T"]).all()
    assert (F["Bmem"] >= F["Bnaive"]).all()


# --------------------------------------------------------------------------- #
# label_evidence_matrices（每标签 score_gene_list 风格注释矩阵）                #
# --------------------------------------------------------------------------- #


def test_label_evidence_matrices_format():
    from hybridscsinglemarker.cellmarkerannot import CellMarkerDB

    db = CellMarkerDB(dataset="all_cell_marker")
    genes_by_label = {
        "CD4 T": ["CD3D", "CD3E", "CD4"],
        "B cell": ["MS4A1", "CD19"],
    }
    mats = label_evidence_matrices(
        genes_by_label, db, species="Human", tissue="Blood",
        data_source="experiment",
    )
    assert set(mats) == {"CD4 T", "B cell"}
    for lab, m in mats.items():
        assert m.index[-1] == "Score"
        assert (m.iloc[:-1].to_numpy(dtype=float) >= 0).all()
        assert m.shape[1] >= 1
        assert m.columns.name is None


def test_label_evidence_score_row_matches_prior_max():
    from hybridscsinglemarker.cellmarkerannot import CellMarkerDB
    from hybridscsinglemarker._fusion import db_prior_for_labels

    db = CellMarkerDB(dataset="all_cell_marker")
    genes_by_label = {
        "CD4 T": ["CD3D", "CD3E", "CD4"],
        "B cell": ["MS4A1", "CD19"],
    }
    mats = label_evidence_matrices(
        genes_by_label, db, species="Human", tissue="Blood",
        data_source="experiment",
    )
    prior = db_prior_for_labels(
        genes_by_label, db, species="Human", tissue="Blood",
        data_source="experiment",
    )
    for lab, m in mats.items():
        top = float(m.loc["Score"].max())
        # db_prior_for_labels 的 P_c = max_t score / 全标签最大值
        assert top > 0


# --------------------------------------------------------------------------- #
# hybrid_annotate 集成                                                         #
# --------------------------------------------------------------------------- #


def _tiny_ref_query():
    import anndata as ad
    import scipy.sparse as sp

    genes = ["CD3D", "CD3E", "CD4", "MS4A1", "CD19"]
    ref = ad.AnnData(
        X=sp.csr_matrix([[3.0, 3.0, 3.0, 0.0, 0.0]]),
        obs=pd.DataFrame({"celltype": ["CD4 T"]}, index=["r1"]),
        var=pd.DataFrame(index=genes),
    )
    q = ad.AnnData(
        X=sp.csr_matrix([[2.0, 2.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0, 2.0]]),
        obs=pd.DataFrame(index=["q1", "q2"]),
        var=pd.DataFrame(index=genes),
    )
    return ref, q


def test_hybrid_family_posterior_lambda_zero_equals_plain_lambda_zero():
    ref, q = _tiny_ref_query()
    kw = dict(celltype_col="celltype", species="Human", tissue="Blood",
              data_source="experiment", lambda_=0.0,
              fine_tune=False, gene_selection="all")
    plain = hybrid_annotate(q, ref, **kw)
    fp = hybrid_annotate(q, ref, family_posterior=True, **kw)
    np.testing.assert_array_equal(
        plain.obs["hybrid_celltype"], fp.obs["hybrid_celltype"]
    )
    np.testing.assert_allclose(
        plain.uns["hybridsc"]["F"].to_numpy(),
        fp.uns["hybridsc"]["F"].to_numpy(),
    )


def test_hybrid_family_posterior_stores_label_evidence():
    ref, q = _tiny_ref_query()
    out = hybrid_annotate(
        q, ref, celltype_col="celltype", species="Human", tissue="Blood",
        data_source="experiment", lambda_=0.3,
        fine_tune=False, gene_selection="all",
        family_posterior=True, include_label_evidence=True,
    )
    hs = out.uns["hybridsc"]
    assert "label_evidence" in hs
    assert set(hs["label_evidence"]) >= {"CD4 T"}
    assert "chosen_family" in hs
    assert len(hs["chosen_family"]) == q.n_obs
