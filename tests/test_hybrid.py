"""hybrid_annotate 端到端与单元测试（合成数据，标记基因为真实 DB marker）。

覆盖设计文档 §6:
    1. λ=0 等价性：融合输出与纯 pysingle 相关一致；
    2. 无参考降级：ref=None 输出与 cellmarkerannot.annotate_cells 一致；
    3. 单参考基本链路：obs 三列、uns 矩阵、状态枚举、置信度范围；
    4. 多参考 + 每参考标签列名映射（celltype / cell_type）；
    5. 边界：DB 先验部分标签无匹配、method="seurat"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import anndata as ad

from hybridscsinglemarker import hybrid_annotate
from hybridscsinglemarker import _layer
from hybridscsinglemarker.cellmarkerannot import CellMarkerDB, annotate_cells
from hybridscsinglemarker.cellmarkerannot.annotation import _resolve_data_sources
from hybridscsinglemarker.pysingle import singleR_annotate

# --------------------------------------------------------------------------- #
# 合成数据：真实 Human/Blood DB marker，让 DB 打分与先验富集都有信号          #
# --------------------------------------------------------------------------- #
CELL_TYPES = ["CD4 T", "CD8 T", "B cell", "Monocyte"]
MARKERS = {
    "CD4 T": ["CD3D", "CD3E", "CD4", "LEF1", "TCF7"],
    "CD8 T": ["CD3D", "CD3E", "CD8A", "CD8B"],
    "B cell": ["MS4A1", "CD19", "CD79A"],
    "Monocyte": ["CD14", "FCGR3A", "LYZ"],
}
HOUSEKEEPING = ["ACTB", "B2M", "GAPDH"]
FILLERS = [f"FGENE_{i:03d}" for i in range(24)]


def _genes() -> list[str]:
    out: list[str] = []
    for ct in CELL_TYPES:
        for g in MARKERS[ct]:
            if g not in out:
                out.append(g)
    return out + HOUSEKEEPING + FILLERS


def _expr_matrix(labels) -> sp.csr_matrix:
    genes = _genes()
    gpos = {g: i for i, g in enumerate(genes)}
    n = len(labels)
    X = np.zeros((n, len(genes)))
    for i, lab in enumerate(labels):
        for g in MARKERS[lab]:
            X[i, gpos[g]] = 3.0
        for g in HOUSEKEEPING:
            X[i, gpos[g]] = 2.0
        X[i] += 0.02 * ((i * 7 + np.arange(len(genes))) % 3)  # 确定性微扰，破平局
    return sp.csr_matrix(X)


def _make_ann(labels, index_prefix, extra_col=None):
    obs = pd.DataFrame({extra_col: labels} if extra_col else {},
                       index=[f"{index_prefix}_{i}" for i in range(len(labels))])
    return ad.AnnData(X=_expr_matrix(labels), obs=obs,
                      var=pd.DataFrame(index=_genes()))


@pytest.fixture(scope="module")
def db():
    return CellMarkerDB()


@pytest.fixture(scope="module")
def ref_single():
    labels = [ct for ct in CELL_TYPES for _ in range(40)]
    return _make_ann(labels, "ref", extra_col="celltype")


@pytest.fixture(scope="module")
def query():
    labels = [ct for ct in CELL_TYPES for _ in range(120)]
    return _make_ann(labels, "q", extra_col="truth")


def _truth(query):
    return query.obs["truth"].to_numpy()


# --------------------------------------------------------------------------- #
# 单参考基本链路                                                               #
# --------------------------------------------------------------------------- #
def test_hybrid_single_ref_runs(query, ref_single):
    out = hybrid_annotate(query, ref_single, species="Human", tissue="Blood")
    assert out is not query            # 默认非破坏（副本写回）
    assert query.obs["truth"].iloc[0] == out.obs["truth"].iloc[0]

    for col in ("hybrid_celltype", "hybrid_confidence", "hybrid_status"):
        assert col in out.obs.columns
    assert len(out.obs) == query.n_obs

    conf = out.obs["hybrid_confidence"].to_numpy(dtype=float)
    assert np.isfinite(conf).all() and ((conf >= 0) & (conf <= 1)).all()

    statuses = set(out.obs["hybrid_status"])
    assert statuses <= set(_layer._ALL_STATUS)

    hs = out.uns["hybridsc"]
    for key in ("S_corr", "F", "softmax_prob", "P_prior", "V",
                "db_celltype", "db_confidence"):
        assert key in hs
    assert hs["S_corr"].shape[0] == query.n_obs
    assert hs["F"].shape == hs["S_corr"].shape
    assert hs["softmax_prob"].shape == hs["S_corr"].shape


def test_hybrid_celltype_matches_truth_majority(query, ref_single):
    out = hybrid_annotate(query, ref_single, species="Human", tissue="Blood")
    match = (out.obs["hybrid_celltype"].to_numpy() == _truth(query)).mean()
    assert match >= 0.9, f"hybrid_celltype 与真值一致率过低: {match:.2f}"


def test_db_prior_has_signal(query, ref_single):
    out = hybrid_annotate(query, ref_single, species="Human", tissue="Blood")
    P = out.uns["hybridsc"]["P_prior"]
    assert len(P) == len(CELL_TYPES)
    assert max(P.values()) > 0, "DB 先验 P_c 全为 0（特征基因未富集到 DB marker）"


def test_h5ad_roundtrip(query, ref_single, tmp_path):
    out = hybrid_annotate(query, ref_single, species="Human", tissue="Blood")
    path = tmp_path / "out.h5ad"
    out.write_h5ad(path)
    rt = ad.read_h5ad(path)
    assert "hybridsc" in rt.uns
    np.testing.assert_array_equal(
        rt.obs["hybrid_celltype"].to_numpy(),
        out.obs["hybrid_celltype"].to_numpy(),
    )
    assert rt.uns["hybridsc"]["S_corr"].shape == (query.n_obs, len(CELL_TYPES))
    assert "pysingle" in rt.uns["hybridsc"]


# --------------------------------------------------------------------------- #
# 1. λ=0 等价性：F ≡ S_corr，c* 与纯 pysingle 一致                             #
# --------------------------------------------------------------------------- #
def test_lambda_zero_equals_pysingle(query, ref_single):
    out = hybrid_annotate(query, ref_single, lambda_=0,
                          species="Human", tissue="Blood")
    pure = singleR_annotate(ref_single, ref_single.obs["celltype"], query)
    pure_labels = pure["labels"].to_numpy()

    assert np.array_equal(
        out.obs["hybrid_celltype"].to_numpy(), pure_labels,
    ), "λ=0 时融合结果应与纯 pysingle 标签一致"
    # F ≡ S_corr（乘法乘 1，位级不变）
    np.testing.assert_array_equal(
        out.uns["hybridsc"]["F"].to_numpy(),
        out.uns["hybridsc"]["S_corr"].to_numpy(),
    )


# --------------------------------------------------------------------------- #
# 2. 无参考降级：status = db_only，与 annotate_cells 一致                      #
# --------------------------------------------------------------------------- #
def test_no_ref_db_only(query, db):
    out = hybrid_annotate(query, ref=None, species="Human", tissue="Blood")
    assert (out.obs["hybrid_status"] == _layer.DB_ONLY).all()

    # hybrid 默认 data_source="experiment"（骨架设计 §4）；直接比较需用同一范围
    ref_db = annotate_cells(query, db, method="weighted",
                            species="Human", tissue="Blood",
                            marker_sources=_resolve_data_sources("experiment"),
                            inplace=False)
    np.testing.assert_array_equal(
        out.obs["hybrid_celltype"].to_numpy(),
        ref_db["celltype_predicted"].to_numpy(),
    )
    np.testing.assert_allclose(
        out.obs["hybrid_confidence"].to_numpy(dtype=float),
        ref_db["confidence"].to_numpy(dtype=float),
        atol=1e-6,
    )


# --------------------------------------------------------------------------- #
# 4. 多参考 + 每参考标签列名映射（celltype / cell_type）                       #
# --------------------------------------------------------------------------- #
def test_multi_ref_column_mapping(query, ref_single):
    ref2 = _make_ann(
        [ct for ct in CELL_TYPES for _ in range(40)],
        "ref2", extra_col="cell_type",
    )
    out = hybrid_annotate(
        query, [ref_single, ref2],
        celltype_col=["celltype", "cell_type"],
        species="Human", tissue="Blood",
    )
    assert set(out.obs["hybrid_status"]) <= set(_layer._ALL_STATUS)
    assert out.obs["hybrid_celltype"].notna().all()
    S = out.uns["hybridsc"]["S_corr"]
    assert set(S.columns) == set(CELL_TYPES)


def test_celltype_col_length_mismatch(query, ref_single):
    with pytest.raises(ValueError, match="celltype_col"):
        hybrid_annotate(query, [ref_single, ref_single],
                        celltype_col=["celltype"], species="Human",
                        tissue="Blood")


def test_ref_missing_label_column(query, ref_single):
    with pytest.raises(ValueError, match="标签列"):
        hybrid_annotate(query, ref_single, celltype_col="nope",
                        species="Human", tissue="Blood")


# --------------------------------------------------------------------------- #
# 5. method="seurat" + 用户特征基因覆盖                                         #
# --------------------------------------------------------------------------- #
def test_method_seurat(query, ref_single):
    out = hybrid_annotate(query, ref_single, method="seurat",
                          species="Human", tissue="Blood", max_genes=200)
    assert out.obs["hybrid_celltype"].notna().all()
    assert set(out.obs["hybrid_status"]) <= set(_layer._ALL_STATUS)


def test_feature_genes_override(query, ref_single):
    # 用任意标签→基因覆盖：先验应按给定基因计算，不崩
    feat = {ct: list(MARKERS[ct]) for ct in CELL_TYPES}
    out = hybrid_annotate(query, ref_single, feature_genes=feat,
                          species="Human", tissue="Blood")
    assert set(out.uns["hybridsc"]["P_prior"]) == set(CELL_TYPES)


# --------------------------------------------------------------------------- #
# 纯单元测试：融合 / 分层                                                      #
# --------------------------------------------------------------------------- #
def test_fuse_scores_lambda_zero_is_identity():
    from hybridscsinglemarker._fusion import fuse_scores
    S = pd.DataFrame(np.random.RandomState(0).rand(5, 3),
                     columns=["a", "b", "c"])
    F = fuse_scores(S, {"a": 1.0, "b": 0.0}, lambda_=0.0)
    np.testing.assert_array_equal(F.to_numpy(), S.to_numpy())
    # 缺标签按 0 处理
    F1 = fuse_scores(S, {"a": 1.0}, lambda_=1.0)
    np.testing.assert_array_equal(F1["b"].to_numpy(), S["b"].to_numpy())


def test_classify_status_bulk_vectorized():
    cr = np.array(["T", "T", "B", None, "T", "NK"])
    cd = np.array(["T", "T", "B", "B", "Mono", None])
    cf = np.array([0.8, 0.1, 0.6, 0.9, 0.7, 0.5])
    st = _layer.classify_status_bulk(cr, cd, cf, threshold=0.3)
    assert list(st) == [
        _layer.CONSISTENT,
        _layer.LOW_CONFIDENCE,
        _layer.CONSISTENT,
        _layer.UNKNOWN,   # ref 无粗族
        _layer.UNKNOWN,   # 粗族冲突（T vs Mono）
        _layer.UNKNOWN,   # db 无粗族
    ]
