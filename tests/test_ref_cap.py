"""逐类型参考封顶单元测试（大参考提速方案 A）。"""
from __future__ import annotations

import numpy as np
import pytest

from hybridscsinglemarker.pysingle.core import _cap_ref_cells

LABELS = np.array(["CD4 T"] * 30 + ["B cell"] * 20 + ["Monocyte"] * 10, dtype=object)


def test_cap_ref_cells_bounds_and_preserves_types():
    keep = _cap_ref_cells(LABELS, cap=10)
    assert keep.size == 30
    for lab in np.unique(LABELS):
        assert (LABELS[keep] == lab).sum() == 10
    assert set(LABELS[keep]) == set(LABELS)


def test_cap_ref_cells_deterministic_same_seed():
    a = _cap_ref_cells(LABELS, cap=10, seed=0)
    b = _cap_ref_cells(LABELS, cap=10, seed=0)
    np.testing.assert_array_equal(a, b)


def test_cap_ref_cells_seed_changes_sampling():
    a = _cap_ref_cells(LABELS, cap=10, seed=0)
    b = _cap_ref_cells(LABELS, cap=10, seed=1)
    assert not np.array_equal(a, b)


def test_cap_ref_cells_noop_when_cap_large():
    keep = _cap_ref_cells(LABELS, cap=100)
    np.testing.assert_array_equal(keep, np.arange(LABELS.size))


def test_cap_ref_cells_exact_size_keeps_in_order():
    # cap == 某一类型的恰好大小：该类型原样保留且保序，其余类型封顶
    keep = _cap_ref_cells(LABELS, cap=20)
    np.testing.assert_array_equal(
        keep[LABELS[keep] == "B cell"], np.arange(30, 50))
    assert keep.size == 20 + 20 + 10


def test_cap_ref_cells_min_cap():
    keep = _cap_ref_cells(LABELS, cap=1)
    assert keep.size == 3
    assert set(LABELS[keep]) == set(LABELS)


import anndata as ad
import pandas as pd
from hybridscsinglemarker.pysingle import singleR_annotate

SIZES = {"CD4 T": 30, "B cell": 20, "Monocyte": 10}
GENE_N = 60

# 合成基因名：各类型信号块头部嵌入真实 Human/Blood DB marker。hybrid_annotate
# 的 DB 打分要求 var_names 含至少一个 in-scope marker（数值型默认名会抛
# ValueError）；约定与 test_hybrid.py 一致（真实 marker 基因名 + 填充基因）。
_DB_MARKERS = {
    "CD4 T": ["CD3D", "CD3E", "CD4", "LEF1", "TCF7"],
    "B cell": ["MS4A1", "CD19", "CD79A"],
    "Monocyte": ["CD14", "FCGR3A", "LYZ"],
}


def _gene_names() -> list[str]:
    out: list[str] = []
    for t, ct in enumerate(SIZES):
        out += _DB_MARKERS[ct] + [f"FGENE_{t}_{i:02d}" for i in range(15 - len(_DB_MARKERS[ct]))]
    out += [f"NOISE_{i:02d}" for i in range(GENE_N - 3 * 15)]
    return out


def _synth_ref() -> ad.AnnData:
    rng = np.random.default_rng(7)
    labels = [ct for ct, n in SIZES.items() for _ in range(n)]
    X = rng.normal(0, 1, (len(labels), GENE_N))
    for row, lab in enumerate(labels):
        t = list(SIZES).index(lab)
        X[row, t * 15:(t + 1) * 15] += 2.0
    return ad.AnnData(X=X, obs={"celltype": labels},
                      var=pd.DataFrame(index=_gene_names()))


def _synth_query(n_per: int = 8) -> ad.AnnData:
    rng = np.random.default_rng(11)
    labels = [ct for ct in SIZES for _ in range(n_per)]
    X = rng.normal(0, 1, (len(labels), GENE_N))
    for row, lab in enumerate(labels):
        t = list(SIZES).index(lab)
        X[row, t * 15:(t + 1) * 15] += 2.0
    return ad.AnnData(X=X, obs={"celltype": labels},
                      var=pd.DataFrame(index=_gene_names()))


REF = _synth_ref()
QUERY = _synth_query()


def _singler(cap, seed=0, **kw):
    out = singleR_annotate(
        REF, REF.obs["celltype"], QUERY,
        fine_tune=True, gene_selection="hvg", max_genes=60,
        max_cells_per_type=cap, ref_cap_seed=seed, **kw,
    )
    return out["labels"].to_numpy(), out["all_scores"].to_numpy()


def test_singler_cap_noop_equals_none():
    lab_none, sc_none = _singler(None)
    lab_big, sc_big = _singler(100)
    np.testing.assert_array_equal(lab_big, lab_none)
    np.testing.assert_array_equal(sc_big, sc_none)


def test_singler_cap_deterministic():
    a = _singler(10, seed=0)
    b = _singler(10, seed=0)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_singler_cap_invalid():
    with pytest.raises(ValueError):
        _singler(0)
    with pytest.raises(ValueError):
        _singler(-1)


def test_singler_cap_actually_trims():
    lab_none, sc_none = _singler(None)
    lab_cap, sc_cap = _singler(10)
    assert not np.array_equal(sc_cap, sc_none)   # 封顶后得分必然改变


def test_singler_cap_seed_reaches_sampling():
    a = _singler(10, seed=0)
    b = _singler(10, seed=1)
    assert not np.array_equal(a[1], b[1])        # 不同种子 → 不同封顶抽样（得分必变）


def test_singler_cap_sparse_ref():
    import scipy.sparse as sp
    ref_sp = ad.AnnData(
        X=sp.csr_matrix(REF.X), obs=REF.obs.copy(),
        var=pd.DataFrame(index=_gene_names()))
    a = singleR_annotate(
        ref_sp, ref_sp.obs["celltype"], QUERY,
        fine_tune=True, gene_selection="hvg", max_genes=60,
        max_cells_per_type=10, ref_cap_seed=0,
    )
    b = singleR_annotate(
        REF, REF.obs["celltype"], QUERY,
        fine_tune=True, gene_selection="hvg", max_genes=60,
        max_cells_per_type=10, ref_cap_seed=0,
    )
    np.testing.assert_array_equal(a["labels"].to_numpy(), b["labels"].to_numpy())
    np.testing.assert_array_equal(a["all_scores"].to_numpy(), b["all_scores"].to_numpy())


# ---------------------------------------------------------------------------
# Task 3：fine-tuning 巨型候选组分块并行
# ---------------------------------------------------------------------------
from hybridscsinglemarker.pysingle.core import _fine_tune


def _synth_query_single_type(n: int = 40) -> ad.AnnData:
    """全部查询细胞同属 CD4 T：首轮所有细胞候选重叠 → 单一巨型组。

    噪声缩到 0.1 使 40 个细胞的候选集保持一致（{B cell, CD4 T}，
    fine_tune_thres=0.3 下全部细胞 2 候选且元组相同 → 单组 40 细胞）。
    """
    rng = np.random.default_rng(13)
    X = rng.normal(0, 0.1, (n, GENE_N))
    for row in range(n):
        X[row, 0:15] += 2.0
        X[row, 15:30] += 2.0          # 与 B cell 共享信号 → 候选不止一个类型
    return ad.AnnData(X=X, obs={"celltype": ["CD4 T"] * n},
                      var=pd.DataFrame(index=_gene_names()))


Q1 = _synth_query_single_type()

SCORES = singleR_annotate(
    REF, REF.obs["celltype"], Q1,
    fine_tune=False, gene_selection="hvg", max_genes=60,
)["all_scores"].to_numpy()


def _ft(scores, n_jobs, chunk):
    """直接调 _fine_tune（REF.X.T 为 (基因×细胞)，Q1.X.T 同）。"""
    return _fine_tune(
        REF.X.T, REF.obs["celltype"].to_numpy(dtype=object), Q1.X.T,
        scores, sorted(SIZES), median_all=None, top_n=5,
        fine_tune_thres=0.3, min_genes=20, n_de_scale=500,
        n_jobs=n_jobs, fine_tune_chunk=chunk,
    )


def test_fine_tune_chunk_equivalence():
    lab_ser, sc_ser = _ft(SCORES, 1, 2500)
    lab_ch1, sc_ch1 = _ft(SCORES, 2, 1)
    lab_chbig, sc_chbig = _ft(SCORES, 2, 10 ** 6)
    np.testing.assert_array_equal(lab_ch1, lab_ser)      # 分块并行 = 串行（逐位一致）
    np.testing.assert_array_equal(lab_chbig, lab_ser)    # 不分块并行 = 串行
    np.testing.assert_array_equal(lab_ch1, lab_chbig)
    np.testing.assert_array_equal(sc_ch1, sc_ser)
    np.testing.assert_array_equal(sc_chbig, sc_ser)
    np.testing.assert_array_equal(sc_ch1, sc_chbig)


def test_fine_tune_single_group_uses_pool(monkeypatch):
    """单一巨型组也必须走并行池（回归保护：禁止恢复 len(groups)>1 条件）。"""
    import hybridscsinglemarker.pysingle.core as core
    from concurrent.futures import ProcessPoolExecutor
    calls = []

    def fake_init(self, *a, **kw):
        calls.append(kw.get("max_workers"))
        ProcessPoolExecutor.__init__(self, *a, **kw)   # 真实初始化，仅记录入参

    monkeypatch.setattr(
        core, "ProcessPoolExecutor",
        type("Spy", (ProcessPoolExecutor,), {"__init__": fake_init}))
    _ft(SCORES, 2, 2500)
    assert calls, "n_jobs=2 时即使单一组也必须创建并行池"


# ---------------------------------------------------------------------------
# Task 4：hybrid_annotate 参考封顶参数透传（lambda_=0 → 纯 pysingle 语义）
# ---------------------------------------------------------------------------
from hybridscsinglemarker import hybrid_annotate


def test_hybrid_cap_passthrough_noop():
    a = hybrid_annotate(QUERY, REF, lambda_=0, species="Human", tissue="Blood")
    b = hybrid_annotate(QUERY, REF, lambda_=0, species="Human", tissue="Blood",
                        max_cells_per_type=100)
    np.testing.assert_array_equal(
        a.obs["hybrid_celltype"].to_numpy(), b.obs["hybrid_celltype"].to_numpy())


def test_hybrid_cap_actual_trim_deterministic():
    a = hybrid_annotate(QUERY, REF, lambda_=0, species="Human", tissue="Blood",
                        max_cells_per_type=10)
    b = hybrid_annotate(QUERY, REF, lambda_=0, species="Human", tissue="Blood",
                        max_cells_per_type=10)
    np.testing.assert_array_equal(
        a.obs["hybrid_celltype"].to_numpy(), b.obs["hybrid_celltype"].to_numpy())
    nc = hybrid_annotate(QUERY, REF, lambda_=0, species="Human", tissue="Blood")
    assert not np.array_equal(
        a.uns["hybridsc"]["S_corr"].to_numpy(), nc.uns["hybridsc"]["S_corr"].to_numpy()), \
        "cap=10 必须真正裁剪参考池（得分必须改变）"


def test_hybrid_cap_explicit_params_seurat_path_safe():
    # 回归保护：max_cells_per_type / ref_cap_seed 必须是显式签名参数。
    # 若仅靠 **kwargs 透传，seurat 路径会把这些参数漏给 seurat_annotate
    # （无该参数）→ TypeError。n_dims/k_score/k_weight 走 kwargs 通道
    # （seurat_annotate 参数），仅为适配 24 细胞小查询（CCA 分量上限）。
    out = hybrid_annotate(QUERY, REF, method="seurat", lambda_=0,
                          species="Human", tissue="Blood",
                          max_cells_per_type=10,
                          n_dims=10, k_score=10, k_weight=10)
    assert out.obs["hybrid_celltype"].notna().all()


# ---------------------------------------------------------------------------
# Task 5：实验脚本 auto_cap_ref 自动封顶阈值
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_cellmarker_experiment import REF_CAP_RULES, auto_cap_ref  # noqa: E402


def test_auto_cap_ref_thresholds():
    assert auto_cap_ref(10_000) is None
    assert auto_cap_ref(49_999) is None
    assert auto_cap_ref(50_000) == 500
    assert auto_cap_ref(119_999) == 500
    assert auto_cap_ref(120_000) == 300
    assert auto_cap_ref(160_000) == 300


def test_ref_cap_rules_synced_between_scripts():
    """两个脚本的自动封顶规则必须保持一致（防漂移）。"""
    from hybrid_pipeline2 import REF_CAP_RULES as R2
    assert REF_CAP_RULES == R2
