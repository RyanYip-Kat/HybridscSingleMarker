"""Skeleton defaults: Experiment-only data source, scale-aware caps and gates.

Implements the acceptance criteria from
``docs/superpowers/specs/2026-08-21-hybridscsinglemarker-skeleton-design.md``
§4 (defaults) and §5–§6 (runtime / quality gates).
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import anndata as ad

from hybridscsinglemarker import hybrid_annotate
from hybridscsinglemarker.cellmarkerannot.annotation import _resolve_data_sources
from hybridscsinglemarker.pysingle import singleR_annotate

import pipeline_utils as pu


# --------------------------------------------------------------------------- #
# Experiment-only default (design §4)                                          #
# --------------------------------------------------------------------------- #


def test_hybrid_annotate_default_data_source_is_experiment():
    sig = inspect.signature(hybrid_annotate)
    assert sig.parameters["data_source"].default == "experiment"


def test_resolve_data_sources_experiment_maps_to_experiment_source():
    assert _resolve_data_sources("experiment") == frozenset({"Experiment"})


def test_resolve_data_sources_all_still_available():
    assert _resolve_data_sources("all") is None


# --------------------------------------------------------------------------- #
# Scale-aware reference capping (design §5.2 / auto rule)                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("n_ref", "expected"),
    [
        (160_000, 300),
        (120_000, 300),
        (119_999, 500),
        (80_000, 500),
        (50_000, 500),
        (49_999, None),
        (20_000, None),
        (0, None),
    ],
)
def test_auto_cap_for_ref_rules(n_ref, expected):
    assert pu.auto_cap_for_ref(n_ref) == expected


def test_auto_cap_returns_integer_within_bounds():
    cap = pu.auto_cap_for_ref(160_000)
    assert isinstance(cap, int) and 200 <= cap <= 500


# --------------------------------------------------------------------------- #
# Flip statistics (design §6.3 quality gate)                                   #
# --------------------------------------------------------------------------- #


def _sample_labels():
    truth = np.array(["T", "T", "B", "NK", "T", "B"])
    base = np.array(["T", "T", "B", "T", "T", "B"])   # NK misclassified as T
    fused = np.array(["T", "T", "B", "NK", "T", "B"])  # fusion fixes cell 3
    return truth, base, fused


def test_flip_stats_counts_and_precision():
    truth, base, fused = _sample_labels()
    st = pu.flip_stats(base, fused, truth)
    assert st["n_good"] == 1          # cell 3 fixed
    assert st["n_bad"] == 0
    assert st["n_both_ok"] == 5
    assert st["n_both_bad"] == 0
    assert st["flip_precision"] == pytest.approx(1.0)


def test_flip_stats_precision_lt_one_when_bad_flip():
    truth = np.array(["T", "B"])
    base = np.array(["T", "B"])
    fused = np.array(["B", "T"])      # both flips are wrong
    st = pu.flip_stats(base, fused, truth)
    assert st["n_both_ok"] == 0
    assert st["n_good"] == 0
    assert st["n_bad"] == 2
    assert st["flip_precision"] == pytest.approx(0.0)


def test_flip_stats_no_flips_gives_nan_precision():
    truth = np.array(["T", "B"])
    base = np.array(["T", "B"])
    st = pu.flip_stats(base, base.copy(), truth)
    assert st["n_good"] == st["n_bad"] == 0
    assert np.isnan(st["flip_precision"])


# --------------------------------------------------------------------------- #
# Scale-aware fusion policy (design §4.1)                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("n_ref", "n_refs", "method", "expected"),
    [
        (160_000, 1, "singler", 0.0),   # >=50k single ref -> pure pysingle
        (80_000, 1, "singler", 0.0),
        (50_000, 1, "singler", 0.0),
        (49_999, 1, "singler", 0.3),
        (20_000, 1, "singler", 0.3),
        (160_000, 2, "singler", 0.3),   # multi-ref keeps fusion
        (160_000, 1, "seurat", 0.0),    # policy applies to seurat too
        (20_000, 1, "seurat", 0.3),
    ],
)
def test_default_lambda_scale_policy(n_ref, n_refs, method, expected):
    assert pu.default_lambda(n_ref=n_ref, n_refs=n_refs, method=method) == expected


# --------------------------------------------------------------------------- #
# λ=0 reuse hook (design §5.3 runtime gate: don't recompute pysingle twice)    #
# --------------------------------------------------------------------------- #


def _tiny_ref_query():
    genes = ["CD3D", "CD3E", "CD4"]
    ref = ad.AnnData(
        X=sp.csr_matrix([[3.0, 0.0, 0.0]]),
        obs=pd.DataFrame({"celltype": ["CD4 T"]}, index=["r1"]),
        var=pd.DataFrame(index=genes),
    )
    q = ad.AnnData(
        X=sp.csr_matrix([[2.0, 2.0, 0.0], [0.0, 0.0, 2.0]]),
        obs=pd.DataFrame(index=["q1", "q2"]),
        var=pd.DataFrame(index=genes),
    )
    return ref, q


def test_reuse_singler_matches_lambda_zero_hybrid():
    ref, q = _tiny_ref_query()
    out = singleR_annotate(ref, ref.obs["celltype"], q,
                           fine_tune=False, gene_selection="all")
    kw = dict(celltype_col="celltype", species="Human", tissue="Blood",
              data_source="experiment", lambda_=0.0,
              fine_tune=False, gene_selection="all")
    r_plain = hybrid_annotate(q, ref, **kw)
    r_reuse = hybrid_annotate(q, ref, reuse_singler=[out], **kw)
    np.testing.assert_array_equal(
        r_plain.obs["hybrid_celltype"], r_reuse.obs["hybrid_celltype"]
    )
    np.testing.assert_allclose(
        r_plain.uns["hybridsc"]["F"].to_numpy(),
        r_reuse.uns["hybridsc"]["F"].to_numpy(),
    )
    np.testing.assert_array_equal(
        r_plain.obs["hybrid_celltype"],
        out["labels"].to_numpy(),
    )


def test_reuse_singler_guard_rejects_positive_lambda():
    ref, q = _tiny_ref_query()
    out = singleR_annotate(ref, ref.obs["celltype"], q,
                           fine_tune=False, gene_selection="all")
    with pytest.raises(ValueError, match="lambda_"):
        hybrid_annotate(
            q, ref, celltype_col="celltype", species="Human", tissue="Blood",
            lambda_=0.3, reuse_singler=[out],
            fine_tune=False, gene_selection="all",
        )


def test_fine_metrics_ignores_nan_predictions():
    """DB-only 路径允许 NaN 预测（无表达 marker 的细胞）；细粒度指标须跳过。"""
    from hybrid_pipeline2 import _compute_fine_metrics

    pred = np.array(["T", "B", np.nan, "NK"], dtype=object)
    truth = np.array(["T", "B", "T", "NK"], dtype=object)
    m = _compute_fine_metrics(pred, truth)
    assert "fine_ari" in m
    assert m["fine_ari"] == m["fine_ari"] or np.isnan(m["fine_ari"])


# --------------------------------------------------------------------------- #
# 可视化修复：UMAP 着色列索引对齐 + 先验图自适应布局                           #
# --------------------------------------------------------------------------- #


def test_umap_coarse_column_aligns_to_obs_names():
    """粗分类列必须按 obs_names 对齐（RangeIndex 错位会让 scanpy 全画成灰点）。"""
    from hybrid_pipeline2 import _obs_coarse_column

    obs = pd.DataFrame(index=["b1", "b2", "b3", "b4"])
    coarse = np.array(["T", None, "B", "NK"], dtype=object)
    col = _obs_coarse_column(obs, coarse, "sc_test")
    assert list(col.index) == ["b1", "b2", "b3", "b4"]
    assert col.isna().sum() == 0          # None → "None" 类别，不允许 NaN
    assert isinstance(col.dtype, pd.CategoricalDtype)
    assert col.iloc[0] == "T"
    assert col.iloc[1] == "None"


def test_prior_layout_scales_with_label_count():
    """先验图布局按标签数自适应：标签多 → 面板更高、字体更小、刻度抽稀。"""
    from hybrid_pipeline2 import _prior_layout

    h_few, fs_few, step_few = _prior_layout(10)
    assert h_few >= 3.4
    assert step_few == 1
    h_many, fs_many, step_many = _prior_layout(46)
    assert h_many > h_few
    assert fs_many < fs_few
    assert step_many >= 2


# --------------------------------------------------------------------------- #
# 实验脚本更新：auto scoring + 规模感知 λ + 翻转闸门                            #
# --------------------------------------------------------------------------- #


def test_experiment_resolve_scoring_auto():
    """auto scoring：≤10k 用 cells（精度优先），>10k 用 profile（速度优先）。"""
    from run_cellmarker_experiment import resolve_scoring

    assert resolve_scoring("auto", 2000) == "cells"
    assert resolve_scoring("auto", 10_000) == "cells"
    assert resolve_scoring("auto", 50_000) == "profile"
    assert resolve_scoring("profile", 2000) == "profile"
    assert resolve_scoring("cells", 50_000) == "cells"


def test_experiment_effective_lambda_uses_scale_policy():
    """scale-aware λ：单参考 ≥50k → 0（纯 pysingle），其余保持 0.3。"""
    from run_cellmarker_experiment import effective_lambda_with

    assert effective_lambda_with(160_000, n_refs=1, scale_aware=True,
                                 lambda_with=0.3) == 0.0
    assert effective_lambda_with(20_000, n_refs=1, scale_aware=True,
                                 lambda_with=0.3) == 0.3
    assert effective_lambda_with(160_000, n_refs=2, scale_aware=True,
                                 lambda_with=0.3) == 0.3
    assert effective_lambda_with(160_000, n_refs=1, scale_aware=False,
                                 lambda_with=0.3) == 0.3


def test_experiment_flatten_flip_keys():
    """翻转统计展平：flip_coarse 的 flip_precision 必须落到 delta['flip_precision']。"""
    from run_cellmarker_experiment import flatten_flip

    delta = flatten_flip(
        {"n_good": 2, "n_bad": 1, "n_both_ok": 5, "n_both_bad": 0,
         "flip_precision": 2 / 3}
    )
    assert delta["flip_precision"] == 2 / 3
    assert delta["flip_n_good"] == 2
    assert delta["flip_n_bad"] == 1
    assert "flip_flip_precision" not in delta
