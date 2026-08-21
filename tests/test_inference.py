"""inference.py 纯逻辑测试：无真值标签的推理指标 + 每标签证据汇总表。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import inference


def _tiny_label_evidence():
    """两张每标签证据矩阵（基因 × DB 细胞类型 + Score 行）。"""
    cd4 = pd.DataFrame(
        {"T cell": [3.0, 2.0], "NK cell": [0.0, 0.0]},
        index=["CD3D", "CD3E"],
    )
    cd4.loc["Score"] = [4.2, 0.0]
    b = pd.DataFrame(
        {"B cell": [5.0], "T cell": [0.0]},
        index=["MS4A1"],
    )
    b.loc["Score"] = [3.1, 0.0]
    return {"CD4 T": cd4, "B cell": b}


def test_evidence_table_from_mats():
    tab = inference.evidence_table_from_mats(_tiny_label_evidence())
    assert list(tab.columns) == [
        "label", "n_matched_genes", "n_db_cell_types",
        "top_db_celltype", "top_score", "mean_score",
    ]
    assert len(tab) == 2
    # 按 top_score 降序：CD4 T (4.2) 在前
    assert tab.iloc[0]["label"] == "CD4 T"
    assert tab.iloc[0]["top_db_celltype"] == "T cell"
    assert tab.iloc[1]["label"] == "B cell"
    assert tab.iloc[1]["n_matched_genes"] == 1


def test_inference_metrics_no_truth():
    import anndata as ad

    adata = ad.AnnData(
        X=np.zeros((4, 2)),
        obs=pd.DataFrame(
            {
                "hybrid_celltype": ["T", "T", "B", None],
                "hybrid_confidence": [0.8, 0.2, 0.5, 0.0],
                "hybrid_status": ["consistent", "low_confidence",
                                  "consistent", "unknown"],
            },
            index=["c1", "c2", "c3", "c4"],
        ),
    )
    adata.uns["hybridsc"] = {
        "coarse_ref": np.array(["T", "T", "B", None], dtype=object),
        "coarse_db": np.array(["T", "NK", "B", "B"], dtype=object),
        "lambda_eff_cell": np.array([0.0, 0.3, 0.2, 0.0]),
    }
    m = inference.inference_metrics(adata)
    assert m["n_cells"] == 4
    assert m["n_predicted"] == 3
    assert m["coverage"] == pytest.approx(0.75)
    assert m["confidence_mean"] == pytest.approx((0.8 + 0.2 + 0.5 + 0.0) / 4)
    assert m["status_counts"]["consistent"] == 2
    assert m["db_consistency_agreement"] == pytest.approx(2 / 3)  # c1,c3 一致；c2 冲突
    assert m["lambda_eff_mean"] == pytest.approx((0.0 + 0.3 + 0.2 + 0.0) / 4)


def test_h5ad_safe_hybridsc_converts_none_and_nan():
    """uns 中的 None / NaN object 数组（无表达 marker 的细胞）须转为空串，
    否则 anndata.write_h5ad 报错（h5py 无法隐式转换非字符串）。"""
    hs = {
        "db_celltype": np.array(["T", None, np.nan, "B"], dtype=object),
        "coarse_ref": np.array(["T", None, None, "B"], dtype=object),
        "coarse_db": np.array(["T", "NK", "B", "B"], dtype=object),
        "chosen_family": None,
        "lambda_eff_cell": None,
        "label_evidence": {},
    }
    clean = inference.h5ad_safe_hybridsc(hs)
    assert clean["db_celltype"].tolist() == ["T", "", "", "B"]
    assert clean["coarse_ref"].tolist() == ["T", "", "", "B"]
    assert "chosen_family" not in clean
    assert "lambda_eff_cell" not in clean
    assert "label_evidence" in clean
