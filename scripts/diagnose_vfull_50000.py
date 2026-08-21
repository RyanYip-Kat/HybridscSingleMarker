#!/usr/bin/env python
"""诊断 1ref-50000 v_full 的翻转结构：dump 预测 + 与纯 pysingle 基线比对。

背景: prototype_fusion_final 显示 v_full（新默认）在 2000–10000 与 2ref-5000
上均 ≥ baseline，但 1ref-50000 仍 −0.8%（翻转精度 0.253，good 124 / bad 366）。
本脚本重跑 v_full（50000），dump 预测，并用原始实验的 without 预测（同参数
纯 pysingle，已验证与我方 baseline 一致 0.9577）作基线，给出翻转的粗族交叉表。

输出: results/experiment/prototype_final/diag_50000/
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_cellmarker_experiment import (  # noqa: E402
    SPECIES, TISSUE, DATA_SOURCE, QUERY, REF1, _subset_cached,
    _clean_coarse, coarse_type_series,
)
from hybridscsinglemarker import hybrid_annotate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N = 50000
OUT = ROOT / "results/experiment/prototype_final/diag_50000"
BASE_CSV = ROOT / "results/experiment/predictions_1ref-50000.csv"
BASE_COL = "pred_fine_without"   # 纯 pysingle（λ=0），已验证与重跑 baseline 一致 0.9577


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    q = _subset_cached(*QUERY, N)
    truth_fine = q.obs[QUERY[1]].to_numpy(dtype=object)
    truth_coarse = coarse_type_series(truth_fine)
    ref = [_subset_cached(*REF1, N)]

    t0 = time.time()
    out = hybrid_annotate(
        q, ref, celltype_col=[REF1[1]], species=SPECIES, tissue=TISSUE,
        data_source=DATA_SOURCE, method="singler", confidence_threshold=0.3,
        n_jobs=16, scoring="cells", fine_tune=True, top_n=5,
        gene_selection="hvg", max_genes=5000, combine_method="max",
        lambda_=0.3, lambda_margin_gate=2.0, family_boost_only=True,
    )
    print(f"[run] v_full elapsed {time.time()-t0:.0f}s", flush=True)

    pred_fine = out.obs["hybrid_celltype"].to_numpy(dtype=object)
    pred_coarse = coarse_type_series(pred_fine)

    # 基线：原始实验 without 预测（同参数纯 pysingle）
    base_df = pd.read_csv(BASE_CSV)
    base_fine = base_df[BASE_COL].to_numpy(dtype=object)
    assert len(base_fine) == q.n_obs, f"{len(base_fine)} vs {q.n_obs}"
    base_coarse = coarse_type_series(base_fine)

    yt = _clean_coarse(truth_coarse).to_numpy()
    yv = _clean_coarse(pred_coarse).to_numpy()
    yb = _clean_coarse(base_coarse).to_numpy()
    cv, cb = yv == yt, yb == yt
    m = yt != "Other"
    good = (cv & ~cb) & m
    bad = (~cv & cb) & m
    both_ok = (cv & cb) & m
    both_bad = (~cv & ~cb) & m
    print(f"n_eval={int(m.sum())}  good={int(good.sum())} bad={int(bad.sum())} "
          f"both_ok={int(both_ok.sum())} both_bad={int(both_bad.sum())}")

    def cross(idx, title):
        print(f"\n-- {title} (n={int(idx.sum())}) --")
        tb = pd.crosstab(
            pd.Series(yt[idx], name="truth"),
            pd.Series(yb[idx], name="pred_base"),
        )
        print(tb.to_string())

    cross(good, "good flip: truth → pred(without)")
    cross(bad, "bad flip:  truth → pred(without)")

    # 翻转细胞的 margin 分布（v_full 内部得分不可得，用预测位置近似）：
    # 只记录 truth/pred 明细到 CSV 供后续分析
    pd.DataFrame({
        "truth_fine": truth_fine, "truth_coarse": yt,
        "pred_base_fine": base_fine, "pred_base_coarse": yb,
        "pred_vfull_fine": pred_fine, "pred_vfull_coarse": yv,
    }).to_csv(OUT / "predictions_50000_vfull.csv", index=False)
    print(f"\n[done] 输出: {OUT}")


if __name__ == "__main__":
    main()
