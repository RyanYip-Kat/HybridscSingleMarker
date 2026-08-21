"""一致性分层与状态判定（hybrid_status 输出列）。

判定表（设计文档 §4.1）:

    | 状态             | 条件                                                       |
    |------------------|------------------------------------------------------------|
    | consistent       | 参考标签粗族与 DB 预测粗族一致，且 hybrid_confidence ≥ threshold |
    | low_confidence   | 粗族一致但 hybrid_confidence < threshold                  |
    | unknown          | 粗族冲突，或无 DB 证据（DB 无匹配 / 范围为空 / 无预测）    |
    | db_only          | 无参考降级链路                                            |

实现: 全 numpy 布尔掩码向量化，无逐细胞循环。粗族为 ``None``（无预测）时
按"缺失"处理（``consistent``/``low_confidence`` 均要求两粗族非空且相等）。
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

CONSISTENT = "consistent"
LOW_CONFIDENCE = "low_confidence"
UNKNOWN = "unknown"
DB_ONLY = "db_only"

_ALL_STATUS = (CONSISTENT, LOW_CONFIDENCE, UNKNOWN, DB_ONLY)


def classify_status(
    coarse_ref: str | None,
    coarse_db: str | None,
    hybrid_confidence: float,
    threshold: float = 0.3,
) -> str:
    """单个细胞的状态判定。"""
    if coarse_ref is None or coarse_db is None or coarse_ref != coarse_db:
        return UNKNOWN
    if hybrid_confidence >= threshold:
        return CONSISTENT
    return LOW_CONFIDENCE


def classify_status_bulk(
    coarse_refs,
    coarse_dbs,
    hybrid_confidences,
    threshold: float = 0.3,
) -> np.ndarray:
    """批量状态判定 → str 数组，长度 n_cells（向量化）。"""
    cr = np.asarray(coarse_refs, dtype=object)
    cd = np.asarray(coarse_dbs, dtype=object)
    cf = np.asarray(hybrid_confidences, dtype=np.float64)

    miss_r = cr == None  # noqa: E711 — 对象数组逐元素 None 判断
    miss_d = cd == None  # noqa: E711
    same = (cr == cd) & ~miss_r & ~miss_d

    status = np.empty(len(cr), dtype=object)
    status[same & (cf >= threshold)] = CONSISTENT
    status[same & (cf < threshold)] = LOW_CONFIDENCE
    status[~same] = UNKNOWN
    return status
