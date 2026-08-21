"""hybridscsinglemarker 骨架冒烟测试：包可导入、入口可调用、状态枚举齐全。"""

from __future__ import annotations

import pytest

import hybridscsinglemarker
from hybridscsinglemarker import __version__, hybrid_annotate
from hybridscsinglemarker import _layer


def test_version():
    assert hybridscsinglemarker.__version__ == __version__
    assert isinstance(__version__, str)


def test_hybrid_annotate_is_callable():
    assert callable(hybrid_annotate)


def test_hybrid_annotate_missing_file_raises_filenotfound():
    # 核心逻辑已实现：不存在的路径应报文件不存在，而非 NotImplementedError
    with pytest.raises(FileNotFoundError):
        hybrid_annotate("dummy.h5ad")


def test_layer_status_enum():
    assert set(_layer._ALL_STATUS) == {
        _layer.CONSISTENT,
        _layer.LOW_CONFIDENCE,
        _layer.UNKNOWN,
        _layer.DB_ONLY,
    }
