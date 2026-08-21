"""Read CellMarker TSV files into cleaned pandas DataFrames."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ._config import (
    COLUMNS,
    FLOAT_ID_COLUMNS,
    get_database_path,
)

# Matches a whole numeric field written as a float, e.g. "353156.0" -> "353156".
_FLOAT_SUFFIX_RE = r"^(\d+)\.0$"


def _default_engine() -> str:
    """Pick the pandas ``read_csv`` engine.

    Prefers the fast pyarrow engine when available, falling back to the classic
    "c" engine otherwise (pyarrow is optional, see the "fast" extra).
    """
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return "c"
    return "pyarrow"


def read_database(
    path: str | os.PathLike[str],
    columns: Sequence[str] = COLUMNS,
    *,
    engine: str | None = None,
    fill_na: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a CellMarker TSV into a cleaned DataFrame.

    The TSV files share one 22-column, tab-delimited, header-row schema with no
    quoting. This function reads them as strings and applies the documented
    data-quality fixes:

    - pandas 3.0 turns empty cells into ``NaN`` even with ``dtype=str``, so they
      are filled with ``""`` (unless ``fill_na=False``).
    - float-like ID strings (``gene_id``/``pmid``/``year``, e.g. ``"353156.0"``)
      have the trailing ``".0"`` stripped.
    - leading/trailing whitespace is stripped from every column.

    Returns a fresh DataFrame whose columns equal ``columns`` in order.
    """
    engine = engine if engine is not None else _default_engine()
    kwargs.setdefault("sep", "\t")
    kwargs.setdefault("dtype", str)
    kwargs.setdefault("header", 0)
    kwargs.setdefault("usecols", list(columns))
    try:
        df = pd.read_csv(path, engine=engine, **kwargs)
    except (ImportError, ValueError, TypeError):
        # The pyarrow engine can be picky about dtype/usecols combinations;
        # fall back to the classic C engine rather than failing.
        df = pd.read_csv(path, engine="c", **kwargs)
    if fill_na:
        df = df.fillna("")
    df = clean_ids(df)
    # Normalize to object dtype so the TSV and parquet read paths (resources.read_parquet)
    # return identical frames — pandas 3.0 ``dtype=str`` otherwise yields StringDtype.
    return df[list(columns)].astype(object)


def strip_float_suffix(
    df: pd.DataFrame,
    columns: Sequence[str] = tuple(FLOAT_ID_COLUMNS),
) -> pd.DataFrame:
    """Strip the trailing ``".0"`` from float-like ID strings
    (e.g. ``"353156.0"`` -> ``"353156"``).

    Non-matching values (including empty strings / NaN) pass through untouched.
    Returns a copy.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].str.replace(_FLOAT_SUFFIX_RE, r"\1", regex=True)
    return df


def normalize_whitespace(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    collapse: bool = False,
) -> pd.DataFrame:
    """Strip leading/trailing whitespace on string columns (all columns by default).

    If ``collapse=True``, internal runs of whitespace are also collapsed to a
    single space. Returns a copy.
    """
    df = df.copy()
    if columns is None:
        columns = list(df.columns)
    for col in columns:
        if col not in df.columns:
            continue
        if collapse:
            df[col] = df[col].str.strip().str.replace(r"\s+", " ", regex=True)
        else:
            df[col] = df[col].str.strip()
    return df


def clean_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience: strip float suffixes, then strip edge whitespace. Returns a copy."""
    return normalize_whitespace(strip_float_suffix(df))


def load_database(dataset: str = "all_cell_marker", **kwargs: Any) -> pd.DataFrame:
    """Resolve a dataset name and read it (see :func:`read_database`)."""
    return read_database(get_database_path(dataset), **kwargs)
