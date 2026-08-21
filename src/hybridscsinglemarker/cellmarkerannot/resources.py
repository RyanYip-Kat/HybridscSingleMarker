"""Source resolution and load dispatch for CellMarker datasets (parquet or TSV).

Answers the question *"where is the data for dataset X, and how do I read it?"*
regardless of format. Resolution priority (highest first):

1. explicit ``path``
2. ``CELLMARKERANNOT_DB_DIR`` override (prefers ``<dataset>.parquet``, else ``<dataset>.txt``)
3. bundled package data (``DATA_DIR/<dataset>.parquet``)
4. project-root database/ (``DATABASE_DIR/<dataset>.txt``)

Config values are read at call time (not import time) so tests can monkeypatch
them, and so the ``CELLMARKERANNOT_DB_DIR`` environment variable takes effect
without a reload.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator, Sequence

import pandas as pd

from . import _config as _cfg
from .io import clean_ids, read_database


def resolve_source(
    dataset: str = "all_cell_marker",
    path: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the highest-priority existing source file for ``dataset``.

    Raises ``ValueError`` for an unknown dataset name and ``FileNotFoundError``
    (listing every location searched) when no source exists.
    """
    if dataset not in _cfg.DATASET_FILES:
        raise ValueError(
            f"Unknown dataset {dataset!r}; choose from {sorted(_cfg.DATASET_FILES)}"
        )

    if path is not None:
        src = Path(path).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Explicit source does not exist: {src}")
        return src

    searched: list[str] = []

    env_dir = os.environ.get("CELLMARKERANNOT_DB_DIR")
    if env_dir:
        for candidate in (_cfg.PARQUET_FILES[dataset], _cfg.DATASET_FILES[dataset]):
            src = Path(env_dir).resolve() / candidate
            searched.append(str(src))
            if src.is_file():
                return src

    bundled = _cfg.DATA_DIR / _cfg.PARQUET_FILES[dataset]
    searched.append(str(bundled))
    if bundled.is_file():
        return bundled

    fallback = _cfg.DATABASE_DIR / _cfg.DATASET_FILES[dataset]
    searched.append(str(fallback))
    if fallback.is_file():
        return fallback

    raise FileNotFoundError(
        f"No source found for dataset {dataset!r}. Searched: " + "; ".join(searched)
        + " (set CELLMARKERANNOT_DB_DIR to point at a database directory)"
    )


def read_parquet(
    path: str | os.PathLike[str],
    columns: Sequence[str] = _cfg.COLUMNS,
    *,
    engine: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a CellMarker parquet file and apply the standard cleaning.

    Cleaning (fill NA + strip ``".0"`` + strip whitespace) is idempotent, so it
    is safe to apply both to pre-cleaned bundled data and to raw fixtures.
    Returns a fresh DataFrame whose columns equal ``columns`` in order.
    """
    engine = engine or "pyarrow"
    df = pd.read_parquet(path, columns=list(columns), engine=engine, **kwargs)
    df = df.fillna("")
    df = clean_ids(df)
    df = df[list(columns)]
    # Normalize to object dtype so the parquet and TSV paths return identical frames.
    return df.astype(object)


def read_dataset(
    dataset: str = "all_cell_marker",
    path: str | os.PathLike[str] | None = None,
    *,
    columns: Sequence[str] = _cfg.COLUMNS,
    engine: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Resolve ``dataset`` and read it, dispatching on file suffix.

    ``.parquet`` files go through :func:`read_parquet`; anything else is read
    as a TSV via :func:`cellmarkerannot.io.read_database`.
    """
    src = resolve_source(dataset, path=path)
    if src.suffix.lower() == ".parquet":
        return read_parquet(src, columns=columns, engine=engine, **kwargs)
    return read_database(src, columns=columns, engine=engine, **kwargs)


def iter_sources() -> Iterator[tuple[str, Path]]:
    """Yield ``(dataset_name, resolved_path)`` for every dataset that has a source.

    Datasets without a resolvable source are skipped.
    """
    for name in _cfg.DATASET_FILES:
        try:
            yield name, resolve_source(name)
        except FileNotFoundError:
            continue
