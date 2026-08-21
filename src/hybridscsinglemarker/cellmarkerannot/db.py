"""CellMarker database access: lazy loading, hierarchical index, and queries.

Each dataset is stored as a cleaned 22-column DataFrame (the canonical store)
plus a nested hierarchy index::

    species -> tissue_class -> tissue_type -> cell_name -> frozenset(marker)

for fast marker lookup. Frames and hierarchy indexes are cached module-wide, so
repeated :class:`CellMarkerDB` construction does not re-read the (potentially
multi-GB) source. :meth:`CellMarkerDB.frame` is a **read-only** contract:
query methods return new objects and never mutate the frame.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from . import resources
from ._config import COLUMNS
from .io import read_database

# Levels of the nested hierarchy index (deepest is cell_name -> markers).
HIERARCHY_KEYS: tuple[str, ...] = ("species", "tissue_class", "tissue_type", "cell_name")

# Cache key = (resolved source path, tuple(columns)). ``engine`` is deliberately
# NOT part of the key: it changes only the read mechanism, not the cleaned content.
_FRAME_CACHE: dict[tuple[Path, tuple[str, ...]], pd.DataFrame] = {}
_HIERARCHY_CACHE: dict[tuple[Path, tuple[str, ...]], dict | None] = {}

# Sentinel distinguishing "hierarchy not built yet" from "built but not buildable".
_UNSET: object = object()


def clear_cache() -> None:
    """Drop cached frames and hierarchy indexes.

    Call after rebuilding the bundled Parquet data, or between tests. Existing
    ``CellMarkerDB`` instances keep working (their references remain valid);
    only newly-loaded instances re-read from disk.
    """
    _FRAME_CACHE.clear()
    _HIERARCHY_CACHE.clear()


def _build_hierarchy(df: pd.DataFrame) -> dict:
    """Build the nested species/tissue/cell-type -> markers index from a frame."""
    grouped = (
        df.groupby(list(HIERARCHY_KEYS), sort=False)["marker"].agg(frozenset)
    )
    hierarchy: dict = {}
    for (species, tissue_class, tissue_type, cell_name), markers in grouped.items():
        hierarchy.setdefault(species, {}).setdefault(tissue_class, {}).setdefault(
            tissue_type, {}
        )[cell_name] = markers
    return hierarchy


def _iter_branches(
    hierarchy: dict,
    species: str | None = None,
    tissue_class: str | None = None,
    tissue_type: str | None = None,
):
    """Yield the innermost ``{cell_name: frozenset(markers)}`` dicts matching the filters."""
    sp_iter = [species] if species is not None else list(hierarchy)
    for sp in sp_iter:
        if sp not in hierarchy:
            continue
        tc_map = hierarchy[sp]
        tc_iter = [tissue_class] if tissue_class is not None else list(tc_map)
        for tc in tc_iter:
            if tc not in tc_map:
                continue
            tt_map = tc_map[tc]
            if tissue_type is not None:
                if tissue_type in tt_map:
                    yield tt_map[tissue_type]
            else:
                yield from tt_map.values()


class CellMarkerDB:
    """Access a single CellMarker dataset (parquet or TSV).

    Construction is cheap and lazy by default: the table is only read into
    memory on first access to :attr:`frame` / :attr:`data`. Loaded frames and
    hierarchy indexes are shared across instances via the module cache.
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        dataset: str = "all_cell_marker",
        *,
        columns: Sequence[str] = COLUMNS,
        lazy: bool = True,
        engine: str | None = None,
    ) -> None:
        """Configure the database handle.

        - ``path``: explicit source file (parquet or TSV). If None, the source
          is resolved from ``dataset`` via :func:`cellmarkerannot.resources.resolve_source`.
        - ``dataset``: one of :data:`cellmarkerannot._config.DATASET_FILES`.
        - ``lazy=False``: read the table eagerly in the constructor.
        """
        self.path: Path = resources.resolve_source(dataset, path=path)
        self.dataset: str = dataset
        self.format: str = self.path.suffix.lower().lstrip(".")
        self._columns: tuple[str, ...] = tuple(columns)
        self._engine: str | None = engine
        self._frame: pd.DataFrame | None = None
        self._hierarchy: dict | None | object = _UNSET
        if not lazy:
            self._ensure_loaded()

    @property
    def frame(self) -> pd.DataFrame:
        """The loaded table, loaded on first access (read-only contract)."""
        if self._frame is None:
            self._ensure_loaded()
        assert self._frame is not None
        return self._frame

    @property
    def data(self) -> pd.DataFrame:
        """Alias for :attr:`frame`."""
        return self.frame

    def _cache_key(self) -> tuple[Path, tuple[str, ...]]:
        return (self.path.resolve(), self._columns)

    def _ensure_loaded(self) -> None:
        """Load :attr:`frame`, reusing the module cache when possible."""
        key = self._cache_key()
        if key in _FRAME_CACHE:
            self._frame = _FRAME_CACHE[key]
            return
        if self.path.suffix.lower() == ".parquet":
            df = resources.read_parquet(
                self.path, columns=self._columns, engine=self._engine
            )
        else:
            df = read_database(self.path, columns=self._columns, engine=self._engine)
        _FRAME_CACHE[key] = df
        self._frame = df

    def _ensure_hierarchy(self) -> dict | None:
        """Return the cached hierarchy index, or None if it cannot be built.

        Building is skipped (returns None) when ``columns`` omits any of the
        hierarchy/marker columns; callers then fall back to DataFrame filtering.
        """
        if self._hierarchy is not _UNSET:
            return self._hierarchy
        key = self._cache_key()
        if key in _HIERARCHY_CACHE:
            self._hierarchy = _HIERARCHY_CACHE[key]
            return self._hierarchy
        required = set(HIERARCHY_KEYS) | {"marker"}
        if not required.issubset(self._columns):
            self._hierarchy = None
            _HIERARCHY_CACHE[key] = None
            return None
        hierarchy = _build_hierarchy(self.frame)
        _HIERARCHY_CACHE[key] = hierarchy
        self._hierarchy = hierarchy
        return hierarchy

    def query(
        self,
        *,
        species: str | None = None,
        tissue_class: str | None = None,
        tissue_type: str | None = None,
        uberon_id: str | None = None,
        disease: str | None = None,
        cell_name_class: str | None = None,
        cell_name: str | None = None,
        cellontology_id: str | None = None,
        marker: str | None = None,
        symbol: str | None = None,
        gene_id: str | None = None,
        gene_type: str | None = None,
        gene_name: str | None = None,
        uniprot_id: str | None = None,
        technology_seq: str | None = None,
        marker_source: str | None = None,
        pmid: str | None = None,
        title: str | None = None,
        journal: str | None = None,
        year: str | int | None = None,
        series_id: str | None = None,
        method_details: str | None = None,
        case_sensitive: bool = False,
    ) -> pd.DataFrame:
        """Return rows matching all provided filters (``None`` = no constraint).

        Filters are matched against the cleaned frame. Values are coerced to
        ``str`` first, so ``year=2020`` matches the cleaned ``"2020"``. Matching
        is case-insensitive by default (``case_sensitive=True`` to opt out).

        Returns a new DataFrame; the underlying frame is never mutated.
        """
        filters = {
            k: v
            for k, v in locals().items()
            if k not in ("self", "case_sensitive") and v is not None
        }
        frame = self.frame
        if not filters:
            return frame.copy()
        mask = pd.Series(True, index=frame.index)
        if case_sensitive:
            for col, val in filters.items():
                mask &= frame[col] == str(val)
        else:
            for col, val in filters.items():
                mask &= frame[col].str.lower() == str(val).lower()
        return frame.loc[mask]

    def markers_for_cell_type(
        self,
        cell_name: str,
        *,
        species: str | None = None,
        tissue_class: str | None = None,
        tissue_type: str | None = None,
        marker_source: str | None = None,
    ) -> list[str]:
        """Sorted unique ``marker`` values for a cell type.

        Uses the hierarchy index for fast lookups when no ``marker_source``
        filter is given; otherwise (or when the index is unavailable) falls back
        to DataFrame filtering. Values are exact-matched (case-sensitive).
        """
        hierarchy = self._ensure_hierarchy()
        if marker_source is not None or hierarchy is None:
            f = self.frame
            mask = f["cell_name"] == cell_name
            for col, val in (
                ("species", species),
                ("tissue_class", tissue_class),
                ("tissue_type", tissue_type),
                ("marker_source", marker_source),
            ):
                if val is not None:
                    mask &= f[col] == val
            return sorted(f.loc[mask, "marker"].dropna().unique().tolist())

        markers: set[str] = set()
        for branch in _iter_branches(hierarchy, species, tissue_class, tissue_type):
            markers.update(branch.get(cell_name, ()))
        return sorted(markers)

    def cell_types_for_marker(
        self,
        marker: str,
        *,
        species: str | None = None,
        tissue_class: str | None = None,
        tissue_type: str | None = None,
    ) -> list[str]:
        """Sorted unique ``cell_name`` values citing this marker."""
        hierarchy = self._ensure_hierarchy()
        if hierarchy is None:
            f = self.frame
            mask = f["marker"] == marker
            for col, val in (
                ("species", species),
                ("tissue_class", tissue_class),
                ("tissue_type", tissue_type),
            ):
                if val is not None:
                    mask &= f[col] == val
            return sorted(f.loc[mask, "cell_name"].dropna().unique().tolist())

        cell_names: set[str] = set()
        for branch in _iter_branches(hierarchy, species, tissue_class, tissue_type):
            for cell_name, marker_set in branch.items():
                if marker in marker_set:
                    cell_names.add(cell_name)
        return sorted(cell_names)

    def available_species(self) -> list[str]:
        """Sorted distinct ``species`` values in the dataset."""
        hierarchy = self._ensure_hierarchy()
        if hierarchy is not None:
            return sorted(hierarchy)
        return sorted(self.frame["species"].dropna().unique().tolist())

    def available_tissues(self, tissue_class: str | None = None) -> list[str]:
        """Sorted distinct ``tissue_type`` values, optionally within a tissue class."""
        hierarchy = self._ensure_hierarchy()
        if hierarchy is None:
            f = self.frame
            mask = pd.Series(True, index=f.index)
            if tissue_class is not None:
                mask &= f["tissue_class"] == tissue_class
            return sorted(f.loc[mask, "tissue_type"].dropna().unique().tolist())

        tissues: set[str] = set()
        if tissue_class is not None:
            for tc_map in hierarchy.values():
                tt_map = tc_map.get(tissue_class)
                if tt_map is not None:
                    tissues.update(tt_map.keys())
        else:
            for tc_map in hierarchy.values():
                for tt_map in tc_map.values():
                    tissues.update(tt_map.keys())
        return sorted(tissues)

    def available_cell_names(
        self,
        *,
        tissue_class: str | None = None,
        species: str | None = None,
    ) -> list[str]:
        """Sorted distinct ``cell_name`` values (optionally filtered)."""
        hierarchy = self._ensure_hierarchy()
        if hierarchy is None:
            f = self.frame
            mask = pd.Series(True, index=f.index)
            for col, val in (
                ("tissue_class", tissue_class),
                ("species", species),
            ):
                if val is not None:
                    mask &= f[col] == val
            return sorted(f.loc[mask, "cell_name"].dropna().unique().tolist())

        names: set[str] = set()
        for branch in _iter_branches(hierarchy, species, tissue_class):
            names.update(branch.keys())
        return sorted(names)

    def available_markers(
        self,
        *,
        cell_name: str | None = None,
        species: str | None = None,
    ) -> list[str]:
        """Sorted distinct ``marker`` values (optionally for a cell type)."""
        if cell_name is None:
            # A single-column scan is the fastest path when no cell type is given.
            return sorted(self.frame["marker"].dropna().unique().tolist())
        hierarchy = self._ensure_hierarchy()
        if hierarchy is None:
            f = self.frame
            mask = f["cell_name"] == cell_name
            if species is not None:
                mask &= f["species"] == species
            return sorted(f.loc[mask, "marker"].dropna().unique().tolist())

        markers: set[str] = set()
        for branch in _iter_branches(hierarchy, species, None, None):
            markers.update(branch.get(cell_name, ()))
        return sorted(markers)

    def __repr__(self) -> str:
        return (
            f"CellMarkerDB(path={str(self.path)!r}, format={self.format!r}, "
            f"dataset={self.dataset!r}, loaded={self._frame is not None})"
        )


def get_database(dataset: str = "all_cell_marker", **kwargs: Any) -> CellMarkerDB:
    """Convenience factory returning a lazy :class:`CellMarkerDB` for ``dataset``."""
    return CellMarkerDB(dataset=dataset, **kwargs)
