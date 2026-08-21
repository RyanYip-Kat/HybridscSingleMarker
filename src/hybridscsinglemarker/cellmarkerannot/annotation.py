"""Public API for per-cell cell-type scoring and annotation.

Three scoring methods (:data:`SCORE_METHODS`) compute, for each cell, a raw score
against every cell type in a ``(species, tissue_type)`` scope:

- **overlap** — number of the cell's expressed marker genes that belong to each
  cell type (a specificity-weighted count).
- **weighted** — sum over expressed markers of ``expression × (evidence × idf)``
  (expression-weighted, marker-specificity-aware).
- **ssgsea** — single-sample gene-set enrichment analysis (Barbie-style random
  walk over expression ranks); computed by a numba ``@njit(parallel=True)`` kernel.

Raw scores are normalized per cell to a 0-1 **confidence** via softmax. The
scoring machinery lives in :mod:`cellmarkerannot._scoring`; this module is a
thin orchestration layer over it, plus the gene-list enrichment matrix
(:func:`score_gene_list`) and the label-merging helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._scoring import (
    DEFAULT_EVIDENCE_WEIGHTS,
    MarkerMatrices,
    SSGSEA_POWER,
    _scope_rows,
    build_evidence_matrix,
    build_marker_matrices,
    hypergeom_score_row,
    minmax_rows,
    overlap_scores,
    softmax_rows,
    ssgsea_scores,
    weighted_scores,
)
from .db import CellMarkerDB

if TYPE_CHECKING:
    from anndata import AnnData

SCORE_METHODS: tuple[str, ...] = ("overlap", "weighted", "ssgsea")


def _resolve_x(adata: AnnData, layer: str | None = None) -> sp.csr_matrix:
    """Return ``adata.X`` (or the named ``layer``) as a ``float32`` CSR matrix.

    Backed sparse datasets are materialized with ``to_memory()`` first (a raw
    ``csr_matrix(backed)`` yields object dtype).
    """
    x = adata.X if layer is None else adata.layers[layer]
    if hasattr(x, "to_memory"):  # anndata backed sparse dataset
        x = x.to_memory()
    if not sp.issparse(x):
        x = sp.csr_matrix(x)
    x = x.tocsr()
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    return x


def _check_method(method: str) -> None:
    if method not in SCORE_METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {SCORE_METHODS}")


def score_cell_types(
    query_markers: Sequence[str],
    db: CellMarkerDB,
    *,
    method: str = "overlap",
    species: str,
    tissue: str,
    evidence_weights: dict[str, float] | None = None,
    marker_sources: Sequence[str] | None = None,
    p: float = SSGSEA_POWER,
    min_overlap: int = 1,
) -> pd.DataFrame:
    """Score candidate cell types against a query marker list.

    Useful for cluster-level annotation: given a set of marker genes, return a
    ranked table of the cell types (in the ``(species, tissue)`` scope) they
    match. The query markers are treated as a single pseudo-cell in which every
    in-scope marker present in the query is expressed once.

    Parameters
    ----------
    query_markers:
        Marker gene symbols to score (exact-match against the in-scope DB markers).
    db, species, tissue:
        Database and the scoring scope. ``tissue`` is matched against the DB's
        ``tissue_type`` column only (case-insensitively) for a precise scope.
    method:
        One of :data:`SCORE_METHODS`. ``ssgsea`` on a bare marker list is
        tie-dominated (all query markers share one "expression"), so prefer
        ``overlap``/``weighted`` there.
    min_overlap:
        Drop cell types with fewer than this many matching query markers before
        ranking.

    Returns a DataFrame indexed by ``cell_name`` (sorted by ``score_raw``
    descending) with columns ``score_raw``, ``n_overlap``, ``n_query``, ``n_db``,
    ``frac_query``, ``frac_db``, ``score`` (min-max confidence over the returned
    cell types), and ``rank``.
    """
    _check_method(method)
    markers = list(dict.fromkeys(query_markers))
    mm = build_marker_matrices(
        db, species, tissue, markers, evidence_weights, marker_sources
    )
    g = len(mm.db_genes)

    # Pseudo-cell expressing every in-scope query marker once.
    x_q = sp.csr_matrix(
        (
            np.ones(g, dtype=np.float32),
            (np.zeros(g, dtype=np.int32), np.arange(g, dtype=np.int32)),
        ),
        shape=(1, g),
    )
    if method == "overlap":
        raw = overlap_scores(x_q, mm.A)[0]
    elif method == "weighted":
        raw = weighted_scores(x_q, mm.W)[0]
    else:  # ssgsea
        raw = ssgsea_scores(x_q, mm, p)[0]

    n_overlap = np.asarray(mm.A.sum(axis=0)).ravel().astype(np.int32)
    n_query = g
    n_db = mm.n_markers

    result = pd.DataFrame(
        {
            "score_raw": raw,
            "n_overlap": n_overlap,
            "n_query": n_query,
            "n_db": n_db,
        },
        index=mm.cell_names,
    )
    result["frac_query"] = np.divide(
        result["n_overlap"], result["n_query"],
        out=np.zeros(len(result), dtype=float),
        where=result["n_query"] > 0,
    )
    result["frac_db"] = np.divide(
        result["n_overlap"], result["n_db"],
        out=np.zeros(len(result), dtype=float),
        where=result["n_db"] > 0,
    )
    result = result.loc[result["n_overlap"] >= min_overlap]
    if result.empty:
        return result
    result["score"] = minmax_rows(result["score_raw"].to_numpy().reshape(1, -1)).ravel()
    result["rank"] = result["score_raw"].rank(method="min", ascending=False).astype(int)
    return result.sort_values("score_raw", ascending=False)


def score_cells(
    adata: AnnData,
    db: CellMarkerDB,
    *,
    method: str = "overlap",
    species: str,
    tissue: str,
    layer: str | None = None,
    evidence_weights: dict[str, float] | None = None,
    marker_sources: Sequence[str] | None = None,
    expr_threshold: float = 1.0,
    normalize: bool = True,
    p: float = SSGSEA_POWER,
) -> pd.DataFrame:
    """Compute raw per-cell scores against every cell type in the scope.

    Returns a DataFrame of shape ``(n_cells, n_cell_types)`` indexed by
    ``adata.obs.index`` with the raw scores (not confidence-normalized; see
    :func:`annotate_cells`). ``tissue`` matches the DB ``tissue_type`` only.

    - ``layer`` selects ``adata.layers[layer]`` (default: ``adata.X``). The
      interpretation depends on the expression scale — for raw counts, consider
      passing a log-normalized layer so high-count cells do not dominate the
      ``weighted`` score.
    - ``marker_sources`` restricts which DB markers are used (default:
      Experimentally validated sources, see ``DEFAULT_MARKER_SOURCES``).
    - ``expr_threshold``: only marker genes with expression strictly above this
      value are counted (default 1.0, tuned for log-normalized data; pass 0 to
      disable). Applies to all three methods.
    - ``normalize``: for the ``weighted`` method, divide each cell type's sum by
      its total marker weight (a specificity-weighted *mean* expression). This
      makes cell types comparable regardless of how many markers they carry —
      without it, broad cell types with hundreds of markers dominate the raw
      sum. Has no effect on ``overlap``/``ssgsea``.
    """
    _check_method(method)
    mm = build_marker_matrices(
        db, species, tissue, list(adata.var_names), evidence_weights, marker_sources
    )
    x = _resolve_x(adata, layer)
    x_sub = x[:, mm.gene_var_pos]
    if expr_threshold and expr_threshold > 0:
        x_sub = x_sub.multiply(x_sub > expr_threshold).tocsr()
    if method == "overlap":
        raw = overlap_scores(x_sub, mm.A)
    elif method == "weighted":
        raw = weighted_scores(x_sub, mm.W)
        if normalize:
            total_w = np.asarray(mm.W.sum(axis=0)).ravel()
            raw = raw / np.where(total_w > 0, total_w, 1.0)
    else:  # ssgsea
        raw = ssgsea_scores(x_sub, mm, p)
    return pd.DataFrame(raw, index=adata.obs.index, columns=mm.cell_names)


def _knee_index(sorted_desc: np.ndarray) -> int:
    """Scree-elbow index: how many leading values to keep before the bend.

    Computes the point of maximum perpendicular distance from the diagonal on
    the (rank, normalized-frequency) curve — the classic "elbow" where adding
    the next cell type yields diminishing returns. A flat curve (no clear
    elbow) returns ``len(sorted_desc)`` (keep everything); a single value
    returns 1.
    """
    v = np.asarray(sorted_desc, dtype=np.float64)
    n = len(v)
    if n <= 1:
        return 1
    span = v.max() - v.min()
    y = (v - v.min()) / (span if span > 0 else 1.0)
    x = np.arange(n, dtype=np.float64)
    x1, y1, x2, y2 = 0.0, y[0], float(n - 1), y[-1]
    dx, dy = x2 - x1, y2 - y1
    denom = np.hypot(dx, dy)
    dist = np.abs(dy * x - dx * y + x2 * y1 - y2 * x1) / denom
    if dist.max() <= 1e-9:  # flat / no clear elbow -> keep everything
        return n
    return int(np.argmax(dist)) + 1


def _coverage_index(sorted_desc: np.ndarray, coverage: float) -> int:
    """Smallest number of leading values whose cumulative share reaches ``coverage``."""
    v = np.asarray(sorted_desc, dtype=np.float64)
    n = v.size
    if n == 0:
        return 0
    total = v.sum()
    if total <= 0:
        return n
    cum = np.cumsum(v) / total
    k = int(np.searchsorted(cum, coverage)) + 1
    return min(max(k, 1), n)


def _merge_labels(
    pred: pd.Series,
    max_cell_types: int | None,
    min_cells: int,
    coverage: float = 0.9,
) -> pd.Series:
    """Merge rare predicted labels into ``"Other"``.

    - ``max_cell_types=0``: return ``pred`` unchanged (no merging).
    - ``max_cell_types`` a positive int: keep the top ``max_cell_types`` labels
      by predicted frequency (after the ``min_cells`` floor).
    - ``max_cell_types=None`` (auto): keep ``max`` of the scree elbow
      (:func:`_knee_index`) and the number needed to cover ``coverage``
      (:func:`_coverage_index`) of the predicted cells — the coverage floor
      bounds how many cells end up in ``"Other"``, while the elbow keeps the
      informative types when the distribution is steep.

    Labels with fewer than ``min_cells`` predicted cells are always collapsed to
    ``"Other"`` (both modes); if that would drop everything the floor is relaxed
    so small datasets still receive labels. ``NaN`` predictions are kept.
    """
    if max_cell_types == 0:
        return pred
    freq = pred.value_counts().sort_values(ascending=False)
    candidates = freq[freq >= min_cells]
    if candidates.empty:
        candidates = freq
    if max_cell_types is None:
        vals = candidates.to_numpy()
        k = max(_knee_index(vals), _coverage_index(vals, coverage))
    else:
        k = min(int(max_cell_types), len(candidates))
    top = set(candidates.index[:k])
    return pred.where(pred.isna() | pred.isin(top), "Other")


def annotate_cells(
    adata: AnnData,
    db: CellMarkerDB,
    *,
    method: str = "weighted",
    species: str,
    tissue: str,
    layer: str | None = None,
    inplace: bool = True,
    evidence_weights: dict[str, float] | None = None,
    marker_sources: Sequence[str] | None = None,
    expr_threshold: float = 1.0,
    normalize: bool = True,
    p: float = SSGSEA_POWER,
    max_cell_types: int | None = None,
    min_cells: int = 100,
    coverage: float = 0.9,
) -> pd.DataFrame | None:
    """Annotate every cell with its most likely cell type.

    Scores each cell against the cell types in the ``(species, tissue)`` scope
    and assigns the argmax cell type. ``confidence`` is the **softmax
    probability** of the predicted cell type over that cell's raw scores — it
    reflects how strongly the winner outscores the alternatives (near 1.0 for a
    clear winner, lower when the top cell types are close), and 0 for cells
    with no expressed marker genes (raw row all zero, which also get a ``NaN``
    prediction).

    ``max_cell_types`` controls how many distinct cell-type labels the result
    may carry (keeping visualization tractable when the scope contains hundreds
    of cell types). ``None`` (default) = **auto mode**: the number is chosen as
    ``max`` of the scree elbow (:func:`_knee_index`) and the count needed to
    cover ``coverage`` (:func:`_coverage_index`, default 0.9) of the predicted
    cells — the coverage floor bounds how many cells fall into ``"Other"``. A
    positive int sets a fixed cap; ``0`` disables merging entirely (every
    predicted label is kept). ``min_cells`` (default 100) additionally drops
    any label predicted for fewer than that many cells to ``"Other"`` (applied
    in auto and fixed-cap modes; lower it for small datasets).

    ``marker_sources``, ``expr_threshold`` and ``normalize`` are forwarded to
    :func:`score_cells`.

    With ``inplace=True``, sets ``adata.obs["celltype_predicted"]`` and
    ``adata.obs["confidence"]`` and returns None; otherwise returns a DataFrame
    with those two columns.
    """
    scores = score_cells(
        adata, db,
        method=method, species=species, tissue=tissue, layer=layer,
        evidence_weights=evidence_weights, marker_sources=marker_sources,
        expr_threshold=expr_threshold, normalize=normalize, p=p,
    )
    raw = scores.to_numpy(dtype=np.float32)
    row_max = raw.max(axis=1)
    pred_idx = raw.argmax(axis=1)
    cell_types = np.asarray(scores.columns)
    pred = np.where(row_max > 0, cell_types[pred_idx], np.nan)
    probs = softmax_rows(raw)
    confidence = np.where(row_max > 0, probs[np.arange(raw.shape[0]), pred_idx], 0.0)

    if max_cell_types != 0:
        pred = _merge_labels(
            pd.Series(pred, index=adata.obs.index),
            max_cell_types,
            min_cells,
            coverage,
        ).to_numpy()

    annotations = pd.DataFrame(
        {"celltype_predicted": pred, "confidence": confidence},
        index=adata.obs.index,
    )
    if inplace:
        adata.obs["celltype_predicted"] = annotations["celltype_predicted"].astype("string")
        adata.obs["confidence"] = annotations["confidence"]
        return None
    return annotations


# ---------------------------------------------------------------------------
# Gene-list evidence matrix + hypergeometric enrichment score
# ---------------------------------------------------------------------------

# User-facing data_source -> set of DB marker_source values (None = every source).
_DATA_SOURCE_ALIASES: dict[str, frozenset[str] | None] = {
    "all": None,
    "experiment": frozenset({"Experiment"}),
    "method": frozenset({"Method"}),
    "review": frozenset({"Review"}),
    "company": frozenset({"Company"}),
    "single_cell": frozenset({"Single-cell sequencing"}),
    "singlecell": frozenset({"Single-cell sequencing"}),
    "single-cell sequencing": frozenset({"Single-cell sequencing"}),
    "single-cell": frozenset({"Single-cell sequencing"}),
    "single cell sequencing": frozenset({"Single-cell sequencing"}),
}


def _resolve_data_sources(data_source: str | Sequence[str]) -> frozenset[str] | None:
    """Map user ``data_source`` value(s) to DB ``marker_source`` values.

    Accepts a single string or a sequence; each item is stripped and lower-cased
    before lookup. ``"all"`` (or a mixed list containing it) resolves to ``None``
    = no source filter. Unknown values raise ``ValueError``.
    """
    values = [data_source] if isinstance(data_source, str) else list(data_source)
    if not values:
        raise ValueError("data_source must be a non-empty string or sequence")
    resolved: set[str] = set()
    for value in values:
        key = value.strip().lower()
        if key == "all":
            return None
        src = _DATA_SOURCE_ALIASES.get(key)
        if src is None:
            raise ValueError(
                f"unknown data_source {value!r}; choose from "
                f"{sorted(k for k in _DATA_SOURCE_ALIASES if k != 'all')} or 'all'"
            )
        resolved.update(src)
    return frozenset(resolved)


def score_gene_list(
    genes: Sequence[str],
    db: CellMarkerDB,
    *,
    species: str,
    tissue: str,
    data_source: str | Sequence[str] = "all",
) -> pd.DataFrame:
    """Return a gene × cell-type support-evidence matrix with an enrichment Score row.

    Index (rows):
        matched genes, in input order (first occurrence; duplicates collapsed).
        Input genes with no supporting record in the scope are dropped.
    Columns:
        in-scope cell types with >= 1 matched gene, sorted.
    Values:
        integer-valued support evidence — the number of in-scope DB rows citing
        each ``(gene, cell_type)`` pair within ``data_source``. The columns are
        float64 (the whole-number evidence values sit in the same columns as the
        float64 Score row).
    Last row (index ``"Score"``):
        per-cell-type comprehensive confidence score::

            score_c = -log10(P_c) * log1p(sum over matched genes of evidence(g, c))

        where ``P_c = scipy.stats.hypergeom.sf(k_c - 1, N_total, K_c, n_query)``
        (``N_total`` = distinct in-scope markers, ``K_c`` = distinct in-scope
        markers of ``c``, ``n_query`` = number of matched genes, ``k_c`` =
        number of matched genes that are markers of ``c``).

    ``tissue`` matches the DB ``tissue_type`` column only (case-insensitive).
    ``data_source`` is one of ``"all"`` (default; every source), ``"experiment"``,
    ``"method"``, ``"single_cell"`` (aliases: ``"singlecell"``,
    ``"single-cell sequencing"``), ``"review"``, ``"company"`` — or a list
    combining several (matched case-insensitively; ``"all"`` overrides). Unknown
    values raise ``ValueError``.

    Note: markers are matched to the input exactly (case-sensitive); the Score
    row is always the *last* row — access it via ``iloc[-1]``.
    """
    marker_sources = _resolve_data_sources(data_source)
    in_scope = _scope_rows(db, species, tissue, marker_sources)
    evidence = build_evidence_matrix(
        db, species, tissue, marker_sources, genes, in_scope=in_scope
    )
    n_total = int(in_scope["marker"].nunique())
    k_per_celltype = in_scope.groupby("cell_name", sort=False)["marker"].nunique()
    score = hypergeom_score_row(
        evidence, N_total=n_total, K_per_celltype=k_per_celltype
    )
    out = pd.concat([evidence, score.to_frame().T])
    out.index.name = None
    out.columns.name = None
    return out
