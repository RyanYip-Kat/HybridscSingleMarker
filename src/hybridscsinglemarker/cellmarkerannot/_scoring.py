"""Internal scoring machinery: marker-matrix construction and the three methods.

Kept private; the public API lives in :mod:`cellmarkerannot.annotation`.

Backend strategy (vectorized, no Python-level per-cell loops):

- **overlap**  : ``(X_sub > 0) @ A`` — counts expressed markers per cell type
                 (scipy.sparse matmul).
- **weighted** : ``X_sub @ W`` — expression values weighted by ``evidence × idf``
                 (scipy.sparse matmul).
- **ssgsea**   : a single-sample GSEA walk, computed per cell by a numba
                 ``@njit(parallel=True)`` kernel (near-C speed, JIT machine code).

``numba`` is imported lazily inside :func:`ssgsea_scores` so that importing this
module (and the package) stays cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

from .db import CellMarkerDB

# Evidence weight by ``marker_source``; for a (marker, cell_type) pair cited by
# several sources the maximum is taken. Unknown sources default to 1.0.
DEFAULT_EVIDENCE_WEIGHTS: dict[str, float] = {
    "Experiment": 1.0,
    "Method": 1.0,
    "Single-cell sequencing": 0.8,
    "Review": 0.6,
    "Company": 0.4,
}

# ``marker_source`` values used by default for scoring. Experimentally validated
# markers only: excluding the computationally-derived (Method) and literature-
# aggregate (Review/Company) markers keeps cell-type marker sets tight — broad
# catch-all cell types otherwise carry ~2000 overlapping markers that dominate
# the scores. Pass ``marker_sources=...`` to override.
DEFAULT_MARKER_SOURCES: tuple[str, ...] = ("Experiment", "Single-cell sequencing")

# Rank power used in the ssGSEA walk (Barbie et al., default 0.25).
SSGSEA_POWER: float = 0.25


@dataclass(frozen=True)
class MarkerMatrices:
    """Precomputed marker membership/weight matrices for one species+tissue scope.

    Attributes
    ----------
    cell_names:
        Sorted cell-type names; the column order of ``A``/``W``.
    db_genes:
        Canonical DB marker strings present in the query gene universe; the row
        order of ``A``/``W`` (ascending position in ``var_names``).
    gene_var_pos:
        ``int64`` position of each ``A`` row within ``var_names`` — lets a caller
        slice ``X[:, gene_var_pos]`` so the column order matches the row order of
        ``A``/``W``.
    A:
        ``float32`` G×C membership matrix (0/1).
    W:
        ``float32`` G×C weight matrix (``evidence × idf``).
    idf:
        ``float32`` (G,) inverse-cell-type-frequency of each marker.
    n_markers:
        ``int32`` (C,) distinct in-scope markers per cell type (the full scope
        count, independent of ``var_names`` matching).
    rows:
        The full in-scope DB rows (before var-name filtering; diagnostics/tests).
    """

    cell_names: list[str]
    db_genes: list[str]
    gene_var_pos: np.ndarray
    A: sp.csr_matrix
    W: sp.csr_matrix
    idf: np.ndarray
    n_markers: np.ndarray
    rows: pd.DataFrame


def _scope_rows(
    db: CellMarkerDB,
    species: str,
    tissue: str,
    marker_sources: Sequence[str] | None = None,
) -> pd.DataFrame:
    """In-scope DB rows for a ``(species, tissue_type)`` scope.

    Rows are restricted to ``species`` (case-insensitive) and to rows whose
    ``tissue_type`` equals ``tissue`` (case-insensitive). Matching by
    ``tissue_type`` only (not ``tissue_class``) keeps the scope precise — a
    broad tissue class such as ``"Blood"`` otherwise pulls in every
    blood-related tissue (aorta, artery, cord blood, ...). ``marker_sources``
    (``None`` = every source, no filter) further restricts ``marker_source``.
    Raises ``ValueError`` if the resulting scope is empty.
    """
    rows = db.query(species=species)
    tl = tissue.lower()
    mask = rows["tissue_type"].str.lower() == tl
    in_scope = rows.loc[mask]
    if marker_sources is not None:
        in_scope = in_scope.loc[in_scope["marker_source"].isin(set(marker_sources))]
    if in_scope.empty:
        raise ValueError(f"no rows in scope species={species!r} tissue={tissue!r}")
    return in_scope


def build_marker_matrices(
    db: CellMarkerDB,
    species: str,
    tissue: str,
    var_names: list[str],
    evidence_weights: dict[str, float] | None = None,
    marker_sources: Sequence[str] | None = None,
) -> MarkerMatrices:
    """Build membership/weight matrices for the ``(species, tissue)`` scope.

    - Rows are restricted to ``species`` (case-insensitive) and to rows whose
      ``tissue_class`` **or** ``tissue_type`` equals ``tissue`` (case-insensitive).
    - ``marker_sources`` further restricts rows to the given ``marker_source``
      values (default: :data:`DEFAULT_MARKER_SOURCES` — experimentally validated
      markers only). Pass the full :data:`MARKER_SOURCES` set to include every
      source.
    - The gene universe is the in-scope markers present in ``var_names`` by
      **exact string match** (absent markers are skipped).
    - ``W[g, c] = evidence_weight(source(g,c)) × idf(g)``, where ``idf(g) =
      1 / log1p(#cell types in scope containing g)`` and the evidence weight is
      the max over ``marker_source`` values for the pair.

    Raises ``ValueError`` if the scope is empty or no marker matches ``var_names``.
    """
    in_scope = _scope_rows(db, species, tissue)
    if marker_sources is not None:
        sources = set(marker_sources)
        in_scope = in_scope.loc[in_scope["marker_source"].isin(sources)]
        if in_scope.empty:
            raise ValueError(
                f"no rows in scope species={species!r} tissue={tissue!r} "
                f"marker_sources={sorted(sources)}"
            )
    else:
        in_scope = in_scope.loc[in_scope["marker_source"].isin(DEFAULT_MARKER_SOURCES)]

    ev = dict(evidence_weights) if evidence_weights else dict(DEFAULT_EVIDENCE_WEIGHTS)

    # All in-scope cell types (column order of A/W) and their full marker counts.
    cell_names = sorted(in_scope["cell_name"].unique())
    n_markers = (
        in_scope.groupby("cell_name", sort=False)["marker"].nunique()
        .reindex(cell_names)
        .fillna(0)
        .to_numpy(dtype=np.int32)
    )

    # --- Gene universe: in-scope markers present in var_names, row order = var position ---
    var_idx = pd.Index(var_names)
    unique_markers = in_scope["marker"].drop_duplicates()
    pos = var_idx.get_indexer(unique_markers)  # -1 for markers absent from var_names
    keep = pos >= 0
    gene_pos = np.sort(pos[keep])
    if gene_pos.size == 0:
        raise ValueError(
            f"no in-scope markers found in var_names "
            f"(species={species!r} tissue={tissue!r})"
        )
    db_genes = [var_names[p] for p in gene_pos]
    gene_lookup = {g: i for i, g in enumerate(db_genes)}
    rows = in_scope.loc[in_scope["marker"].isin(db_genes)]

    # --- Membership A (G×C, 0/1) ---
    pairs = rows[["marker", "cell_name"]].drop_duplicates()
    col_idx = {c: i for i, c in enumerate(cell_names)}
    r = np.asarray([gene_lookup[g] for g in pairs["marker"]], dtype=np.int32)
    c = np.asarray([col_idx[cn] for cn in pairs["cell_name"]], dtype=np.int32)
    G, C = len(db_genes), len(cell_names)
    A = sp.csr_matrix(
        (np.ones(len(r), dtype=np.float32), (r, c)), shape=(G, C)
    )

    # --- Evidence weights: max over marker_source per (marker, cell_name) ---
    src = rows[["marker", "cell_name", "marker_source"]].drop_duplicates()
    ew = src["marker_source"].map(ev).fillna(1.0).astype(np.float32)
    src = src.assign(ew=ew)
    maxew = src.groupby(["marker", "cell_name"], sort=False)["ew"].max()
    pair_index = pairs.set_index(["marker", "cell_name"]).index
    ew_vals = maxew.reindex(pair_index).to_numpy(dtype=np.float32)

    # --- IDF within scope ---
    nct = rows.groupby("marker", sort=False)["cell_name"].nunique()
    idf = (1.0 / np.log1p(nct.reindex(db_genes).to_numpy())).astype(np.float32)

    # --- W ---
    wvals = ew_vals * idf[r]
    W = sp.csr_matrix((wvals, (r, c)), shape=(G, C))

    return MarkerMatrices(
        cell_names=cell_names,
        db_genes=db_genes,
        gene_var_pos=gene_pos,
        A=A,
        W=W,
        idf=idf,
        n_markers=n_markers,
        rows=in_scope,
    )


def overlap_scores(X_sub: sp.csr_matrix, A: sp.csr_matrix) -> np.ndarray:
    """Per-cell counts of expressed markers per cell type. Returns (cells, C) float32."""
    xb = (X_sub > 0).astype(np.float32)
    return (xb @ A).toarray().astype(np.float32)


def weighted_scores(X_sub: sp.csr_matrix, W: sp.csr_matrix) -> np.ndarray:
    """Per-cell weighted sums (expression × evidence×idf). Returns (cells, C) float32."""
    return (X_sub @ W).toarray().astype(np.float32)


def minmax_rows(raw: np.ndarray) -> np.ndarray:
    """Per-row min-max normalize to [0, 1].

    Rows whose max equals min (including all-zero "no evidence" rows) map to 0.0.
    Returns a new ``float32`` array.
    """
    raw = np.asarray(raw, dtype=np.float32)
    row_min = raw.min(axis=1, keepdims=True)
    row_max = raw.max(axis=1, keepdims=True)
    span = row_max - row_min
    out = np.zeros_like(raw)
    np.divide(raw - row_min, span, out=out, where=span > 0)
    return out


def softmax_rows(raw: np.ndarray) -> np.ndarray:
    """Row-wise softmax (numerically stable). Returns a new ``float64`` array.

    Used as the per-cell confidence: the predicted cell type's softmax
    probability reflects how strongly it outscores the alternatives (an
    all-equal row yields the uniform ``1/C``). Callers usually zero-out the
    "no evidence" rows separately.
    """
    raw = np.asarray(raw, dtype=np.float64)
    shifted = raw - raw.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# ssGSEA
# ---------------------------------------------------------------------------

# The raw Python kernel; compiled lazily by numba on first use (see
# ``_get_ssgsea_kernel``). ``prange`` is resolved by numba at compile time.
def _ssgsea_es_raw(x_indptr, x_indices, x_data, a_indptr, a_indices, p, out):
    """Hits-only single-sample GSEA walk.

    Parameters are the CSR arrays of the expression matrix ``X_sub`` (cells×G)
    and of the membership matrix ``A`` (G×C). Because ``X_sub`` columns and
    ``A`` rows are both in ascending ``var_names`` order, ``x_indices[j]`` is
    already the matching ``A`` row index — no gene mapping is needed.
    """
    n_cells = x_indptr.shape[0] - 1
    n_cell_types = out.shape[1]
    for i in prange(n_cells):
        start = x_indptr[i]
        end = x_indptr[i + 1]
        n = end - start
        vals = np.empty(n, np.float64)
        rows = np.empty(n, np.int64)
        cnt = 0
        for j in range(start, end):
            if x_data[j] > 0:  # skip explicit zeros
                vals[cnt] = x_data[j]
                rows[cnt] = x_indices[j]
                cnt += 1
        if cnt == 0:
            continue
        order = np.argsort(vals[:cnt], kind="mergesort")  # ascending expression
        # pass 1: per cell type, total rank^p (Z_c) and hit count (N_h)
        zsum = np.zeros(n_cell_types, np.float64)
        nhit = np.zeros(n_cell_types, np.int64)
        for j in range(cnt):
            w = (j + 1) ** p  # rank = j+1 (1 = lowest expression)
            g = rows[order[j]]
            for t in range(a_indptr[g], a_indptr[g + 1]):
                cc = a_indices[t]
                zsum[cc] += w
                nhit[cc] += 1
        # pass 2: walk in descending expression order, ES peaks only at hits
        run = np.zeros(n_cell_types, np.float64)
        tcnt = np.zeros(n_cell_types, np.int64)
        es = np.zeros(n_cell_types, np.float64)
        for jj in range(cnt):
            j = cnt - 1 - jj  # descending expression index
            pos = jj + 1  # 1-based position in the descending walk
            w = (j + 1) ** p
            g = rows[order[j]]
            for t in range(a_indptr[g], a_indptr[g + 1]):
                cc = a_indices[t]
                tcnt[cc] += 1
                run[cc] += w
                denom = cnt - nhit[cc]
                miss = (pos - tcnt[cc]) / denom if denom > 0 else 0.0
                cand = run[cc] / zsum[cc] - miss
                if cand > es[cc]:
                    es[cc] = cand
        for cc in range(n_cell_types):
            out[i, cc] = es[cc]


_ssgsea_kernel = None


def _get_ssgsea_kernel():
    """Return the numba-compiled SSGSEA kernel, compiling lazily on first use."""
    global _ssgsea_kernel
    if _ssgsea_kernel is None:
        try:
            from numba import njit
            from numba import prange as _prange
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ssgsea scoring requires numba; install with `pip install numba`"
            ) from exc
        _ssgsea_es_raw.__globals__["prange"] = _prange
        _ssgsea_kernel = njit(parallel=True, cache=True)(_ssgsea_es_raw)
    return _ssgsea_kernel


def ssgsea_scores(
    X_sub: sp.csr_matrix,
    mm: MarkerMatrices,
    p: float = SSGSEA_POWER,
) -> np.ndarray:
    """Per-cell ssGSEA enrichment scores against all cell types.

    Ranks are computed over the expressed, marker-relevant genes only (see the
    plan notes: zero-inflated tails contribute nothing and would merely rescale
    the scores). Returns ``(cells, C)`` float32.
    """
    kernel = _get_ssgsea_kernel()
    out = np.zeros((X_sub.shape[0], mm.A.shape[1]), dtype=np.float32)
    x = X_sub.tocsr()
    kernel(
        x.indptr.astype(np.int32),
        x.indices.astype(np.int32),
        x.data.astype(np.float32),
        mm.A.indptr.astype(np.int32),
        mm.A.indices.astype(np.int32),
        np.float64(p),
        out,
    )
    return out


# ---------------------------------------------------------------------------
# Gene-list evidence matrix + hypergeometric enrichment score
# ---------------------------------------------------------------------------


def build_evidence_matrix(
    db: CellMarkerDB,
    species: str,
    tissue: str,
    marker_sources: Sequence[str] | None,
    genes: Sequence[str],
    *,
    in_scope: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a ``(matched genes × cell types)`` integer support-evidence matrix.

    Each cell equals the number of in-scope DB rows citing that ``(marker,
    cell_name)`` pair within ``marker_sources`` (``None`` = every source, no
    source filter). Rows are the input ``genes`` with >= 1 supporting record in
    scope, in input order (first occurrence; duplicates collapsed). Columns are
    the in-scope cell types with >= 1 matched gene, sorted.

    ``in_scope`` optionally reuses a pre-computed :func:`_scope_rows` result to
    avoid re-scanning the (large) database frame. Raises ``ValueError`` if the
    scope is empty or no input gene matches.
    """
    if in_scope is None:
        in_scope = _scope_rows(db, species, tissue, marker_sources)
    genes = list(dict.fromkeys(genes))
    in_scope = in_scope.loc[in_scope["marker"].isin(genes)]
    if in_scope.empty:
        raise ValueError(
            "none of the input genes have a supporting record in scope "
            f"(species={species!r} tissue={tissue!r})"
        )
    in_scope_markers = set(in_scope["marker"])
    matched = [g for g in genes if g in in_scope_markers]  # input order
    cell_types = sorted(in_scope["cell_name"].unique())
    counts = (
        in_scope.groupby(["marker", "cell_name"], sort=False)
        .size()
        .rename("evidence")
        .reset_index()
    )
    return (
        counts.pivot(index="marker", columns="cell_name", values="evidence")
        .reindex(index=matched, columns=cell_types)
        .fillna(0)
        .astype(np.int32)
    )


def hypergeom_score_row(
    evidence: pd.DataFrame,
    *,
    N_total: int,
    K_per_celltype: pd.Series,
) -> pd.Series:
    """Per-cell-type enrichment score ``-log10(P_c) * log1p(evsum_c)``.

    ``P_c = P(X >= k_c)`` with ``X ~ Hypergeom(N_total, K_c, n_query)``, where
    ``n_query`` = number of matched genes (evidence rows), ``k_c`` = number of
    matched genes that are markers of cell type ``c`` (nonzero evidence column),
    ``evsum_c`` = column sum of the evidence. Cell types absent from
    ``K_per_celltype`` get ``K_c = 0`` -> ``P_c = 1`` -> score 0. Returns a
    ``float64`` Series named "Score".
    """
    n_query = evidence.shape[0]
    k = (evidence > 0).sum(axis=0).to_numpy(dtype=np.int64)  # k_c
    ev_sum = evidence.sum(axis=0).to_numpy(dtype=np.float64)
    k_arr = K_per_celltype.reindex(evidence.columns).fillna(0).astype(np.int64).to_numpy()
    pvals = np.ones(len(evidence.columns), dtype=np.float64)
    pos = k > 0
    pvals[pos] = stats.hypergeom.sf(k[pos] - 1, N_total, k_arr[pos], n_query)
    logp = np.where(pvals > 0, -np.log10(pvals), 0.0)  # P > 0 always; defensive
    return pd.Series(logp * np.log1p(ev_sum), index=evidence.columns, name="Score")
