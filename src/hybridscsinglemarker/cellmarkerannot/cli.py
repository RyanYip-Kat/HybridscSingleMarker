"""Command-line interface for cellmarkerannot.

Three subcommands:

- ``query``    — filter the CellMarker database rows (species / tissue / cell type / marker).
- ``annotate`` — annotate every cell of an AnnData file with its most likely cell
  type (:func:`cellmarkerannot.annotate_cells`, incl. auto label-merging).
- ``score``    — gene-list cell-type enrichment: an integer support-evidence
  matrix with a hypergeometric Score row (:func:`cellmarkerannot.score_gene_list`),
  optionally plotted as a heatmap.

Heavy imports (anndata, the database, matplotlib) are done lazily inside each
handler so ``cellmarkerannot --help`` / ``--version`` stay fast.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .__version__ import __version__

_QUERY_FILTERS = (
    "species",
    "tissue_class",
    "tissue_type",
    "cell_name_class",
    "cell_name",
    "marker",
    "marker_source",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser with ``query`` / ``annotate`` / ``score`` subcommands."""
    from .annotation import SCORE_METHODS

    parser = argparse.ArgumentParser(
        prog="cellmarkerannot",
        description=(
            "Cell-type annotation toolkit built on the CellMarker 3.0 marker database."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    query = sub.add_parser("query", help="query the CellMarker database")
    query.add_argument(
        "--dataset", default="all_cell_marker",
        help="dataset name (default: all_cell_marker)",
    )
    query.add_argument("--species", help="filter by species (Human / Mouse)")
    query.add_argument("--tissue-class", dest="tissue_class", help="filter by tissue class")
    query.add_argument("--tissue-type", dest="tissue_type", help="filter by tissue type")
    query.add_argument(
        "--cell-name-class", dest="cell_name_class", help="filter by cell type class"
    )
    query.add_argument("--cell-name", dest="cell_name", help="filter by cell type name")
    query.add_argument("--marker", help="filter by marker gene")
    query.add_argument("--marker-source", dest="marker_source", help="filter by marker source")
    query.add_argument("--limit", type=int, default=20, help="max rows to print (default: 20)")
    query.add_argument("-o", "--output", help="write the full result to this TSV path")

    annotate = sub.add_parser("annotate", help="annotate an AnnData .h5ad file")
    annotate.add_argument("--input", required=True, help="path to .h5ad (AnnData)")
    annotate.add_argument("--dataset", default="all_cell_marker")
    annotate.add_argument("--species", required=True, help="scoring species (Human / Mouse)")
    annotate.add_argument("--tissue", required=True, help="scoring tissue")
    annotate.add_argument("--method", choices=SCORE_METHODS, default="weighted")
    annotate.add_argument("--layer", default=None, help="AnnData layer (default: adata.X)")
    annotate.add_argument(
        "--expr-threshold", type=float, default=1.0,
        help="only count markers expressed above this (default: 1.0)",
    )
    annotate.add_argument(
        "--no-normalize", action="store_true",
        help="do not size-normalize the weighted score",
    )
    annotate.add_argument(
        "--max-cell-types", type=int, default=None,
        help="max predicted labels (default: auto knee+coverage; 0 = keep all)",
    )
    annotate.add_argument(
        "--min-cells", type=int, default=100,
        help="drop labels predicted for fewer cells (default: 100)",
    )
    annotate.add_argument(
        "--coverage", type=float, default=0.9,
        help="auto-mode coverage floor for label merging (default: 0.9)",
    )
    annotate.add_argument(
        "-o", "--output", default="annotated_output",
        help="output directory (default: annotated_output/)",
    )
    annotate.add_argument(
        "--save-adata", action="store_true", help="also write the annotated .h5ad",
    )

    score = sub.add_parser("score", help="gene-list cell-type enrichment matrix")
    score.add_argument("--genes", required=True, help="comma-separated marker genes")
    score.add_argument("--dataset", default="all_cell_marker")
    score.add_argument("--species", required=True)
    score.add_argument("--tissue", required=True)
    score.add_argument(
        "--data-source", default="all",
        help="source filter for the evidence counts (all/method/experiment/review/company/single_cell)",
    )
    score.add_argument(
        "-o", "--output", default=None,
        help="output directory (writes evidence matrix CSV + heatmap PNG)",
    )
    return parser


def _db(dataset: str):
    from .db import CellMarkerDB

    return CellMarkerDB(dataset=dataset)


def cmd_query(args: argparse.Namespace) -> int:
    """Load a CellMarkerDB, apply query filters, print (and optionally save) rows."""
    db = _db(args.dataset)
    filters = {name: getattr(args, name) for name in _QUERY_FILTERS if getattr(args, name)}
    result = db.query(**filters)
    if result.empty:
        print("no rows match the filters")
        return 0
    print(result.head(args.limit).to_string(max_colwidth=40))
    if args.output:
        result.to_csv(args.output, sep="\t", index=False)
        print(f"wrote {len(result)} rows to {args.output}")
    return 0


def cmd_annotate(args: argparse.Namespace) -> int:
    """Load AnnData + CellMarkerDB, annotate cells, save the result."""
    import anndata as ad

    from .annotation import annotate_cells

    db = _db(args.dataset)
    adata = ad.read_h5ad(args.input)
    annotate_cells(
        adata, db,
        method=args.method, species=args.species, tissue=args.tissue, layer=args.layer,
        expr_threshold=args.expr_threshold, normalize=not args.no_normalize,
        max_cell_types=args.max_cell_types, min_cells=args.min_cells,
        coverage=args.coverage, inplace=True,
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    obs_path = out / "annotated_obs.csv"
    adata.obs.to_csv(obs_path)
    print(f"annotated {adata.n_obs} cells -> {obs_path}")
    print("top predicted cell types:")
    for ct, n in adata.obs["celltype_predicted"].value_counts().head(10).items():
        print(f"  {str(ct):<34} {int(n):>7,}")
    if args.save_adata:
        h5_path = out / "annotated.h5ad"
        adata.write_h5ad(h5_path)
        print(f"wrote annotated AnnData -> {h5_path}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Compute a gene-list evidence matrix + enrichment Score row, optionally plot it."""
    from .annotation import score_gene_list

    db = _db(args.dataset)
    genes = [g.strip() for g in args.genes.split(",") if g.strip()]
    if not genes:
        raise ValueError("--genes must contain at least one gene")
    mat = score_gene_list(genes, db, species=args.species, tissue=args.tissue,
                          data_source=args.data_source)
    top = mat.iloc[-1].sort_values(ascending=False)
    print(f"matched {len(mat) - 1} genes across {mat.shape[1]} cell types")
    print("top cell types by Score:")
    for ct, s in top.head(10).items():
        print(f"  {str(ct):<34} {float(s):.4f}")
    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        mat_path = out / "gene_scores_matrix.csv"
        mat.to_csv(mat_path)
        print(f"wrote evidence matrix -> {mat_path}")
        from .plotting import plot_gene_scores

        fig = plot_gene_scores(mat, title=f"{args.species} / {args.tissue} marker scores",
                               save_path=out / "gene_scores_heatmap.png")
        import matplotlib.pyplot as plt

        plt.close(fig)
        print(f"wrote heatmap -> {out / 'gene_scores_heatmap.png'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch to the requested command. Returns an exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {"query": cmd_query, "annotate": cmd_annotate, "score": cmd_score}
    try:
        return handlers[args.command](args)
    except Exception as exc:  # FileNotFoundError, ValueError, ...
        print(f"error: {exc}", file=sys.stderr)
        return 2
