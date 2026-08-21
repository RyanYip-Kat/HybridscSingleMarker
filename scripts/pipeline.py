"""End-to-end annotation pipeline for ``cellmarkerannot``.

Loads the bundled CellMarker database, reads an AnnData file (backed), annotates
every cell against the ``(species, tissue)`` scope, runs a gene-list enrichment
heatmap on a curated marker panel, prints a concise report, and writes outputs.

Usage::

    uv run python scripts/pipeline.py                    # full 58,677 x 33,421
    uv run python scripts/pipeline.py --subset 200       # fast smoke run

The pipeline is also importable::

    from pipeline import run_pipeline
    results = run_pipeline("testdata/pbmc5k_querydata.h5ad", subset=200)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless; must precede any pyplot import

import matplotlib.pyplot as plt

import anndata as ad
import pandas as pd
import scanpy as sc

from hybridscsinglemarker.cellmarkerannot import SCORE_METHODS, CellMarkerDB, annotate_cells, score_gene_list
from hybridscsinglemarker.cellmarkerannot._scoring import build_marker_matrices
from hybridscsinglemarker.cellmarkerannot.plotting import plot_gene_scores

_REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):  # allow `python scripts/pipeline.py` without install
    for _p in (_REPO_ROOT / "src", _REPO_ROOT):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

DEFAULT_INPUT = _REPO_ROOT / "testdata" / "vkhQ8_querydata.h5ad"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "pipeline_output"

# Compact PBMC marker panel covering T / B / NK / Monocyte / DC
# (all present in the Human/Blood scope of the bundled database).
DEFAULT_PBMC_GENES: tuple[str, ...] = (
    "CD3D", "CD4", "CD8A",        # T
    "MS4A1", "CD19",              # B
    "NKG7", "GNLY", "NCAM1",      # NK
    "CD14", "FCGR3A",             # Monocyte
    "FCER1A", "CLEC9A",           # DC
)

# pbmc5k granular ground-truth `celltype` -> broad category (all 21 values).
_TRUTH_BROAD: dict[str, str] = {
    "CD4 Naive": "T", "CD4 Tcm": "T", "CD4 Tem": "T",
    "CD4+CD8": "T", "CD4-CD8": "T", "CD8 CTL": "T",
    "CD8 Naive": "T", "CD8 Tem": "T", "T-mito": "T",
    "Memory BC": "B", "Naive BC": "B", "ASC": "B",
    "NK1": "NK", "NK2": "NK", "NK3": "NK",
    "CD14": "Mono", "CD16": "Mono",
    "cDC": "DC", "pDC": "DC",
    "MEG": "Other", "RBC": "Other",
}


def _predicted_broad(cell: str) -> str:
    """Map a predicted (DB) cell-type name to a broad category."""
    c = cell.lower()
    if "t cell" in c or "t-cell" in c or c.startswith(("cd4", "cd8")):
        return "T"
    if "b cell" in c or "b-cell" in c or "plasma" in c or "plasmablast" in c or "antibody" in c:
        return "B"
    if "nk" in c or "natural killer" in c:
        return "NK"
    if "monocyte" in c or "myeloid" in c:
        return "Mono"
    if "dendritic" in c or c in ("dc", "cdc", "pdc"):
        return "DC"
    return "Other"


def _broad_agreement(obs: pd.DataFrame) -> float | None:
    """Fraction of annotated cells whose broad category matches the ground truth.

    Returns None when the obs has no ``celltype`` column or no comparable cells.
    Cells whose truth maps to "Other" or that have no prediction are excluded.
    """
    if "celltype" not in obs.columns:
        return None
    truth = obs["celltype"].map(_TRUTH_BROAD)
    pred = obs["celltype_predicted"].dropna().map(_predicted_broad)
    pair = pd.DataFrame({"t": truth, "p": pred}).dropna()
    pair = pair.loc[pair["t"] != "Other"]
    if len(pair) == 0:
        return None
    return float((pair["t"] == pair["p"]).mean())


def run_pipeline(
    input_path: str | Path,
    *,
    dataset: str = "all_cell_marker",
    db: CellMarkerDB | None = None,
    species: str = "Human",
    tissue: str = "Blood",
    method: str = "weighted",
    layer: str | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    subset: int | None = None,
    save_obs: bool = True,
    save_heatmap: bool = True,
    save_adata: bool = False,
    genes: Sequence[str] | None = None,
    data_source: str = "all",
    top_n: int = 8,
    title: str | None = None,
    max_cell_types: int | None = None,
    min_cells: int = 100,
    coverage: float = 0.9,
) -> dict:
    """Run the full annotation pipeline and return a results summary dict.

    Steps: load the AnnData (backed), optionally keep the first ``subset`` cells,
    annotate every cell via :func:`cellmarkerannot.annotate_cells` (inplace),
    compute report statistics (+ broad-level agreement against a ground-truth
    ``celltype`` column when present), run ``score_gene_list`` + heatmap, and
    write outputs (annotated obs CSV, heatmap PNG, optionally an annotated h5ad).

    The ``db`` argument lets tests inject a small synthetic database; by default
    the bundled ``dataset`` is used.
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"input file not found: {input_path}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    db = db or CellMarkerDB(dataset=dataset)

    adata = ad.read_h5ad(input_path, backed="r")
    if subset is not None:
        adata = adata[:subset].to_memory()  # materialize only the first N cells

    n_cell_types_scored = len(
        build_marker_matrices(db, species, tissue, list(adata.var_names)).cell_names
    )

    annotate_cells(
        adata, db, method=method, species=species, tissue=tissue,
        layer=layer, inplace=True,
        max_cell_types=max_cell_types, min_cells=min_cells, coverage=coverage,
    )

    outputs: dict[str, str | None] = {
        "obs_csv": None, "heatmap_png": None, "annotated_h5ad": None, "umap_png": None,
    }
    gene_info: dict[str, int] | None = None

    if "X_umap" in adata.obsm:
        adata.obs["celltype_predicted"] = adata.obs["celltype_predicted"].astype("category")
        ax = sc.pl.umap(
            adata,
            color="celltype_predicted",
            legend_loc="right margin",
            show=False,
        )
        umap_path = out / "umap_celltype_predicted.png"
        fig = getattr(ax, "figure", None) or plt.gcf()
        fig.savefig(umap_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        outputs["umap_png"] = str(umap_path)

    pred = adata.obs["celltype_predicted"].dropna()
    n_cells = adata.n_obs
    frac_predicted = float(len(pred) / n_cells)
    mean_confidence = float(adata.obs["confidence"].mean())
    max_confidence = float(adata.obs["confidence"].max())
    prediction_counts = pred.value_counts().head(top_n).astype(int).to_dict()
    broad = _broad_agreement(adata.obs)

    if save_obs:
        obs_path = out / "annotated_obs.csv"
        adata.obs.to_csv(obs_path)
        outputs["obs_csv"] = str(obs_path)

    if save_heatmap:
        gene_list = list(genes) if genes is not None else list(DEFAULT_PBMC_GENES)
        try:
            mat = score_gene_list(
                gene_list, db, species=species, tissue=tissue, data_source=data_source
            )
        except ValueError as exc:  # no matched genes for this scope/panel
            print(f"[warn] skipping gene-list heatmap: {exc}")
        else:
            heatmap_path = out / "gene_scores_heatmap.png"
            fig = plot_gene_scores(
                mat,
                title=title or f"{species} / {tissue} marker scores",
                save_path=heatmap_path,
            )

            plt.close(fig)
            outputs["heatmap_png"] = str(heatmap_path)
            gene_info = {"n_matched": mat.shape[0] - 1, "n_cell_types": mat.shape[1]}

    if save_adata:
        adata_path = out / "annotated.h5ad"
        if adata_path.resolve() == input_path.resolve():
            raise ValueError(
                "annotated.h5ad would overwrite the input file; "
                "choose a different --output-dir"
            )
        adata.write_h5ad(adata_path)
        outputs["annotated_h5ad"] = str(adata_path)

    return {
        "input_path": str(input_path),
        "output_dir": str(out),
        "species": species,
        "tissue": tissue,
        "method": method,
        "layer": layer,
        "n_cells": int(n_cells),
        "n_genes": int(adata.n_vars),
        "n_cell_types_scored": int(n_cell_types_scored),
        "frac_predicted": frac_predicted,
        "mean_confidence": mean_confidence,
        "max_confidence": max_confidence,
        "prediction_counts": prediction_counts,
        "broad_agreement": broad,
        "outputs": outputs,
        "gene_list": gene_info,
    }


def _print_report(r: dict) -> None:
    """Print a concise, aligned report of the pipeline results."""
    n_cells = r["n_cells"]
    print("pipeline report")
    print(f"  input        {r['input_path']} ({n_cells:,} x {r['n_genes']:,})")
    print(f"  scope        {r['species']} / {r['tissue']}  method={r['method']} layer={r['layer']}")
    print(f"  predicted    {round(n_cells * r['frac_predicted']):,} / {n_cells:,} "
          f"({r['frac_predicted'] * 100:.1f}%)")
    print(f"  confidence   mean {r['mean_confidence']:.2f}  max {r['max_confidence']:.2f}")
    print("  top cell types")
    for ct, n in list(r["prediction_counts"].items()):
        print(f"    {ct:<34} {n:>7,} ({n / n_cells * 100:.1f}%)")
    if r["broad_agreement"] is not None:
        print(f"  broad agreement  {r['broad_agreement'] * 100:.1f}%")
    print("  outputs")
    for key, value in r["outputs"].items():
        if value:
            print(f"    {key:<16} {value}")
    if r["gene_list"] is not None:
        print(f"  gene list    {r['gene_list']['n_matched']} matched / "
              f"{r['gene_list']['n_cell_types']} cell types")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        prog="pipeline", description="End-to-end cellmarkerannot annotation pipeline"
    )
    p.add_argument(
        "input", nargs="?", default=str(DEFAULT_INPUT),
        help=f".h5ad input (default: {DEFAULT_INPUT.name})",
    )
    p.add_argument("--dataset", default="all_cell_marker")
    p.add_argument("--species", default="Human")
    p.add_argument("--tissue", default="Blood")
    p.add_argument("--method", choices=SCORE_METHODS, default="overlap")
    p.add_argument("--layer", default=None, help="AnnData layer (default: adata.X)")
    p.add_argument("-o", "--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument(
        "--subset", type=int, default=None,
        help="annotate only the first N cells (fast smoke run)",
    )
    p.add_argument("--save-adata", action="store_true", help="also write annotated .h5ad")
    p.add_argument("--no-heatmap", action="store_true", help="skip the gene-list heatmap")
    p.add_argument("--no-obs", action="store_true", help="skip annotated_obs.csv")
    p.add_argument(
        "--genes", default=None,
        help="comma-separated marker genes for score_gene_list",
    )
    p.add_argument("--data-source", default="all", help="data_source for score_gene_list")
    p.add_argument("--title", default=None, help="heatmap title")
    p.add_argument(
        "--max-cell-types", type=int, default=None,
        help="max predicted cell-type labels (default: auto scree-knee)",
    )
    p.add_argument(
        "--min-cells", type=int, default=100,
        help="drop labels predicted for fewer cells (default: 100)",
    )
    p.add_argument(
        "--coverage", type=float, default=0.9,
        help="auto-mode coverage floor for label merging (default: 0.9)",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    genes = [g.strip() for g in args.genes.split(",") if g.strip()] if args.genes else None
    try:
        results = run_pipeline(
            args.input,
            dataset=args.dataset,
            species=args.species,
            tissue=args.tissue,
            method=args.method,
            layer=args.layer,
            output_dir=args.output_dir,
            subset=args.subset,
            save_obs=not args.no_obs,
            save_heatmap=not args.no_heatmap,
            save_adata=args.save_adata,
            genes=genes,
            data_source=args.data_source,
            title=args.title,
            max_cell_types=args.max_cell_types,
            min_cells=args.min_cells,
            coverage=args.coverage,
        )
    except Exception as exc:  # FileNotFoundError, ValueError, ...
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
