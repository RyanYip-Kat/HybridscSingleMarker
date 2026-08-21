# cellmarkerannot

Cell-type annotation toolkit built on the **CellMarker 3.0** marker database.

This package ships the CellMarker database as **bundled Parquet data** inside the
package, so it works out of the box after install — no external data directory
required. It loads the database, builds an in-memory species/tissue/cell-type
hierarchy, queries markers and cell types, and (planned) annotates single-cell
data by matching query markers against database markers.

> **Status**: `_config`, `io`, `resources`, `db`, `build_db`, the scoring layer
> (`annotation` / `_scoring`, three methods), the gene-list enrichment + heatmap
> (`score_gene_list` / `plot_gene_scores`), and the `cli` commands
> (`query` / `annotate` / `score`) are all implemented.

## Install

Requires Python >= 3.10. Uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync --inexact --extra test            # tests (add --extra plot for heatmaps)
```

`--inexact` matters: `uv sync` otherwise prunes undeclared packages
(e.g. scanpy / matplotlib) already present in `.venv`.

**pyarrow is a hard dependency** — it reads the bundled Parquet store.
**matplotlib** is optional (the `plot` extra) — needed only for `plot_gene_scores`.

## Built-in database

All five datasets are pre-converted from the source TSVs in `database/` into
compressed Parquet files shipped inside the package:

| dataset | rows | bundled size |
|---|---:|---:|
| `all_cell_marker` | 2,537,570 | 35.6 MB |
| `human_cell_marker` | 1,779,944 | 18.8 MB |
| `mouse_cell_marker` | 757,626 | 9.5 MB |
| `single_cell_marker` | 418,933 | 5.2 MB |
| `method_cell_marker` | 2,318,927 | 32.1 MB |

The `.parquet` files (≈100 MB total) live in `src/cellmarkerannot/data/` and are
included in the wheel. A `manifest.json` records row counts and build metadata.

To rebuild after the source data changes:

```bash
uv run python -m cellmarkerannot.build_db --overwrite
```

(and call `cellmarkerannot.db.clear_cache()` in any live session).

## Data source resolution

A dataset is resolved by priority:

1. explicit `path` passed to `CellMarkerDB` / `read_dataset`
2. `CELLMARKERANNOT_DB_DIR` (a directory containing `<dataset>.parquet` or `.txt`)
3. bundled package data (`src/cellmarkerannot/data/<dataset>.parquet`)
4. project-root `database/<dataset>.txt`

If nothing resolves, `FileNotFoundError` is raised listing the searched paths.

## Cell-type annotation

Three vectorized scoring methods compute, per cell, a score against every cell
type in a `(species, tissue)` scope:

| method | score | backend |
|---|---|---|
| `overlap` | # of expressed markers matched | scipy.sparse matmul |
| `weighted` | specificity-weighted mean expression (`evidence × idf`) | scipy.sparse matmul |
| `ssgsea` | single-sample gene-set enrichment (Barbie-style walk) | numba `@njit(parallel=True)` |

Defaults (tunable): markers restricted to experimentally validated sources
(`Experiment` + `Single-cell sequencing`, see `DEFAULT_MARKER_SOURCES` — broad
catch-all cell types otherwise carry ~2000 overlapping markers that dominate the
scores); expression threshold `> 1.0` (log-normalized data); the `weighted`
score is size-normalized so cell types are comparable regardless of marker count.
Per-cell ``confidence`` is the **softmax probability** of the predicted cell
type over the raw scores — near 1.0 for a clear winner, low when the top cell
types are close, and 0 for cells with no expressed markers.

```python
import anndata as ad
from cellmarkerannot import CellMarkerDB, annotate_cells, score_cells

adata = ad.read_h5ad("testdata/pbmc5k_querydata.h5ad")  # or your data
db = CellMarkerDB(dataset="all_cell_marker")

# Raw per-cell scores, cells × cell types (58k × ~600 for Human/Blood):
scores = score_cells(adata, db, method="weighted", species="Human", tissue="Blood")

# Predictions + 0-1 confidence (sets adata.obs by default):
annotate_cells(adata, db, method="weighted", species="Human", tissue="Blood")

# Score a bare marker list (cluster-level annotation):
score_cell_types(["CD3D", "CD3E", "IL7R"], db, species="Human", tissue="Blood")

# Gene list → support-evidence matrix + enrichment Score + heatmap:
from cellmarkerannot import plot_gene_scores, score_gene_list

mat = score_gene_list(
    ["CD3D", "CD3E", "IL7R", "CD19"], db,
    species="Human", tissue="Blood", data_source="all",
)  # genes × cell types of integer support evidence, final row = "Score"
fig = plot_gene_scores(mat, title="PBMC markers", save_path="markers.png")
```

`score_gene_list`'s per-cell-type **Score** is
`-log10(P_hypergeom) × log1p(Σ matched-gene evidence)`, i.e. a hypergeometric
enrichment p-value (P(X ≥ k) for drawing the query genes from the in-scope marker
universe) weighted by the support evidence. `data_source` selects the marker
source(s): `all`, `experiment`, `method`, `single_cell`, `review`, `company` (or
a list).

> **Prediction quality**: the methods are verified against hand-computed
> references. Against the *aggregated* CellMarker scope, predictions are
> biologically plausible but not exact — the DB's granular cell-type names do not
> map one-to-one onto common ground-truth labels, and marker sets overlap across
> related cell types. The scope already matches `tissue_type` precisely (not
> `tissue_class`); for further gains, prefer a curated cell-type set.

> For a complete scripted end-to-end example, see
> [Pipeline / end-to-end example](#pipeline--end-to-end-example).

## Pipeline / end-to-end example

`scripts/pipeline.py` ties the library together end-to-end: load the bundled
database, load an AnnData file (backed), annotate every cell (`weighted`,
Human/Blood by default), run the gene-list enrichment + heatmap on a curated
PBMC marker panel, print a concise report, and write outputs.

Run it from the repo root against the included test file:

```bash
uv run python scripts/pipeline.py               # full 58,677 x 33,421 (~15 s, ~1 GB RAM)
uv run python scripts/pipeline.py --subset 200  # fast smoke run (first 200 cells)
```

| arg | default | meaning |
|---|---|---|
| `input` | `testdata/pbmc5k_querydata.h5ad` | `.h5ad` path |
| `--dataset` | `all_cell_marker` | bundled dataset |
| `--species` / `--tissue` | `Human` / `Blood` | scoring scope |
| `--method` | `weighted` | `overlap` / `weighted` / `ssgsea` |
| `--layer` | (none) | use `adata.X`, or `counts` / `data` |
| `-o, --output-dir` | `pipeline_output/` | outputs directory |
| `--subset N` | (all) | annotate only the first N cells |
| `--save-adata` | off | also write `annotated.h5ad` |
| `--no-heatmap` / `--no-obs` | off | disable the gene-list heatmap / obs CSV |
| `--genes A,B,...` | curated PBMC panel | marker list for `score_gene_list` |
| `--data-source` | `all` | source filter for the gene-list step |
| `--max-cell-types` | (auto) | max predicted labels; omit = auto knee, `0` = keep all |
| `--min-cells` | 100 | drop labels predicted for fewer cells |

Example report (abridged):

```
pipeline report
  input        testdata/pbmc5k_querydata.h5ad (58,677 x 33,421)
  scope        Human / Blood  method=weighted layer=None
  predicted    55,321 / 58,677 (94.3%)
  confidence   mean 0.42  max 0.98
  top cell types   T cell 12,345 (22.3%)   ...
  broad agreement  47.2%
```

Outputs written to the output directory:

- `annotated_obs.csv` — the annotated `obs` (original columns + `celltype_predicted`, `confidence`).
- `gene_scores_heatmap.png` — evidence heatmap + Score bars for the marker panel.
- `annotated.h5ad` (only with `--save-adata`) — the annotated AnnData (the subset, if `--subset` was used).

Programmatic use:

```python
from pipeline import run_pipeline  # from the scripts/ directory

results = run_pipeline("testdata/pbmc5k_querydata.h5ad", subset=200, output_dir="out")
print(results["prediction_counts"])
```

The real-data smoke is also wired into the test suite as an opt-in slow test:
`uv run pytest tests/test_pipeline.py -m slow`.

## Quickstart

### Python API

```python
from cellmarkerannot import CellMarkerDB, clear_cache

db = CellMarkerDB(dataset="all_cell_marker")        # lazy; loads from bundled parquet
db.frame.shape                                      # (2537570, 22)

# Filter markers by species / tissue / cell type:
db.markers_for_cell_type("T cell", species="Human", tissue_class="Blood")[:10]

# Which cell types cite a marker?
db.cell_types_for_marker("CD3D", species="Human")[:10]

# Discovery:
db.available_species()          # ['Human', 'Mouse']
db.available_tissues("Blood")
db.available_cell_names(tissue_class="Blood")
db.available_markers(cell_name="T cell")

# Arbitrary column filters (case-insensitive by default):
rows = db.query(species="human", marker_source="Method", cell_name="T cell")
rows.head()

# Frames are cached module-wide; drop the cache after rebuilding data:
clear_cache()
```

### CLI

```bash
uv run cellmarkerannot --version
uv run cellmarkerannot query --dataset mouse_cell_marker --species Mouse --marker Cd3d
uv run cellmarkerannot score --species Human --tissue Blood --genes CD3D,CD3E,CD19
uv run cellmarkerannot annotate --input data.h5ad --species Human --tissue Blood --save-adata
```

- `query` — filter the database rows and optionally save to TSV.
- `score` — gene-list evidence matrix + enrichment Score row, with optional heatmap.
- `annotate` — per-cell annotation (auto label-merging by default) writing
  `annotated_obs.csv` (and `.h5ad` with `--save-adata`).

## Data reference

- `README.txt` — the five CellMarker 3.0 datasets (`all`, `human`, `mouse`,
  `single_cell`, `method`).
- `database/DATA_STRUCTURE.md` — full 22-column schema, data-quality notes
  (`.0`-suffixed numeric IDs, empty columns, value sets), and inter-file
  relationships.

## Project layout

```
src/cellmarkerannot/
  _config.py        # schema constants, dataset files, path/data-dir resolution
  io.py             # TSV reading + id / whitespace cleaning (implemented)
  resources.py      # source resolution + parquet/TSV load dispatch (implemented)
  db.py             # CellMarkerDB: lazy load, hierarchy index, query, caching
  build_db.py       # TSV -> bundled Parquet conversion (python -m ...)
  _scoring.py       # marker matrices, overlap/weighted/ssgsea cores, numba kernel
  annotation.py     # score_cell_types / score_cells / annotate_cells / score_gene_list
  plotting.py       # plot_gene_scores heatmap (lazy matplotlib)
  cli.py            # argparse CLI: query / annotate / score commands
  data/             # bundled Parquet datasets + manifest.json (shipped in wheel)
scripts/
  pipeline.py       # end-to-end example pipeline (uv run python scripts/pipeline.py)
tests/              # pytest suite
```

## Development

```bash
uv run pytest                      # run the test suite
uv run cellmarkerannot --help
```
