"""cellmarkerannot: cell-type annotation toolkit on the CellMarker 3.0 database."""

from .__version__ import __version__
from . import resources
from ._config import (
    COLUMNS,
    COLUMN_SPECS,
    DATA_DIR,
    DATASET_FILES,
    DATABASE_DIR,
    MANIFEST_FILE,
    PARQUET_FILES,
    SCHEMA_VERSION,
    get_database_path,
)
from .io import read_database
from .resources import iter_sources, read_dataset, read_parquet, resolve_source
from .db import CellMarkerDB, clear_cache, get_database
from .annotation import (
    SCORE_METHODS,
    annotate_cells,
    score_cell_types,
    score_cells,
    score_gene_list,
)
from .plotting import plot_gene_scores

__all__ = [
    "__version__",
    "COLUMNS",
    "COLUMN_SPECS",
    "DATA_DIR",
    "DATASET_FILES",
    "DATABASE_DIR",
    "MANIFEST_FILE",
    "PARQUET_FILES",
    "SCHEMA_VERSION",
    "get_database_path",
    "read_database",
    "iter_sources",
    "read_dataset",
    "read_parquet",
    "resolve_source",
    "CellMarkerDB",
    "clear_cache",
    "get_database",
    "SCORE_METHODS",
    "annotate_cells",
    "score_cell_types",
    "score_cells",
    "score_gene_list",
    "plot_gene_scores",
    "resources",
]
