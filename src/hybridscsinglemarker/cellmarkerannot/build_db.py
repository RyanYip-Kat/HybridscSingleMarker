"""Convert the CellMarker source TSVs into the bundled Parquet database.

Run as::

    uv run python -m cellmarkerannot.build_db [--overwrite]

Reads each ``<dataset>.txt`` from ``src_dir`` (default: the project-root
``database/``), applies the standard cleaning from :mod:`cellmarkerannot.io`,
and writes a ``<dataset>.parquet`` plus a ``manifest.json`` into ``out_dir``
(default: the package data dir ``src/cellmarkerannot/data/``, which is shipped
inside the wheel).

After rebuilding in a live session, call ``cellmarkerannot.db.clear_cache()``
so already-loaded frames are re-read from the new files.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from . import _config as _cfg
from .io import read_database


def _sha256(path: Path) -> str:
    """Compute the SHA-256 digest of a file in chunks (memory-friendly)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pyarrow_version() -> str:
    try:
        import pyarrow

        return pyarrow.__version__
    except ImportError:  # pragma: no cover - pyarrow is a hard dependency
        return "not-installed"


def build_dataset(
    src: Path,
    dst: Path,
    *,
    columns: Iterable[str] = _cfg.COLUMNS,
    compression: str = "zstd",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert one TSV into a Parquet file. Returns a manifest record."""
    if dst.exists() and not overwrite:
        print(f"[skip] {dst.name} exists (use --overwrite)")
        return {"dataset": dst.stem, "skipped": True, "path": str(dst)}

    print(f"[read]  {src} ({src.stat().st_size / 1e6:.1f} MB)")
    t0 = time.perf_counter()
    df = read_database(src, columns=list(columns))
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, compression=compression, index=False)
    rows = len(df)
    del df
    gc.collect()  # free memory before loading the next dataset
    record = {
        "dataset": dst.stem,
        "rows": rows,
        "path": str(dst),
        "bytes": dst.stat().st_size,
        "sha256": _sha256(dst),
        "read_seconds": round(time.perf_counter() - t0, 2),
    }
    print(f"[write] {dst} rows={rows} {record['bytes'] / 1e6:.1f} MB")
    return record


def build_all(
    src_dir: Path = _cfg.DATABASE_DIR,
    out_dir: Path = _cfg.DATA_DIR,
    *,
    datasets: Iterable[str] | None = None,
    overwrite: bool = False,
    compression: str = "zstd",
    write_manifest: bool = True,
) -> dict[str, dict[str, Any]]:
    """Convert the source TSVs under ``src_dir`` into Parquet under ``out_dir``.

    ``datasets`` limits the conversion to a subset of :data:`DATASET_FILES`.
    Returns a ``{dataset: manifest_record}`` mapping.
    """
    names = list(datasets) if datasets is not None else list(_cfg.DATASET_FILES)
    results: dict[str, dict[str, Any]] = {}
    for name in names:
        src = src_dir / _cfg.DATASET_FILES[name]
        if not src.is_file():
            raise FileNotFoundError(f"Missing source TSV: {src}")
        results[name] = build_dataset(
            src,
            out_dir / _cfg.PARQUET_FILES[name],
            overwrite=overwrite,
            compression=compression,
        )
    if write_manifest:
        _write_manifest(out_dir, results)
    return results


def _write_manifest(out_dir: Path, results: dict[str, dict[str, Any]]) -> None:
    """Write ``manifest.json`` describing the built datasets."""
    manifest = {
        "schema_version": _cfg.SCHEMA_VERSION,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pyarrow": _pyarrow_version(),
        "pandas": pd.__version__,
        "datasets": results,
    }
    (out_dir / _cfg.MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m cellmarkerannot.build_db``."""
    parser = argparse.ArgumentParser(
        prog="python -m cellmarkerannot.build_db",
        description="Convert CellMarker TSV files into the bundled Parquet database.",
    )
    parser.add_argument(
        "--datasets",
        help="comma-separated subset of " + ",".join(_cfg.DATASET_FILES),
    )
    parser.add_argument("--src-dir", default=str(_cfg.DATABASE_DIR))
    parser.add_argument(
        "--out-dir",
        default=str(_cfg.DATA_DIR),
        help=f"default: bundled package data ({_cfg.DATA_DIR})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing parquet files (default: skip existing)",
    )
    parser.add_argument("--no-manifest", action="store_true")
    parser.add_argument("--compression", default="zstd")
    args = parser.parse_args(argv)

    datasets = args.datasets.split(",") if args.datasets else None
    print(f"Writing bundled Parquet to {args.out_dir}")
    print("NOTE: after rebuilding, call cellmarkerannot.db.clear_cache() in live sessions.")
    build_all(
        Path(args.src_dir),
        Path(args.out_dir),
        datasets=datasets,
        overwrite=args.overwrite,
        compression=args.compression,
        write_manifest=not args.no_manifest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
