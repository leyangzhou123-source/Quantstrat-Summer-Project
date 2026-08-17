from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _rank_expr(column: str, date_column: str) -> pl.Expr:
    rank = pl.col(column).rank(method="average").over(date_column)
    count = pl.col(column).count().over(date_column)
    scaled = -1.0 + 2.0 * ((rank - 1.0) / (count - 1.0))
    return pl.when(count > 1).then(scaled).otherwise(None).cast(pl.Float32).alias(column)


def rebuild_rank_fixed_panel(
    input_path: Path,
    manifest_path: Path,
    output_path: Path,
    output_manifest_path: Path,
    reuse_existing_output: bool = False,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    stock_characteristics = list(manifest["stock_characteristics"])
    date_column = (
        "yyyymm"
        if "yyyymm" in pl.scan_parquet(input_path).collect_schema().names()
        else manifest["date"]
    )

    if output_path.exists() and not reuse_existing_output:
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output_path}")
    if output_manifest_path.exists():
        raise FileExistsError(
            f"Output manifest already exists; refusing to overwrite: {output_manifest_path}"
        )

    scan = pl.scan_parquet(input_path)
    missing = sorted(set(stock_characteristics).difference(scan.collect_schema().names()))
    if missing:
        raise ValueError(f"Manifest stock characteristics missing from panel: {missing}")

    if not reuse_existing_output:
        fixed = scan.with_columns(
            [_rank_expr(column, date_column) for column in stock_characteristics]
        )
        fixed.sink_parquet(output_path, compression="zstd", statistics=True)

    audit = (
        pl.scan_parquet(output_path)
        .select(
            [
                pl.len().alias("rows"),
                pl.col("permno").n_unique().alias("permnos"),
                pl.col(date_column).n_unique().alias("months"),
                pl.col(date_column).min().alias("yyyymm_start"),
                pl.col(date_column).max().alias("yyyymm_end"),
                pl.sum_horizontal(
                    [pl.col(c).is_null().cast(pl.Int64) for c in stock_characteristics]
                ).alias("characteristic_null_cells"),
                pl.sum_horizontal(
                    [(~pl.col(c).is_between(-1, 1)).cast(pl.Int64) for c in stock_characteristics]
                ).alias("characteristic_outside_rank_cells"),
                pl.min_horizontal([pl.col(c).min() for c in stock_characteristics]).alias(
                    "ranked_characteristic_min"
                ),
                pl.max_horizontal([pl.col(c).max() for c in stock_characteristics]).alias(
                    "ranked_characteristic_max"
                ),
            ]
        )
        .collect()
        .to_dicts()[0]
    )

    fixed_manifest = dict(manifest)
    fixed_manifest["panel"] = _relative(output_path)
    fixed_manifest["rank_fix_source_panel"] = _relative(input_path)
    fixed_manifest["rank_formula"] = "2 * (rank - 1) / (N - 1) - 1 within each yyyymm"
    fixed_manifest["processing"] = list(manifest.get("processing", [])) + [
        "Re-ranked stock characteristics within each yyyymm using the full endpoint [-1, 1] formula.",
        "Existing source parquet was left unchanged.",
    ]
    fixed_manifest["completeness"] = {
        **dict(manifest.get("completeness", {})),
        **{k: int(v) if isinstance(v, (np.integer, int)) else v for k, v in audit.items()},
    }
    output_manifest_path.write_text(json.dumps(fixed_manifest, indent=2, default=str) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a non-overwriting model panel copy with corrected characteristic ranks."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--reuse-existing-output",
        action="store_true",
        help="Audit an already-written output parquet and create only its manifest.",
    )
    args = parser.parse_args()

    audit = rebuild_rank_fixed_panel(
        input_path=args.input,
        manifest_path=args.manifest,
        output_path=args.output,
        output_manifest_path=args.output_manifest,
        reuse_existing_output=args.reuse_existing_output,
    )
    print(json.dumps(audit, indent=2, default=str))


if __name__ == "__main__":
    main()
