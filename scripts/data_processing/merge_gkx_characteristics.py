from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
EXTERNAL_DIR = ROOT / "data" / "external"

DEFAULT_GKX_ZIP = EXTERNAL_DIR / "gkx_datashare.zip"
DEFAULT_CURRENT = PROCESSED_DIR / "stock_characteristics.parquet"
DEFAULT_BACKUP = PROCESSED_DIR / "stock_characteristics_before_gkx_merge.parquet"
DEFAULT_STAGED_GKX = PROCESSED_DIR / "gkx_stock_characteristics_1957_2016.parquet"
DEFAULT_OUTPUT = PROCESSED_DIR / "stock_characteristics.parquet"
DEFAULT_REPORT = PROCESSED_DIR / "stock_characteristics_gkx_merge_report.json"


KEY_COLUMNS = ["permno", "yyyymm"]


def parquet_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema.names


def read_gkx_columns(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as archive, archive.open("datashare.csv") as handle:
        return pd.read_csv(handle, nrows=0).columns.tolist()


def stage_gkx_zip_to_parquet(
    zip_path: Path,
    output_path: Path,
    start_yyyymm: int,
    end_yyyymm: int,
    chunksize: int,
) -> dict:
    if output_path.exists():
        output_path.unlink()

    writer: pq.ParquetWriter | None = None
    rows = 0
    min_yyyymm: int | None = None
    max_yyyymm: int | None = None
    duplicate_keys = 0

    with zipfile.ZipFile(zip_path) as archive, archive.open("datashare.csv") as handle:
        for chunk in pd.read_csv(handle, chunksize=chunksize):
            chunk["yyyymm"] = (pd.to_numeric(chunk["DATE"], errors="coerce") // 100).astype("Int64")
            chunk = chunk[
                (chunk["yyyymm"] >= start_yyyymm) & (chunk["yyyymm"] <= end_yyyymm)
            ].copy()
            if chunk.empty:
                continue
            duplicate_keys += int(chunk.duplicated(KEY_COLUMNS).sum())
            chunk["yyyymm"] = chunk["yyyymm"].astype("int64")
            chunk["_gkx_present"] = True
            chunk = chunk.drop(columns=["DATE"])
            cols = KEY_COLUMNS + [c for c in chunk.columns if c not in KEY_COLUMNS]
            chunk = chunk[cols]
            rows += len(chunk)
            min_yyyymm = (
                int(chunk["yyyymm"].min())
                if min_yyyymm is None
                else min(min_yyyymm, int(chunk["yyyymm"].min()))
            )
            max_yyyymm = (
                int(chunk["yyyymm"].max())
                if max_yyyymm is None
                else max(max_yyyymm, int(chunk["yyyymm"].max()))
            )

            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)

    if writer is not None:
        writer.close()
    if rows == 0:
        raise RuntimeError("No GKX rows were staged. Check the date filter and source zip.")

    return {
        "path": str(output_path.relative_to(ROOT)),
        "rows": rows,
        "min_yyyymm": min_yyyymm,
        "max_yyyymm": max_yyyymm,
        "duplicate_keys_within_chunks": duplicate_keys,
    }


def merge_characteristics(
    current_path: Path,
    gkx_path: Path,
    output_path: Path,
    backup_path: Path,
    report_path: Path,
) -> dict:
    current_cols = parquet_columns(current_path)
    gkx_cols = parquet_columns(gkx_path)
    gkx_feature_cols = [c for c in gkx_cols if c not in {*KEY_COLUMNS, "sic2", "_gkx_present"}]
    current_feature_cols = [c for c in current_cols if c not in KEY_COLUMNS]
    overlapping_features = [c for c in current_feature_cols if c in gkx_feature_cols]
    added_features = [c for c in gkx_feature_cols if c not in current_feature_cols]
    current_only_features = [c for c in current_feature_cols if c not in gkx_feature_cols]

    if not backup_path.exists():
        shutil.copy2(current_path, backup_path)

    current = pl.scan_parquet(current_path)
    gkx = pl.scan_parquet(gkx_path)

    joined = current.join(gkx, on=KEY_COLUMNS, how="left", suffix="__gkx")
    present = pl.col("_gkx_present").fill_null(False)
    expressions = [pl.col("permno"), pl.col("yyyymm")]
    for col in gkx_feature_cols:
        if col in current_feature_cols:
            expressions.append(
                pl.when(present).then(pl.col(f"{col}__gkx")).otherwise(pl.col(col)).alias(col)
            )
        else:
            expressions.append(pl.col(col).alias(col))
    for col in current_only_features:
        expressions.append(pl.col(col))

    merged = joined.select(expressions).sort(KEY_COLUMNS)
    tmp_output = output_path.with_suffix(".tmp.parquet")
    if tmp_output.exists():
        tmp_output.unlink()
    merged.sink_parquet(tmp_output, compression="zstd")
    tmp_output.replace(output_path)

    current_keys = pl.scan_parquet(backup_path).select(KEY_COLUMNS)
    gkx_keys = pl.scan_parquet(gkx_path).select(KEY_COLUMNS + ["_gkx_present"])
    key_check = (
        current_keys.join(gkx_keys, on=KEY_COLUMNS, how="left")
        .select(
            [
                pl.len().alias("current_rows"),
                pl.col("_gkx_present").fill_null(False).sum().alias("rows_replaced_from_gkx"),
                (~pl.col("_gkx_present").fill_null(False))
                .sum()
                .alias("rows_kept_from_current_no_gkx_key"),
            ]
        )
        .collect()
    )
    extra_gkx = (
        gkx_keys.join(current_keys, on=KEY_COLUMNS, how="anti").select(pl.len()).collect().item()
    )
    output_info = (
        pl.scan_parquet(output_path)
        .select(
            [
                pl.len().alias("output_rows"),
                pl.col("yyyymm").min().alias("min_yyyymm"),
                pl.col("yyyymm").max().alias("max_yyyymm"),
                pl.col("permno").n_unique().alias("permnos"),
            ]
        )
        .collect()
    )

    report = {
        "input_current": str(current_path.relative_to(ROOT)),
        "backup_current": str(backup_path.relative_to(ROOT)),
        "staged_gkx": str(gkx_path.relative_to(ROOT)),
        "output": str(output_path.relative_to(ROOT)),
        "current_feature_count": len(current_feature_cols),
        "gkx_feature_count": len(gkx_feature_cols),
        "output_feature_count": len(gkx_feature_cols) + len(current_only_features),
        "overlapping_features_replaced_by_gkx": overlapping_features,
        "added_gkx_features": added_features,
        "current_only_features_retained": current_only_features,
        "current_rows": int(key_check["current_rows"][0]),
        "rows_replaced_from_gkx": int(key_check["rows_replaced_from_gkx"][0]),
        "rows_kept_from_current_no_gkx_key": int(key_check["rows_kept_from_current_no_gkx_key"][0]),
        "gkx_keys_outside_current": int(extra_gkx),
        "output_rows": int(output_info["output_rows"][0]),
        "output_min_yyyymm": int(output_info["min_yyyymm"][0]),
        "output_max_yyyymm": int(output_info["max_yyyymm"][0]),
        "output_permnos": int(output_info["permnos"][0]),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge Dacheng Xiu/GKX firm characteristics into current stock characteristics."
    )
    parser.add_argument("--gkx-zip", type=Path, default=DEFAULT_GKX_ZIP)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--backup", type=Path, default=DEFAULT_BACKUP)
    parser.add_argument("--staged-gkx", type=Path, default=DEFAULT_STAGED_GKX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--start-yyyymm", type=int, default=195703)
    parser.add_argument("--end-yyyymm", type=int, default=201612)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()

    staged = stage_gkx_zip_to_parquet(
        zip_path=args.gkx_zip,
        output_path=args.staged_gkx,
        start_yyyymm=args.start_yyyymm,
        end_yyyymm=args.end_yyyymm,
        chunksize=args.chunksize,
    )
    report = merge_characteristics(
        current_path=args.current,
        gkx_path=args.staged_gkx,
        output_path=args.output,
        backup_path=args.backup,
        report_path=args.report,
    )
    print(json.dumps({"staged": staged, "merge": report}, indent=2))


if __name__ == "__main__":
    main()
