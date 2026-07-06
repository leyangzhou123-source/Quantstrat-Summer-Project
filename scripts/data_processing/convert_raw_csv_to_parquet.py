from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import time

import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq


DEFAULT_DIRECTORIES = [
    Path("data/raw/02_stock_characteristics"),
    Path("data/raw/crsp"),
]


def read_header(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle))


def stream_csv_to_parquet(
    csv_path: Path,
    temp_path: Path,
    compression: str | None,
    all_string_columns: bool = False,
) -> int:
    read_options = pv.ReadOptions(block_size=64 * 1024 * 1024)
    convert_options = pv.ConvertOptions(strings_can_be_null=True)
    if all_string_columns:
        column_names = read_header(csv_path)
        read_options = pv.ReadOptions(
            block_size=64 * 1024 * 1024,
            column_names=column_names,
            skip_rows=1,
        )
        convert_options = pv.ConvertOptions(
            column_types={name: pa.string() for name in column_names},
            strings_can_be_null=True,
        )

    rows = 0
    writer: pq.ParquetWriter | None = None
    try:
        reader = pv.open_csv(
            csv_path,
            read_options=read_options,
            convert_options=convert_options,
        )
        for batch in reader:
            if writer is None:
                writer = pq.ParquetWriter(
                    temp_path,
                    batch.schema,
                    compression=compression,
                    use_dictionary=True,
                    write_statistics=True,
                )
            writer.write_batch(batch)
            rows += batch.num_rows
    finally:
        if writer is not None:
            writer.close()
    return rows


def convert_csv_to_parquet(csv_path: Path, compression: str | None = "zstd") -> dict[str, object]:
    parquet_path = csv_path.with_suffix(".parquet")
    temp_path = csv_path.with_suffix(".parquet.tmp")

    if parquet_path.exists():
        return {
            "source": str(csv_path),
            "target": str(parquet_path),
            "status": "skipped_existing_parquet",
        }

    if temp_path.exists():
        temp_path.unlink()

    start = time.time()
    fallback_all_strings = False

    try:
        rows = stream_csv_to_parquet(csv_path, temp_path, compression)
    except pa.ArrowInvalid:
        temp_path.unlink(missing_ok=True)
        fallback_all_strings = True
        rows = stream_csv_to_parquet(
            csv_path,
            temp_path,
            compression,
            all_string_columns=True,
        )

    metadata = pq.read_metadata(temp_path)
    if metadata.num_rows != rows:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Row-count mismatch for {csv_path}: streamed={rows}, parquet={metadata.num_rows}"
        )

    temp_path.replace(parquet_path)
    csv_size = csv_path.stat().st_size
    parquet_size = parquet_path.stat().st_size
    csv_path.unlink()

    return {
        "source": str(csv_path),
        "target": str(parquet_path),
        "status": "converted",
        "rows": rows,
        "csv_bytes": csv_size,
        "parquet_bytes": parquet_size,
        "seconds": round(time.time() - start, 3),
        "fallback_all_strings": fallback_all_strings,
    }


def discover_csv_files(directories: list[Path]) -> list[Path]:
    files: list[Path] = []
    for directory in directories:
        files.extend(path for path in directory.glob("*.csv") if path.is_file())
    return sorted(files, key=lambda path: path.stat().st_size)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert raw CSV files to Parquet and remove CSVs after successful conversion."
    )
    parser.add_argument(
        "directories",
        nargs="*",
        type=Path,
        default=DEFAULT_DIRECTORIES,
        help="Directories containing CSV files to convert.",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        choices=["zstd", "snappy", "gzip", "brotli", "none"],
        help="Parquet compression codec.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("data/raw/parquet_conversion_log.jsonl"),
        help="JSONL conversion log path.",
    )
    args = parser.parse_args()

    compression = None if args.compression == "none" else args.compression
    csv_files = discover_csv_files(args.directories)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(csv_files)} CSV files to convert.")
    print(f"Initial free space: {shutil.disk_usage(Path.cwd()).free:,} bytes")

    converted = 0
    with args.log.open("a", encoding="utf-8") as log:
        for index, csv_path in enumerate(csv_files, start=1):
            try:
                print(f"[{index}/{len(csv_files)}] Converting {csv_path}")
                result = convert_csv_to_parquet(csv_path, compression=compression)
                print(json.dumps(result, sort_keys=True))
                log.write(json.dumps(result, sort_keys=True) + "\n")
                log.flush()
                if result["status"] == "converted":
                    converted += 1
            except Exception as exc:
                error = {
                    "source": str(csv_path),
                    "status": "failed",
                    "error": repr(exc),
                }
                print(json.dumps(error, sort_keys=True), file=sys.stderr)
                log.write(json.dumps(error, sort_keys=True) + "\n")
                log.flush()
                return 1

    print(f"Converted {converted} CSV files.")
    print(f"Final free space: {shutil.disk_usage(Path.cwd()).free:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
