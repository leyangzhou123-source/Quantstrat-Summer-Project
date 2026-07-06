from __future__ import annotations

from pathlib import Path
import json
import time

import polars as pl


SORT_SPECS: dict[Path, list[str]] = {
    Path("data/raw/01_stock_returns.parquet"): ["month", "permno"],
    Path("data/raw/03_industry_dummies.parquet"): ["month", "permno"],
    Path("data/raw/04_fama_french_factors.parquet"): ["month"],
    Path("data/raw/05_welch_goyal_macro.parquet"): ["month"],
}


def is_sorted(path: Path, sort_cols: list[str]) -> bool:
    df = pl.scan_parquet(path).select(sort_cols).collect()
    return df.equals(df.sort(sort_cols))


def sort_file(path: Path, sort_cols: list[str]) -> dict[str, object]:
    start = time.time()
    temp_path = path.with_suffix(path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()

    original_rows = pl.scan_parquet(path).select(pl.len()).collect().item()
    (
        pl.scan_parquet(path)
        .sort(sort_cols)
        .sink_parquet(temp_path, compression="zstd")
    )
    sorted_rows = pl.scan_parquet(temp_path).select(pl.len()).collect().item()
    if original_rows != sorted_rows:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Row-count mismatch for {path}: {original_rows} vs {sorted_rows}")
    if not is_sorted(temp_path, sort_cols):
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Sorted verification failed for {path}")

    temp_path.replace(path)
    return {
        "path": str(path),
        "sort_cols": sort_cols,
        "rows": original_rows,
        "seconds": round(time.time() - start, 3),
        "status": "sorted",
    }


def main() -> int:
    log_path = Path("reports/raw_parquet_sort_log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    specs = dict(SORT_SPECS)
    for path in sorted(Path("data/raw/02_stock_characteristics").glob("*.parquet")):
        specs[path] = ["yyyymm", "permno"]

    with log_path.open("w") as log:
        for path, sort_cols in specs.items():
            if not path.exists():
                result = {
                    "path": str(path),
                    "sort_cols": sort_cols,
                    "status": "missing",
                }
            else:
                result = sort_file(path, sort_cols)
            print(json.dumps(result, sort_keys=True))
            log.write(json.dumps(result, sort_keys=True) + "\n")
            log.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
