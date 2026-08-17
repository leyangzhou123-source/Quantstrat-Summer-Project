from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

DEFAULT_OUTPUT = PROCESSED_DIR / "model_penal.parquet"
DEFAULT_MANIFEST = PROCESSED_DIR / "model_penal_manifest.json"
DEFAULT_RETURNS_TARGET = PROCESSED_DIR / "model_penal_returns_target.parquet"

KEY_COLUMNS = ["permno", "yyyymm"]
RETURN_KEEP_COLUMNS = [
    "permno",
    "permco",
    "date",
    "month",
    "yyyymm",
    "ret",
    "retx",
    "prc",
    "shrout",
    "vol",
    "ticker",
    "comnam",
    "shrcd",
    "exchcd",
    "siccd",
    "me",
]
MACRO_RENAME = {
    "dp": "macro_dp",
    "ep": "macro_ep",
    "bm": "macro_bm",
    "ntis": "macro_ntis",
    "tbl": "macro_tbl",
    "tms": "macro_tms",
    "dfy": "macro_dfy",
    "svar": "macro_svar",
}
MACRO_COLUMNS = list(MACRO_RENAME.values())


def parquet_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema.names


def characteristic_columns(path: Path) -> list[str]:
    return [c for c in parquet_columns(path) if c not in KEY_COLUMNS]


def industry_dummy_columns(path: Path) -> list[str]:
    return [c for c in parquet_columns(path) if c.startswith("sic2_")]


def add_yyyymm_from_month(frame: pl.LazyFrame) -> pl.LazyFrame:
    return frame.with_columns(
        (pl.col("month").dt.year() * 100 + pl.col("month").dt.month())
        .cast(pl.Int64)
        .alias("yyyymm")
    )


def expected_next_yyyymm() -> pl.Expr:
    year = pl.col("month").dt.year()
    month = pl.col("month").dt.month()
    return (
        pl.when(month == 12)
        .then((year + 1) * 100 + 1)
        .otherwise(year * 100 + month + 1)
        .cast(pl.Int64)
    )


def ranked_characteristics(path: Path, char_cols: list[str]) -> pl.LazyFrame:
    base = pl.scan_parquet(path).select(KEY_COLUMNS + char_cols)
    filled = base.with_columns(
        [
            pl.col(col).fill_null(pl.col(col).median().over("yyyymm")).fill_null(0.0).alias(col)
            for col in char_cols
        ]
    )
    n_by_month = pl.len().over("yyyymm")
    return filled.with_columns(
        [
            (
                -1.0
                + 2.0
                * ((pl.col(col).rank(method="average").over("yyyymm") - 1.0) / (n_by_month - 1.0))
            ).alias(col)
            for col in char_cols
        ]
    )


def returns_with_target(returns_path: Path, macro_path: Path) -> pl.LazyFrame:
    returns = add_yyyymm_from_month(pl.scan_parquet(returns_path)).select(RETURN_KEEP_COLUMNS)
    macro = (
        pl.scan_parquet(macro_path)
        .rename(MACRO_RENAME)
        .select(["yyyymm", "rf_welch_goyal"] + MACRO_COLUMNS)
    )
    panel = returns.join(macro, on="yyyymm", how="left").with_columns(
        [
            pl.coalesce([pl.col("me"), pl.col("prc").abs() * pl.col("shrout")]).alias("me"),
            (pl.col("ret") - pl.col("rf_welch_goyal")).alias("ret_excess"),
        ]
    )
    return (
        panel.sort(["permno", "yyyymm"])
        .with_columns(
            [
                pl.col("me").forward_fill().backward_fill().over("permno").alias("me"),
                pl.col("ret_excess").shift(-1).over("permno").alias("ret_excess_lead1"),
                pl.col("yyyymm").shift(-1).over("permno").alias("next_yyyymm"),
                expected_next_yyyymm().alias("expected_next_yyyymm"),
            ]
        )
        .filter(pl.col("next_yyyymm") == pl.col("expected_next_yyyymm"))
        .filter(pl.col("ret_excess_lead1").is_not_null())
        .drop(["next_yyyymm", "expected_next_yyyymm"])
    )


def stage_returns_target(returns_path: Path, macro_path: Path, output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
    target = returns_with_target(returns_path, macro_path).sort(["yyyymm", "permno"])
    target.sink_parquet(output_path, compression="zstd")


def collect_year_batch(
    year: int,
    returns_target_path: Path,
    characteristics_path: Path,
    industry_path: Path,
    ff_path: Path,
    char_cols: list[str],
    industry_cols: list[str],
) -> pd.DataFrame:
    start_yyyymm = year * 100 + 1
    end_yyyymm = year * 100 + 12
    base = pl.scan_parquet(returns_target_path).filter(
        pl.col("yyyymm").is_between(start_yyyymm, end_yyyymm)
    )
    chars = ranked_characteristics(characteristics_path, char_cols).filter(
        pl.col("yyyymm").is_between(start_yyyymm, end_yyyymm)
    )
    industry = (
        add_yyyymm_from_month(pl.scan_parquet(industry_path))
        .filter(pl.col("yyyymm").is_between(start_yyyymm, end_yyyymm))
        .select(KEY_COLUMNS + ["sic2"] + industry_cols)
    )
    ff = (
        pl.scan_parquet(ff_path)
        .rename({"rf": "rf_fama_french"})
        .select(["yyyymm", "mktrf", "smb", "hml", "rmw", "cma", "umd", "rf_fama_french"])
    )
    batch = (
        base.join(chars, on=KEY_COLUMNS, how="inner")
        .join(industry, on=KEY_COLUMNS, how="left")
        .join(ff, on="yyyymm", how="left")
        .with_columns([pl.col(c).fill_null(0).cast(pl.Int8).alias(c) for c in industry_cols])
        .with_columns([pl.col(c).fill_null(0.0).alias(c) for c in MACRO_COLUMNS])
        .with_columns(
            [
                (pl.col(char_col) * pl.col(macro_col)).alias(f"{char_col}__x__{macro_col}")
                for char_col in char_cols
                for macro_col in MACRO_COLUMNS
            ]
        )
        .sort(["month", "permno"])
        .collect()
    )
    return batch.to_pandas()


def write_model_penal_by_year(
    returns_target_path: Path,
    characteristics_path: Path,
    industry_path: Path,
    ff_path: Path,
    output_path: Path,
    char_cols: list[str],
    industry_cols: list[str],
    start_year: int,
    end_year: int,
) -> None:
    if output_path.exists():
        output_path.unlink()
    writer: pq.ParquetWriter | None = None
    try:
        for year in range(start_year, end_year + 1):
            print(f"Writing model_penal batch for {year}", flush=True)
            batch = collect_year_batch(
                year=year,
                returns_target_path=returns_target_path,
                characteristics_path=characteristics_path,
                industry_path=industry_path,
                ff_path=ff_path,
                char_cols=char_cols,
                industry_cols=industry_cols,
            )
            if batch.empty:
                continue
            table = pa.Table.from_pandas(batch, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def build_model_penal(
    returns_path: Path,
    characteristics_path: Path,
    industry_path: Path,
    macro_path: Path,
    ff_path: Path,
    returns_target_path: Path,
    output_path: Path,
    manifest_path: Path,
    start_year: int,
    end_year: int,
) -> dict:
    char_cols = characteristic_columns(characteristics_path)
    industry_cols = industry_dummy_columns(industry_path)
    macro_interactions = [
        f"{char_col}__x__{macro_col}" for char_col in char_cols for macro_col in MACRO_COLUMNS
    ]

    print("Staging returns with next-month excess-return target", flush=True)
    stage_returns_target(returns_path, macro_path, returns_target_path)
    write_model_penal_by_year(
        returns_target_path=returns_target_path,
        characteristics_path=characteristics_path,
        industry_path=industry_path,
        ff_path=ff_path,
        output_path=output_path,
        char_cols=char_cols,
        industry_cols=industry_cols,
        start_year=start_year,
        end_year=end_year,
    )

    scan = pl.scan_parquet(output_path)
    total_feature_columns = char_cols + MACRO_COLUMNS + macro_interactions + industry_cols
    completeness = scan.select(
        [
            pl.len().alias("rows"),
            pl.col("permno").n_unique().alias("permnos"),
            pl.col("yyyymm").min().alias("min_yyyymm"),
            pl.col("yyyymm").max().alias("max_yyyymm"),
            pl.col("month").min().alias("min_month"),
            pl.col("month").max().alias("max_month"),
            pl.col("ret_excess_lead1").null_count().alias("target_nulls"),
            pl.col("me").null_count().alias("market_equity_nulls"),
            pl.sum_horizontal([pl.col(c).null_count() for c in char_cols]).alias(
                "characteristic_null_cells"
            ),
            pl.sum_horizontal([pl.col(c).null_count() for c in MACRO_COLUMNS]).alias(
                "macro_null_cells"
            ),
            pl.sum_horizontal([pl.col(c).null_count() for c in macro_interactions]).alias(
                "macro_interaction_null_cells"
            ),
            pl.sum_horizontal([pl.col(c).null_count() for c in industry_cols]).alias(
                "industry_dummy_null_cells"
            ),
            pl.min_horizontal([pl.col(c).min() for c in char_cols]).alias(
                "ranked_characteristic_min"
            ),
            pl.max_horizontal([pl.col(c).max() for c in char_cols]).alias(
                "ranked_characteristic_max"
            ),
        ]
    ).collect()

    by_month = (
        scan.group_by("yyyymm")
        .agg(pl.len().alias("rows"))
        .select(
            [
                pl.col("rows").min().alias("min_rows_per_month"),
                pl.col("rows").median().alias("median_rows_per_month"),
                pl.col("rows").max().alias("max_rows_per_month"),
            ]
        )
        .collect()
    )

    manifest = {
        "panel": str(output_path.relative_to(ROOT)),
        "rows": int(completeness["rows"][0]),
        "columns": len(parquet_columns(output_path)),
        "permnos": int(completeness["permnos"][0]),
        "yyyymm_start": int(completeness["min_yyyymm"][0]),
        "yyyymm_end": int(completeness["max_yyyymm"][0]),
        "month_start": str(completeness["min_month"][0]),
        "month_end": str(completeness["max_month"][0]),
        "asset_id": "permno",
        "date": "month",
        "target": "ret_excess_lead1",
        "target_risk_free_rate": "rf_welch_goyal",
        "weight": "me",
        "stock_characteristics": char_cols,
        "stock_characteristic_count": len(char_cols),
        "macro_predictors": MACRO_COLUMNS,
        "macro_predictor_count": len(MACRO_COLUMNS),
        "macro_interactions": macro_interactions,
        "macro_interaction_count": len(macro_interactions),
        "industry_dummies": industry_cols,
        "industry_dummy_count": len(industry_cols),
        "total_model_feature_count": len(total_feature_columns),
        "completeness": {
            "target_nulls": int(completeness["target_nulls"][0]),
            "market_equity_nulls": int(completeness["market_equity_nulls"][0]),
            "characteristic_null_cells_after_median_fill_and_rank": int(
                completeness["characteristic_null_cells"][0]
            ),
            "macro_null_cells": int(completeness["macro_null_cells"][0]),
            "macro_interaction_null_cells": int(completeness["macro_interaction_null_cells"][0]),
            "industry_dummy_null_cells": int(completeness["industry_dummy_null_cells"][0]),
            "ranked_characteristic_min": float(completeness["ranked_characteristic_min"][0]),
            "ranked_characteristic_max": float(completeness["ranked_characteristic_max"][0]),
            "min_rows_per_month": int(by_month["min_rows_per_month"][0]),
            "median_rows_per_month": float(by_month["median_rows_per_month"][0]),
            "max_rows_per_month": int(by_month["max_rows_per_month"][0]),
        },
        "inputs": {
            "returns": str(returns_path.relative_to(ROOT)),
            "returns_target_staged": str(returns_target_path.relative_to(ROOT)),
            "characteristics": str(characteristics_path.relative_to(ROOT)),
            "industry_dummies": str(industry_path.relative_to(ROOT)),
            "welch_goyal_macros": str(macro_path.relative_to(ROOT)),
            "fama_french_factors": str(ff_path.relative_to(ROOT)),
        },
        "processing": [
            "Built one row per stock-month.",
            "Constructed ret_excess = ret - rf_welch_goyal.",
            "Constructed ret_excess_lead1 from the next contiguous calendar month by permno.",
            "Cross-sectionally median-filled and ranked all 94 stock characteristics by yyyymm into [-1, 1].",
            "Added 8 Welch-Goyal macro predictors with macro_ prefixes.",
            "Added all 94 x 8 stock-characteristic-by-macro interactions.",
            "Added observed SIC2 industry dummies.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the real paper-style model_penal parquet from processed inputs."
    )
    parser.add_argument("--returns", type=Path, default=PROCESSED_DIR / "stock_returns.parquet")
    parser.add_argument(
        "--characteristics", type=Path, default=PROCESSED_DIR / "stock_characteristics.parquet"
    )
    parser.add_argument("--industry", type=Path, default=PROCESSED_DIR / "industry_dummies.parquet")
    parser.add_argument("--macro", type=Path, default=PROCESSED_DIR / "welch_goyal_macros.parquet")
    parser.add_argument("--ff", type=Path, default=PROCESSED_DIR / "fama_french_factors.parquet")
    parser.add_argument("--returns-target", type=Path, default=DEFAULT_RETURNS_TARGET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--start-year", type=int, default=1957)
    parser.add_argument("--end-year", type=int, default=2016)
    args = parser.parse_args()

    manifest = build_model_penal(
        returns_path=args.returns,
        characteristics_path=args.characteristics,
        industry_path=args.industry,
        macro_path=args.macro,
        ff_path=args.ff,
        returns_target_path=args.returns_target,
        output_path=args.output,
        manifest_path=args.manifest,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in [
                    "panel",
                    "rows",
                    "columns",
                    "yyyymm_start",
                    "yyyymm_end",
                    "total_model_feature_count",
                ]
            },
            indent=2,
        )
    )
    print(json.dumps(manifest["completeness"], indent=2))


if __name__ == "__main__":
    main()
