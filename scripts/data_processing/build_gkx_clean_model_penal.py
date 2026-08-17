from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
GKX_URL = "https://dachxiu.chicagobooth.edu/download/datashare.zip"


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=120) as response, output.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def write_gkx_stage(
    zip_path: Path,
    output_path: Path,
    stock_characteristics: list[str],
    start_yyyymm: int,
    end_yyyymm: int,
    chunksize: int,
) -> None:
    if output_path.exists():
        output_path.unlink()
    usecols = ["permno", "DATE", "sic2", *stock_characteristics]
    writer = None
    try:
        with zipfile.ZipFile(zip_path) as archive, archive.open("datashare.csv") as handle:
            for chunk in pd.read_csv(handle, usecols=usecols, chunksize=chunksize):
                chunk["yyyymm"] = (chunk["DATE"].astype("int64") // 100).astype("int64")
                chunk = chunk[
                    (chunk["yyyymm"] >= start_yyyymm) & (chunk["yyyymm"] <= end_yyyymm)
                ].drop(columns=["DATE"])
                if chunk.empty:
                    continue
                chunk["permno"] = chunk["permno"].astype("int64")
                table = pl.from_pandas(chunk).to_arrow()
                if writer is None:
                    import pyarrow.parquet as pq

                    writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def rank_gkx_characteristics(stage_path: Path, stock_characteristics: list[str]) -> pl.LazyFrame:
    base = pl.scan_parquet(stage_path)
    filled = base.with_columns(
        [
            pl.col(col).fill_null(pl.col(col).median().over("yyyymm")).fill_null(0.0).alias(col)
            for col in stock_characteristics
        ]
    )
    n_by_month = pl.len().over("yyyymm")
    return filled.with_columns(
        [
            (
                -1.0
                + 2.0
                * ((pl.col(col).rank(method="average").over("yyyymm") - 1.0) / (n_by_month - 1.0))
            )
            .cast(pl.Float32)
            .alias(col)
            for col in stock_characteristics
        ]
    )


def build_clean_panel(
    current_path: Path,
    manifest_path: Path,
    stage_path: Path,
    output_path: Path,
    output_manifest_path: Path,
    min_target: float | None,
    max_target: float | None,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text())
    stock_characteristics = list(manifest["stock_characteristics"])
    macro_predictors = list(manifest["macro_predictors"])
    old_industry = list(manifest["industry_dummies"])
    non_feature_cols = [
        col
        for col in pl.scan_parquet(current_path).collect_schema().names()
        if col not in stock_characteristics and col not in old_industry
    ]

    current = pl.scan_parquet(current_path).select(non_feature_cols)
    gkx = rank_gkx_characteristics(stage_path, stock_characteristics)
    gkx_features = gkx.select(["permno", "yyyymm", "sic2", *stock_characteristics])
    joined = (
        current.join(gkx_features, on=["permno", "yyyymm"], how="inner", suffix="_gkx")
        .filter(pl.col("me") > 0)
        .with_columns(
            [
                pl.when(pl.col("sic2_gkx").is_not_null())
                .then(pl.col("sic2_gkx").cast(pl.Int64))
                .otherwise(pl.col("sic2").cast(pl.Int64))
                .alias("sic2")
            ]
        )
        .drop("sic2_gkx")
    )
    if min_target is not None or max_target is not None:
        low = -float("inf") if min_target is None else float(min_target)
        high = float("inf") if max_target is None else float(max_target)
        joined = joined.with_columns(
            pl.col(manifest["target"]).clip(low, high).alias(manifest["target"])
        )

    observed_sic2 = (
        joined.select("sic2")
        .unique()
        .collect()
        .get_column("sic2")
        .drop_nulls()
        .cast(pl.Int64)
        .sort()
        .to_list()
    )
    industry_cols = [f"sic2_{int(code)}" for code in observed_sic2]
    joined = joined.with_columns(
        [
            (pl.col("sic2") == int(code)).cast(pl.Int8).alias(f"sic2_{int(code)}")
            for code in observed_sic2
        ]
    )
    output_cols = [
        *non_feature_cols,
        *stock_characteristics,
        *industry_cols,
    ]
    output_cols = list(dict.fromkeys(output_cols))
    if output_path.exists():
        output_path.unlink()
    joined.select(output_cols).sort(["month", "permno"]).sink_parquet(
        output_path, compression="zstd", statistics=True
    )

    clean = pl.scan_parquet(output_path)
    audit = (
        clean.select(
            [
                pl.len().alias("rows"),
                pl.struct(["permno", "yyyymm"]).n_unique().alias("unique_permno_yyyymm"),
                pl.col("permno").n_unique().alias("permnos"),
                pl.col("yyyymm").n_unique().alias("months"),
                pl.col("yyyymm").min().alias("yyyymm_start"),
                pl.col("yyyymm").max().alias("yyyymm_end"),
                (pl.col("me") <= 0).sum().alias("me_nonpositive"),
                pl.sum_horizontal([pl.col(c) for c in industry_cols])
                .eq(1)
                .sum()
                .alias("industry_one_hot"),
                pl.sum_horizontal([pl.col(c) for c in industry_cols])
                .eq(0)
                .sum()
                .alias("industry_no_dummy"),
                pl.sum_horizontal([pl.col(c) for c in industry_cols])
                .gt(1)
                .sum()
                .alias("industry_multi_dummy"),
                pl.sum_horizontal(
                    [pl.col(c).is_null().cast(pl.Int64) for c in stock_characteristics]
                ).alias("characteristic_null_cells"),
                pl.sum_horizontal(
                    [(~pl.col(c).is_between(-1, 1)).cast(pl.Int64) for c in stock_characteristics]
                ).alias("characteristic_outside_rank_cells"),
                pl.sum_horizontal(
                    [pl.col(c).is_null().cast(pl.Int64) for c in macro_predictors]
                ).alias("macro_null_cells"),
                pl.col(manifest["target"]).min().alias("target_min"),
                pl.col(manifest["target"]).max().alias("target_max"),
                (pl.col(manifest["target"]).abs() > 1).sum().alias("target_abs_gt_100pct"),
                (pl.col(manifest["target"]).abs() > 5).sum().alias("target_abs_gt_500pct"),
            ]
        )
        .collect()
        .to_dicts()[0]
    )

    clean_manifest = dict(manifest)
    clean_manifest["panel"] = str(output_path.relative_to(ROOT))
    clean_manifest["columns"] = clean.collect_schema().names()
    clean_manifest["rows"] = int(audit["rows"])
    clean_manifest["permnos"] = int(audit["permnos"])
    clean_manifest["yyyymm_start"] = int(audit["yyyymm_start"])
    clean_manifest["yyyymm_end"] = int(audit["yyyymm_end"])
    clean_manifest["month_start"] = str(clean.select(pl.col("month").min()).collect().item())
    clean_manifest["month_end"] = str(clean.select(pl.col("month").max()).collect().item())
    clean_manifest["industry_dummies"] = industry_cols
    clean_manifest["industry_dummy_count"] = len(industry_cols)
    clean_manifest["total_model_feature_count"] = (
        len(stock_characteristics) + len(macro_predictors) + len(industry_cols)
    )
    clean_manifest["processing"] = [
        "Clean GKX-aligned panel built from current returns/macros and official Dacheng Xiu datashare characteristics.",
        "Matched current panel rows to official GKX permno-month keys.",
        "Replaced all 94 ranked stock characteristics with official GKX values ranked by yyyymm.",
        "Rebuilt industry dummies from official GKX sic2.",
        "Dropped rows with nonpositive market equity.",
    ]
    if min_target is not None or max_target is not None:
        clean_manifest["processing"].append(
            f"Clipped {manifest['target']} to [{min_target}, {max_target}]."
        )
    clean_manifest["completeness"] = {
        k: int(v) if isinstance(v, (np.integer, int)) else v for k, v in audit.items()
    }
    output_manifest_path.write_text(json.dumps(clean_manifest, indent=2) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a GKX-aligned clean model_penal parquet.")
    parser.add_argument("--current", type=Path, default=PROCESSED_DIR / "model_penal.parquet")
    parser.add_argument(
        "--manifest", type=Path, default=PROCESSED_DIR / "model_penal_manifest.json"
    )
    parser.add_argument(
        "--output", type=Path, default=PROCESSED_DIR / "model_penal_gkx_clean.parquet"
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=PROCESSED_DIR / "model_penal_gkx_clean_manifest.json",
    )
    parser.add_argument(
        "--zip-path", type=Path, default=Path("/private/tmp/gkx_datashare_clean_build.zip")
    )
    parser.add_argument(
        "--stage-path", type=Path, default=Path("/private/tmp/gkx_datashare_clean_stage.parquet")
    )
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--min-target", type=float, default=-1.0)
    parser.add_argument("--max-target", type=float, default=1.0)
    parser.add_argument("--keep-download", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    try:
        print("Downloading official GKX datashare...", flush=True)
        download(GKX_URL, args.zip_path)
        print("Staging official characteristics...", flush=True)
        write_gkx_stage(
            zip_path=args.zip_path,
            output_path=args.stage_path,
            stock_characteristics=manifest["stock_characteristics"],
            start_yyyymm=int(manifest["yyyymm_start"]),
            end_yyyymm=int(manifest["yyyymm_end"]),
            chunksize=args.chunksize,
        )
        print("Building clean parquet...", flush=True)
        audit = build_clean_panel(
            current_path=args.current,
            manifest_path=args.manifest,
            stage_path=args.stage_path,
            output_path=args.output,
            output_manifest_path=args.output_manifest,
            min_target=args.min_target,
            max_target=args.max_target,
        )
        print(json.dumps(audit, indent=2, default=str), flush=True)
    finally:
        if not args.keep_download:
            for path in [args.zip_path, args.stage_path]:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass


if __name__ == "__main__":
    main()
