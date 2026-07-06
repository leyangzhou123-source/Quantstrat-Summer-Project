from __future__ import annotations

import json
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[2]
RETURNS_PATH = ROOT / "data" / "raw" / "01_stock_returns.parquet"
FF_PATH = ROOT / "data" / "raw" / "04_fama_french_factors.parquet"
CHAR_PATH = ROOT / "data" / "processed" / "paper_73_characteristics_ranked.parquet"
INDUSTRY_PATH = ROOT / "data" / "raw" / "03_industry_dummies.parquet"
MACRO_PATH = ROOT / "data" / "raw" / "05_welch_goyal_macro_monthly.parquet"

OUT_RETURNS_PATH = ROOT / "data" / "processed" / "paper_returns_excess.parquet"
OUT_PANEL_PATH = ROOT / "data" / "processed" / "paper_model_inputs_73.parquet"
REPORT_PATH = ROOT / "reports" / "paper_input_conversion_report.md"
LOG_PATH = ROOT / "reports" / "paper_input_conversion_log.jsonl"


def yyyymm_expr(date_col: str) -> pl.Expr:
    return (pl.col(date_col).dt.year() * 100 + pl.col(date_col).dt.month()).cast(pl.Int64)


def load_returns() -> pl.DataFrame:
    return (
        pl.scan_parquet(RETURNS_PATH)
        .select(
            pl.col("permno").cast(pl.Int64),
            pl.col("permco").cast(pl.Int64),
            pl.col("month").cast(pl.Date),
            yyyymm_expr("month").alias("yyyymm"),
            pl.col("ret").cast(pl.Float64, strict=False),
            pl.col("retx").cast(pl.Float64, strict=False),
            pl.col("prc").cast(pl.Float64, strict=False),
            pl.col("shrout").cast(pl.Float64, strict=False),
            pl.col("me").cast(pl.Float64, strict=False).alias("market_equity"),
            pl.col("siccd").cast(pl.Float64, strict=False),
            pl.col("shrcd").cast(pl.Float64, strict=False),
            pl.col("exchcd").cast(pl.Float64, strict=False),
        )
        .with_columns((pl.col("siccd") // 100).cast(pl.Int64).alias("sic2"))
        .sort(["permno", "yyyymm"])
        .collect()
    )


def load_risk_free() -> pl.DataFrame:
    return (
        pl.scan_parquet(FF_PATH)
        .select(
            pl.col("month").cast(pl.Date),
            yyyymm_expr("month").alias("yyyymm"),
            pl.col("rf").cast(pl.Float64, strict=False),
            pl.col("mktrf").cast(pl.Float64, strict=False),
            pl.col("smb").cast(pl.Float64, strict=False),
            pl.col("hml").cast(pl.Float64, strict=False),
            pl.col("umd").cast(pl.Float64, strict=False),
        )
        .unique(subset=["yyyymm"])
        .sort("yyyymm")
        .collect()
    )


def convert_returns() -> pl.DataFrame:
    returns = load_returns()
    rf = load_risk_free()
    converted = (
        returns.join(rf, on="yyyymm", how="left", suffix="_ff")
        .with_columns(
            (pl.col("ret") - pl.col("rf")).alias("ret_excess"),
            (pl.col("retx") - pl.col("rf")).alias("retx_excess"),
        )
        .sort(["permno", "yyyymm"])
        .with_columns(
            pl.col("ret_excess").shift(-1).over("permno").alias("ret_excess_lead1"),
            pl.col("yyyymm").shift(-1).over("permno").alias("target_yyyymm"),
        )
        .with_columns(
            (
                ((pl.col("yyyymm") // 100) * 12 + (pl.col("yyyymm") % 100) + 1)
                == ((pl.col("target_yyyymm") // 100) * 12 + (pl.col("target_yyyymm") % 100))
            ).alias("has_next_calendar_month")
        )
    )
    converted.write_parquet(OUT_RETURNS_PATH, compression="zstd")
    return converted


def build_panel(converted_returns: pl.DataFrame) -> pl.DataFrame:
    chars = pl.read_parquet(CHAR_PATH)
    macro = (
        pl.scan_parquet(MACRO_PATH)
        .select(
            pl.col("yyyymm").cast(pl.Int64),
            pl.col("dp").cast(pl.Float32),
            pl.col("ep").cast(pl.Float32),
            pl.col("bm").cast(pl.Float32).alias("macro_bm"),
            pl.col("ntis").cast(pl.Float32),
            pl.col("tbl").cast(pl.Float32),
            pl.col("tms").cast(pl.Float32),
            pl.col("dfy").cast(pl.Float32),
            pl.col("svar").cast(pl.Float32),
        )
        .collect()
    )
    industries = (
        pl.scan_parquet(INDUSTRY_PATH)
        .select(
            pl.col("permno").cast(pl.Int64),
            yyyymm_expr("month").alias("yyyymm"),
            pl.exclude(["permno", "month", "sic2"]).cast(pl.Float32, strict=False),
        )
        .collect()
    )
    panel = (
        converted_returns.join(chars, on=["permno", "yyyymm"], how="inner")
        .join(macro, on="yyyymm", how="left")
        .join(industries, on=["permno", "yyyymm"], how="left", suffix="_industry")
        .filter(pl.col("ret_excess_lead1").is_not_null())
        .filter(pl.col("has_next_calendar_month"))
        .sort(["yyyymm", "permno"])
    )
    panel.write_parquet(OUT_PANEL_PATH, compression="zstd")
    return panel


def macro_profile() -> dict:
    macro = pl.read_parquet(MACRO_PATH)
    return {
        "rows": int(macro.height),
        "yyyymm_min": int(macro["yyyymm"].min()),
        "yyyymm_max": int(macro["yyyymm"].max()),
        "bm_nulls": int(macro["bm"].null_count()) if "bm" in macro.columns else None,
        "columns": macro.columns,
    }


def quality(converted: pl.DataFrame, panel: pl.DataFrame) -> dict:
    char_cols = [
        c
        for c in pl.read_parquet(CHAR_PATH, n_rows=1).columns
        if c not in {"permno", "yyyymm"}
    ]
    industry_cols = [c for c in panel.columns if c.startswith("sic2_")]
    macro_cols = ["dp", "ep", "macro_bm", "ntis", "tbl", "tms", "dfy", "svar"]
    return {
        "returns_rows": int(converted.height),
        "returns_min_yyyymm": int(converted["yyyymm"].min()),
        "returns_max_yyyymm": int(converted["yyyymm"].max()),
        "rf_missing_rows": int(converted["rf"].null_count()),
        "ret_excess_missing_rows": int(converted["ret_excess"].null_count()),
        "lead_target_missing_rows": int(converted["ret_excess_lead1"].null_count()),
        "calendar_next_month_rows": int(converted["has_next_calendar_month"].sum()),
        "panel_rows": int(panel.height),
        "panel_columns": int(len(panel.columns)),
        "panel_min_yyyymm": int(panel["yyyymm"].min()),
        "panel_max_yyyymm": int(panel["yyyymm"].max()),
        "panel_permnos": int(panel["permno"].n_unique()),
        "characteristic_count": len(char_cols),
        "macro_predictor_count": len(macro_cols),
        "macro_missing_rows": int(panel.select([pl.col(c).null_count() for c in macro_cols]).sum_horizontal().item()),
        "industry_dummy_count": len(industry_cols),
        "target_missing_rows": int(panel["ret_excess_lead1"].null_count()),
        "panel_sorted": bool(
            panel.select(["yyyymm", "permno"]).equals(
                panel.select(["yyyymm", "permno"]).sort(["yyyymm", "permno"])
            )
        ),
        "target_min": float(panel["ret_excess_lead1"].min()),
        "target_max": float(panel["ret_excess_lead1"].max()),
        "target_mean": float(panel["ret_excess_lead1"].mean()),
    }


def write_report(q: dict, macro: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Paper input conversion report",
        "",
        "Based on Gu, Kelly, and Xiu (2020), the converted modeling target is next-month individual stock excess return. The paper states that it obtains monthly CRSP total stock returns and the Treasury-bill rate, then calculates individual excess returns. Characteristics are cross-sectionally ranked each period to `[-1, 1]`, missing characteristics are replaced by the monthly cross-sectional median, and SIC codes become 74 industry dummies.",
        "",
        "## Inputs that require conversion",
        "",
        "| input | paper requirement | local source | conversion performed | output |",
        "|---|---|---|---|---|",
        "| Individual stock returns | Predict excess return over risk-free rate | `data/raw/01_stock_returns.parquet` plus `data/raw/04_fama_french_factors.parquet` | `ret_excess = ret - rf`; `retx_excess = retx - rf` | `data/processed/paper_returns_excess.parquet` |",
        "| Prediction target | Forecast `r_{i,t+1}` using information at month `t` | converted stock excess returns | `ret_excess_lead1 = groupby(permno).shift(-1)` and require next calendar month | `data/processed/paper_model_inputs_73.parquet` |",
        "| Risk-free rate | Treasury-bill proxy for `rf` | Fama-French `rf` column | merged by `yyyymm`; already monthly decimal units | both converted outputs |",
        "| Stock characteristics | Rank to `[-1, 1]`, median-impute by month | `data/processed/paper_73_characteristics_ranked.parquet` | already converted in prior step; merged into panel | `data/processed/paper_model_inputs_73.parquet` |",
        "| Industry information | 74 SIC2 industry dummies | `data/raw/03_industry_dummies.parquet` | merged by `permno`, `yyyymm` | `data/processed/paper_model_inputs_73.parquet` |",
        "| Fama-French market/factor returns | Benchmark factors; `mktrf` is already market excess return | `data/raw/04_fama_french_factors.parquet` | no additional RF subtraction for `mktrf`; merged alongside RF/factors | converted returns output |",
        "| Macro predictors | Eight monthly Welch-Goyal predictors interacted with characteristics | `data/raw/05_welch_goyal_macro_monthly.parquet` | merged by `yyyymm`; `bm` is named `macro_bm` to avoid collision with the stock characteristic `bm` | `data/processed/paper_model_inputs_73.parquet` |",
        "",
        "## Output quality",
        "",
        f"- Converted returns rows: {q['returns_rows']:,}",
        f"- Converted returns date range: {q['returns_min_yyyymm']} to {q['returns_max_yyyymm']}",
        f"- RF missing rows after merge: {q['rf_missing_rows']:,}",
        f"- Excess-return missing rows: {q['ret_excess_missing_rows']:,}",
        f"- Lead-target missing rows before filtering: {q['lead_target_missing_rows']:,}",
        f"- Rows with next calendar-month target: {q['calendar_next_month_rows']:,}",
        f"- Final model-input rows: {q['panel_rows']:,}",
        f"- Final model-input columns: {q['panel_columns']:,}",
        f"- Final model-input date range: {q['panel_min_yyyymm']} to {q['panel_max_yyyymm']}",
        f"- Unique permnos: {q['panel_permnos']:,}",
        f"- Characteristics included: {q['characteristic_count']}",
        f"- Macro predictors included: {q['macro_predictor_count']}",
        f"- Macro missing cells in final panel: {q['macro_missing_rows']:,}",
        f"- SIC2 industry dummy columns included: {q['industry_dummy_count']}",
        f"- Final target missing rows: {q['target_missing_rows']:,}",
        f"- Sorted by `yyyymm`, `permno`: {q['panel_sorted']}",
        f"- Target mean/min/max: {q['target_mean']:.6f}, {q['target_min']:.6f}, {q['target_max']:.6f}",
        "",
        "## Macro data quality",
        "",
        f"- Current macro rows: {macro['rows']:,}",
        f"- Current macro date range: {macro['yyyymm_min']} to {macro['yyyymm_max']}",
        f"- Current macro `bm` null rows: {macro['bm_nulls']:,}",
        "- The paper needs monthly `dp`, `ep`, `bm`, `ntis`, `tbl`, `tms`, `dfy`, and `svar`; the downloaded file now covers this requirement for the 1957-2016 replication window.",
        "- Macro-characteristic interactions are not materialized in this file to avoid creating hundreds of additional columns; they can be generated from the 73 ranked characteristic columns and these 8 macro columns before model fitting.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_RETURNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    converted = convert_returns()
    panel = build_panel(converted)
    q = quality(converted, panel)
    macro = macro_profile()
    write_report(q, macro)

    with LOG_PATH.open("w") as fh:
        fh.write(json.dumps({"type": "quality", **q}) + "\n")
        fh.write(json.dumps({"type": "macro_gap", **macro}) + "\n")

    print(f"Wrote {OUT_RETURNS_PATH.relative_to(ROOT)}")
    print(f"Wrote {OUT_PANEL_PATH.relative_to(ROOT)}")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
