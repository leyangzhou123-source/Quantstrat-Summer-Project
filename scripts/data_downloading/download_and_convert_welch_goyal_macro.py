from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DIR = ROOT / "data" / "external"
RAW_OUT = EXTERNAL_DIR / "welch_goyal_updated_2025.xlsx"
RAW_COMBINED_OUT = EXTERNAL_DIR / "welch_goyal_all_data_2024.xlsx"
PARQUET_OUT = ROOT / "data" / "raw" / "05_welch_goyal_macro_monthly.parquet"
REPORT_OUT = ROOT / "reports" / "welch_goyal_macro_quality_report.md"
LOG_OUT = ROOT / "reports" / "welch_goyal_macro_quality_log.jsonl"


DOWNLOADS = {
    RAW_OUT: "https://docs.google.com/spreadsheets/d/1qwpl2R_DNujpU5YUkk8lacP1tTeMb9iJ/export?format=xlsx",
    RAW_COMBINED_OUT: "https://docs.google.com/spreadsheets/d/10_nkOkJPvq4eZgNl-1ys63PzhbnM3S2y/export?format=xlsx",
}


def download() -> list[dict]:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for path, url in DOWNLOADS.items():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=90) as response:
            data = response.read()
        path.write_bytes(data)
        results.append(
            {
                "file": str(path.relative_to(ROOT)),
                "url": url,
                "bytes": len(data),
                "signature": data[:4].hex(),
            }
        )
    return results


def convert_monthly() -> pd.DataFrame:
    monthly = pd.read_excel(RAW_OUT, sheet_name="Monthly")
    monthly.columns = [str(c).strip() for c in monthly.columns]

    required = ["yyyymm", "Index", "D12", "E12", "b/m", "tbl", "AAA", "BAA", "lty", "ntis", "Rfree", "svar"]
    missing = [c for c in required if c not in monthly.columns]
    if missing:
        raise ValueError(f"Missing required Welch-Goyal columns: {missing}")

    for col in required:
        monthly[col] = pd.to_numeric(monthly[col], errors="coerce")

    out = pd.DataFrame()
    out["yyyymm"] = monthly["yyyymm"].astype("Int64")
    out["month"] = pd.to_datetime(out["yyyymm"].astype(str) + "01", format="%Y%m%d") + pd.offsets.MonthEnd(0)

    div_price = monthly["D12"] / monthly["Index"]
    earn_price = monthly["E12"] / monthly["Index"]
    out["dp"] = np.log(div_price.where(div_price > 0))
    out["ep"] = np.log(earn_price.where(earn_price > 0))
    out["bm"] = monthly["b/m"]
    out["ntis"] = monthly["ntis"]
    out["tbl"] = monthly["tbl"]
    out["tms"] = monthly["lty"] - monthly["tbl"]
    out["dfy"] = monthly["BAA"] - monthly["AAA"]
    out["svar"] = monthly["svar"]
    out["rf_welch_goyal"] = monthly["Rfree"]

    out = out[(out["yyyymm"] >= 195703) & (out["yyyymm"] <= 201612)].copy()
    out = out.sort_values("yyyymm").reset_index(drop=True)

    # Keep numeric columns in decimal/rate units as supplied by Welch-Goyal.
    PARQUET_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PARQUET_OUT, index=False)
    return out


def quality(df: pd.DataFrame, downloads: list[dict]) -> dict:
    cols = ["dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar"]
    nulls = {col: int(df[col].isna().sum()) for col in cols}
    stats = {
        col: {
            "min": float(df[col].min()) if df[col].notna().any() else None,
            "max": float(df[col].max()) if df[col].notna().any() else None,
            "mean": float(df[col].mean()) if df[col].notna().any() else None,
        }
        for col in cols
    }
    y = df["yyyymm"].astype(int)
    month_index = (y // 100) * 12 + (y % 100)
    return {
        "downloads": downloads,
        "rows": int(len(df)),
        "yyyymm_min": int(df["yyyymm"].min()),
        "yyyymm_max": int(df["yyyymm"].max()),
        "expected_months_195703_201612": 718,
        "is_complete_monthly_sequence": bool(month_index.diff().dropna().eq(1).all()),
        "nulls": nulls,
        "stats": stats,
        "columns": list(df.columns),
    }


def write_report(q: dict) -> None:
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Welch-Goyal monthly macro predictor quality report",
        "",
        "Downloaded public data from Amit Goyal's website and converted the monthly predictors required by Gu, Kelly, and Xiu (2020).",
        "",
        "## Downloaded files",
        "",
        "| file | bytes | source |",
        "|---|---:|---|",
    ]
    for item in q["downloads"]:
        lines.append(f"| `{item['file']}` | {item['bytes']:,} | {item['url']} |")

    lines.extend(
        [
            "",
            "## Converted output",
            "",
            f"- Output: `{PARQUET_OUT.relative_to(ROOT)}`",
            f"- Rows: {q['rows']:,}",
            f"- Date range: {q['yyyymm_min']} to {q['yyyymm_max']}",
            f"- Expected months in replication window: {q['expected_months_195703_201612']}",
            f"- Complete monthly sequence: {q['is_complete_monthly_sequence']}",
            "",
            "## Variable definitions used",
            "",
            "| output | construction from Goyal workbook |",
            "|---|---|",
            "| `dp` | `log(D12 / Index)` |",
            "| `ep` | `log(E12 / Index)` |",
            "| `bm` | `b/m` |",
            "| `ntis` | `ntis` |",
            "| `tbl` | `tbl` |",
            "| `tms` | `lty - tbl` |",
            "| `dfy` | `BAA - AAA` |",
            "| `svar` | `svar` |",
            "| `rf_welch_goyal` | `Rfree`, kept for comparison to the Fama-French RF already used in the return target |",
            "",
            "## Missing values",
            "",
            "| variable | null rows | min | mean | max |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for col, n in q["nulls"].items():
        st = q["stats"][col]
        lines.append(
            f"| `{col}` | {n:,} | {st['min']:.6f} | {st['mean']:.6f} | {st['max']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This replaces the earlier annual-only `data/raw/05_welch_goyal_macro.parquet` for the replication's monthly macro interaction design.",
            "- The file is sorted by `yyyymm` and covers March 1957 through December 2016, matching the paper's sample window.",
        ]
    )
    REPORT_OUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    downloads = download()
    df = convert_monthly()
    q = quality(df, downloads)
    write_report(q)
    with LOG_OUT.open("w") as fh:
        fh.write(json.dumps(q) + "\n")
    print(f"Wrote {PARQUET_OUT.relative_to(ROOT)}")
    print(f"Wrote {REPORT_OUT.relative_to(ROOT)}")
    print(f"Wrote {LOG_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
