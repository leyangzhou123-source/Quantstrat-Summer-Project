from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
AQR_DIR = ROOT / "data" / "external" / "aqr"
REPORT_PATH = ROOT / "reports" / "aqr_vme_dataset_report.md"
LOG_PATH = ROOT / "reports" / "aqr_vme_dataset_log.jsonl"


DATASETS = [
    {
        "source_file": "Value-and-Momentum-Everywhere-Factors-Monthly.xlsx",
        "sheet": "VME Factors",
        "header_row": 21,
        "out_file": "value_momentum_everywhere_factors_monthly.parquet",
        "source_url": "https://www.aqr.com/Insights/Datasets/Value-and-Momentum-Everywhere-Factors-Monthly",
        "description": "Updated monthly zero-cost long/short VME value and momentum factors.",
    },
    {
        "source_file": "Value-and-Momentum-Everywhere-Portfolios-Monthly.xlsx",
        "sheet": "VME Portfolios",
        "header_row": 20,
        "out_file": "value_momentum_everywhere_portfolios_monthly.parquet",
        "source_url": "https://www.aqr.com/Insights/Datasets/Value-and-Momentum-Everywhere-Portfolios-Monthly",
        "description": "Updated monthly long-only tertile VME value and momentum portfolios.",
    },
    {
        "source_file": "Value-and-Momentum-Everywhere-Original-Paper-Data.xlsx",
        "sheet": "VME Factors",
        "header_row": 14,
        "out_file": "value_momentum_everywhere_original_factors.parquet",
        "source_url": "https://www.aqr.com/Insights/Datasets/Value-and-Momentum-Everywhere-Original-Paper-Data",
        "description": "Original paper long/short VME factors from Asness, Moskowitz, and Pedersen (2013).",
    },
    {
        "source_file": "Value-and-Momentum-Everywhere-Original-Paper-Data.xlsx",
        "sheet": "VME Portfolios",
        "header_row": 12,
        "out_file": "value_momentum_everywhere_original_portfolios.parquet",
        "source_url": "https://www.aqr.com/Insights/Datasets/Value-and-Momentum-Everywhere-Original-Paper-Data",
        "description": "Original paper long-only VME test portfolios from Asness, Moskowitz, and Pedersen (2013).",
    },
]


def clean_columns(columns: list[object]) -> list[str]:
    cleaned = []
    seen: dict[str, int] = {}
    for col in columns:
        name = str(col).strip().replace(" ", "_").replace("/", "_").replace("-", "_")
        if name.lower() in {"date", "nan"}:
            name = "date"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        cleaned.append(name)
    return cleaned


def read_aqr_sheet(spec: dict) -> pd.DataFrame:
    path = AQR_DIR / spec["source_file"]
    df = pd.read_excel(path, sheet_name=spec["sheet"], header=spec["header_row"])
    df = df.dropna(how="all").copy()
    df.columns = clean_columns(list(df.columns))
    if "date" not in df.columns:
        raise ValueError(f"No date column found in {path.name} / {spec['sheet']}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()
    df["yyyymm"] = df["date"].dt.year * 100 + df["date"].dt.month
    value_cols = [c for c in df.columns if c not in {"date", "yyyymm"}]
    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["date", "yyyymm"] + value_cols].sort_values("date").reset_index(drop=True)
    return df


def summarize(df: pd.DataFrame, spec: dict) -> dict:
    value_cols = [c for c in df.columns if c not in {"date", "yyyymm"}]
    return {
        "file": spec["out_file"],
        "source_file": spec["source_file"],
        "sheet": spec["sheet"],
        "rows": int(len(df)),
        "columns": int(len(value_cols)),
        "date_min": df["date"].min().date().isoformat() if len(df) else None,
        "date_max": df["date"].max().date().isoformat() if len(df) else None,
        "missing_share": float(df[value_cols].isna().sum().sum() / (len(df) * len(value_cols))) if len(df) and value_cols else None,
        "source_url": spec["source_url"],
        "description": spec["description"],
    }


def write_report(results: list[dict]) -> None:
    lines = [
        "# AQR Value and Momentum Everywhere dataset report",
        "",
        "Downloaded and converted public AQR Value and Momentum Everywhere datasets for supporting work and robustness tests.",
        "",
        "## Main finding",
        "",
        "These files are useful as global factor/portfolio benchmarks, but they do not replace the missing firm-level accounting characteristics required for the 94-characteristic stock-level replication. The AQR data are portfolio/factor returns, not permno-level predictors such as `currat`, `roic`, `stdacc`, or `tb`.",
        "",
        "## Converted files",
        "",
        "| parquet | source workbook | sheet | rows | factor/portfolio columns | date min | date max | missing share |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| `{item['file']}` | `{item['source_file']}` | `{item['sheet']}` | {item['rows']:,} | {item['columns']:,} | {item['date_min']} | {item['date_max']} | {item['missing_share']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Methodology use",
            "",
            "- Use `value_momentum_everywhere_factors_monthly.parquet` as external global value/momentum factor benchmarks in robustness checks.",
            "- Use `value_momentum_everywhere_portfolios_monthly.parquet` to compare long-only value/momentum portfolio behavior across AQR's eight markets/asset classes.",
            "- Use original-paper files only when matching the published Asness, Moskowitz, and Pedersen sample ending in July 2011.",
            "- Do not merge these directly into the 94 firm-level characteristic matrix, because they are aggregate factor/portfolio returns rather than stock-level characteristics.",
            "",
            "## References",
            "",
            "- AQR Data Library, Value and Momentum Everywhere: Factors, Monthly, https://www.aqr.com/Insights/Datasets/Value-and-Momentum-Everywhere-Factors-Monthly",
            "- AQR Data Library, Value and Momentum Everywhere: Portfolios, Monthly, https://www.aqr.com/Insights/Datasets/Value-and-Momentum-Everywhere-Portfolios-Monthly",
            "- AQR Data Library, Value and Momentum Everywhere: Original Paper Data, https://www.aqr.com/Insights/Datasets/Value-and-Momentum-Everywhere-Original-Paper-Data",
            "- Asness, Clifford S., Tobias J. Moskowitz, and Lasse H. Pedersen. 2013. Value and Momentum Everywhere. The Journal of Finance 68(3): 929-985.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    AQR_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with LOG_PATH.open("w") as log:
        for spec in DATASETS:
            df = read_aqr_sheet(spec)
            out_path = AQR_DIR / spec["out_file"]
            df.to_parquet(out_path, index=False)
            item = summarize(df, spec)
            results.append(item)
            log.write(json.dumps(item) + "\n")
    write_report(results)
    print(f"Converted {len(results)} AQR VME datasets")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
