from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DIR = ROOT / "data" / "external"
RAW_DIR = ROOT / "data" / "raw"
REPORT_PATH = ROOT / "reports" / "fama_french_factor_quality_report.md"
LOG_PATH = ROOT / "reports" / "fama_french_factor_quality_log.jsonl"

FF5_ZIP = EXTERNAL_DIR / "F-F_Research_Data_5_Factors_2x3_CSV.zip"
MOM_ZIP = EXTERNAL_DIR / "F-F_Momentum_Factor_CSV.zip"
FF5_OUT = RAW_DIR / "04_fama_french_5_factors_monthly.parquet"
MOM_OUT = RAW_DIR / "04_fama_french_momentum_monthly.parquet"
COMBINED_OUT = RAW_DIR / "04_fama_french_5_plus_momentum_monthly.parquet"

URLS = {
    FF5_ZIP: "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip",
    MOM_ZIP: "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip",
}


def download() -> list[dict]:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for path, url in URLS.items():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=90) as response:
            data = response.read()
        path.write_bytes(data)
        results.append({"file": str(path.relative_to(ROOT)), "url": url, "bytes": len(data)})
    return results


def read_ken_french_zip(path: Path, value_names: dict[str, str]) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        csv_name = [name for name in zf.namelist() if name.lower().endswith(".csv")][0]
        text = zf.read(csv_name).decode("latin1")

    lines = text.splitlines()
    header_idx = None
    for idx, line in enumerate(lines):
        first = line.split(",", 1)[0].strip()
        if first in {"", "YYYYMM"} and any(key in line for key in value_names):
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError(f"Could not find monthly header in {path}")

    data_lines = [lines[header_idx]]
    for line in lines[header_idx + 1 :]:
        first = line.split(",", 1)[0].strip()
        if not first:
            break
        if not first.isdigit() or len(first) != 6:
            break
        data_lines.append(line)

    df = pd.read_csv(io.StringIO("\n".join(data_lines)))
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "yyyymm"})
    df["yyyymm"] = pd.to_numeric(df["yyyymm"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["yyyymm"]).copy()
    df["yyyymm"] = df["yyyymm"].astype(int)
    df["month"] = pd.to_datetime(df["yyyymm"].astype(str) + "01", format="%Y%m%d") + pd.offsets.MonthEnd(0)

    rename = {}
    for col in df.columns:
        normalized = col.strip()
        if normalized in value_names:
            rename[col] = value_names[normalized]
    df = df.rename(columns=rename)
    keep = ["yyyymm", "month"] + list(value_names.values())
    df = df[keep]
    for col in value_names.values():
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0
    return df.sort_values("yyyymm").reset_index(drop=True)


def convert() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ff5 = read_ken_french_zip(
        FF5_ZIP,
        {
            "Mkt-RF": "mktrf",
            "SMB": "smb",
            "HML": "hml",
            "RMW": "rmw",
            "CMA": "cma",
            "RF": "rf",
        },
    )
    mom = read_ken_french_zip(MOM_ZIP, {"Mom": "umd"})

    ff5_window = ff5[(ff5["yyyymm"] >= 195703) & (ff5["yyyymm"] <= 201612)].copy()
    mom_window = mom[(mom["yyyymm"] >= 195703) & (mom["yyyymm"] <= 201612)].copy()
    combined = ff5_window.merge(mom_window[["yyyymm", "umd"]], on="yyyymm", how="left")

    FF5_OUT.parent.mkdir(parents=True, exist_ok=True)
    ff5_window.to_parquet(FF5_OUT, index=False)
    mom_window.to_parquet(MOM_OUT, index=False)
    combined.to_parquet(COMBINED_OUT, index=False)
    return ff5_window, mom_window, combined


def quality(df: pd.DataFrame, name: str, cols: list[str]) -> dict:
    y = df["yyyymm"].astype(int)
    month_idx = (y // 100) * 12 + (y % 100)
    return {
        "name": name,
        "rows": int(len(df)),
        "yyyymm_min": int(df["yyyymm"].min()),
        "yyyymm_max": int(df["yyyymm"].max()),
        "complete_monthly_sequence": bool(month_idx.diff().dropna().eq(1).all()),
        "nulls": {col: int(df[col].isna().sum()) for col in cols},
        "means": {col: float(df[col].mean()) for col in cols},
        "mins": {col: float(df[col].min()) for col in cols},
        "maxs": {col: float(df[col].max()) for col in cols},
    }


def write_report(downloads: list[dict], qualities: list[dict]) -> None:
    lines = [
        "# Fama-French factor data quality report",
        "",
        "Downloaded official Kenneth French monthly factor files and converted the replication-window observations to decimal returns.",
        "",
        "## Downloaded files",
        "",
        "| file | bytes | source |",
        "|---|---:|---|",
    ]
    for item in downloads:
        lines.append(f"| `{item['file']}` | {item['bytes']:,} | {item['url']} |")

    lines.extend(
        [
            "",
            "## Converted outputs",
            "",
            f"- FF5: `{FF5_OUT.relative_to(ROOT)}`",
            f"- Momentum: `{MOM_OUT.relative_to(ROOT)}`",
            f"- FF5 plus momentum: `{COMBINED_OUT.relative_to(ROOT)}`",
            "",
        ]
    )
    for q in qualities:
        lines.extend(
            [
                f"### {q['name']}",
                "",
                f"- Rows: {q['rows']:,}",
                f"- Date range: {q['yyyymm_min']} to {q['yyyymm_max']}",
                f"- Complete monthly sequence: {q['complete_monthly_sequence']}",
                "",
                "| variable | null rows | mean | min | max |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for col, nulls in q["nulls"].items():
            lines.append(
                f"| `{col}` | {nulls:,} | {q['means'][col]:.6f} | {q['mins'][col]:.6f} | {q['maxs'][col]:.6f} |"
            )
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines))


def main() -> None:
    downloads = download()
    ff5, mom, combined = convert()
    qualities = [
        quality(ff5, "Fama-French 5 factors", ["mktrf", "smb", "hml", "rmw", "cma", "rf"]),
        quality(mom, "Fama-French momentum", ["umd"]),
        quality(combined, "Fama-French 5 factors plus momentum", ["mktrf", "smb", "hml", "rmw", "cma", "rf", "umd"]),
    ]
    write_report(downloads, qualities)
    with LOG_PATH.open("w") as fh:
        for item in downloads:
            fh.write(json.dumps({"type": "download", **item}) + "\n")
        for item in qualities:
            fh.write(json.dumps({"type": "quality", **item}) + "\n")
    print(f"Wrote {FF5_OUT.relative_to(ROOT)}")
    print(f"Wrote {MOM_OUT.relative_to(ROOT)}")
    print(f"Wrote {COMBINED_OUT.relative_to(ROOT)}")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
