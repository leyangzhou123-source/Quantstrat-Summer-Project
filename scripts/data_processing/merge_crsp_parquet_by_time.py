from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
CRSP_DIR = ROOT / "data" / "raw" / "crsp"
OUT_DIR = ROOT / "data" / "raw" / "crsp_merged"
REPORT_PATH = ROOT / "reports" / "crsp_merge_structure_report.md"
LOG_PATH = ROOT / "reports" / "crsp_merge_log.jsonl"


DATASETS = {
    "01_crsp_daily_stock_1925_1982": {
        "description": "62-column CRSP daily stock/security extract.",
        "date_col": "date",
        "files": ["o9cpuircioqtybnu.parquet"],
    },
    "02_ccm_linked_daily_1983_2022": {
        "description": "83-column CRSP/Compustat linked daily security extracts, arranged by source-period time order.",
        "date_col": "datadate",
        "files": [
            "cpglgfuthgeycbbq.parquet",
            "4vpqzv2cafyswsoi.parquet",
            "rct5mvx4bblpj9r9.parquet",
            "iywscyc3rpjgscbk.parquet",
        ],
        "excluded_files": {
            "mkzmnj8jjqf8ugmd.parquet": "Duplicate 1983-1989 slice already contained in cpglgfuthgeycbbq.parquet.",
        },
    },
    "03_ccm_link_bridge_2021": {
        "description": "11-column CRSP/Compustat link bridge for 2021; kept separate because it is not the same table as the 83-column linked daily extracts.",
        "date_col": "datadate",
        "files": ["ivm2y80t1mjgai1u.parquet"],
    },
}


def date_bounds(path: Path, date_col: str) -> tuple[str | None, str | None]:
    pf = pq.ParquetFile(path)
    mins: list[str] = []
    maxs: list[str] = []
    for row_group in range(pf.metadata.num_row_groups):
        arr = pc.cast(pf.read_row_group(row_group, columns=[date_col])[date_col], pa.string())
        arr = arr.drop_null()
        if len(arr) == 0:
            continue
        mins.append(pc.min(arr).as_py())
        maxs.append(pc.max(arr).as_py())
    return (min(mins), max(maxs)) if mins else (None, None)


def year_counts(path: Path, date_col: str) -> Counter:
    pf = pq.ParquetFile(path)
    counts: Counter = Counter()
    for row_group in range(pf.metadata.num_row_groups):
        arr = pc.cast(pf.read_row_group(row_group, columns=[date_col])[date_col], pa.string())
        years = pc.utf8_slice_codeunits(arr, 0, 4)
        for item in pc.value_counts(years).to_pylist():
            counts[str(item["values"])] += int(item["counts"])
    return counts


def profile_file(path: Path) -> dict:
    pf = pq.ParquetFile(path)
    cols = pf.schema_arrow.names
    date_col = "datadate" if "datadate" in cols else "date" if "date" in cols else None
    mn, mx = date_bounds(path, date_col) if date_col else (None, None)
    return {
        "file": path.name,
        "rows": pf.metadata.num_rows,
        "columns": len(cols),
        "column_names": cols,
        "schema": str(pf.schema_arrow),
        "date_col": date_col,
        "date_min": mn,
        "date_max": mx,
        "year_counts": dict(sorted(year_counts(path, date_col).items())) if date_col else {},
    }


def assemble_dataset(name: str, spec: dict, profiles_by_file: dict[str, dict]) -> dict:
    out_path = OUT_DIR / name
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    copied = []
    total_rows = 0
    all_years: Counter = Counter()
    for idx, file_name in enumerate(spec["files"], start=1):
        profile = profiles_by_file[file_name]
        start = str(profile["date_min"]).replace("-", "")
        end = str(profile["date_max"]).replace("-", "")
        target = out_path / f"{idx:02d}_{start}_{end}_{file_name}"
        shutil.copy2(CRSP_DIR / file_name, target)
        copied.append(
            {
                "source": file_name,
                "target": str(target.relative_to(ROOT)),
                "rows": profile["rows"],
                "date_min": profile["date_min"],
                "date_max": profile["date_max"],
                "columns": profile["columns"],
            }
        )
        total_rows += profile["rows"]
        all_years.update(profile["year_counts"])

    return {
        "dataset": name,
        "description": spec["description"],
        "output": str(out_path.relative_to(ROOT)),
        "rows": total_rows,
        "date_min": copied[0]["date_min"] if copied else None,
        "date_max": copied[-1]["date_max"] if copied else None,
        "columns": copied[0]["columns"] if copied else 0,
        "copied_files": copied,
        "year_counts": dict(sorted(all_years.items())),
        "excluded_files": spec.get("excluded_files", {}),
    }


def write_report(raw_profiles: list[dict], results: list[dict]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CRSP merge and structure report",
        "",
        "The raw CRSP folder contains multiple extracts with different schemas. I merged compatible files into chronological parquet dataset folders and kept incompatible tables separate to avoid mixing unlike variables.",
        "",
        "## Raw file structure",
        "",
        "| file | rows | columns | date column | date min | date max |",
        "|---|---:|---:|---|---|---|",
    ]
    for p in raw_profiles:
        lines.append(
            f"| `{p['file']}` | {p['rows']:,} | {p['columns']} | `{p['date_col']}` | {p['date_min']} | {p['date_max']} |"
        )

    lines.extend(["", "## Merged/arranged outputs", ""])
    for result in results:
        lines.extend(
            [
                f"### `{result['dataset']}`",
                "",
                result["description"],
                "",
                f"- Output folder: `{result['output']}`",
                f"- Rows: {result['rows']:,}",
                f"- Date range: {result['date_min']} to {result['date_max']}",
                f"- Common columns within this output: {result['columns']}",
                "- Time arrangement: file names are prefixed in chronological order and include their date range.",
                "",
                "| order | source file | output file | rows | date min | date max |",
                "|---:|---|---|---:|---|---|",
            ]
        )
        for idx, copied in enumerate(result["copied_files"], start=1):
            lines.append(
                f"| {idx} | `{copied['source']}` | `{copied['target']}` | {copied['rows']:,} | {copied['date_min']} | {copied['date_max']} |"
            )
        if result["excluded_files"]:
            lines.extend(["", "Excluded files:"])
            for file_name, reason in result["excluded_files"].items():
                lines.append(f"- `{file_name}`: {reason}")
        lines.extend(["", "Year coverage:"])
        lines.append(", ".join(f"{year}: {rows:,}" for year, rows in result["year_counts"].items()))
        lines.append("")

    lines.extend(
        [
            "## Important structure notes",
            "",
            "- `01_crsp_daily_stock_1925_1982` is the actual CRSP daily stock/security extract with CRSP identifiers such as `PERMNO`, `PERMCO`, `RET`, `RETX`, `PRC`, `VOL`, `SHROUT`, and delisting fields.",
            "- `02_ccm_linked_daily_1983_2022` uses linked Compustat/CRSP identifiers such as `GVKEY`, `LPERMNO`, `LPERMCO`, `datadate`, `prccd`, `cshtrd`, dividends, company descriptors, SIC/GICS/NAICS fields, and link metadata.",
            "- `03_ccm_link_bridge_2021` is a compact 2021 bridge, not a full daily return/price file. It fills the calendar gap in link information but does not replace a full 2021 daily security extract.",
            "- `mkzmnj8jjqf8ugmd.parquet` duplicates the 1983-1989 part of `cpglgfuthgeycbbq.parquet`, so it is excluded from the arranged merge to prevent double counting.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_profiles = [profile_file(path) for path in sorted(CRSP_DIR.glob("*.parquet"))]
    profiles_by_file = {p["file"]: p for p in raw_profiles}
    results = [assemble_dataset(name, spec, profiles_by_file) for name, spec in DATASETS.items()]

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w") as fh:
        for profile in raw_profiles:
            fh.write(json.dumps({"type": "raw_profile", **profile}) + "\n")
        for result in results:
            fh.write(json.dumps({"type": "merged_output", **result}) + "\n")

    write_report(raw_profiles, results)
    print(f"Wrote arranged CRSP parquet datasets under {OUT_DIR.relative_to(ROOT)}")
    print(f"Wrote structure report to {REPORT_PATH.relative_to(ROOT)}")
    print(f"Wrote machine-readable log to {LOG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
