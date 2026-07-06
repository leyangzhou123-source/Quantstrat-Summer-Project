from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import openassetpricing as oap
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_OUT = ROOT / "data" / "external" / "osap_missing_proxy_signals.parquet"
CHAR_DIR = ROOT / "data" / "raw" / "02_stock_characteristics"
RETURNS_PATH = ROOT / "data" / "raw" / "01_stock_returns.parquet"
REPORT_PATH = ROOT / "reports" / "osap_missing_characteristics_download_report.md"
REMAINING_PATH = ROOT / "reports" / "paper_missing_characteristics_remaining_after_osap.txt"
LOG_PATH = ROOT / "reports" / "osap_missing_characteristics_download_log.jsonl"


MISSING_21 = [
    "currat",
    "depr",
    "pchcapx_ia",
    "pchcurrat",
    "pchdepr",
    "pchgm_pchsale",
    "pchquick",
    "pchsale_pchinvt",
    "pchsale_pchrect",
    "pchsale_pchxsga",
    "pchsaleinv",
    "quick",
    "roavol",
    "roic",
    "salecash",
    "salerec",
    "secured",
    "securedind",
    "stdacc",
    "stdcf",
    "tb",
]


EXACT_ATTEMPTS = [
    "currat",
    "depr",
    "pchcurrat",
    "pchdepr",
    "pchgm_pchsale",
    "pchquick",
    "pchsaleinv",
    "quick",
    "roavol",
    "roic",
    "salecash",
    "salerec",
    "secured",
    "securedind",
]


PROXY_DOWNLOADS = {
    "pchsale_pchinvt": "GrSaleToGrInv",
    "pchsale_pchxsga": "GrSaleToGrOverhead",
    "pchcapx_ia": "grcapx",
}


def save_feature(df: pd.DataFrame, feature: str, source_col: str) -> dict:
    out = (
        df[["permno", "yyyymm", source_col]]
        .rename(columns={source_col: feature})
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[feature])
        .copy()
    )
    out["permno"] = out["permno"].astype("int64")
    out["yyyymm"] = out["yyyymm"].astype("int64")
    out = out.sort_values(["yyyymm", "permno"]).reset_index(drop=True)
    path = CHAR_DIR / f"{feature}.parquet"
    out.to_parquet(path, index=False)
    return {
        "feature": feature,
        "source": source_col,
        "path": str(path.relative_to(ROOT)),
        "rows": int(len(out)),
        "yyyymm_min": int(out["yyyymm"].min()) if len(out) else None,
        "yyyymm_max": int(out["yyyymm"].max()) if len(out) else None,
        "permnos": int(out["permno"].nunique()) if len(out) else 0,
    }


def derive_pchcapx_ia(proxy_df: pd.DataFrame) -> dict:
    grcapx = proxy_df[["permno", "yyyymm", "grcapx"]].dropna(subset=["grcapx"]).copy()
    returns = pd.read_parquet(RETURNS_PATH, columns=["permno", "month", "siccd"])
    returns["yyyymm"] = pd.to_datetime(returns["month"]).dt.year * 100 + pd.to_datetime(returns["month"]).dt.month
    returns["sic2"] = np.floor(pd.to_numeric(returns["siccd"], errors="coerce") / 100.0)
    sic = returns[["permno", "yyyymm", "sic2"]].drop_duplicates()
    merged = grcapx.merge(sic, on=["permno", "yyyymm"], how="inner")
    merged["pchcapx_ia"] = merged["grcapx"] - merged.groupby(["yyyymm", "sic2"])["grcapx"].transform("mean")
    return save_feature(merged, "pchcapx_ia", "pchcapx_ia")


def main() -> None:
    EXTERNAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    CHAR_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    openap = oap.OpenAP()

    exact_results = []
    for signal in EXACT_ATTEMPTS:
        df = openap.dl_signal("pandas", [signal])
        exact_results.append(
            {
                "signal": signal,
                "download_rows": int(len(df)),
                "status": "not_available_as_firm_signal" if len(df) == 0 else "downloaded",
            }
        )

    proxy_df = openap.dl_signal("pandas", list(PROXY_DOWNLOADS.values()))
    proxy_df.to_parquet(EXTERNAL_OUT, index=False)

    created = []
    created.append(save_feature(proxy_df, "pchsale_pchinvt", "GrSaleToGrInv"))
    created.append(save_feature(proxy_df, "pchsale_pchxsga", "GrSaleToGrOverhead"))
    created.append(derive_pchcapx_ia(proxy_df))

    filled = {item["feature"] for item in created}
    remaining = [x for x in MISSING_21 if x not in filled]
    REMAINING_PATH.write_text("\n".join(remaining) + "\n")

    lines = [
        "# OSAP missing-characteristic download report",
        "",
        "Downloaded the firm-level OSAP signals that are actually available through the `openassetpricing` downloader and used them to fill compatible GKY paper acronyms.",
        "",
        f"Raw OSAP proxy download: `{EXTERNAL_OUT.relative_to(ROOT)}`",
        "",
        "## Created characteristic files",
        "",
        "| GKY characteristic | OSAP source / construction | rows | date min | date max | permnos |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in created:
        source = item["source"]
        if item["feature"] == "pchcapx_ia":
            source = "industry-adjusted `grcapx`: grcapx minus same-month SIC2 mean"
        lines.append(
            f"| `{item['feature']}` | {source} | {item['rows']:,} | {item['yyyymm_min']} | {item['yyyymm_max']} | {item['permnos']:,} |"
        )

    lines.extend(
        [
            "",
            "## Exact OSAP attempts that were not downloadable as firm-level signals",
            "",
            "| signal | result | rows |",
            "|---|---|---:|",
        ]
    )
    for item in exact_results:
        lines.append(f"| `{item['signal']}` | `{item['status']}` | {item['download_rows']:,} |")

    lines.extend(["", "## Remaining missing characteristics", ""])
    for signal in remaining:
        lines.append(f"- `{signal}`")
    lines.append("")
    lines.append(
        "The remaining signals are accounting/placebo signals in OSAP documentation but are not exposed by the standard firm-level signal downloader. They need WRDS/Compustat construction or a separate source that distributes placebo firm-level signals."
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n")
    with LOG_PATH.open("w") as fh:
        for item in exact_results:
            fh.write(json.dumps({"type": "exact_attempt", **item}) + "\n")
        for item in created:
            fh.write(json.dumps({"type": "created", **item}) + "\n")
        for signal in remaining:
            fh.write(json.dumps({"type": "remaining", "signal": signal}) + "\n")

    print(f"Wrote {EXTERNAL_OUT.relative_to(ROOT)}")
    print(f"Created {len(created)} characteristic files")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")
    print(f"Wrote {REMAINING_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
