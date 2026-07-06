from __future__ import annotations

import json
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[2]
CHAR_DIR = ROOT / "data" / "raw" / "02_stock_characteristics"
RETURNS_PATH = ROOT / "data" / "raw" / "01_stock_returns.parquet"
OUT_PATH = ROOT / "data" / "processed" / "paper_73_characteristics_ranked.parquet"
REPORT_PATH = ROOT / "reports" / "paper_73_characteristics_quality_report.md"
MISSING_PATH = ROOT / "reports" / "paper_21_missing_characteristics.txt"
LOG_PATH = ROOT / "reports" / "paper_73_characteristics_quality_log.jsonl"


PAPER_94 = [
    "absacc",
    "acc",
    "aeavol",
    "age",
    "agr",
    "baspread",
    "beta",
    "betasq",
    "bm",
    "bm_ia",
    "cash",
    "cashdebt",
    "cashpr",
    "cfp",
    "cfp_ia",
    "chatoia",
    "chcsho",
    "chempia",
    "chinv",
    "chmom",
    "chpmia",
    "chtx",
    "cinvest",
    "convind",
    "currat",
    "depr",
    "divi",
    "divo",
    "dolvol",
    "dy",
    "ear",
    "egr",
    "ep",
    "gma",
    "grcapx",
    "grltnoa",
    "herf",
    "hire",
    "idiovol",
    "ill",
    "indmom",
    "invest",
    "lev",
    "lgr",
    "maxret",
    "mom12m",
    "mom1m",
    "mom36m",
    "mom6m",
    "ms",
    "mvel1",
    "mve_ia",
    "nincr",
    "operprof",
    "orgcap",
    "pchcapx_ia",
    "pchcurrat",
    "pchdepr",
    "pchgm_pchsale",
    "pchquick",
    "pchsale_pchinvt",
    "pchsale_pchrect",
    "pchsale_pchxsga",
    "pchsaleinv",
    "pctacc",
    "pricedelay",
    "ps",
    "quick",
    "rd",
    "rd_mve",
    "rd_sale",
    "realestate",
    "retvol",
    "roaq",
    "roavol",
    "roeq",
    "roic",
    "rsup",
    "salecash",
    "saleinv",
    "salerec",
    "secured",
    "securedind",
    "sgr",
    "sin",
    "sp",
    "std_dolvol",
    "std_turn",
    "stdacc",
    "stdcf",
    "tang",
    "tb",
    "turn",
    "zerotrade",
]


SOURCE_MAP = {
    "absacc": "AbnormalAccruals",
    "acc": "Accruals",
    "aeavol": "EarningsSurprise",
    "age": "FirmAge",
    "agr": "AssetGrowth",
    "baspread": "BidAskSpread",
    "beta": "Beta",
    "betasq": "betasq",
    "bm": "BM",
    "bm_ia": "BM",
    "cash": "Cash",
    "cashdebt": "NetDebtPrice",
    "cashpr": "CashProd",
    "cfp": "cfp",
    "cfp_ia": "cfp",
    "chatoia": "ChAssetTurnover",
    "chcsho": "ShareIss1Y",
    "chempia": "hire",
    "chinv": "ChInv",
    "chmom": "chmom",
    "chpmia": "AOP",
    "chtx": "ChTax",
    "cinvest": "Investment",
    "convind": "ConvDebt",
    "divi": "DivInit",
    "divo": "DivOmit",
    "dolvol": "DolVol",
    "dy": "DivYieldST",
    "ear": "AnnouncementReturn",
    "egr": "ChEQ",
    "ep": "EP",
    "gma": "GP",
    "grcapx": "grcapx",
    "grltnoa": "GrLTNOA",
    "herf": "Herf",
    "hire": "hire",
    "idiovol": "IdioVol3F",
    "ill": "Illiquidity",
    "indmom": "IndMom",
    "invest": "Investment",
    "lev": "Leverage",
    "lgr": "GrLTNOA",
    "maxret": "MaxRet",
    "mom12m": "Mom12m",
    "mom1m": "mom1m",
    "mom36m": "mom36m",
    "mom6m": "Mom6m",
    "ms": "MS",
    "mvel1": "mvel1",
    "mve_ia": "mve_ia",
    "nincr": "NumEarnIncrease",
    "operprof": "OperProf",
    "orgcap": "OrgCap",
    "pctacc": "PctAcc",
    "pricedelay": "PriceDelayRsq",
    "ps": "PS",
    "rd": "RD",
    "rd_mve": "RDcap",
    "rd_sale": "RDS",
    "realestate": "realestate",
    "retvol": "retvol",
    "roaq": "roaq",
    "roeq": "RoE",
    "rsup": "RevenueSurprise",
    "saleinv": "GrSaleToGrInv",
    "sgr": "GrSaleToGrOverhead",
    "sin": "sinAlgo",
    "sp": "SP",
    "std_dolvol": "std_dolvol",
    "std_turn": "std_turn",
    "tang": "tang",
    "turn": "turn",
    "zerotrade": "zerotrade12M",
}


MISSING_21 = [char for char in PAPER_94 if char not in SOURCE_MAP]


def monthly_base() -> pl.DataFrame:
    return (
        pl.scan_parquet(RETURNS_PATH)
        .select(
            pl.col("permno").cast(pl.Int64),
            (
                pl.col("month").dt.year() * 100 + pl.col("month").dt.month()
            ).cast(pl.Int64).alias("yyyymm"),
        )
        .unique()
        .sort(["yyyymm", "permno"])
        .collect()
    )


def load_feature(paper_name: str, source_name: str) -> tuple[pl.DataFrame, dict]:
    path = CHAR_DIR / f"{source_name}.parquet"
    df = pl.read_parquet(path)
    value_cols = [c for c in df.columns if c not in {"permno", "yyyymm"}]
    if len(value_cols) != 1:
        raise ValueError(f"{path} should have exactly one value column, found {value_cols}")
    value_col = value_cols[0]
    duplicates = (
        df.group_by(["permno", "yyyymm"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    df = (
        df.select(
            pl.col("permno").cast(pl.Int64),
            pl.col("yyyymm").cast(pl.Int64),
            pl.col(value_col).cast(pl.Float64, strict=False).alias(paper_name),
        )
        .group_by(["permno", "yyyymm"])
        .agg(pl.col(paper_name).mean())
        .sort(["yyyymm", "permno"])
    )
    profile = {
        "paper_characteristic": paper_name,
        "source_file": f"{source_name}.parquet",
        "source_value_column": value_col,
        "source_rows": int(df.height),
        "source_yyyymm_min": int(df["yyyymm"].min()) if df.height else None,
        "source_yyyymm_max": int(df["yyyymm"].max()) if df.height else None,
        "duplicate_keys_collapsed": int(duplicates),
    }
    return df, profile


def rank_columns(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    exprs = []
    for col in cols:
        filled = pl.col(col).fill_null(pl.col(col).median().over("yyyymm"))
        n = filled.count().over("yyyymm")
        rank = filled.rank(method="average").over("yyyymm")
        ranked = (
            pl.when(n <= 1)
            .then(0.0)
            .otherwise(-1.0 + 2.0 * ((rank - 1.0) / (n - 1.0)))
            .cast(pl.Float32)
            .alias(col)
        )
        exprs.append(ranked)
    return df.with_columns(exprs).sort(["yyyymm", "permno"])


def quality_metrics(raw: pl.DataFrame, ranked: pl.DataFrame, cols: list[str]) -> dict:
    total_rows = raw.height
    metrics = []
    for col in cols:
        missing = raw[col].null_count()
        ranked_missing = ranked[col].null_count()
        min_val = ranked[col].min()
        max_val = ranked[col].max()
        metrics.append(
            {
                "characteristic": col,
                "raw_missing": int(missing),
                "raw_missing_rate": float(missing / total_rows),
                "ranked_missing": int(ranked_missing),
                "ranked_min": float(min_val) if min_val is not None else None,
                "ranked_max": float(max_val) if max_val is not None else None,
            }
        )

    first = ranked.select(["yyyymm", "permno"]).head(50_000)
    sorted_check = first.equals(first.sort(["yyyymm", "permno"]))
    return {
        "rows": int(ranked.height),
        "columns": int(len(ranked.columns)),
        "characteristic_count": len(cols),
        "yyyymm_min": int(ranked["yyyymm"].min()),
        "yyyymm_max": int(ranked["yyyymm"].max()),
        "permno_count": int(ranked["permno"].n_unique()),
        "is_sorted_sample_check": bool(sorted_check),
        "metrics": metrics,
    }


def write_report(profiles: list[dict], quality: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    worst_missing = sorted(quality["metrics"], key=lambda x: x["raw_missing_rate"], reverse=True)[:15]
    out_of_bounds = [
        m
        for m in quality["metrics"]
        if m["ranked_min"] is not None
        and (m["ranked_min"] < -1.0001 or m["ranked_max"] > 1.0001)
    ]

    lines = [
        "# Paper 73 characteristic panel quality report",
        "",
        f"Output file: `{OUT_PATH.relative_to(ROOT)}`",
        "",
        "The file contains the 73 currently covered characteristics from the 94-characteristic list in Gu, Kelly, and Xiu (2020), using paper acronyms as columns.",
        "",
        "Each characteristic is median-imputed within month and cross-sectionally ranked within `yyyymm` to the interval `[-1, 1]`. Rows are sorted by `yyyymm`, then `permno`.",
        "",
        "## Panel shape",
        "",
        f"- Rows: {quality['rows']:,}",
        f"- Columns: {quality['columns']} (`permno`, `yyyymm`, plus {quality['characteristic_count']} characteristics)",
        f"- Date range: {quality['yyyymm_min']} to {quality['yyyymm_max']}",
        f"- Unique permnos: {quality['permno_count']:,}",
        f"- Time/order check on sample: {quality['is_sorted_sample_check']}",
        f"- Ranked value bounds violations: {len(out_of_bounds)}",
        "",
        "## Source crosswalk",
        "",
        "| paper acronym | source file | source value column | rows | date min | date max | duplicate keys collapsed |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for p in profiles:
        lines.append(
            f"| `{p['paper_characteristic']}` | `{p['source_file']}` | `{p['source_value_column']}` | {p['source_rows']:,} | {p['source_yyyymm_min']} | {p['source_yyyymm_max']} | {p['duplicate_keys_collapsed']:,} |"
        )

    lines.extend(
        [
            "",
            "## Highest raw missing rates before monthly median imputation",
            "",
            "| characteristic | raw missing rate | raw missing rows | ranked missing rows | ranked min | ranked max |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for m in worst_missing:
        lines.append(
            f"| `{m['characteristic']}` | {m['raw_missing_rate']:.2%} | {m['raw_missing']:,} | {m['ranked_missing']:,} | {m['ranked_min']:.4f} | {m['ranked_max']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Remaining 21 missing characteristics",
            "",
        ]
    )
    for char in MISSING_21:
        lines.append(f"- `{char}`")

    REPORT_PATH.write_text("\n".join(lines) + "\n")
    MISSING_PATH.write_text("\n".join(MISSING_21) + "\n")


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    covered = [char for char in PAPER_94 if char in SOURCE_MAP]
    if len(covered) != 73:
        raise ValueError(f"Expected 73 covered characteristics, found {len(covered)}")
    if len(MISSING_21) != 21:
        raise ValueError(f"Expected 21 missing characteristics, found {len(MISSING_21)}")

    panel = monthly_base()
    profiles = []
    for paper_name in covered:
        feature, profile = load_feature(paper_name, SOURCE_MAP[paper_name])
        profiles.append(profile)
        panel = panel.join(feature, on=["permno", "yyyymm"], how="left")

    ranked = rank_columns(panel, covered)
    ranked.write_parquet(OUT_PATH, compression="zstd")
    quality = quality_metrics(panel, ranked, covered)
    write_report(profiles, quality)

    with LOG_PATH.open("w") as fh:
        fh.write(json.dumps({"type": "quality", **{k: v for k, v in quality.items() if k != "metrics"}}) + "\n")
        for item in profiles:
            fh.write(json.dumps({"type": "source", **item}) + "\n")
        for item in quality["metrics"]:
            fh.write(json.dumps({"type": "metric", **item}) + "\n")
        for char in MISSING_21:
            fh.write(json.dumps({"type": "missing", "characteristic": char}) + "\n")

    print(f"Wrote {OUT_PATH.relative_to(ROOT)}")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")
    print(f"Wrote {MISSING_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
