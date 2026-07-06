from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RETURNS_PATH = ROOT / "data" / "raw" / "01_stock_returns.parquet"
CHAR_DIR = ROOT / "data" / "raw" / "02_stock_characteristics"
REPORT_PATH = ROOT / "reports" / "missing_crsp_characteristics_report.md"
LOG_PATH = ROOT / "reports" / "missing_crsp_characteristics_log.jsonl"


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


EXISTING_MAPPING = {
    "absacc": "AbnormalAccruals",
    "acc": "Accruals",
    "aeavol": "EarningsSurprise",
    "age": "FirmAge",
    "agr": "AssetGrowth",
    "baspread": "BidAskSpread",
    "beta": "Beta",
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
    "mom1m": "MomRev",
    "mom36m": "LRreversal",
    "mom6m": "Mom6m",
    "ms": "MS",
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
    "retvol": "RealizedVol",
    "roaq": "roaq",
    "roeq": "RoE",
    "rsup": "RevenueSurprise",
    "saleinv": "GrSaleToGrInv",
    "sgr": "GrSaleToGrOverhead",
    "sin": "sinAlgo",
    "sp": "SP",
    "std_turn": "std_turn",
    "tang": "tang",
    "turn": "ShareVol",
    "zerotrade": "zerotrade12M",
}


CRSP_DERIVED = {
    "mvel1": "log market equity: log(abs(prc) * shrout)",
    "mve_ia": "industry-adjusted size: mvel1 minus same-month SIC2 mean mvel1",
    "mom1m": "one-month momentum/reversal: current-month return",
    "mom36m": "long-term momentum/reversal: cumulative return from t-36 through t-13",
    "chmom": "change in six-month momentum: mom6m minus mom6m lagged six months",
    "turn": "share turnover: monthly volume divided by shares outstanding",
    "std_dolvol": "36-month rolling standard deviation of log dollar volume",
    "retvol": "12-month rolling standard deviation of monthly returns",
}


DERIVED_FROM_EXISTING = {
    "betasq": "Beta squared: Beta^2 from data/raw/02_stock_characteristics/Beta.parquet",
}


def yyyymm_from_month(month: pd.Series) -> pd.Series:
    dt = pd.to_datetime(month)
    return (dt.dt.year * 100 + dt.dt.month).astype("int64")


def rolling_prod_minus_one(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    return (1.0 + s).rolling(window=window, min_periods=min_periods).apply(np.prod, raw=True) - 1.0


def write_feature(df: pd.DataFrame, name: str, values: pd.Series) -> dict:
    out = pd.DataFrame(
        {
            "permno": df["permno"].astype("int64"),
            "yyyymm": df["yyyymm"].astype("int64"),
            name: pd.to_numeric(values, errors="coerce"),
        }
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=[name])
    out = out.sort_values(["yyyymm", "permno"]).reset_index(drop=True)
    path = CHAR_DIR / f"{name}.parquet"
    out.to_parquet(path, index=False)
    return {
        "feature": name,
        "path": str(path.relative_to(ROOT)),
        "rows": int(len(out)),
        "yyyymm_min": int(out["yyyymm"].min()) if len(out) else None,
        "yyyymm_max": int(out["yyyymm"].max()) if len(out) else None,
        "permno_count": int(out["permno"].nunique()) if len(out) else 0,
    }


def build_from_monthly_crsp() -> list[dict]:
    cols = ["permno", "month", "ret", "prc", "shrout", "vol", "siccd", "me"]
    df = pd.read_parquet(RETURNS_PATH, columns=cols)
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values(["permno", "month"]).reset_index(drop=True)
    df["yyyymm"] = yyyymm_from_month(df["month"])
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    df["prc"] = pd.to_numeric(df["prc"], errors="coerce").abs()
    df["shrout"] = pd.to_numeric(df["shrout"], errors="coerce")
    df["vol"] = pd.to_numeric(df["vol"], errors="coerce")
    df["me_raw"] = pd.to_numeric(df["me"], errors="coerce")
    df["me_calc"] = df["me_raw"].where(df["me_raw"].notna(), df["prc"] * df["shrout"])
    df["sic2"] = np.floor(pd.to_numeric(df["siccd"], errors="coerce") / 100.0)

    grouped = df.groupby("permno", sort=False)
    df["mvel1"] = np.log(df["me_calc"].where(df["me_calc"] > 0))
    df["mve_ia"] = df["mvel1"] - df.groupby(["yyyymm", "sic2"])["mvel1"].transform("mean")
    df["mom1m"] = df["ret"]
    df["mom6m_internal"] = grouped["ret"].transform(lambda s: rolling_prod_minus_one(s.shift(1), 6, 4))
    df["mom36m"] = grouped["ret"].transform(lambda s: rolling_prod_minus_one(s.shift(13), 24, 18))
    df["chmom"] = df["mom6m_internal"] - grouped["mom6m_internal"].shift(6)

    shares_outstanding = df["shrout"] * 1000.0
    df["turn"] = df["vol"] / shares_outstanding.where(shares_outstanding > 0)
    dollar_volume = df["prc"] * df["vol"]
    df["log_dolvol"] = np.log(dollar_volume.where(dollar_volume > 0))
    df["std_dolvol"] = grouped["log_dolvol"].transform(lambda s: s.rolling(36, min_periods=24).std())
    df["retvol"] = grouped["ret"].transform(lambda s: s.rolling(12, min_periods=8).std())

    results = []
    for name in CRSP_DERIVED:
        results.append(write_feature(df, name, df[name]))
    return results


def build_betasq() -> dict | None:
    beta_path = CHAR_DIR / "Beta.parquet"
    if not beta_path.exists():
        return None
    beta = pd.read_parquet(beta_path)
    value_col = [c for c in beta.columns if c not in {"permno", "yyyymm"}][0]
    beta["betasq"] = pd.to_numeric(beta[value_col], errors="coerce") ** 2
    out = beta[["permno", "yyyymm", "betasq"]].replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["betasq"]).sort_values(["yyyymm", "permno"]).reset_index(drop=True)
    path = CHAR_DIR / "betasq.parquet"
    out.to_parquet(path, index=False)
    return {
        "feature": "betasq",
        "path": str(path.relative_to(ROOT)),
        "rows": int(len(out)),
        "yyyymm_min": int(out["yyyymm"].min()) if len(out) else None,
        "yyyymm_max": int(out["yyyymm"].max()) if len(out) else None,
        "permno_count": int(out["permno"].nunique()) if len(out) else 0,
    }


def coverage_rows(created: list[dict]) -> list[dict]:
    files = {p.stem.lower(): p.stem for p in CHAR_DIR.glob("*.parquet")}
    created_names = {item["feature"] for item in created}
    rows = []
    for char in PAPER_94:
        exact = char in files
        mapped = EXISTING_MAPPING.get(char)
        created_now = char in created_names
        if created_now:
            status = "created_from_crsp"
            source = f"{char}.parquet"
        elif exact:
            status = "exact_file"
            source = f"{files[char]}.parquet"
        elif mapped and (CHAR_DIR / f"{mapped}.parquet").exists():
            status = "covered_by_existing_proxy"
            source = f"{mapped}.parquet"
        else:
            status = "not_covered_here"
            source = "Requires Compustat/accounting fields not present in filtered CRSP monthly file"
        rows.append({"paper_acronym": char, "status": status, "source_or_note": source})
    return rows


def write_report(created: list[dict], rows: list[dict]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    counts = pd.Series([r["status"] for r in rows]).value_counts().to_dict()
    lines = [
        "# Missing CRSP-derived stock characteristics",
        "",
        "This report compares the 94 stock characteristics in Table A.6 of Gu, Kelly, and Xiu (2020) to the current `data/raw/02_stock_characteristics` folder and creates the missing variables that can be built from the filtered monthly CRSP panel.",
        "",
        "Source used for new CRSP-derived variables: `data/raw/01_stock_returns.parquet`.",
        "",
        "## Created files",
        "",
        "| feature | formula/source | rows | yyyymm min | yyyymm max | permnos |",
        "|---|---|---:|---:|---:|---:|",
    ]
    formulas = {**CRSP_DERIVED, **DERIVED_FROM_EXISTING}
    for item in created:
        lines.append(
            f"| `{item['feature']}` | {formulas[item['feature']]} | {item['rows']:,} | {item['yyyymm_min']} | {item['yyyymm_max']} | {item['permno_count']:,} |"
        )

    lines.extend(
        [
            "",
            "## Coverage summary",
            "",
            f"- Exact paper-acronym files now present: {counts.get('exact_file', 0)}",
            f"- Newly created from CRSP in this run: {counts.get('created_from_crsp', 0)}",
            f"- Covered by existing proxy/renamed file: {counts.get('covered_by_existing_proxy', 0)}",
            f"- Still not covered by this folder/CRSP monthly data: {counts.get('not_covered_here', 0)}",
            "",
            "## 94-characteristic crosswalk",
            "",
            "| paper acronym | status | source or note |",
            "|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(f"| `{row['paper_acronym']}` | `{row['status']}` | {row['source_or_note']} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Variables such as `currat`, `quick`, `depr`, `secured`, `tb`, `stdacc`, and `stdcf` require Compustat balance-sheet, income-statement, or debt-detail fields and cannot be recovered from CRSP prices, returns, volume, and shares alone.",
            "- `mve_ia` uses two-digit SIC from the filtered CRSP monthly panel and subtracts the same-month SIC2 industry mean of `mvel1`.",
            "- `chmom`, `mom36m`, `std_dolvol`, and `retvol` are rolling monthly approximations from the available monthly CRSP file. Exact Green-Hand-Zhang definitions may differ if constructed from daily CRSP or original signal code.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    created = build_from_monthly_crsp()
    betasq = build_betasq()
    if betasq:
        created.append(betasq)

    rows = coverage_rows(created)
    write_report(created, rows)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w") as fh:
        for item in created:
            fh.write(json.dumps({"type": "created", **item}) + "\n")
        for row in rows:
            fh.write(json.dumps({"type": "coverage", **row}) + "\n")

    print(f"Created {len(created)} CRSP-derived characteristic files in {CHAR_DIR.relative_to(ROOT)}")
    print(f"Wrote report to {REPORT_PATH.relative_to(ROOT)}")
    print(f"Wrote log to {LOG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
