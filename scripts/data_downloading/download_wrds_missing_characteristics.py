from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import wrds


ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / ".env"
LINK_PATH = ROOT / "data" / "external" / "crsp_compustat_linking.parquet"
RETURNS_PATH = ROOT / "data" / "raw" / "01_stock_returns.parquet"
CHAR_DIR = ROOT / "data" / "raw" / "02_stock_characteristics"
RAW_DIR = ROOT / "data" / "external" / "wrds_compustat_missing_characteristics"
REPORT_PATH = ROOT / "reports" / "wrds_missing_characteristics_report.md"
LOG_PATH = ROOT / "reports" / "wrds_missing_characteristics_log.jsonl"
REMAINING_PATH = ROOT / "reports" / "paper_missing_characteristics_remaining_after_wrds.txt"


ANNUAL_SIGNALS = [
    "currat",
    "depr",
    "pchcurrat",
    "pchdepr",
    "pchgm_pchsale",
    "pchquick",
    "pchsale_pchrect",
    "pchsaleinv",
    "quick",
    "roic",
    "salecash",
    "salerec",
    "secured",
    "securedind",
    "stdacc",
    "stdcf",
    "tb",
]

QUARTERLY_SIGNALS = ["roavol"]


FORMULAS = {
    "currat": "act / lct",
    "depr": "dp / ppent",
    "pchcurrat": "one-year percentage change in currat",
    "pchdepr": "one-year percentage change in depr",
    "pchgm_pchsale": "percentage change in gross margin (sale - cogs) minus percentage change in sale",
    "pchquick": "one-year percentage change in quick",
    "pchsale_pchrect": "percentage change in sale minus percentage change in rect",
    "pchsaleinv": "one-year percentage change in sale / invt",
    "quick": "(act - invt) / lct",
    "roavol": "rolling 16-quarter standard deviation of roaq, where roaq = ibq / lag(atq)",
    "roic": "(ebit - nopi) / (ceq + lt - che)",
    "salecash": "sale / che",
    "salerec": "sale / rect",
    "secured": "dm / dltt",
    "securedind": "1 if secured > 0 else 0",
    "stdacc": "rolling 5-year standard deviation of annual accruals scaled by average assets",
    "stdcf": "rolling 5-year standard deviation of annual cash flow scaled by average assets",
    "tb": "tax income to book income approximation: ((txfed + txfo) / 0.35) / ib",
}


def load_env(path: Path = ENV_PATH) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    for key in ["WRDS_USERNAME", "WRDS_PASSWORD", "WRDS_HOST", "WRDS_PORT", "WRDS_DB"]:
        if os.environ.get(key):
            values[key] = os.environ[key]
    return values


def connect_wrds() -> wrds.Connection:
    env = load_env()
    username = env.get("WRDS_USERNAME", "")
    password = env.get("WRDS_PASSWORD", "")
    if not username or not password:
        raise RuntimeError(
            "Missing WRDS credentials. Fill .env with WRDS_USERNAME and WRDS_PASSWORD, "
            "then rerun: python scripts/download_wrds_missing_characteristics.py"
        )
    db = wrds.Connection(
        autoconnect=False,
        wrds_username=username,
        wrds_password=password,
        wrds_hostname=env.get("WRDS_HOST", "wrds-pgdata.wharton.upenn.edu"),
        wrds_port=int(env.get("WRDS_PORT", "9737")),
        wrds_dbname=env.get("WRDS_DB", "wrds"),
        wrds_connect_args={"sslmode": "require", "connect_timeout": int(env.get("WRDS_CONNECT_TIMEOUT", "60"))},
    )
    try:
        db._Connection__make_sa_engine_conn(raise_err=True)
        db.load_library_list()
    except Exception as exc:
        raise RuntimeError(
            "WRDS connection failed using the credentials in .env. "
            "Please verify WRDS_USERNAME/WRDS_PASSWORD and that this WRDS account has "
            "PostgreSQL access to Compustat/CRSP. The original database error is shown above."
        ) from exc
    return db


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.where(den.abs() > 1e-12)
    return num / den


def pct_change_by_gvkey(df: pd.DataFrame, col: str) -> pd.Series:
    lag = df.groupby("gvkey", sort=False)[col].shift(1)
    return safe_div(df[col], lag) - 1.0


def query_compustat(db: wrds.Connection) -> tuple[pd.DataFrame, pd.DataFrame]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    annual_sql = """
        select gvkey, datadate, fyear, fyr, indfmt, datafmt, popsrc, consol,
               at, act, lct, che, invt, dp, ppent, sale, cogs, rect, xsga,
               capx, ebit, nopi, ceq, lt, dltt, dm, ib, oancf, dlc, txp,
               txfed, txfo, txt
        from comp.funda
        where indfmt='INDL' and datafmt='STD' and popsrc='D' and consol='C'
          and datadate between '1950-01-01' and '2017-12-31'
    """
    quarterly_sql = """
        select gvkey, datadate, fyearq, fqtr, indfmt, datafmt, popsrc, consol,
               atq, ibq
        from comp.fundq
        where indfmt='INDL' and datafmt='STD' and popsrc='D' and consol='C'
          and datadate between '1950-01-01' and '2017-12-31'
    """
    annual = db.raw_sql(annual_sql, date_cols=["datadate"])
    quarterly = db.raw_sql(quarterly_sql, date_cols=["datadate"])
    annual.to_parquet(RAW_DIR / "comp_funda_missing_characteristics.parquet", index=False)
    quarterly.to_parquet(RAW_DIR / "comp_fundq_missing_characteristics.parquet", index=False)
    return annual, quarterly


def prepare_annual(annual: pd.DataFrame) -> pd.DataFrame:
    annual = annual.copy()
    annual["gvkey"] = annual["gvkey"].astype(str).str.zfill(6)
    annual = annual.sort_values(["gvkey", "datadate"]).reset_index(drop=True)
    numeric_cols = [c for c in annual.columns if c not in {"gvkey", "datadate", "indfmt", "datafmt", "popsrc", "consol"}]
    for col in numeric_cols:
        annual[col] = pd.to_numeric(annual[col], errors="coerce")

    annual["currat"] = safe_div(annual["act"], annual["lct"])
    annual["quick"] = safe_div(annual["act"] - annual["invt"], annual["lct"])
    annual["depr"] = safe_div(annual["dp"], annual["ppent"])
    annual["gm"] = annual["sale"] - annual["cogs"]
    annual["sale_to_invt"] = safe_div(annual["sale"], annual["invt"])
    annual["secured"] = safe_div(annual["dm"], annual["dltt"])
    annual["securedind"] = (annual["secured"] > 0).astype(float)
    annual["roic"] = safe_div(annual["ebit"] - annual["nopi"], annual["ceq"] + annual["lt"] - annual["che"])
    annual["salecash"] = safe_div(annual["sale"], annual["che"])
    annual["salerec"] = safe_div(annual["sale"], annual["rect"])
    annual["tb"] = safe_div((annual["txfed"].fillna(0) + annual["txfo"].fillna(0)) / 0.35, annual["ib"])

    annual["pchcurrat"] = pct_change_by_gvkey(annual, "currat")
    annual["pchquick"] = pct_change_by_gvkey(annual, "quick")
    annual["pchdepr"] = pct_change_by_gvkey(annual, "depr")
    annual["pchgm_pchsale"] = pct_change_by_gvkey(annual, "gm") - pct_change_by_gvkey(annual, "sale")
    annual["pchsale_pchrect"] = pct_change_by_gvkey(annual, "sale") - pct_change_by_gvkey(annual, "rect")
    annual["pchsaleinv"] = pct_change_by_gvkey(annual, "sale_to_invt")

    lag_at = annual.groupby("gvkey", sort=False)["at"].shift(1)
    avg_at = (annual["at"] + lag_at) / 2.0
    delta_act = annual.groupby("gvkey", sort=False)["act"].diff()
    delta_che = annual.groupby("gvkey", sort=False)["che"].diff()
    delta_lct = annual.groupby("gvkey", sort=False)["lct"].diff()
    delta_dlc = annual.groupby("gvkey", sort=False)["dlc"].diff()
    delta_txp = annual.groupby("gvkey", sort=False)["txp"].diff()
    accrual_component = delta_act - delta_che - delta_lct + delta_dlc + delta_txp - annual["dp"]
    annual["acc_for_std"] = safe_div(accrual_component, avg_at)
    annual["cf_for_std"] = safe_div(annual["ib"] - accrual_component, avg_at)
    annual["stdacc"] = annual.groupby("gvkey", sort=False)["acc_for_std"].transform(lambda x: x.rolling(5, min_periods=3).std())
    annual["stdcf"] = annual.groupby("gvkey", sort=False)["cf_for_std"].transform(lambda x: x.rolling(5, min_periods=3).std())

    annual["available_date"] = annual["datadate"] + pd.DateOffset(months=6) + pd.offsets.MonthEnd(0)
    return annual


def prepare_quarterly(quarterly: pd.DataFrame) -> pd.DataFrame:
    quarterly = quarterly.copy()
    quarterly["gvkey"] = quarterly["gvkey"].astype(str).str.zfill(6)
    quarterly = quarterly.sort_values(["gvkey", "datadate"]).reset_index(drop=True)
    quarterly["atq"] = pd.to_numeric(quarterly["atq"], errors="coerce")
    quarterly["ibq"] = pd.to_numeric(quarterly["ibq"], errors="coerce")
    lag_atq = quarterly.groupby("gvkey", sort=False)["atq"].shift(1)
    quarterly["roaq"] = safe_div(quarterly["ibq"], lag_atq)
    quarterly["roavol"] = quarterly.groupby("gvkey", sort=False)["roaq"].transform(lambda x: x.rolling(16, min_periods=8).std())
    quarterly["available_date"] = quarterly["datadate"] + pd.DateOffset(months=4) + pd.offsets.MonthEnd(0)
    return quarterly


def link_to_permno(obs: pd.DataFrame, signal_cols: list[str]) -> pd.DataFrame:
    link = pd.read_parquet(LINK_PATH)
    link["gvkey"] = link["gvkey"].astype(str).str.zfill(6)
    link["linkdt"] = pd.to_datetime(link["linkdt"])
    link["linkenddt_filled"] = pd.to_datetime(link["linkenddt_filled"])
    link = link[link["lpermno"].notna()].copy()
    link["permno"] = link["lpermno"].astype("int64")
    link = link[link["LINKTYPE"].isin(["LC", "LU"]) & link["LINKPRIM"].isin(["P", "C"])]

    keep = ["gvkey", "available_date"] + signal_cols
    merged = obs[keep].merge(link[["gvkey", "permno", "linkdt", "linkenddt_filled"]], on="gvkey", how="inner")
    merged = merged[
        (merged["available_date"] >= merged["linkdt"])
        & (merged["available_date"] <= merged["linkenddt_filled"])
    ].copy()
    merged["yyyymm"] = merged["available_date"].dt.year * 100 + merged["available_date"].dt.month
    return merged[["permno", "yyyymm"] + signal_cols]


def expand_to_monthly(linked_obs: pd.DataFrame, signal: str, base: pd.DataFrame) -> pd.DataFrame:
    obs = linked_obs[["permno", "yyyymm", signal]].dropna(subset=[signal]).copy()
    obs = obs.replace([np.inf, -np.inf], np.nan).dropna(subset=[signal])
    obs = obs.groupby(["permno", "yyyymm"], as_index=False)[signal].mean()
    obs = obs.sort_values(["permno", "yyyymm"])
    obs_groups = {permno: group for permno, group in obs.groupby("permno", sort=False)}

    pieces = []
    for permno, base_g in base.groupby("permno", sort=False):
        obs_g = obs_groups.get(permno)
        if obs_g is None:
            continue
        if obs_g.empty:
            continue
        merged = pd.merge_asof(
            base_g.sort_values("yyyymm"),
            obs_g.sort_values("yyyymm"),
            on="yyyymm",
            by="permno",
            direction="backward",
        )
        pieces.append(merged[["permno", "yyyymm", signal]])
    if not pieces:
        return pd.DataFrame(columns=["permno", "yyyymm", signal])
    out = pd.concat(pieces, ignore_index=True)
    out = out.dropna(subset=[signal]).sort_values(["yyyymm", "permno"]).reset_index(drop=True)
    return out


def save_signal(df: pd.DataFrame, signal: str) -> dict:
    path = CHAR_DIR / f"{signal}.parquet"
    df.to_parquet(path, index=False)
    return {
        "signal": signal,
        "path": str(path.relative_to(ROOT)),
        "rows": int(len(df)),
        "yyyymm_min": int(df["yyyymm"].min()) if len(df) else None,
        "yyyymm_max": int(df["yyyymm"].max()) if len(df) else None,
        "permnos": int(df["permno"].nunique()) if len(df) else 0,
        "formula": FORMULAS[signal],
    }


def write_report(results: list[dict], remaining: list[str]) -> None:
    lines = [
        "# WRDS missing-characteristics construction report",
        "",
        "Constructed remaining Compustat-accounting characteristics from WRDS `comp.funda` and `comp.fundq`, linked to CRSP `permno` using `data/external/crsp_compustat_linking.parquet`.",
        "",
        "## Created files",
        "",
        "| signal | rows | date min | date max | permnos | formula |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in results:
        lines.append(
            f"| `{item['signal']}` | {item['rows']:,} | {item['yyyymm_min']} | {item['yyyymm_max']} | {item['permnos']:,} | {item['formula']} |"
        )
    lines.extend(["", "## Remaining", ""])
    if remaining:
        for signal in remaining:
            lines.append(f"- `{signal}`")
    else:
        lines.append("No requested WRDS characteristics remain unbuilt.")
    lines.extend(
        [
            "",
            "## Formula caveats",
            "",
            "- `stdacc`, `stdcf`, and `tb` are constructed from common accounting definitions but should be verified against the original Green-Hand-Zhang SAS code before final replication tables.",
            "- Annual variables are made available six months after fiscal `datadate`; quarterly `roavol` is made available four months after fiscal quarter `datadate`.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    CHAR_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = connect_wrds()
    try:
        annual_raw, quarterly_raw = query_compustat(db)
    finally:
        db.close()

    annual = prepare_annual(annual_raw)
    quarterly = prepare_quarterly(quarterly_raw)
    annual_linked = link_to_permno(annual, ANNUAL_SIGNALS)
    quarterly_linked = link_to_permno(quarterly, QUARTERLY_SIGNALS)

    base = pd.read_parquet(RETURNS_PATH, columns=["permno", "month"])
    base["permno"] = base["permno"].astype("int64")
    base["month"] = pd.to_datetime(base["month"])
    base["yyyymm"] = base["month"].dt.year * 100 + base["month"].dt.month
    base = base[["permno", "yyyymm"]].drop_duplicates().sort_values(["permno", "yyyymm"])

    results = []
    for signal in ANNUAL_SIGNALS:
        out = expand_to_monthly(annual_linked, signal, base)
        results.append(save_signal(out, signal))
    for signal in QUARTERLY_SIGNALS:
        out = expand_to_monthly(quarterly_linked, signal, base)
        results.append(save_signal(out, signal))

    remaining = [item["signal"] for item in results if item["rows"] == 0]
    REMAINING_PATH.write_text("\n".join(remaining) + ("\n" if remaining else ""))
    write_report(results, remaining)

    with LOG_PATH.open("w") as fh:
        for item in results:
            fh.write(json.dumps({"type": "created", **item}) + "\n")
        for signal in remaining:
            fh.write(json.dumps({"type": "remaining", "signal": signal}) + "\n")

    print(f"Created {len(results)} WRDS characteristic files")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")
    print(f"Wrote {REMAINING_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
