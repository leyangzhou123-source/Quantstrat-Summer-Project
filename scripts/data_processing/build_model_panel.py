from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from quantstrat.features.interactions import add_macro_interactions
from quantstrat.features.ranking import rank_characteristics

RAW_DIR = ROOT / "data" / "raw"
EXTERNAL_DIR = ROOT / "data" / "external"
PROCESSED_DIR = ROOT / "data" / "processed"

CCM_LINK_URL = "https://braverock.com/brian/CRSP/crsp_compustat_linking.csv"
WELCH_GOYAL_MONTHLY_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1bM7vCWd3WOt95Sf9qjLPZjoiafgF_8EG/gviz/tq?tqx=out:csv&sheet=Monthly"
)
FAMA_FRENCH_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)

DAILY_PRICE_COLUMNS = [
    "GVKEY",
    "LPERMNO",
    "LPERMCO",
    "datadate",
    "tic",
    "cusip",
    "conm",
    "prccd",
    "prchd",
    "prcld",
    "ajexdi",
    "trfd",
    "cshoc",
    "cshtrd",
    "sic",
    "gsector",
    "ggroup",
    "exchg",
    "secstat",
    "tpci",
]

CHARACTERISTIC_COLUMNS = [
    "mvel1",
    "price",
    "dolvol",
    "turn",
    "zerotrade",
    "retvol",
    "maxret",
    "std_dolvol",
    "std_turn",
    "ill",
    "mom1m",
    "mom6m",
    "mom12m",
    "mom36m",
    "chmom",
    "beta",
    "betasq",
    "idiovol",
]

MACRO_COLUMNS = [
    "macro_dp",
    "macro_ep",
    "macro_bm",
    "macro_ntis",
    "macro_tbl",
    "macro_tms",
    "macro_dfy",
    "macro_svar",
]


def parse_yyyymm_date(value: pd.Series) -> pd.Series:
    text = value.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    return pd.to_datetime(text, errors="coerce")


def download_url(url: str, output: Path, overwrite: bool = False) -> bool:
    if output.exists() and not overwrite:
        return True
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            output.write_bytes(response.read())
        return True
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"Warning: could not download {url}: {exc}", file=sys.stderr)
        return False


def download_public_inputs(overwrite: bool = False) -> None:
    download_url(CCM_LINK_URL, EXTERNAL_DIR / "crsp_compustat_linking.csv", overwrite=overwrite)
    download_url(
        WELCH_GOYAL_MONTHLY_URL, EXTERNAL_DIR / "welch_goyal_monthly_raw.csv", overwrite=overwrite
    )
    download_url(FAMA_FRENCH_DAILY_URL, EXTERNAL_DIR / "fama_french_daily.zip", overwrite=overwrite)


def raw_price_files() -> list[Path]:
    files = []
    for path in sorted(RAW_DIR.glob("*.csv")):
        cols = set(pd.read_csv(path, nrows=0).columns)
        if {"datadate", "prccd", "cshoc"}.issubset(cols):
            files.append(path)
    if not files:
        raise FileNotFoundError(f"No daily price CSVs with prccd/cshoc found in {RAW_DIR}")
    return files


def make_numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def aggregate_daily_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk["date"] = parse_yyyymm_date(chunk["datadate"])
    chunk = chunk.dropna(subset=["date", "LPERMNO", "prccd"])
    make_numeric(
        chunk,
        [
            "GVKEY",
            "LPERMNO",
            "LPERMCO",
            "prccd",
            "prchd",
            "prcld",
            "ajexdi",
            "trfd",
            "cshoc",
            "cshtrd",
            "sic",
            "gsector",
            "ggroup",
            "exchg",
        ],
    )
    chunk = chunk.dropna(subset=["LPERMNO", "prccd"])
    chunk["permno"] = chunk["LPERMNO"].astype("int64")
    chunk["gvkey"] = chunk["GVKEY"].astype("Int64")
    chunk["month"] = chunk["date"].dt.to_period("M").dt.to_timestamp("M")
    chunk["ajexdi"] = chunk["ajexdi"].replace(0, np.nan).fillna(1.0)
    chunk["trfd"] = chunk["trfd"].replace(0, np.nan).fillna(1.0)
    chunk["adj_close"] = chunk["prccd"].abs() / chunk["ajexdi"] * chunk["trfd"]
    chunk["dollar_vol"] = chunk["prccd"].abs() * chunk["cshtrd"].fillna(0.0)
    chunk["turnover"] = chunk["cshtrd"] / chunk["cshoc"].replace(0, np.nan)
    chunk = chunk.sort_values(["permno", "date"])
    chunk["daily_ret"] = chunk.groupby("permno", sort=False)["adj_close"].pct_change()
    chunk.loc[chunk["daily_ret"].abs() > 5, "daily_ret"] = np.nan

    grouped = chunk.groupby(["permno", "month"], sort=False)
    monthly = grouped.agg(
        gvkey=("gvkey", "last"),
        lpermco=("LPERMCO", "last"),
        tic=("tic", "last"),
        cusip=("cusip", "last"),
        conm=("conm", "last"),
        sic=("sic", "last"),
        gsector=("gsector", "last"),
        ggroup=("ggroup", "last"),
        exchg=("exchg", "last"),
        secstat=("secstat", "last"),
        tpci=("tpci", "last"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        first_adj_close=("adj_close", "first"),
        last_adj_close=("adj_close", "last"),
        cshoc=("cshoc", "last"),
        dollar_vol_mean=("dollar_vol", "mean"),
        dollar_vol_std=("dollar_vol", "std"),
        turnover_mean=("turnover", "mean"),
        turnover_std=("turnover", "std"),
        daily_ret_std=("daily_ret", "std"),
        daily_ret_max=("daily_ret", "max"),
        ill_raw=("daily_ret", lambda x: np.nan),
        trading_days=("date", "count"),
        zero_volume_days=("cshtrd", lambda x: int((x.fillna(0.0) <= 0).sum())),
    ).reset_index()

    ill = (
        chunk.assign(
            ill_component=chunk["daily_ret"].abs() / chunk["dollar_vol"].replace(0, np.nan)
        )
        .groupby(["permno", "month"], sort=False)["ill_component"]
        .mean()
        .reset_index(name="ill")
    )
    monthly = monthly.drop(columns=["ill_raw"]).merge(ill, on=["permno", "month"], how="left")
    return monthly


def combine_partial_months(parts: list[pd.DataFrame]) -> pd.DataFrame:
    monthly = pd.concat(parts, ignore_index=True)
    monthly = monthly.sort_values(["permno", "month", "last_date"])
    agg = monthly.groupby(["permno", "month"], sort=False).agg(
        gvkey=("gvkey", "last"),
        lpermco=("lpermco", "last"),
        tic=("tic", "last"),
        cusip=("cusip", "last"),
        conm=("conm", "last"),
        sic=("sic", "last"),
        gsector=("gsector", "last"),
        ggroup=("ggroup", "last"),
        exchg=("exchg", "last"),
        secstat=("secstat", "last"),
        tpci=("tpci", "last"),
        first_date=("first_date", "min"),
        last_date=("last_date", "max"),
        first_adj_close=("first_adj_close", "first"),
        last_adj_close=("last_adj_close", "last"),
        cshoc=("cshoc", "last"),
        dollar_vol_mean=("dollar_vol_mean", "mean"),
        dollar_vol_std=("dollar_vol_std", "mean"),
        turnover_mean=("turnover_mean", "mean"),
        turnover_std=("turnover_std", "mean"),
        daily_ret_std=("daily_ret_std", "mean"),
        daily_ret_max=("daily_ret_max", "max"),
        ill=("ill", "mean"),
        trading_days=("trading_days", "sum"),
        zero_volume_days=("zero_volume_days", "sum"),
    )
    return agg.reset_index()


def build_monthly_from_raw(
    chunk_rows: int,
    sample_files: int | None = None,
    max_chunks_per_file: int | None = None,
) -> pd.DataFrame:
    parts = []
    files = raw_price_files()
    if sample_files is not None:
        files = files[:sample_files]
    for path in files:
        usecols = [c for c in DAILY_PRICE_COLUMNS if c in pd.read_csv(path, nrows=0).columns]
        print(f"Reading {path.relative_to(ROOT)}")
        for chunk_number, chunk in enumerate(
            pd.read_csv(path, usecols=usecols, chunksize=chunk_rows, low_memory=False),
            start=1,
        ):
            part = aggregate_daily_chunk(chunk)
            if not part.empty:
                parts.append(part)
            if max_chunks_per_file is not None and chunk_number >= max_chunks_per_file:
                break
    if not parts:
        raise RuntimeError("No usable rows were found in the raw daily files")
    monthly = combine_partial_months(parts)
    monthly["ret"] = (
        monthly["last_adj_close"] / monthly.groupby("permno", sort=False)["last_adj_close"].shift(1)
        - 1.0
    )
    monthly["market_equity"] = monthly["last_adj_close"].abs() * monthly["cshoc"]
    monthly["sic2"] = (monthly["sic"].fillna(0).astype("int64") // 100).astype("int64")
    return monthly.sort_values(["permno", "month"]).reset_index(drop=True)


def add_return_characteristics(monthly: pd.DataFrame) -> pd.DataFrame:
    panel = monthly.copy()
    panel["price"] = panel["last_adj_close"].abs()
    panel["mvel1"] = np.log(panel["market_equity"].replace(0, np.nan))
    panel["dolvol"] = np.log(panel["dollar_vol_mean"].replace(0, np.nan))
    panel["turn"] = panel["turnover_mean"]
    panel["zerotrade"] = panel["zero_volume_days"] / panel["trading_days"].replace(0, np.nan)
    panel["retvol"] = panel["daily_ret_std"]
    panel["maxret"] = panel["daily_ret_max"]
    panel["std_dolvol"] = np.log1p(panel["dollar_vol_std"].clip(lower=0))
    panel["std_turn"] = panel["turnover_std"]
    panel["ill"] = panel["ill"]

    grouped = panel.groupby("permno", sort=False)["ret"]
    panel["mom1m"] = grouped.shift(1)
    for window, name in [(6, "mom6m"), (12, "mom12m"), (36, "mom36m")]:
        panel[name] = grouped.transform(
            lambda x, w=window: (
                (1.0 + x.shift(1)).rolling(w, min_periods=max(2, w // 2)).apply(np.prod, raw=True)
                - 1.0
            )
        )
    panel["chmom"] = panel["mom6m"] - grouped.transform(
        lambda x: (1.0 + x.shift(7)).rolling(6, min_periods=3).apply(np.prod, raw=True) - 1.0
    )
    return panel


def load_fama_french_daily() -> pd.DataFrame | None:
    path = EXTERNAL_DIR / "fama_french_daily.zip"
    if not path.exists():
        return None
    with zipfile.ZipFile(path) as zf:
        name = zf.namelist()[0]
        text = zf.read(name).decode("latin1")
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(",Mkt-RF"))
    end = next(i for i in range(start + 1, len(lines)) if not lines[i].strip())
    ff = pd.read_csv(io.StringIO("\n".join(lines[start:end])))
    ff = ff.rename(columns={ff.columns[0]: "date", "Mkt-RF": "mktrf", "RF": "rf"})
    ff["date"] = pd.to_datetime(ff["date"].astype(str), format="%Y%m%d", errors="coerce")
    for col in ["mktrf", "SMB", "HML", "rf"]:
        if col in ff.columns:
            ff[col] = pd.to_numeric(ff[col], errors="coerce") / 100.0
    return ff[["date", "mktrf", "rf"]].dropna()


def add_beta_features(panel: pd.DataFrame) -> pd.DataFrame:
    ff = load_fama_french_daily()
    if ff is None:
        panel["beta"] = np.nan
        panel["betasq"] = np.nan
        panel["idiovol"] = np.nan
        return panel
    market = (
        ff.assign(month=ff["date"].dt.to_period("M").dt.to_timestamp("M"))
        .groupby("month")["mktrf"]
        .sum()
    )
    panel = panel.merge(market.rename("market_ret").reset_index(), on="month", how="left")
    panel["rf"] = 0.0
    rows = []
    for _, group in panel.sort_values("month").groupby("permno", sort=False):
        cov = group["ret"].rolling(36, min_periods=12).cov(group["market_ret"])
        var = group["market_ret"].rolling(36, min_periods=12).var()
        beta = cov / var
        idiovol = (group["ret"] - beta * group["market_ret"]).rolling(36, min_periods=12).std()
        rows.append(pd.DataFrame({"_idx": group.index, "beta": beta, "idiovol": idiovol}))
    values = pd.concat(rows, ignore_index=True).set_index("_idx")
    panel.loc[values.index, "beta"] = values["beta"]
    panel.loc[values.index, "idiovol"] = values["idiovol"]
    panel["betasq"] = panel["beta"] ** 2
    return panel.drop(columns=["market_ret"])


def load_welch_goyal_monthly() -> pd.DataFrame:
    path = EXTERNAL_DIR / "welch_goyal_monthly_raw.csv"
    if not path.exists():
        return pd.DataFrame(columns=["month", "rf_welch_goyal", *MACRO_COLUMNS])
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    numeric_columns = [
        "yyyymm",
        "Index",
        "D12",
        "E12",
        "b/m",
        "b.m",
        "tbl",
        "AAA",
        "BAA",
        "lty",
        "ntis",
        "Rfree",
        "svar",
    ]
    for column in numeric_columns:
        if column in raw.columns:
            raw[column] = pd.to_numeric(
                raw[column].astype(str).str.replace(",", "", regex=False), errors="coerce"
            )
    raw["month"] = pd.to_datetime(
        raw["yyyymm"].astype("Int64").astype(str), format="%Y%m", errors="coerce"
    ) + pd.offsets.MonthEnd(0)
    raw["IndexDiv"] = raw["Index"] + raw["D12"]
    raw["dp"] = np.log(raw["D12"]) - np.log(raw["Index"])
    raw["ep"] = np.log(raw["E12"]) - np.log(raw["Index"])
    raw["tms"] = raw["lty"] - raw["tbl"]
    raw["dfy"] = raw["BAA"] - raw["AAA"]
    bm_col = "b/m" if "b/m" in raw.columns else "b.m"
    keep = raw[["month", "Rfree", "dp", "ep", bm_col, "ntis", "tbl", "tms", "dfy", "svar"]].copy()
    keep = keep.rename(
        columns={
            "Rfree": "rf_welch_goyal",
            "dp": "macro_dp",
            "ep": "macro_ep",
            bm_col: "macro_bm",
            "ntis": "macro_ntis",
            "tbl": "macro_tbl",
            "tms": "macro_tms",
            "dfy": "macro_dfy",
            "svar": "macro_svar",
        }
    )
    return keep.dropna(subset=["month"]).sort_values("month")


def fill_and_rank_characteristics(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    for col in CHARACTERISTIC_COLUMNS:
        if col not in panel.columns:
            panel[col] = np.nan
        panel[col] = panel.groupby("month")[col].transform(lambda x: x.fillna(x.dropna().median()))
        panel[col] = panel[col].fillna(0.0)
    return rank_characteristics(panel, "month", CHARACTERISTIC_COLUMNS)


def add_industry_dummies(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dummies = pd.get_dummies(panel["sic2"].fillna(0).astype("int64"), prefix="sic2", dtype=np.int8)
    panel = pd.concat([panel, dummies], axis=1)
    return panel, dummies.columns.tolist()


def build_final_panel(
    chunk_rows: int,
    download_public: bool,
    overwrite_downloads: bool,
    max_chunks_per_file: int | None,
) -> tuple[pd.DataFrame, dict]:
    if download_public:
        download_public_inputs(overwrite=overwrite_downloads)

    monthly = build_monthly_from_raw(chunk_rows=chunk_rows, max_chunks_per_file=max_chunks_per_file)
    source_start = monthly["first_date"].min()
    source_end = monthly["last_date"].max()
    panel = add_return_characteristics(monthly)
    panel = add_beta_features(panel)
    panel = fill_and_rank_characteristics(panel)

    macro = load_welch_goyal_monthly()
    panel = panel.merge(macro, on="month", how="left")
    for col in ["rf_welch_goyal", *MACRO_COLUMNS]:
        panel[col] = panel[col].ffill().bfill().fillna(0.0)

    panel["ret_excess"] = panel["ret"] - panel["rf_welch_goyal"]
    panel["ret_excess_lead1"] = panel.groupby("permno", sort=False)["ret_excess"].shift(-1)
    panel["next_month"] = panel.groupby("permno", sort=False)["month"].shift(-1)
    panel = panel[panel["next_month"] == panel["month"] + pd.offsets.MonthEnd(1)].copy()
    panel = panel.dropna(subset=["ret_excess_lead1", "market_equity"]).copy()
    panel, industry_cols = add_industry_dummies(panel)
    panel = add_macro_interactions(panel, CHARACTERISTIC_COLUMNS, MACRO_COLUMNS)

    keep_cols = [
        "month",
        "permno",
        "gvkey",
        "lpermco",
        "tic",
        "cusip",
        "conm",
        "sic",
        "sic2",
        "gsector",
        "ggroup",
        "exchg",
        "secstat",
        "tpci",
        "ret",
        "ret_excess",
        "ret_excess_lead1",
        "market_equity",
        "rf_welch_goyal",
        *CHARACTERISTIC_COLUMNS,
        *MACRO_COLUMNS,
        *[f"{char}__x__{macro}" for char in CHARACTERISTIC_COLUMNS for macro in MACRO_COLUMNS],
        *industry_cols,
    ]
    panel = panel[keep_cols].sort_values(["month", "permno"]).reset_index(drop=True)
    for column in ["tic", "cusip", "conm", "secstat", "tpci"]:
        if column in panel.columns:
            panel[column] = panel[column].astype("string")
    manifest = {
        "panel": "data/processed/no_wrds_model_panel.parquet",
        "rows": len(panel),
        "columns": int(panel.shape[1]),
        "source_daily_start": str(source_start.date()),
        "source_daily_end": str(source_end.date()),
        "panel_start": str(panel["month"].min().date()) if len(panel) else None,
        "panel_end": str(panel["month"].max().date()) if len(panel) else None,
        "latest_source_file_start": "2022-01-03",
        "target": "ret_excess_lead1",
        "target_risk_free_rate": "rf_welch_goyal",
        "date": "month",
        "asset_id": "permno",
        "weight": "market_equity",
        "characteristics": CHARACTERISTIC_COLUMNS,
        "macro_predictors": MACRO_COLUMNS,
        "macro_interactions": [
            f"{char}__x__{macro}" for char in CHARACTERISTIC_COLUMNS for macro in MACRO_COLUMNS
        ],
        "industry_dummies": industry_cols,
        "notes": [
            "No WRDS access was used.",
            "Firm characteristics are the subset computable from supplied daily CRSP/Compustat-style files.",
            "Characteristics are cross-sectionally median-filled and ranked into [-1, 1] by month.",
            "The supplied 2021 CSV has link records only, not daily prices/returns, so no 2021 return panel rows are produced.",
        ],
    }
    return panel, manifest


def write_sparse_industry_interactions(panel: pd.DataFrame, industry_cols: list[str]) -> dict:
    blocks = []
    char = sparse.csr_matrix(panel[CHARACTERISTIC_COLUMNS].to_numpy(dtype=np.float32, copy=False))
    names = []
    for industry_col in industry_cols:
        indicator = sparse.csr_matrix(panel[industry_col].to_numpy(dtype=np.float32, copy=False)).T
        blocks.append(char.multiply(indicator))
        names.extend(f"{c}__x__{industry_col}" for c in CHARACTERISTIC_COLUMNS)
    matrix = (
        sparse.hstack(blocks, format="csr", dtype=np.float32)
        if blocks
        else sparse.csr_matrix((len(panel), 0))
    )
    matrix_path = PROCESSED_DIR / "no_wrds_industry_characteristic_interactions.npz"
    names_path = PROCESSED_DIR / "no_wrds_industry_characteristic_interaction_names.json"
    sparse.save_npz(matrix_path, matrix)
    names_path.write_text(json.dumps(names, indent=2) + "\n")
    return {
        "matrix": str(matrix_path.relative_to(ROOT)),
        "feature_names": str(names_path.relative_to(ROOT)),
        "shape": list(matrix.shape),
        "nnz": int(matrix.nnz),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a no-WRDS paper-style stock-month modeling panel from supplied CRSP/Compustat CSVs."
    )
    parser.add_argument("--chunk-rows", type=int, default=500_000)
    parser.add_argument(
        "--download-public",
        action="store_true",
        help="Download public CCM, Welch-Goyal, and FF inputs.",
    )
    parser.add_argument("--overwrite-downloads", action="store_true")
    parser.add_argument("--skip-sparse-industry-interactions", action="store_true")
    parser.add_argument(
        "--max-chunks-per-file",
        type=int,
        default=None,
        help="Smoke-test limiter; omit for full data.",
    )
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel, manifest = build_final_panel(
        chunk_rows=args.chunk_rows,
        download_public=args.download_public,
        overwrite_downloads=args.overwrite_downloads,
        max_chunks_per_file=args.max_chunks_per_file,
    )
    panel_path = PROCESSED_DIR / "no_wrds_model_panel.parquet"
    panel.to_parquet(panel_path, index=False)

    if not args.skip_sparse_industry_interactions:
        manifest["industry_characteristic_sparse_interactions"] = (
            write_sparse_industry_interactions(panel, manifest["industry_dummies"])
        )

    manifest_path = PROCESSED_DIR / "no_wrds_model_panel_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {panel_path.relative_to(ROOT)} with {len(panel):,} rows")
    print(f"Panel months: {manifest['panel_start']} to {manifest['panel_end']}")
    print(f"Latest supplied daily price data starts: {manifest['latest_source_file_start']}")
    print(f"Wrote {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
