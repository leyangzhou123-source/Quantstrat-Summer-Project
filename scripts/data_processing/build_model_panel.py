from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from quantstrat.features.interactions import build_sparse_industry_characteristic_interactions

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def characteristic_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {"permno", "yyyymm", "month", "date"}]


def industry_dummy_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("sic2_")]


def build_panel(sample_rows: int | None = None) -> tuple[pd.DataFrame, list[str], list[str]]:
    returns = pd.read_parquet(RAW_DIR / "stock_returns.parquet")
    chars = pd.read_parquet(RAW_DIR / "stock_characteristics.parquet")
    industry = pd.read_parquet(RAW_DIR / "industry_dummies.parquet")
    macro = pd.read_parquet(RAW_DIR / "welch_goyal_macros.parquet")
    ff = pd.read_parquet(RAW_DIR / "fama_french_factors.parquet")

    returns["month"] = pd.to_datetime(returns["month"])
    returns["yyyymm"] = returns["month"].dt.year * 100 + returns["month"].dt.month
    returns = returns.sort_values(["permno", "month"]).copy()

    rf = macro[["yyyymm", "rf_welch_goyal"]].copy()
    ff_keep = ff[["yyyymm", "rf", "mktrf", "smb", "hml", "rmw", "cma", "umd"]].rename(
        columns={"rf": "rf_fama_french"}
    )
    panel = returns.merge(rf, on="yyyymm", how="left").merge(ff_keep, on="yyyymm", how="left")
    panel["ret_excess"] = panel["ret"] - panel["rf_welch_goyal"]
    panel["ret_excess_lead1"] = panel.groupby("permno", sort=False)["ret_excess"].shift(-1)
    panel["next_month"] = panel.groupby("permno", sort=False)["month"].shift(-1)
    panel = panel[panel["next_month"] == panel["month"] + pd.offsets.MonthEnd(1)].copy()

    char_cols = characteristic_columns(chars)
    panel = panel.merge(chars, on=["permno", "yyyymm"], how="inner")

    industry["month"] = pd.to_datetime(industry["month"])
    industry["yyyymm"] = industry["month"].dt.year * 100 + industry["month"].dt.month
    industry_cols = industry_dummy_columns(industry)
    panel = panel.merge(industry[["permno", "yyyymm", "sic2"] + industry_cols], on=["permno", "yyyymm"], how="left")

    macro_source_cols = ["dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar"]
    macro_rename = {col: f"macro_{col}" for col in macro_source_cols}
    macro = macro[["yyyymm"] + macro_source_cols].rename(columns=macro_rename)
    macro_cols = list(macro_rename.values())
    panel = panel.merge(macro, on="yyyymm", how="left")

    panel = panel.rename(columns={"me": "market_equity"})
    required = ["ret_excess_lead1", "market_equity"] + char_cols + industry_cols + macro_cols
    panel = panel.dropna(subset=["ret_excess_lead1"] + char_cols).sort_values(["month", "permno"]).reset_index(drop=True)
    panel[required] = panel[required].fillna(0.0)

    if sample_rows is not None:
        panel = panel.head(sample_rows).copy()

    return panel, char_cols, industry_cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-style model panel and sparse industry interactions.")
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--skip-interactions", action="store_true")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel, char_cols, industry_cols = build_panel(sample_rows=args.sample_rows)

    panel_out = PROCESSED_DIR / "model_panel.parquet"
    panel.to_parquet(panel_out, index=False)

    manifest = {
        "panel": str(panel_out.relative_to(ROOT)),
        "rows": int(len(panel)),
        "columns": int(panel.shape[1]),
        "characteristics": char_cols,
        "industry_dummies": industry_cols,
        "target": "ret_excess_lead1",
        "target_risk_free_rate": "rf_welch_goyal",
        "date": "month",
        "asset_id": "permno",
        "weight": "market_equity",
    }

    if not args.skip_interactions:
        interaction = build_sparse_industry_characteristic_interactions(
            panel=panel,
            characteristic_columns=char_cols,
            industry_dummy_columns=industry_cols,
        )
        matrix_out = PROCESSED_DIR / "industry_characteristic_interactions.npz"
        names_out = PROCESSED_DIR / "industry_characteristic_interaction_names.json"
        sparse.save_npz(matrix_out, interaction.matrix)
        names_out.write_text(json.dumps(interaction.feature_names, indent=2) + "\n")
        manifest["industry_characteristic_interactions"] = {
            "matrix": str(matrix_out.relative_to(ROOT)),
            "feature_names": str(names_out.relative_to(ROOT)),
            "shape": list(interaction.matrix.shape),
            "nnz": int(interaction.matrix.nnz),
        }

    manifest_out = PROCESSED_DIR / "model_panel_manifest.json"
    manifest_out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {panel_out.relative_to(ROOT)}")
    print(f"Wrote {manifest_out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
