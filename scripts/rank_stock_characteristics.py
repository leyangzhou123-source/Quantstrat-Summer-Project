"""Cross-sectionally rank stock characteristics and save a Parquet copy.

Example (from the repository root):
    python scripts/rank_stock_characteristics.py --input "D:\\New folder (2)\\Documents\\Masters - 2nd Time's the Charm\\UIUC\\QuantStrat Research Paper\\Quantstrat Data\\stock_characteristics.parquet"

The output keeps every original column name.  Characteristic values are
replaced with their within-date scaled ranks:

    2 * (rank - 1) / (N - 1) - 1

where N is the count of non-missing values for that characteristic on that
date.  Missing observations stay missing.  Tied values receive their average
rank; a date with only one non-missing observation receives NaN for that
characteristic because the requested formula has a zero denominator.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# These identifiers are not characteristics even if they have a numeric dtype.
DEFAULT_IDENTIFIER_COLUMNS = {
    "permno",
    "permco",
    "gvkey",
    "cusip",
    "ncusip",
    "ticker",
    "sic",
    "sic2",
    "exchange",
    "exchcd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input Parquet file.")
    parser.add_argument(
        "--time-column",
        default="yyyymm",
        help="Column that defines each cross-section (default: yyyymm).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet file. Defaults to <input stem>_ranked.parquet beside the input.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help="Optional exact list of characteristic columns to rank. Defaults to numeric non-ID columns.",
    )
    parser.add_argument(
        "--identifier-columns",
        nargs="*",
        default=(),
        help="Additional columns to exclude when --columns is omitted (for example: market_equity).",
    )
    parser.add_argument(
        "--force-if-already-scaled",
        action="store_true",
        help="Proceed even if every selected value already lies in the [-1, 1] rank range.",
    )
    return parser.parse_args()


def choose_characteristics(
    panel: pd.DataFrame,
    time_column: str,
    columns: list[str] | None,
    identifier_columns: tuple[str, ...] | list[str],
) -> list[str]:
    if time_column not in panel.columns:
        raise ValueError(
            f"Time column {time_column!r} was not found. Available columns: {panel.columns.tolist()}"
        )

    if columns is not None:
        missing = sorted(set(columns).difference(panel.columns))
        if missing:
            raise ValueError(f"Requested characteristic columns are missing: {missing}")
        non_numeric = [column for column in columns if not pd.api.types.is_numeric_dtype(panel[column])]
        if non_numeric:
            raise ValueError(f"Characteristic columns must be numeric: {non_numeric}")
        return columns

    excluded = DEFAULT_IDENTIFIER_COLUMNS | {time_column.lower()} | {
        column.lower() for column in identifier_columns
    }
    return [
        column
        for column in panel.columns
        if column.lower() not in excluded and pd.api.types.is_numeric_dtype(panel[column])
    ]


def scale_ranks(panel: pd.DataFrame, time_column: str, columns: list[str]) -> pd.DataFrame:
    """Return a copy with each selected column replaced by its scaled date rank."""
    ranked = panel.copy()
    grouped = ranked.groupby(time_column, sort=False)

    # Both rank and count are calculated separately for every characteristic,
    # so incomplete coverage in one characteristic does not affect another.
    ranks = grouped[columns].rank(method="average", na_option="keep")
    counts = grouped[columns].transform("count")
    ranked[columns] = 2.0 * (ranks - 1.0) / (counts - 1.0) - 1.0
    ranked[columns] = ranked[columns].where(counts > 1)
    return ranked


def appears_already_scaled(panel: pd.DataFrame, columns: list[str]) -> bool:
    """Conservatively flag data whose selected finite values all lie in [-1, 1]."""
    values = panel[columns]
    return bool((values.min(skipna=True) >= -1.0).all() and (values.max(skipna=True) <= 1.0).all())


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    output_path = args.output or input_path.with_name(f"{input_path.stem}_ranked.parquet")
    output_path = output_path.expanduser().resolve()
    if output_path == input_path:
        raise ValueError("Output path must differ from the input path.")

    print(f"Reading {input_path}")
    panel = pd.read_parquet(input_path)
    characteristics = choose_characteristics(
        panel, args.time_column, args.columns, args.identifier_columns
    )
    if not characteristics:
        raise ValueError("No numeric characteristic columns were selected for ranking.")
    if appears_already_scaled(panel, characteristics) and not args.force_if_already_scaled:
        raise ValueError(
            "All selected values already lie in [-1, 1], which suggests that this file has "
            "already been rank-scaled. Verify the source before ranking again. If you are "
            "certain it is not already ranked, rerun with --force-if-already-scaled."
        )

    print(f"Ranking {len(characteristics)} columns within each {args.time_column!r} cross-section")
    ranked = scale_ranks(panel, args.time_column, characteristics)
    ranked.to_parquet(output_path, index=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
