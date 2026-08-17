from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "feature_engineering"))

import run_feature_engineering_nn5_experiment as base
from run_feature_engineering_nn5_expanded_search import (
    ACCOUNTING_BALANCE,
    GROWTH_OPERATIONS,
    HIGH_IC_CORE,
    INTANGIBLE_ATTENTION,
    MACRO_ALL,
)


def build_industry_block_groups(manifest_path: Path) -> dict[str, dict[str, list[str]]]:
    manifest = json.loads(manifest_path.read_text())
    chars = manifest.get("stock_characteristics", manifest.get("characteristics", []))
    industries = manifest.get("industry_dummies", [])

    core_60 = list(
        dict.fromkeys(HIGH_IC_CORE + ACCOUNTING_BALANCE[:10] + GROWTH_OPERATIONS[:10] + MACRO_ALL)
    )
    core_90 = list(
        dict.fromkeys(
            HIGH_IC_CORE
            + ACCOUNTING_BALANCE
            + GROWTH_OPERATIONS[:18]
            + INTANGIBLE_ATTENTION[:8]
            + MACRO_ALL
        )
    )
    quality_value_momentum = list(
        dict.fromkeys(
            [
                "mvel1",
                "bm",
                "bm_ia",
                "ep",
                "cfp",
                "cfp_ia",
                "ps",
                "dy",
                "mom1m",
                "mom6m",
                "mom12m",
                "mom36m",
                "chmom",
                "indmom",
                "operprof",
                "gma",
                "roic",
                "roaq",
                "roeq",
                "cashdebt",
                "acc",
                "stdacc",
                "stdcf",
                "lev",
                "baspread",
                "ill",
                "retvol",
                "idiovol",
                "maxret",
                "turn",
                "dolvol",
            ]
            + MACRO_ALL
        )
    )

    risky = {
        "baspread",
        "ill",
        "retvol",
        "idiovol",
        "maxret",
        "beta",
        "betasq",
        "zerotrade",
        "std_dolvol",
        "std_turn",
    }
    low_growth_ops = {
        "grcapx",
        "grltnoa",
        "hire",
        "herf",
        "chatoia",
        "chempia",
        "chpmia",
        "pchsale_pchinvt",
        "pchsale_pchrect",
        "pchsale_pchxsga",
    }

    return {
        "full_no_interactions_180": {
            "characteristics": chars,
            "macro": MACRO_ALL,
            "industry": industries,
        },
        "full_minus_macro_172": {
            "characteristics": chars,
            "industry": industries,
        },
        "all_chars_macro_top_industry_113": {
            "characteristics": chars,
            "macro": MACRO_ALL,
            "industry": industries[:18],
        },
        "core60_all_industry_138": {
            "core_economic_features": core_60,
            "industry": industries,
        },
        "core90_all_industry_168": {
            "core_economic_features": core_90,
            "industry": industries,
        },
        "quality_value_momentum_all_industry_118": {
            "quality_value_momentum": quality_value_momentum,
            "industry": industries,
        },
        "full_minus_risk_170": {
            "characteristics": [feature for feature in chars if feature not in risky],
            "macro": MACRO_ALL,
            "industry": industries,
        },
        "full_minus_weak_growth_ops_170": {
            "characteristics": [feature for feature in chars if feature not in low_growth_ops],
            "macro": MACRO_ALL,
            "industry": industries,
        },
        "all_chars_all_industry_no_macro_172": {
            "characteristics": chars,
            "industry": industries,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NN5 industry-block feature search.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-prefix", type=str, default="nn5_industry_block_feature_search")
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--skip-predictions", action="store_true")
    parser.add_argument("--checkpoint-each-split", action="store_true")
    parser.add_argument("--first-test-year", type=int, default=None)
    parser.add_argument("--last-test-year", type=int, default=None)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = base.build_base_config(config_path)
    manifest_path = ROOT / cfg["data"]["manifest_path"]
    groups = build_industry_block_groups(manifest_path)

    original = deepcopy(base.MANUAL_GROUPS)
    try:
        base.MANUAL_GROUPS = groups
        sys.argv = [
            "scripts/feature_engineering/run_feature_engineering_nn5_experiment.py",
            "--config",
            str(args.config),
            "--out-prefix",
            args.out_prefix,
        ]
        if args.groups:
            sys.argv.extend(["--groups", *args.groups])
        if args.skip_predictions:
            sys.argv.append("--skip-predictions")
        if args.checkpoint_each_split:
            sys.argv.append("--checkpoint-each-split")
        if args.first_test_year is not None:
            sys.argv.extend(["--first-test-year", str(args.first_test_year)])
        if args.last_test_year is not None:
            sys.argv.extend(["--last-test-year", str(args.last_test_year)])
        base.main()
    finally:
        base.MANUAL_GROUPS = original


if __name__ == "__main__":
    main()
