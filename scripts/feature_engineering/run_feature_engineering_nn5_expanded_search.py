from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "feature_engineering"))

import run_feature_engineering_nn5_experiment as base

HIGH_IC_CORE = [
    "baspread",
    "idiovol",
    "retvol",
    "maxret",
    "ill",
    "mvel1",
    "ep",
    "cfp",
    "bm",
    "bm_ia",
    "mom12m",
    "mom6m",
    "mom1m",
    "chmom",
    "indmom",
    "operprof",
    "gma",
    "roic",
    "roaq",
    "roeq",
    "cashdebt",
    "cashpr",
    "dy",
    "ps",
    "lev",
    "turn",
    "dolvol",
    "zerotrade",
    "std_dolvol",
    "std_turn",
]

ACCOUNTING_BALANCE = [
    "acc",
    "absacc",
    "pctacc",
    "stdacc",
    "stdcf",
    "cash",
    "currat",
    "quick",
    "depr",
    "tang",
    "tb",
    "secured",
    "securedind",
    "realestate",
]

GROWTH_OPERATIONS = [
    "agr",
    "egr",
    "sgr",
    "invest",
    "lgr",
    "grcapx",
    "grltnoa",
    "hire",
    "herf",
    "chatoia",
    "chinv",
    "chempia",
    "chpmia",
    "pchcapx_ia",
    "pchcurrat",
    "pchdepr",
    "pchgm_pchsale",
    "pchquick",
    "pchsale_pchinvt",
    "pchsale_pchrect",
    "pchsale_pchxsga",
    "pchsaleinv",
    "saleinv",
    "salecash",
    "salerec",
]

INTANGIBLE_ATTENTION = [
    "rd",
    "rd_mve",
    "rd_sale",
    "orgcap",
    "age",
    "convind",
    "sin",
    "ms",
    "nincr",
    "ear",
    "chtx",
    "cinvest",
    "aeavol",
    "roavol",
    "rsup",
    "sp",
    "mve_ia",
    "pricedelay",
]

MACRO_ALL = [
    "macro_dp",
    "macro_ep",
    "macro_bm",
    "macro_ntis",
    "macro_tbl",
    "macro_tms",
    "macro_dfy",
    "macro_svar",
]

INDUSTRY_BROAD = [
    "sic2_20",
    "sic2_24",
    "sic2_28",
    "sic2_35",
    "sic2_36",
    "sic2_37",
    "sic2_38",
    "sic2_49",
    "sic2_50",
    "sic2_51",
    "sic2_52",
    "sic2_53",
    "sic2_54",
    "sic2_60",
    "sic2_65",
    "sic2_67",
    "sic2_73",
    "sic2_87",
]


EXPANDED_GROUPS: dict[str, dict[str, list[str]]] = {
    "balanced_45_high_ic": {
        "value_momentum_risk": HIGH_IC_CORE[:24],
        "accounting_balance": ACCOUNTING_BALANCE[:8],
        "macro": ["macro_tms", "macro_tbl", "macro_dfy", "macro_ntis", "macro_dp"],
        "industry": INDUSTRY_BROAD[:8],
    },
    "balanced_60_with_operations": {
        "high_ic_core": HIGH_IC_CORE,
        "accounting_balance": ACCOUNTING_BALANCE[:10],
        "growth_operations": GROWTH_OPERATIONS[:10],
        "macro": MACRO_ALL,
        "industry": INDUSTRY_BROAD[:10],
    },
    "balanced_80_full_signal": {
        "high_ic_core": HIGH_IC_CORE,
        "accounting_balance": ACCOUNTING_BALANCE,
        "growth_operations": GROWTH_OPERATIONS[:18],
        "intangibles_attention": INTANGIBLE_ATTENTION[:8],
        "macro": MACRO_ALL,
        "industry": INDUSTRY_BROAD[:12],
    },
    "chars_all_macro_no_industry_102": {
        "high_ic_core": HIGH_IC_CORE,
        "accounting_balance": ACCOUNTING_BALANCE,
        "growth_operations": GROWTH_OPERATIONS,
        "intangibles_attention": INTANGIBLE_ATTENTION,
        "macro": MACRO_ALL,
    },
    "chars_all_macro_industry18_120": {
        "high_ic_core": HIGH_IC_CORE,
        "accounting_balance": ACCOUNTING_BALANCE,
        "growth_operations": GROWTH_OPERATIONS,
        "intangibles_attention": INTANGIBLE_ATTENTION,
        "macro": MACRO_ALL,
        "industry": INDUSTRY_BROAD,
    },
    "risk_value_quality_55": {
        "risk_liquidity": [
            "baspread",
            "idiovol",
            "retvol",
            "maxret",
            "ill",
            "beta",
            "betasq",
            "turn",
            "dolvol",
            "zerotrade",
            "std_dolvol",
            "std_turn",
        ],
        "value": ["mvel1", "bm", "bm_ia", "ep", "cfp", "cfp_ia", "ps", "dy"],
        "quality": ["operprof", "gma", "roic", "roaq", "roeq", "cashdebt", "cash", "lev"],
        "momentum": ["mom1m", "mom6m", "mom12m", "mom36m", "chmom", "indmom"],
        "macro": MACRO_ALL,
        "industry": INDUSTRY_BROAD[:12],
    },
    "value_momentum_quality_growth_70": {
        "value": ["mvel1", "bm", "bm_ia", "ep", "cfp", "cfp_ia", "ps", "dy"],
        "momentum": ["mom1m", "mom6m", "mom12m", "mom36m", "chmom", "indmom"],
        "quality_profitability": ["operprof", "gma", "roic", "roaq", "roeq", "cashdebt"],
        "growth_investment": GROWTH_OPERATIONS[:18],
        "accounting": ["acc", "absacc", "pctacc", "stdacc", "stdcf"],
        "risk_liquidity": ["baspread", "ill", "retvol", "idiovol", "maxret", "turn", "dolvol"],
        "macro": MACRO_ALL,
        "industry": INDUSTRY_BROAD[:10],
    },
    "top_signal_no_macro_70": {
        "high_ic_core": HIGH_IC_CORE,
        "accounting_balance": ACCOUNTING_BALANCE,
        "growth_operations": GROWTH_OPERATIONS[:14],
        "intangibles_attention": INTANGIBLE_ATTENTION[:8],
        "industry": INDUSTRY_BROAD[:8],
    },
    "top_signal_no_industry_80": {
        "high_ic_core": HIGH_IC_CORE,
        "accounting_balance": ACCOUNTING_BALANCE,
        "growth_operations": GROWTH_OPERATIONS[:20],
        "intangibles_attention": INTANGIBLE_ATTENTION[:8],
        "macro": MACRO_ALL,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run expanded manual NN5 feature search.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-prefix", type=str, default="nn5_expanded_feature_search")
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--skip-predictions", action="store_true")
    parser.add_argument("--checkpoint-each-split", action="store_true")
    parser.add_argument("--first-test-year", type=int, default=None)
    parser.add_argument("--last-test-year", type=int, default=None)
    args = parser.parse_args()

    original = deepcopy(base.MANUAL_GROUPS)
    try:
        base.MANUAL_GROUPS = EXPANDED_GROUPS
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
