from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantstrat.Engine.engine import ResearchEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the quantstrat research engine.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default.yaml",
        help="Path to a YAML or TOML config file.",
    )
    parser.add_argument(
        "--paper-rolling",
        action="store_true",
        help="Use the paper's annual rolling/expanding refit protocol.",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    engine = ResearchEngine.from_config(config_path, project_root=ROOT)
    result = engine.run_paper_rolling() if args.paper_rolling else engine.run()
    out_dir = ROOT / "reports" / "model_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    result.predictions.to_parquet(out_dir / "engine_predictions.parquet", index=False)
    result.metrics.to_csv(out_dir / "engine_metrics.csv", index=False)
    print(result.metrics.to_string(index=False))


if __name__ == "__main__":
    main()
