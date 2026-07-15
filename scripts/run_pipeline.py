from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantstrat.engine import ResearchEngine


def main() -> None:
    engine = ResearchEngine.from_config(ROOT / "configs" / "default.yaml", project_root=ROOT)
    result = engine.run()
    out_dir = ROOT / "reports" / "model_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    result.predictions.to_parquet(out_dir / "engine_predictions.parquet", index=False)
    result.metrics.to_csv(out_dir / "engine_metrics.csv", index=False)
    print(result.metrics.to_string(index=False))


if __name__ == "__main__":
    main()
