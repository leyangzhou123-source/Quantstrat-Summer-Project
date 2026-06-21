from __future__ import annotations

from quantstrat.utils.config import load_config


def main() -> None:
    config = load_config()
    enabled_models = ", ".join(config["models"]["enabled"])
    print(f"Loaded project config for {config['project']['name']}")
    print(f"Enabled model families: {enabled_models}")
    print("Next step: implement ingestion, feature assembly, tuning, and evaluation.")


if __name__ == "__main__":
    main()

