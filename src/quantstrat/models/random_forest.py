"""Train and evaluate a Random Forest on the GKX-style stock panel.

Framework (matches the group presentation):

1. Split the panel into train 1957-03..1974-12, validation 1975-01..1986-12,
   test 1987-01..2016-11 (time-ordered, no shuffling).
2. Fit a RandomForestRegressor on the train period for each hyperparameter
   combination (tree depth x feature subsampling) and pick the combination
   with the best validation OOS R^2 (zero-forecast benchmark).
3. Calibrate the selected model's forecasts with a validation-period
   regression y = a + b * y_hat.
4. Report test OOS R^2 / MSE / MAE / monthly Pearson IC (raw and calibrated)
   and the value-weighted forecast-decile long-short portfolio.

The default grid follows Gu-Kelly-Xiu (2020): shallow trees (depth 2-6),
300 trees, and a small feature subsample at each split.

Usage:
    python models/train_random_forest.py                 # full data run
    python models/train_random_forest.py --smoke         # synthetic end-to-end check
    python models/train_random_forest.py --max-depths 2 4 6 --n-estimators 300
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from pipeline import RAW_DIR, FeatureBuilder, prepare_model_data
from evaluation import evaluate_on_test, out_of_sample_r2, save_results

TARGET = "ret_excess_lead1"


def parse_max_features(value: str) -> float | int | str:
    if value == "sqrt":
        return "sqrt"
    number = float(value)
    if number >= 1.0:
        return int(number)
    return number


def predict_in_chunks(
    model: RandomForestRegressor, frame: pd.DataFrame, builder: FeatureBuilder
) -> pd.Series:
    predictions = pd.Series(np.nan, index=frame.index, dtype=float, name="forecast")
    for index, X_chunk in builder.transform_by_month(frame):
        predictions.loc[index] = model.predict(X_chunk)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-depths", type=int, nargs="+", default=[2, 4, 6],
                        help="Tree depths to tune on validation.")
    parser.add_argument("--max-features", type=parse_max_features, nargs="+",
                        default=["sqrt", 50],
                        help="Feature subsample per split: 'sqrt', an integer count, "
                             "or a fraction in (0, 1).")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-samples", type=float, default=0.5,
                        help="Bootstrap sample fraction per tree.")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR,
                        help="Directory holding the five raw data files.")
    parser.add_argument("--processed-panel", type=Path, default=None,
                        help="Optional prebuilt data/processed/model_panel.parquet.")
    parser.add_argument("--sample-rows", type=int, default=None,
                        help="Optional row cap for quick experiments.")
    parser.add_argument("--no-interactions", action="store_true",
                        help="Drop characteristic x macro interaction features.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="Run end-to-end on generated synthetic data.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = args.raw_dir
    if args.smoke:
        from synthetic import write_synthetic_raw_files

        raw_dir = write_synthetic_raw_files(seed=args.seed)
        args.processed_panel = Path("/nonexistent")  # force building from synthetic raw
        args.max_depths = [2, 4]
        args.max_features = ["sqrt"]
        args.n_estimators = 50

    started = time.time()
    train, validation, test, builder = prepare_model_data(
        raw_dir=raw_dir,
        processed_panel=args.processed_panel,
        sample_rows=args.sample_rows,
        interactions=not args.no_interactions,
    )
    print(
        f"panel loaded: train={len(train):,} validation={len(validation):,} "
        f"test={len(test):,} rows, {builder.n_features} features"
    )

    X_train = builder.transform(train)
    y_train = train[TARGET].to_numpy(dtype=float)
    X_val = builder.transform(validation)
    y_val = validation[TARGET]

    validation_curve = []
    best = {"params": None, "oos_r2": -np.inf, "model": None}
    for depth, max_features in product(args.max_depths, args.max_features):
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=depth,
            max_features=max_features,
            max_samples=args.max_samples,
            bootstrap=True,
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )
        model.fit(X_train, y_train)
        val_pred = pd.Series(model.predict(X_val), index=validation.index)
        val_r2 = out_of_sample_r2(y_val, val_pred)
        params = {"max_depth": depth, "max_features": max_features}
        validation_curve.append({**params, "validation_oos_r2": float(val_r2)})
        print(f"max_depth={depth} max_features={max_features!s:>6}  "
              f"validation OOS R^2 = {val_r2:.6f}")
        if val_r2 > best["oos_r2"]:
            best = {"params": params, "oos_r2": float(val_r2), "model": model}

    print(f"selected {best['params']} (validation OOS R^2 = {best['oos_r2']:.6f})")
    model = best["model"]
    validation_forecast = pd.Series(model.predict(X_val), index=validation.index)
    del X_train, X_val

    test_forecast = predict_in_chunks(model, test, builder)

    metrics, decile_table, predictions = evaluate_on_test(
        test, test_forecast, validation, validation_forecast, target=TARGET
    )
    best_params = dict(best["params"])
    if not isinstance(best_params["max_features"], str):
        best_params["max_features"] = float(best_params["max_features"])
    metrics = {
        "model": "random_forest",
        "selected_params": best_params,
        "n_estimators": args.n_estimators,
        "max_samples": args.max_samples,
        "validation_oos_r2": best["oos_r2"],
        "validation_curve": [
            {**{k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in row.items()}}
            for row in validation_curve
        ],
        "n_features": builder.n_features,
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        **metrics,
        "runtime_seconds": round(time.time() - started, 1),
    }

    out = save_results("random_forest", metrics, decile_table, predictions, args.output_dir)
    print(json.dumps({k: v for k, v in metrics.items() if k != "validation_curve"}, indent=2))
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
