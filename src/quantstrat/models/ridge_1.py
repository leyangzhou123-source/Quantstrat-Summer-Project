"""Train and evaluate a Ridge regression on the GKX-style stock panel.

Framework (matches the group presentation):

1. Split the panel into train 1957-03..1974-12, validation 1975-01..1986-12,
   test 1987-01..2016-11 (time-ordered, no shuffling).
2. Fit Ridge on the train period for each regularization strength lambda
   (``--alphas``) and pick the lambda with the best validation OOS R^2
   (zero-forecast benchmark) without ever touching the test set.
3. Calibrate the selected model's forecasts with a validation-period
   regression y = a + b * y_hat.
4. Report test OOS R^2 / MSE / MAE / monthly Pearson IC (raw and calibrated)
   and the value-weighted forecast-decile long-short portfolio.

Usage:
    python models/train_ridge.py                       # full data run
    python models/train_ridge.py --smoke               # synthetic end-to-end check
    python models/train_ridge.py --alphas 0.1 1 10 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from pipeline import RAW_DIR, FeatureBuilder, prepare_model_data
from evaluation import evaluate_on_test, out_of_sample_r2, save_results

TARGET = "ret_excess_lead1"
DEFAULT_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0]


def predict_in_chunks(
    model: Ridge, scaler: StandardScaler, frame: pd.DataFrame, builder: FeatureBuilder
) -> pd.Series:
    predictions = pd.Series(np.nan, index=frame.index, dtype=float, name="forecast")
    for index, X_chunk in builder.transform_by_month(frame):
        predictions.loc[index] = model.predict(scaler.transform(X_chunk))
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS,
                        help="Ridge regularization strengths to tune on validation.")
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
        args.alphas = [0.1, 10.0, 1_000.0]

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
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)

    X_val = scaler.transform(builder.transform(validation))
    y_val = validation[TARGET]

    validation_curve = []
    best = {"alpha": None, "oos_r2": -np.inf, "model": None}
    for alpha in args.alphas:
        model = Ridge(alpha=alpha, fit_intercept=True, random_state=args.seed)
        model.fit(X_train, y_train)
        val_pred = pd.Series(model.predict(X_val), index=validation.index)
        val_r2 = out_of_sample_r2(y_val, val_pred)
        validation_curve.append({"alpha": alpha, "validation_oos_r2": float(val_r2)})
        print(f"alpha={alpha:>10g}  validation OOS R^2 = {val_r2:.6f}")
        if val_r2 > best["oos_r2"]:
            best = {"alpha": alpha, "oos_r2": float(val_r2), "model": model}

    print(f"selected alpha={best['alpha']} (validation OOS R^2 = {best['oos_r2']:.6f})")
    model = best["model"]
    validation_forecast = pd.Series(model.predict(X_val), index=validation.index)
    del X_train, X_val

    test_forecast = predict_in_chunks(model, scaler, test, builder)

    metrics, decile_table, predictions = evaluate_on_test(
        test, test_forecast, validation, validation_forecast, target=TARGET
    )
    metrics = {
        "model": "ridge",
        "selected_alpha": best["alpha"],
        "validation_oos_r2": best["oos_r2"],
        "validation_curve": validation_curve,
        "n_features": builder.n_features,
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        **metrics,
        "runtime_seconds": round(time.time() - started, 1),
    }

    out = save_results("ridge", metrics, decile_table, predictions, args.output_dir)
    print(json.dumps({k: v for k, v in metrics.items() if k != "validation_curve"}, indent=2))
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
