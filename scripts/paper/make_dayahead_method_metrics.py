#!/usr/bin/env python
"""Create a multi-method day-ahead metrics table for the paper."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/carbon_forecast_matplotlib")

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from make_horizon_comparison_tables import (  # noqa: E402
    DAYAHEAD_EXTRA,
    DIRECT_SEEDS,
    LABELS,
    REGIONS,
    fit_direct_mlp_train_val,
    true_ci,
)

SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fuel_mix_trainer import (  # noqa: E402
    build_supervised_table,
    feature_columns_from_table,
    load_config,
    split_train_val_test,
)

OUT_DIR = REPO_ROOT / "results" / "experiments" / "method_metrics"


def _rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(y)) ** 2)))


def _mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(y))))


def _corr(pred: np.ndarray, y: np.ndarray) -> float:
    pred = np.asarray(pred)
    y = np.asarray(y)
    if pred.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(pred, y)[0, 1])


def _r2(pred: np.ndarray, y: np.ndarray) -> float:
    pred = np.asarray(pred)
    y = np.asarray(y)
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom < 1e-12:
        return float("nan")
    return float(1.0 - np.sum((y - pred) ** 2) / denom)


def _metric_row(
    region: str,
    method: str,
    pred: np.ndarray,
    y: np.ndarray,
    balance_pct: float | None = None,
) -> dict:
    mean_ci = float(np.mean(y))
    rmse = _rmse(pred, y)
    return {
        "region": LABELS[region],
        "method": method,
        "mean_ci": mean_ci,
        "ci_mae": _mae(pred, y),
        "ci_rmse": rmse,
        "rel_rmse_pct": 100.0 * rmse / mean_ci if mean_ci else float("nan"),
        "ci_corr": _corr(pred, y),
        "ci_r2": _r2(pred, y),
        "balance_pct": balance_pct,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    fuelmix_metrics = pd.read_csv(REPO_ROOT / "results" / "fuelmix_dayahead" / "multiseed" / "benchmark_ensemble.csv")

    for region in REGIONS + DAYAHEAD_EXTRA:
        cfg = load_config(str(REPO_ROOT / "configs" / "dayahead" / f"{region}_fuelmix_dayahead.json"))
        data = build_supervised_table(cfg)
        feature_cols = feature_columns_from_table(data, cfg)
        train_df, val_df, test_df = split_train_val_test(
            data,
            train_frac=cfg.get("split", {}).get("train_frac", 0.7),
            val_frac=cfg.get("split", {}).get("val_frac", 0.15),
        )
        y_train = true_ci(train_df, cfg)
        y_val = true_ci(val_df, cfg)
        y_test = true_ci(test_df, cfg)

        persistence_pred = test_df["historical_ci_24h_lag"].to_numpy(dtype=float)
        rows.append(_metric_row(region, "24h persistence", persistence_pred, y_test))

        direct_runs = [
            fit_direct_mlp_train_val(
                train_df[feature_cols].to_numpy(dtype=float),
                y_train,
                val_df[feature_cols].to_numpy(dtype=float),
                y_val,
                test_df[feature_cols].to_numpy(dtype=float),
                seed=seed,
            )
            for seed in DIRECT_SEEDS
        ]
        direct_preds = [run["test_pred"] for run in direct_runs]
        direct_pred = np.mean(direct_preds, axis=0)
        direct_row = _metric_row(region, "Direct MLP", direct_pred, y_test)
        direct_row["direct_seed_rmse_std"] = float(np.std([_rmse(p, y_test) for p in direct_preds], ddof=1))
        direct_row["direct_best_epochs"] = ",".join(
            str(run["best_epoch"]) for run in direct_runs
        )
        rows.append(direct_row)

        ensemble = pd.read_csv(REPO_ROOT / "results" / "fuelmix_dayahead" / "multiseed" / f"{region}_ensemble_predictions.csv")
        fuelmix_pred = ensemble["ci_pred"].to_numpy(dtype=float)
        region_fm = fuelmix_metrics[fuelmix_metrics["region"].str.lower().str.replace("-", "") == region.replace("isone", "isone")]
        if region_fm.empty:
            region_fm = fuelmix_metrics[fuelmix_metrics["region"] == LABELS[region]]
        balance_pct = None
        if not region_fm.empty and "balance_pct" in region_fm:
            balance_pct = float(region_fm.iloc[0]["balance_pct"])
        rows.append(_metric_row(region, "Fuel-mix ens.", fuelmix_pred, y_test, balance_pct=balance_pct))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "dayahead_method_metrics.csv", index=False)

    print(out[[
        "method",
        "region",
        "ci_rmse",
        "rel_rmse_pct",
        "ci_corr",
        "ci_r2",
        "balance_pct",
    ]].round(4).to_string(index=False))
    print(f"wrote {OUT_DIR / 'dayahead_method_metrics.csv'}")


if __name__ == "__main__":
    main()
