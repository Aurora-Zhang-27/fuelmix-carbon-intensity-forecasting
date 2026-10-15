"""
fuel_mix_trainer.py — region-agnostic FuelMixForecaster training pipeline
=========================================================================

Config-driven module that works for any ISO as long as you give it:

    1. A fuel-mix CSV (actual generation by fuel, hourly or 5-min)
    2. A load-forecast CSV (either zonal like ISO-NE, or flat system-level)
    3. A JSON config listing fuel_cols / emission_factors / load_locations
       (empty ⇒ flat system load) / fuel_col_map (raw → canonical)

The model architecture (share + total decomposition), the loss
(L_fuel + 50·L_ci + 0.2·L_balance + 0.5·L_fossil, eq. 18 of the paper),
and the training loop are identical to the ISONE case. Only the I/O +
column wiring is parameterized.

Typical usage — from the CLI:

    python src/run_fuelmix.py --config configs/nyiso_fuelmix.json

Or directly in Python:

    from fuel_mix_trainer import run_experiment, load_config
    cfg = load_config("configs/nyiso_fuelmix.json")
    result = run_experiment(cfg, device="cpu")
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# 1) Config helpers
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path) as f:
        cfg = json.load(f)
    # Stash the resolved root so path-resolution downstream is centralised
    cfg["_config_path"] = str(path)
    return cfg


def _abspath(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else REPO_ROOT / pp


def _read_csvs(paths) -> pd.DataFrame:
    """Read one CSV path or concatenate a list of yearly CSVs in order."""
    if isinstance(paths, (str, Path)):
        return pd.read_csv(_abspath(paths))
    return pd.concat([pd.read_csv(_abspath(p)) for p in paths], ignore_index=True)


def set_random_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2) Data loading
# ============================================================

def _detect_time_col(df: pd.DataFrame) -> str:
    for c in ["interval_start_utc", "UTC time", "timestamp",
              "Interval Start", "Time"]:
        if c in df.columns:
            return c
    raise ValueError(f"No recognized time column in: {list(df.columns)}")


def load_fuel_mix(cfg: dict) -> pd.DataFrame:
    """
    Returns a dataframe with columns: timestamp (utc), <canonical fuel_cols>,
    total_generation_mw. Handles either ISONE's bespoke CSV or gridstatusio
    _raw CSV by applying `fuel_col_map` if present.
    """
    df = _read_csvs(cfg["paths"]["fuel_mix_csv"])

    tcol = _detect_time_col(df)
    df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    df = df.dropna(subset=[tcol]).set_index(tcol)

    fuel_cols = cfg["dataset"]["fuel_cols"]
    fuel_col_map = cfg["dataset"].get("fuel_col_map", {})
    signed_flow_map = cfg["dataset"].get("signed_flow_map", {})
    clip_negative = cfg["dataset"].get("clip_negative_fuel_values", False)

    # Start with all raw cols summed into canonical buckets (if map provided);
    # otherwise assume the raw CSV already uses canonical names.
    if fuel_col_map:
        out = pd.DataFrame(index=df.index)
        for raw_col, canonical in fuel_col_map.items():
            if raw_col not in df.columns:
                print(f"    ! raw fuel col missing, skipping: {raw_col}")
                continue
            series = pd.to_numeric(df[raw_col], errors="coerce").fillna(0.0)
            if clip_negative:
                series = series.clip(lower=0.0)
            if canonical in out.columns:
                out[canonical] = out[canonical] + series.values
            else:
                out[canonical] = series.values
        for raw_col, flow_spec in signed_flow_map.items():
            if raw_col not in df.columns:
                print(f"    ! raw signed-flow col missing, skipping: {raw_col}")
                continue
            series = pd.to_numeric(df[raw_col], errors="coerce").fillna(0.0)
            positive_col = flow_spec["positive"]
            negative_col = flow_spec["negative"]
            out[positive_col] = out.get(positive_col, 0.0) + series.clip(lower=0.0)
            out[negative_col] = out.get(negative_col, 0.0) + (-series).clip(lower=0.0)
        # Zero-fill any canonical fuels that weren't in the map
        for c in fuel_cols:
            if c not in out.columns:
                out[c] = 0.0
    else:
        out = pd.DataFrame(index=df.index)
        for c in fuel_cols:
            if c in df.columns:
                out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
                if clip_negative:
                    out[c] = out[c].clip(lower=0.0)
            else:
                out[c] = 0.0

    # Hourly mean (gridstatusio server-side resample already did this for
    # our new data, but ISO-NE's bespoke CSV is 5-min)
    hourly = out[fuel_cols].resample("h").mean().fillna(0.0)
    hourly["total_generation_mw"] = hourly[fuel_cols].sum(axis=1)
    return hourly.reset_index().rename(columns={tcol: "timestamp"})


def load_load_forecast(cfg: dict) -> pd.DataFrame:
    """
    Returns a wide dataframe indexed by timestamp with at minimum a
    `system_load_forecast` column, plus optional `load_<zone>` / `pct_<zone>`
    columns for ISONE-style zonal data.
    """
    df = _read_csvs(cfg["paths"]["load_forecast_csv"])

    tcol = _detect_time_col(df)
    df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    df = df.dropna(subset=[tcol])

    load_locations = cfg["dataset"].get("load_locations", [])
    load_col = cfg["dataset"].get("load_col", "load_forecast")
    enforce_asof = cfg["dataset"].get("enforce_asof_publication", False)
    required_lead_hours = float(
        cfg["dataset"].get("required_forecast_lead_hours", 0.0)
    )
    load_filter_col = cfg["dataset"].get("load_filter_col")
    load_filter_value = cfg["dataset"].get("load_filter_value")
    if load_filter_col:
        if load_filter_col not in df.columns:
            raise KeyError(
                f"load_filter_col='{load_filter_col}' not in load CSV "
                f"(got cols={list(df.columns)})"
            )
        df = df[df[load_filter_col] == load_filter_value].copy()
        if df.empty:
            raise ValueError(
                f"No load rows remain after {load_filter_col}={load_filter_value!r}"
            )

    if load_locations and "location" in df.columns:
        # ISONE-style zonal: filter to the listed zones, pivot wide
        df = df[df["location"].isin(load_locations)].copy()

        if "publish_time_utc" in df.columns:
            df["publish_time_utc"] = pd.to_datetime(
                df["publish_time_utc"], utc=True, errors="coerce"
            )
            if enforce_asof:
                cutoff = df[tcol] - pd.to_timedelta(required_lead_hours, unit="h")
                df = df[df["publish_time_utc"] <= cutoff]
            df = (
                df.sort_values([tcol, "location", "publish_time_utc"])
                  .groupby([tcol, "location"], as_index=False)
                  .last()
            )

        wide = df.pivot_table(index=tcol, columns="location",
                              values=load_col, aggfunc="last"
                              ).reindex(columns=load_locations)

        pct = None
        if "regional_percentage" in df.columns:
            pct = df.pivot_table(index=tcol, columns="location",
                                 values="regional_percentage", aggfunc="last"
                                 ).reindex(columns=load_locations)

        wide.columns = [f"load_{c}" for c in wide.columns]
        out = wide.copy()
        if pct is not None:
            pct.columns = [f"pct_{c}" for c in pct.columns]
            out = pd.concat([out, pct], axis=1)
        out["system_load_forecast"] = wide.sum(axis=1)
    else:
        # Flat system-level (NYISO/PJM/ERCOT from gridstatusio)
        df = df.sort_values(tcol)
        # If multiple publish vintages per interval, keep the latest
        if "publish_time_utc" in df.columns:
            df["publish_time_utc"] = pd.to_datetime(
                df["publish_time_utc"], utc=True, errors="coerce"
            )
            if enforce_asof:
                cutoff = df[tcol] - pd.to_timedelta(required_lead_hours, unit="h")
                df = df[df["publish_time_utc"] <= cutoff]
            df = (df.sort_values([tcol, "publish_time_utc"])
                    .groupby(tcol, as_index=False).last())
        if load_col not in df.columns:
            raise KeyError(
                f"load_col='{load_col}' not in load CSV (got cols={list(df.columns)}). "
                "Set dataset.load_col in config to the right column name.")
        passthrough_cols = [
            c for c in df.columns
            if c.startswith("ci_")
            or c in {
                "local_generation_mw",
                "import_mw",
                "import_availability",
                "observed_local_ci",
                "import_share",
            }
        ]
        keep_cols = [load_col, *passthrough_cols]
        out = df.set_index(tcol)[keep_cols].rename(
            columns={load_col: "system_load_forecast"})

    # Hourly resample (guards against sub-hourly publish times)
    out = out.resample("h").mean()
    max_interp = int(cfg["dataset"].get("max_load_interpolation_hours", 0))
    if max_interp > 0:
        # Fill only gaps of at most max_interp hours; longer gaps stay missing
        # (and are dropped later) rather than being partly bridged.
        missing = out["system_load_forecast"].isna()
        run_length = missing.groupby((~missing).cumsum()).transform("sum")
        short = missing & (run_length <= max_interp)
        filled = out.interpolate(method="time", limit_area="inside")
        out.loc[short] = filled.loc[short]

    # Time-encoding features
    idx = out.index
    out["hour"] = idx.hour
    out["dow"] = idx.dayofweek
    out["month"] = idx.month
    out["sin_hour"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["cos_hour"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["sin_dow"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["cos_dow"] = np.cos(2 * np.pi * out["dow"] / 7)
    out["sin_month"] = np.sin(2 * np.pi * out["month"] / 12)
    out["cos_month"] = np.cos(2 * np.pi * out["month"] / 12)

    return out.reset_index().rename(columns={tcol: "timestamp"})


def load_renewable_forecast(cfg: dict) -> pd.DataFrame | None:
    """Load optional hourly renewable forecast features."""
    path = cfg.get("paths", {}).get("renewable_forecast_csv")
    if not path:
        return None
    df = _read_csvs(path)
    tcol = _detect_time_col(df)
    df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    df = df.dropna(subset=[tcol]).set_index(tcol)

    keep = [c for c in df.columns if c.startswith("renewable_forecast_")]
    if not keep:
        return None
    out = df[keep].apply(pd.to_numeric, errors="coerce").resample("h").mean()
    return out.reset_index().rename(columns={tcol: "timestamp"})


def build_supervised_table(cfg: dict) -> pd.DataFrame:
    X_df = load_load_forecast(cfg)
    R_df = load_renewable_forecast(cfg)
    if R_df is not None:
        X_df = X_df.merge(R_df, on="timestamp", how="left")
        renewable_cols = [
            c for c in X_df.columns if c.startswith("renewable_forecast_")
        ]
        if renewable_cols:
            X_df[renewable_cols] = (
                X_df[renewable_cols]
                .ffill()
                .bfill()
                .fillna(0.0)
            )
        if "renewable_forecast_total_mw" in X_df.columns:
            X_df["renewable_forecast_net_load_mw"] = (
                X_df["system_load_forecast"]
                - X_df["renewable_forecast_total_mw"]
            )
    Y_df = load_fuel_mix(cfg)
    data = (X_df.merge(Y_df, on="timestamp", how="inner")
                .sort_values("timestamp")
                .reset_index(drop=True))
    missing_load = data["system_load_forecast"].isna()

    feature_cfg = cfg.get("features", {})
    if feature_cfg.get("load_lags", True):
        load_lag_hours = feature_cfg.get("load_lag_hours", [1, 24])
        for h in load_lag_hours:
            h = int(h)
            data[f"load_{h}h_lag"] = (
                data["system_load_forecast"]
                .shift(h)
                .fillna(data["system_load_forecast"])
            )
        if 1 in [int(h) for h in load_lag_hours]:
            data["load_diff"] = data["system_load_forecast"].diff().fillna(0.0)

    if feature_cfg.get("historical_fuel_lags", False):
        lag_hours = feature_cfg.get("historical_lag_hours", [1, 24])
        fuel_cols = cfg["dataset"]["fuel_cols"]
        emission_factors = cfg["emission_factors"]
        _, ci = fuel_mix_to_carbon_intensity_np(
            data[fuel_cols].to_numpy(dtype=np.float32),
            emission_factors,
            fuel_cols,
            data["system_load_forecast"].to_numpy(dtype=np.float32),
        )
        data["historical_ci"] = ci

        renewable_cols = [c for c in ["wind", "solar", "hydro"]
                          if c in data.columns]
        fossil_cols = [c for c in cfg.get("fossil_fuels", [])
                       if c in data.columns]
        total = data[fuel_cols].sum(axis=1).replace(0, np.nan)
        if renewable_cols:
            data["historical_renewable_share"] = (
                data[renewable_cols].sum(axis=1) / total
            ).fillna(0.0)
        if fossil_cols:
            data["historical_fossil_share"] = (
                data[fossil_cols].sum(axis=1) / total
            ).fillna(0.0)

        historical_base_cols = [
            *fuel_cols,
            "historical_ci",
            "historical_renewable_share",
            "historical_fossil_share",
        ]
        historical_base_cols = [c for c in historical_base_cols if c in data.columns]
        max_lag = max(int(h) for h in lag_hours)
        for h in lag_hours:
            h = int(h)
            for c in historical_base_cols:
                data[f"{c}_{h}h_lag"] = data[c].shift(h)
        data = data.iloc[max_lag:].reset_index(drop=True)

    # Drop hours without a published load forecast and hours whose lagged
    # inputs fall in such a gap (rows are hourly and contiguous before this).
    lag_hours = [int(h) for h in feature_cfg.get("load_lag_hours", [1, 24])]
    if feature_cfg.get("historical_fuel_lags", False):
        lag_hours += [int(h) for h in feature_cfg.get("historical_lag_hours", [1, 24])]
    bad = missing_load.copy()
    for h in set(lag_hours):
        bad |= missing_load.shift(h, fill_value=False)
    bad = bad.iloc[len(missing_load) - len(data):].to_numpy()
    if bad.any():
        data = data[~bad].reset_index(drop=True)
    data.attrs["dropped_missing_load_hours"] = int(bad.sum())
    return data


def feature_columns_from_table(df: pd.DataFrame, cfg: dict) -> list[str]:
    feature_cfg = cfg.get("features", {})
    load_locations = cfg["dataset"].get("load_locations", [])
    zone_load = ([f"load_{c}" for c in load_locations]
                 if feature_cfg.get("zonal_load", bool(load_locations)) else [])
    zone_pct = ([f"pct_{c}" for c in load_locations]
                if feature_cfg.get("zonal_percentage", bool(load_locations)) else [])

    system_cols = (["system_load_forecast"]
                   if feature_cfg.get("system_load", True) else [])
    time_cols = [
        "sin_hour", "cos_hour",
        "sin_dow", "cos_dow",
        "sin_month", "cos_month",
    ] if feature_cfg.get("time", True) else []
    if feature_cfg.get("load_lags", True):
        load_lag_hours = feature_cfg.get("load_lag_hours", [1, 24])
        lag_cols = [f"load_{int(h)}h_lag" for h in load_lag_hours]
        if 1 in [int(h) for h in load_lag_hours]:
            lag_cols.append("load_diff")
    else:
        lag_cols = []

    optional_cols = []
    if feature_cfg.get("historical_fuel_lags", False):
        optional_cols.extend([
            c for c in df.columns
            if c.endswith("h_lag") and (
                c.startswith("historical_")
                or any(c.startswith(f"{fuel}_") for fuel in cfg["dataset"]["fuel_cols"])
            )
        ])
    if feature_cfg.get("renewable_forecast", False):
        optional_cols.extend([
            c for c in df.columns
            if c.startswith("renewable_forecast_")
            or c in ("wind_forecast", "solar_forecast", "net_load_forecast")
        ])
    if feature_cfg.get("weather", False):
        optional_cols.extend([c for c in df.columns if c.startswith("weather_")])

    # Keep only cols actually present in df (e.g. zonal features absent for
    # flat-load regions; pct absent when CSV has no regional_percentage)
    ordered = zone_load + zone_pct + system_cols + time_cols + lag_cols + optional_cols
    return [c for c in ordered if c in df.columns]


# ============================================================
# 3) Data validation & run metadata
# ============================================================

def _json_safe(value: Any):
    """Convert numpy/pandas scalars and timestamps to JSON-safe values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _expected_hours_from_config(cfg: dict) -> int | None:
    ds = cfg.get("data_source", {})
    start = ds.get("start_date")
    end = ds.get("end_date")
    if not start or not end:
        return None
    try:
        delta = pd.Timestamp(end, tz="UTC") - pd.Timestamp(start, tz="UTC")
        return int(delta.total_seconds() // 3600)
    except Exception:
        return None


def validate_supervised_table(
    df: pd.DataFrame,
    cfg: dict,
    feature_cols: list[str],
) -> dict:
    """Build a compact data-quality report before training."""
    region = cfg["dataset"].get("region_name", "UNKNOWN")
    fuel_cols = cfg["dataset"]["fuel_cols"]
    emission_factors = cfg["emission_factors"]
    required_cols = ["timestamp", "system_load_forecast", *fuel_cols]
    warnings: list[str] = []
    errors: list[str] = []

    report: dict[str, Any] = {
        "region": region,
        "status": "ok",
        "rows": int(len(df)),
        "required_columns": required_cols,
        "feature_columns": feature_cols,
        "warnings": warnings,
        "errors": errors,
    }

    if df.empty:
        errors.append("supervised table is empty")
        report["status"] = "error"
        return report
    if not feature_cols:
        errors.append("no feature columns selected")

    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        errors.append(f"missing required columns: {missing_required}")

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    bad_ts = int(ts.isna().sum())
    if bad_ts:
        errors.append(f"unparseable timestamps: {bad_ts}")

    ts_valid = ts.dropna().sort_values()
    if not ts_valid.empty:
        report["time_start"] = ts_valid.iloc[0]
        report["time_end"] = ts_valid.iloc[-1]
        report["duplicate_timestamps"] = int(ts.duplicated().sum())
        diffs = ts_valid.diff().dropna()
        non_hourly = int((diffs != pd.Timedelta(hours=1)).sum())
        report["non_hourly_steps"] = non_hourly
        if non_hourly:
            warnings.append(f"non-hourly timestamp steps detected: {non_hourly}")

        expected_hours = _expected_hours_from_config(cfg)
        if expected_hours is not None:
            report["expected_hours_from_config"] = expected_hours
            report["coverage_pct"] = round(len(df) / max(expected_hours, 1) * 100, 2)
            if len(df) < 0.95 * expected_hours:
                warnings.append(
                    f"row coverage below 95% of requested window: "
                    f"{len(df)}/{expected_hours}"
                )

    numeric_cols = [c for c in ["system_load_forecast", *fuel_cols, *feature_cols]
                    if c in df.columns]
    nan_counts = df[numeric_cols].isna().sum()
    nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
    report["nan_counts"] = nan_counts.astype(int).to_dict()
    if len(nan_counts):
        warnings.append(f"NaNs found in numeric columns: {dict(nan_counts.head(8))}")
        critical_nan_cols = [
            c for c in nan_counts.index
            if c in feature_cols or c in required_cols
        ]
        if critical_nan_cols:
            errors.append(f"NaNs in training-critical columns: {critical_nan_cols}")

    if "system_load_forecast" in df.columns:
        load = pd.to_numeric(df["system_load_forecast"], errors="coerce")
        neg_load = int((load < 0).sum())
        zero_load = int((load <= 0).sum())
        report["load_min_mw"] = float(load.min())
        report["load_mean_mw"] = float(load.mean())
        report["negative_load_rows"] = neg_load
        report["nonpositive_load_rows"] = zero_load
        if neg_load:
            errors.append(f"negative system load rows: {neg_load}")
        elif zero_load:
            warnings.append(f"non-positive system load rows: {zero_load}")

    negative_fuel = {}
    fuel_df = pd.DataFrame(index=df.index)
    for c in fuel_cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            fuel_df[c] = s
            nneg = int((s < 0).sum())
            if nneg:
                negative_fuel[c] = nneg
    report["negative_fuel_rows"] = negative_fuel
    if negative_fuel:
        errors.append(f"negative fuel generation rows: {negative_fuel}")

    if set(fuel_cols).issubset(df.columns) and "system_load_forecast" in df.columns:
        _, ci = fuel_mix_to_carbon_intensity_np(
            df[fuel_cols].to_numpy(dtype=np.float32),
            emission_factors,
            fuel_cols,
            df["system_load_forecast"].to_numpy(dtype=np.float32),
        )
        ci = pd.Series(ci)
        report["ci_min_t_per_mwh"] = float(ci.min())
        report["ci_mean_t_per_mwh"] = float(ci.mean())
        report["ci_max_t_per_mwh"] = float(ci.max())
        neg_ci = int((ci < -1e-9).sum())
        report["negative_ci_rows"] = neg_ci
        if neg_ci:
            errors.append(f"negative carbon-intensity rows: {neg_ci}")

    report["status"] = "error" if errors else ("warning" if warnings else "ok")
    return report


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_safe(payload), f, indent=2)


# ============================================================
# 4) Splits & standardization
# ============================================================

def split_train_val_test(df: pd.DataFrame, train_frac: float = 0.7,
                         val_frac: float = 0.15):
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return (df.iloc[:n_train].copy(),
            df.iloc[n_train:n_train + n_val].copy(),
            df.iloc[n_train + n_val:].copy())


def standardize_from_train(train_df, val_df, test_df, feature_cols):
    mu = train_df[feature_cols].mean()
    sigma = train_df[feature_cols].std().replace(0, 1.0)
    train_x = ((train_df[feature_cols] - mu) / sigma).to_numpy(dtype=np.float32)
    val_x = ((val_df[feature_cols] - mu) / sigma).to_numpy(dtype=np.float32)
    test_x = ((test_df[feature_cols] - mu) / sigma).to_numpy(dtype=np.float32)
    return train_x, val_x, test_x, mu, sigma


def standardize_targets_from_train(train_arr, val_arr, test_arr, eps=1e-6):
    mu = train_arr.mean(axis=0)
    sigma = train_arr.std(axis=0)
    sigma = np.where(sigma < eps, 1.0, sigma)
    return (
        ((train_arr - mu) / sigma).astype(np.float32),
        ((val_arr - mu) / sigma).astype(np.float32),
        ((test_arr - mu) / sigma).astype(np.float32),
        mu.astype(np.float32),
        sigma.astype(np.float32),
    )


# ============================================================
# 4) Carbon intensity utilities
# ============================================================

def fuel_mix_to_carbon_intensity_np(fuel_mix_mw: np.ndarray,
                                    emission_factors: dict,
                                    fuel_cols: list[str],
                                    demand_mw: np.ndarray):
    ef = np.array([emission_factors.get(c, 0.0) for c in fuel_cols],
                  dtype=np.float32)
    emissions_tph = np.sum(fuel_mix_mw * ef[None, :], axis=1)
    ci = emissions_tph / np.maximum(demand_mw, 1e-6)
    return emissions_tph, ci


def fuel_mix_to_carbon_intensity_torch(fuel_mix_mw, ef_torch, demand_mw):
    emissions_tph = torch.sum(fuel_mix_mw * ef_torch[None, :], dim=1)
    ci = emissions_tph / torch.clamp(demand_mw, min=1e-6)
    return emissions_tph, ci


# ============================================================
# 5) Models
# ============================================================

class FuelMixForecaster(nn.Module):
    """Features -> physical fuel-mix MW via shares and total generation."""
    def __init__(self, xdim, nfuel, hidden=128, total_scale=1.0):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(xdim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.share_head = nn.Linear(hidden, nfuel)
        self.total_head = nn.Linear(hidden, 1)
        self.register_buffer(
            "total_scale", torch.tensor(float(total_scale), dtype=torch.float32)
        )

    def forward(self, x):
        h = self.backbone(x)
        shares = torch.softmax(self.share_head(h), dim=1)
        total_gen = (
            torch.nn.functional.softplus(self.total_head(h)).squeeze(-1)
            * self.total_scale
        )
        fuel_mix = shares * total_gen[:, None]
        return fuel_mix, shares, total_gen


class SignedFlowFuelMixForecaster(nn.Module):
    """Local fuel shares plus independent nonnegative import/export/storage flows."""

    def __init__(self, xdim, fuel_cols, signed_flow_cols, hidden=128,
                 local_total_scale=1.0, flow_scales=None):
        super().__init__()
        self.fuel_cols = list(fuel_cols)
        self.signed_flow_cols = list(signed_flow_cols)
        self.local_fuel_cols = [
            c for c in self.fuel_cols if c not in self.signed_flow_cols
        ]
        self.backbone = nn.Sequential(
            nn.Linear(xdim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.local_share_head = nn.Linear(hidden, len(self.local_fuel_cols))
        self.local_total_head = nn.Linear(hidden, 1)
        self.flow_head = nn.Linear(hidden, len(self.signed_flow_cols))
        self.register_buffer(
            "local_total_scale",
            torch.tensor(float(local_total_scale), dtype=torch.float32),
        )
        self.register_buffer(
            "flow_scales",
            torch.tensor(flow_scales, dtype=torch.float32),
        )

    def forward(self, x):
        h = self.backbone(x)
        local_shares = torch.softmax(self.local_share_head(h), dim=1)
        local_total = (
            torch.nn.functional.softplus(self.local_total_head(h)).squeeze(-1)
            * self.local_total_scale
        )
        local_mix = local_shares * local_total[:, None]
        flows = torch.nn.functional.softplus(self.flow_head(h)) * self.flow_scales[None, :]

        local_idx = {c: i for i, c in enumerate(self.local_fuel_cols)}
        flow_idx = {c: i for i, c in enumerate(self.signed_flow_cols)}
        fuel_mix = torch.stack([
            flows[:, flow_idx[c]] if c in flow_idx else local_mix[:, local_idx[c]]
            for c in self.fuel_cols
        ], dim=1)
        total_positive_components = fuel_mix.sum(dim=1)
        shares = fuel_mix / torch.clamp(total_positive_components[:, None], min=1e-6)
        return fuel_mix, shares, total_positive_components


# ============================================================
# 6) Trainer
# ============================================================

class FuelMixTrainer:
    """
    Implements the paper's eq. 18 training objective:
        L = L_fuel + β_c·L_ci + β_b·L_bal + β_f·L_fossil

    Config-driven on fuel_cols, emission_factors, fossil_fuels. A direct
    refactor of ISONEFuelMixTrainer with ISONE-specific constants lifted.
    """
    def __init__(self, feature_cols, fuel_cols, emission_factors,
                 fossil_fuels=None, loss_weights=None,
                 hidden=128, lr=1e-3, device="cpu",
                 fuel_mu=None, fuel_sigma=None,
                 demand_mu=None, demand_sigma=None,
                 ci_pred_denominator="load", balance_coefficients=None,
                 signed_flow_cols=None):
        self.feature_cols = feature_cols
        self.fuel_cols = fuel_cols
        self.emission_factors = emission_factors
        self.device = device

        # Fossil-fuels for L_fossil term. Default = classical fossils seen
        # in CarbonCast schema + ISO-NE extras if they happen to be present.
        if fossil_fuels is None:
            candidates = ["coal", "natural_gas", "nat_gas", "oil",
                          "landfill_gas", "refuse", "dual_fuel"]
            fossil_fuels = [f for f in candidates if f in fuel_cols]
        self.fossil_fuels = fossil_fuels
        self.fossil_idx = [fuel_cols.index(f) for f in fossil_fuels
                           if f in fuel_cols]

        self.loss_weights = loss_weights or {
            "fuel": 1.0, "ci": 50.0, "balance": 0.2, "fossil": 0.5,
        }
        if ci_pred_denominator not in {"load", "predicted_generation"}:
            raise ValueError(
                "ci_pred_denominator must be 'load' or 'predicted_generation'"
            )
        self.ci_pred_denominator = ci_pred_denominator

        self.ef_torch = torch.tensor(
            [emission_factors.get(c, 0.0) for c in fuel_cols],
            dtype=torch.float32, device=device,
        )
        balance_coefficients = balance_coefficients or {}
        self.balance_coefficients = torch.tensor(
            [balance_coefficients.get(c, 1.0) for c in fuel_cols],
            dtype=torch.float32,
            device=device,
        )

        signed_flow_cols = signed_flow_cols or []
        if signed_flow_cols:
            local_idx = [i for i, c in enumerate(fuel_cols) if c not in signed_flow_cols]
            flow_idx = [fuel_cols.index(c) for c in signed_flow_cols]
            local_scale = (
                float(np.sum(np.asarray(fuel_mu)[local_idx]))
                if fuel_mu is not None else demand_mu
            )
            flow_scales = [
                max(float(np.asarray(fuel_mu)[i]), float(np.asarray(fuel_sigma)[i]), 1.0)
                for i in flow_idx
            ]
            self.model = SignedFlowFuelMixForecaster(
                xdim=len(feature_cols),
                fuel_cols=fuel_cols,
                signed_flow_cols=signed_flow_cols,
                hidden=hidden,
                local_total_scale=local_scale,
                flow_scales=flow_scales,
            ).to(device)
        else:
            self.model = FuelMixForecaster(
                xdim=len(feature_cols),
                nfuel=len(fuel_cols),
                hidden=hidden,
                total_scale=demand_mu if demand_mu is not None else 1.0,
            ).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.history: list[dict] = []

        def _to(v, shape_hint=None):
            if v is None:
                return None
            return torch.tensor(v, dtype=torch.float32, device=device)

        self.fuel_mu = _to(fuel_mu)
        self.fuel_sigma = _to(fuel_sigma)
        self.demand_mu = (torch.tensor([demand_mu], dtype=torch.float32,
                                       device=device)
                          if demand_mu is not None else None)
        self.demand_sigma = (torch.tensor([demand_sigma], dtype=torch.float32,
                                          device=device)
                             if demand_sigma is not None else None)

    def denorm_fuel(self, fuel_std):
        return fuel_std * self.fuel_sigma[None, :] + self.fuel_mu[None, :]

    def denorm_demand(self, demand_std):
        return demand_std * self.demand_sigma + self.demand_mu

    def norm_fuel(self, fuel_mw):
        return (fuel_mw - self.fuel_mu[None, :]) / self.fuel_sigma[None, :]

    def norm_demand(self, demand_mw):
        return (demand_mw - self.demand_mu) / self.demand_sigma

    def ci_prediction_denominator(self, total_gen_pred_mw, demand_true_mw):
        if self.ci_pred_denominator == "predicted_generation":
            return total_gen_pred_mw
        return demand_true_mw

    def net_supply(self, fuel_mix_mw):
        return torch.sum(
            fuel_mix_mw * self.balance_coefficients[None, :], dim=1
        )

    def loss_fn(self, fuel_pred_std, fuel_true_std,
                ci_pred, ci_true,
                total_gen_pred_std, demand_true_std):
        lw = self.loss_weights
        loss_fuel = torch.mean((fuel_pred_std - fuel_true_std) ** 2)
        loss_ci = torch.mean((ci_pred - ci_true) ** 2)
        loss_balance = torch.mean((total_gen_pred_std - demand_true_std) ** 2)

        if self.fossil_idx:
            loss_fossil = torch.mean(
                (fuel_pred_std[:, self.fossil_idx]
                 - fuel_true_std[:, self.fossil_idx]) ** 2
            )
        else:
            loss_fossil = torch.tensor(0.0, device=self.device)

        loss = (lw["fuel"] * loss_fuel
                + lw["ci"] * loss_ci
                + lw["balance"] * loss_balance
                + lw["fossil"] * loss_fossil)
        return loss, {
            "loss": float(loss.item()),
            "loss_fuel": float(loss_fuel.item()),
            "loss_ci": float(loss_ci.item()),
            "loss_balance": float(loss_balance.item()),
            "loss_fossil": float(loss_fossil.item()),
        }

    def step(self, x_batch, fuel_true_std_batch, demand_true_std_batch):
        self.model.train()
        self.optimizer.zero_grad()

        fuel_pred_mw, _, total_gen_pred_mw = self.model(x_batch)

        fuel_true_mw = self.denorm_fuel(fuel_true_std_batch)
        demand_true_mw = self.denorm_demand(demand_true_std_batch)
        fuel_pred_std = self.norm_fuel(fuel_pred_mw)
        net_supply_pred_mw = self.net_supply(fuel_pred_mw)
        total_gen_pred_std = self.norm_demand(net_supply_pred_mw)

        ci_pred_den = self.ci_prediction_denominator(total_gen_pred_mw, demand_true_mw)
        _, ci_pred = fuel_mix_to_carbon_intensity_torch(
            fuel_pred_mw, self.ef_torch, ci_pred_den)
        _, ci_true = fuel_mix_to_carbon_intensity_torch(
            fuel_true_mw, self.ef_torch, demand_true_mw)

        loss, logs = self.loss_fn(fuel_pred_std, fuel_true_std_batch,
                                  ci_pred, ci_true,
                                  total_gen_pred_std, demand_true_std_batch)
        loss.backward()
        self.optimizer.step()
        self.history.append(logs)
        return logs

    @torch.no_grad()
    def evaluate(self, x, fuel_true_std, demand_true_std):
        self.model.eval()
        fuel_pred_mw, shares_pred, total_gen_pred_mw = self.model(x)

        fuel_true_mw = self.denorm_fuel(fuel_true_std)
        demand_true_mw = self.denorm_demand(demand_true_std)
        net_supply_pred_mw = self.net_supply(fuel_pred_mw)

        ci_pred_den = self.ci_prediction_denominator(total_gen_pred_mw, demand_true_mw)
        _, ci_pred = fuel_mix_to_carbon_intensity_torch(
            fuel_pred_mw, self.ef_torch, ci_pred_den)
        _, ci_true = fuel_mix_to_carbon_intensity_torch(
            fuel_true_mw, self.ef_torch, demand_true_mw)

        ci_pred_np = ci_pred.cpu().numpy()
        ci_true_np = ci_true.cpu().numpy()
        metrics = {
            "fuel_rmse_mw": float(torch.sqrt(torch.mean(
                (fuel_pred_mw - fuel_true_mw) ** 2)).item()),
            "ci_mae_t_per_mwh": float(torch.mean(
                torch.abs(ci_pred - ci_true)).item()),
            "ci_rmse_t_per_mwh": float(torch.sqrt(torch.mean(
                (ci_pred - ci_true) ** 2)).item()),
            "balance_mae_mw": float(torch.mean(
                torch.abs(net_supply_pred_mw - demand_true_mw)).item()),
        }
        metrics.update(evaluate_trend_metrics(ci_pred_np, ci_true_np))
        outputs = {
            "fuel_pred": fuel_pred_mw.cpu().numpy(),
            "fuel_true": fuel_true_mw.cpu().numpy(),
            "shares_pred": shares_pred.cpu().numpy(),
            "total_gen_pred": total_gen_pred_mw.cpu().numpy(),
            "net_supply_pred": net_supply_pred_mw.cpu().numpy(),
            "ci_pred": ci_pred_np,
            "ci_true": ci_true_np,
            "demand_true": demand_true_mw.cpu().numpy(),
        }
        return metrics, outputs


# ============================================================
# 7) Utilities
# ============================================================

def to_torch(arr, device):
    return torch.tensor(arr, dtype=torch.float32, device=device)


def make_model_arrays(
    df,
    feature_cols,
    fuel_cols,
    emission_factors,
    demand_col="system_load_forecast",
):
    X = df[feature_cols].to_numpy(dtype=np.float32)
    Y_fuel = df[fuel_cols].to_numpy(dtype=np.float32)
    if demand_col not in df.columns:
        raise KeyError(f"demand_col='{demand_col}' not found in supervised table")
    demand = df[demand_col].to_numpy(dtype=np.float32)
    _, Y_ci = fuel_mix_to_carbon_intensity_np(
        Y_fuel, emission_factors, fuel_cols, demand)
    return X, Y_fuel, demand.astype(np.float32), Y_ci.astype(np.float32)


def iterate_minibatches(x, y_fuel, y_demand, batch_size=64, shuffle=True):
    n = x.shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, n, batch_size):
        j = idx[i:i + batch_size]
        yield x[j], y_fuel[j], y_demand[j]


def evaluate_ci_metrics(ci_pred, ci_true, prefix=""):
    err = ci_pred - ci_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    safe = np.abs(ci_true) > 1e-6
    mape = float(np.mean(np.abs(err[safe] / ci_true[safe])) * 100)
    return {f"{prefix}ci_mae": mae,
            f"{prefix}ci_rmse": rmse,
            f"{prefix}ci_mape": mape}


def evaluate_trend_metrics(ci_pred, ci_true) -> dict:
    """Trend-oriented diagnostics for carbon-intensity forecasts."""
    ci_pred = np.asarray(ci_pred, dtype=float)
    ci_true = np.asarray(ci_true, dtype=float)
    err = ci_pred - ci_true
    true_std = float(np.std(ci_true))
    pred_std = float(np.std(ci_pred))
    corr = (float(np.corrcoef(ci_true, ci_pred)[0, 1])
            if true_std > 0 and pred_std > 0 else np.nan)
    sse = float(np.sum(err ** 2))
    sst = float(np.sum((ci_true - np.mean(ci_true)) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else np.nan
    std_ratio = float(pred_std / true_std) if true_std > 0 else np.nan

    true_ramp = np.diff(ci_true)
    pred_ramp = np.diff(ci_pred)
    if len(true_ramp) and np.std(true_ramp) > 0 and np.std(pred_ramp) > 0:
        ramp_corr = float(np.corrcoef(true_ramp, pred_ramp)[0, 1])
    else:
        ramp_corr = np.nan
    ramp_rmse = (float(np.sqrt(np.mean((pred_ramp - true_ramp) ** 2)))
                 if len(true_ramp) else np.nan)
    direction_acc = (float(np.mean(np.sign(true_ramp) == np.sign(pred_ramp)) * 100)
                     if len(true_ramp) else np.nan)

    return {
        "ci_corr": corr,
        "ci_r2": r2,
        "ci_true_std": true_std,
        "ci_pred_std": pred_std,
        "ci_std_ratio": std_ratio,
        "ci_ramp_rmse": ramp_rmse,
        "ci_ramp_corr": ramp_corr,
        "ci_direction_acc_pct": direction_acc,
    }


# ============================================================
# 8) Plotting (optional — caller can skip)
# ============================================================

def plot_training_history(trainer, savepath):
    import matplotlib.pyplot as plt
    hist = pd.DataFrame(trainer.history)
    plt.figure(figsize=(8, 4))
    for col, label in [("loss", "total"), ("loss_fuel", "fuel"),
                       ("loss_ci", "carbon"), ("loss_balance", "balance")]:
        if col in hist.columns:
            plt.plot(hist.index, hist[col], label=label)
    plt.xlabel("Iteration"); plt.ylabel("Loss")
    plt.title("Training losses"); plt.legend()
    plt.tight_layout(); plt.savefig(savepath, dpi=200); plt.close()
    print(f"Saved {savepath}")


def representative_window_start(ci_true, ci_pred, window=48):
    """Pick a window whose RMSE is closest to full-test RMSE."""
    n = len(ci_true)
    if n <= window:
        return 0
    full_rmse = np.sqrt(np.mean((ci_pred - ci_true) ** 2))
    best_start = 0
    best_gap = float("inf")
    for start in range(0, n - window + 1):
        end = start + window
        window_rmse = np.sqrt(np.mean((ci_pred[start:end] - ci_true[start:end]) ** 2))
        gap = abs(window_rmse - full_rmse)
        if gap < best_gap:
            best_gap = gap
            best_start = start
    return best_start


def plot_48h_outputs(outputs, fuel_cols, savepath_prefix):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(savepath_prefix) or ".", exist_ok=True)

    window = min(48, len(outputs["ci_true"]))
    start = representative_window_start(
        outputs["ci_true"], outputs["ci_pred"], window=window)
    end = start + window
    h = np.arange(window)
    ci_true = outputs["ci_true"][start:end]
    ci_pred = outputs["ci_pred"][start:end]
    demand = outputs["demand_true"][start:end]
    fuel_true = outputs["fuel_true"][start:end]
    fuel_pred = outputs["fuel_pred"][start:end]

    plt.figure(figsize=(9, 3.8))
    plt.plot(h, ci_true, label="true CI")
    plt.plot(h, ci_pred, label="forecasted CI")
    plt.xlabel("Hour in selected test window"); plt.ylabel("tCO2/MWh")
    plt.title("48-hour carbon-intensity forecast")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{savepath_prefix}_emission.png", dpi=200); plt.close()

    plt.figure(figsize=(9, 3.8))
    plt.plot(h, demand, label="system load forecast")
    plt.plot(h, fuel_true.sum(axis=1), label="true total generation")
    plt.plot(h, fuel_pred.sum(axis=1), label="pred total generation")
    plt.xlabel("Hour in selected test window"); plt.ylabel("MW")
    plt.title("48-hour demand & total generation")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{savepath_prefix}_system.png", dpi=200); plt.close()

    topk = min(5, len(fuel_cols))
    fuel_idx = np.argsort(fuel_true.mean(axis=0))[::-1][:topk]
    plt.figure(figsize=(10, 5))
    for j in fuel_idx:
        plt.plot(h, fuel_true[:, j], "-", label=f"{fuel_cols[j]}_true")
        plt.plot(h, fuel_pred[:, j], "--", label=f"{fuel_cols[j]}_pred")
    plt.xlabel("Hour in selected test window"); plt.ylabel("MW")
    plt.title("48-hour fuel-mix forecast – top fuels")
    plt.legend(ncol=2, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(f"{savepath_prefix}_fuelmix.png", dpi=200); plt.close()

    print(f"Saved 48-hour plots to {savepath_prefix}_*.png")


# ============================================================
# 9) End-to-end runner
# ============================================================

def run_experiment(cfg: dict, device: str = "cpu", make_plots: bool = True,
                   seed: int | None = None
                   ) -> dict:
    region = cfg["dataset"].get("region_name", "UNKNOWN")
    if seed is None:
        seed = cfg.get("training", {}).get("seed")
    set_random_seed(seed)
    print(f"[{region}] building supervised table")
    data = build_supervised_table(cfg)

    max_samples = cfg.get("debug", {}).get("max_samples")
    if max_samples and len(data) > max_samples:
        data = data.iloc[:max_samples].copy()
        print(f"[{region}] limiting to {max_samples} samples")

    feature_cols = feature_columns_from_table(data, cfg)
    fuel_cols = cfg["dataset"]["fuel_cols"]
    emission_factors = cfg["emission_factors"]
    fuelmix_denominator_col = cfg["dataset"].get(
        "fuelmix_denominator_col", "system_load_forecast"
    )
    ci_pred_denominator = cfg["dataset"].get("ci_pred_denominator", "load")
    save_dir = _abspath(cfg["output"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    quality_report = validate_supervised_table(data, cfg, feature_cols)
    write_json(save_dir / "data_quality_report.json", quality_report)

    print(f"[{region}] dataset size: {data.shape}")
    print(f"[{region}] feature cols ({len(feature_cols)}): {feature_cols}")
    print(f"[{region}] fuel cols ({len(fuel_cols)}): {fuel_cols}")
    print(f"[{region}] data quality: {quality_report['status']} "
          f"(report: {save_dir / 'data_quality_report.json'})")
    for msg in quality_report.get("warnings", []):
        print(f"  ! warning: {msg}")
    if quality_report.get("errors"):
        for msg in quality_report["errors"]:
            print(f"  ! error: {msg}")
        raise ValueError(
            f"[{region}] data validation failed; see "
            f"{save_dir / 'data_quality_report.json'}"
        )

    split_cfg = cfg.get("split", {})
    train_df, val_df, test_df = split_train_val_test(
        data,
        train_frac=split_cfg.get("train_frac", 0.7),
        val_frac=split_cfg.get("val_frac", 0.15),
    )
    print(f"[{region}] train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    # Feature standardization
    train_x_np, val_x_np, test_x_np, *_ = standardize_from_train(
        train_df, val_df, test_df, feature_cols)

    # Target arrays
    _, Yfuel_tr, Dtr, Yci_tr = make_model_arrays(
        train_df, feature_cols, fuel_cols, emission_factors, fuelmix_denominator_col)
    _, Yfuel_va, Dva, Yci_va = make_model_arrays(
        val_df, feature_cols, fuel_cols, emission_factors, fuelmix_denominator_col)
    _, Yfuel_te, Dte, Yci_te = make_model_arrays(
        test_df, feature_cols, fuel_cols, emission_factors, fuelmix_denominator_col)

    (Yfuel_tr_std, Yfuel_va_std, Yfuel_te_std,
     fuel_mu, fuel_sigma) = standardize_targets_from_train(
        Yfuel_tr, Yfuel_va, Yfuel_te)

    (Dtr_std, Dva_std, Dte_std,
     demand_mu_arr, demand_sigma_arr) = standardize_targets_from_train(
        Dtr[:, None], Dva[:, None], Dte[:, None])
    Dtr_std = Dtr_std.reshape(-1)
    Dva_std = Dva_std.reshape(-1)
    Dte_std = Dte_std.reshape(-1)
    demand_mu = float(demand_mu_arr.reshape(-1)[0])
    demand_sigma = float(demand_sigma_arr.reshape(-1)[0])

    # Torch tensors
    Xtr = to_torch(train_x_np, device)
    Xva = to_torch(val_x_np, device)
    Xte = to_torch(test_x_np, device)
    YF_tr = to_torch(Yfuel_tr_std, device)
    YF_va = to_torch(Yfuel_va_std, device)
    YF_te = to_torch(Yfuel_te_std, device)
    D_tr = to_torch(Dtr_std, device)
    D_va = to_torch(Dva_std, device)
    D_te = to_torch(Dte_std, device)

    # Trainer
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    trainer = FuelMixTrainer(
        feature_cols=feature_cols,
        fuel_cols=fuel_cols,
        emission_factors=emission_factors,
        fossil_fuels=cfg.get("fossil_fuels"),
        loss_weights=cfg.get("loss_weights"),
        hidden=model_cfg.get("hidden_size", 128),
        lr=training_cfg.get("learning_rate", 1e-3),
        device=device,
        fuel_mu=fuel_mu, fuel_sigma=fuel_sigma,
        demand_mu=demand_mu, demand_sigma=demand_sigma,
        ci_pred_denominator=ci_pred_denominator,
        balance_coefficients=cfg["dataset"].get("balance_coefficients"),
        signed_flow_cols=cfg["dataset"].get("signed_flow_cols"),
    )

    # Training loop with early stopping
    epochs = training_cfg.get("max_epochs", 40)
    batch_size = training_cfg.get("batch_size", 64)
    patience = training_cfg.get("patience", 5)
    min_epochs = training_cfg.get("min_epochs", 0)

    best_val_rmse = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        logs = []
        for xb, yb_fuel, yb_dem in iterate_minibatches(
                Xtr, YF_tr, D_tr, batch_size=batch_size):
            logs.append(trainer.step(xb, yb_fuel, yb_dem))
        mean_log = {k: np.mean([z[k] for z in logs]) for k in logs[0]}
        val_metrics, _ = trainer.evaluate(Xva, YF_va, D_va)
        val_rmse = val_metrics["ci_rmse_t_per_mwh"]
        print(f"[{region}] epoch {epoch:03d}  "
              f"train loss {mean_log['loss']:.4f}  "
              f"val CI RMSE {val_rmse:.5f}")

        if epoch >= min_epochs and val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.clone()
                          for k, v in trainer.model.state_dict().items()}
            no_improve = 0
        elif epoch >= min_epochs:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{region}] early stopping at epoch {epoch}")
                break

    if best_state is not None:
        trainer.model.load_state_dict(best_state)
    print(f"[{region}] restored best model (val CI RMSE = {best_val_rmse:.5f})")

    # Final evaluation
    train_metrics, _ = trainer.evaluate(Xtr, YF_tr, D_tr)
    val_metrics, _ = trainer.evaluate(Xva, YF_va, D_va)
    test_metrics, test_outputs = trainer.evaluate(Xte, YF_te, D_te)

    print(f"\n[{region}] test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.6f}")

    # Save trained model
    torch.save(trainer.model.state_dict(), save_dir / "fuelmix_model.pt")

    # Save metrics CSVs
    pd.DataFrame([{"region": region, **test_metrics}]).to_csv(
        save_dir / "test_metrics.csv", index=False)
    pd.DataFrame(trainer.history).to_csv(
        save_dir / "training_history.csv", index=False)

    # Also drop per-hour prediction series for the test window
    pd.DataFrame({
        "timestamp": test_df["timestamp"].to_numpy(),
        "ci_true": test_outputs["ci_true"],
        "ci_pred": test_outputs["ci_pred"],
        "demand_true": test_outputs["demand_true"],
        "total_gen_pred": test_outputs["total_gen_pred"],
        "net_supply_pred": test_outputs["net_supply_pred"],
    }).to_csv(save_dir / "test_predictions.csv", index=False)

    # Fuel-level series for contribution diagnostics (predicted and realized MW)
    fuel_frame = {
        "timestamp": test_df["timestamp"].to_numpy(),
        "demand_true": test_outputs["demand_true"],
    }
    for j, fuel in enumerate(fuel_cols):
        fuel_frame[f"pred_{fuel}"] = test_outputs["fuel_pred"][:, j]
        fuel_frame[f"true_{fuel}"] = test_outputs["fuel_true"][:, j]
    pd.DataFrame(fuel_frame).to_csv(
        save_dir / "test_fuel_predictions.csv", index=False)

    if make_plots:
        try:
            plot_training_history(
                trainer, str(save_dir / "training_loss.png"))
            plot_48h_outputs(
                test_outputs, fuel_cols, str(save_dir / "test48"))
        except Exception as ex:
            print(f"  ! plotting skipped: {ex}")

    run_summary = {
        "region": region,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": cfg.get("_config_path"),
        "data_source": cfg.get("data_source", {}),
        "seed": seed,
        "device": device,
        "rows": len(data),
        "time_start": data["timestamp"].min(),
        "time_end": data["timestamp"].max(),
        "split_rows": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "feature_columns": feature_cols,
        "fuel_columns": fuel_cols,
        "loss_weights": cfg.get("loss_weights"),
        "model": cfg.get("model", {}),
        "training": cfg.get("training", {}),
        "best_val_ci_rmse_t_per_mwh": best_val_rmse,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "data_quality_status": quality_report["status"],
        "data_quality_report": str(save_dir / "data_quality_report.json"),
        "outputs": {
            "model": str(save_dir / "fuelmix_model.pt"),
            "test_metrics": str(save_dir / "test_metrics.csv"),
            "test_predictions": str(save_dir / "test_predictions.csv"),
            "test_fuel_predictions": str(save_dir / "test_fuel_predictions.csv"),
            "training_history": str(save_dir / "training_history.csv"),
        },
    }
    write_json(save_dir / "run_summary.json", run_summary)
    print(f"[{region}] wrote run summary: {save_dir / 'run_summary.json'}")

    return {
        "region": region,
        "feature_cols": feature_cols,
        "fuel_cols": fuel_cols,
        "data_quality_report": quality_report,
        "run_summary": run_summary,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_outputs": test_outputs,
        "trainer": trainer,
        "save_dir": str(save_dir),
    }


if __name__ == "__main__":
    # Quick smoke-test hook: requires a config path
    import sys
    if len(sys.argv) < 2:
        print("usage: python src/fuel_mix_trainer.py <config.json>")
        sys.exit(1)
    cfg = load_config(sys.argv[1])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    run_experiment(cfg, device=device)
