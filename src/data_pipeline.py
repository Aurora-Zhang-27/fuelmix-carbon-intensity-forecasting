"""
Unified Data Pipeline for Multi-Region Carbon Emission Forecasting
===================================================================
Pulls raw grid data from gridstatusio and caches the raw fuel-mix and
load-forecast CSVs used by `run_fuelmix.py` / `fuel_mix_trainer.py`.

Expected config block (add under a new "data_source" key in each region's JSON):

    "data_source": {
        "provider": "gridstatusio",
        "start_date": "2023-01-01",
        "end_date":   "2024-12-31",
        "chunk_days": 30,
        "sleep_sec":  2,

        "fuel_mix": {
            "dataset": "nyiso_fuel_mix",        # gridstatusio dataset name
            "fuel_col_map": {                   # raw col -> CarbonCast fuel col
                "dual_fuel":   "nat_gas",
                "natural_gas": "nat_gas",
                "nuclear":     "nuclear",
                "hydro":       "hydro",
                "wind":        "wind",
                "other_renewables": "other",
                "other_fossil_fuels": "oil"
            }
        },

        "load_forecast": {
            "dataset": "nyiso_load_forecast",
            "load_col": "load_forecast"         # raw col for load forecast
        },

        # If weather is available via a separate source (meteostat, NOAA),
        # a future adapter will populate this block. For now pipeline skips
        # weather if weather_csv already exists on disk, OR falls back to
        # coarse proxies (lag features of fuel mix).
        "weather": {
            "provider": "existing_csv"          # uses paths.weather_csv as-is
        }
    }

Produces (paths taken from config["paths"]):
    - {region}_fuel_mix_raw.csv       (cached raw fuel mix, for debugging)
    - {region}_load_forecast_raw.csv  (cached raw load forecast)
    - fuel_mix_csv       — cached raw fuel mix
    - load_forecast_csv  — cached raw load forecast
    - renewable_forecast_csv, when configured

Dependency: pip install gridstatusio pandas
"""

from __future__ import annotations

import os
import time
import json
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

os.environ.setdefault("GSIO_SKIP_VERSION_CHECK", "true")


# ─────────────────────────────────────────────────────────────────────────────
# Date chunking
# ─────────────────────────────────────────────────────────────────────────────

def date_chunks(start: str, end: str, days: int):
    """Yield (chunk_start, chunk_end) as YYYY-MM-DD strings, half-open [s, e)."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end,   "%Y-%m-%d")
    while s < e:
        chunk_end = min(s + timedelta(days=days), e)
        yield s.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        s = chunk_end


# ─────────────────────────────────────────────────────────────────────────────
# gridstatusio fetch helpers
# ─────────────────────────────────────────────────────────────────────────────

_RESAMPLE_KWARG_CACHE: Optional[str] = None

# gridstatusio REST API only accepts these exact strings as the resample
# frequency (error message returned by the server when anything else is
# passed). We also accept common pandas-style aliases and map them.
_API_VALID_FREQS = {
    "1 minute", "5 minutes", "10 minutes", "15 minutes",
    "1 hour", "1 day", "1 week", "1 month", "1 year",
}
_PANDAS_TO_API = {
    # minutes
    "1min": "1 minute", "1t": "1 minute", "1 min": "1 minute",
    "5min": "5 minutes", "5t": "5 minutes", "5 min": "5 minutes",
    "10min": "10 minutes", "10t": "10 minutes", "10 min": "10 minutes",
    "15min": "15 minutes", "15t": "15 minutes", "15 min": "15 minutes",
    # hours
    "1h": "1 hour", "h": "1 hour", "60min": "1 hour",
    # days / weeks / months / years
    "1d": "1 day", "d": "1 day",
    "1w": "1 week", "w": "1 week",
    "1m": "1 month", "m": "1 month", "1mo": "1 month",
    "1y": "1 year", "y": "1 year", "1a": "1 year",
}


def _normalize_freq(freq: str) -> str:
    """pandas-style or API-native → gridstatusio-accepted string."""
    if freq is None:
        return freq
    f = str(freq).strip().lower()
    if f in _API_VALID_FREQS:
        return f
    if f in _PANDAS_TO_API:
        return _PANDAS_TO_API[f]
    # unrecognized — pass through; server will complain, user sees clear error
    return freq


def _resample_kwargs(freq: Optional[str]) -> dict:
    """
    Return the kwargs dict to request server-side downsampling from the
    gridstatusio client. Different versions of the client use different
    parameter names (`resample_frequency` in 0.5+, `resample` in older
    releases). We inspect the signature once and cache the winner. The
    value is normalized from pandas-style (e.g. "1h") to the natural-
    language strings the gridstatusio REST API expects (e.g. "1 hour").

    Pass freq=None or "none" to skip server-side resampling.
    """
    if freq in (None, "", "none", "None"):
        return {}
    normalized = _normalize_freq(freq)
    global _RESAMPLE_KWARG_CACHE
    if _RESAMPLE_KWARG_CACHE is None:
        try:
            import inspect
            from gridstatusio import GridStatusClient
            sig = inspect.signature(GridStatusClient.get_dataset)
            params = sig.parameters
            if "resample_frequency" in params:
                _RESAMPLE_KWARG_CACHE = "resample_frequency"
            elif "resample" in params:
                _RESAMPLE_KWARG_CACHE = "resample"
            else:
                # Unknown — don't pass a kwarg; client-side pandas resample
                # in raw_fuel_to_carboncast() will still catch it.
                _RESAMPLE_KWARG_CACHE = ""
        except Exception:
            _RESAMPLE_KWARG_CACHE = ""
    if not _RESAMPLE_KWARG_CACHE:
        return {}
    return {_RESAMPLE_KWARG_CACHE: normalized}


def _fetch_dataset(client, dataset_name: str, start: str, end: str,
                   chunk_days: int, sleep_sec: float,
                   extra_kwargs: Optional[dict] = None) -> pd.DataFrame:
    """Call client.get_dataset in chunks and concat. Tolerant to empty chunks."""
    chunks = list(date_chunks(start, end, chunk_days))
    frames = []
    for i, (s, e) in enumerate(chunks):
        print(f"  [{i+1:>3}/{len(chunks)}] {dataset_name}  {s} → {e} ...",
              end=" ", flush=True)
        try:
            kwargs = {"start": s, "end": e, "verbose": False}
            if extra_kwargs:
                kwargs.update(extra_kwargs)
            df = client.get_dataset(dataset_name, **kwargs)
            frames.append(df)
            print(f"✓ {len(df):>6} rows")
        except Exception as ex:
            print(f"✗ {ex}")
        if i < len(chunks) - 1:
            time.sleep(sleep_sec)
    if not frames:
        raise RuntimeError(f"No data returned for {dataset_name} "
                           f"over {start} → {end}")
    return pd.concat(frames, ignore_index=True)


def _detect_time_col(df: pd.DataFrame) -> str:
    """Find the column that carries the interval start timestamp."""
    for c in ["interval_start_utc", "interval_start_local",
              "Interval Start", "Time", "timestamp"]:
        if c in df.columns:
            return c
    raise ValueError(f"No recognized time column in: {list(df.columns)}")


def latest_published_hourly_forecast(
    raw: pd.DataFrame,
    forecast_cols: dict[str, str],
    datetime_out: str = "timestamp",
) -> pd.DataFrame:
    """
    Collapse a rolling forecast table to one row per target hour.

    gridstatusio forecast datasets often contain many publish vintages for the
    same `interval_start_utc`. To avoid target leakage, we keep the latest row
    whose publish time is not after the target interval start.
    """
    df = raw.copy()
    tcol = _detect_time_col(df)
    if "publish_time_utc" not in df.columns:
        raise ValueError("renewable forecast data needs publish_time_utc")

    df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    df["publish_time_utc"] = pd.to_datetime(
        df["publish_time_utc"], utc=True, errors="coerce"
    )
    df = df.dropna(subset=[tcol, "publish_time_utc"])
    df = df[df["publish_time_utc"] <= df[tcol]].copy()
    if df.empty:
        raise RuntimeError("No non-leaking forecast vintages found")

    df = (
        df.sort_values([tcol, "publish_time_utc"])
          .groupby(tcol, as_index=False)
          .tail(1)
    )

    out = pd.DataFrame({datetime_out: df[tcol]})
    for raw_col, out_col in forecast_cols.items():
        if raw_col in df.columns:
            out[out_col] = pd.to_numeric(df[raw_col], errors="coerce")
        else:
            out[out_col] = np.nan

    return (
        out.set_index(datetime_out)
           .resample("h")
           .mean(numeric_only=True)
           .reset_index()
    )


def build_ercot_renewable_forecast(
    wind_raw: pd.DataFrame,
    solar_raw: pd.DataFrame,
) -> pd.DataFrame:
    """Build hourly ERCOT renewable forecast features from wind/solar feeds."""
    wind = latest_published_hourly_forecast(
        wind_raw,
        {
            "stwpf_system_wide": "renewable_forecast_wind_mw",
            "wgrpp_system_wide": "renewable_forecast_wind_potential_mw",
            "cop_hsl_system_wide": "renewable_forecast_wind_hsl_mw",
        },
    )
    solar = latest_published_hourly_forecast(
        solar_raw,
        {
            "stppf_system_wide": "renewable_forecast_solar_mw",
            "pvgrpp_system_wide": "renewable_forecast_solar_potential_mw",
            "cop_hsl_system_wide": "renewable_forecast_solar_hsl_mw",
        },
    )

    out = wind.merge(solar, on="timestamp", how="outer").sort_values("timestamp")
    out["renewable_forecast_total_mw"] = (
        out["renewable_forecast_wind_mw"].fillna(0.0)
        + out["renewable_forecast_solar_mw"].fillna(0.0)
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Transformations: raw gridstatusio → CarbonCast schema
# ─────────────────────────────────────────────────────────────────────────────

CARBONCAST_FUEL_COLS = [
    "coal", "nat_gas", "nuclear", "oil", "hydro",
    "wind", "solar", "other",
]


def raw_fuel_to_carboncast(raw: pd.DataFrame, fuel_col_map: dict,
                           datetime_out: str = "UTC time") -> pd.DataFrame:
    """
    Convert a gridstatusio fuel mix dataframe to CarbonCast-style hourly
    actuals with columns:  UTC time, coal, nat_gas, nuclear, oil, hydro,
    wind, solar, other.

    `fuel_col_map` maps raw column names to CarbonCast fuel names.
    If multiple raw cols map to the same CarbonCast fuel, they are summed.
    Resamples to hourly mean (most gridstatusio feeds are 5-min).
    """
    df = raw.copy()
    tcol = _detect_time_col(df)
    df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    df = df.dropna(subset=[tcol])

    # Keep only mapped columns, sum into CarbonCast buckets
    out = pd.DataFrame({datetime_out: df[tcol]})
    for raw_col, cc_col in fuel_col_map.items():
        if raw_col not in df.columns:
            print(f"    ! raw fuel col missing, skipping: {raw_col}")
            continue
        series = pd.to_numeric(df[raw_col], errors="coerce").fillna(0.0)
        if cc_col in out.columns:
            out[cc_col] = out[cc_col] + series.values
        else:
            out[cc_col] = series.values

    # Ensure every CarbonCast fuel column exists (zero-fill missing)
    for cc_col in CARBONCAST_FUEL_COLS:
        if cc_col not in out.columns:
            out[cc_col] = 0.0

    # Resample to hourly means (gridstatusio typically 5-min granular)
    out = (out
           .set_index(datetime_out)
           .resample("1h")
           .mean(numeric_only=True)
           .reset_index())

    # Order columns
    out = out[[datetime_out] + CARBONCAST_FUEL_COLS]
    return out


def add_carbon_intensity(fuel_df: pd.DataFrame, emission_factors: dict,
                         datetime_col: str = "UTC time") -> pd.DataFrame:
    """
    Compute carbon_intensity (kg CO2e / MWh) from per-fuel generation and
    emission factors. Emission factors may be keyed either by plain fuel
    name ("nat_gas") or by the structured-surrogate naming
    ("avg_nat_gas_production_forecast"); we accept both.
    """
    df = fuel_df.copy()

    # Normalize emission_factors keys to plain fuel names
    ef = {}
    for k, v in emission_factors.items():
        if k.startswith("avg_") and k.endswith("_production_forecast"):
            fuel = k[len("avg_"):-len("_production_forecast")]
        else:
            fuel = k
        ef[fuel] = float(v)

    fuels_in_df = [c for c in CARBONCAST_FUEL_COLS if c in df.columns]
    weighted = np.zeros(len(df), dtype=float)
    total_gen = np.zeros(len(df), dtype=float)
    for fuel in fuels_in_df:
        factor = ef.get(fuel, 0.0)
        gen = df[fuel].astype(float).fillna(0.0).values
        weighted += factor * gen
        total_gen += gen

    with np.errstate(divide="ignore", invalid="ignore"):
        ci = np.where(total_gen > 0, weighted / total_gen * 1000.0, np.nan)
        # factor is in kg/kWh, *1000 → kg/MWh (matches CarbonCast scale ~150-500)
    df.insert(1, "carbon_intensity", ci)
    return df


def build_fuel_production_forecasts(fuel_hourly: pd.DataFrame,
                                    lookback_hours: int = 24,
                                    datetime_col: str = "UTC time") -> pd.DataFrame:
    """
    Day-ahead fuel production "forecast" as a persistence baseline:
    value at time t uses the actual at t - 24h. This is the same simple
    baseline CarbonCast uses for its "day-ahead" column when a native
    fuel-type forecast is unavailable. Output columns:
       avg_{fuel}_production_forecast
    """
    df = fuel_hourly.copy().sort_values(datetime_col).reset_index(drop=True)
    fuels = [c for c in CARBONCAST_FUEL_COLS if c in df.columns]
    for fuel in fuels:
        df[f"avg_{fuel}_production_forecast"] = df[fuel].shift(lookback_hours)
    keep = [datetime_col] + [f"avg_{f}_production_forecast" for f in fuels]
    return df[keep]


# ─────────────────────────────────────────────────────────────────────────────
# Output file builders (CarbonCast schema)
# ─────────────────────────────────────────────────────────────────────────────

def write_emission_csv(fuel_hourly_with_ci: pd.DataFrame, out_path: str,
                       datetime_col: str = "UTC time"):
    """
    Matches existing {REGION}_direct_emissions.csv shape:
        index, UTC time, carbon_intensity, nat_gas, nuclear, oil, hydro, wind, other, ...
    """
    df = fuel_hourly_with_ci.copy()
    # Columns: datetime, carbon_intensity, then fuels that are present+non-zero
    fuel_cols = [c for c in CARBONCAST_FUEL_COLS
                 if c in df.columns and df[c].abs().sum() > 0]
    out = df[[datetime_col, "carbon_intensity"] + fuel_cols].copy()
    out[datetime_col] = pd.to_datetime(out[datetime_col], utc=True
                                       ).dt.tz_convert(None)
    out.to_csv(out_path, index=True)
    print(f"  wrote emission_csv  rows={len(out)}  cols={list(out.columns)}")


def write_forecast_csv(fuel_fcst: pd.DataFrame, weather_df: Optional[pd.DataFrame],
                       out_path: str, datetime_col: str = "UTC time"):
    """
    Matches existing {REGION}_96hr_forecasts.csv shape:
        UTC time, forecast_avg_*_wMean..., avg_*_production_forecast...
    """
    out = fuel_fcst.copy()
    out[datetime_col] = pd.to_datetime(out[datetime_col], utc=True
                                       ).dt.tz_convert(None)
    if weather_df is not None and len(weather_df) > 0:
        wx = weather_df.copy()
        wx[datetime_col] = pd.to_datetime(wx[datetime_col], errors="coerce"
                                          ).dt.tz_localize(None, nonexistent="NaT",
                                                           ambiguous="NaT") \
                           if wx[datetime_col].dtype.name.startswith("datetime64[ns,") \
                           else pd.to_datetime(wx[datetime_col], errors="coerce")
        out = out.merge(wx, on=datetime_col, how="left")
        # Reorder: weather first, then fuel forecasts
        wx_cols = [c for c in wx.columns if c != datetime_col]
        fcst_cols = [c for c in out.columns
                     if c.startswith("avg_") and c.endswith("_production_forecast")]
        out = out[[datetime_col] + wx_cols + fcst_cols]
    out.to_csv(out_path, index=False)
    print(f"  wrote forecast_csv  rows={len(out)}  cols={list(out.columns)}")


# ─────────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ─────────────────────────────────────────────────────────────────────────────

def fetch_region(config: dict, api_key: Optional[str] = None,
                 force_refresh: bool = False,
                 project_root: Optional[str] = None) -> dict:
    """
    End-to-end: raw gridstatusio → CarbonCast CSVs on disk.

    Returns a dict of absolute output paths for downstream training or
    diagnostics.
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    region_name = config["dataset"].get("region_name", "UNKNOWN")
    ds = config.get("data_source")
    if ds is None:
        raise ValueError(f"config has no 'data_source' block; cannot auto-fetch "
                         f"for region {region_name}. Add a data_source key or "
                         f"run with --skip-fetch to use pre-existing CSVs.")

    paths = {k: os.path.join(project_root, v) if not os.path.isabs(v) else v
             for k, v in config["paths"].items()}
    for p in paths.values():
        os.makedirs(os.path.dirname(p), exist_ok=True)

    emission_csv = paths["emission_csv"]
    forecast_csv = paths["forecast_csv"]
    weather_csv  = paths.get("weather_csv")

    # Short-circuit if already cached
    if not force_refresh and os.path.exists(emission_csv) and os.path.exists(forecast_csv):
        print(f"[fetch_region:{region_name}] cache hit — reusing\n"
              f"    {emission_csv}\n    {forecast_csv}")
        return {"emission_csv": emission_csv, "forecast_csv": forecast_csv,
                "weather_csv": weather_csv, "cached": True}

    # gridstatusio client
    provider = ds.get("provider", "gridstatusio")
    if provider != "gridstatusio":
        raise NotImplementedError(f"provider={provider} not supported yet")

    from gridstatusio import GridStatusClient
    if api_key is None:
        api_key = os.environ.get("GRIDSTATUS_API_KEY")
    if not api_key:
        raise RuntimeError("gridstatusio api_key missing: pass via --api_key "
                           "or set GRIDSTATUS_API_KEY env var")
    client = GridStatusClient(api_key=api_key)

    start = ds["start_date"]
    end   = ds["end_date"]
    chunk_days = int(ds.get("chunk_days", 30))
    sleep_sec  = float(ds.get("sleep_sec", 2))

    # ── 1. Raw fuel mix ──────────────────────────────────────────────────────
    fm = ds["fuel_mix"]
    # Server-side resample: ask gridstatusio to return pre-aggregated hourly
    # data instead of native 5-min. This cuts API payload ~12x and is
    # effectively free since we downsample to hourly anyway.
    fm_extra = _resample_kwargs(fm.get("resample", "1h"))
    if fm.get("extra_kwargs"):
        fm_extra.update(fm["extra_kwargs"])
    print(f"\n[{region_name}] fetching fuel mix: {fm['dataset']}  "
          f"(resample={fm.get('resample', '1h')})")
    fuel_raw = _fetch_dataset(client, fm["dataset"], start, end,
                              chunk_days, sleep_sec,
                              extra_kwargs=fm_extra)
    raw_dir = os.path.join(project_root, "data", region_name.upper(), "_raw")
    os.makedirs(raw_dir, exist_ok=True)
    fuel_raw.to_csv(os.path.join(raw_dir, f"{region_name}_fuel_mix_raw.csv"),
                    index=False)

    fuel_hourly = raw_fuel_to_carboncast(fuel_raw, fm["fuel_col_map"])

    # ── 2. Add carbon_intensity and write emission_csv ───────────────────────
    emission_factors = config.get("emission_factors", {})
    if not emission_factors:
        print("    ! no emission_factors in config; carbon_intensity will be NaN")
    fuel_with_ci = add_carbon_intensity(fuel_hourly, emission_factors)
    write_emission_csv(fuel_with_ci, emission_csv)

    # ── 3. Build fuel production forecasts (persistence baseline for now) ────
    fuel_fcst = build_fuel_production_forecasts(fuel_hourly, lookback_hours=24)

    # ── 4. Optional load forecast fetch (not used by structured_surrogate
    #      directly, but saved for future use) ───────────────────────────────
    lf = ds.get("load_forecast")
    if lf and lf.get("dataset"):
        try:
            lf_extra = _resample_kwargs(lf.get("resample", "1h"))
            if lf.get("extra_kwargs"):
                lf_extra.update(lf["extra_kwargs"])
            print(f"\n[{region_name}] fetching load forecast: {lf['dataset']}  "
                  f"(resample={lf.get('resample', '1h')})")
            load_raw = _fetch_dataset(client, lf["dataset"], start, end,
                                      chunk_days, sleep_sec,
                                      extra_kwargs=lf_extra)
            load_raw.to_csv(os.path.join(raw_dir,
                            f"{region_name}_load_forecast_raw.csv"), index=False)
        except Exception as ex:
            print(f"    ! load forecast fetch failed, continuing: {ex}")

    # ── 5. Weather — reuse existing CSV if present; else warn ────────────────
    weather_df = None
    weather_provider = (ds.get("weather") or {}).get("provider", "existing_csv")
    if weather_provider == "none":
        print(f"    weather disabled (provider=none); forecast_csv will have "
              f"fuel forecasts only")
    elif weather_csv and os.path.exists(weather_csv):
        weather_df = pd.read_csv(weather_csv)
        # Normalize the datetime col name
        for c in weather_df.columns:
            if c.lower().strip() in ("utc time", "interval_start_utc"):
                weather_df = weather_df.rename(columns={c: "UTC time"})
                break
        print(f"    using existing weather_csv: {weather_csv}  "
              f"({len(weather_df)} rows)")
    else:
        print(f"    ! no weather_csv found at {weather_csv}; "
              f"forecast_csv will lack weather features")

    write_forecast_csv(fuel_fcst, weather_df, forecast_csv)

    return {"emission_csv": emission_csv, "forecast_csv": forecast_csv,
            "weather_csv": weather_csv, "cached": False}


# ─────────────────────────────────────────────────────────────────────────────
# Module-level test hook (no API call)
# ─────────────────────────────────────────────────────────────────────────────

def _self_test():
    """Smoke-test the transformation helpers on synthetic data."""
    print("=== data_pipeline self-test ===")
    rng = pd.date_range("2024-01-01", periods=48, freq="5min", tz="UTC")
    raw = pd.DataFrame({
        "interval_start_utc": rng,
        "dual_fuel": np.random.uniform(1000, 3000, len(rng)),
        "natural_gas": np.random.uniform(2000, 5000, len(rng)),
        "nuclear": np.random.uniform(4000, 4200, len(rng)),
        "hydro": np.random.uniform(500, 1500, len(rng)),
        "wind": np.random.uniform(0, 2000, len(rng)),
        "other_renewables": np.random.uniform(100, 500, len(rng)),
    })
    cmap = {
        "dual_fuel": "nat_gas",
        "natural_gas": "nat_gas",
        "nuclear": "nuclear",
        "hydro": "hydro",
        "wind": "wind",
        "other_renewables": "other",
    }
    hourly = raw_fuel_to_carboncast(raw, cmap)
    assert "nat_gas" in hourly.columns
    assert len(hourly) == 4, f"expected 4 hourly rows, got {len(hourly)}"

    with_ci = add_carbon_intensity(hourly, {
        "nat_gas": 0.40, "nuclear": 0.0, "hydro": 0.0,
        "wind": 0.0, "other": 0.20,
    })
    assert "carbon_intensity" in with_ci.columns
    assert with_ci["carbon_intensity"].notna().any()

    fcst = build_fuel_production_forecasts(hourly, lookback_hours=1)
    assert "avg_nat_gas_production_forecast" in fcst.columns
    print("  all checks passed")


if __name__ == "__main__":
    _self_test()
