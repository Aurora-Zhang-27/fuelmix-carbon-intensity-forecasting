"""
run_fuelmix.py — unified fetch→train CLI for the FuelMixForecaster method
==========================================================================

This is the runner for the paper's actual method: load forecast -> fuel
shares + total -> fuel mix -> carbon intensity.

Usage:

    # Train on already-downloaded data (no API call)
    python src/run_fuelmix.py --config configs/nyiso_fuelmix.json --stage train

    # Fetch raw fuel_mix + load_forecast from gridstatusio, then train
    python src/run_fuelmix.py --config configs/pjm_fuelmix.json --stage all

    # Just fetch, don't train
    python src/run_fuelmix.py --config configs/ercot_fuelmix.json --stage fetch

    # Force a fresh API pull
    python src/run_fuelmix.py --config configs/pjm_fuelmix.json --stage fetch \
        --force_refresh

    # App-like run: choose dates from the command line
    python src/run_fuelmix.py --config configs/ercot_fuelmix.json --stage all \
        --start_date 2024-01-01 --end_date 2024-12-31 --api_key YOUR_KEY
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    raise SystemExit(
        "Missing dependency: torch. Install the project dependencies with:\n"
        "    python -m pip install -r requirements.txt\n"
        "Make sure this is the same python used to run src/run_fuelmix.py."
    ) from exc

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(region: str) -> Path:
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"fuelmix_{region.lower()}_{stamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return log_path


# ─────────────────────────────────────────────────────────────────────────────
# Stage: fetch  (raw fuel_mix + load_forecast via gridstatusio)
# ─────────────────────────────────────────────────────────────────────────────

def stage_fetch(cfg: dict, api_key: str | None, force_refresh: bool) -> dict:
    """
    Pulls raw fuel_mix and load_forecast from gridstatusio and writes them
    to the paths specified in cfg["paths"]. No CarbonCast-style transform —
    fuel_mix_trainer.py applies fuel_col_map at load time, so we just cache
    the raw API responses.
    """
    from data_pipeline import (
        _fetch_dataset,
        _resample_kwargs,
    )

    region = cfg["dataset"].get("region_name", "UNKNOWN")
    logging.info("=" * 70)
    logging.info(f"STAGE: fetch   region={region}   force_refresh={force_refresh}")
    logging.info("=" * 70)

    ds = cfg.get("data_source")
    if ds is None:
        raise ValueError(
            f"config has no 'data_source' block; cannot auto-fetch. "
            f"If you already have fuel_mix_csv and load_forecast_csv on disk, "
            f"run --stage train directly.")
    if ds.get("provider", "gridstatusio") != "gridstatusio":
        raise NotImplementedError(
            f"provider={ds.get('provider')} not supported yet")

    # Resolve output paths
    fuel_mix_csv = cfg["paths"]["fuel_mix_csv"]
    load_csv = cfg["paths"]["load_forecast_csv"]
    renewable_csv = cfg["paths"].get("renewable_forecast_csv")
    if not os.path.isabs(fuel_mix_csv):
        fuel_mix_csv = str(REPO_ROOT / fuel_mix_csv)
    if not os.path.isabs(load_csv):
        load_csv = str(REPO_ROOT / load_csv)
    if renewable_csv and not os.path.isabs(renewable_csv):
        renewable_csv = str(REPO_ROOT / renewable_csv)
    os.makedirs(os.path.dirname(fuel_mix_csv), exist_ok=True)
    os.makedirs(os.path.dirname(load_csv), exist_ok=True)
    if renewable_csv:
        os.makedirs(os.path.dirname(renewable_csv), exist_ok=True)

    # ── Per-file cache check ─────────────────────────────────────────────
    # All-or-nothing was wasting API quota: if renewable was missing but
    # fuel_mix + load already cached, we'd refetch the latter two too.
    # Now each piece is fetched independently iff its target file is absent
    # (or force_refresh is set).
    rf = ds.get("renewable_forecast")
    need_fuel = force_refresh or not os.path.exists(fuel_mix_csv)
    need_load = force_refresh or not os.path.exists(load_csv)
    need_renewable = bool(rf and renewable_csv) and (
        force_refresh or not os.path.exists(renewable_csv)
    )

    if not (need_fuel or need_load or need_renewable):
        logging.info(f"[{region}] cache hit — reusing all raw CSVs:")
        logging.info(f"    {fuel_mix_csv}")
        logging.info(f"    {load_csv}")
        if renewable_csv:
            logging.info(f"    {renewable_csv}")
        return {"fuel_mix_csv": fuel_mix_csv, "load_forecast_csv": load_csv,
                "renewable_forecast_csv": renewable_csv,
                "cached": True}

    logging.info(f"[{region}] fetch plan: fuel_mix={need_fuel}  "
                 f"load={need_load}  renewable={need_renewable}")
    for label, path, will_fetch in [
        ("fuel_mix", fuel_mix_csv, need_fuel),
        ("load",     load_csv,     need_load),
        ("renewable",renewable_csv, need_renewable),
    ]:
        if path and not will_fetch and os.path.exists(path):
            logging.info(f"    [skip] {label}: cached at {path}")

    # gridstatusio client (only instantiate if we'll actually call the API)
    client = None
    if need_fuel or need_load or need_renewable:
        from gridstatusio import GridStatusClient
        if api_key is None:
            api_key = os.environ.get("GRIDSTATUS_API_KEY")
        if not api_key:
            raise RuntimeError("gridstatusio api_key missing: pass via --api_key "
                               "or set GRIDSTATUS_API_KEY env var")
        client = GridStatusClient(api_key=api_key)

    start = ds["start_date"]
    end = ds["end_date"]
    chunk_days = int(ds.get("chunk_days", 30))
    sleep_sec = float(ds.get("sleep_sec", 2))

    # ── Fuel mix ──
    if need_fuel:
        fm = ds["fuel_mix"]
        fm_extra = _resample_kwargs(fm.get("resample", "1h"))
        if fm.get("extra_kwargs"):
            fm_extra.update(fm["extra_kwargs"])
        logging.info(f"[{region}] fetching fuel mix: {fm['dataset']}  "
                     f"(resample={fm.get('resample', '1h')})")
        fuel_raw = _fetch_dataset(client, fm["dataset"], start, end,
                                  chunk_days, sleep_sec, extra_kwargs=fm_extra)
        fuel_raw.to_csv(fuel_mix_csv, index=False)
        logging.info(f"    wrote {fuel_mix_csv}  rows={len(fuel_raw)}")

    # ── Load forecast ──
    if need_load:
        lf = ds["load_forecast"]
        logging.info(f"[{region}] fetching load forecast: {lf['dataset']}  "
                     f"(resample={lf.get('resample', '1h')})")
        lf_extra = _resample_kwargs(lf.get("resample", "1h"))
        if lf.get("extra_kwargs"):
            lf_extra.update(lf["extra_kwargs"])
        load_raw = _fetch_dataset(client, lf["dataset"], start, end,
                                  chunk_days, sleep_sec,
                                  extra_kwargs=lf_extra)
        load_raw.to_csv(load_csv, index=False)
        logging.info(f"    wrote {load_csv}  rows={len(load_raw)}")

    # ── Optional renewable forecast features ──
    if need_renewable:
        from data_pipeline import build_ercot_renewable_forecast

        wind = rf["wind"]
        solar = rf["solar"]
        wind_extra = _resample_kwargs(wind.get("resample", "1h"))
        solar_extra = _resample_kwargs(solar.get("resample", "1h"))

        logging.info(f"[{region}] fetching wind forecast: {wind['dataset']}  "
                     f"(resample={wind.get('resample', '1h')})")
        wind_raw = _fetch_dataset(client, wind["dataset"], start, end,
                                  chunk_days, sleep_sec,
                                  extra_kwargs=wind_extra)

        logging.info(f"[{region}] fetching solar forecast: {solar['dataset']}  "
                     f"(resample={solar.get('resample', '1h')})")
        solar_raw = _fetch_dataset(client, solar["dataset"], start, end,
                                   chunk_days, sleep_sec,
                                   extra_kwargs=solar_extra)

        renewable = build_ercot_renewable_forecast(wind_raw, solar_raw)
        renewable.to_csv(renewable_csv, index=False)
        logging.info(f"    wrote {renewable_csv}  rows={len(renewable)}")

    return {"fuel_mix_csv": fuel_mix_csv, "load_forecast_csv": load_csv,
            "renewable_forecast_csv": renewable_csv, "cached": False}


# ─────────────────────────────────────────────────────────────────────────────
# Stage: train
# ─────────────────────────────────────────────────────────────────────────────

def stage_train(cfg: dict, device: str, seed: int | None = None) -> dict:
    from fuel_mix_trainer import run_experiment
    region = cfg["dataset"].get("region_name", "UNKNOWN")
    logging.info("=" * 70)
    logging.info(f"STAGE: train   region={region}   device={device}   seed={seed}")
    logging.info("=" * 70)
    result = run_experiment(cfg, device=device, make_plots=True, seed=seed)
    logging.info(f"[{region}] test metrics:")
    for k, v in result["test_metrics"].items():
        logging.info(f"    {k}: {v:.6f}")
    logging.info(f"[{region}] outputs saved under {result['save_dir']}")
    return result


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> bool:
    """Apply command-line overrides to the loaded config."""
    changed_fetch_window = False
    ds = cfg.get("data_source")
    if ds is not None:
        if args.start_date:
            ds["start_date"] = args.start_date
            changed_fetch_window = True
        if args.end_date:
            ds["end_date"] = args.end_date
            changed_fetch_window = True
        if args.chunk_days is not None:
            ds["chunk_days"] = args.chunk_days
        if args.sleep_sec is not None:
            ds["sleep_sec"] = args.sleep_sec
    return changed_fetch_window


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="Path to region fuelmix config JSON")
    ap.add_argument("--stage", default="train",
                    choices=["fetch", "train", "all"],
                    help="Which stage(s) to run (default: train)")
    ap.add_argument("--api_key", default=None,
                    help="gridstatusio API key (falls back to $GRIDSTATUS_API_KEY)")
    ap.add_argument("--force_refresh", action="store_true",
                    help="Force re-fetch from API even if cached raw CSVs exist")
    ap.add_argument("--start_date", default=None,
                    help="Override config data_source.start_date, YYYY-MM-DD")
    ap.add_argument("--end_date", default=None,
                    help="Override config data_source.end_date, YYYY-MM-DD")
    ap.add_argument("--chunk_days", type=int, default=None,
                    help="Override config data_source.chunk_days")
    ap.add_argument("--sleep_sec", type=float, default=None,
                    help="Override config data_source.sleep_sec")
    ap.add_argument("--device", default=None,
                    help="'cpu' or 'cuda'; default auto-detects")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed for reproducible paper runs")
    args = ap.parse_args()

    from fuel_mix_trainer import load_config
    cfg = load_config(args.config)
    changed_fetch_window = apply_cli_overrides(cfg, args)
    if changed_fetch_window and args.stage in ("fetch", "all"):
        args.force_refresh = True

    region = cfg["dataset"].get("region_name", "UNKNOWN")
    log_path = setup_logging(region)
    logging.info(f"config:   {args.config}")
    logging.info(f"log file: {log_path}")
    if cfg.get("data_source"):
        logging.info(f"date window: {cfg['data_source']['start_date']} -> "
                     f"{cfg['data_source']['end_date']}")
        if changed_fetch_window:
            logging.info("date override detected; force_refresh enabled")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        if args.stage in ("fetch", "all"):
            stage_fetch(cfg, api_key=args.api_key,
                        force_refresh=args.force_refresh)
        if args.stage in ("train", "all"):
            stage_train(cfg, device=device, seed=args.seed)
    except Exception as ex:
        logging.exception(f"fuelmix pipeline failed at stage={args.stage}: {ex}")
        sys.exit(1)

    logging.info("pipeline complete")


if __name__ == "__main__":
    main()
