#!/usr/bin/env python
"""Build fair baseline tables for short-horizon and day-ahead settings."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/carbon_forecast_matplotlib")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fuel_mix_trainer import (  # noqa: E402
    build_supervised_table,
    feature_columns_from_table,
    fuel_mix_to_carbon_intensity_np,
    load_config,
    split_train_val_test,
)

REGIONS = ["isone", "nyiso", "pjm", "ercot", "caiso", "miso", "ieso"]
# Day-ahead-only variants (ERCOT with wind/solar forecasts), e.g. "ercot_rnw0,ercot_rnw24".
DAYAHEAD_EXTRA = [r for r in os.environ.get("DAYAHEAD_EXTRA_REGIONS", "").split(",") if r.strip()]
LABELS = {
    "isone": "ISO-NE",
    "nyiso": "NYISO",
    "pjm": "PJM",
    "ercot": "ERCOT",
    "caiso": "CAISO",
    "miso": "MISO",
    "ieso": "IESO",
    "ercot_rnw0": "ERCOT+WS latest",
    "ercot_rnw24": "ERCOT+WS 24h",
    "sim_clean": "Sim clean",
    "sim_boundary": "Sim boundary",
}


def label_for_region(region: str) -> str:
    if region in LABELS:
        return LABELS[region]
    if region.startswith("sim_"):
        return region.replace("_", " ")
    return region
OUT_DIR = REPO_ROOT / "results" / "experiments" / "horizon_comparison"
DIRECT_SEEDS = [int(s) for s in os.environ.get("DIRECT_SEEDS", "7,42,123").split(",") if s.strip()]


class DirectCIMLP(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def true_ci(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    target_col = cfg.get("dataset", {}).get("ci_target_col")
    if target_col:
        if target_col not in df.columns:
            raise KeyError(f"ci_target_col='{target_col}' not found in supervised table")
        return df[target_col].to_numpy(dtype=float)
    fuel_cols = cfg["dataset"]["fuel_cols"]
    _, ci = fuel_mix_to_carbon_intensity_np(
        df[fuel_cols].to_numpy(dtype=float),
        cfg["emission_factors"],
        fuel_cols,
        df["system_load_forecast"].to_numpy(dtype=float),
    )
    return ci.astype(float)


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(y)) ** 2)))


def fit_direct_mlp(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_test: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_mu = x_fit.mean(axis=0)
    x_sigma = x_fit.std(axis=0)
    x_sigma = np.where(x_sigma < 1e-8, 1.0, x_sigma)
    y_mu = float(y_fit.mean())
    y_sigma = float(y_fit.std() if y_fit.std() > 1e-8 else 1.0)

    x_train = torch.tensor((x_fit - x_mu) / x_sigma, dtype=torch.float32)
    y_train = torch.tensor((y_fit - y_mu) / y_sigma, dtype=torch.float32)
    x_eval = torch.tensor((x_test - x_mu) / x_sigma, dtype=torch.float32)

    model = DirectCIMLP(x_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(300):
        opt.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(x_eval).numpy()
    return pred * y_sigma + y_mu


def fit_direct_mlp_train_val(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    seed: int = 42,
    max_epochs: int = 300,
    patience: int = 5,
    min_epochs: int = 0,
    device: str = "cpu",
) -> dict:
    """Fit Direct MLP on train only and select its epoch by validation RMSE.

    Feature and target normalization statistics are computed exclusively from
    ``x_train`` and ``y_train``. Validation observations are used only for
    early stopping, matching the information boundary used by Fuel-mix.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_mu = x_train.mean(axis=0)
    x_sigma = x_train.std(axis=0)
    x_sigma = np.where(x_sigma < 1e-8, 1.0, x_sigma)
    y_mu = float(y_train.mean())
    y_sigma = float(y_train.std() if y_train.std() > 1e-8 else 1.0)

    xtr = torch.tensor((x_train - x_mu) / x_sigma, dtype=torch.float32, device=device)
    ytr = torch.tensor((y_train - y_mu) / y_sigma, dtype=torch.float32, device=device)
    xva = torch.tensor((x_val - x_mu) / x_sigma, dtype=torch.float32, device=device)
    xte = torch.tensor((x_test - x_mu) / x_sigma, dtype=torch.float32, device=device)

    model = DirectCIMLP(xtr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val_rmse = float("inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(xtr), ytr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(xva).cpu().numpy() * y_sigma + y_mu
        val_rmse = rmse(val_pred, y_val)
        if epoch >= min_epochs and val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        elif epoch >= min_epochs:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(xte).cpu().numpy() * y_sigma + y_mu
    return {
        "test_pred": test_pred,
        "best_val_rmse": best_val_rmse,
        "best_epoch": best_epoch,
    }


def build_setting_rows(
    setting: str,
    config_dir: Path,
    ensemble_dir: Path,
    persistence_col: str,
) -> list[dict]:
    rows = []
    for region in REGIONS + (DAYAHEAD_EXTRA if setting == "day_ahead" else []):
        if setting == "short_horizon":
            cfg_path = config_dir / f"{region}_fuelmix.json"
        else:
            cfg_path = config_dir / f"{region}_fuelmix_dayahead.json"
        cfg = load_config(str(cfg_path))
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
        direct_rmses = [rmse(pred, y_test) for pred in direct_preds]
        direct_pred = np.mean(direct_preds, axis=0)
        ensemble = pd.read_csv(
            ensemble_dir / f"{region}_ensemble_predictions.csv",
            parse_dates=["timestamp"],
        )
        if len(ensemble) != len(y_test):
            raise ValueError(
                f"{setting}/{region}: ensemble length {len(ensemble)} "
                f"does not match test length {len(y_test)}"
            )
        persist = test_df[persistence_col].to_numpy(dtype=float)
        fuelmix_pred = ensemble["ci_pred"].to_numpy(dtype=float)

        rows.append({
            "setting": setting,
            "region": label_for_region(region),
            "persistence_baseline": persistence_col.replace("historical_ci_", "").replace("_lag", " persistence"),
            "persistence_rmse": rmse(persist, y_test),
            "direct_mlp_rmse": rmse(direct_pred, y_test),
            "direct_mlp_rmse_mean": float(np.mean(direct_rmses)),
            "direct_mlp_rmse_std": float(np.std(direct_rmses, ddof=1)) if len(direct_rmses) > 1 else 0.0,
            "fuelmix_ensemble_rmse": rmse(fuelmix_pred, y_test),
            "fuelmix_minus_persistence": rmse(fuelmix_pred, y_test) - rmse(persist, y_test),
            "fuelmix_minus_direct_mlp": rmse(fuelmix_pred, y_test) - rmse(direct_pred, y_test),
            "direct_mlp_seeds": ",".join(str(s) for s in DIRECT_SEEDS),
            "direct_best_epochs": ",".join(str(run["best_epoch"]) for run in direct_runs),
            "n_test": len(y_test),
            "feature_count": len(feature_cols),
        })
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(build_setting_rows(
        "short_horizon",
        REPO_ROOT / "configs",
        REPO_ROOT / "results" / "fuelmix" / "multiseed",
        "historical_ci_1h_lag",
    ))
    rows.extend(build_setting_rows(
        "day_ahead",
        REPO_ROOT / "configs" / "dayahead",
        REPO_ROOT / "results" / "fuelmix_dayahead" / "multiseed",
        "historical_ci_24h_lag",
    ))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "horizon_baseline_comparison.csv", index=False)

    short = out[out["setting"] == "short_horizon"]
    day = out[out["setting"] == "day_ahead"]
    short.to_csv(OUT_DIR / "short_horizon_baselines.csv", index=False)
    day.to_csv(OUT_DIR / "dayahead_baselines.csv", index=False)

    print("Short-horizon fair comparison:")
    print(short[["region", "persistence_rmse", "direct_mlp_rmse", "fuelmix_ensemble_rmse"]].round(4).to_string(index=False))
    print("\nDay-ahead fair comparison:")
    print(day[["region", "persistence_rmse", "direct_mlp_rmse", "fuelmix_ensemble_rmse"]].round(4).to_string(index=False))
    print(f"\nwrote {OUT_DIR / 'horizon_baseline_comparison.csv'}")


if __name__ == "__main__":
    main()
