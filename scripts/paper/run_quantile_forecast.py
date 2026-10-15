#!/usr/bin/env python
"""Quantile carbon-intensity forecasting diagnostics.

This exploratory script tests whether calibrated interval forecasts add value
beyond point CI RMSE. It compares:

1. Persistence interval: 1h persistence plus empirical train residual quantiles.
2. Direct quantile MLP: selected features -> CI q10/q50/q90.
3. Fuelmix-informed quantile MLP: selected features plus a trained fuel-mix
   surrogate's point CI and total-generation outputs -> CI q10/q50/q90.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/carbon_forecast_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fuel_mix_trainer import (  # noqa: E402
    FuelMixTrainer,
    build_supervised_table,
    feature_columns_from_table,
    fuel_mix_to_carbon_intensity_np,
    iterate_minibatches,
    load_config,
    split_train_val_test,
    standardize_from_train,
    standardize_targets_from_train,
    to_torch,
)


REGIONS = ["isone", "nyiso", "pjm", "ercot", "caiso", "miso", "ieso"]
LABELS = {
    "isone": "ISO-NE",
    "nyiso": "NYISO",
    "pjm": "PJM",
    "ercot": "ERCOT",
    "caiso": "CAISO",
    "miso": "MISO",
    "ieso": "IESO",
}
QUANTILES = np.array([0.10, 0.50, 0.90], dtype=np.float32)
CONFORMAL_ALPHA = 0.18
OUT_DIR = REPO_ROOT / "results" / "experiments" / "quantile_forecast"
PAPER_FIG_DIR = REPO_ROOT / "paper_figures"


class QuantileMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = 96, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, len(QUANTILES)),
        )

    def forward(self, x):
        raw = self.net(x)
        # Enforce non-crossing quantiles by making q50/q90 positive increments
        # above q10. This keeps intervals well formed during training.
        q10 = raw[:, 0]
        q50 = q10 + torch.nn.functional.softplus(raw[:, 1])
        q90 = q50 + torch.nn.functional.softplus(raw[:, 2])
        return torch.stack([q10, q50, q90], dim=1)


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })


def true_ci(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    fuel_cols = cfg["dataset"]["fuel_cols"]
    _, ci = fuel_mix_to_carbon_intensity_np(
        df[fuel_cols].to_numpy(dtype=float),
        cfg["emission_factors"],
        fuel_cols,
        df["system_load_forecast"].to_numpy(dtype=float),
    )
    return ci.astype(np.float32)


def pinball_np(y: np.ndarray, q_pred: np.ndarray) -> float:
    y = y.reshape(-1, 1)
    qs = QUANTILES.reshape(1, -1)
    err = y - q_pred
    return float(np.mean(np.maximum(qs * err, (qs - 1.0) * err)))


def pinball_loss_torch(q_pred, y):
    qs = torch.tensor(QUANTILES, dtype=torch.float32, device=q_pred.device)[None, :]
    err = y[:, None] - q_pred
    return torch.mean(torch.maximum(qs * err, (qs - 1.0) * err))


def interval_metrics(region: str, method: str, y_true: np.ndarray,
                     q_pred: np.ndarray) -> dict:
    q10, q50, q90 = q_pred[:, 0], q_pred[:, 1], q_pred[:, 2]
    coverage = np.mean((y_true >= q10) & (y_true <= q90))
    width = np.mean(q90 - q10)
    rmse = np.sqrt(np.mean((q50 - y_true) ** 2))
    mae = np.mean(np.abs(q50 - y_true))
    return {
        "region": LABELS[region],
        "method": method,
        "coverage_80": float(coverage),
        "mean_width": float(width),
        "q50_mae": float(mae),
        "q50_rmse": float(rmse),
        "pinball_loss": pinball_np(y_true, q_pred),
    }


def conformal_calibrate(
    val_q: np.ndarray,
    y_val: np.ndarray,
    test_q: np.ndarray,
    alpha: float = 0.20,
) -> tuple[np.ndarray, float]:
    """Split-conformal calibration for a central quantile interval."""
    scores = np.maximum(val_q[:, 0] - y_val, y_val - val_q[:, 2])
    scores = np.maximum(scores, 0.0)
    n = len(scores)
    if n == 0:
        raise ValueError("Calibration set must be nonempty")
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    adjustment = float(np.partition(scores, rank - 1)[rank - 1]) if rank <= n else float("inf")
    out = test_q.copy()
    out[:, 0] -= adjustment
    out[:, 2] += adjustment
    return out, adjustment


def conformal_curve(
    val_q: np.ndarray,
    y_val: np.ndarray,
    test_q: np.ndarray,
    y_test: np.ndarray,
    method: str,
    region: str,
    alphas: np.ndarray | None = None,
) -> list[dict]:
    """Coverage-width points from split-conformal interval expansion."""
    if alphas is None:
        alphas = np.linspace(0.05, 0.50, 19)
    rows = []
    for alpha in alphas:
        q_cal, adj = conformal_calibrate(val_q, y_val, test_q, alpha=float(alpha))
        q10, q90 = q_cal[:, 0], q_cal[:, 2]
        rows.append({
            "region": LABELS[region],
            "method": method,
            "alpha": float(alpha),
            "conformal_adjustment": adj,
            "coverage": float(np.mean((y_test >= q10) & (y_test <= q90))),
            "mean_width": float(np.mean(q90 - q10)),
        })
    return rows


def persistence_curve(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    region: str,
    alphas: np.ndarray | None = None,
) -> list[dict]:
    """Coverage-width points for empirical residual persistence intervals."""
    if alphas is None:
        alphas = np.linspace(0.05, 0.50, 19)
    train_lag = train_df["historical_ci_1h_lag"].to_numpy(dtype=np.float32)
    test_lag = test_df["historical_ci_1h_lag"].to_numpy(dtype=np.float32)
    residual = y_train - train_lag
    rows = []
    for alpha in alphas:
        lo = alpha / 2.0
        hi = 1.0 - alpha / 2.0
        r_lo, r_hi = np.quantile(residual, [lo, hi])
        q10 = test_lag + r_lo
        q90 = test_lag + r_hi
        rows.append({
            "region": LABELS[region],
            "method": "persistence_empirical_interval",
            "alpha": float(alpha),
            "conformal_adjustment": np.nan,
            "coverage": float(np.mean((y_test >= q10) & (y_test <= q90))),
            "mean_width": float(np.mean(q90 - q10)),
        })
    return rows


def standardize_arrays(x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray):
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return (x_train - mu) / sigma, (x_val - mu) / sigma, (x_test - mu) / sigma


def train_quantile_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    seed: int = 42,
    epochs: int = 400,
    dropout: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train, x_val, x_test = standardize_arrays(x_train, x_val, x_test)
    y_mu = float(y_train.mean())
    y_sigma = float(y_train.std() if y_train.std() > 1e-8 else 1.0)
    y_train_std = (y_train - y_mu) / y_sigma
    y_val_std = (y_val - y_mu) / y_sigma

    Xtr = torch.tensor(x_train, dtype=torch.float32)
    ytr = torch.tensor(y_train_std, dtype=torch.float32)
    Xva = torch.tensor(x_val, dtype=torch.float32)
    yva = torch.tensor(y_val_std, dtype=torch.float32)
    Xte = torch.tensor(x_test, dtype=torch.float32)

    model = QuantileMLP(x_train.shape[1], dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_val = float("inf")
    wait = 0
    idx = np.arange(len(x_train))

    for _ in range(epochs):
        model.train()
        np.random.shuffle(idx)
        for start in range(0, len(idx), 128):
            j = idx[start:start + 128]
            pred = model(Xtr[j])
            loss = pinball_loss_torch(pred, ytr[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(pinball_loss_torch(model(Xva), yva).item())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 25:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_q = model(Xva).numpy() * y_sigma + y_mu
        test_q = model(Xte).numpy() * y_sigma + y_mu
    return val_q, test_q


def train_fuelmix_point_outputs(
    cfg: dict,
    feature_cols: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = 42,
) -> tuple[dict, dict, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    fuel_cols = cfg["dataset"]["fuel_cols"]
    emission_factors = cfg["emission_factors"]
    train_x_np, val_x_np, test_x_np, *_ = standardize_from_train(
        train_df, val_df, test_df, feature_cols
    )

    def arrays(df):
        y_fuel = df[fuel_cols].to_numpy(dtype=np.float32)
        demand = df["system_load_forecast"].to_numpy(dtype=np.float32)
        return y_fuel, demand

    yfuel_tr, dtr = arrays(train_df)
    yfuel_va, dva = arrays(val_df)
    yfuel_te, dte = arrays(test_df)
    (yf_tr_std, yf_va_std, yf_te_std, fuel_mu, fuel_sigma) = (
        standardize_targets_from_train(yfuel_tr, yfuel_va, yfuel_te)
    )
    (dtr_std, dva_std, dte_std, demand_mu_arr, demand_sigma_arr) = (
        standardize_targets_from_train(dtr[:, None], dva[:, None], dte[:, None])
    )
    dtr_std = dtr_std.reshape(-1)
    dva_std = dva_std.reshape(-1)
    dte_std = dte_std.reshape(-1)
    demand_mu = float(demand_mu_arr.reshape(-1)[0])
    demand_sigma = float(demand_sigma_arr.reshape(-1)[0])

    device = "cpu"
    trainer = FuelMixTrainer(
        feature_cols=feature_cols,
        fuel_cols=fuel_cols,
        emission_factors=emission_factors,
        fossil_fuels=cfg.get("fossil_fuels"),
        loss_weights=cfg.get("loss_weights"),
        hidden=cfg.get("model", {}).get("hidden_size", 128),
        lr=cfg.get("training", {}).get("learning_rate", 1e-3),
        device=device,
        fuel_mu=fuel_mu,
        fuel_sigma=fuel_sigma,
        demand_mu=demand_mu,
        demand_sigma=demand_sigma,
    )

    Xtr = to_torch(train_x_np, device)
    Xva = to_torch(val_x_np, device)
    Xte = to_torch(test_x_np, device)
    YF_tr = to_torch(yf_tr_std, device)
    YF_va = to_torch(yf_va_std, device)
    YF_te = to_torch(yf_te_std, device)
    D_tr = to_torch(dtr_std, device)
    D_va = to_torch(dva_std, device)
    D_te = to_torch(dte_std, device)

    best_state = None
    best_val = float("inf")
    wait = 0
    for _ in range(60):
        for xb, yb_fuel, yb_dem in iterate_minibatches(
            Xtr, YF_tr, D_tr, batch_size=cfg.get("training", {}).get("batch_size", 64)
        ):
            trainer.step(xb, yb_fuel, yb_dem)
        val_metrics, _ = trainer.evaluate(Xva, YF_va, D_va)
        val_rmse = val_metrics["ci_rmse_t_per_mwh"]
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.clone() for k, v in trainer.model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 8:
                break
    if best_state is not None:
        trainer.model.load_state_dict(best_state)

    _, out_tr = trainer.evaluate(Xtr, YF_tr, D_tr)
    _, out_va = trainer.evaluate(Xva, YF_va, D_va)
    _, out_te = trainer.evaluate(Xte, YF_te, D_te)
    return out_tr, out_va, out_te


def persistence_interval(train_df: pd.DataFrame, test_df: pd.DataFrame,
                         y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    lag_col = "historical_ci_1h_lag"
    train_lag = train_df[lag_col].to_numpy(dtype=np.float32)
    test_lag = test_df[lag_col].to_numpy(dtype=np.float32)
    residual = y_train - train_lag
    r10, r50, r90 = np.quantile(residual, QUANTILES)
    return np.column_stack([test_lag + r10, test_lag + r50, test_lag + r90])


def multiseed_intervals(region: str, y_test: np.ndarray) -> dict[str, np.ndarray]:
    base = REPO_ROOT / "results" / "fuelmix" / "multiseed" / "predictions"
    preds = []
    for seed_dir in sorted(base.glob("seed_*")):
        path = seed_dir / f"{region}_test_predictions.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if len(df) == len(y_test):
            preds.append(df["ci_pred"].to_numpy(dtype=np.float32))
    if not preds:
        return {}
    arr = np.column_stack(preds)
    return {
        "multiseed_minmax_interval": np.column_stack([
            arr.min(axis=1),
            np.median(arr, axis=1),
            arr.max(axis=1),
        ]),
        "multiseed_p10_p90_interval": np.column_stack([
            np.quantile(arr, 0.10, axis=1),
            np.quantile(arr, 0.50, axis=1),
            np.quantile(arr, 0.90, axis=1),
        ]),
    }


def plot_region_intervals(region: str, y_true: np.ndarray,
                          direct_q: np.ndarray, informed_q: np.ndarray,
                          persist_q: np.ndarray, window: int = 96) -> None:
    start = 0
    if len(y_true) > window:
        err = np.abs(informed_q[:, 1] - y_true)
        rolling = np.convolve(err, np.ones(window) / window, mode="valid")
        start = int(np.argmin(np.abs(rolling - np.mean(err))))
    sl = slice(start, start + min(window, len(y_true)))
    x = np.arange(len(y_true[sl]))

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(x, y_true[sl], color="#111827", label="Observed")
    for q, label, color, alpha in [
        (persist_q, "Persistence interval", "#10B981", 0.18),
        (direct_q, "Direct quantile MLP", "#F59E0B", 0.18),
        (informed_q, "Fuelmix-informed quantile MLP", "#2563EB", 0.22),
    ]:
        ax.fill_between(x, q[sl, 0], q[sl, 2], color=color, alpha=alpha)
        ax.plot(x, q[sl, 1], color=color, linestyle="--", label=label)
    ax.set_title(f"{LABELS[region]} 80% CI intervals")
    ax.set_xlabel("Hour in selected test window")
    ax.set_ylabel("CI (tCO2/MWh)")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"{region}_quantile_intervals_96h.{ext}",
                    bbox_inches="tight")
    plt.close(fig)


def run_region(region: str) -> tuple[list[dict], dict[str, np.ndarray], list[dict]]:
    cfg = load_config(f"configs/{region}_fuelmix.json")
    data = build_supervised_table(cfg)
    split_cfg = cfg.get("split", {})
    train_df, val_df, test_df = split_train_val_test(
        data,
        train_frac=split_cfg.get("train_frac", 0.7),
        val_frac=split_cfg.get("val_frac", 0.15),
    )
    feature_cols = feature_columns_from_table(data, cfg)
    y_train = true_ci(train_df, cfg)
    y_val = true_ci(val_df, cfg)
    y_test = true_ci(test_df, cfg)

    x_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    x_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    x_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    direct_val_q, direct_q = train_quantile_model(
        x_train, y_train, x_val, y_val, x_test
    )

    point_tr, point_va, point_te = train_fuelmix_point_outputs(
        cfg, feature_cols, train_df, val_df, test_df
    )
    def augment(x, point):
        add = np.column_stack([
            point["ci_pred"],
        ]).astype(np.float32)
        return np.column_stack([x, add])

    informed_val_q, informed_q = train_quantile_model(
        augment(x_train, point_tr), y_train,
        augment(x_val, point_va), y_val,
        augment(x_test, point_te),
    )
    direct_conf_q, direct_adj = conformal_calibrate(
        direct_val_q, y_val, direct_q, alpha=CONFORMAL_ALPHA
    )
    informed_conf_q, informed_adj = conformal_calibrate(
        informed_val_q, y_val, informed_q, alpha=CONFORMAL_ALPHA
    )
    persist_q = persistence_interval(train_df, test_df, y_train, y_test)

    rows = [
        interval_metrics(region, "persistence_empirical_interval", y_test, persist_q),
        interval_metrics(region, "direct_quantile_mlp", y_test, direct_q),
        {
            **interval_metrics(
                region, "direct_quantile_mlp_conformal", y_test, direct_conf_q
            ),
            "conformal_adjustment": direct_adj,
        },
        interval_metrics(region, "fuelmix_informed_quantile_mlp", y_test, informed_q),
        {
            **interval_metrics(
                region,
                "fuelmix_informed_quantile_mlp_conformal",
                y_test,
                informed_conf_q,
            ),
            "conformal_adjustment": informed_adj,
        },
    ]
    for method, q in multiseed_intervals(region, y_test).items():
        rows.append(interval_metrics(region, method, y_test, q))
    curve_rows = []
    curve_rows.extend(persistence_curve(train_df, test_df, y_train, y_test, region))
    curve_rows.extend(
        conformal_curve(
            direct_val_q, y_val, direct_q, y_test,
            "direct_quantile_mlp_conformal", region,
        )
    )
    curve_rows.extend(
        conformal_curve(
            informed_val_q, y_val, informed_q, y_test,
            "fuelmix_informed_quantile_mlp_conformal", region,
        )
    )
    plot_region_intervals(region, y_test, direct_conf_q, informed_conf_q, persist_q)
    return rows, {
        "y_test": y_test,
        "direct_q": direct_q,
        "informed_q": informed_q,
        "persist_q": persist_q,
    }, curve_rows


def plot_summary(metrics: pd.DataFrame) -> None:
    methods = [
        "persistence_empirical_interval",
        "direct_quantile_mlp",
        "direct_quantile_mlp_conformal",
        "fuelmix_informed_quantile_mlp",
        "fuelmix_informed_quantile_mlp_conformal",
    ]
    labels = [
        "Persistence",
        "Direct",
        "Direct+conformal",
        "Fuelmix-informed",
        "Fuelmix+conformal",
    ]
    colors = ["#10B981", "#F59E0B", "#FDBA74", "#2563EB", "#93C5FD"]

    for col, ylabel, name in [
        ("coverage_80", "Empirical coverage of 80% interval", "quantile_coverage"),
        ("mean_width", "Mean interval width (tCO2/MWh)", "quantile_width"),
        ("pinball_loss", "Mean pinball loss", "quantile_pinball_loss"),
        ("q50_rmse", "q50 RMSE (tCO2/MWh)", "quantile_q50_rmse"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        x = np.arange(len(REGIONS))
        width = 0.16
        for i, method in enumerate(methods):
            vals = []
            for region in REGIONS:
                row = metrics[
                    (metrics["region"] == LABELS[region]) & (metrics["method"] == method)
                ]
                vals.append(float(row[col].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + (i - 2) * width, vals, width=width, label=labels[i],
                   color=colors[i])
        if col == "coverage_80":
            ax.axhline(0.80, color="#111827", linestyle="--", linewidth=1.0,
                       label="Target 80%")
        ax.set_xticks(x, [LABELS[r] for r in REGIONS])
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight")
        plt.close(fig)


def plot_tradeoff_curves(curves: pd.DataFrame) -> None:
    labels = {
        "persistence_empirical_interval": "Persistence",
        "direct_quantile_mlp_conformal": "Direct+conformal",
        "fuelmix_informed_quantile_mlp_conformal": "Fuelmix+conformal",
    }
    colors = {
        "persistence_empirical_interval": "#10B981",
        "direct_quantile_mlp_conformal": "#F59E0B",
        "fuelmix_informed_quantile_mlp_conformal": "#2563EB",
    }
    ncols = 3
    nrows = int(np.ceil(len(REGIONS) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(11.4, 2.9 * nrows), sharex=False, sharey=False
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, region in zip(axes, [LABELS[r] for r in REGIONS]):
        sub = curves[curves["region"] == region]
        for method, label in labels.items():
            ss = sub[sub["method"] == method].sort_values("coverage")
            ax.plot(
                ss["coverage"],
                ss["mean_width"],
                marker="o",
                markersize=3,
                label=label,
                color=colors[method],
            )
        ax.axvline(0.80, color="#111827", linestyle="--", linewidth=1.0)
        ax.set_title(region)
        ax.set_xlabel("Empirical coverage")
        ax.set_ylabel("Mean interval width")
    for ax in axes[len(REGIONS):]:
        ax.axis("off")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"coverage_width_tradeoff.{ext}", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=False)
    for ax, region in zip(axes, ["ISO-NE", "ERCOT"]):
        sub = curves[curves["region"] == region]
        for method, label in labels.items():
            ss = sub[sub["method"] == method].sort_values("coverage")
            ax.plot(
                ss["coverage"],
                ss["mean_width"],
                marker="o",
                markersize=3,
                label=label,
                color=colors[method],
            )
        ax.axvline(0.80, color="#111827", linestyle="--", linewidth=1.0)
        ax.set_title(region)
        ax.set_xlabel("Empirical coverage")
        ax.set_ylabel("Mean interval width")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(
            PAPER_FIG_DIR / f"interval_tradeoff_isone_ercot.{ext}",
            bbox_inches="tight",
        )
    plt.close(fig)

    ercot = curves[curves["region"] == "ERCOT"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for method, label in labels.items():
        ss = ercot[ercot["method"] == method].sort_values("coverage")
        ax.plot(
            ss["coverage"],
            ss["mean_width"],
            marker="o",
            markersize=3,
            label=label,
            color=colors[method],
        )
    ax.axvline(0.80, color="#111827", linestyle="--", linewidth=1.0)
    ax.set_title("ERCOT coverage-width tradeoff")
    ax.set_xlabel("Empirical coverage")
    ax.set_ylabel("Mean interval width")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"ercot_coverage_width_tradeoff.{ext}", bbox_inches="tight")
        fig.savefig(PAPER_FIG_DIR / f"ercot_coverage_width_tradeoff.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_near_target_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare sharpness among methods close to or above target coverage."""
    rows = []
    for region in [LABELS[r] for r in REGIONS]:
        sub = metrics[metrics["region"] == region].copy()
        eligible = sub[sub["coverage_80"] >= 0.78].copy()
        if eligible.empty:
            eligible = sub.iloc[[int((sub["coverage_80"] - 0.80).abs().argmin())]].copy()
        eligible = eligible.sort_values(["mean_width", "pinball_loss"]).reset_index(drop=True)
        best = eligible.iloc[0]
        rows.append({
            "region": region,
            "sharpest_near_80_method": best["method"],
            "coverage_80": best["coverage_80"],
            "mean_width": best["mean_width"],
            "q50_rmse": best["q50_rmse"],
            "pinball_loss": best["pinball_loss"],
            "eligible_methods": "; ".join(
                f"{r.method} (cov={r.coverage_80:.3f}, width={r.mean_width:.4f})"
                for r in eligible.itertuples(index=False)
            ),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "near_80_width_summary.csv", index=False)
    return out


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    curve_rows = []
    for region in REGIONS:
        print(f"[{LABELS[region]}] training quantile diagnostics")
        region_rows, _, region_curve_rows = run_region(region)
        rows.extend(region_rows)
        curve_rows.extend(region_curve_rows)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_DIR / "quantile_metrics.csv", index=False)
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(OUT_DIR / "coverage_width_tradeoff.csv", index=False)
    plot_summary(metrics)
    plot_tradeoff_curves(curves)
    near_target = write_near_target_summary(metrics)
    print(metrics.round(4).to_string(index=False))
    print("\nSharpest intervals among methods with coverage >= 78%:")
    print(near_target.round(4).to_string(index=False))
    print(f"\nwrote {OUT_DIR / 'quantile_metrics.csv'}")
    print(f"wrote {OUT_DIR / 'near_80_width_summary.csv'}")
    print(f"wrote {OUT_DIR / 'coverage_width_tradeoff.csv'}")


if __name__ == "__main__":
    main()
