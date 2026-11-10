#!/usr/bin/env python
"""Region-configurable 2x2 decision-focused training diagnostic.

Runs {Direct CI, Fuel-mix surrogate} x {MSE, MSE+SPO+} on the same ERCOT
rolling-origin windows. This is intentionally a one-shot diagnostic: lambda is
chosen from validation decision regret, and test metrics are evaluated once.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/carbon_forecast_matplotlib")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from scipy.stats import wilcoxon
except ModuleNotFoundError:  # pragma: no cover
    wilcoxon = None

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
for path in (SCRIPT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fuel_mix_trainer import (  # noqa: E402
    FuelMixTrainer,
    build_supervised_table,
    feature_columns_from_table,
    fuel_mix_to_carbon_intensity_torch,
    iterate_minibatches,
    load_config,
    make_model_arrays,
    set_random_seed,
    standardize_from_train,
    standardize_targets_from_train,
    to_torch,
)
from make_horizon_comparison_tables import DirectCIMLP, true_ci  # noqa: E402
from run_decision_eval import cap_label, decision_weights, rank_corr, top_k_overlap  # noqa: E402

DEFAULT_REGION = "ercot"
REGION_LABELS = {
    "isone": "ISO-NE",
    "nyiso": "NYISO",
    "pjm": "PJM",
    "ercot": "ERCOT",
    "caiso": "CAISO",
    "miso": "MISO",
    "ieso": "IESO",
    "sim_clean": "Sim clean",
    "sim_boundary": "Sim boundary",
}
METHODS = ["Climatology", "24h persistence", "Direct MSE", "Direct SPO+", "Fuel-mix MSE", "Fuel-mix SPO+"]
OUT_DIR = REPO_ROOT / "results" / "experiments" / "spoplus_2x2_ercot"


def parse_cap(value: str) -> float:
    value = value.strip().lower()
    if value.endswith("h"):
        return 1.0 / float(value[:-1])
    if "/" in value:
        num, den = value.split("/", 1)
        return float(num) / float(den)
    return float(value)


def parse_floats(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_starts(value: str) -> list[pd.Timestamp]:
    return [pd.Timestamp(v.strip(), tz="UTC") for v in value.split(",") if v.strip()]


def label_for_region(region: str) -> str:
    if region in REGION_LABELS:
        return REGION_LABELS[region]
    if region.startswith("sim_"):
        return region.replace("_", " ")
    return region


def complete_daily(values: np.ndarray) -> np.ndarray:
    n_days = len(values) // 24
    return np.asarray(values[: n_days * 24]).reshape(n_days, 24)


def solve_weights_matrix(costs: np.ndarray, cap: float) -> np.ndarray:
    return np.vstack([decision_weights(row, cap) for row in costs])


def decision_metrics(pred: np.ndarray, true: np.ndarray, cap: float) -> dict:
    pred_days = complete_daily(pred)
    true_days = complete_daily(true)
    if len(pred_days) == 0:
        raise ValueError("No complete daily blocks for decision evaluation")

    pred_w = solve_weights_matrix(pred_days, cap)
    oracle_w = solve_weights_matrix(true_days, cap)
    flat_w = np.ones_like(true_days) / 24.0

    method_emission = np.sum(pred_w * true_days, axis=1)
    oracle_emission = np.sum(oracle_w * true_days, axis=1)
    flat_emission = np.sum(flat_w * true_days, axis=1)
    saving = flat_emission - oracle_emission
    safe = saving > 1e-12
    normalized_regret = np.full_like(saving, np.nan, dtype=float)
    capture = np.full_like(saving, np.nan, dtype=float)
    normalized_regret[safe] = (method_emission[safe] - oracle_emission[safe]) / saving[safe]
    capture[safe] = (flat_emission[safe] - method_emission[safe]) / saving[safe]

    return {
        "mean_emission": float(np.nanmean(method_emission)),
        "mean_normalized_regret": float(np.nanmean(normalized_regret)),
        "mean_capture": float(np.nanmean(capture)),
        "median_capture": float(np.nanmedian(capture)),
        "mean_spearman": float(np.nanmean([
            rank_corr(p, y) for p, y in zip(pred_days, true_days)
        ])),
        "mean_topk_overlap": float(np.nanmean([
            top_k_overlap(p, y, cap) for p, y in zip(pred_days, true_days)
        ])),
        "mean_oracle_saving_pct": float(np.nanmean(
            100.0 * saving / np.maximum(flat_emission, 1e-12)
        )),
    }


def spo_plus_loss(c_pred: torch.Tensor, c_true_np: np.ndarray, cap: float) -> torch.Tensor:
    n_days = min(c_pred.shape[0], len(c_true_np)) // 24
    if n_days == 0:
        return c_pred.sum() * 0.0
    c_day = c_pred[: n_days * 24].reshape(n_days, 24)
    true_day = np.asarray(c_true_np[: n_days * 24], dtype=float).reshape(n_days, 24)
    perturbed = (2.0 * c_day.detach().cpu().numpy()) - true_day
    oracle_w = solve_weights_matrix(true_day, cap)
    perturbed_w = solve_weights_matrix(perturbed, cap)
    grad = torch.tensor(2.0 * (oracle_w - perturbed_w), dtype=c_pred.dtype, device=c_pred.device)
    return torch.mean(torch.sum(grad.detach() * c_day, dim=1))


def complete_days(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only UTC days with all 24 hourly rows (unchanged for gap-free data)."""
    day = frame["timestamp"].dt.floor("D")
    return frame[day.map(day.value_counts()).to_numpy() == 24].copy()


def window_splits(data: pd.DataFrame, starts: list[pd.Timestamp], test_days: int, val_days: int):
    for test_start in starts:
        test_end = test_start + pd.Timedelta(days=test_days)
        val_start = test_start - pd.Timedelta(days=val_days)
        train_df = data[data["timestamp"] < val_start].copy()
        val_df = data[(data["timestamp"] >= val_start) & (data["timestamp"] < test_start)].copy()
        test_df = data[(data["timestamp"] >= test_start) & (data["timestamp"] < test_end)].copy()
        val_df = complete_days(val_df)
        test_df = complete_days(test_df)
        if len(train_df) < 24 * 60 or len(val_df) < 24 or len(test_df) < 24:
            raise ValueError(
                f"{test_start.date()}: insufficient split sizes "
                f"train={len(train_df)} val={len(val_df)} test={len(test_df)}"
            )
        yield test_start.strftime("%Y-%m-%d"), train_df, val_df, test_df


def fit_direct_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict,
    seed: int,
    device: str,
    cap: float,
    lambda_spo: float,
    use_spo: bool,
    max_epochs: int,
    patience: int,
    min_epochs: int,
) -> dict:
    set_random_seed(seed)
    x_tr_np, x_va_np, x_te_np, *_ = standardize_from_train(train_df, val_df, test_df, feature_cols)
    y_tr = true_ci(train_df, cfg)
    y_va = true_ci(val_df, cfg)
    y_te = true_ci(test_df, cfg)

    y_mu = float(np.mean(y_tr))
    y_sigma = float(np.std(y_tr) if np.std(y_tr) > 1e-8 else 1.0)
    y_tr_std = (y_tr - y_mu) / y_sigma

    x_tr = torch.tensor(x_tr_np, dtype=torch.float32, device=device)
    x_va = torch.tensor(x_va_np, dtype=torch.float32, device=device)
    x_te = torch.tensor(x_te_np, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr_std, dtype=torch.float32, device=device)

    model = DirectCIMLP(x_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    mse_loss = nn.MSELoss()

    best_state = None
    best_val_regret = float("inf")
    no_improve = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        pred_std = model(x_tr)
        pred_ci = pred_std * y_sigma + y_mu
        loss = mse_loss(pred_std, y_tr_t)
        if use_spo:
            loss = loss + lambda_spo * spo_plus_loss(pred_ci, y_tr, cap)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = (model(x_va).cpu().numpy() * y_sigma) + y_mu
        val_regret = decision_metrics(val_pred, y_va, cap)["mean_normalized_regret"]
        if epoch >= min_epochs and val_regret < best_val_regret:
            best_val_regret = val_regret
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        elif epoch >= min_epochs:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = (model(x_va).cpu().numpy() * y_sigma) + y_mu
        test_pred = (model(x_te).cpu().numpy() * y_sigma) + y_mu
    return {"val_pred": val_pred, "test_pred": test_pred, "y_val": y_va, "y_test": y_te}


def fit_fuelmix_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict,
    seed: int,
    device: str,
    cap: float,
    lambda_spo: float,
    use_spo: bool,
    max_epochs: int,
    patience: int,
    min_epochs: int,
) -> dict:
    set_random_seed(seed)
    fuel_cols = cfg["dataset"]["fuel_cols"]
    emission_factors = cfg["emission_factors"]
    fuelmix_denominator_col = cfg["dataset"].get("fuelmix_denominator_col", "system_load_forecast")
    ci_pred_denominator = cfg["dataset"].get("ci_pred_denominator", "load")

    train_x_np, val_x_np, test_x_np, *_ = standardize_from_train(
        train_df, val_df, test_df, feature_cols
    )
    _, yfuel_tr, dtr, _ = make_model_arrays(train_df, feature_cols, fuel_cols, emission_factors, fuelmix_denominator_col)
    _, yfuel_va, dva, _ = make_model_arrays(val_df, feature_cols, fuel_cols, emission_factors, fuelmix_denominator_col)
    _, yfuel_te, dte, _ = make_model_arrays(test_df, feature_cols, fuel_cols, emission_factors, fuelmix_denominator_col)
    y_tr_ci = true_ci(train_df, cfg)
    y_va_ci = true_ci(val_df, cfg)
    y_te_ci = true_ci(test_df, cfg)

    (yfuel_tr_std, yfuel_va_std, yfuel_te_std, fuel_mu, fuel_sigma) = standardize_targets_from_train(
        yfuel_tr, yfuel_va, yfuel_te
    )
    (dtr_std, dva_std, dte_std, demand_mu_arr, demand_sigma_arr) = standardize_targets_from_train(
        dtr[:, None], dva[:, None], dte[:, None]
    )
    dtr_std = dtr_std.reshape(-1)
    dva_std = dva_std.reshape(-1)
    dte_std = dte_std.reshape(-1)
    demand_mu = float(demand_mu_arr.reshape(-1)[0])
    demand_sigma = float(demand_sigma_arr.reshape(-1)[0])

    xtr = to_torch(train_x_np, device)
    xva = to_torch(val_x_np, device)
    xte = to_torch(test_x_np, device)
    yf_tr = to_torch(yfuel_tr_std, device)
    yf_va = to_torch(yfuel_va_std, device)
    yf_te = to_torch(yfuel_te_std, device)
    d_tr = to_torch(dtr_std, device)
    d_va = to_torch(dva_std, device)
    d_te = to_torch(dte_std, device)

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
        fuel_mu=fuel_mu,
        fuel_sigma=fuel_sigma,
        demand_mu=demand_mu,
        demand_sigma=demand_sigma,
        ci_pred_denominator=ci_pred_denominator,
    )

    best_state = None
    best_val_regret = float("inf")
    no_improve = 0
    batch_size = int(training_cfg.get("batch_size", 64))
    for epoch in range(1, max_epochs + 1):
        for xb, yb_fuel, yb_dem in iterate_minibatches(
            xtr, yf_tr, d_tr, batch_size=batch_size, shuffle=True
        ):
            trainer.step(xb, yb_fuel, yb_dem)

        if use_spo:
            trainer.model.train()
            trainer.optimizer.zero_grad()
            fuel_pred_mw, _, total_gen_pred_mw = trainer.model(xtr)
            fuel_true_mw = trainer.denorm_fuel(yf_tr)
            demand_true_mw = trainer.denorm_demand(d_tr)
            fuel_pred_std = trainer.norm_fuel(fuel_pred_mw)
            total_gen_pred_std = trainer.norm_demand(total_gen_pred_mw)
            ci_pred_den = trainer.ci_prediction_denominator(total_gen_pred_mw, demand_true_mw)
            _, ci_pred = fuel_mix_to_carbon_intensity_torch(
                fuel_pred_mw, trainer.ef_torch, ci_pred_den
            )
            _, ci_true = fuel_mix_to_carbon_intensity_torch(
                fuel_true_mw, trainer.ef_torch, demand_true_mw
            )
            base_loss, _ = trainer.loss_fn(
                fuel_pred_std, yf_tr, ci_pred, ci_true, total_gen_pred_std, d_tr
            )
            loss = base_loss + lambda_spo * spo_plus_loss(ci_pred, y_tr_ci, cap)
            loss.backward()
            trainer.optimizer.step()

        val_metrics, val_outputs = trainer.evaluate(xva, yf_va, d_va)
        val_regret = decision_metrics(val_outputs["ci_pred"], y_va_ci, cap)["mean_normalized_regret"]
        if epoch >= min_epochs and val_regret < best_val_regret:
            best_val_regret = val_regret
            best_state = {k: v.clone() for k, v in trainer.model.state_dict().items()}
            no_improve = 0
        elif epoch >= min_epochs:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        trainer.model.load_state_dict(best_state)
    _, val_outputs = trainer.evaluate(xva, yf_va, d_va)
    _, test_outputs = trainer.evaluate(xte, yf_te, d_te)
    return {
        "val_pred": val_outputs["ci_pred"],
        "test_pred": test_outputs["ci_pred"],
        "y_val": y_va_ci,
        "y_test": y_te_ci,
    }


def ensemble_predictions(runs: list[dict], key: str) -> np.ndarray:
    return np.mean([run[key] for run in runs], axis=0)


def paired_vs_persistence(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pivot = daily.pivot_table(
        index=["window", "day_index"], columns="method", values="decision_emission", aggfunc="first"
    )
    for method in ["Direct MSE", "Direct SPO+", "Fuel-mix MSE", "Fuel-mix SPO+"]:
        diff = pivot[method] - pivot["24h persistence"]
        p_value = float("nan")
        if wilcoxon is not None and np.any(np.abs(diff.to_numpy()) > 1e-12):
            p_value = float(wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
        rows.append({
            "method": method,
            "baseline": "24h persistence",
            "n_days": int(diff.notna().sum()),
            "mean_delta_emission": float(diff.mean()),
            "median_delta_emission": float(diff.median()),
            "wilcoxon_p_two_sided": p_value,
        })
    return pd.DataFrame(rows)


def build_daily_rows(
    region_label: str,
    window: str,
    method: str,
    pred: np.ndarray,
    y_true: np.ndarray,
    cap: float,
) -> list[dict]:
    pred_days = complete_daily(pred)
    true_days = complete_daily(y_true)
    rows = []
    for day_idx, (pred_day, true_day) in enumerate(zip(pred_days, true_days)):
        weights = decision_weights(pred_day, cap)
        oracle_w = decision_weights(true_day, cap)
        flat_w = np.ones(24) / 24.0
        method_emission = float(weights @ true_day)
        oracle_emission = float(oracle_w @ true_day)
        flat_emission = float(flat_w @ true_day)
        saving = flat_emission - oracle_emission
        rows.append({
            "region": region_label,
            "window": window,
            "day_index": day_idx,
            "method": method,
            "decision_emission": method_emission,
            "oracle_emission": oracle_emission,
            "flat_emission": flat_emission,
            "oracle_saving_pct": 100.0 * saving / flat_emission if flat_emission else np.nan,
            "normalized_regret": (method_emission - oracle_emission) / saving if saving > 1e-12 else np.nan,
            "capture": (flat_emission - method_emission) / saving if saving > 1e-12 else np.nan,
            "spearman": rank_corr(pred_day, true_day),
            "topk_overlap": top_k_overlap(pred_day, true_day, cap),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-starts", default="2024-06-01,2024-08-01,2024-10-01")
    parser.add_argument("--test-days", type=int, default=56)
    parser.add_argument("--val-days", type=int, default=30)
    parser.add_argument("--cap", type=parse_cap, default=1 / 6)
    parser.add_argument("--seeds", type=parse_ints, default=[7, 42, 123])
    parser.add_argument("--lambdas", type=parse_floats, default=[0.1, 1.0, 10.0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if args.out_dir == OUT_DIR and args.region != DEFAULT_REGION:
        args.out_dir = REPO_ROOT / "results" / "experiments" / f"spoplus_2x2_{args.region}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    starts = parse_starts(args.test_starts)
    region_label = label_for_region(args.region)
    cfg = load_config(REPO_ROOT / "configs" / "dayahead" / f"{args.region}_fuelmix_dayahead.json")
    data = build_supervised_table(cfg)
    feature_cols = feature_columns_from_table(data, cfg)

    split_rows = list(window_splits(data, starts, args.test_days, args.val_days))
    val_curve_rows = []
    cache: dict[tuple, list[dict]] = {}

    # MSE baselines with the same validation-regret early stopping.
    for window, train_df, val_df, test_df in split_rows:
        for seed in args.seeds:
            cache[("direct", "mse", 0.0, window, seed)] = [
                fit_direct_model(
                    train_df, val_df, test_df, feature_cols, cfg, seed, args.device,
                    args.cap, 0.0, False, args.max_epochs, args.patience, args.min_epochs,
                )
            ]
            cache[("fuelmix", "mse", 0.0, window, seed)] = [
                fit_fuelmix_model(
                    train_df, val_df, test_df, feature_cols, cfg, seed, args.device,
                    args.cap, 0.0, False, args.max_epochs, args.patience, args.min_epochs,
                )
            ]

    # Lambda sweep for SPO+; choose one global lambda by validation regret.
    lambda_val_scores = []
    for lambda_spo in args.lambdas:
        method_val_regrets = []
        for architecture in ["direct", "fuelmix"]:
            for window, train_df, val_df, test_df in split_rows:
                runs = []
                for seed in args.seeds:
                    key = (architecture, "spo", lambda_spo, window, seed)
                    if architecture == "direct":
                        run = fit_direct_model(
                            train_df, val_df, test_df, feature_cols, cfg, seed, args.device,
                            args.cap, lambda_spo, True, args.max_epochs, args.patience, args.min_epochs,
                        )
                    else:
                        run = fit_fuelmix_model(
                            train_df, val_df, test_df, feature_cols, cfg, seed, args.device,
                            args.cap, lambda_spo, True, args.max_epochs, args.patience, args.min_epochs,
                        )
                    cache[key] = [run]
                    runs.append(run)
                val_pred = ensemble_predictions(runs, "val_pred")
                y_val = runs[0]["y_val"]
                val_metrics = decision_metrics(val_pred, y_val, args.cap)
                method_val_regrets.append(val_metrics["mean_normalized_regret"])
                val_curve_rows.append({
                    "architecture": architecture,
                    "lambda_spo": lambda_spo,
                    "window": window,
                    **val_metrics,
                })
        lambda_val_scores.append({
            "lambda_spo": lambda_spo,
            "mean_val_normalized_regret": float(np.mean(method_val_regrets)),
        })

    lambda_curve = pd.DataFrame(lambda_val_scores)
    selected_lambda = float(lambda_curve.sort_values("mean_val_normalized_regret").iloc[0]["lambda_spo"])

    daily_rows = []
    prediction_rows = []
    for window, train_df, val_df, test_df in split_rows:
        y_test = true_ci(test_df, cfg)
        fit_df = pd.concat([train_df, val_df], ignore_index=True)
        y_fit = true_ci(fit_df, cfg)
        hour_climatology = (
            pd.DataFrame({"hour": fit_df["hour"].to_numpy(), "ci": y_fit})
            .groupby("hour")["ci"]
            .mean()
        )
        baseline_preds = {
            "Climatology": test_df["hour"].map(hour_climatology).to_numpy(dtype=float),
            "24h persistence": test_df["historical_ci_24h_lag"].to_numpy(dtype=float),
        }
        method_preds = {}
        for method_name, architecture, objective, lambda_value in [
            ("Direct MSE", "direct", "mse", 0.0),
            ("Fuel-mix MSE", "fuelmix", "mse", 0.0),
            ("Direct SPO+", "direct", "spo", selected_lambda),
            ("Fuel-mix SPO+", "fuelmix", "spo", selected_lambda),
        ]:
            runs = [
                cache[(architecture, objective, lambda_value, window, seed)][0]
                for seed in args.seeds
            ]
            method_preds[method_name] = ensemble_predictions(runs, "test_pred")

        for method_name, pred in {**baseline_preds, **method_preds}.items():
            daily_rows.extend(build_daily_rows(region_label, window, method_name, pred, y_test, args.cap))

        pred_df = pd.DataFrame({
            "window": window,
            "timestamp": test_df["timestamp"].to_numpy(),
            "ci_true": y_test,
            **{name.replace(" ", "_").replace("+", "plus").lower(): pred
               for name, pred in {**baseline_preds, **method_preds}.items()},
        })
        prediction_rows.append(pred_df)

    daily = pd.DataFrame(daily_rows)
    summary_rows = []
    for method, group in daily.groupby("method", sort=False):
        summary_rows.append({
            "method": method,
            "n_days": int(group[["window", "day_index"]].drop_duplicates().shape[0]),
            "mean_capture": float(group["capture"].mean()),
            "median_capture": float(group["capture"].median()),
            "mean_normalized_regret": float(group["normalized_regret"].mean()),
            "mean_spearman": float(group["spearman"].mean()),
            "mean_topk_overlap": float(group["topk_overlap"].mean()),
            "mean_oracle_saving_pct": float(group["oracle_saving_pct"].mean()),
        })
    summary = pd.DataFrame(summary_rows)
    pairwise = paired_vs_persistence(daily)
    val_curve_details = pd.DataFrame(val_curve_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)

    prefix = f"{args.region}_spoplus"
    summary.to_csv(args.out_dir / f"{prefix}_2x2_summary.csv", index=False)
    daily.to_csv(args.out_dir / f"{prefix}_2x2_daily.csv", index=False)
    pairwise.to_csv(args.out_dir / f"{prefix}_2x2_vs_persistence.csv", index=False)
    lambda_curve.to_csv(args.out_dir / f"{prefix}_lambda_curve.csv", index=False)
    val_curve_details.to_csv(args.out_dir / f"{prefix}_lambda_curve_by_window.csv", index=False)
    predictions.to_csv(args.out_dir / f"{prefix}_2x2_predictions.csv", index=False)

    print("Validation lambda curve:")
    print(lambda_curve.round(5).to_string(index=False))
    print(f"selected_lambda={selected_lambda}")
    print("\nTest summary:")
    print(summary.round(5).to_string(index=False))
    print("\nVs persistence:")
    print(pairwise.round(5).to_string(index=False))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
