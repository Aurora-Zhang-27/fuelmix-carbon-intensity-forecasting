#!/usr/bin/env python
"""Rolling-origin decision evaluation for one region.

This is a diagnostic-only experiment: it repeats the existing day-ahead
FuelMixForecaster and Direct MLP training objective over multiple chronological
test windows, then evaluates the same deferrable-load scheduling metric used by
run_decision_eval.py. No model architecture, loss, or feature-set changes are
introduced here.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/carbon_forecast_matplotlib")

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
for path in (SCRIPT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from make_horizon_comparison_tables import (  # noqa: E402
    DIRECT_SEEDS,
    fit_direct_mlp_train_val,
    true_ci,
)
from run_decision_eval import (  # noqa: E402
    METHOD_ORDER,
    PLOT_METHODS,
    cap_label,
    decision_weights,
    rank_corr,
    top_k_decomposition,
    top_k_overlap,
)
from fuel_mix_trainer import (  # noqa: E402
    FuelMixTrainer,
    build_supervised_table,
    feature_columns_from_table,
    iterate_minibatches,
    load_config,
    make_model_arrays,
    set_random_seed,
    standardize_from_train,
    standardize_targets_from_train,
    to_torch,
)

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


def parse_cap(value: str) -> float:
    value = value.strip().lower()
    if value.endswith("h"):
        return 1.0 / float(value[:-1])
    if "/" in value:
        num, den = value.split("/", 1)
        return float(num) / float(den)
    return float(value)


def parse_starts(value: str) -> list[pd.Timestamp]:
    return [pd.Timestamp(v.strip(), tz="UTC") for v in value.split(",") if v.strip()]


def train_fuelmix_split(
    cfg: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
    device: str,
) -> dict[str, np.ndarray]:
    """Train the existing fuel-mix surrogate on a supplied chronological split."""
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
        balance_coefficients=cfg["dataset"].get("balance_coefficients"),
        signed_flow_cols=cfg["dataset"].get("signed_flow_cols"),
    )

    epochs = int(training_cfg.get("max_epochs", 40))
    batch_size = int(training_cfg.get("batch_size", 64))
    patience = int(training_cfg.get("patience", 5))
    min_epochs = int(training_cfg.get("min_epochs", 0))
    best_val_rmse = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        for xb, yb_fuel, yb_dem in iterate_minibatches(
            xtr, yf_tr, d_tr, batch_size=batch_size, shuffle=True
        ):
            trainer.step(xb, yb_fuel, yb_dem)
        val_metrics, _ = trainer.evaluate(xva, yf_va, d_va)
        val_rmse = val_metrics["ci_rmse_t_per_mwh"]
        if epoch >= min_epochs and val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.clone() for k, v in trainer.model.state_dict().items()}
            no_improve = 0
        elif epoch >= min_epochs:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        trainer.model.load_state_dict(best_state)
    val_metrics, val_outputs = trainer.evaluate(xva, yf_va, d_va)
    test_metrics, outputs = trainer.evaluate(xte, yf_te, d_te)
    return {
        "val_ci_pred": val_outputs["ci_pred"],
        "val_ci_rmse": val_metrics["ci_rmse_t_per_mwh"],
        "ci_pred": outputs["ci_pred"],
        "test_ci_rmse": test_metrics["ci_rmse_t_per_mwh"],
        "fuel_pred": outputs["fuel_pred"],
        "total_gen_pred": outputs["total_gen_pred"],
        "net_supply_pred": outputs["net_supply_pred"],
    }


def fuel_share_columns(
    fuel_pred: np.ndarray,
    fuel_cols: list[str],
    fossil_fuels: list[str],
) -> dict[str, np.ndarray]:
    total = np.maximum(fuel_pred.sum(axis=1), 1e-9)
    out: dict[str, np.ndarray] = {}
    for idx, fuel in enumerate(fuel_cols):
        out[f"fuelmix_pred_{fuel}_mw"] = fuel_pred[:, idx]
        out[f"fuelmix_pred_{fuel}_share"] = fuel_pred[:, idx] / total
    fossil_idx = [fuel_cols.index(fuel) for fuel in fossil_fuels if fuel in fuel_cols]
    clean_idx = [
        fuel_cols.index(fuel)
        for fuel in ["nuclear", "hydro", "wind", "solar"]
        if fuel in fuel_cols
    ]
    out["fuelmix_pred_fossil_share"] = (
        fuel_pred[:, fossil_idx].sum(axis=1) / total
        if fossil_idx else np.zeros(len(fuel_pred))
    )
    out["fuelmix_pred_clean_share"] = (
        fuel_pred[:, clean_idx].sum(axis=1) / total
        if clean_idx else np.zeros(len(fuel_pred))
    )
    out["fuelmix_pred_total_generation_mw"] = fuel_pred.sum(axis=1)
    return out


def complete_days(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only UTC days with all 24 hourly rows (unchanged for gap-free data)."""
    day = frame["timestamp"].dt.floor("D")
    return frame[day.map(day.value_counts()).to_numpy() == 24].copy()


def complete_daily(values: np.ndarray) -> np.ndarray:
    n_days = len(values) // 24
    return np.asarray(values[: n_days * 24]).reshape(n_days, 24)


def evaluate_window(
    region_label: str,
    window_name: str,
    test_df: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    caps: list[float],
) -> tuple[list[dict], list[dict]]:
    y_true = predictions["Oracle"]
    true_blocks = complete_daily(y_true)
    timestamps = complete_daily(test_df["timestamp"].to_numpy())
    pred_blocks = {name: complete_daily(values) for name, values in predictions.items()}
    n_days = true_blocks.shape[0]
    daily_rows = []
    tail_rows = []

    for cap in caps:
        for day_idx in range(n_days):
            true = true_blocks[day_idx]
            flat_w = np.ones(24, dtype=float) / 24.0
            oracle_w = decision_weights(true, cap)
            flat_emission = float(flat_w @ true)
            oracle_emission = float(oracle_w @ true)
            saving_abs = flat_emission - oracle_emission
            saving_pct = 100.0 * saving_abs / flat_emission if flat_emission else float("nan")

            for method in METHOD_ORDER:
                pred = pred_blocks[method][day_idx]
                if method == "Oracle":
                    weights = oracle_w
                    spearman = 1.0
                    overlap = 1.0
                elif method == "Flat":
                    weights = flat_w
                    spearman = float("nan")
                    overlap = float("nan")
                else:
                    weights = decision_weights(pred, cap)
                    spearman = rank_corr(pred, true)
                    overlap = top_k_overlap(pred, true, cap)
                    tail = top_k_decomposition(pred, true, cap)
                    tail.update({
                        "region": region_label,
                        "window": window_name,
                        "cap": cap,
                        "cap_label": cap_label(cap),
                        "day_index": day_idx,
                        "block_start": timestamps[day_idx, 0],
                        "method": method,
                    })
                    tail_rows.append(tail)

                emission = float(weights @ true)
                if saving_abs > 1e-12:
                    normalized_regret = (emission - oracle_emission) / saving_abs
                    capture = (flat_emission - emission) / saving_abs
                else:
                    normalized_regret = float("nan")
                    capture = float("nan")

                daily_rows.append({
                    "region": region_label,
                    "window": window_name,
                    "cap": cap,
                    "cap_label": cap_label(cap),
                    "day_index": day_idx,
                    "block_start": timestamps[day_idx, 0],
                    "method": method,
                    "decision_emission": emission,
                    "flat_emission": flat_emission,
                    "oracle_emission": oracle_emission,
                    "oracle_saving_abs": saving_abs,
                    "oracle_saving_pct": saving_pct,
                    "normalized_regret": normalized_regret,
                    "capture": capture,
                    "spearman": spearman,
                    "topk_overlap": overlap,
                })
    return daily_rows, tail_rows


def summarize(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (region, cap, method), group in daily.groupby(["region", "cap", "method"], sort=False):
        rows.append({
            "region": region,
            "cap": cap,
            "cap_label": cap_label(float(cap)),
            "method": method,
            "n_days": int(group[["window", "day_index"]].drop_duplicates().shape[0]),
            "mean_capture": float(group["capture"].mean()),
            "median_capture": float(group["capture"].median()),
            "mean_normalized_regret": float(group["normalized_regret"].mean()),
            "mean_spearman": float(group["spearman"].mean()),
            "mean_topk_overlap": float(group["topk_overlap"].mean()),
            "mean_oracle_saving_pct": float(group["oracle_saving_pct"].mean()),
        })
    return pd.DataFrame(rows)


def paired_tests_rolling(daily: pd.DataFrame) -> pd.DataFrame:
    try:
        from scipy.stats import wilcoxon
    except ModuleNotFoundError:  # pragma: no cover
        wilcoxon = None

    rows = []
    baselines = ["Climatology", "24h persistence"]
    methods = ["Direct MLP", "Fuel-mix ens."]
    for (region, cap), group in daily.groupby(["region", "cap"]):
        pivot = group.pivot_table(
            index=["window", "day_index"],
            columns="method",
            values="decision_emission",
            aggfunc="first",
        )
        for method in methods:
            for baseline in baselines:
                if method not in pivot or baseline not in pivot:
                    continue
                paired = pivot[[method, baseline]].dropna()
                diff = paired[method] - paired[baseline]
                p_value = float("nan")
                if wilcoxon is not None and len(diff) > 0 and np.any(np.abs(diff.to_numpy()) > 1e-12):
                    p_value = float(wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
                rows.append({
                    "region": region,
                    "cap": cap,
                    "cap_label": cap_label(float(cap)),
                    "method": method,
                    "baseline": baseline,
                    "n_days": int(len(diff)),
                    "mean_delta_emission": float(diff.mean()),
                    "median_delta_emission": float(diff.median()),
                    "wilcoxon_p_two_sided": p_value,
                })
    return pd.DataFrame(rows)


def plot_capture(daily: pd.DataFrame, out_dir: Path, region: str, region_label: str) -> None:
    import matplotlib.pyplot as plt

    cap_df = daily[(daily["cap_label"] == "1over6") & (daily["method"].isin(PLOT_METHODS))]
    values = [
        cap_df[cap_df["method"] == method]["capture"].dropna().to_numpy()
        for method in PLOT_METHODS
    ]
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.boxplot(values, tick_labels=["Clim.", "Persist", "MLP", "Fuel"], showfliers=False)
    ax.axhline(0.0, color="0.7", linewidth=0.8)
    ax.axhline(1.0, color="0.7", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Capture ratio")
    ax.set_title(f"{region_label} rolling-origin scheduling capture, cap=1/6")
    ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{region}_rolling_capture_boxplot_cap_1over6.pdf")
    fig.savefig(out_dir / f"{region}_rolling_capture_boxplot_cap_1over6.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="nyiso")
    parser.add_argument("--test-starts", default="2024-06-01,2024-08-01,2024-10-01")
    parser.add_argument("--test-days", type=int, default=56)
    parser.add_argument("--val-days", type=int, default=30)
    parser.add_argument("--caps", nargs="+", type=parse_cap, default=[1 / 6])
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--make-plots", action="store_true")
    args = parser.parse_args()

    region_label = label_for_region(args.region)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    starts = parse_starts(args.test_starts)
    out_dir = args.out_dir or (
        REPO_ROOT / "results" / "experiments" / f"decision_eval_rolling_{args.region}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(REPO_ROOT / "configs" / "dayahead" / f"{args.region}_fuelmix_dayahead.json")
    data = build_supervised_table(cfg)
    feature_cols = feature_columns_from_table(data, cfg)
    fuel_cols = cfg["dataset"]["fuel_cols"]
    fossil_fuels = cfg.get("fossil_fuels", [])
    y_all = true_ci(data, cfg)
    data = data.copy()
    data["ci_true_eval"] = y_all

    daily_rows = []
    tail_rows = []
    prediction_rows = []
    rmse_rows = []
    protocol_rows = []

    for test_start in starts:
        test_end = test_start + pd.Timedelta(days=args.test_days)
        val_start = test_start - pd.Timedelta(days=args.val_days)
        train_df = data[data["timestamp"] < val_start].copy()
        val_df = data[(data["timestamp"] >= val_start) & (data["timestamp"] < test_start)].copy()
        test_df = data[(data["timestamp"] >= test_start) & (data["timestamp"] < test_end)].copy()
        test_df = complete_days(test_df)
        n_test_days = len(test_df) // 24
        window_name = test_start.strftime("%Y-%m-%d")

        if len(train_df) < 24 * 60 or len(val_df) < 24 * 10 or len(test_df) < 24:
            raise ValueError(
                f"{window_name}: insufficient rows train={len(train_df)} "
                f"val={len(val_df)} test={len(test_df)}"
            )

        print(
            f"[{region_label} {window_name}] train={len(train_df)} val={len(val_df)} "
            f"test={len(test_df)} days={n_test_days}"
        )
        fit_df = pd.concat([train_df, val_df], ignore_index=True)
        y_fit = true_ci(fit_df, cfg)
        y_train = true_ci(train_df, cfg)
        y_val = true_ci(val_df, cfg)
        y_test = true_ci(test_df, cfg)

        hour_climatology = (
            pd.DataFrame({"hour": fit_df["hour"].to_numpy(), "ci": y_fit})
            .groupby("hour")["ci"]
            .mean()
        )
        climatology = test_df["hour"].map(hour_climatology).to_numpy(dtype=float)
        direct_runs = [
            fit_direct_mlp_train_val(
                train_df[feature_cols].to_numpy(dtype=float),
                y_train,
                val_df[feature_cols].to_numpy(dtype=float),
                y_val,
                test_df[feature_cols].to_numpy(dtype=float),
                seed=seed,
                device=args.device,
            )
            for seed in DIRECT_SEEDS
        ]
        direct_preds = [run["test_pred"] for run in direct_runs]
        direct_pred = np.mean(direct_preds, axis=0)
        protocol_rows.append({
            "region": region_label,
            "window": window_name,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "direct_fit_rows": int(len(train_df)),
            "fuelmix_fit_rows": int(len(train_df)),
            "climatology_history_rows": int(len(fit_df)),
            "direct_best_epochs": ",".join(str(run["best_epoch"]) for run in direct_runs),
            "direct_best_val_rmse": ",".join(
                f"{run['best_val_rmse']:.10g}" for run in direct_runs
            ),
        })

        fuelmix_runs = [
            train_fuelmix_split(cfg, train_df, val_df, test_df, feature_cols, seed, args.device)
            for seed in seeds
        ]
        fuelmix_pred = np.mean([run["ci_pred"] for run in fuelmix_runs], axis=0)
        fuelmix_fuel_pred = np.mean([run["fuel_pred"] for run in fuelmix_runs], axis=0)

        predictions = {
            "Oracle": y_test,
            "Flat": np.zeros_like(y_test),
            "Climatology": climatology,
            "24h persistence": test_df["historical_ci_24h_lag"].to_numpy(dtype=float),
            "Direct MLP": direct_pred,
            "Fuel-mix ens.": fuelmix_pred,
        }
        for method in ["Climatology", "24h persistence", "Direct MLP", "Fuel-mix ens."]:
            err = predictions[method] - y_test
            rmse_rows.append({
                "region": region_label,
                "window": window_name,
                "method": method,
                "n_test": int(len(y_test)),
                "sse": float(np.sum(err ** 2)),
                "ci_rmse_t_per_mwh": float(np.sqrt(np.mean(err ** 2))),
                "ci_mae": float(np.mean(np.abs(err))),
                "mean_ci": float(np.mean(y_test)),
            })
        window_daily, window_tail = evaluate_window(region_label, window_name, test_df, predictions, args.caps)
        daily_rows.extend(window_daily)
        tail_rows.extend(window_tail)

        pred_df = pd.DataFrame({
            "window": window_name,
            "timestamp": test_df["timestamp"].to_numpy(),
            "ci_true": y_test,
            "climatology": climatology,
            "persistence_24h": predictions["24h persistence"],
            "direct_mlp": direct_pred,
            "fuelmix_ens": fuelmix_pred,
        })
        for col, values in fuel_share_columns(fuelmix_fuel_pred, fuel_cols, fossil_fuels).items():
            pred_df[col] = values
        prediction_rows.append(pred_df)

    daily = pd.DataFrame(daily_rows)
    tail = pd.DataFrame(tail_rows)
    summary = summarize(daily)
    tests = paired_tests_rolling(daily)
    predictions_out = pd.concat(prediction_rows, ignore_index=True)

    daily.to_csv(out_dir / f"{args.region}_rolling_decision_daily.csv", index=False)
    tail.to_csv(out_dir / f"{args.region}_rolling_tail_daily.csv", index=False)
    summary.to_csv(out_dir / f"{args.region}_rolling_decision_summary.csv", index=False)
    tests.to_csv(out_dir / f"{args.region}_rolling_pairwise_wilcoxon.csv", index=False)
    predictions_out.to_csv(out_dir / f"{args.region}_rolling_predictions.csv", index=False)
    rmse_df = pd.DataFrame(rmse_rows)
    pooled = (
        rmse_df.groupby(["region", "method"], as_index=False)
        .agg(n_test=("n_test", "sum"), sse=("sse", "sum"))
    )
    pooled["ci_rmse_t_per_mwh"] = np.sqrt(pooled["sse"] / pooled["n_test"])
    pooled["n_windows"] = rmse_df.groupby(["region", "method"])["window"].nunique().to_numpy()

    rmse_df.to_csv(out_dir / f"{args.region}_rolling_rmse_by_window.csv", index=False)
    pooled.to_csv(out_dir / f"{args.region}_rolling_rmse_pooled.csv", index=False)
    pd.DataFrame(protocol_rows).to_csv(
        out_dir / f"{args.region}_rolling_protocol_audit.csv", index=False
    )
    if args.make_plots:
        plot_capture(daily, out_dir, args.region, region_label)

    focus = summary[
        (summary["cap_label"] == "1over6")
        & (summary["method"].isin(["Climatology", "24h persistence", "Direct MLP", "Fuel-mix ens."]))
    ][[
        "method",
        "n_days",
        "mean_capture",
        "median_capture",
        "mean_spearman",
        "mean_topk_overlap",
        "mean_oracle_saving_pct",
    ]]
    print(focus.round(4).to_string(index=False))
    print("\nRolling-origin pooled RMSE:")
    print(pooled.sort_values("ci_rmse_t_per_mwh").round(6).to_string(index=False))
    print("\nPairwise tests:")
    print(tests[tests["cap_label"] == "1over6"].round(5).to_string(index=False))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
