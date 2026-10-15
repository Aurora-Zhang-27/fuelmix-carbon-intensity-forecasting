#!/usr/bin/env python
"""Evaluate carbon-aware scheduling decisions from day-ahead CI forecasts.

This script does not train the fuel-mix surrogate. It reuses saved day-ahead
fuel-mix ensemble predictions, recomputes the direct-MLP baseline with the same
feature set, and evaluates whether each forecast selects low-carbon hours well
for a simple deferrable-load scheduling problem.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/carbon_forecast_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    wilcoxon = None

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
for path in (SCRIPT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from make_horizon_comparison_tables import (  # noqa: E402
    DIRECT_SEEDS,
    LABELS,
    REGIONS,
    fit_direct_mlp_train_val,
    label_for_region,
    true_ci,
)
from fuel_mix_trainer import (  # noqa: E402
    build_supervised_table,
    feature_columns_from_table,
    load_config,
    split_train_val_test,
)

OUT_DIR = REPO_ROOT / "results" / "experiments" / "decision_eval"
METHOD_ORDER = [
    "Oracle",
    "Flat",
    "Climatology",
    "24h persistence",
    "Direct MLP",
    "Fuel-mix ens.",
]
PLOT_METHODS = ["Climatology", "24h persistence", "Direct MLP", "Fuel-mix ens."]


def parse_cap(value: str) -> float:
    """Parse a cap such as 0.1667, 1/6, or 6h into max hourly energy."""
    value = value.strip().lower()
    if value.endswith("h"):
        hours = float(value[:-1])
        if hours <= 0:
            raise argparse.ArgumentTypeError("cap hours must be positive")
        return 1.0 / hours
    if "/" in value:
        num, den = value.split("/", 1)
        cap = float(num) / float(den)
    else:
        cap = float(value)
    if not (0 < cap <= 1):
        raise argparse.ArgumentTypeError("cap must be in (0, 1]")
    return cap


def cap_label(cap: float) -> str:
    hours = 1.0 / cap
    if abs(hours - round(hours)) < 1e-6:
        return f"1over{int(round(hours))}"
    return f"{cap:.3f}".replace(".", "p")


def rank_corr(pred: np.ndarray, true: np.ndarray) -> float:
    pred_rank = pd.Series(pred).rank(method="average").to_numpy()
    true_rank = pd.Series(true).rank(method="average").to_numpy()
    if pred_rank.std() < 1e-12 or true_rank.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(pred_rank, true_rank)[0, 1])


def decision_weights(cost: np.ndarray, cap: float) -> np.ndarray:
    """Solve min cost^T w s.t. sum(w)=1, 0<=w<=cap by sorting costs."""
    cost = np.asarray(cost, dtype=float)
    order = np.argsort(cost)
    weights = np.zeros_like(cost, dtype=float)
    remaining = 1.0
    for idx in order:
        amount = min(cap, remaining)
        weights[idx] = amount
        remaining -= amount
        if remaining <= 1e-12:
            break
    if remaining > 1e-8:
        raise ValueError(f"cap={cap} cannot allocate full unit energy")
    return weights


def top_k_overlap(pred: np.ndarray, true: np.ndarray, cap: float) -> float:
    k = int(math.ceil(1.0 / cap - 1e-12))
    pred_set = set(np.argsort(pred)[:k].tolist())
    true_set = set(np.argsort(true)[:k].tolist())
    return len(pred_set & true_set) / k


def top_k_decomposition(pred: np.ndarray, true: np.ndarray, cap: float) -> dict:
    """Describe the bottom-k selection error that drives scheduling regret.

    When cap=1/k, the scheduling decision places equal mass on the k predicted
    cleanest hours. The regret is exactly the realized-CI mean of the selected
    hours minus the realized-CI mean of the oracle bottom-k hours.
    """
    k = int(math.ceil(1.0 / cap - 1e-12))
    selected = set(np.argsort(pred)[:k].tolist())
    oracle = set(np.argsort(true)[:k].tolist())
    false_selected = sorted(selected - oracle)
    missed_oracle = sorted(oracle - selected)
    overlap = len(selected & oracle)

    selected_vals = true[list(selected)]
    oracle_vals = true[list(oracle)]
    false_vals = true[false_selected] if false_selected else np.array([], dtype=float)
    missed_vals = true[missed_oracle] if missed_oracle else np.array([], dtype=float)

    selected_mean = float(np.mean(selected_vals))
    oracle_mean = float(np.mean(oracle_vals))
    false_mean = float(np.mean(false_vals)) if len(false_vals) else float("nan")
    missed_mean = float(np.mean(missed_vals)) if len(missed_vals) else float("nan")
    return {
        "k": k,
        "topk_overlap_count": overlap,
        "topk_overlap": overlap / k,
        "miss_count": k - overlap,
        "miss_rate": 1.0 - overlap / k,
        "selected_true_ci_mean": selected_mean,
        "oracle_true_ci_mean": oracle_mean,
        "tail_excess_ci": selected_mean - oracle_mean,
        "false_selected_count": len(false_selected),
        "false_selected_true_ci_mean": false_mean,
        "missed_oracle_true_ci_mean": missed_mean,
        "replacement_gap": false_mean - missed_mean if len(false_vals) else 0.0,
    }


def complete_daily_blocks(values: np.ndarray, block_size: int = 24) -> np.ndarray:
    values = np.asarray(values)
    n_blocks = len(values) // block_size
    return values[: n_blocks * block_size].reshape(n_blocks, block_size)


def method_predictions(region: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    cfg = load_config(str(REPO_ROOT / "configs" / "dayahead" / f"{region}_fuelmix_dayahead.json"))
    data = build_supervised_table(cfg)
    feature_cols = feature_columns_from_table(data, cfg)
    train_df, val_df, test_df = split_train_val_test(
        data,
        train_frac=cfg.get("split", {}).get("train_frac", 0.7),
        val_frac=cfg.get("split", {}).get("val_frac", 0.15),
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
        )
        for seed in DIRECT_SEEDS
    ]
    direct_preds = [run["test_pred"] for run in direct_runs]
    direct_pred = np.mean(direct_preds, axis=0)

    ensemble = pd.read_csv(
        REPO_ROOT / "results" / "fuelmix_dayahead" / "multiseed" / f"{region}_ensemble_predictions.csv",
        parse_dates=["timestamp"],
    )
    if len(ensemble) != len(test_df):
        raise ValueError(
            f"{region}: fuel-mix ensemble length {len(ensemble)} does not match "
            f"test length {len(test_df)}"
        )

    test_meta = test_df[["timestamp", "hour"]].reset_index(drop=True).copy()
    test_meta["ci_true"] = y_test

    return test_meta, {
        "Oracle": y_test,
        "Flat": np.zeros_like(y_test),
        "Climatology": climatology,
        "24h persistence": test_df["historical_ci_24h_lag"].to_numpy(dtype=float),
        "Direct MLP": direct_pred,
        "Fuel-mix ens.": ensemble["ci_pred"].to_numpy(dtype=float),
    }


def evaluate_region(region: str, caps: list[float]) -> tuple[list[dict], list[dict], list[dict]]:
    meta, preds = method_predictions(region)
    label = label_for_region(region)

    timestamps = complete_daily_blocks(meta["timestamp"].to_numpy())
    true_blocks = complete_daily_blocks(meta["ci_true"].to_numpy(dtype=float))
    n_blocks = true_blocks.shape[0]
    start_times = timestamps[:, 0]
    pred_blocks = {name: complete_daily_blocks(values) for name, values in preds.items()}

    daily_rows: list[dict] = []
    tail_rows: list[dict] = []
    for cap in caps:
        for day_idx in range(n_blocks):
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
                        "region": label,
                        "cap": cap,
                        "cap_label": cap_label(cap),
                        "day_index": day_idx,
                        "block_start": start_times[day_idx],
                        "method": method,
                    })
                    tail_rows.append(tail)

                emission = float(weights @ true)
                if saving_abs > 1e-12:
                    regret_norm = (emission - oracle_emission) / saving_abs
                    capture = (flat_emission - emission) / saving_abs
                else:
                    regret_norm = float("nan")
                    capture = float("nan")

                daily_rows.append({
                    "region": label,
                    "cap": cap,
                    "cap_label": cap_label(cap),
                    "day_index": day_idx,
                    "block_start": start_times[day_idx],
                    "method": method,
                    "decision_emission": emission,
                    "flat_emission": flat_emission,
                    "oracle_emission": oracle_emission,
                    "oracle_saving_abs": saving_abs,
                    "oracle_saving_pct": saving_pct,
                    "normalized_regret": regret_norm,
                    "capture": capture,
                    "spearman": spearman,
                    "topk_overlap": overlap,
                })

    daily = pd.DataFrame(daily_rows)
    summary_rows: list[dict] = []
    for (cap, method), group in daily.groupby(["cap", "method"], sort=False):
        summary_rows.append({
            "region": label,
            "cap": cap,
            "cap_label": cap_label(cap),
            "method": method,
            "n_days": int(group["day_index"].nunique()),
            "mean_decision_emission": float(group["decision_emission"].mean()),
            "median_decision_emission": float(group["decision_emission"].median()),
            "mean_normalized_regret": float(group["normalized_regret"].mean()),
            "median_normalized_regret": float(group["normalized_regret"].median()),
            "mean_capture": float(group["capture"].mean()),
            "median_capture": float(group["capture"].median()),
            "mean_spearman": float(group["spearman"].mean()),
            "mean_topk_overlap": float(group["topk_overlap"].mean()),
            "mean_oracle_saving_abs": float(group["oracle_saving_abs"].mean()),
            "mean_oracle_saving_pct": float(group["oracle_saving_pct"].mean()),
        })
    return daily_rows, summary_rows, tail_rows


def summarize_tail(tail: pd.DataFrame) -> pd.DataFrame:
    if tail.empty:
        return pd.DataFrame()
    rows = []
    for (region, cap, method), group in tail.groupby(["region", "cap", "method"], sort=False):
        rows.append({
            "region": region,
            "cap": cap,
            "cap_label": cap_label(float(cap)),
            "method": method,
            "n_days": int(group["day_index"].nunique()),
            "k": int(group["k"].iloc[0]),
            "mean_miss_count": float(group["miss_count"].mean()),
            "mean_miss_rate": float(group["miss_rate"].mean()),
            "mean_topk_overlap": float(group["topk_overlap"].mean()),
            "median_topk_overlap": float(group["topk_overlap"].median()),
            "mean_tail_excess_ci": float(group["tail_excess_ci"].mean()),
            "median_tail_excess_ci": float(group["tail_excess_ci"].median()),
            "mean_replacement_gap": float(group["replacement_gap"].mean()),
            "median_replacement_gap": float(group["replacement_gap"].median()),
            "mean_false_selected_ci": float(group["false_selected_true_ci_mean"].mean()),
            "mean_missed_oracle_ci": float(group["missed_oracle_true_ci_mean"].mean()),
        })
    return pd.DataFrame(rows)


def summarize_miss_penalty(tail: pd.DataFrame) -> pd.DataFrame:
    if tail.empty:
        return pd.DataFrame()
    rows = []
    for (region, cap, method), group in tail.groupby(["region", "cap", "method"], sort=False):
        positive_miss = group[group["miss_count"] > 0].copy()
        penalty_q = positive_miss["replacement_gap"].quantile([0.25, 0.5, 0.75, 0.9]) if not positive_miss.empty else pd.Series(dtype=float)
        rows.append({
            "region": region,
            "cap": cap,
            "cap_label": cap_label(float(cap)),
            "method": method,
            "n_days": int(group["day_index"].nunique()),
            "k": int(group["k"].iloc[0]),
            "mean_miss_count": float(group["miss_count"].mean()),
            "mean_miss_rate": float(group["miss_rate"].mean()),
            "days_with_any_miss_pct": 100.0 * float((group["miss_count"] > 0).mean()),
            "mean_replacement_gap_all_days": float(group["replacement_gap"].mean()),
            "mean_replacement_gap_miss_days": float(positive_miss["replacement_gap"].mean()) if not positive_miss.empty else 0.0,
            "replacement_gap_p25_miss_days": float(penalty_q.loc[0.25]) if not penalty_q.empty else 0.0,
            "replacement_gap_p50_miss_days": float(penalty_q.loc[0.5]) if not penalty_q.empty else 0.0,
            "replacement_gap_p75_miss_days": float(penalty_q.loc[0.75]) if not penalty_q.empty else 0.0,
            "replacement_gap_p90_miss_days": float(penalty_q.loc[0.9]) if not penalty_q.empty else 0.0,
            "tail_excess_ci_mean": float(group["tail_excess_ci"].mean()),
        })
    return pd.DataFrame(rows)


def paired_tests(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baselines = ["Climatology", "24h persistence"]
    methods = ["Direct MLP", "Fuel-mix ens."]
    for (region, cap), group in daily.groupby(["region", "cap"]):
        pivot = group.pivot_table(
            index="day_index",
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


def plot_capture_boxplots(daily: pd.DataFrame, out_dir: Path) -> None:
    for cap_label_value, cap_df in daily[daily["method"].isin(PLOT_METHODS)].groupby("cap_label"):
        regions = list(daily["region"].drop_duplicates())
        methods = PLOT_METHODS
        fig, axes = plt.subplots(1, len(regions), figsize=(13.2, 3.2), sharey=True)
        if len(regions) == 1:
            axes = [axes]
        for ax, region in zip(axes, regions):
            values = [
                cap_df[(cap_df["region"] == region) & (cap_df["method"] == method)]["capture"].dropna().to_numpy()
                for method in methods
            ]
            ax.boxplot(values, tick_labels=["Clim.", "Persist", "MLP", "Fuel"], showfliers=False)
            ax.axhline(0.0, color="0.7", linewidth=0.8)
            ax.axhline(1.0, color="0.7", linewidth=0.8, linestyle="--")
            ax.set_title(region)
            ax.tick_params(axis="x", labelrotation=35)
        axes[0].set_ylabel("Capture ratio")
        fig.suptitle(f"Daily carbon-aware scheduling capture, cap={cap_label_value.replace('over', '/')}")
        fig.tight_layout()
        fig.savefig(out_dir / f"decision_capture_boxplot_cap_{cap_label_value}.pdf")
        fig.savefig(out_dir / f"decision_capture_boxplot_cap_{cap_label_value}.png", dpi=200)
        plt.close(fig)


def plot_replacement_gap_boxplots(tail: pd.DataFrame, out_dir: Path) -> None:
    plot_df = tail[
        (tail["region"] == "ERCOT")
        & (tail["cap_label"] == "1over6")
        & (tail["method"].isin(PLOT_METHODS))
        & (tail["miss_count"] > 0)
    ]
    if plot_df.empty:
        return
    values = [
        plot_df[plot_df["method"] == method]["replacement_gap"].dropna().to_numpy()
        for method in PLOT_METHODS
    ]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.boxplot(values, tick_labels=["Clim.", "Persist", "MLP", "Fuel"], showfliers=False)
    ax.axhline(0.0, color="0.7", linewidth=0.8)
    ax.set_ylabel("Replacement gap (tCO2/MWh)")
    ax.set_title("ERCOT bottom-6 miss penalty, cap=1/6")
    ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    fig.savefig(out_dir / "ercot_replacement_gap_boxplot_cap_1over6.pdf")
    fig.savefig(out_dir / "ercot_replacement_gap_boxplot_cap_1over6.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regions",
        nargs="+",
        default=REGIONS,
        help="Region slugs with configs/dayahead/{region}_fuelmix_dayahead.json.",
    )
    parser.add_argument(
        "--caps",
        nargs="+",
        type=parse_cap,
        default=[1 / 3, 1 / 6, 1 / 12],
        help="Hourly load caps. Use decimals, fractions like 1/6, or hours like 6h.",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    daily_rows: list[dict] = []
    summary_rows: list[dict] = []
    tail_rows: list[dict] = []
    for region in args.regions:
        region_daily, region_summary, region_tail = evaluate_region(region, args.caps)
        daily_rows.extend(region_daily)
        summary_rows.extend(region_summary)
        tail_rows.extend(region_tail)

    daily = pd.DataFrame(daily_rows)
    summary = pd.DataFrame(summary_rows)
    tail = pd.DataFrame(tail_rows)
    tail_summary = summarize_tail(tail)
    miss_penalty = summarize_miss_penalty(tail)
    tests = paired_tests(daily)

    daily.to_csv(args.out_dir / "decision_daily_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "decision_summary_by_method.csv", index=False)
    tail.to_csv(args.out_dir / "decision_tail_decomposition_daily.csv", index=False)
    tail_summary.to_csv(args.out_dir / "decision_tail_decomposition_summary.csv", index=False)
    miss_penalty.to_csv(args.out_dir / "decision_miss_penalty_summary.csv", index=False)
    tests.to_csv(args.out_dir / "decision_pairwise_wilcoxon.csv", index=False)
    plot_capture_boxplots(daily, args.out_dir)
    plot_replacement_gap_boxplots(tail, args.out_dir)

    focus = summary[
        (summary["cap_label"] == "1over6")
        & (summary["method"].isin(["Climatology", "24h persistence", "Direct MLP", "Fuel-mix ens."]))
    ][[
        "region",
        "method",
        "mean_capture",
        "median_capture",
        "mean_spearman",
        "mean_topk_overlap",
        "mean_oracle_saving_pct",
    ]]
    print(focus.round(4).to_string(index=False))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
