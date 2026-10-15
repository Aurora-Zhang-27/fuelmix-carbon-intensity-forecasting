#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
DEVICE="${DEVICE:-cpu}"
SEEDS_CSV="${SEEDS:-7,42,123}"
REGIONS=("$@")
if [[ ${#REGIONS[@]} -eq 0 ]]; then
    REGIONS=(isone nyiso pjm ercot caiso miso ieso)
fi

IFS=',' read -r -a SEED_LIST <<< "$SEEDS_CSV"
OUT_DIR="results/fuelmix_dayahead/multiseed"
PRED_DIR="$OUT_DIR/predictions"
mkdir -p "$OUT_DIR" "$PRED_DIR"

for seed in "${SEED_LIST[@]}"; do
    seed="$(echo "$seed" | xargs)"
    echo "=== day-ahead seed ${seed} ==="
    for region in "${REGIONS[@]}"; do
        echo ">>> train day-ahead ${region}"
        "$PYTHON_BIN" src/run_fuelmix.py \
            --config "configs/dayahead/${region}_fuelmix_dayahead.json" \
            --stage train \
            --seed "$seed" \
            --device "$DEVICE"
    done

    REGIONS_CSV="$(IFS=,; echo "${REGIONS[*]}")"
    export REGIONS_CSV
    "$PYTHON_BIN" - <<'PY'
import os
import pandas as pd

rows = []
for region in [r for r in os.environ["REGIONS_CSV"].split(",") if r]:
    m = pd.read_csv(f"results/fuelmix_dayahead/{region}/test_metrics.csv").iloc[0]
    p = pd.read_csv(f"results/fuelmix_dayahead/{region}/test_predictions.csv")
    mean_ci = p["ci_true"].mean()
    mean_dem = p["demand_true"].mean()
    rows.append({
        "region": region.upper(),
        "mean_ci": round(mean_ci, 4),
        "ci_mae": round(m["ci_mae_t_per_mwh"], 4),
        "ci_rmse": round(m["ci_rmse_t_per_mwh"], 4),
        "rel_rmse_pct": round(m["ci_rmse_t_per_mwh"] / mean_ci * 100, 1),
        "ci_corr": round(m.get("ci_corr", float("nan")), 4),
        "ci_r2": round(m.get("ci_r2", float("nan")), 4),
        "ci_std_ratio": round(m.get("ci_std_ratio", float("nan")), 4),
        "ci_ramp_corr": round(m.get("ci_ramp_corr", float("nan")), 4),
        "direction_acc_pct": round(m.get("ci_direction_acc_pct", float("nan")), 1),
        "balance_mae_mw": round(m["balance_mae_mw"], 0),
        "balance_pct": round(m["balance_mae_mw"] / mean_dem * 100, 1),
    })

df = pd.DataFrame(rows)
df.to_csv("results/fuelmix_dayahead/benchmark_all_regions.csv", index=False)
print(df.to_string(index=False))
PY

    cp "results/fuelmix_dayahead/benchmark_all_regions.csv" \
        "$OUT_DIR/benchmark_seed_${seed}.csv"
    for region in "${REGIONS[@]}"; do
        mkdir -p "$PRED_DIR/seed_${seed}"
        cp "results/fuelmix_dayahead/${region}/test_predictions.csv" \
            "$PRED_DIR/seed_${seed}/${region}_test_predictions.csv"
    done
done

REGIONS_CSV="$(IFS=,; echo "${REGIONS[*]}")"
export REGIONS_CSV OUT_DIR PRED_DIR SEEDS_CSV
"$PYTHON_BIN" - <<'PY'
import os
import numpy as np
import pandas as pd

out_dir = os.environ["OUT_DIR"]
pred_dir = os.environ["PRED_DIR"]
seeds = [s.strip() for s in os.environ["SEEDS_CSV"].split(",") if s.strip()]
regions = [r for r in os.environ["REGIONS_CSV"].split(",") if r]

frames = []
for seed in seeds:
    df = pd.read_csv(f"{out_dir}/benchmark_seed_{seed}.csv")
    df["seed"] = int(seed)
    frames.append(df)
all_runs = pd.concat(frames, ignore_index=True)
all_runs.to_csv(f"{out_dir}/benchmark_multiseed_runs.csv", index=False)

metric_cols = [
    "mean_ci", "ci_mae", "ci_rmse", "rel_rmse_pct", "ci_corr", "ci_r2",
    "ci_std_ratio", "ci_ramp_corr", "direction_acc_pct",
    "balance_mae_mw", "balance_pct",
]
summary = all_runs.groupby("region")[metric_cols].agg(["mean", "std"])
summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
summary = summary.reset_index()
summary.to_csv(f"{out_dir}/benchmark_multiseed_summary.csv", index=False)

def trend_metrics(ci_pred, ci_true):
    err = ci_pred - ci_true
    true_std = float(np.std(ci_true))
    pred_std = float(np.std(ci_pred))
    corr = (
        float(np.corrcoef(ci_true, ci_pred)[0, 1])
        if true_std > 0 and pred_std > 0 else np.nan
    )
    sse = float(np.sum(err ** 2))
    sst = float(np.sum((ci_true - np.mean(ci_true)) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else np.nan
    std_ratio = float(pred_std / true_std) if true_std > 0 else np.nan
    true_ramp = np.diff(ci_true)
    pred_ramp = np.diff(ci_pred)
    ramp_corr = (
        float(np.corrcoef(true_ramp, pred_ramp)[0, 1])
        if len(true_ramp) and np.std(true_ramp) > 0 and np.std(pred_ramp) > 0
        else np.nan
    )
    direction_acc = (
        float(np.mean(np.sign(true_ramp) == np.sign(pred_ramp)) * 100)
        if len(true_ramp) else np.nan
    )
    return corr, r2, std_ratio, ramp_corr, direction_acc

ensemble_rows = []
for region in regions:
    pred_frames = [
        pd.read_csv(f"{pred_dir}/seed_{seed}/{region}_test_predictions.csv")
        for seed in seeds
    ]
    base = pred_frames[0][["timestamp", "ci_true", "demand_true"]].copy()
    ci_pred = np.mean([p["ci_pred"].to_numpy(dtype=float) for p in pred_frames], axis=0)
    total_gen_pred = np.mean([
        p["total_gen_pred"].to_numpy(dtype=float) for p in pred_frames
    ], axis=0)

    ci_true = base["ci_true"].to_numpy(dtype=float)
    demand = base["demand_true"].to_numpy(dtype=float)
    err = ci_pred - ci_true
    mean_ci = float(np.mean(ci_true))
    ci_mae = float(np.mean(np.abs(err)))
    ci_rmse = float(np.sqrt(np.mean(err ** 2)))
    balance_mae = float(np.mean(np.abs(total_gen_pred - demand)))
    corr, r2, std_ratio, ramp_corr, direction_acc = trend_metrics(ci_pred, ci_true)

    ensemble_rows.append({
        "region": region.upper(),
        "mean_ci": round(mean_ci, 4),
        "ci_mae": round(ci_mae, 4),
        "ci_rmse": round(ci_rmse, 4),
        "rel_rmse_pct": round(ci_rmse / max(mean_ci, 1e-9) * 100, 1),
        "ci_corr": round(corr, 4),
        "ci_r2": round(r2, 4),
        "ci_std_ratio": round(std_ratio, 4),
        "ci_ramp_corr": round(ramp_corr, 4),
        "direction_acc_pct": round(direction_acc, 1),
        "balance_mae_mw": round(balance_mae, 0),
        "balance_pct": round(balance_mae / max(float(np.mean(demand)), 1e-9) * 100, 1),
        "seeds": ",".join(seeds),
    })

    out = base.copy()
    out["ci_pred"] = ci_pred
    out["total_gen_pred"] = total_gen_pred
    out.to_csv(f"{out_dir}/{region}_ensemble_predictions.csv", index=False)

ensemble = pd.DataFrame(ensemble_rows)
ensemble.to_csv(f"{out_dir}/benchmark_ensemble.csv", index=False)

print(summary.round(4).to_string(index=False))
print("\nDay-ahead ensemble:")
print(ensemble.to_string(index=False))
print(f"\nwrote {out_dir}/benchmark_ensemble.csv")
PY
