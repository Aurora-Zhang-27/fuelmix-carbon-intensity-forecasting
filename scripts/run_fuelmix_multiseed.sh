#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
DEVICE="${DEVICE:-cpu}"
STAGE="${STAGE:-train}"
SEEDS_CSV="${SEEDS:-7,42,123}"
REGIONS=("$@")
if [[ ${#REGIONS[@]} -eq 0 ]]; then
    REGIONS=(isone nyiso pjm ercot caiso miso ieso)
fi
ALL_REGIONS=(isone nyiso pjm ercot caiso miso ieso)
BENCHMARK_SUFFIX="selected_regions"
if [[ "${REGIONS[*]}" == "${ALL_REGIONS[*]}" ]]; then
    BENCHMARK_SUFFIX="all_regions"
fi

IFS=',' read -r -a SEED_LIST <<< "$SEEDS_CSV"
OUT_DIR="results/fuelmix/multiseed"
PRED_DIR="$OUT_DIR/predictions"
mkdir -p "$OUT_DIR"
mkdir -p "$PRED_DIR"

INNER_STAGE="$STAGE"
if [[ "$STAGE" == "all" ]]; then
    echo "=== shared fetch before multiseed training ==="
    STAGE=fetch DEVICE="$DEVICE" PYTHON_BIN="$PYTHON_BIN" \
        ./scripts/run_fuelmix_benchmark.sh "${REGIONS[@]}"
    INNER_STAGE="train"
elif [[ "$STAGE" == "fetch" ]]; then
    echo "=== fetch only ==="
    STAGE=fetch DEVICE="$DEVICE" PYTHON_BIN="$PYTHON_BIN" \
        ./scripts/run_fuelmix_benchmark.sh "${REGIONS[@]}"
    echo "fetch complete; rerun with STAGE=train to train multiseed models"
    exit 0
fi

for seed in "${SEED_LIST[@]}"; do
    seed="$(echo "$seed" | xargs)"
    echo "=== seed ${seed} ==="
    SEED="$seed" DEVICE="$DEVICE" STAGE="$INNER_STAGE" FORCE_REFRESH=0 \
        PYTHON_BIN="$PYTHON_BIN" ./scripts/run_fuelmix_benchmark.sh "${REGIONS[@]}"
    cp "results/fuelmix/benchmark_${BENCHMARK_SUFFIX}.csv" \
        "$OUT_DIR/benchmark_seed_${seed}.csv"
    for region in "${REGIONS[@]}"; do
        mkdir -p "$PRED_DIR/seed_${seed}"
        cp "results/fuelmix/${region}/test_predictions.csv" \
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
summary = (
    all_runs.groupby("region")[metric_cols]
    .agg(["mean", "std"])
)
summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
summary = summary.reset_index()
summary.to_csv(f"{out_dir}/benchmark_multiseed_summary.csv", index=False)

def trend_metrics(ci_pred, ci_true):
    ci_pred = np.asarray(ci_pred, dtype=float)
    ci_true = np.asarray(ci_true, dtype=float)
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
    pred_frames = []
    for seed in seeds:
        p = pd.read_csv(f"{pred_dir}/seed_{seed}/{region}_test_predictions.csv")
        pred_frames.append(p)

    base = pred_frames[0][["timestamp", "ci_true", "demand_true"]].copy()
    ci_pred = np.mean([p["ci_pred"].to_numpy(dtype=float) for p in pred_frames], axis=0)
    if all("total_gen_pred" in p.columns for p in pred_frames):
        total_gen_pred = np.mean([
            p["total_gen_pred"].to_numpy(dtype=float) for p in pred_frames
        ], axis=0)
    else:
        total_gen_pred = None

    ci_true = base["ci_true"].to_numpy(dtype=float)
    demand = base["demand_true"].to_numpy(dtype=float)
    err = ci_pred - ci_true
    mean_ci = float(np.mean(ci_true))
    ci_mae = float(np.mean(np.abs(err)))
    ci_rmse = float(np.sqrt(np.mean(err ** 2)))
    corr, r2, std_ratio, ramp_corr, direction_acc = trend_metrics(ci_pred, ci_true)
    if total_gen_pred is not None:
        balance_mae = float(np.mean(np.abs(total_gen_pred - demand)))
    else:
        # Backward compatibility for old predictions; fresh runs include this.
        balance_mae = np.nan

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
        "balance_mae_mw": round(balance_mae, 0) if np.isfinite(balance_mae) else np.nan,
        "balance_pct": (
            round(balance_mae / max(float(np.mean(demand)), 1e-9) * 100, 1)
            if np.isfinite(balance_mae) else np.nan
        ),
        "seeds": ",".join(seeds),
    })

    out = base.copy()
    out["ci_pred"] = ci_pred
    if total_gen_pred is not None:
        out["total_gen_pred"] = total_gen_pred
    out.to_csv(f"{out_dir}/{region}_ensemble_predictions.csv", index=False)

ensemble = pd.DataFrame(ensemble_rows)
ensemble.to_csv(f"{out_dir}/benchmark_ensemble.csv", index=False)

print(summary.round(4).to_string(index=False))
print("\nEnsemble:")
print(ensemble.to_string(index=False))
print(f"\nwrote {out_dir}/benchmark_multiseed_runs.csv")
print(f"wrote {out_dir}/benchmark_multiseed_summary.csv")
print(f"wrote {out_dir}/benchmark_ensemble.csv")
PY

echo
echo "note: results/fuelmix/{region}/ now contains the final seed from SEEDS=${SEEDS_CSV}."
echo "rerun SEED=42 DEVICE=${DEVICE} ./scripts/run_fuelmix_benchmark.sh ${REGIONS[*]} to restore the paper seed outputs if needed."
echo "ensemble outputs are in ${OUT_DIR}/benchmark_ensemble.csv."
