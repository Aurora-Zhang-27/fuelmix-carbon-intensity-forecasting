#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SEED="${SEED:-42}"
DEVICE="${DEVICE:-}"
STAGE="${STAGE:-train}"
API_KEY="${GRIDSTATUS_API_KEY:-${API_KEY:-}}"
START_DATE="${START_DATE:-}"
END_DATE="${END_DATE:-}"
CHUNK_DAYS="${CHUNK_DAYS:-}"
SLEEP_SEC="${SLEEP_SEC:-}"
FORCE_REFRESH="${FORCE_REFRESH:-0}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
REGIONS=("$@")
if [[ ${#REGIONS[@]} -eq 0 ]]; then
    REGIONS=(isone nyiso pjm ercot caiso miso ieso)
fi

for region in "${REGIONS[@]}"; do
    echo ">>> ${STAGE} ${region}"
    cmd=("$PYTHON_BIN" src/run_fuelmix.py
        --config "configs/${region}_fuelmix.json"
        --stage "$STAGE"
        --seed "$SEED")
    if [[ -n "$DEVICE" ]]; then
        cmd+=(--device "$DEVICE")
    fi
    if [[ -n "$API_KEY" ]]; then
        cmd+=(--api_key "$API_KEY")
    fi
    if [[ -n "$START_DATE" ]]; then
        cmd+=(--start_date "$START_DATE")
    fi
    if [[ -n "$END_DATE" ]]; then
        cmd+=(--end_date "$END_DATE")
    fi
    if [[ -n "$CHUNK_DAYS" ]]; then
        cmd+=(--chunk_days "$CHUNK_DAYS")
    fi
    if [[ -n "$SLEEP_SEC" ]]; then
        cmd+=(--sleep_sec "$SLEEP_SEC")
    fi
    if [[ "$FORCE_REFRESH" == "1" || "$FORCE_REFRESH" == "true" ]]; then
        cmd+=(--force_refresh)
    fi
    "${cmd[@]}"
done

if [[ "$STAGE" == "fetch" ]]; then
    echo "fetch complete; rerun with STAGE=train or STAGE=all to train models"
    exit 0
fi

REGIONS_CSV="$(IFS=,; echo "${REGIONS[*]}")"
export REGIONS_CSV
"$PYTHON_BIN" - <<'PY'
import os
import pandas as pd

regions = [r for r in os.environ["REGIONS_CSV"].split(",") if r]
all_regions = ["isone", "nyiso", "pjm", "ercot", "caiso", "miso", "ieso"]
suffix = "all_regions" if regions == all_regions else "selected_regions"
rows = []
for region in regions:
    m = pd.read_csv(f"results/fuelmix/{region}/test_metrics.csv").iloc[0]
    p = pd.read_csv(f"results/fuelmix/{region}/test_predictions.csv")
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
print(df.to_string(index=False))
df.to_csv(f"results/fuelmix/benchmark_{suffix}.csv", index=False)
print(f"\nwrote results/fuelmix/benchmark_{suffix}.csv")
PY
