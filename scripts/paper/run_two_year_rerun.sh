#!/usr/bin/env bash
# Two-year (2024-2025) rerun following PROTOCOL_2024_2025.md.
# Every fit is single-threaded so retraining reproduces the tables exactly.
# Usage: CACHE=/tmp/carbon_two_year bash scripts/paper/run_two_year_rerun.sh [phase ...]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
PY="${PYTHON_BIN:-python3}"
CACHE="${CACHE:?set CACHE to a directory outside the package}"
REGIONS=(isone nyiso pjm ercot caiso miso ieso)
VARIANTS=(ercot_rnw0 ercot_rnw24)
W2025="2025-02-01,2025-04-01,2025-06-01,2025-08-01,2025-10-01"
W2024="2024-06-01,2024-08-01,2024-10-01"
export DAYAHEAD_EXTRA_REGIONS="ercot_rnw0,ercot_rnw24"
PHASES=("$@")
[[ ${#PHASES[@]} -eq 0 ]] && PHASES=(fixed rolling decision intervals contribution sensitivity tables)

for phase in "${PHASES[@]}"; do
  echo "=== phase: $phase ($(date '+%H:%M:%S'))"
  case "$phase" in
    fixed)
      PYTHON_BIN="$PY" SEEDS=7,42,123 DEVICE=cpu bash scripts/paper/run_dayahead_multiseed.sh "${REGIONS[@]}" "${VARIANTS[@]}"
      PYTHON_BIN="$PY" STAGE=train SEEDS=7,42,123 DEVICE=cpu bash scripts/run_fuelmix_multiseed.sh "${REGIONS[@]}"
      "$PY" scripts/paper/make_horizon_comparison_tables.py
      "$PY" scripts/paper/make_dayahead_method_metrics.py
      "$PY" scripts/paper/collect_paper_tables.py ;;
    rolling)
      for r in "${REGIONS[@]}" "${VARIANTS[@]}"; do
        "$PY" scripts/paper/run_rolling_decision_eval.py --region "$r" --seeds 7,42,123 --test-starts "$W2025"
      done
      for r in "${REGIONS[@]}"; do
        "$PY" scripts/paper/run_rolling_decision_eval.py --region "$r" --seeds 7,42,123 --test-starts "$W2024" \
          --out-dir "results/experiments/decision_eval_rolling_${r}_2024w"
      done ;;
    decision)
      "$PY" scripts/paper/run_ercot_spoplus_2x2.py --region ercot --seeds 7,42,123 --lambdas 10 --test-starts "$W2025"
      for r in "${REGIONS[@]}"; do
        "$PY" scripts/paper/run_ercot_spoplus_2x2.py --region "$r" --seeds 7,42,123 --lambdas 10 --test-starts "$W2025" \
          --out-dir "results/experiments/spoplus_2x2_all/$r"
      done
      "$PY" scripts/paper/run_ercot_seed_robustness.py --cache-dir "$CACHE/seeds" --test-starts "$W2025" ;;
    intervals)
      "$PY" scripts/paper/run_interval_robustness.py --cache-dir "$CACHE/intervals"
      "$PY" scripts/paper/run_interval_robustness.py --setting dayahead \
        --out-dir results/robustness_intervals_dayahead --cache-dir "$CACHE/intervals_dayahead" ;;
    contribution)
      "$PY" scripts/paper/run_fuel_contribution_eval.py --cache-dir "$CACHE/contribution"
      "$PY" scripts/paper/make_ci_definition_check.py
      "$PY" scripts/paper/make_region_table.py ;;
    sensitivity)
      for r in "${REGIONS[@]}"; do
        "$PY" scripts/paper/run_loss_sensitivity.py --cache-dir "$CACHE/loss_sensitivity" --region "$r"
      done
      "$PY" scripts/paper/run_loss_sensitivity.py --cache-dir "$CACHE/loss_sensitivity" --summarize ;;
    tables)
      "$PY" scripts/paper/collect_paper_tables.py ;;
    *) echo "unknown phase $phase"; exit 1 ;;
  esac
done
echo "=== done ($(date '+%H:%M:%S'))"
