#!/usr/bin/env python3
"""Assemble the paper's compact tables from per-region script outputs.

Each step is skipped, with a message, when its inputs are absent:
1. Concatenate the per-region outputs of run_rolling_decision_eval.py
   (results/experiments/decision_eval_rolling_<region>/) into
   results/rolling_rmse_aligned/seven_regions_*.csv.
2. Copy make_horizon_comparison_tables.py's day-ahead table to
   results/aligned_protocol/dayahead_baselines_aligned.csv.
3. Refresh results/summary/ from the retained sources.
"""
from __future__ import annotations
import argparse,shutil
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
REGIONS=['isone','nyiso','pjm','ercot','caiso','miso','ieso']
ROLLING={'rolling_decision_summary':'seven_regions_rolling_decision_summary_aligned.csv',
         'rolling_pairwise_wilcoxon':'seven_regions_rolling_pairwise_wilcoxon_aligned.csv',
         'rolling_protocol_audit':'seven_regions_rolling_protocol_audit.csv',
         'rolling_rmse_by_window':'seven_regions_rolling_rmse_by_window_aligned.csv',
         'rolling_rmse_pooled':'seven_regions_rolling_rmse_pooled_aligned.csv'}
SCALAR=['Climatology','24h persistence','Direct MLP','Fuel-mix ens.']
# file, classification, source, description; sources without a CSV are kept as retained.
SUMMARY=[
    ('protocol_aligned_dayahead_rmse.csv','protocol_aligned_primary','results/aligned_protocol/dayahead_baselines_aligned.csv','Protocol-aligned seven-region day-ahead-like RMSE comparison.'),
    ('protocol_aligned_dayahead_method_metrics.csv','protocol_aligned_primary','results/experiments/method_metrics/dayahead_method_metrics.csv','Protocol-aligned metrics by method.'),
    ('protocol_aligned_rolling_rmse_pooled.csv','protocol_aligned_primary','results/rolling_rmse_aligned/seven_regions_rolling_rmse_pooled_aligned.csv','Protocol-aligned rolling-origin pooled RMSE.'),
    ('protocol_aligned_rolling_decision_capture.csv','protocol_aligned_primary','results/rolling_rmse_aligned/seven_regions_rolling_decision_summary_aligned.csv','Protocol-aligned rolling-origin decision results at cap=1/6 (three seeds, prediction-error early stopping).'),
    ('ci_definition_check.csv','protocol_aligned_primary','scripts/paper/make_ci_definition_check.py','Load-normalized proxy versus generation-normalized intensity; written directly by the script.'),
    ('fuel_contribution_skill.csv','protocol_aligned_primary','results/attribution/fuel_contribution_skill.csv','Day-ahead-like predicted versus realized fuel contributions with 24h fuel persistence, fixed-split test period.'),
    ('fuel_contribution_spikes.csv','protocol_aligned_primary','results/attribution/fuel_contribution_spikes.csv','Fuel composition of the rise in top-decile realized-proxy test hours.'),
    ('prediction_intervals_robustness.csv','robustness_primary','results/robustness_intervals/metrics.csv','Independent calibration and matched chronological OOF interval ablation; raw and calibrated results.'),
    ('ercot_seed_robustness.csv','robustness_primary','results/robustness_seeds/ensemble_summary.csv','Fixed-lambda ERCOT three- and ten-seed ensembles; individual-seed and paired-block diagnostics in robustness_seeds.'),
    ('posthoc_caiso_signed_flow.csv','posthoc_diagnostic','retained single-year (2024) result','CAISO signed-flow correction and validation-selected loss weights on 2024 data.')]


def collect_rolling(rolling_root,suffix='',out=ROOT/'results/rolling_rmse_aligned',regions=REGIONS):
    dirs=[rolling_root/f'decision_eval_rolling_{r}{suffix}' for r in regions]
    if not all(d.is_dir() for d in dirs):
        print(f'skip rolling{suffix}: per-region outputs not all present under {rolling_root}');return
    out.mkdir(parents=True,exist_ok=True)
    for stem,name in ROLLING.items():
        frame=pd.concat([pd.read_csv(d/f'{r}_{stem}.csv') for r,d in zip(regions,dirs)],ignore_index=True)
        frame.to_csv(out/name,index=False);print('wrote',out/name)


def collect_dayahead():
    src=ROOT/'results/experiments/horizon_comparison/dayahead_baselines.csv'
    if not src.exists():print(f'skip day-ahead table: {src} not present');return
    dst=ROOT/'results/aligned_protocol/dayahead_baselines_aligned.csv';dst.parent.mkdir(parents=True,exist_ok=True)
    pd.read_csv(src).to_csv(dst,index=False);print('wrote',dst)


def refresh_summary():
    for name,_,source,_ in SUMMARY:
        src=ROOT/source;dst=ROOT/'results/summary'/name
        if name=='protocol_aligned_rolling_decision_capture.csv':
            frame=pd.read_csv(src);frame[frame.method.isin(SCALAR)].to_csv(dst,index=False)
        elif src.suffix=='.csv':shutil.copyfile(src,dst)
    print(f'refreshed results/summary/ ({len(SUMMARY)} tables)')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rolling-root',type=Path,default=ROOT/'results/experiments')
    args=ap.parse_args()
    collect_rolling(args.rolling_root)
    collect_rolling(args.rolling_root,'_2024w',ROOT/'results/rolling_2024_windows')
    collect_rolling(args.rolling_root,'',ROOT/'results/rolling_ercot_ws',['ercot','ercot_rnw0','ercot_rnw24'])
    collect_dayahead();refresh_summary()


if __name__=='__main__':main()
