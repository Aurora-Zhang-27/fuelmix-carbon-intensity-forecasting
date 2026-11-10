#!/usr/bin/env python3
"""One-at-a-time loss-weight sensitivity on the fixed day-ahead split.

Each region is refitted with one weight changed at a time (CI 25 or 100,
balance 0 or 1, fossil 0 or 1) and seeds 7, 42, 123. The weights used in the
paper are not selected from this; it shows how much the validation RMSE moves.

Fit one region:  run_loss_sensitivity.py --cache-dir DIR --region ercot
Summarize:       run_loss_sensitivity.py --cache-dir DIR --summarize
"""
from __future__ import annotations
import argparse,copy,json,sys
from pathlib import Path
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from fuel_mix_trainer import load_config,run_experiment  # noqa: E402
VARIANTS={'base':{},'ci_25':{'ci':25.0},'ci_100':{'ci':100.0},'bal_0':{'balance':0.0},
          'bal_1':{'balance':1.0},'fossil_0':{'fossil':0.0},'fossil_1':{'fossil':1.0}}
SEEDS=(7,42,123)


def fit(region,cache):
    base=load_config(ROOT/f'configs/dayahead/{region}_fuelmix_dayahead.json');rows=[]
    for name,upd in VARIANTS.items():
        for seed in SEEDS:
            cfg=copy.deepcopy(base);cfg['loss_weights'].update(upd)
            cfg['output']['save_dir']=str(cache/'runs'/region/name/f'seed_{seed}')
            r=run_experiment(cfg,device='cpu',make_plots=False,seed=seed)
            rows.append(dict(region=region,variant=name,seed=seed,val_rmse=r['val_metrics']['ci_rmse_t_per_mwh'],
                             test_rmse=r['test_metrics']['ci_rmse_t_per_mwh']))
    (cache/f'{region}.json').write_text(json.dumps(rows,indent=1))


def summarize(cache):
    d=pd.concat([pd.DataFrame(json.loads(f.read_text())) for f in sorted(cache.glob('*.json'))])
    m=d.groupby(['region','variant'])[['val_rmse','test_rmse']].mean().reset_index()
    b=m[m.variant=='base'].set_index('region')
    for c in ['val_rmse','test_rmse']:m[c+'_ratio_to_base']=m[c]/m.region.map(b[c])
    out=ROOT/'results/summary/loss_weight_sensitivity.csv';m.to_csv(out,index=False);print('wrote',out)


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache-dir',type=Path,required=True)
    ap.add_argument('--region')
    ap.add_argument('--summarize',action='store_true')
    args=ap.parse_args();args.cache_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    if args.region:fit(args.region,args.cache_dir)
    if args.summarize:summarize(args.cache_dir)


if __name__=='__main__':main()
