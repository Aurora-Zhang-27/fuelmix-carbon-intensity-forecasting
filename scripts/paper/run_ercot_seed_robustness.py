#!/usr/bin/env python3
"""Fixed-lambda ERCOT initialization audit, distinct from ensemble-size effects.

Ten seeds and three historical windows are fixed in advance. Lambda=10 is the
previous validation-selected weight, not selected again using test outcomes.
Report individual seeds and the original-three/all-ten prediction ensembles.
Confidence intervals resample 3/7/14-day circular blocks within each window;
seeds are never treated as additional observed calendar days.
"""
from __future__ import annotations
import argparse,json,sys
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT/'src')]
from run_ercot_spoplus_2x2 import (load_config,build_supervised_table,feature_columns_from_table,
    parse_starts,window_splits,fit_direct_model,fit_fuelmix_model,decision_metrics,
    build_daily_rows,true_ci)
SEEDS=[7,42,123,202,303,404,505,606,707,808]
METHODS={'Direct MSE':('direct',False),'Fuel-mix MSE':('fuelmix',False),
         'Direct SPO+':('direct',True),'Fuel-mix SPO+':('fuelmix',True)}
PAIRS=[('Direct SPO+','Fuel-mix MSE'),('Direct SPO+','Direct MSE'),
       ('Fuel-mix SPO+','Fuel-mix MSE')]+[(m,'24h persistence') for m in METHODS]


def block_interval(windows,block_length,draws=5000,seed=20261014):
    """Conditional CI: fixed observed windows, resampled serial blocks per window."""
    rng=np.random.default_rng(seed)
    boot=np.zeros(draws)
    total=sum(len(x) for x in windows)
    for x in windows:
        x=np.asarray(x,dtype=float)
        assert np.isfinite(x).all() and len(x)>0
        n=len(x);nblocks=int(np.ceil(n/block_length))
        starts=rng.integers(0,n,size=(draws,nblocks))
        idx=(starts[:,:,None]+np.arange(block_length))%n
        boot+=x[idx.reshape(draws,-1)[:,:n]].sum(axis=1)/total
    return np.quantile(boot,[.025,.975])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir',type=Path,default=ROOT/'results/robustness_seeds')
    ap.add_argument('--cache-dir',type=Path,required=True)
    ap.add_argument('--test-starts',default='2024-06-01,2024-08-01,2024-10-01')
    args=ap.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    args.out_dir.mkdir(parents=True,exist_ok=True);args.cache_dir.mkdir(parents=True,exist_ok=True)
    cfg=load_config(ROOT/'configs/dayahead/ercot_fuelmix_dayahead.json')
    data=build_supervised_table(cfg);features=feature_columns_from_table(data,cfg)
    starts=parse_starts(args.test_starts)
    splits=list(window_splits(data,starts,56,30))
    protocol=dict(seeds=SEEDS,windows=[str(x) for x in starts],test_days=56,selection_days=30,
        lambda_spo=10.,lambda_source='prior three-seed validation sweep; held fixed',
        max_epochs=40,patience=5,min_epochs=0,cap=1/6,threads=1,
        ensemble_sizes=[3,10],ensemble3_seeds=SEEDS[:3],bootstrap_draws=5000,
        bootstrap_blocks_days=[3,7,14],bootstrap_seed=20261014,
        numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__)
    pp=args.out_dir/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol,'Protocol changed; use another output directory'
    else:pp.write_text(json.dumps(protocol,indent=2))
    runs={};baselines={};bywindow=[];daily=[]
    for window,tr,va,te in splits:
        y=true_ci(te,cfg)
        history=pd.concat([tr,va],ignore_index=True)
        climatology=pd.DataFrame({'hour':history.hour.to_numpy(),'y':true_ci(history,cfg)}).groupby('hour').y.mean()
        baselines[window]={'24h persistence':te.historical_ci_24h_lag.to_numpy(dtype=float),
                           'Climatology':te.hour.map(climatology).to_numpy(dtype=float)}
        for seed in SEEDS:
            for method,(architecture,spo) in METHODS.items():
                key=(window,seed,method)
                path=args.cache_dir/(window+'_'+str(seed)+'_'+method.replace(' ','_').replace('+','plus')+'.npz')
                if path.exists():
                    with np.load(path) as z:run={k:z[k] for k in z.files}
                else:
                    print(window,'seed',seed,method,flush=True)
                    fn=fit_direct_model if architecture=='direct' else fit_fuelmix_model
                    run=fn(tr,va,te,features,cfg,seed,'cpu',1/6,10. if spo else 0.,spo,40,5,0)
                    assert np.isfinite(run['test_pred']).all()
                    np.savez_compressed(path,**run)
                assert np.array_equal(run['y_test'],y)
                runs[key]=run['test_pred']
                metrics=decision_metrics(run['test_pred'],y,1/6)
                bywindow.append(dict(window=window,seed=seed,method=method,**metrics))
                daily.extend(dict(seed=seed,**r) for r in build_daily_rows('ERCOT',window,method,run['test_pred'],y,1/6))
            print(window,'seed',seed,'DONE',flush=True)
        pd.DataFrame(bywindow).to_csv(args.out_dir/'by_window_seed.csv',index=False)
    seed_daily=pd.DataFrame(daily)
    seed_rows=[]
    for (seed,method),g in seed_daily.groupby(['seed','method'],sort=False):
        seed_rows.append(dict(seed=seed,method=method,n_days=len(g),mean_capture=g.capture.mean(),
                              mean_emission=g.decision_emission.mean()))
    per_seed=pd.DataFrame(seed_rows)
    per_seed.to_csv(args.out_dir/'per_seed.csv',index=False)
    per_seed.groupby('method').mean_capture.agg(['mean','std','min','max']).reset_index().to_csv(args.out_dir/'seed_summary.csv',index=False)
    seed_pair=[]
    pivot=per_seed.pivot(index='seed',columns='method',values='mean_capture')
    for method,baseline in PAIRS[:3]:
        for seed in SEEDS:seed_pair.append(dict(seed=seed,method=method,baseline=baseline,delta_capture=pivot.loc[seed,method]-pivot.loc[seed,baseline]))
    pd.DataFrame(seed_pair).to_csv(args.out_dir/'paired_seed_differences.csv',index=False)
    ensemble_daily=[]
    for size in [3,10]:
        for window,tr,va,te in splits:
            y=true_ci(te,cfg)
            preds={m:np.mean([runs[(window,s,m)] for s in SEEDS[:size]],axis=0) for m in METHODS}
            for method,pred in {**preds,**baselines[window]}.items():
                ensemble_daily.extend(dict(ensemble_size=size,**r) for r in build_daily_rows('ERCOT',window,method,pred,y,1/6))
    ens=pd.DataFrame(ensemble_daily)
    summary=ens.groupby(['ensemble_size','method'],sort=False).agg(mean_capture=('capture','mean'),
        mean_emission=('decision_emission','mean'),n_days=('capture','size')).reset_index()
    summary.to_csv(args.out_dir/'ensemble_summary.csv',index=False)
    intervals=[]
    for size in [3,10]:
        subset=ens[ens.ensemble_size==size]
        for method,baseline in PAIRS:
            c=subset.pivot(index=['window','day_index'],columns='method',values='capture')
            e=subset.pivot(index=['window','day_index'],columns='method',values='decision_emission')
            dc=c[method]-c[baseline];de=e[method]-e[baseline]
            for block in [3,7,14]:
                clo,chi=block_interval([g.to_numpy() for _,g in dc.groupby(level=0,sort=False)],block)
                elo,ehi=block_interval([g.to_numpy() for _,g in de.groupby(level=0,sort=False)],block)
                intervals.append(dict(ensemble_size=size,method=method,baseline=baseline,block_days=block,
                    delta_capture=dc.mean(),capture_ci_low=clo,capture_ci_high=chi,
                    delta_emission=de.mean(),emission_ci_low=elo,emission_ci_high=ehi,
                    wilcoxon_days_exploratory=wilcoxon(de.to_numpy()).pvalue))
    pd.DataFrame(intervals).to_csv(args.out_dir/'paired_block_intervals.csv',index=False)
    seed_daily.to_csv(args.cache_dir/'seed_daily.csv',index=False)
    ens.to_csv(args.cache_dir/'ensemble_daily.csv',index=False)
    print(summary.to_string(index=False),flush=True)
    print(f'DONE seed robustness: {40*len(starts)} fits; individual seeds and ensembles kept distinct.',flush=True)
if __name__=='__main__':main()
