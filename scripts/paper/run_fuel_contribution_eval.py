#!/usr/bin/env python3
"""Check whether predicted fuel contributions track realized ones (day-ahead-like).

Retrains the locked three-seed day-ahead-like Fuel-mix models (same configs,
seeds, and fixed 70/15/15 split as the point-forecast table) and compares the
predicted contribution eta_f * m_hat_f / L_hat with the realized contribution
eta_f * m_f / L_hat on every test hour. The baseline is 24-hour fuel
persistence, eta_f * m_f(t-24) / L_hat(t-24), whose fuel sum is the 24-hour
CI persistence forecast. Contributions summing to the predicted proxy only
show that the decomposition is consistent; agreement with realized
contributions is the evidence of interest. Neither is a causal attribution.

Spike composition compares mean contributions in the top-decile realized-proxy
test hours with the remaining test hours. The ERCOT case is the 72-hour test
window with the largest realized-proxy range, chosen from realized data only;
it is illustrative. ERCOT variants with wind/solar forecasts are included
when their configs exist.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from fuel_mix_trainer import build_supervised_table,load_config,run_experiment
REGIONS={'isone':'ISO-NE','nyiso':'NYISO','pjm':'PJM','ercot':'ERCOT',
         'caiso':'CAISO','miso':'MISO','ieso':'IESO',
         'ercot_rnw0':'ERCOT+WS latest','ercot_rnw24':'ERCOT+WS 24h'}
WINDOW_HOURS=72
CASE_TAGS={'ercot':'fuelmix','ercot_rnw0':'fuelmix_ws_latest','ercot_rnw24':'fuelmix_ws24'}


def fit_or_load(region,seed,cache):
    path=cache/region/f'seed_{seed}'/'test_fuel_predictions.csv'
    if not path.exists():
        cfg=load_config(ROOT/f'configs/dayahead/{region}_fuelmix_dayahead.json')
        cfg['output']['save_dir']=str(path.parent)
        run_experiment(cfg,device='cpu',make_plots=False,seed=seed)
    frame=pd.read_csv(path)
    frame['timestamp']=pd.to_datetime(frame['timestamp'],utc=True)
    return frame


def scores(pred,true):
    err=pred-true;sst=np.sum((true-true.mean())**2)
    return dict(mae=np.mean(np.abs(err)),rmse=np.sqrt(np.mean(err**2)),
                corr=np.corrcoef(pred,true)[0,1] if np.std(pred)>0 else np.nan,
                r2=1-np.sum(err**2)/sst if sst>0 else np.nan)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache-dir',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,default=ROOT/'results/attribution')
    ap.add_argument('--seeds',default='7,42,123')
    args=ap.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    seeds=[int(s) for s in args.seeds.split(',')]
    args.out_dir.mkdir(parents=True,exist_ok=True)
    skill=[];spikes=[];case=None
    for region,label in REGIONS.items():
        if not (ROOT/f'configs/dayahead/{region}_fuelmix_dayahead.json').exists():continue
        cfg=load_config(ROOT/f'configs/dayahead/{region}_fuelmix_dayahead.json')
        ef=cfg['emission_factors'];fuels=[f for f in cfg['dataset']['fuel_cols'] if ef.get(f,0)>0]
        runs=[fit_or_load(region,s,args.cache_dir) for s in seeds]
        base=runs[0][['timestamp','demand_true']]
        assert all(r['timestamp'].equals(base['timestamp']) for r in runs)
        load=base['demand_true'].to_numpy(float)
        # Realized contributions over the full table supply the t-24 persistence values.
        data=build_supervised_table(cfg).set_index('timestamp')
        lag=base['timestamp']-pd.Timedelta(hours=24)
        real={f:runs[0][f'true_{f}'].to_numpy(float)*ef[f]/load for f in fuels}
        persist={f:(data.loc[lag,f]*ef[f]/data.loc[lag,'system_load_forecast']).to_numpy(float) for f in fuels}
        seed_pred=[{f:r[f'pred_{f}'].to_numpy(float)*ef[f]/load for f in fuels} for r in runs]
        ens={f:np.mean([p[f] for p in seed_pred],axis=0) for f in fuels}
        ci=sum(real.values());ci_ens=sum(ens.values());ci_persist=sum(persist.values())
        top=ci>=np.quantile(ci,.9)
        for f in fuels:
            share=real[f].mean()/ci.mean()
            for method,pred in [('Fuel-mix ens.',ens[f]),('24h fuel persistence',persist[f])]:
                skill.append(dict(region=label,fuel=f,method=method,n_test=len(ci),
                    mean_realized_share=share,**scores(pred,real[f])))
            seed_scores=pd.DataFrame([scores(p[f],real[f]) for p in seed_pred]).mean()
            skill.append(dict(region=label,fuel=f,method='Fuel-mix seed mean',n_test=len(ci),
                mean_realized_share=share,**seed_scores.to_dict()))
            for method,pred in [('Realized',real[f]),('Fuel-mix ens.',ens[f]),('24h fuel persistence',persist[f])]:
                spikes.append(dict(region=label,fuel=f,method=method,n_spike_hours=int(top.sum()),
                    spike_minus_other=pred[top].mean()-pred[~top].mean()))
        if region in CASE_TAGS:
            stamp=base['timestamp']
            if case is None:
                # Largest realized range over contiguous 72-hour test windows.
                ranges=pd.Series(ci).rolling(WINDOW_HOURS).agg(lambda w:w.max()-w.min())
                end=int(ranges.idxmax());sel=np.zeros(len(ci),bool);sel[end-WINDOW_HOURS+1:end+1]=True
                case=pd.DataFrame({'timestamp':stamp[sel].dt.strftime('%Y-%m-%d %H:%M:%S%z').values,
                    'ci_realized':ci[sel],'ci_persistence':ci_persist[sel]})
                for f in fuels:
                    case[f'{f}_realized']=real[f][sel];case[f'{f}_persistence']=persist[f][sel]
                case_stamps=stamp[sel].values
            assert (stamp[sel].values==case_stamps).all()
            tag=CASE_TAGS[region];case[f'ci_{tag}']=ci_ens[sel]
            for f in fuels:case[f'{f}_{tag}']=ens[f][sel]
    if case is not None:case.to_csv(args.out_dir/'ercot_dayahead_case_72h.csv',index=False)
    skill=pd.DataFrame(skill);spikes=pd.DataFrame(spikes)
    total=spikes.groupby(['region','method']).spike_minus_other.transform('sum')
    spikes['share_of_spike_rise']=spikes.spike_minus_other/total
    real_rise=spikes[spikes.method=='Realized'].set_index(['region','fuel']).spike_minus_other
    spikes['fraction_of_realized_rise']=spikes.spike_minus_other.values/real_rise.loc[list(zip(spikes.region,spikes.fuel))].values
    skill.to_csv(args.out_dir/'fuel_contribution_skill.csv',index=False)
    spikes.to_csv(args.out_dir/'fuel_contribution_spikes.csv',index=False)
    protocol=dict(setting='day-ahead-like fixed 70/15/15 split',seeds=seeds,
        contribution='eta_f * fuel_MW / latest load forecast',baseline='24h fuel persistence',
        spike_hours='top decile of realized proxy in the test split',
        ercot_window_hours=WINDOW_HOURS,
        window_selection='largest realized-proxy range among contiguous 72-hour ERCOT test windows; illustrative only')
    (args.out_dir/'fuel_contribution_protocol.json').write_text(json.dumps(protocol,indent=2))
    pd.set_option('display.width',200)
    print(skill.round(4).to_string(index=False));print(spikes.round(4).to_string(index=False))


if __name__=='__main__':main()
