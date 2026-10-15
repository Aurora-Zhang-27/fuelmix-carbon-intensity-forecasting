#!/usr/bin/env python3
"""Compare the load-normalized target with a generation-normalized intensity.

The paper's target is c_load = sum_f eta_f m_f / L_hat, where L_hat is the
latest load forecast published no later than the target hour. It factors as
c_gen * G / L_hat with G = sum_f m_f over the reported categories (for CAISO
these include positive net imports and battery discharge). Low correlation
between c_load and c_gen shows that the definition and accounting boundary
matter; it does not identify interchange or any single cause, because G/L_hat
also contains load-forecast error, storage, losses, and reporting conventions.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from fuel_mix_trainer import build_supervised_table,load_config,split_train_val_test
REGIONS={'isone':'ISO-NE','nyiso':'NYISO','pjm':'PJM','ercot':'ERCOT',
         'caiso':'CAISO','miso':'MISO','ieso':'IESO'}


def main():
    rows=[]
    for region,label in REGIONS.items():
        cfg=load_config(ROOT/f'configs/dayahead/{region}_fuelmix_dayahead.json')
        data=build_supervised_table(cfg);fuels=cfg['dataset']['fuel_cols']
        split=cfg['split'];_,_,test=split_train_val_test(data,split['train_frac'],split['val_frac'])
        for period,frame in [('full year',data),('fixed-split test',test)]:
            m=frame[fuels].to_numpy(float)
            ef=np.array([cfg['emission_factors'].get(f,0.) for f in fuels])
            gen=np.nansum(m,axis=1);load=frame['system_load_forecast'].to_numpy(float)
            emis=np.nansum(m*ef,axis=1);ratio=gen/load
            c_load=emis/load;c_gen=emis/gen
            ok=np.isfinite(c_load)&np.isfinite(c_gen)
            rows.append(dict(region=label,period=period,hours=int(ok.sum()),
                mean_c_load=c_load[ok].mean(),mean_c_gen=c_gen[ok].mean(),
                ratio_mean=ratio[ok].mean(),ratio_p05=np.quantile(ratio[ok],.05),
                ratio_p95=np.quantile(ratio[ok],.95),
                corr_c_load_c_gen=np.corrcoef(c_load[ok],c_gen[ok])[0,1]))
    out=pd.DataFrame(rows)
    path=ROOT/'results/summary/ci_definition_check.csv'
    out.to_csv(path,index=False)
    print(out.round(4).to_string(index=False));print(f'wrote {path}')


if __name__=='__main__':main()
