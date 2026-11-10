#!/usr/bin/env python3
"""Region overview for 2025 (the paper's region table).

For each region, over the 2025 rows of the day-ahead-like supervised table:
mean load forecast (GW), mean CI proxy (tCO2/MWh), number of reported fuel
categories F, the three largest sources by share of reported generation, and
the combined wind and solar share. Writes results/summary/region_overview_2025.csv.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from fuel_mix_trainer import build_supervised_table,fuel_mix_to_carbon_intensity_np,load_config  # noqa: E402
REGIONS={'isone':'ISO-NE','nyiso':'NYISO','pjm':'PJM','ercot':'ERCOT','caiso':'CAISO','miso':'MISO','ieso':'IESO'}
NAMES={'nat_gas':'gas','natural_gas':'gas'}


def main():
    rows=[]
    for code,label in REGIONS.items():
        cfg=load_config(ROOT/f'configs/dayahead/{code}_fuelmix_dayahead.json')
        data=build_supervised_table(cfg)
        fuels=cfg['dataset']['fuel_cols']
        year=data[data.timestamp.dt.year==2025]
        _,ci=fuel_mix_to_carbon_intensity_np(year[fuels].to_numpy(float),cfg['emission_factors'],fuels,
                                             year['system_load_forecast'].to_numpy(float))
        share=year[fuels].sum()/year[fuels].to_numpy().sum()
        top=share.sort_values(ascending=False).head(3)
        rows.append(dict(region=label,mean_load_forecast_gw=year['system_load_forecast'].mean()/1000,
                         mean_ci_proxy=ci.mean(),n_fuel_categories=len(fuels),
                         largest_sources=', '.join(f'{NAMES.get(f,f)} {100*s:.0f}%' for f,s in top.items()),
                         wind_solar_share=sum(share.get(f,0.0) for f in ('wind','solar'))))
    out=ROOT/'results/summary/region_overview_2025.csv'
    pd.DataFrame(rows).to_csv(out,index=False);print('wrote',out)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__=='__main__':main()
