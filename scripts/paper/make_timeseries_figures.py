#!/usr/bin/env python3
"""Time-series figures from the retained rolling predictions.

timeseries_week (and timeseries_week_column, single-column layout): realized
proxy and the three point forecasts in every region
over one fixed calendar week, the first full Monday-Sunday UTC week of the
October 2025 rolling test window (2025-10-06 to 2025-10-12). The week is fixed
by date, not chosen by error.

ercot_case_stack: the ERCOT case window of ercot_contribution_case (largest
realized-proxy range among 72-hour test windows, chosen from realized data
only), with the realized generation by fuel added above the proxy and the gas
contribution.
"""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).resolve().parent))
from polish_paper_figures import BLUE,GRAY,INK,ORANGE,ORDER,style  # noqa: E402
from fuel_mix_trainer import build_supervised_table,load_config  # noqa: E402
CODES={'ISO-NE':'isone','NYISO':'nyiso','PJM':'pjm','ERCOT':'ercot','CAISO':'caiso','MISO':'miso','IESO':'ieso'}
WEEK=(pd.Timestamp('2025-10-06',tz='UTC'),pd.Timestamp('2025-10-13',tz='UTC'))
GRID='#E3E5E7'
# Fuel colors (validated; red and aqua are kept apart by coal and labelled on the areas).
FUELS=[('Nuclear + other','#6250d6'),('Gas','#e34948'),('Coal','#5b5f66'),('Wind','#1baf7a'),('Solar','#eda100')]
METHODS=[('ci_true','Realized',INK,'-',1.15),('persistence_24h','24h persistence',GRAY,':',1.0),
         ('direct_mlp','Direct',BLUE,'-',.9),('fuelmix_ens','Fuel-mix',ORANGE,'-',.9)]


def save(fig,name,args):
    fig.savefig(args.output_dir/(name+'.pdf'),bbox_inches='tight',pad_inches=.035)
    fig.savefig(args.preview_dir/(name+'.png'),dpi=220,bbox_inches='tight',pad_inches=.035)
    plt.close(fig)


def week(args):
    fig,axes=plt.subplots(4,2,figsize=(7.1,5.4),layout='constrained')
    for ax,region in zip(axes.flat,ORDER):
        x=pd.read_csv(ROOT/f'results/experiments/decision_eval_rolling_{CODES[region]}/{CODES[region]}_rolling_predictions.csv')
        x['timestamp']=pd.to_datetime(x.timestamp,utc=True)
        x=x[(x.window=='2025-10-01')&(x.timestamp>=WEEK[0])&(x.timestamp<WEEK[1])]
        assert len(x)==168,(region,len(x))
        t=(x.timestamp-WEEK[0]).dt.total_seconds().to_numpy()/3600
        for col,label,color,ls,lw in METHODS:
            ax.plot(t,x[col],color=color,ls=ls,lw=lw,label=label,zorder=3 if col=='ci_true' else 2)
        ax.set_title(region,loc='left',pad=2)
        ax.set_xlim(0,168);ax.set_xticks(range(0,169,24))
        ax.set_xticklabels([d.strftime('%a %d') for d in pd.date_range(WEEK[0],periods=8,freq='D')] ,fontsize=7)
        ax.grid(axis='y',color=GRID,lw=.45);ax.set_axisbelow(True)
        ax.tick_params(axis='y',labelsize=7)
    for ax in axes[:,0]:ax.set_ylabel('CI (tCO$_2$/MWh)')
    leg=axes[3,1];leg.axis('off')
    leg.legend(handles=[Line2D([],[],color=c,ls=s,lw=w+.3,label=l) for _,l,c,s,w in METHODS],
               loc='center',frameon=False,fontsize=8,title='October 6–12, 2025 (UTC)',title_fontsize=8)
    save(fig,'timeseries_week',args)


def week_column(args):
    """Single-column version of timeseries_week: shared time axis, 6-7 pt text."""
    fig,axes=plt.subplots(4,2,figsize=(3.5,2.95),sharex=True,layout='constrained')
    for ax,region in zip(axes.flat,ORDER):
        x=pd.read_csv(ROOT/f'results/experiments/decision_eval_rolling_{CODES[region]}/{CODES[region]}_rolling_predictions.csv')
        x['timestamp']=pd.to_datetime(x.timestamp,utc=True)
        x=x[(x.window=='2025-10-01')&(x.timestamp>=WEEK[0])&(x.timestamp<WEEK[1])]
        assert len(x)==168,(region,len(x))
        t=(x.timestamp-WEEK[0]).dt.total_seconds().to_numpy()/3600
        for col,label,color,ls,lw in METHODS:
            ax.plot(t,x[col],color=color,ls=ls,lw=lw*.75,label=label,zorder=3 if col=='ci_true' else 2)
        ax.set_title(region,loc='left',pad=1,fontsize=6.5)
        ax.set_xlim(0,168);ax.set_xticks(range(0,169,24))
        ax.grid(axis='y',color=GRID,lw=.4);ax.set_axisbelow(True)
        ax.tick_params(labelsize=5.5,length=2,pad=1.5)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    days=[str(d.day) for d in pd.date_range(WEEK[0],periods=8,freq='D')]
    for ax in (axes[3,0],axes[2,1]):
        ax.set_xticks(range(0,169,24),days);ax.tick_params(labelbottom=True)
        ax.set_xlabel('October 2025 (UTC)',fontsize=6,labelpad=1)
    fig.supylabel('CI (tCO$_2$/MWh)',fontsize=6.5)
    leg=axes[3,1];leg.axis('off')
    leg.legend(handles=[Line2D([],[],color=c,ls=s,lw=w,label=l) for _,l,c,s,w in METHODS],
               loc='center',frameon=False,fontsize=6.5)
    save(fig,'timeseries_week_column',args)


def ercot_case(args,out_name='ercot_case_stack',figsize=(3.5,4.6)):
    x=pd.read_csv(ROOT/'results/attribution/ercot_dayahead_case_72h.csv')
    x['timestamp']=pd.to_datetime(x.timestamp,utc=True)
    assert len(x)==72
    cfg=load_config(ROOT/'configs/dayahead/ercot_fuelmix_dayahead.json')
    gen=build_supervised_table(cfg).set_index('timestamp').loc[x.timestamp]
    groups={'Nuclear + other':gen[['nuclear','hydro','other']].sum(axis=1),'Gas':gen['nat_gas'],
            'Coal':gen['coal'],'Wind':gen['wind'],'Solar':gen['solar']}
    fig,axes=plt.subplots(3,1,figsize=figsize,sharex=True,layout='constrained',
                          gridspec_kw={'height_ratios':[1.15,1,.85]})
    t=np.arange(72);base=np.zeros(72)
    for name,color in FUELS:
        v=groups[name].to_numpy(float)/1000
        axes[0].fill_between(t,base,base+v,color=color,lw=0,alpha=.9)
        axes[0].plot(t,base+v,color='white',lw=.8)  # surface gap between bands
        # Direct label at the centre of the band's widest stretch (smoothed), skipped for thin bands.
        smooth=pd.Series(v).rolling(9,center=True,min_periods=1).mean().to_numpy()
        mid=6+int(np.argmax(smooth[6:-6]))  # keep labels clear of the plot edges
        if v[mid]>6:axes[0].text(mid,base[mid]+v[mid]/2,name,ha='center',va='center',fontsize=6.5,
                                 color='white' if name in ('Gas','Coal') else INK)
        base=base+v
    axes[0].set_ylabel('Generation (GW)');axes[0].set_ylim(0,78);axes[0].set_yticks([0,20,40,60])
    axes[0].legend(handles=[plt.Rectangle((0,0),1,1,color=c,lw=0) for _,c in FUELS],labels=[n for n,_ in FUELS],
                   loc='upper center',ncol=5,frameon=False,fontsize=6,handlelength=.9,handletextpad=.3,
                   columnspacing=.7,borderaxespad=.1)
    lines=[('realized','Realized',INK,'-',1.2),('persistence','24h persistence',GRAY,':',1.0),
           ('fuelmix','Fuel-mix',BLUE,'--',1.0),('fuelmix_ws24','Fuel-mix + wind/solar',ORANGE,'-',1.0)]
    for tag,label,color,ls,lw in lines:
        axes[1].plot(t,x['ci_'+tag],color=color,ls=ls,lw=lw,label=label)
        if tag!='persistence':axes[2].plot(t,x['nat_gas_'+tag],color=color,ls=ls,lw=lw)
    axes[1].set_ylabel('CI (tCO$_2$/MWh)');axes[2].set_ylabel('Gas contribution\n(tCO$_2$/MWh)')
    axes[1].set_ylim(0,.5);axes[2].set_ylim(0,.32)
    fig.legend(*axes[1].get_legend_handles_labels(),loc='outside upper center',ncol=2,frameon=False,fontsize=7)
    for ax in axes:
        ax.set_xlim(0,71);ax.grid(axis='y',lw=.4,color=GRID);ax.set_axisbelow(True)
    idx=[0,24,48,71]
    axes[2].set_xticks(idx,[x.timestamp.iloc[i].strftime('%b %d\n%H:%M') for i in idx]);axes[2].set_xlabel('2025 UTC')
    save(fig,out_name,args)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,default=ROOT/'paper_figures')
    ap.add_argument('--preview-dir',type=Path,required=True)
    args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True);args.preview_dir.mkdir(parents=True,exist_ok=True)
    style();week(args);week_column(args);ercot_case(args)
    ercot_case(args,'ercot_case_stack_short',(3.5,3.45))


if __name__=='__main__':main()
