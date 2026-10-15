#!/usr/bin/env python3
"""Render publication figures from the retained, reviewed compact CSVs."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
ROOT=Path(__file__).resolve().parents[2]
ORDER=['ISO-NE','NYISO','PJM','ERCOT','CAISO','MISO','IESO']
BLUE='#2463A6';ORANGE='#B16B2D';PURPLE='#7655A0';GRAY='#687078';INK='#252A30'


def style():
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,'xtick.labelsize':7.5,
        'ytick.labelsize':7.5,'legend.fontsize':7.5,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42,
        'ps.fonttype':42,'figure.facecolor':'white','savefig.facecolor':'white'})


def save(fig,name,args):
    fig.savefig(args.output_dir/(name+'.pdf'),bbox_inches='tight',pad_inches=.035)
    fig.savefig(args.preview_dir/(name+'.png'),dpi=220,bbox_inches='tight',pad_inches=.035)
    plt.close(fig)


def intervals(args):
    x=pd.read_csv(ROOT/'results/robustness_intervals/metrics.csv')
    x=x[x.calibrated]
    assert len(x)==28
    fig,axes=plt.subplots(2,1,figsize=(3.5,4.55),layout='constrained')
    methods=[('Persistence',GRAY,'o'),('Direct',BLUE,'s'),('In-sample',ORANGE,'D'),('OOF',PURPLE,'^')]
    y=np.arange(7)
    for j,(method,color,marker) in enumerate(methods):
        rows=x[x.method==method].set_index('region').loc[ORDER]
        for ax,col in zip(axes,['coverage_80','mean_width']):
            ax.scatter(rows[col],y+(j-1.5)*.16,c=color,marker=marker,s=21,
                linewidths=.5,edgecolors='white',zorder=3,label=method)
    for ax in axes:
        ax.set_yticks(y,ORDER);ax.invert_yaxis();ax.grid(axis='x',color='#E3E5E7',lw=.45)
        ax.set_axisbelow(True)
    axes[0].axvline(.8,ls='--',lw=.8,color=INK)
    axes[0].set_xlim(.45,1.0);axes[0].xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    axes[0].set_xlabel('Test coverage of nominal 80% intervals')
    axes[0].set_title('(a) Coverage after independent calibration',loc='left')
    axes[1].set_xlim(0,None);axes[1].set_xlabel('Mean interval width (tCO$_2$/MWh)')
    axes[1].set_title('(b) Width at the same nominal level',loc='left')
    handles=[Line2D([],[],ls='',marker=m,color=c,markersize=4,label=k) for k,c,m in methods]
    fig.legend(handles=handles,loc='outside upper center',ncol=2,frameon=False,
        columnspacing=1.3,handletextpad=.4)
    save(fig,'interval_robustness',args)


def attribution(args):
    x=pd.read_csv(ROOT/'results/attribution/ercot_dayahead_case_72h.csv')
    fuels=['coal','nat_gas','other']
    assert len(x)==72 and np.allclose(x[[f+'_realized' for f in fuels]].sum(axis=1),x.ci_realized)
    for tag in ['fuelmix','fuelmix_ws24']:
        assert np.allclose(x[[f+'_'+tag for f in fuels]].sum(axis=1),x['ci_'+tag],atol=1e-6)
    fig,axes=plt.subplots(2,1,figsize=(3.5,3.35),sharex=True,layout='constrained')
    t=np.arange(len(x))
    lines=[('realized','Realized',INK,'-',1.25),('fuelmix','Fuel-mix',BLUE,'--',1.1),
           ('fuelmix_ws24','Fuel-mix + wind/solar',ORANGE,'-',1.1),('persistence','24h persist.',GRAY,':',1)]
    for tag,label,color,ls,lw in lines:
        axes[0].plot(t,x['ci_'+tag],color=color,ls=ls,lw=lw,label=label)
        if tag!='persistence':axes[1].plot(t,x['nat_gas_'+tag],color=color,ls=ls,lw=lw)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=2,frameon=False,fontsize=7)
    axes[0].set_ylabel('Proxy (tCO$_2$/MWh)');axes[1].set_ylabel('Gas contribution\n(tCO$_2$/MWh)')
    axes[0].set_ylim(0,.5);axes[0].set_yticks([0,.2,.4]);axes[1].set_ylim(0,.32);axes[1].set_yticks([0,.1,.2,.3])
    for ax in axes:
        ax.set_xlim(0,71);ax.grid(axis='y',lw=.4,color='#E3E5E7');ax.set_axisbelow(True)
    idx=[0,24,48,71];dates=pd.to_datetime(x.timestamp,utc=True)
    axes[1].set_xticks(idx,[dates.iloc[i].strftime('%b %d\n%H:%M') for i in idx])
    axes[1].set_xlabel('2025 UTC')
    save(fig,'ercot_contribution_case',args)


def history(args):
    def ratios(path):
        w=pd.read_csv(ROOT/path).pivot_table(index=['region','window'],columns='method',values='ci_rmse_t_per_mwh')
        r=(w['Fuel-mix ens.']/w['24h persistence']).unstack('window')
        r.columns=[c[5:] for c in r.columns]
        return r.loc[ORDER]
    r24=ratios('results/rolling_2024_windows/seven_regions_rolling_rmse_by_window_aligned.csv')
    r25=ratios('results/rolling_rmse_aligned/seven_regions_rolling_rmse_by_window_aligned.csv')
    fig,ax=plt.subplots(figsize=(3.5,3.3),layout='constrained')
    y=np.arange(7)
    for j,(month,label,color) in enumerate([('06-01','June',ORANGE),('08-01','August',BLUE),('10-01','October',PURPLE)]):
        yy=y+(j-1)*.24
        ax.hlines(yy,r25[month],r24[month],color=color,lw=.9,alpha=.6)
        ax.scatter(r24[month],yy,facecolors='white',edgecolors=color,s=20,lw=.9,zorder=3)
        ax.scatter(r25[month],yy,color=color,s=20,zorder=3,label=label)
    ax.axvline(1,color=INK,lw=.7,ls='--')
    ax.set_yticks(y,ORDER);ax.invert_yaxis();ax.set_xscale('log')
    ticks=[.75,1,1.5,2,3];ax.set_xticks(ticks,[f'{v:g}' for v in ticks]);ax.minorticks_off()
    ax.set_xlabel('Fuel-mix RMSE / 24h persistence RMSE')
    ax.grid(axis='x',color='#E3E5E7',lw=.45);ax.set_axisbelow(True)
    handles=[Line2D([],[],ls='',marker='o',color=c,label=l) for l,c in [('June',ORANGE),('August',BLUE),('October',PURPLE)]]
    handles+=[Line2D([],[],ls='',marker='o',markerfacecolor='white',markeredgecolor=GRAY,color=GRAY,label='2024 (120-242 days)'),
              Line2D([],[],ls='',marker='o',color=GRAY,label='2025 (485-607 days)')]
    fig.legend(handles=handles,loc='outside upper center',ncol=3,frameon=False,fontsize=7,columnspacing=1,handletextpad=.3)
    save(fig,'history_contrast',args)


def overview(args):
    rmse=pd.read_csv(ROOT/'results/aligned_protocol/dayahead_baselines_aligned.csv').set_index('region').loc[ORDER]
    dec=pd.read_csv(ROOT/'results/rolling_rmse_aligned/seven_regions_rolling_decision_summary_aligned.csv')
    fig,axes=plt.subplots(2,1,figsize=(3.5,4.5),layout='constrained')
    y=np.arange(7)
    for method,col,color,marker,off in [('Direct MLP','direct_mlp_rmse',BLUE,'o',-.13),('Fuel-mix','fuelmix_ensemble_rmse',ORANGE,'^',.13)]:
        gain=100*(rmse.persistence_rmse-rmse[col])/rmse.persistence_rmse
        axes[0].scatter(gain,y+off,c=color,marker=marker,s=25,zorder=3,label=method)
    axes[0].legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.19))
    gains_a=pd.concat([100*(rmse.persistence_rmse-rmse[c])/rmse.persistence_rmse for c in ['direct_mlp_rmse','fuelmix_ensemble_rmse']])
    axes[0].set_xlim(min(-5,gains_a.min()-3),gains_a.max()+3);axes[0].set_xlabel('RMSE reduction vs. 24h persistence (%)')
    axes[0].set_title('(a) Fixed-split day-ahead-like accuracy, 2024-2025',loc='left',pad=23)
    gains=[]
    for region in ORDER:
        sub=dec[dec.region==region].set_index('method')
        gains.append(sub.loc['Fuel-mix ens.','mean_capture']-sub.loc[['Climatology','24h persistence','Direct MLP'],'mean_capture'].max())
    axes[1].hlines(y,0,gains,color=GRAY,lw=1)
    axes[1].scatter(gains,y,color=ORANGE,s=24,zorder=3)
    for i,gain in enumerate(gains):axes[1].text(gain+(.009 if gain>=0 else -.009),i,f'{gain:+.3f}',ha='left' if gain>=0 else 'right',va='center',fontsize=7)
    pad=.25*(max(gains)-min(gains))
    axes[1].set_xlim(min(0,min(gains))-pad,max(0,max(gains))+pad);axes[1].set_xlabel('Fuel-mix capture minus best scalar capture')
    axes[1].set_title('(b) Three-seed rolling scheduling, 2025 windows',loc='left')
    for ax in axes:
        ax.axvline(0,color=INK,lw=.65);ax.set_yticks(y,ORDER);ax.invert_yaxis()
        ax.grid(axis='x',color='#E3E5E7',lw=.45);ax.set_axisbelow(True)
    save(fig,'seven_region_overview',args)


def seeds(args):
    p=pd.read_csv(ROOT/'results/robustness_seeds/per_seed.csv')
    ens=pd.read_csv(ROOT/'results/robustness_seeds/ensemble_summary.csv')
    ci=pd.read_csv(ROOT/'results/robustness_seeds/paired_block_intervals.csv')
    methods=['Direct MSE','Direct SPO+','Fuel-mix MSE','Fuel-mix SPO+']
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.7),layout='constrained',gridspec_kw={'width_ratios':[1.03,1]})
    for j,method in enumerate(methods):
        vals=p[p.method==method].sort_values('seed').mean_capture.to_numpy()
        axes[0].scatter(j+np.linspace(-.13,.13,10),vals,c='#9DA3A9',s=16,zorder=2)
        axes[0].errorbar(j,vals.mean(),yerr=vals.std(ddof=1),color=INK,marker='_',ms=11,capsize=3,lw=.85,zorder=3)
        for size,color,marker,off in [(3,ORANGE,'D',-.20),(10,BLUE,'s',.20)]:
            val=ens[(ens.method==method)&(ens.ensemble_size==size)].mean_capture.iloc[0]
            axes[0].scatter(j+off,val,c=color,marker=marker,s=24,zorder=4)
    axes[0].set_xticks(range(4),['Direct\nMSE','Direct\nSPO+','Fuel-mix\nMSE','Fuel-mix\nSPO+'])
    days=int(p.n_days.iloc[0]);lo=min(p.mean_capture.min(),ens.mean_capture.min())
    axes[0].set_ylabel(f'Mean capture over {days} days');axes[0].set_ylim(lo-.02,max(p.mean_capture.max(),ens.mean_capture.max())+.02)
    axes[0].grid(axis='y',color='#E3E5E7',lw=.45);axes[0].set_axisbelow(True)
    axes[0].set_title('(a) Training variation and ensemble size',loc='left')
    pairs=[('Direct SPO+','Fuel-mix MSE'),('Direct SPO+','Direct MSE'),('Fuel-mix SPO+','Fuel-mix MSE')]
    for j,(m,b) in enumerate(pairs):
        for size,color,marker,off in [(3,ORANGE,'D',-.13),(10,BLUE,'s',.13)]:
            row=ci[(ci.method==m)&(ci.baseline==b)&(ci.ensemble_size==size)&(ci.block_days==7)].iloc[0]
            axes[1].errorbar(row.delta_capture,j+off,
                xerr=[[row.delta_capture-row.capture_ci_low],[row.capture_ci_high-row.delta_capture]],
                fmt=marker,color=color,ms=4,capsize=2,lw=1)
    axes[1].set_yticks(range(3),['Direct SPO+\n− Fuel-mix MSE','Direct SPO+\n− Direct MSE','Fuel-mix SPO+\n− Fuel-mix MSE'])
    axes[1].invert_yaxis();axes[1].axvline(0,color=INK,lw=.7)
    sub=ci[(ci.block_days==7)&ci.baseline.ne('24h persistence')]
    lo,hi=sub.capture_ci_low.min(),sub.capture_ci_high.max();pad=.1*(hi-lo)
    axes[1].set_xlim(min(lo,0)-pad,hi+pad);axes[1].set_xlabel('Mean capture difference (95% interval)')
    axes[1].grid(axis='x',color='#E3E5E7',lw=.45);axes[1].set_axisbelow(True)
    axes[1].set_title('(b) Seven-day blocks within fixed windows',loc='left')
    handles=[Line2D([],[],ls='',marker='o',color='#9DA3A9',label='Individual seed'),
        Line2D([],[],ls='-',marker='_',color=INK,label='Seed mean ± SD'),
        Line2D([],[],ls='',marker='D',color=ORANGE,label='3-seed ensemble'),
        Line2D([],[],ls='',marker='s',color=BLUE,label='10-seed ensemble')]
    fig.legend(handles=handles,loc='outside upper center',ncol=4,frameon=False)
    save(fig,'ercot_seed_robustness',args)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,default=ROOT.parent/'paper/paper_figures')
    ap.add_argument('--preview-dir',type=Path,required=True)
    ap.add_argument('--only',nargs='+',choices=['intervals','attribution','overview','seeds','history'])
    args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True);args.preview_dir.mkdir(parents=True,exist_ok=True)
    style()
    for fn in [intervals,attribution,overview,seeds,history]:
        if not args.only or fn.__name__ in args.only:fn(args)
if __name__=='__main__':main()
