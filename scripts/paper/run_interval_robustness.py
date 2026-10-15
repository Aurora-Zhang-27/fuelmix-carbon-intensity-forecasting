#!/usr/bin/env python3
"""Chronological OOF ablation; checkpoint selection and calibration are disjoint.
Outer split 70/15/15; validation is split in chronological halves. Three
expanding folds follow a 40% training warm-up; each fold reserves the last 20%
of prior history for selection. All quantile variants use the same OOF rows.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT/'src')]
from run_quantile_forecast import (build_supervised_table,load_config,split_train_val_test,
    feature_columns_from_table,true_ci,train_quantile_model,train_fuelmix_point_outputs,
    conformal_calibrate,interval_metrics)
REGIONS=['nyiso','caiso','isone','pjm','ercot','miso','ieso']
LABELS=dict(nyiso='NYISO',caiso='CAISO',isone='ISO-NE',pjm='PJM',ercot='ERCOT',miso='MISO',ieso='IESO')


def frames_split(data,cfg):
    tr,va,te=split_train_val_test(data,cfg['split']['train_frac'],cfg['split']['val_frac'])
    mid=len(va)//2
    sel,cal=va.iloc[:mid],va.iloc[mid:]
    assert tr.timestamp.max()<sel.timestamp.min()<sel.timestamp.max()<cal.timestamp.min()<cal.timestamp.max()<te.timestamp.min()
    return tr,sel,cal,te


def fold_slices(n):
    edges=np.linspace(int(n*.4),n,4,dtype=int)
    return [(0,int(.8*lo),int(lo),int(hi)) for lo,hi in zip(edges[:-1],edges[1:])]


def run_region(region,args):
    out=args.out_dir/region
    out.mkdir(parents=True,exist_ok=True)
    if args.resume and (out/'metrics.csv').exists():
        print(region,'resume completed',flush=True);return
    cfg=load_config(ROOT/f'configs/{region}_fuelmix.json')
    data=build_supervised_table(cfg)
    tr,sel,cal,te=frames_split(data,cfg)
    features=feature_columns_from_table(data,cfg)
    pool=pd.concat([cal,te],ignore_index=True)
    nc=len(cal)
    audit=[]
    def record(role,df):
        audit.append(dict(region=LABELS[region],role=role,n=len(df),first=str(df.timestamp.min()),last=str(df.timestamp.max())))
    for role,df in [('train',tr),('selection',sel),('calibration',cal),('test',te)]:record(role,df)
    print(region,'full-history surrogate',len(tr),len(sel),len(cal),len(te),flush=True)
    full_tr,full_sel,full_pool=train_fuelmix_point_outputs(cfg,features,tr,sel,pool,seed=args.seed)
    oof=np.full(len(tr),np.nan,dtype=np.float32)
    for i,(start,split,lo,hi) in enumerate(fold_slices(len(tr))):
        fit,stop,held=tr.iloc[start:split],tr.iloc[split:lo],tr.iloc[lo:hi]
        assert fit.timestamp.max()<stop.timestamp.min() and stop.timestamp.max()<held.timestamp.min()
        print(region,'OOF fold',i+1,len(fit),len(stop),len(held),flush=True)
        _,_,pred=train_fuelmix_point_outputs(cfg,features,fit,stop,held,seed=args.seed)
        oof[lo:hi]=pred['ci_pred']
        for role,df in [('fit',fit),('selection',stop),('heldout',held)]:record(f'fold{i+1}_{role}',df)
    mask=np.isfinite(oof)
    assert mask.sum()==len(tr)-fold_slices(len(tr))[0][2]
    record('matched_quantile_train',tr.loc[mask])
    xt=tr[features].to_numpy(dtype=np.float32)[mask]
    xs=sel[features].to_numpy(dtype=np.float32)
    xp=pool[features].to_numpy(dtype=np.float32)
    yt,ys,yp=true_ci(tr,cfg)[mask],true_ci(sel,cfg),true_ci(pool,cfg)
    yc,ye=yp[:nc],yp[nc:]
    def aug(x,c):return np.column_stack([x,np.asarray(c,dtype=np.float32)])
    variants={'Direct':(xt,xs,xp),
        'In-sample':(aug(xt,full_tr['ci_pred'][mask]),aug(xs,full_sel['ci_pred']),aug(xp,full_pool['ci_pred'])),
        'OOF':(aug(xt,oof[mask]),aug(xs,full_sel['ci_pred']),aug(xp,full_pool['ci_pred']))}
    rows,curves=[],[]
    predictions=pd.DataFrame(dict(timestamp=te.timestamp.to_numpy(),ci_true=ye))
    calibration_predictions=pd.DataFrame(dict(timestamp=cal.timestamp.to_numpy(),ci_true=yc))
    def evaluate(method,qc,qt,ntrain,nsel):
        calibrated,adjustment=conformal_calibrate(qc,yc,qt,.2)
        for flag,q in [(False,qt),(True,calibrated)]:
            cq=qc.copy()
            if flag:
                cq[:,0]-=adjustment
                cq[:,2]+=adjustment
            rows.append({**interval_metrics(region,method,ye,q),'calibrated':flag,
                'calibration_coverage':float(np.mean((yc>=cq[:,0])&(yc<=cq[:,2]))),
                'adjustment':adjustment if flag else 0.,'n_train':ntrain,'n_selection':nsel,
                'n_calibration':nc,'n_test':len(te),'seed':args.seed})
        for alpha in np.linspace(.05,.5,19):
            q,_=conformal_calibrate(qc,yc,qt,float(alpha))
            curves.append(dict(region=LABELS[region],method=method,alpha=alpha,
                coverage=np.mean((ye>=q[:,0])&(ye<=q[:,2])),mean_width=np.mean(q[:,2]-q[:,0])))
        for j,name in enumerate(['q10','q50','q90']):predictions[method+'_'+name]=calibrated[:,j]
        for j,name in enumerate(['q10','q50','q90']):calibration_predictions[method+'_'+name]=qc[:,j]
    for method,(xtr,xsel,xpool) in variants.items():
        print(region,'quantile',method,flush=True)
        _,q=train_quantile_model(xtr,yt,xsel,ys,xpool,seed=args.seed,epochs=args.epochs)
        assert np.isfinite(q).all() and np.all(np.diff(q,axis=1)>=0)
        evaluate(method,q[:nc],q[nc:],len(yt),len(sel))
    residual=true_ci(tr,cfg)-tr.historical_ci_1h_lag.to_numpy()
    q=pool.historical_ci_1h_lag.to_numpy()[:,None]+np.quantile(residual,[.1,.5,.9])
    evaluate('Persistence',q[:nc],q[nc:],len(tr),0)
    pd.DataFrame(audit).to_csv(out/'audit.csv',index=False)
    pd.DataFrame(curves).to_csv(out/'curves.csv',index=False)
    args.cache_dir.mkdir(parents=True,exist_ok=True)
    predictions.to_csv(args.cache_dir/f'{region}_test_intervals.csv',index=False)
    calibration_predictions.to_csv(args.cache_dir/f'{region}_calibration_raw_intervals.csv',index=False)
    np.savez_compressed(args.cache_dir/f'{region}_features.npz',oof=oof,mask=mask,
        full_train=full_tr['ci_pred'],selection=full_sel['ci_pred'],pool=full_pool['ci_pred'])
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    print(region,'DONE',[(r['method'],round(r['coverage_80'],3),round(r['mean_width'],4)) for r in rows if r['calibrated']],flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--regions',nargs='+',default=REGIONS,choices=REGIONS)
    ap.add_argument('--out-dir',type=Path,default=ROOT/'results/robustness_intervals')
    ap.add_argument('--cache-dir',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--epochs',type=int,default=400)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    protocol=dict(seed=args.seed,epochs=args.epochs,outer_split=[.7,.15,.15],validation_selection_fraction=.5,
        oof_warmup_fraction=.4,oof_folds=3,fold_selection_fraction=.2,alpha=.2,point_epochs=60,
        point_patience=8,quantile_patience=25,comparison='matched post-warmup rows',
        numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__)
    path=args.out_dir/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Protocol changed; use another output directory'
    else:path.write_text(json.dumps(protocol,indent=2))
    for region in args.regions:run_region(region,args)
    for filename in ['metrics','audit','curves']:
        files=sorted(args.out_dir.glob(f'*/{filename}.csv'))
        if files:pd.concat([pd.read_csv(p) for p in files],ignore_index=True).to_csv(args.out_dir/f'{filename}.csv',index=False)
if __name__=='__main__':main()
