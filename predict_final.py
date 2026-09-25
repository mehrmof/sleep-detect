"""Frozen RC1 judging inference: process each supplied EDF as-is, without labels."""
from pathlib import Path
import argparse, json, os, sys
ROOT=Path(__file__).parent;sys.path.insert(0,str(ROOT/'research_tools'));os.environ.setdefault('OMP_NUM_THREADS','2')
import joblib, numpy as np, pandas as pd, pyedflib
import cnn_c01 as c
from advanced_experiments import Config, transform_record
from sleep_pipeline import ART, STAGES, extract

PROBS=[f'P({s})' for s in STAGES];WEIGHTS=(.6,.2,.2);N3_ALPHA=.6

def emg_for_file(path,n_epochs):
    from scipy.signal import butter,sosfiltfilt
    e=pyedflib.EdfReader(str(path))
    try:
        names=e.getSignalLabels();hits=[i for i,x in enumerate(names) if x.strip().lower()=='emg submental']
        if len(hits)!=1:raise ValueError(f'{path.name}: exact EMG submental count={len(hits)}')
        i=hits[0];fs=float(e.getSampleFrequency(i));sig=e.readSignal(i).astype('float64')
        if fs==100:
            sig=sosfiltfilt(butter(4,10,btype='highpass',fs=fs,output='sos'),sig);ns=len(sig)//100;env=np.sqrt(np.mean(np.square(np.abs(sig[:ns*100]).reshape(ns,100)),axis=1))
        elif fs==1:env=sig
        else:raise ValueError(f'{path.name}: unsupported EMG rate {fs}')
        if len(env)<n_epochs*30:raise ValueError(f'{path.name}: EMG shorter than complete PSG epochs')
        z=env[:n_epochs*30].reshape(n_epochs,30)
        return np.stack([np.median(z,1),np.quantile(z,.75,axis=1)-np.quantile(z,.25,axis=1),np.quantile(z,.9,axis=1)],1).astype('float32')
    finally:e.close()

def cnn_waveforms(path,n_epochs):
    e=pyedflib.EdfReader(str(path))
    try:
        names=e.getSignalLabels();idx=[]
        for name in c.CHANNELS:
            if names.count(name)!=1:raise ValueError(f'{path.name}: exact channel {name} count={names.count(name)}')
            i=names.index(name)
            if float(e.getSampleFrequency(i))!=100 or e.getPhysicalDimension(i)!='uV':raise ValueError(f'{path.name}: bad {name} contract')
            idx.append(i)
        arr=np.empty((n_epochs,3,3000),dtype='float16')
        for ci,i in enumerate(idx):
            sig=e.readSignal(i,start=0,n=n_epochs*3000).astype('float32')
            med=np.median(sig);iqr=np.quantile(sig,.75)-np.quantile(sig,.25);arr[:,ci]=np.clip((sig-med)/(iqr+1e-8),-10,10).reshape(n_epochs,3000).astype('float16')
        return arr
    finally:e.close()

def cnn_probs(arr,stride,device):
    import torch
    starts=list(range(0,len(arr),stride));sumprob=np.zeros((len(arr),5));counts=np.zeros(len(arr));C=c.model_class()
    for fold in range(1,6):
        model=C().to(device);ck=torch.load(ART/'CNN_C01'/f'fold_{fold}'/'best.pt',map_location=device,weights_only=True);model.load_state_dict(ck['model']);model.eval()
        with torch.inference_mode():
            for q in range(0,len(starts),16):
                ss=starts[q:q+16];batch=[];length=[]
                for s in ss:
                    x=np.asarray(arr[s:s+20],dtype='float32');n=len(x);length.append(n)
                    if n<20:x=np.concatenate([x,np.repeat(x[-1:],20-n,0)])
                    batch.append(x)
                x=torch.from_numpy(np.stack(batch)).to(device)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):p=torch.softmax(model(x).float(),-1).cpu().numpy()
                for j,s in enumerate(ss):sumprob[s:s+length[j]]+=p[j,:length[j]];counts[s:s+length[j]]+=1
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
    if np.any(counts==0):raise ValueError('CNN inference coverage failure')
    return sumprob/counts[:,None]

def predict_one(path,output):
    import torch
    path=Path(path);raw,q,_=extract(path);n=len(raw);cfg=Config(representation='raw_norm',context_radius=2,rolling_radius=15,balanced=True,emg=True)
    emg=emg_for_file(path,n);x,_=transform_record(raw,np.ones(n,bool),cfg,emg,None);bundle=joblib.load(ART/'N06'/'model.joblib');pn=bundle['model'].predict_proba(x)
    arr=cnn_waveforms(path,n);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');p20=cnn_probs(arr,20,device);p10=cnn_probs(arr,10,device)
    p=WEIGHTS[0]*pn+WEIGHTS[1]*p20+WEIGHTS[2]*p10;p[:,3]*=N3_ALPHA;p/=p.sum(1,keepdims=True)
    if not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-5):raise ValueError('invalid final probabilities')
    out=pd.DataFrame({'recording_id':path.stem,'epoch_index':np.arange(n),'onset':np.arange(n)*30,'duration':30,'predicted_stage':np.asarray(STAGES)[p.argmax(1)],'quality_flags':q,'model_version':'FINAL_CLASSIFIER_RC1'})
    for i,s in enumerate(STAGES):out[f'P({s})']=p[:,i]
    Path(output).parent.mkdir(parents=True,exist_ok=True);out.to_csv(output,index=False);return len(out)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();src=Path(a.input)
    files=[src] if src.is_file() else sorted(p for p in src.rglob('*.edf') if 'hypnogram' not in p.name.lower())
    if not files:raise SystemExit('No PSG EDF files found')
    target=Path(a.output);multi=len(files)>1 or src.is_dir();target.mkdir(parents=True,exist_ok=True) if multi else target.parent.mkdir(parents=True,exist_ok=True)
    report=[]
    for p in files:
        out=target/f'{p.stem}.csv' if multi else target
        try:report.append({'file':str(p),'status':'ok','epochs':predict_one(p,out),'output':str(out)})
        except Exception as exc:report.append({'file':str(p),'status':'failed','error':str(exc)})
    if multi:(target/'batch_report.json').write_text(json.dumps(report,indent=2))
    if any(x['status']=='failed' for x in report):raise SystemExit(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
