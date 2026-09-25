from pathlib import Path
import argparse, json, hashlib, time, warnings
import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from concurrent.futures import ThreadPoolExecutor
FEATURE_WORKERS=ThreadPoolExecutor(max_workers=3)
ROOT = Path(__file__).parent
DATA = ROOT / 'sleep-edf-database-expanded-1.0.0'
ART = ROOT / 'artifacts'
CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal']
STAGES = ['Wake','N1','N2','N3','REM']
MAPPING = {'Sleep stage W':0,'Sleep stage 1':1,'Sleep stage 2':2,'Sleep stage 3':3,'Sleep stage 4':3,'Sleep stage R':4}
VERSION = 'physical-welch-v1'

def save(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def labels(path,n):
    import mne
    a=mne.read_annotations(str(path))
    y=np.full(n,-1,dtype=np.int8); covered=np.zeros(n,bool)
    for o,d,s in zip(a.onset,a.duration,a.description):
        if abs(o/30-round(o/30))>1e-6 or abs(d/30-round(d/30))>1e-6:
            raise ValueError(f'Non-30-second annotation boundary: {path} {o} {d}')
        lo=max(0,round(o/30)); hi=min(n,round((o+d)/30))
        if hi<=lo: continue
        if covered[lo:hi].any(): raise ValueError(f'Overlapping annotations: {path}')
        if s not in MAPPING and s not in ['Sleep stage ?','Movement time']:
            raise ValueError(f'Unknown annotation {s}')
        y[lo:hi]=MAPPING.get(s,-1); covered[lo:hi]=True
    return y

def features(x,parallel=True):
    # x: channel, epoch, sample, physical microvolts; no fitted transformation.
    if parallel and len(x)>1:
        return np.concatenate(list(FEATURE_WORKERS.map(lambda z:features(z[None],False),x)),axis=1)
    out=[]
    for z in x:
        f,p=welch(z,fs=100,nperseg=500,noverlap=250,window='hamming',average='median',axis=-1)
        bands=[(.4,1),(1,4),(4,8),(8,12),(12,16),(16,30)]
        powers=np.stack([p[:,(f>=a)&(f<b)].sum(axis=1)*.2 for a,b in bands],axis=1)
        total=powers.sum(axis=1,keepdims=True)+1e-12
        sd=z.std(axis=1); diff=np.diff(z,axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            shape=np.stack([skew(z,axis=1),kurtosis(z,axis=1)],axis=1)
        stat=np.stack([np.log1p(sd),np.log1p(np.ptp(z,axis=1)),np.log1p(np.quantile(z,.75,axis=1)-np.quantile(z,.25,axis=1)),np.log1p(np.sqrt((z*z).mean(axis=1))),diff.std(axis=1)/(sd+1e-9),np.mean(np.diff(np.signbit(z),axis=1),axis=1)],axis=1)
        out.extend([np.log10(powers+1e-12),powers/total,stat,np.nan_to_num(shape)])
    return np.concatenate(out,axis=1).astype('float32')

def extract(path):
    e=pyedflib.EdfReader(str(path))
    try:
        names=e.getSignalLabels(); idx=[]
        for c in CHANNELS:
            if c not in names: raise ValueError(f'{path.name}: missing {c}')
            i=names.index(c); idx.append(i)
            if e.getSampleFrequency(i)!=100 or e.getPhysicalDimension(i)!='uV':
                raise ValueError(f'{path.name}: {c} requires 100 Hz and uV')
        n=int(min(e.getNSamples()[i] for i in idx)//3000)
        if n<1: raise ValueError(f'{path.name}: no complete 30-second epoch')
        blocks=[]; flags=[]
        for lo in range(0,n,32):
            count=min(32,n-lo)
            x=np.stack([e.readSignal(i,start=lo*3000,n=count*3000).reshape(count,3000) for i in idx])
            finite=np.isfinite(x).all(axis=(0,2))
            if not finite.all(): raise ValueError(f'{path.name}: nonfinite physical samples')
            flags.append((np.ptp(x[0],axis=1)==0).astype('uint8')+2*(np.abs(x)>500).any(axis=(0,2)).astype('uint8'))
            blocks.append(features(x))
        return np.concatenate(blocks),np.concatenate(flags),e.getStartdatetime().isoformat()
    finally: e.close()

def prepare():
    ART.mkdir(exist_ok=True); (ART/'cache').mkdir(exist_ok=True)
    files=sorted(DATA.rglob('*-PSG.edf')); subjects=sorted({p.name[:5] for p in files})
    splitpath=ART/'split.json'
    if splitpath.exists(): split=json.loads(splitpath.read_text())
    else:
        rng=np.random.default_rng(20260923); split={k:[] for k in ['train','validation','test']}
        for cohort,nt,nv in [('SC',47,16),('ST',13,4)]:
            group=np.array([s for s in subjects if s.startswith(cohort)]); rng.shuffle(group)
            for k,v in zip(split,[group[:nt],group[nt:nt+nv],group[nt+nv:]]): split[k]+=v.tolist()
        save(splitpath,split)
    assert len(sum(split.values(),[]))==len(set(sum(split.values(),[])))==len(subjects)==100
    assert set(sum(split.values(),[]))==set(subjects)
    rows=[]
    for j,p in enumerate(files):
        target=ART/'cache'/f'{p.stem}.npz'
        partition=next(k for k,v in split.items() if p.name[:5] in v)
        rows.append(dict(file=p.name,subject=p.name[:5],cohort=p.name[:2],partition=partition))
        if partition=='test': continue # sealed, not even feature extraction tonight
        if not target.exists():
            t=time.perf_counter(); x,q,start=extract(p)
            hyp=list(p.parent.glob(p.name[:6]+'*Hypnogram.edf'))
            if len(hyp)!=1: raise ValueError(f'Ambiguous annotation pairing: {p}')
            with p.open('rb') as f: ph=f.read(184)[168:184]
            with hyp[0].open('rb') as f: hh=f.read(184)[168:184]
            if ph!=hh: raise ValueError(f'PSG/hypnogram start mismatch: {p}')
            y=labels(hyp[0],len(x))
            temporary=target.with_suffix('.tmp.npz')
            np.savez_compressed(temporary,x=x,y=y,q=q,start=start,version=VERSION)
            temporary.replace(target)
            print(f'{j+1}/{len(files)} {p.name} {len(x)} epochs {time.perf_counter()-t:.1f}s',flush=True)
    pd.DataFrame(rows).to_csv(ART/'manifest.csv',index=False)

def context(x):
    return np.concatenate([x,np.vstack([x[:1],x[:-1]]),np.vstack([x[1:],x[-1:]])],axis=1)

def load(part,ctx):
    split=json.loads((ART/'split.json').read_text()); xs=[]; ys=[]; ids=[]
    for p in sorted((ART/'cache').glob('*.npz')):
        if p.name[:5] not in split[part]: continue
        with np.load(p) as z:
            assert str(z['version'])==VERSION
            x=z['x']; y=z['y']; mask=y>=0
            xs.append((context(x) if ctx else x)[mask]); ys.append(y[mask]); ids.extend([p.stem]*int(mask.sum()))
    return np.concatenate(xs),np.concatenate(ys),np.array(ids)

def metrics(y,p):
    from sklearn.metrics import f1_score,confusion_matrix,cohen_kappa_score,recall_score,precision_score
    k=float(cohen_kappa_score(y,p,labels=list(range(5)))) if len(np.unique(np.r_[y,p]))>1 else None
    return dict(macro_f1=float(f1_score(y,p,labels=range(5),average='macro',zero_division=0)),per_class_f1=f1_score(y,p,labels=range(5),average=None,zero_division=0).tolist(),precision=precision_score(y,p,labels=range(5),average=None,zero_division=0).tolist(),recall=recall_score(y,p,labels=range(5),average=None,zero_division=0).tolist(),kappa=k if k is None or np.isfinite(k) else None,confusion_matrix=confusion_matrix(y,p,labels=range(5)).tolist(),epochs=len(y))

def train(exp,balanced=False,ctx=False):
    import lightgbm as lgb, joblib, psutil, threading
    manifest=pd.read_csv(ART/'manifest.csv')
    expected=set(manifest.loc[manifest.partition!='test','file'].str.replace('.edf','.npz',regex=False))
    actual={p.name for p in (ART/'cache').glob('*.npz')}
    if actual!=expected: raise ValueError('Incomplete or unexpected feature cache; finish prepare first')
    dest=ART/exp
    if dest.exists(): raise ValueError('Experiment exists; use a new experiment ID')
    dest.mkdir(); t=time.perf_counter()
    stop=threading.Event(); peak=[psutil.Process().memory_info().rss]
    def monitor():
        while not stop.wait(.1): peak[0]=max(peak[0],psutil.Process().memory_info().rss)
    threading.Thread(target=monitor,daemon=True).start()
    x,y,_=load('train',ctx); vx,vy,ids=load('validation',ctx)
    if set(np.unique(y))!=set(range(5)): raise ValueError('Training population must contain all five classes')
    model=lgb.LGBMClassifier(n_estimators=250,learning_rate=.06,num_leaves=31,max_bin=63,min_child_samples=100,colsample_bytree=.9,reg_lambda=1,n_jobs=2,random_state=20260923,class_weight='balanced' if balanced else None,verbosity=-1,deterministic=True,force_col_wise=True)
    model.fit(x,y)
    pred=model.predict(vx); result=metrics(vy,pred)
    result['cohorts']={c:metrics(vy[np.char.startswith(ids,c)],pred[np.char.startswith(ids,c)]) for c in ['SC','ST']}
    result['subjects']={s:metrics(vy[np.char.startswith(ids,s)],pred[np.char.startswith(ids,s)]) for s in sorted({i[:5] for i in ids})}
    result.update(seconds=time.perf_counter()-t,rss_mb=psutil.Process().memory_info().rss/2**20,balanced=balanced,context=ctx,training_epochs=len(y),feature_count=x.shape[1],parameters=model.get_params(),stage_order=STAGES)
    stop.set();result['sampled_peak_rss_mb']=peak[0]/2**20
    result['majority_baseline']=metrics(vy,np.full(len(vy),int(np.bincount(y).argmax())))
    save(dest/'metrics.json',result)
    provenance=dict(split_sha256=hashlib.sha256((ART/'split.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),dataset_manifest_sha256=hashlib.sha256((DATA/'SHA256SUMS.txt').read_bytes()).hexdigest(),training_subjects=json.loads((ART/'split.json').read_text())['train'])
    save(dest/'provenance.json',provenance)
    joblib.dump(dict(model=model,context=ctx,version=VERSION,stages=STAGES,channels=CHANNELS,experiment=exp,provenance=provenance),dest/'model.joblib')
    pd.DataFrame(dict(recording_id=ids,reference=vy,prediction=pred)).to_csv(dest/'validation_predictions.csv',index=False)
    print(json.dumps({k:result[k] for k in ['macro_f1','per_class_f1','seconds','rss_mb'] }),flush=True)

def predict(modelpath,inputpath,output):
    import joblib
    b=joblib.load(modelpath)
    if b['version']!=VERSION: raise ValueError('Preprocessing version mismatch')
    x,q,start=extract(Path(inputpath)); x=context(x) if b['context'] else x
    prob=b['model'].predict_proba(x)
    if prob.shape!=(len(x),5) or not np.isfinite(prob).all() or not np.allclose(prob.sum(axis=1),1): raise ValueError('Invalid probabilities')
    frame=pd.DataFrame(dict(recording_id=Path(inputpath).stem,epoch_index=np.arange(len(x)),onset=np.arange(len(x))*30,duration=30,predicted_stage=np.array(STAGES)[prob.argmax(axis=1)],quality_flags=q,model_version=b['experiment']))
    for i,s in enumerate(STAGES): frame['P('+s+')']=prob[:,i]
    Path(output).parent.mkdir(parents=True,exist_ok=True); frame.to_csv(output,index=False)
    print(f'{len(frame)} predictions saved to {output}',flush=True)

def predict_input(modelpath,inputpath,output):
    source=Path(inputpath)
    if not source.is_dir(): return predict(modelpath,source,output)
    files=sorted(p for p in source.rglob('*') if p.suffix.lower()=='.edf' and 'hypnogram' not in p.name.lower())
    if not files: raise ValueError('No input EDF recordings found')
    if len({p.stem for p in files})!=len(files): raise ValueError('Duplicate recording basenames in input directory')
    target=Path(output);target.mkdir(parents=True,exist_ok=True);report=[]
    for path in files:
        try:
            predict(modelpath,path,target/(path.stem+'.csv'))
            report.append(dict(file=str(path),status='ok'))
        except (ValueError,OSError) as exc: report.append(dict(file=str(path),status='failed',error=str(exc)))
    save(target/'batch_report.json',report)
    if any(r['status']=='failed' for r in report): raise ValueError('Some EDF inputs failed; inspect batch_report.json')

def sqi(y):
    y=np.asarray(y); sleep=np.flatnonzero(y>0)
    if len(y)==0 or np.any((y<0)|(y>4)): return {'score':None,'reason':'missing_or_invalid_stages'}
    if not len(sleep): return {'score':None,'reason':'no_sleep'}
    a,b=int(sleep[0]),int(sleep[-1])+1; period=y[a:b]; tst=int((y>0).sum())*.5
    return dict(score=float(100*(period>0).mean()),name='Sleep-period continuity index',tst_minutes=tst,observed_latency_minutes=a*.5,waso_minutes=float((period==0).sum()*.5),sleep_period_minutes=len(period)*.5,wake_bouts=int(np.sum((period[1:]==0)&(period[:-1]!=0))),stage_minutes={s:float((y==i).sum()*.5) for i,s in enumerate(STAGES)},interpretation='Descriptive continuity within the supplied window; not PSQI or a validated global sleep-quality score.')

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare'); t=sub.add_parser('train'); t.add_argument('--id',required=True);t.add_argument('--balanced',action='store_true');t.add_argument('--context',action='store_true')
    i=sub.add_parser('predict');i.add_argument('--model',required=True);i.add_argument('--input',required=True);i.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='prepare': prepare()
    elif a.command=='train':
        if (ART/'window_policy.json').exists():
            from policy_improvement import train_selected
            train_selected(a.id,a.balanced,a.context)
        else:train(a.id,a.balanced,a.context)
    else: predict_input(a.model,a.input,a.output)
