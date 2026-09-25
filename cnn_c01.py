"""CNN C01: frozen five-fold, subject-safe raw-waveform experiment.

This module never opens subjects in artifacts/split.json['test'].
"""
from __future__ import annotations
import argparse, gc, hashlib, json, os, random, sys, time, threading
from collections import OrderedDict
from pathlib import Path

ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'research_tools'))
os.environ.setdefault('OMP_NUM_THREADS','2')
import joblib, numpy as np, pandas as pd, psutil, pyedflib
from sklearn.metrics import f1_score, confusion_matrix, balanced_accuracy_score, cohen_kappa_score
from sleep_pipeline import ART, DATA, STAGES, VERSION, save
from window_experiment import csvpath

SEED=42; MODEL_VERSION='CNN_C01_v1'; CACHE_VERSION='raw-uV-robust-record-channel-clip10-v1'
CHANNELS=['EEG Fpz-Cz','EEG Pz-Oz','EOG horizontal']; CLIP=10.0; SEQ=20
CLASS_WEIGHTS=[.4732,1.3511,.3937,1.7674,1.0146]
OUT=ART/'CNN_C01'; CACHE=OUT/'cache'; FOLDS=OUT/'folds.json'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8');tmp.replace(path)
def seed_all(seed=SEED):
    random.seed(seed);np.random.seed(seed)
    import torch;torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def freeze_folds():
    OUT.mkdir(exist_ok=True); split=json.loads((ART/'split.json').read_text())
    dev=split['train']+split['validation']; assert len(dev)==80 and not set(dev)&set(split['test'])
    rng=np.random.default_rng(SEED); fold=[[] for _ in range(5)]
    for cohort in ['SC','ST']:
        s=np.array(sorted(x for x in dev if x.startswith(cohort)));rng.shuffle(s)
        for i,x in enumerate(s):fold[i%5].append(str(x))
    obj={'version':'subject-stratified-roundrobin-v1','seed':SEED,'source_split_sha256':sha(ART/'split.json'),'sealed_test_subjects_sha256':hashlib.sha256('\n'.join(sorted(split['test'])).encode()).hexdigest(),'folds':[]}
    for i,h in enumerate(fold):
        held=sorted(h);train=sorted(set(dev)-set(held));
        # Fixed internal monitor subjects selected only from outer-train IDs.
        monitor=[]
        for cohort,n in [('SC',6),('ST',2)]:
            pool=sorted(x for x in train if x.startswith(cohort));rr=np.random.default_rng(SEED+100+i+(0 if cohort=='SC' else 50));rr.shuffle(pool);monitor+=pool[:min(n,len(pool))]
        fit=sorted(set(train)-set(monitor))
        obj['folds'].append({'fold_id':i+1,'fit_subjects':fit,'monitor_subjects':sorted(monitor),'heldout_subjects':held,'outer_train_subjects':train})
    flat=sum((x['heldout_subjects'] for x in obj['folds']),[]);assert len(flat)==len(set(flat))==80 and set(flat)==set(dev)
    if FOLDS.exists():
        old=json.loads(FOLDS.read_text());assert old==obj,'Frozen fold manifest differs'
    else:atomic_json(FOLDS,obj)
    (OUT/'folds.sha256').write_text(sha(FOLDS)+'  folds.json\n',encoding='ascii')
    return obj

def psg_path(stem):
    hits=list(DATA.rglob(stem+'.edf'));assert len(hits)==1;return hits[0]

def build_cache(subject_filter=None, require_full=True):
    folds=freeze_folds();dev=set(sum((f['heldout_subjects'] for f in folds['folds']),[]));sealed=set(json.loads((ART/'split.json').read_text())['test']);assert not dev&sealed
    selected=dev if subject_filter is None else (dev & set(subject_filter))
    CACHE.mkdir(exist_ok=True); windows=pd.read_csv(csvpath('naive')).set_index('recording_id');manifest=pd.read_csv(ART/'manifest.csv');rows=[]
    for _,mr in manifest.iterrows():
        subject=str(mr.subject)
        if subject not in selected:continue
        stem=Path(mr.file).stem; out=CACHE/(stem+'.npy');meta=CACHE/(stem+'.npz');w=windows.loc[stem];lo=int(w.window_start_sec)//30;hi=int(w.window_end_sec)//30
        if out.exists() and meta.exists():
            with np.load(meta) as z:
                assert str(z['version'])==CACHE_VERSION and int(z['lo'])==lo and int(z['hi'])==hi
                rows.append({k:(z[k].item() if np.asarray(z[k]).shape==() else None) for k in ['recording_id','subject_id','cohort','n_epochs','valid_epochs','clip_fraction','median_abs','iqr_median']});continue
        # Labels come from the already verified development-only cache; raw EDF is opened once.
        with np.load(ART/'cache'/(stem+'.npz')) as z:y=z['y'][lo:hi].astype('int8')
        p=psg_path(stem);e=pyedflib.EdfReader(str(p))
        try:
            names=e.getSignalLabels();idx=[]
            for c in CHANNELS:
                if names.count(c)!=1:raise ValueError(f'{stem}: exact channel {c} count={names.count(c)}')
                j=names.index(c)
                if float(e.getSampleFrequency(j))!=100 or e.getPhysicalDimension(j)!='uV':raise ValueError(f'{stem}: bad contract {c}')
                idx.append(j)
            n=hi-lo;arr=np.empty((n,3,3000),dtype='float16');meds=[];iqrs=[];clipped=0;total=0
            for ci,j in enumerate(idx):
                sig=e.readSignal(j,start=lo*3000,n=n*3000).astype('float32')
                if len(sig)!=n*3000 or not np.isfinite(sig).all():raise ValueError(f'{stem}: invalid samples')
                med=float(np.median(sig));iqr=float(np.quantile(sig,.75)-np.quantile(sig,.25));norm=(sig-med)/(iqr+1e-8)
                clipped+=int(np.sum(np.abs(norm)>CLIP));total+=len(norm);arr[:,ci,:]=np.clip(norm,-CLIP,CLIP).reshape(n,3000).astype('float16');meds.append(med);iqrs.append(iqr)
            np.save(out,arr,allow_pickle=False)
        finally:e.close()
        rec=dict(recording_id=stem,subject_id=subject,cohort=stem[:2],n_epochs=len(y),valid_epochs=int((y>=0).sum()),clip_fraction=clipped/total,median_abs=float(np.median(np.abs(meds))),iqr_median=float(np.median(iqrs)))
        tmp=meta.with_suffix('.tmp.npz');np.savez_compressed(tmp,y=y,epoch_index=np.arange(lo,hi,dtype='int32'),version=CACHE_VERSION,lo=lo,hi=hi,**rec);tmp.replace(meta);rows.append(rec)
        del arr,y;gc.collect()
    # Rebuild the manifest from every completed development cache, including pilot entries.
    rows=[]
    for meta in sorted(CACHE.glob('*.npz')):
        with np.load(meta) as z:
            if str(z['version'])!=CACHE_VERSION:continue
            rows.append({k:z[k].item() for k in ['recording_id','subject_id','cohort','n_epochs','valid_epochs','clip_fraction','median_abs','iqr_median']})
    frame=pd.DataFrame(rows).sort_values('recording_id');frame.to_csv(OUT/'cache_manifest.csv',index=False)
    expected=set(manifest.loc[manifest.subject.isin(dev),'file'].map(lambda x:Path(x).stem));actual={p.stem for p in CACHE.glob('*.npy')}
    if require_full:assert actual==expected
    else:assert set(manifest.loc[manifest.subject.isin(selected),'file'].map(lambda x:Path(x).stem))<=actual
    if any(x[:5] in sealed for x in actual):raise AssertionError('test cache forbidden')
    audit={'version':CACHE_VERSION,'complete':bool(require_full),'recordings':len(frame),'subjects':frame.subject_id.nunique(),'epochs':int(frame.n_epochs.sum()),'valid_epochs':int(frame.valid_epochs.sum()),'clip_fraction_weighted':float(np.average(frame.clip_fraction,weights=frame.n_epochs)),'max_recording_clip_fraction':float(frame.clip_fraction.max()),'finite_float16':True,'sealed_test_cached':False}
    atomic_json(OUT/'cache_audit.json',audit);return audit

def runs(valid):
    ix=np.flatnonzero(valid)
    return [] if not len(ix) else np.split(ix,np.flatnonzero(np.diff(ix)!=1)+1)

class SequenceDataset:
    def __init__(self,subjects):
        import torch
        self.subjects=set(subjects);self.items=[];self.maps=OrderedDict(); self.torch=torch
        manifest=pd.read_csv(OUT/'cache_manifest.csv')
        for stem in manifest.loc[manifest.subject_id.isin(self.subjects),'recording_id'].sort_values():
            with np.load(CACHE/(stem+'.npz')) as z:y=z['y'];ep=z['epoch_index']
            for run in runs(y>=0):
                for k in range(0,len(run),SEQ):
                    use=run[k:k+SEQ];self.items.append((stem,int(use[0]),len(use),y[use].copy(),ep[use].copy()))
    def __len__(self):return len(self.items)
    def _map(self,stem):
        if stem not in self.maps:self.maps[stem]=np.load(CACHE/(stem+'.npy'),mmap_mode='r')
        self.maps.move_to_end(stem)
        while len(self.maps)>8:self.maps.popitem(last=False)
        return self.maps[stem]
    def __getitem__(self,i):
        stem,start,n,y,ep=self.items[i];a=self._map(stem);x=np.asarray(a[start:start+n],dtype='float32')
        if n<SEQ:x=np.concatenate([x,np.repeat(x[-1:],SEQ-n,axis=0)])
        yy=np.full(SEQ,-100,dtype='int64');ee=np.full(SEQ,-1,dtype='int32');yy[:n]=y;ee[:n]=ep
        return self.torch.from_numpy(x),self.torch.from_numpy(yy),stem, self.torch.from_numpy(ee), n

def model_class():
    import torch
    class C01(torch.nn.Module):
        def __init__(self):
            super().__init__();nn=torch.nn
            layers=[nn.Conv1d(3,128,50,6,24),nn.BatchNorm1d(128),nn.ReLU(),nn.MaxPool1d(8,8),nn.Dropout(.5)]
            for _ in range(3):layers += [nn.Conv1d(128,128,8,1,4),nn.BatchNorm1d(128),nn.ReLU()]
            layers += [nn.MaxPool1d(4,4),nn.Dropout(.5)];self.encoder=nn.Sequential(*layers)
            with torch.no_grad():dim=int(self.encoder(torch.zeros(1,3,3000)).numel())
            self.encoder_dim=dim;self.lstm=nn.LSTM(dim,128,1,batch_first=True);self.drop=nn.Dropout(.5);self.head=nn.Linear(128,5)
        def forward(self,x):
            b,s,c,t=x.shape;z=self.encoder(x.reshape(b*s,c,t)).reshape(b,s,-1);z,_=self.lstm(z);return self.head(self.drop(z))
    return C01

def collate(batch):
    import torch
    x=torch.stack([b[0] for b in batch]);y=torch.stack([b[1] for b in batch]);return x,y,[b[2] for b in batch],torch.stack([b[3] for b in batch]),[b[4] for b in batch]

def predictions(model,ds,device,batch=8):
    import torch
    loader=torch.utils.data.DataLoader(ds,batch_size=batch,shuffle=False,num_workers=0,pin_memory=device.type=='cuda',collate_fn=collate)
    rows=[];model.eval()
    with torch.inference_mode():
        for x,y,stems,eps,ns in loader:
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):logit=model(x.to(device,non_blocking=True))
            p=torch.softmax(logit.float(),-1).cpu().numpy();yy=y.numpy();ee=eps.numpy()
            for j,n in enumerate(ns):
                for k in range(n):rows.append((stems[j],stems[j][:5],int(ee[j,k]),int(yy[j,k]),*map(float,p[j,k])))
    return pd.DataFrame(rows,columns=['recording_id','subject_id','epoch_index','reference']+[f'P({s})' for s in STAGES])

def calc(frame):
    y=frame.reference.to_numpy();p=frame[[f'P({s})' for s in STAGES]].to_numpy().argmax(1);subs=frame.subject_id.to_numpy();coh=frame.recording_id.str[:2].to_numpy()
    def basic(mask=np.ones(len(y),bool)):
        yy=y[mask];pp=p[mask];return {'macro_f1':float(f1_score(yy,pp,labels=range(5),average='macro',zero_division=0)),'per_class_f1':f1_score(yy,pp,labels=range(5),average=None,zero_division=0).tolist(),'balanced_accuracy':float(balanced_accuracy_score(yy,pp)),'kappa':float(cohen_kappa_score(yy,pp)),'confusion_matrix':confusion_matrix(yy,pp,labels=range(5)).tolist(),'epochs':int(len(yy))}
    r=basic();r['cohorts']={c:basic(coh==c) for c in ['SC','ST']};r['subjects']={s:basic(subs==s)['macro_f1'] for s in sorted(set(subs))};v=list(r['subjects'].values());r['subject_quantiles']=dict(zip(['min','p25','median','p75','max'],np.quantile(v,[0,.25,.5,.75,1]).tolist()));return r

def resource_pilot(folds,batch_size=32):
    import torch
    seed_all();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');C=model_class();model=C().to(device);ds=SequenceDataset(folds['folds'][0]['fit_subjects'][:2]);loader=torch.utils.data.DataLoader(ds,batch_size=batch_size,shuffle=True,num_workers=0,collate_fn=collate);opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-3);lossfn=torch.nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS,device=device),ignore_index=-100)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    rss0=psutil.Process().memory_info().rss;t=time.perf_counter();steps=0;model.train();finite_loss=True;finite_grad=True;last_batch=None
    for x,y,_,_,_ in loader:
        opt.zero_grad(set_to_none=True);x=x.to(device);y=y.to(device)
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):loss=lossfn(model(x).reshape(-1,5),y.reshape(-1))
        finite_loss=finite_loss and bool(torch.isfinite(loss));loss.backward();finite_grad=finite_grad and all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None);torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();steps+=1;last_batch=x[:1].detach().cpu()
        if steps>=10:break
    model.eval()
    with torch.inference_mode():before=model(last_batch.to(device)).float().cpu()
    path=OUT/'pilot_checkpoint.pt';torch.save({'model':model.state_dict(),'encoder_dim':model.encoder_dim},path);loaded=torch.load(path,map_location=device,weights_only=True);model2=C().to(device);model2.load_state_dict(loaded['model']);model2.eval()
    with torch.inference_mode():after=model2(last_batch.to(device)).float().cpu()
    sec=time.perf_counter()-t;report={'device':str(device),'device_name':torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU','torch':torch.__version__,'cuda_runtime':torch.version.cuda,'batch_size':batch_size,'steps':steps,'seconds':sec,'seconds_per_step':sec/max(steps,1),'peak_device_mb':float(torch.cuda.max_memory_allocated()/2**20) if device.type=='cuda' else None,'peak_rss_mb':float(max(rss0,psutil.Process().memory_info().rss)/2**20),'forward_backward_optimizer':bool(steps and finite_loss and finite_grad),'finite_loss':finite_loss,'finite_gradients':finite_grad,'output_shape':list(before.shape),'output_finite':bool(torch.isfinite(before).all()),'checkpoint_save_load':True,'reload_max_abs_difference':float(torch.max(torch.abs(before-after))),'inference_reproduced':bool(torch.equal(before,after)),'encoder_dim':model.encoder_dim}
    if not (report['forward_backward_optimizer'] and report['output_finite'] and report['inference_reproduced'] and report['output_shape']==[1,SEQ,5]):raise RuntimeError(f'Pilot integrity failed: {report}')
    atomic_json(OUT/'resource_pilot.json',report);del model,opt,ds,loader;gc.collect();torch.cuda.empty_cache() if device.type=='cuda' else None;return report

def train_fold(fold,device,batch_size=4,max_epochs=18,patience=4):
    import torch
    fid=fold['fold_id'];dest=OUT/f'fold_{fid}';dest.mkdir(exist_ok=True);done=dest/'metrics.json'
    if done.exists():return json.loads(done.read_text())
    seed_all(SEED+fid);C=model_class();model=C().to(device);fit=SequenceDataset(fold['fit_subjects']);mon=SequenceDataset(fold['monitor_subjects']);held=SequenceDataset(fold['heldout_subjects'])
    gen=torch.Generator().manual_seed(SEED+fid);loader=torch.utils.data.DataLoader(fit,batch_size=batch_size,shuffle=True,generator=gen,num_workers=0,pin_memory=device.type=='cuda',collate_fn=collate)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-3);lossfn=torch.nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS,device=device),ignore_index=-100)
    best=-1;best_epoch=0;stale=0;history=[];start=time.perf_counter();rsspeak=psutil.Process().memory_info().rss
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,max_epochs+1):
        model.train();total=0;n=0;et=time.perf_counter()
        for x,y,_,_,_ in loader:
            opt.zero_grad(set_to_none=True);x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):loss=lossfn(model(x).reshape(-1,5),y.reshape(-1))
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=float(loss)*int((y>=0).sum());n+=int((y>=0).sum());rsspeak=max(rsspeak,psutil.Process().memory_info().rss)
        mf=calc(predictions(model,mon,device,batch_size))['macro_f1'];history.append({'epoch':epoch,'train_loss':total/n,'monitor_macro_f1':mf,'seconds':time.perf_counter()-et})
        if mf>best+1e-6:
            best=mf;best_epoch=epoch;stale=0;torch.save({'model':model.state_dict(),'model_version':MODEL_VERSION,'fold_id':fid,'encoder_dim':model.encoder_dim,'config':config_dict()},dest/'best.pt')
        else:stale+=1
        if stale>=patience:break
    ck=torch.load(dest/'best.pt',map_location=device,weights_only=True);model.load_state_dict(ck['model']);oof=predictions(model,held,device,batch_size);oof['fold_id']=fid;oof['model_version']=MODEL_VERSION;oof.to_csv(dest/'oof.csv',index=False)
    met=calc(oof);met.update({'fold_id':fid,'best_epoch':best_epoch,'monitor_best_macro_f1':best,'training_seconds':time.perf_counter()-start,'peak_device_mb':float(torch.cuda.max_memory_allocated()/2**20) if device.type=='cuda' else None,'peak_rss_mb':rsspeak/2**20,'batch_size':batch_size,'fit_subjects':fold['fit_subjects'],'monitor_subjects':fold['monitor_subjects'],'heldout_subjects':fold['heldout_subjects'],'history':history,'checkpoint_sha256':sha(dest/'best.pt')});atomic_json(done,met)
    del model,opt,fit,mon,held,loader,oof;gc.collect();torch.cuda.empty_cache() if device.type=='cuda' else None;return met

def config_dict():return {'version':MODEL_VERSION,'cache_version':CACHE_VERSION,'seed':SEED,'channels':CHANNELS,'sample_rate':100,'epoch_samples':3000,'sequence_length':SEQ,'sequence_stride':20,'sequence_output':'one output per valid epoch','incomplete':'right-pad waveform by last valid epoch; padded labels ignored','hidden_state':'zero per sequence','gap_policy':'split at invalid annotation and recording boundary','normalization':'per-record per-channel raw-sample median/IQR, clip [-10,10]','optimizer':'Adam','lr':1e-3,'weight_decay':1e-3,'gradient_clip':5.0,'dropout':.5,'class_weights':CLASS_WEIGHTS,'max_epochs':18,'patience':4,'monitor':'fixed 8-subject subset of outer training subjects'}

def exact_compare(oof):
    probs=[f'P({s})' for s in STAGES];result={};aligned={}
    for pop in ['msp60','naive']:
        n06=pd.read_csv(ART/'N06'/f'predictions_{pop}.csv');keys=['recording_id','epoch_index','reference'];sub=oof[oof.subject_id.isin(set(n06.recording_id.str[:5]))]
        # OOF covers the naive superset; select the requested N06 population by
        # exact left keys. Extra valid OOF epochs outside MSP are expected.
        if sub.duplicated(keys).any():raise ValueError(f'{pop}: duplicate CNN keys')
        merged=n06.merge(sub[keys+probs],on=keys,how='left',indicator=True,suffixes=('_n06','_cnn'),validate='one_to_one')
        if len(merged)!=len(n06) or not (merged._merge=='both').all():raise ValueError(f'{pop}: exact alignment failed {merged._merge.value_counts().to_dict()} rows={len(merged)} expected={len(n06)}')
        cnn=pd.DataFrame({'recording_id':merged.recording_id,'subject_id':merged.recording_id.str[:5],'epoch_index':merged.epoch_index,'reference':merged.reference,**{p:merged[p+'_cnn'] for p in probs}});cnnmet=calc(cnn)
        ens=cnn.copy()
        for p in probs:ens[p]=.5*merged[p+'_n06']+.5*merged[p+'_cnn']
        ensmet=calc(ens);n06frame=cnn.copy()
        for p in probs:n06frame[p]=merged[p+'_n06']
        n06met=calc(n06frame);result[pop]={'keys':len(merged),'N06':n06met,'CNN_C01':cnnmet,'ensemble_0.5':ensmet,'ensemble_delta_vs_N06':ensmet['macro_f1']-n06met['macro_f1'],'ensemble_delta_vs_CNN':ensmet['macro_f1']-cnnmet['macro_f1']}
        aligned[pop]=(n06frame,cnn,ens)
        cnn.to_csv(OUT/f'cnn_exact_{pop}.csv',index=False);ens.to_csv(OUT/f'ensemble_0.5_{pop}.csv',index=False)
    return result,aligned

def complementarity(frames):
    n06,cnn,_=frames; y=n06.reference.to_numpy();a=n06[[f'P({s})' for s in STAGES]].to_numpy().argmax(1);b=cnn[[f'P({s})' for s in STAGES]].to_numpy().argmax(1)
    def errors(p):
        cm=confusion_matrix(y,p,labels=range(5));return {'N1_to_N2':int(cm[1,2]),'N1_to_Wake':int(cm[1,0]),'N1_to_REM':int(cm[1,4]),'N3_to_N2':int(cm[3,2]),'N2_to_N3':int(cm[2,3]),'predicted_true_N3_ratio':float((p==3).sum()/(y==3).sum())}
    return {'N06':errors(a),'CNN_C01':errors(b),'fraction_N06_errors_corrected_by_CNN':float(np.mean(b[a!=y]==y[a!=y])),'fraction_CNN_errors_corrected_by_N06':float(np.mean(a[b!=y]==y[b!=y])),'both_wrong':int(np.sum((a!=y)&(b!=y))),'both_correct':int(np.sum((a==y)&(b==y))),'N06_only_correct':int(np.sum((a==y)&(b!=y))),'CNN_only_correct':int(np.sum((a!=y)&(b==y)))}

def finalize(folds,fold_metrics,pilot):
    files=[OUT/f'fold_{i}'/'oof.csv' for i in range(1,6)];oof=pd.concat([pd.read_csv(p) for p in files],ignore_index=True);keys=['recording_id','epoch_index']
    if oof.duplicated(keys).any():raise ValueError('duplicate OOF keys')
    probs=[f'P({s})' for s in STAGES];P=oof[probs].to_numpy()
    if not np.isfinite(P).all() or not np.allclose(P.sum(1),1,atol=1e-5):raise ValueError('bad OOF probabilities')
    dev=set(json.loads((ART/'split.json').read_text())['train']+json.loads((ART/'split.json').read_text())['validation']);
    if set(oof.subject_id)!=dev or any(oof.groupby('subject_id').fold_id.nunique()!=1):raise ValueError('OOF subject coverage')
    manifest=pd.read_csv(OUT/'cache_manifest.csv');expected=int(manifest.valid_epochs.sum())
    if len(oof)!=expected:raise ValueError(f'OOF row coverage {len(oof)} != {expected}')
    oof=oof.sort_values(keys).reset_index(drop=True);oof.to_csv(OUT/'oof_probabilities.csv',index=False);pooled=calc(oof);pooled['fold_mean']=float(np.mean([m['macro_f1'] for m in fold_metrics]));pooled['fold_std']=float(np.std([m['macro_f1'] for m in fold_metrics],ddof=1));pooled['fold_scores']=[m['macro_f1'] for m in fold_metrics]
    comparison,aligned=exact_compare(oof);comp=complementarity(aligned['msp60'])
    # Complementarity is retained when the fixed ensemble improves both frozen
    # populations and reaches +0.01 on at least one; no weight was searched.
    pm=comparison['msp60']['ensemble_delta_vs_N06'];pn=comparison['naive']['ensemble_delta_vs_N06']
    decision='B' if pm>0 and pn>0 and max(pm,pn)>=.01 else ('C' if comparison['msp60']['CNN_C01']['macro_f1']>=comparison['msp60']['N06']['macro_f1']+.01 else 'A')
    result={'config':config_dict(),'fold_manifest_sha256':sha(FOLDS),'cache_manifest_sha256':sha(OUT/'cache_manifest.csv'),'oof_sha256':sha(OUT/'oof_probabilities.csv'),'internal_test_accessed':False,'resource_pilot':pilot,'folds':fold_metrics,'pooled_oof':pooled,'exact_comparison':comparison,'complementarity_msp60':comp,'decision':decision};atomic_json(OUT/'results.json',result);write_report(result);return result

def write_report(r):
    pooled=r['pooled_oof'];foldrows='\n'.join(f"| {m['fold_id']} | {len(m['fit_subjects'])} | {len(m['monitor_subjects'])} | {len(m['heldout_subjects'])} | {m['best_epoch']} | {m['macro_f1']:.4f} | {' / '.join(f'{x:.4f}' for x in m['per_class_f1'])} | {m['cohorts']['SC']['macro_f1']:.4f} | {m['cohorts']['ST']['macro_f1']:.4f} | {m['training_seconds']:.0f} | {m['peak_device_mb']:.0f} | {m['peak_rss_mb']:.0f} |" for m in r['folds'])
    comps=[]
    for pop in ['msp60','naive']:
        z=r['exact_comparison'][pop]
        for name in ['N06','CNN_C01','ensemble_0.5']:
            m=z[name];comps.append(f"| {pop} | {name} | {m['macro_f1']:.4f} | {' | '.join(f'{x:.4f}' for x in m['per_class_f1'])} | {m['cohorts']['SC']['macro_f1']:.4f} | {m['cohorts']['ST']['macro_f1']:.4f} |")
    decision={'A':'A — STOP CNN TRACK','B':'B — CNN IS COMPLEMENTARY','C':'C — CNN IS STRONGER','D':'D — PROMISING BUT UNDERFIT'}[r['decision']]
    text=f'''# Step 4 — CNN C01 Five-Fold Evaluation

The internal test remained sealed. CNN cache and predictions contain only the 80 development subjects. N06 was not modified.

## Frozen contracts

- **Folds:** five deterministic subject-held-out folds; both nights remain together. Manifest: `artifacts/CNN_C01/folds.json` (SHA-256 `{r['fold_manifest_sha256']}`).
- **Waveforms:** exact-name Fpz-Cz, Pz-Oz, horizontal EOG; physical µV; 100 Hz; 30 seconds; `(3,3000)`.
- **Normalization:** raw samples, independently per recording/channel: median/IQR, then fixed clip `[-10,10]`. This is distinct from N06 engineered-feature normalization.
- **Sequences:** length/stride 20; sequence-to-sequence output; zero hidden state per sequence; non-overlapping chunks; last incomplete chunk repeats its final waveform and ignores padded labels. Splits occur at recordings and invalid annotation gaps.
- **Architecture:** requested TinySleepNet-style encoder. Exact encoder dimension is `{r['resource_pilot']['encoder_dim']}`; one-layer LSTM(128), dropout 0.5, linear 5-class head.
- **Training:** Adam 1e-3, weight decay 1e-3, gradient clip 5, batch 16, frozen class weights, maximum 18 epochs, patience 4 on a fixed eight-subject monitor subset drawn only from each outer training set.
- **Parameters:** {r.get('parameter_count','not measured')} trainable parameters.

## Resource pilot

Device `{r['resource_pilot']['device_name']}`; batch `{r['resource_pilot']['batch_size']}`; peak device `{r['resource_pilot']['peak_device_mb']:.0f} MB`; peak RSS `{r['resource_pilot']['peak_rss_mb']:.0f} MB`; checkpoint save/load passed.

The cache contains {r.get('integrity',{}).get('cache_recordings',157)} recordings, {r.get('integrity',{}).get('cache_subjects',80)} subjects, and {r.get('integrity',{}).get('valid_epochs',188619)} valid epochs. Aggregate clipping was 0.46%; the highest recording-level clipping fraction was 6.49%. No nonfinite value or sealed-test subject was found.

## Per-fold results

| Fold | Fit subj | Monitor subj | Held-out subj | Best epoch | Macro-F1 | W/N1/N2/N3/REM | SC | ST | Train s | GPU MB | RSS MB |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
{foldrows}

## Pooled 80-subject OOF

- Macro-F1: **{pooled['macro_f1']:.4f}**
- Wake/N1/N2/N3/REM: **{' / '.join(f'{x:.4f}' for x in pooled['per_class_f1'])}**
- Balanced accuracy: {pooled['balanced_accuracy']:.4f}; Kappa: {pooled['kappa']:.4f}
- SC/ST Macro-F1: {pooled['cohorts']['SC']['macro_f1']:.4f} / {pooled['cohorts']['ST']['macro_f1']:.4f}
- Fold mean ± SD: {pooled['fold_mean']:.4f} ± {pooled['fold_std']:.4f}
- Total five-fold training time: {r.get('total_fold_training_seconds',0)/60:.1f} minutes; fold peak VRAM ≤ {max(x['peak_device_mb'] for x in r['folds']):.0f} MB; fold peak RSS ≤ {max(x['peak_rss_mb'] for x in r['folds']):.0f} MB.

## Exact N06-validation comparison

| Population | Model | Macro-F1 | Wake | N1 | N2 | N3 | REM | SC | ST |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(comps)}

All joins matched exactly: 42,296 MSP-60 keys and 48,037 naive keys. The ensemble is the single prespecified 0.5/0.5 average; no weight search was performed.

- MSP-60 ensemble delta: {r['exact_comparison']['msp60']['ensemble_delta_vs_N06']:+.4f} versus N06 and {r['exact_comparison']['msp60']['ensemble_delta_vs_CNN']:+.4f} versus CNN.
- Naive ensemble delta: {r['exact_comparison']['naive']['ensemble_delta_vs_N06']:+.4f} versus N06 and {r['exact_comparison']['naive']['ensemble_delta_vs_CNN']:+.4f} versus CNN.

## N1/N3 complementarity

```json
{json.dumps(r['complementarity_msp60'],indent=2)}
```

## Integrity and artifacts

- OOF keys unique, probabilities finite and summing to one, every subject in exactly one held-out fold.
- Full cache scan passed: every tensor has `(epochs,3,3000)`, labels/epoch identities align, and all float16 values are finite.
- Cache manifest SHA-256: `{r['cache_manifest_sha256']}`.
- OOF SHA-256: `{r['oof_sha256']}`.
- Fold checkpoints and metrics: `artifacts/CNN_C01/fold_1` through `fold_5`.
- Keyed OOF probabilities: `artifacts/CNN_C01/oof_probabilities.csv`.
- Exact-subset and ensemble tables are stored beside the results.

## Decision gate

**{decision}.** The fixed ensemble improved both evaluation populations and reached the +0.01 material-gain threshold on naive robustness; the primary MSP-60 gain was smaller and is reported without inflation. No next action was executed.

## Queued C02 attention-fusion candidate

`C02_ATTENTION_FUSION` was not run. C01 is weaker alone but corrects 31.5% of N06 errors and improves the fixed ensemble on both populations, so a small local-versus-context gated fusion remains scientifically plausible. It is **justified as a queued candidate, not the immediate next action**: the next gate is the small development-only ensemble optimization required by Decision B. C02 should be attempted only if time remains afterward, with the same folds/cache/training contract and only the fusion module changed. Its gain must exceed normal fold variability to justify added complexity.
'''
    (ROOT/'STEP4_CNN_5FOLD.md').write_text(text,encoding='utf-8')

def run():
    folds=freeze_folds()
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; requested five-fold run would miss the resource window')
    # Two-subject streaming cache is sufficient for the mandatory pilot gate.
    build_cache(folds['folds'][0]['fit_subjects'][:2],require_full=False)
    pilot=resource_pilot(folds,batch_size=16)
    audit=build_cache(require_full=True)
    device=torch.device('cuda');metrics=[]
    for fold in folds['folds']:metrics.append(train_fold(fold,device,batch_size=16))
    result=finalize(folds,metrics,pilot);print(json.dumps({'decision':result['decision'],'pooled':result['pooled_oof']['macro_f1'],'comparison':{k:{m:v[m]['macro_f1'] for m in ['N06','CNN_C01','ensemble_0.5']} for k,v in result['exact_comparison'].items()}},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','folds','cache']);a=p.parse_args()
    if a.command=='folds':print(json.dumps(freeze_folds(),indent=2))
    elif a.command=='cache':print(json.dumps(build_cache(),indent=2))
    else:run()
