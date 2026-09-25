"""Controlled A/B/C: ONLY local training window population changes."""
import sys,json,time,hashlib,inspect,threading,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'research_tools'))
import numpy as np,pandas as pd,joblib,psutil
from sleep_pipeline import ART,ROOT,VERSION,CHANNELS,STAGES,metrics,labels,save,context
from window_policy import WIN,VERSION as WV

def csvpath(policy):return WIN/f'{WV}-{policy}.csv'

def load_population(part,policy,ctx=False):
    split=json.loads((ART/'split.json').read_text())
    assert part!='test','Test performance remains sealed'
    windows=pd.read_csv(csvpath(policy)).set_index('recording_id');xs=[];ys=[];ids=[];epochs=[]
    for path in sorted((ART/'cache').glob('*.npz')):
        if path.name[:5] not in split[part]:continue
        row=windows.loc[path.stem];lo=int(row.window_start_sec)//30;hi=int(row.window_end_sec)//30
        assert row.window_start_sec%30==row.window_end_sec%30==0
        with np.load(path) as z:
            assert str(z['version'])==VERSION
            x=z['x'];y=z['y'];assert 0<=lo<=hi<=len(y)
            mask=y[lo:hi]>=0
            segment=x[lo:hi]
            if ctx and len(segment):segment=context(segment)
            xs.append(segment[mask]);ys.append(y[lo:hi][mask]);ids.extend([path.stem]*int(mask.sum()));epochs.extend((np.arange(lo,hi)[mask]).tolist())
    return np.concatenate(xs),np.concatenate(ys),np.array(ids),np.array(epochs)

def evaluate(y,pred,ids):
    result=metrics(y,pred);result['accuracy_secondary']=float((y==pred).mean())
    result['cohorts']={c:metrics(y[np.char.startswith(ids,c)],pred[np.char.startswith(ids,c)]) for c in ['SC','ST']}
    result['subjects']={s:metrics(y[np.char.startswith(ids,s)],pred[np.char.startswith(ids,s)]) for s in sorted({i[:5] for i in ids})}
    subject_scores=[m['macro_f1'] for m in result['subjects'].values()]
    result['per_subject_macro_f1_quantiles']=dict(zip(['min','p25','median','p75','max'],np.quantile(subject_scores,[0,.25,.5,.75,1]).tolist()))
    return result

def experiment(name,policy):
    import lightgbm as lgb
    dest=ART/name
    if dest.exists():raise ValueError('Experiment ID exists; never overwrite')
    frozen=json.loads((ROOT/'audit'/'window_audit.json').read_text())['frozen_contract']
    assert hashlib.sha256((ART/'split.json').read_bytes()).hexdigest()==frozen['split.json']
    assert hashlib.sha256(inspect.getsource(labels).encode()).hexdigest()==frozen['label_function_sha256']
    assert hashlib.sha256(inspect.getsource(metrics).encode()).hexdigest()==frozen['metric_function_sha256']
    base=joblib.load(ART/'E01'/'model.joblib')
    assert base['context'] is False and base['version']==VERSION
    params=base['model'].get_params();assert params['class_weight'] is None
    dest.mkdir();start=time.perf_counter();stop=threading.Event();peak=[psutil.Process().memory_info().rss]
    def monitor():
        while not stop.wait(.1):peak[0]=max(peak[0],psutil.Process().memory_info().rss)
    threading.Thread(target=monitor,daemon=True).start()
    x,y,_,_=load_population('train',policy)
    if policy=='untrimmed':
        model=base['model'];reused='E01'
        baseline=json.loads((ART/'E01'/'metrics.json').read_text());assert baseline['training_epochs']==len(y)
        assert base['provenance']['split_sha256']==frozen['split.json']
    else:
        model=lgb.LGBMClassifier(**params);model.fit(x,y);reused=None
    result=dict(experiment=name,training_policy=policy,training_epochs=len(y),training_class_counts=np.bincount(y,minlength=5).tolist(),parameters=params,context=False,class_weighting=False,feature_version=VERSION,reused_checkpoint=reused,split_sha256=frozen['split.json'],train_window_sha256=hashlib.sha256(csvpath(policy).read_bytes()).hexdigest(),validation={})
    del x,y
    for validation_policy in ['msp60','naive']:
        vx,vy,ids,epochs=load_population('validation',validation_policy);pred=model.predict(vx)
        result['validation'][validation_policy]=evaluate(vy,pred,ids)
        result['validation'][validation_policy]['window_csv_sha256']=hashlib.sha256(csvpath(validation_policy).read_bytes()).hexdigest()
        pd.DataFrame(dict(recording_id=ids,epoch_index=epochs,reference=vy,prediction=pred)).to_csv(dest/f'predictions_{validation_policy}.csv',index=False)
        del vx,vy,ids,epochs,pred
    stop.set();result.update(seconds=time.perf_counter()-start,sampled_peak_rss_mb=peak[0]/2**20)
    save(dest/'window_metrics.json',result)
    bundle=dict(model=model,context=False,version=VERSION,stages=STAGES,channels=CHANNELS,experiment=name,local_training_window=policy,provenance={k:result[k] for k in ['split_sha256','train_window_sha256','parameters']})
    joblib.dump(bundle,dest/'model.joblib')
    print(json.dumps(dict(experiment=name,training_policy=policy,seconds=result['seconds'],peak_mb=result['sampled_peak_rss_mb'],scores={k:dict(macro_f1=v['macro_f1'],per_class_f1=v['per_class_f1'],SC=v['cohorts']['SC']['macro_f1'],ST=v['cohorts']['ST']['macro_f1']) for k,v in result['validation'].items()}),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--policy',choices=['untrimmed','naive','msp60'],required=True);a=p.parse_args();experiment(a.id,a.policy)
