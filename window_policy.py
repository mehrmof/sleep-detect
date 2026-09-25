"""Label-derived LOCAL window audit. Never imported by judging inference."""
import sys,json,hashlib
from pathlib import Path
from datetime import timedelta
sys.path.insert(0,str(Path(__file__).parent/'research_tools'))
import numpy as np,pandas as pd,pyedflib
from sleep_pipeline import ROOT,DATA,ART,STAGES,labels,save
OUT=ROOT/'audit';WIN=OUT/'windows';VERSION='window-v1'

def bouts(mask):
    edge=np.diff(np.r_[False,mask,False].astype('int8'))
    return list(zip(np.flatnonzero(edge==1).tolist(),np.flatnonzero(edge==-1).tolist()))

def bounds(y,policy):
    if policy=='untrimmed':return 0,len(y)
    sleep=np.flatnonzero(y>0)
    if not len(sleep):return 0,0
    if policy.startswith('msp'):
        threshold=int(policy[3:])*2
        cuts=sorted({int(np.searchsorted(sleep,b)) for a,b in bouts(y==0) if b-a>threshold})
        cuts=[c for c in cuts if 0<c<len(sleep)]
        episodes=np.split(sleep,cuts)
        # Largest number of actual sleep epochs; earliest start breaks ties.
        sleep=max(episodes,key=lambda e:(len(e),-int(e[0])))
    elif policy!='naive': raise ValueError(policy)
    return max(0,int(sleep[0])-60),min(len(y),int(sleep[-1])+61)

def describe(y,lo,hi,start):
    z=y[lo:hi]; w=bouts(z==0)
    longest=max(w,key=lambda ab:ab[1]-ab[0]) if w else None
    counts=[int((z==i).sum()) for i in range(5)]
    return dict(window_start_sec=lo*30,window_end_sec=hi*30,duration_hours=(hi-lo)/120,
                wake_epochs=counts[0],sleep_epochs=sum(counts[1:]),
                longest_wake_minutes=(longest[1]-longest[0])*.5 if longest else 0,
                longest_wake_start_sec=(lo+longest[0])*30 if longest else None,
                longest_wake_start_clock=(start+timedelta(seconds=(lo+longest[0])*30)).isoformat() if longest else None,
                wake_bouts_over_60_minutes=sum(b-a>120 for a,b in w),
                invalid_epochs=int((z<0).sum()),**{s:counts[i] for i,s in enumerate(STAGES)})

def audit():
    WIN.mkdir(parents=True,exist_ok=True);rows=[];episodes=[];policies=['untrimmed','naive','msp30','msp60','msp120']
    frozen={k:hashlib.sha256((ART/k).read_bytes()).hexdigest() for k in ['split.json']}
    import inspect
    from sleep_pipeline import metrics
    frozen.update(label_function_sha256=hashlib.sha256(inspect.getsource(labels).encode()).hexdigest(),metric_function_sha256=hashlib.sha256(inspect.getsource(metrics).encode()).hexdigest())
    for path in sorted(DATA.rglob('*-PSG.edf')):
        with pyedflib.EdfReader(str(path)) as reader:
            n=int(reader.getFileDuration()//30);start=reader.getStartdatetime()
        hyp=list(path.parent.glob(path.name[:6]+'*Hypnogram.edf'));assert len(hyp)==1
        y=labels(hyp[0],n)
        for policy in policies:
            lo,hi=bounds(y,policy); row=dict(recording_id=path.stem,cohort=path.name[:2],policy=policy,rule_version=f'{VERSION}-{policy}',**describe(y,lo,hi,start))
            row['flagged']=row['duration_hours']>12 or row['longest_wake_minutes']>90
            rows.append(row)
        # Explicit sequence evidence: sleep before/after each internal long Wake.
        for a,b in bouts(y==0):
            if b-a>120 and (y[:a]>0).any() and (y[b:]>0).any():
                episodes.append(dict(recording_id=path.stem,cohort=path.name[:2],wake_start_sec=a*30,wake_end_sec=b*30,wake_minutes=(b-a)*.5,wake_start_clock=(start+timedelta(seconds=a*30)).isoformat(),sleep_minutes_before=float((y[:a]>0).sum()*.5),sleep_minutes_after=float((y[b:]>0).sum()*.5)))
    frame=pd.DataFrame(rows); frame.to_csv(OUT/'window_per_recording.csv',index=False)
    frame[frame.flagged].to_csv(OUT/'window_flagged.csv',index=False)
    pd.DataFrame(episodes).to_csv(OUT/'long_wake_episode_evidence.csv',index=False)
    for policy in policies:
        dest=WIN/f'{VERSION}-{policy}.csv';content=frame.loc[frame.policy==policy,['recording_id','window_start_sec','window_end_sec','rule_version']].to_csv(index=False)
        if dest.exists() and dest.read_text()!=content:raise ValueError(f'Immutable window version differs: {dest}')
        dest.write_text(content,encoding='utf-8')
    dist=[];duration=[];sensitivity=[]
    for policy in policies:
        for cohort in ['SC','ST','combined']:
            z=frame[(frame.policy==policy)&(True if cohort=='combined' else frame.cohort==cohort)]
            counts=z[STAGES].sum();total=int(counts.sum())
            dist.append(dict(policy=policy,cohort=cohort,total_epochs=total,**{s:int(counts[s]) for s in STAGES},**{s+'_pct':float(100*counts[s]/total) for s in STAGES}))
            q=np.quantile(z.duration_hours,[0,.25,.5,.75,.9,1])
            duration.append(dict(policy=policy,cohort=cohort,recordings=len(z),**dict(zip(['min','p25','median','p75','p90','max'],q.tolist())),in_7_to_10_hours=int(z.duration_hours.between(7,10).sum()),flagged=int(z.flagged.sum())))
    naive=frame[frame.policy=='naive'].set_index('recording_id')
    for policy in ['msp30','msp60','msp120']:
        z=frame[frame.policy==policy].set_index('recording_id');changed=(z.window_start_sec!=naive.window_start_sec)|(z.window_end_sec!=naive.window_end_sec)
        for cohort in ['SC','ST','combined']:
            ids=z.index if cohort=='combined' else z.index[z.cohort==cohort]
            removed=naive.loc[ids,STAGES].sum()-z.loc[ids,STAGES].sum()
            sensitivity.append(dict(policy=policy,cohort=cohort,different_recordings=int(changed.loc[ids].sum()),removed_sleep_epochs=int(removed.iloc[1:].sum()),**{s+'_removed':int(removed[s]) for s in STAGES}))
    pd.DataFrame(dist).to_csv(OUT/'window_class_distributions.csv',index=False)
    pd.DataFrame(duration).to_csv(OUT/'window_duration_distributions.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/'window_threshold_sensitivity.csv',index=False)
    expected_full=[289856,25175,88983,19454,34184];expected_naive=[70140,25175,88983,19454,34184]
    measured={p:[next(r for r in dist if r['policy']==p and r['cohort']=='combined')[s] for s in STAGES] for p in ['untrimmed','naive']}
    checks=dict(untrimmed_matches_supplied_counts=measured['untrimmed']==expected_full,naive_matches_supplied_counts=measured['naive']==expected_naive,measured=measured)
    differences={}
    for left,right in [('msp30','msp60'),('msp60','msp120')]:
        a=frame[frame.policy==left].set_index('recording_id');b=frame[frame.policy==right].set_index('recording_id')
        differences[left+'_vs_'+right]=a.index[(a.window_start_sec!=b.window_start_sec)|(a.window_end_sec!=b.window_end_sec)].tolist()
    save(OUT/'window_audit.json',dict(version=VERSION,frozen_contract=frozen,count_checks=checks,class_distributions=dist,duration_distributions=duration,sensitivity=sensitivity,threshold_boundary_differences=differences,nap_interpretation='Measured separated sleep episodes support nap-like patterns; labels/clock alone cannot prove daytime naps or intended overnight opportunity.'))
    print(json.dumps(dict(checks=checks,combined_distributions=[r for r in dist if r['cohort']=='combined'],combined_sensitivity=[r for r in sensitivity if r['cohort']=='combined'],duration=[r for r in duration if r['policy'] in ['naive','msp60']]),indent=2))

if __name__=='__main__':audit()
