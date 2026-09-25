"""Controlled N03-N08 experiments and final window-policy re-run.

The internal test partition is deliberately inaccessible in this module.
"""
from __future__ import annotations
import argparse, hashlib, inspect, json, os, re, sys, threading, time
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "research_tools"))
os.environ.setdefault("OMP_NUM_THREADS", "2")

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy.signal import butter, sosfiltfilt

from sleep_pipeline import ART, CHANNELS, DATA, STAGES, VERSION, extract, labels, metrics, save
from window_experiment import csvpath, evaluate

SEED = 20260923
PRIMARY = "msp60"
ROBUSTNESS = "naive"
BAND_BASE = np.array([c * 20 + j for c in range(3) for j in range(12)])


@dataclass(frozen=True)
class Config:
    representation: str = "raw"       # raw | norm | raw_norm
    context_radius: int = 1
    rolling_radius: int = 0
    balanced: bool = False
    emg: bool = False
    metadata: bool = False
    viterbi: bool = False


def robust_normalize(x: np.ndarray) -> np.ndarray:
    """Per-recording robust scaling, with no learned/external state."""
    med = np.median(x, axis=0)
    iqr = np.quantile(x, .75, axis=0) - np.quantile(x, .25, axis=0)
    return np.clip((x - med) / (iqr + 1e-8), -20, 20).astype("float32")


def _base(x: np.ndarray, representation: str) -> tuple[np.ndarray, np.ndarray]:
    n = robust_normalize(x)
    if representation == "raw": return x.astype("float32", copy=False), BAND_BASE
    if representation == "norm": return n, BAND_BASE
    if representation == "raw_norm":
        return np.concatenate([x, n], axis=1), np.r_[BAND_BASE, BAND_BASE + x.shape[1]]
    raise ValueError(representation)


def _runs(valid: np.ndarray) -> list[np.ndarray]:
    idx = np.flatnonzero(valid)
    return [] if not len(idx) else list(np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1))


def _shift_edge(a: np.ndarray, offset: int) -> np.ndarray:
    ix = np.clip(np.arange(len(a)) + offset, 0, len(a) - 1)
    return a[ix]


def _rolling(a: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    padded = np.pad(a, ((radius, radius), (0, 0)), mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1, axis=0)
    return view.mean(axis=2).astype("float32"), view.std(axis=2).astype("float32")


def transform_record(raw: np.ndarray, valid: np.ndarray, cfg: Config,
                     emg: np.ndarray | None = None,
                     metadata: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Build model input. N04+ never crosses invalid annotation gaps."""
    base, band_idx = _base(raw, cfg.representation)
    if cfg.emg:
        if emg is None or len(emg) != len(raw): raise ValueError("EMG length mismatch")
        en = robust_normalize(emg)
        if cfg.representation == "raw": eb = emg
        elif cfg.representation == "norm": eb = en
        else: eb = np.concatenate([emg, en], axis=1)
        base = np.concatenate([base, eb.astype("float32")], axis=1)
    # N02/N03 exact context behavior. N04+ splits on annotation gaps.
    strict = cfg.context_radius >= 2 or cfg.rolling_radius > 0
    out_dim = base.shape[1] * (2 * cfg.context_radius + 1)
    out_dim += (2 * len(band_idx) + 1) if cfg.rolling_radius else 0
    out_dim += 2 if cfg.metadata else 0
    out = np.zeros((len(base), out_dim), dtype="float32")
    truncated = np.zeros(len(base), dtype="uint8")
    groups = _runs(valid) if strict else [np.arange(len(base))]
    for run in groups:
        if not len(run): continue
        z = base[run]
        parts = [z] + [_shift_edge(z, -k) for k in range(1, cfg.context_radius + 1)] + [_shift_edge(z, k) for k in range(1, cfg.context_radius + 1)]
        if cfg.rolling_radius:
            mean, std = _rolling(z[:, band_idx], cfg.rolling_radius)
            trunc = ((np.arange(len(z)) < max(cfg.context_radius, cfg.rolling_radius)) |
                     (np.arange(len(z)) >= len(z) - max(cfg.context_radius, cfg.rolling_radius))).astype("float32")
            truncated[run] = trunc.astype("uint8")
            parts += [mean, std, trunc[:, None]]
        row = np.concatenate(parts, axis=1)
        if cfg.metadata:
            if metadata is None: raise ValueError("Metadata requested but unavailable")
            row = np.concatenate([row, np.tile(np.asarray(metadata, dtype="float32"), (len(row), 1))], axis=1)
        out[run] = row
    return out, truncated


def psg_path(stem: str) -> Path:
    hits = list(DATA.rglob(stem + ".edf"))
    if len(hits) != 1: raise ValueError(f"PSG lookup failed for {stem}: {len(hits)}")
    return hits[0]


def parse_edf_metadata(path: Path) -> tuple[float, float] | None:
    with path.open("rb") as f:
        h = f.read(88)
    patient = h[8:88].decode("latin-1", "replace").strip()
    age = re.search(r"(?<!\d)(\d{1,3})\s*(?:yr|year)", patient, re.I)
    sex = re.search(r"\b(Female|Male|F|M)\b", patient, re.I)
    if not age or not sex: return None
    years = int(age.group(1)); token = sex.group(1).lower()
    if not 0 < years < 120: return None
    return float(years), float(token.startswith("f"))


def _emg_cache_path(stem: str) -> Path:
    return ART / "emg_cache" / f"{stem}.npz"


def emg_features(stem: str) -> np.ndarray:
    """Return per-epoch median/IQR/p90 of harmonized 1-Hz envelope."""
    cached = _emg_cache_path(stem)
    if cached.exists():
        with np.load(cached) as z:
            if str(z["version"]) != "emg-envelope-v1": raise ValueError("EMG cache version")
            return z["x"]
    import pyedflib
    path = psg_path(stem); e = pyedflib.EdfReader(str(path))
    try:
        names = e.getSignalLabels()
        matches = [i for i, n in enumerate(names) if n.strip().lower() == "emg submental"]
        if len(matches) != 1: raise ValueError(f"{stem}: EMG name count={len(matches)}")
        i = matches[0]; fs = float(e.getSampleFrequency(i)); sig = e.readSignal(i).astype("float64")
        if fs == 100:
            sig = sosfiltfilt(butter(4, 10, btype="highpass", fs=fs, output="sos"), sig)
            nsec = len(sig) // 100
            env = np.sqrt(np.mean(np.square(np.abs(sig[:nsec * 100]).reshape(nsec, 100)), axis=1))
        elif fs == 1:
            env = sig
        else: raise ValueError(f"{stem}: unexpected EMG sampling rate {fs}")
        n = len(env) // 30; z = env[:n * 30].reshape(n, 30)
        x = np.stack([np.median(z, axis=1), np.quantile(z, .75, axis=1) - np.quantile(z, .25, axis=1), np.quantile(z, .9, axis=1)], axis=1).astype("float32")
    finally: e.close()
    cached.parent.mkdir(exist_ok=True)
    tmp = cached.with_suffix(".tmp.npz"); np.savez_compressed(tmp, x=x, version="emg-envelope-v1"); tmp.replace(cached)
    return x


def load_records(part: str, policy: str, cfg: Config):
    if part == "test": raise AssertionError("Internal test remains closed")
    split = json.loads((ART / "split.json").read_text())
    windows = pd.read_csv(csvpath(policy)).set_index("recording_id")
    xs=[]; ys=[]; ids=[]; epochs=[]; trunc=[]; slices={}; cursor=0
    for path in sorted((ART / "cache").glob("*.npz")):
        if path.name[:5] not in split[part]: continue
        stem = path.stem; row = windows.loc[stem]
        lo, hi = int(row.window_start_sec)//30, int(row.window_end_sec)//30
        with np.load(path) as z:
            if str(z["version"]) != VERSION: raise ValueError("Feature cache version")
            raw, y = z["x"][lo:hi], z["y"][lo:hi]
        valid = y >= 0
        emg = emg_features(stem)[lo:hi] if cfg.emg else None
        md = parse_edf_metadata(psg_path(stem)) if cfg.metadata else None
        x, tf = transform_record(raw, valid, cfg, emg, md)
        use = np.flatnonzero(valid); n=len(use)
        xs.append(x[use]); ys.append(y[use]); ids.extend([stem]*n); epochs.extend((lo+use).tolist()); trunc.extend(tf[use].tolist())
        slices[stem] = (cursor, cursor+n); cursor += n
    return np.concatenate(xs), np.concatenate(ys), np.asarray(ids), np.asarray(epochs), np.asarray(trunc), slices


def base_params(balanced=False):
    p = joblib.load(ART / "N02" / "model.joblib")["model"].get_params()
    p["class_weight"] = "balanced" if balanced else None
    p["random_state"] = SEED
    return p


def transition_model(policy: str):
    split = json.loads((ART / "split.json").read_text()); windows=pd.read_csv(csvpath(policy)).set_index("recording_id")
    counts=np.ones((5,5), dtype="float64"); initial=np.ones(5, dtype="float64")
    for path in sorted((ART/"cache").glob("*.npz")):
        if path.name[:5] not in split["train"]: continue
        row=windows.loc[path.stem]; lo=int(row.window_start_sec)//30; hi=int(row.window_end_sec)//30
        with np.load(path) as z: y=z["y"][lo:hi]
        for run in _runs(y>=0):
            s=y[run]; initial[s[0]] += 1
            np.add.at(counts, (s[:-1],s[1:]), 1)
    return initial/initial.sum(), counts/counts.sum(axis=1,keepdims=True), counts


def viterbi_decode(prob: np.ndarray, initial: np.ndarray, trans: np.ndarray) -> np.ndarray:
    eps=1e-12; n=len(prob); dp=np.empty((n,5)); back=np.empty((n,5),dtype=np.int8)
    dp[0]=np.log(initial+eps)+np.log(prob[0]+eps); back[0]=0
    lt=np.log(trans+eps)
    for t in range(1,n):
        score=dp[t-1][:,None]+lt; back[t]=score.argmax(axis=0); dp[t]=score.max(axis=0)+np.log(prob[t]+eps)
    out=np.empty(n,dtype=np.int8);out[-1]=dp[-1].argmax()
    for t in range(n-2,-1,-1):out[t]=back[t+1,out[t+1]]
    return out


def decode_by_record(prob, ids, initial, trans):
    out=np.empty(len(prob),dtype=np.int8)
    for rid in dict.fromkeys(ids.tolist()):
        ix=np.flatnonzero(ids==rid);out[ix]=viterbi_decode(prob[ix],initial,trans)
    return out


def sequence_counts(pred, ids):
    rows=[]
    for rid in dict.fromkeys(ids.tolist()):
        p=pred[ids==rid]
        rows.append(dict(recording_id=rid,wake_bouts=int((p[0]==0)+np.sum((p[1:]==0)&(p[:-1]!=0))),stage_transitions=int(np.sum(p[1:]!=p[:-1]))))
    return pd.DataFrame(rows)


def score_prob(y, prob, ids, use_viterbi=False, tm=None):
    raw=prob.argmax(axis=1)
    pred=decode_by_record(prob,ids,*tm[:2]) if use_viterbi else raw
    return evaluate(y,pred,ids),pred,raw


def fit_one(exp: str, cfg: Config, training_policy="naive", viterbi_tm=None):
    import lightgbm as lgb
    dest=ART/exp
    if (dest/"metrics.json").exists():
        return json.loads((dest/"metrics.json").read_text())
    if dest.exists(): raise ValueError(f"Incomplete experiment exists: {exp}")
    dest.mkdir(); start=time.perf_counter(); stop=threading.Event(); peak=[psutil.Process().memory_info().rss]
    def monitor():
        while not stop.wait(.1): peak[0]=max(peak[0],psutil.Process().memory_info().rss)
    threading.Thread(target=monitor,daemon=True).start()
    x,y,_,_,_,_=load_records("train",training_policy,cfg)
    model=lgb.LGBMClassifier(**base_params(cfg.balanced)); model.fit(x,y)
    train_counts=np.bincount(y,minlength=5).tolist(); feature_count=x.shape[1]; del x,y
    tm=viterbi_tm
    if cfg.viterbi and tm is None: tm=transition_model(training_policy)
    result=dict(experiment=exp,config=asdict(cfg),training_policy=training_policy,training_class_counts=train_counts,feature_count=feature_count,validation={})
    for pop in [PRIMARY,ROBUSTNESS]:
        vx,vy,ids,epochs,trunc,_=load_records("validation",pop,cfg)
        prob=model.predict_proba(vx); m,pred,raw=score_prob(vy,prob,ids,cfg.viterbi,tm)
        result["validation"][pop]=m
        frame=pd.DataFrame(dict(recording_id=ids,epoch_index=epochs,reference=vy,prediction=pred,raw_prediction=raw,context_truncated=trunc))
        for i,s in enumerate(STAGES):frame[f"P({s})"]=prob[:,i]
        frame.to_csv(dest/f"predictions_{pop}.csv",index=False)
        del vx,vy,ids,epochs,trunc,prob,pred,raw,frame
    stop.set(); result.update(seconds=time.perf_counter()-start,sampled_peak_rss_mb=peak[0]/2**20)
    save(dest/"metrics.json",result)
    bundle=dict(model=model,config=asdict(cfg),version=VERSION,stages=STAGES,channels=CHANNELS,experiment=exp,training_policy=training_policy)
    if tm is not None: bundle.update(viterbi_initial=tm[0],viterbi_transition=tm[1],viterbi_counts=tm[2])
    joblib.dump(bundle,dest/"model.joblib")
    return result


def load_n02():
    old=json.loads((ART/"N02"/"window_metrics.json").read_text())
    return dict(experiment="N02",config=asdict(Config()),training_policy="naive",seconds=old["seconds"],validation=old["validation"])


def better(a,b):
    """Select higher primary macro-F1; robustness breaks exact ties."""
    ka=(a["validation"][PRIMARY]["macro_f1"],a["validation"][ROBUSTNESS]["macro_f1"])
    kb=(b["validation"][PRIMARY]["macro_f1"],b["validation"][ROBUSTNESS]["macro_f1"])
    return a if ka>=kb else b


def config_of(result): return Config(**result["config"])


def verify_metadata_all():
    files=sorted(DATA.rglob("*-PSG.edf")); failed=[p.name for p in files if parse_edf_metadata(p) is None]
    if failed: raise ValueError(f"Metadata parsing failed: {failed}")
    return len(files)


def verify_n03_single_edf(cfg: Config):
    """Compare cache and direct single-EDF transforms over identical full input."""
    split=json.loads((ART/"split.json").read_text()); p=next(p for p in sorted((ART/"cache").glob("*.npz")) if p.name[:5] in split["validation"])
    with np.load(p) as z: cached=z["x"]
    direct,_,_=extract(psg_path(p.stem))
    if cached.shape!=direct.shape or not np.allclose(cached,direct,rtol=1e-6,atol=1e-6): raise ValueError("Direct EDF/cache feature mismatch")
    a,_=transform_record(cached,np.ones(len(cached),bool),cfg)
    b,_=transform_record(direct,np.ones(len(direct),bool),cfg)
    if not np.array_equal(a,b): raise ValueError("Single-EDF normalization mismatch")
    return dict(recording=p.stem,epochs=len(a),max_abs_difference=float(np.max(np.abs(a-b))),external_state=False)


def n07_evaluate(source_result):
    source=source_result["experiment"]; src=ART/source; dest=ART/"N07"
    if dest.exists(): raise ValueError("Experiment exists: N07")
    dest.mkdir(); start=time.perf_counter(); tm=transition_model(source_result["training_policy"])
    result=dict(experiment="N07",config={**source_result["config"],"viterbi":True},training_policy=source_result["training_policy"],validation={},transition_counts=tm[2].astype(int).tolist())
    for pop in [PRIMARY,ROBUSTNESS]:
        f=pd.read_csv(src/f"predictions_{pop}.csv"); prob=f[[f"P({s})" for s in STAGES]].to_numpy(); y=f.reference.to_numpy();ids=f.recording_id.astype(str).to_numpy(dtype=str)
        before=f.raw_prediction.to_numpy(); after=decode_by_record(prob,ids,*tm[:2])
        m=evaluate(y,after,ids); bm=evaluate(y,before,ids)
        sb=sequence_counts(before,ids);sa=sequence_counts(after,ids);seq=sb.merge(sa,on="recording_id",suffixes=("_before","_after"))
        seq["wake_bouts_delta"]=seq.wake_bouts_after-seq.wake_bouts_before;seq["stage_transitions_delta"]=seq.stage_transitions_after-seq.stage_transitions_before
        seq.to_csv(dest/f"sequence_effect_{pop}.csv",index=False)
        result["validation"][pop]=m;result["validation"][pop]["before_viterbi"]=bm
        result["validation"][pop]["sequence_effect_totals"]={c:int(seq[c].sum()) for c in seq.columns if c!="recording_id"}
        f["prediction"]=after;f.to_csv(dest/f"predictions_{pop}.csv",index=False)
    result["seconds"]=time.perf_counter()-start;save(dest/"metrics.json",result)
    b=joblib.load(src/"model.joblib");b.update(config=result["config"],experiment="N07",viterbi_initial=tm[0],viterbi_transition=tm[1],viterbi_counts=tm[2]);joblib.dump(b,dest/"model.joblib")
    return result


def run_all():
    summary=[load_n02()]; incumbent=summary[0]
    # N03 has two explicitly requested variants.
    n03a=fit_one("N03_norm",Config(representation="norm"));summary.append(n03a)
    n03b=fit_one("N03_rawnorm",Config(representation="raw_norm"));summary.append(n03b)
    incumbent=better(incumbent,better(n03a,n03b)); n03verify=verify_n03_single_edf(config_of(better(n03a,n03b)))
    # N04 wider context and band-only rolling features.
    c=config_of(incumbent); n04=fit_one("N04",Config(**{**asdict(c),"context_radius":2,"rolling_radius":15}));summary.append(n04);incumbent=better(incumbent,n04)
    # N05 interaction of context winner and balanced class weights.
    c=config_of(incumbent);n05=fit_one("N05",Config(**{**asdict(c),"balanced":True}));summary.append(n05);incumbent=better(incumbent,n05)
    # N06 channel-name EMG harmonization.
    c=config_of(incumbent);n06=fit_one("N06",Config(**{**asdict(c),"emg":True}));summary.append(n06);incumbent=better(incumbent,n06)
    # N07 is deterministic post-processing of the current winner.
    n07=n07_evaluate(incumbent);summary.append(n07);incumbent=better(incumbent,n07)
    # N08 metadata challenger; preserve incumbent as explicit fallback.
    underlying=Config(**{**asdict(config_of(incumbent)),"viterbi":False,"metadata":True})
    n08base=fit_one("N08",underlying)
    if config_of(incumbent).viterbi:
        # Apply a training-only transition model without retraining the metadata classifier.
        tmp=n07_evaluate_named(n08base,"N08_viterbi")
        n08=tmp
    else:n08=n08base
    summary.append(n08base)
    if n08 is not n08base:summary.append(n08)
    incumbent=better(incumbent,n08)
    metadata_count=verify_metadata_all()
    # Final configuration window-policy rerun. Viterbi matrix is re-estimated per training policy.
    final_cfg=config_of(incumbent);rerun=[]
    for policy,suffix in [("untrimmed","U"),("naive","N"),("msp60","M")]:
        r=fit_one(f"FINAL_WINDOW_{suffix}",final_cfg,policy);rerun.append(r);summary.append(r)
    final=rerun[0]
    for r in rerun[1:]:final=better(final,r)
    state=dict(summary=summary,incumbent_before_window=incumbent["experiment"],final_winner=final["experiment"],n03_single_edf_verification=n03verify,metadata_parse_count=metadata_count)
    save(ART/"advanced_experiment_state.json",state)
    write_report(state)
    package_release(final,incumbent)


def n07_evaluate_named(source_result,name):
    source=source_result["experiment"];dest=ART/name
    if dest.exists():raise ValueError(f"Experiment exists: {name}")
    dest.mkdir();start=time.perf_counter();tm=transition_model(source_result["training_policy"])
    result=dict(experiment=name,config={**source_result["config"],"viterbi":True},training_policy=source_result["training_policy"],validation={})
    for pop in [PRIMARY,ROBUSTNESS]:
        f=pd.read_csv(ART/source/f"predictions_{pop}.csv");prob=f[[f"P({s})" for s in STAGES]].to_numpy();y=f.reference.to_numpy();ids=f.recording_id.astype(str).to_numpy(dtype=str);pred=decode_by_record(prob,ids,*tm[:2]);result["validation"][pop]=evaluate(y,pred,ids);f["prediction"]=pred;f.to_csv(dest/f"predictions_{pop}.csv",index=False)
    result["seconds"]=time.perf_counter()-start;save(dest/"metrics.json",result)
    b=joblib.load(ART/source/"model.joblib");b.update(config=result["config"],experiment=name,viterbi_initial=tm[0],viterbi_transition=tm[1],viterbi_counts=tm[2]);joblib.dump(b,dest/"model.joblib")
    return result


def package_release(final, fallback):
    import shutil
    dest=ART/"release_candidate";dest.mkdir(exist_ok=True)
    shutil.copy2(ART/final["experiment"]/"model.joblib",dest/"model.joblib")
    # Always preserve a no-metadata fallback when selected model needs metadata.
    fb=fallback
    if config_of(final).metadata:
        candidates=[r for r in json.loads((ART/"advanced_experiment_state.json").read_text())["summary"] if not r["config"].get("metadata",False) and r["training_policy"]==final["training_policy"]]
        if candidates:
            fb=max(candidates,key=lambda r:r["validation"][PRIMARY]["macro_f1"])
        shutil.copy2(ART/fb["experiment"]/"model.joblib",dest/"model_no_metadata.joblib")
    save(dest/"release.json",dict(selected=final["experiment"],fallback=fb["experiment"] if config_of(final).metadata else None,internal_test_opened=False))


def _fmt(v): return f"{v:.4f}"


def write_report(state):
    rows=[];n02=state["summary"][0]["validation"][PRIMARY]["macro_f1"]
    for r in state["summary"]:
        p=r["validation"][PRIMARY];q=r["validation"][ROBUSTNESS]
        rows.append(f"| {r['experiment']} | {r['training_policy']} | {_fmt(p['macro_f1'])} | {_fmt(q['macro_f1'])} | "+" | ".join(_fmt(x) for x in p["per_class_f1"])+f" | {p['macro_f1']-n02:+.4f} | {r.get('seconds',0):.1f} |")
    final=next(r for r in state["summary"] if r["experiment"]==state["final_winner"]);m=final["validation"][PRIMARY];cm=np.asarray(m["confusion_matrix"]);true=cm.sum(1);pred=cm.sum(0)
    matrix="\n".join(["| "+STAGES[i]+" | "+" | ".join(map(str,cm[i]))+f" | {true[i]} |" for i in range(5)])
    rer=[r for r in state["summary"] if r["experiment"].startswith("FINAL_WINDOW_")]
    win="\n".join(f"| {r['training_policy']} | {_fmt(r['validation'][PRIMARY]['macro_f1'])} | {_fmt(r['validation'][ROBUSTNESS]['macro_f1'])} | {r['validation'][PRIMARY]['cohorts']['SC']['macro_f1']:.4f} | {r['validation'][PRIMARY]['cohorts']['ST']['macro_f1']:.4f} |" for r in rer)
    ordered=[]
    best=state["summary"][0]
    for r in state["summary"][1:]:
        if r["experiment"].startswith("FINAL_WINDOW_"):continue
        if r["validation"][PRIMARY]["macro_f1"]>best["validation"][PRIMARY]["macro_f1"]:best=r
        else:ordered.append((r["experiment"],r["validation"][PRIMARY]["macro_f1"]-best["validation"][PRIMARY]["macro_f1"]))
    rejected="\n".join(f"- **{n}**: rejected at its gate; primary Macro-F1 was {d:+.4f} relative to the incumbent available at that gate." for n,d in ordered) or "- None."
    text=f"""# Controlled Algorithm Experiment Report

Generated: 2026-09-23. Seed: `{SEED}`. Frozen split and label/metric contracts were retained. The internal test partition was not loaded, featurized, trained on, or scored.

## Complete results

Primary = MSP-60 validation; robustness = naive-window validation. Stage order is Wake, N1, N2, N3, REM.

| Configuration | Train window | Primary Macro-F1 | Robustness Macro-F1 | Wake F1 | N1 F1 | N2 F1 | N3 F1 | REM F1 | Delta vs N02 | Wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

N03 single-EDF verification: `{json.dumps(state['n03_single_edf_verification'])}`. The exact same stateless normalization function processed the direct EDF and cached physical features. Header age/sex parsed for `{state['metadata_parse_count']}/197` files. N08 retains a no-metadata fallback checkpoint.

## Final winner confusion matrix

Release candidate: **{final['experiment']}**, trained on **{final['training_policy']}**. Rows are true stages and columns are predicted stages.

| True / Predicted | Wake | N1 | N2 | N3 | REM | True total |
|---|---:|---:|---:|---:|---:|---:|
{matrix}
| **Predicted total** | {' | '.join(map(str,pred))} | {pred.sum()} |

## Window-policy re-run with final configuration

| Training population | MSP-60 Macro-F1 | Naive Macro-F1 | SC MSP-60 | ST MSP-60 |
|---|---:|---:|---:|---:|
{win}

The selected policy is the row with the strongest primary result, using robustness as the tie-breaker. No validation population, preprocessing definition, hyperparameter, seed, or split changed across these three fits.

## Rejected experiments

{rejected}

## N07 sequence impact

See `artifacts/N07/sequence_effect_msp60.csv` and `sequence_effect_naive.csv` for recording-level wake-bout and transition counts before/after Viterbi. SQI-facing probabilities remain the unsmoothed model probabilities; Viterbi affects only the discrete classification output.

## Release recommendation

Ship **{final['experiment']}** because it is the best controlled primary-validation configuration after reopening the training-window decision under the final preprocessing. Its MSP-60 Macro-F1 is **{m['macro_f1']:.4f}** and naive robustness Macro-F1 is **{final['validation'][ROBUSTNESS]['macro_f1']:.4f}**. The package is in `artifacts/release_candidate/`. Metadata fallback is mandatory if the selected configuration uses age/sex. All claims here are validation results; the internal test remains sealed.
"""
    (ROOT/"EXPERIMENT_REPORT.md").write_text(text,encoding="utf-8")


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("command",choices=["run"]);a=p.parse_args();run_all()
