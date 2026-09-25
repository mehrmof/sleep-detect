"""E13: overlapping inference for frozen C01 checkpoints (no retraining)."""
from pathlib import Path
from collections import OrderedDict
import json, sys, time

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "research_tools"))
import numpy as np
import pandas as pd
import cnn_c01 as c

OUT = c.ART / "CNN_E13_OVERLAP"
PROBS = [f"P({s})" for s in c.STAGES]


class OverlapDataset:
    def __init__(self, subjects, stride=10):
        import torch
        self.torch = torch
        self.items = []
        self.maps = OrderedDict()
        manifest = pd.read_csv(c.OUT / "cache_manifest.csv")
        for stem in manifest.loc[manifest.subject_id.isin(set(subjects)), "recording_id"].sort_values():
            with np.load(c.CACHE / (stem + ".npz")) as z:
                y, ep = z["y"], z["epoch_index"]
            for run in c.runs(y >= 0):
                # Starts every 10 valid epochs. Partial final sequences are padded exactly as C01.
                for k in range(0, len(run), stride):
                    use = run[k:k + c.SEQ]
                    self.items.append((stem, int(use[0]), len(use), y[use].copy(), ep[use].copy()))

    def __len__(self): return len(self.items)
    def _map(self, stem):
        if stem not in self.maps: self.maps[stem] = np.load(c.CACHE / (stem + ".npy"), mmap_mode="r")
        self.maps.move_to_end(stem)
        while len(self.maps) > 8: self.maps.popitem(last=False)
        return self.maps[stem]
    def __getitem__(self, i):
        stem, start, n, y, ep = self.items[i]
        x = np.asarray(self._map(stem)[start:start+n], dtype="float32")
        if n < c.SEQ: x = np.concatenate([x, np.repeat(x[-1:], c.SEQ-n, axis=0)])
        yy = np.full(c.SEQ, -100, dtype="int64"); ee = np.full(c.SEQ, -1, dtype="int32")
        yy[:n] = y; ee[:n] = ep
        return self.torch.from_numpy(x), self.torch.from_numpy(yy), stem, self.torch.from_numpy(ee), n


def predict_average(model, ds, device, batch=16):
    import torch
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0,
                                         pin_memory=True, collate_fn=c.collate)
    rows=[]; model.eval()
    with torch.inference_mode():
        for x,y,stems,eps,ns in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                p=torch.softmax(model(x.to(device,non_blocking=True)).float(),-1).cpu().numpy()
            yy=y.numpy(); ee=eps.numpy()
            for j,n in enumerate(ns):
                for k in range(n): rows.append((stems[j],int(ee[j,k]),int(yy[j,k]),*map(float,p[j,k])))
    raw=pd.DataFrame(rows,columns=["recording_id","epoch_index","reference"]+PROBS)
    if raw.groupby(["recording_id","epoch_index"]).reference.nunique().max()!=1: raise ValueError("label conflict")
    avg=raw.groupby(["recording_id","epoch_index","reference"],as_index=False)[PROBS].mean()
    avg["subject_id"]=avg.recording_id.str[:5]
    return avg, len(raw)


def main():
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    OUT.mkdir(exist_ok=True)
    folds=json.loads((c.OUT/"folds.json").read_text())["folds"]
    all_oof=[]; t0=time.perf_counter(); raw_rows=0
    C=c.model_class(); device=torch.device("cuda")
    for fold in folds:
        fid=fold["fold_id"]; model=C().to(device)
        ck=torch.load(c.OUT/f"fold_{fid}"/"best.pt",map_location=device,weights_only=True)
        model.load_state_dict(ck["model"])
        frame,nraw=predict_average(model,OverlapDataset(fold["heldout_subjects"]),device)
        frame["fold_id"]=fid; frame["model_version"]="CNN_C01_E13_overlap_inference_stride10"
        frame.to_csv(OUT/f"fold_{fid}_oof.csv",index=False); all_oof.append(frame); raw_rows+=nraw
    oof=pd.concat(all_oof,ignore_index=True).sort_values(["recording_id","epoch_index"])
    base=pd.read_csv(c.OUT/"oof_probabilities.csv")
    keys=["recording_id","epoch_index","reference"]
    if len(oof)!=len(base) or oof.duplicated(keys).any() or len(base[keys].merge(oof[keys],on=keys))!=len(base):
        raise ValueError("E13 coverage differs from C01")
    if not np.allclose(oof[PROBS].sum(1),1,atol=1e-5): raise ValueError("bad probabilities")
    oof.to_csv(OUT/"oof_probabilities.csv",index=False)
    pooled=c.calc(oof)
    # exact_compare writes derived tables via c.OUT; redirect only after cache inference is complete.
    c.OUT=OUT
    comparison,aligned=c.exact_compare(oof)
    result={"version":"CNN_C01_E13_overlap_inference_stride10","controlled_change":"inference sequence stride 20 to 10; average repeated epoch probabilities","training_changed":False,"raw_prediction_rows":raw_rows,"unique_oof_rows":len(oof),"runtime_seconds":time.perf_counter()-t0,"pooled_oof":pooled,"exact_comparison":comparison,"complementarity_msp60":c.complementarity(aligned["msp60"]),"internal_test_accessed":False}
    (OUT/"results.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({"pooled":pooled["macro_f1"],"msp":comparison["msp60"]["CNN_C01"]["macro_f1"],"naive":comparison["naive"]["CNN_C01"]["macro_f1"],"runtime_s":result["runtime_seconds"]},indent=2))


if __name__ == "__main__":
    # Reuse immutable C01 cache/checkpoints; redirect exact-comparison artifacts.
    c.OUT = c.ART / "CNN_C01"
    main()
