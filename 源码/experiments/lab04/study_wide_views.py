"""Frozen wider-backbone two-view study: train/validation only until explicitly evaluated.

One additional deterministic horizontal flip per image; no new labels, external
weights or test-class selection. Complete negative grid and paired scores persist.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .data import image_records, normalized_tensor, digest_file
from .experiment import load_checkpoint, log, summarize, write_json
from .graph_calibration import GraphConfig, apply_transform, predict
from .multiview import combine_views
from .refine import tasks_for, validation_partition
from .study_cached_refinement import load_wide, paired_stats
from .supervised_whitening import fit


def extract_views(args, split):
    if split not in ("train", "val"):
        raise ValueError("Test data forbidden during validation selection")
    out=Path(args.output); source=Path(args.wide); cache=out/f"{split}_views.npz"; stamp=out/f"{split}_views.json"
    original=load_wide(source,split)
    identity={"checkpoint_sha256":digest_file(source/"best.pt"),"original_cache_sha256":digest_file(source/f"{split}_features.npz"),
              "source_sha256":digest_file(Path(__file__)),"preprocessing_sha256":digest_file(Path(__file__).with_name("data.py")),"view":"horizontal flip of normalized tensor"}
    if cache.exists() and stamp.exists():
        metadata=json.loads(stamp.read_text(encoding="utf-8"))
        if metadata["identity"]!=identity or metadata["sha256"]!=digest_file(cache):
            raise ValueError("Wide view cache fingerprint changed")
        with np.load(cache,allow_pickle=False) as src: return {k:src[k] for k in src.files}
    records=image_records(args.data_root,split); by_id={r[2]:r for r in records}
    model,_=load_checkpoint(source/"best.pt"); model.eval(); values=[]; start_time=time.perf_counter()
    with torch.inference_mode():
        for start in range(0,len(original["ids"]),32):
            images=[]
            for id in original["ids"][start:start+32]:
                with Image.open(by_id[str(id)][1]) as image:
                    images.append(np.asarray(image.convert("RGB").resize((84,84),Image.Resampling.BICUBIC)))
            values.append(model.features(normalized_tensor(np.stack(images)).flip(-1)).numpy())
            if (start//32+1)%200==0: log("wideview_encoding",split=split,images=start+len(images),seconds=time.perf_counter()-start_time)
    data={"original":original["features"],"flipped":np.concatenate(values),"labels":original["labels"],"ids":original["ids"]}
    temporary=cache.with_suffix(".tmp.npz"); np.savez_compressed(temporary,**data); temporary.replace(cache)
    write_json(stamp,{"identity":identity,"sha256":digest_file(cache),"seconds":time.perf_counter()-start_time})
    log("wideview_cache_complete",split=split,images=len(data["ids"])); return data


def run(args):
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True); torch.set_num_threads(args.threads)
    wide=Path(args.wide); artifacts=Path(args.artifacts)
    filenames=("study_wide_views.py","study_cached_refinement.py","multiview.py","supervised_whitening.py","graph_calibration.py","calibration.py","refine.py","data.py")
    inputs=[wide/"best.pt",artifacts/"wide_configuration.json",artifacts/"wide_transform.npz"]
    inputs += [wide/f"{split}_{kind}" for split in ("train","val") for kind in ("features.npz","cache.json")]
    candidates=[{"views":views,"exponent":exponent,"residual_mix":mix} for views in ("mean","concat") for exponent in (.25,.5,.75) for mix in (0.,.5,1.)]
    plan={"seed":args.seed,"search_episodes":160,"confirmation_episodes":400,"shortlist":4,"candidates":candidates,"test_accessed":False,
          "inputs":{str(p):digest_file(p) for p in inputs},"source_hashes":{f:digest_file(Path(__file__).with_name(f)) for f in filenames},
          "selection_rule":"Search top four, choose confirmation paired lower bound vs frozen wide reference. Retain original wide unless lower bound > 0. No automatic publication.",
          "limitations":"Repeated same validation classes. Confirmation selects the method, so is not independent final evaluation. Two views add encoding cost."}
    path=output/"selection_plan.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8"))!=plan: raise ValueError("Immutable study plan changed")
    if not path.exists(): write_json(path,plan)
    destination=output/"selection.json"
    if destination.exists() and json.loads(destination.read_text(encoding="utf-8")).get("status")=="complete":
        log("wideview_study_already_complete"); return
    train=extract_views(args,"train"); val=extract_views(args,"val")
    train_classes={str(id).split("/")[1] for id in train["ids"]}; val_classes={str(id).split("/")[1] for id in val["ids"]}
    if train_classes & val_classes or len(train_classes)!=64 or len(val_classes)!=16: raise ValueError("Class leakage or incomplete training/validation pool")
    search,confirm=validation_partition(val["labels"],args.seed)
    if set(search)&set(confirm): raise ValueError("Overlapping selection images")
    frozen=json.loads((artifacts/"wide_configuration.json").read_text(encoding="utf-8"))
    scores={}; journal=output/"scores.jsonl"
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            v=json.loads(line);scores[v["key"]]=v["values"]
    def evaluate(key,features,tasks,config):
        if key in scores:return scores[key]
        start=time.perf_counter(); values=[]
        for task in tasks:
            answer=predict(features[task["support_indices"]],task["support_labels"],features[task["query_indices"]],config)
            values.append(float(np.mean(answer["predictions"]==task["query_labels"])))
        with journal.open("a",encoding="utf-8",newline="\n") as stream:stream.write(json.dumps({"key":key,"values":values},allow_nan=False)+"\n")
        scores[key]=values;log("wideview_score",key=key,accuracy=float(np.mean(values)),seconds=time.perf_counter()-start);return values
    result={"status":"selecting","plan_sha256":digest_file(path),"shots":{},"test_accessed":False}; transforms={}
    for shot in (1,5):
        config=GraphConfig(**frozen["selected"][str(shot)]["config"])
        tasks=tasks_for(val["labels"][search],args.seed+100+shot,shot,160); confirmation=tasks_for(val["labels"][confirm],args.seed+200+shot,shot,400)
        with np.load(artifacts/"wide_transform.npz",allow_pickle=False) as src:
            prefix=f"shot{shot}_"; reference_transform={k[len(prefix):]:src[k] for k in src.files if k.startswith(prefix)}
        reference=apply_transform(val["original"],reference_transform)
        ref_search=evaluate(f"{shot}/reference/search",reference[search],tasks,config)
        ref_confirm=evaluate(f"{shot}/reference/confirmation",reference[confirm],confirmation,config)
        rows=[]
        for index,candidate in enumerate(candidates):
            training=combine_views(train["original"],train["flipped"],candidate["views"])
            validation=combine_views(val["original"],val["flipped"],candidate["views"])
            transform=fit(training,train["labels"],strength=1.,exponent=candidate["exponent"],residual_mix=candidate["residual_mix"])
            values=evaluate(f"{shot}/{index}/search",apply_transform(validation,transform)[search],tasks,config)
            rows.append({"index":index,"candidate":candidate,"config":config.to_dict(),"search":summarize(values),"search_paired":paired_stats(values,ref_search)})
        shortlist=sorted(rows,key=lambda r:(-r["search"]["accuracy"],r["index"]))[:4]
        for row in shortlist:
            c=row["candidate"];training=combine_views(train["original"],train["flipped"],c["views"]);validation=combine_views(val["original"],val["flipped"],c["views"])
            transform=fit(training,train["labels"],strength=1.,exponent=c["exponent"],residual_mix=c["residual_mix"])
            values=evaluate(f"{shot}/{row['index']}/confirmation",apply_transform(validation,transform)[confirm],confirmation,config)
            row["confirmation"]=summarize(values);row["confirmation_paired"]=paired_stats(values,ref_confirm)
        selected=max(shortlist,key=lambda r:(r["confirmation_paired"]["mean"]-r["confirmation_paired"]["ci95"],-r["index"]))
        c=selected["candidate"];training=combine_views(train["original"],train["flipped"],c["views"])
        transform=fit(training,train["labels"],strength=1.,exponent=c["exponent"],residual_mix=c["residual_mix"])
        transforms.update({f"shot{shot}_{k}":v for k,v in transform.items()})
        result["shots"][str(shot)]={"selected":selected,"candidates":rows,"reference":{"search":summarize(ref_search),"confirmation":summarize(ref_confirm)},
                                  "promising":selected["confirmation_paired"]["mean"]-selected["confirmation_paired"]["ci95"]>0}
        write_json(destination,result);log("wideview_selected",shot=shot,selected=selected)
    np.savez_compressed(output/"transform.npz",**transforms)
    result.update({"status":"complete","completed_utc":datetime.now(timezone.utc).isoformat(),"transform_sha256":digest_file(output/"transform.npz")})
    write_json(destination,result)
    write_json(output/"selection_lock.json",{"status":"locked","test_accessed":False,"plan_sha256":digest_file(path),"selection_sha256":digest_file(destination),
        "transform_sha256":digest_file(output/"transform.npz"),"checkpoint_sha256":digest_file(wide/"best.pt"),"locked_utc":datetime.now(timezone.utc).isoformat()})
    log("wideview_study_complete")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",default=".local/lab04-wide-views-20261125")
    p.add_argument("--data-root",default=".local/mini-imagenet");p.add_argument("--wide",default=".local/lab04-v3-wide")
    p.add_argument("--artifacts",default="outputs/lab04");p.add_argument("--threads",type=int,default=2);p.add_argument("--seed",type=int,default=20261127)
    run(p.parse_args())


if __name__=="__main__":main()
