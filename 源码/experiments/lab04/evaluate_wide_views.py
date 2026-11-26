"""One locked test of a validation-selected wider two-view candidate.

Uses the original 1200 paired tasks; no search, threshold choice or auto-publish.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import digest_file, load_split, normalized_tensor, sample_episode, selection_digest
from .experiment import load_checkpoint, log, summarize, write_json
from .graph_calibration import GraphConfig, apply_transform, predict
from .multiview import combine_views


def read(path): return json.loads(path.read_text(encoding="utf-8"))


def run(args):
    source=Path(args.candidate); core=Path(args.artifacts); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    lock=read(source/"selection_lock.json"); selection=read(source/"selection.json"); selection_plan=read(source/"selection_plan.json")
    if lock.get("status")!="locked" or lock.get("test_accessed") is not False or selection.get("status")!="complete": raise ValueError("Completed validation lock is required")
    for name,key in (("selection_plan.json","plan_sha256"),("selection.json","selection_sha256"),("transform.npz","transform_sha256")):
        if digest_file(source/name)!=lock[key]: raise ValueError("Validation evidence changed: "+name)
    for name,digest in selection_plan["source_hashes"].items():
        if digest_file(Path(__file__).with_name(name))!=digest: raise ValueError("Validation source changed")
    if not all(r["promising"] for r in selection["shots"].values()): raise ValueError("Both validation settings must pass the predeclared gate")
    checkpoint=core/"wide.pt"
    if digest_file(checkpoint)!=lock["checkpoint_sha256"]: raise ValueError("Frozen wider weights changed")
    metadata=read(core/"results.json")
    sources=("evaluate_wide_views.py","models.py","data.py","calibration.py","graph_calibration.py","multiview.py")
    references=("results.json","episodes.json","graph_episodes.json","multiview_episodes.json","wide_episodes.json","wide_results.json")
    plan={"status":"locked","selection_lock_sha256":digest_file(source/"selection_lock.json"),"selection_sha256":lock["selection_sha256"],
          "transform_sha256":lock["transform_sha256"],"checkpoint_sha256":lock["checkpoint_sha256"],"seed":metadata["protocol"]["seed"],
          "episodes_per_shot":600,"query_per_class":15,"test_pool_sha256":metadata["protocol"]["test_selection_sha256"],
          "reference_files":{name:digest_file(core/name) for name in references},"source_hashes":{name:digest_file(Path(__file__).with_name(name)) for name in sources}}
    plan_path=out/"evaluation_plan.json"
    if plan_path.exists() and read(plan_path)!=plan: raise ValueError("Evaluation plan changed; do not rerun against different evidence")
    if not plan_path.exists(): write_json(plan_path,plan)
    if (out/"wideview_results.json").exists() and read(out/"wideview_results.json").get("status")=="complete":
        log("wideview_test_already_complete"); return
    torch.set_num_threads(args.threads)
    test=load_split(args.data_root,"test",max_per_class=600,seed=metadata["protocol"]["seed"]+4)
    if selection_digest(test)!=plan["test_pool_sha256"]: raise ValueError("Test pool not paired")
    cache=out/"test_views.npz"; stamp=out/"test_views.json"
    identity={"checkpoint_sha256":plan["checkpoint_sha256"],"test_pool_sha256":selection_digest(test),"source_sha256":plan["source_hashes"]}
    if cache.exists() or stamp.exists():
        if not (cache.is_file() and stamp.is_file()): raise ValueError("Partial test cache; inspect rather than silently trust")
        manifest=read(stamp)
        if manifest["identity"]!=identity or manifest["sha256"]!=digest_file(cache): raise ValueError("Unverified test view cache")
        with np.load(cache,allow_pickle=False) as src: features={k:src[k] for k in src.files}
        if not np.array_equal(features["ids"],test.ids): raise ValueError("Test cache sample mismatch")
    else:
        model,_=load_checkpoint(checkpoint);model.eval();normal=[];flipped=[];started=time.perf_counter()
        with torch.inference_mode():
            for start in range(0,len(test.images),32):
                tensor=normalized_tensor(test.images[start:start+32])
                normal.append(model.features(tensor).numpy());flipped.append(model.features(tensor.flip(-1)).numpy())
                if (start//32+1)%100==0:log("wideview_test_encoding",images=start+len(tensor),seconds=time.perf_counter()-started)
        features={"original":np.concatenate(normal),"flipped":np.concatenate(flipped),"ids":np.array(test.ids),"labels":test.labels}
        np.savez_compressed(cache,**features);write_json(stamp,{"identity":identity,"sha256":digest_file(cache)})
    core_tasks={(r["shots"],r["episode"]):r for r in read(core/"episodes.json")["episodes"]}
    extension_tasks={name:{(r["shots"],r["episode"]):r for r in read(core/f"{name}_episodes.json")["episodes"]} for name in ("graph","multiview","wide")}
    configuration={"status":"locked","test_accessed":False,"locked_utc":lock["locked_utc"],"base_weights_sha256":plan["checkpoint_sha256"],
                   "selected":{shot:{**r["selected"]["candidate"],"config":r["selected"]["config"]} for shot,r in selection["shots"].items()},
                   "selection_sha256":lock["selection_sha256"],"plan_sha256":lock["plan_sha256"],"transform_sha256":lock["transform_sha256"]}
    journal=out/"wideview_episodes.jsonl"
    records=[json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()] if journal.exists() else []
    done={(r["shots"],r["episode"]):r for r in records}
    if len(done)!=len(records):raise ValueError("Duplicate resumed task")
    for shot in (1,5):
        cfg=configuration["selected"][str(shot)]
        with np.load(source/"transform.npz",allow_pickle=False) as src:
            prefix=f"shot{shot}_";transform={key[len(prefix):]:src[key] for key in src.files if key.startswith(prefix)}
        embedded=apply_transform(combine_views(features["original"],features["flipped"],cfg["views"]),transform)
        graph=GraphConfig(**cfg["config"]);rng=np.random.default_rng(metadata["protocol"]["seed"]+200+shot)
        for episode in range(600):
            task=sample_episode(test.labels,rng,shots=shot,queries=15);key=(shot,episode);reference=core_tasks[key]
            support=[test.ids[i] for i in task["support_indices"]];query=[test.ids[i] for i in task["query_indices"]]
            if support!=reference["support_ids"] or query!=reference["query_ids"] or task["query_labels"].tolist()!=reference["query_labels"]:raise ValueError("Test task mismatch")
            if key in done:
                if done[key]["support_ids"]!=support or done[key]["query_ids"]!=query:raise ValueError("Resume task mismatch")
                continue
            started=time.perf_counter()
            answer=predict(embedded[task["support_indices"]],task["support_labels"],embedded[task["query_indices"]],graph)
            elapsed=time.perf_counter()-started
            row={"shots":shot,"episode":episode,"support_ids":support,"query_ids":query,"query_labels":task["query_labels"].tolist(),
                 "predictions":answer["predictions"].tolist(),"probabilities":answer["probabilities"].tolist(),
                 "accuracy":float(np.mean(answer["predictions"]==task["query_labels"])),"seconds":elapsed,
                 "baseline_accuracy":reference["methods"]["baseline"]["accuracy"],"full_accuracy":reference["methods"]["full"]["accuracy"],
                 **{f"{name}_accuracy":values[key]["accuracy"] for name,values in extension_tasks.items()}}
            with journal.open("a",encoding="utf-8",newline="\n") as stream: stream.write(json.dumps(row,allow_nan=False)+"\n");stream.flush()
            records.append(row)
            if (episode+1)%150==0:log("wideview_test",shot=shot,episodes=episode+1)
    rows=[]
    for shot in (1,5):
        tasks=sorted((r for r in records if r["shots"]==shot),key=lambda r:r["episode"])
        if len(tasks)!=600:raise ValueError("Incomplete test")
        values=np.array([r["accuracy"] for r in tasks]); row={"method":"wideview","shots":shot,**summarize(values),
            "mean_episode_seconds":float(np.mean([r["seconds"] for r in tasks])),"timing_scope":"inference_from_cached_features","transductive":True,"balanced_queries":True}
        for name in ("baseline","full","graph","multiview","wide"):
            delta=values-np.array([r[f"{name}_accuracy"] for r in tasks]);mean_key="paired_delta" if name=="baseline" else f"paired_vs_{name}"
            ci_key="paired_ci95" if name=="baseline" else f"paired_vs_{name}_ci95"
            row[mean_key]=float(delta.mean());row[ci_key]=float(1.96*delta.std(ddof=1)/600**.5)
        rows.append(row)
    write_json(out/"wideview_episodes.json",{"episodes":records})
    write_json(out/"wideview_configuration.json",configuration)
    (out/"wideview_transform.npz").write_bytes((source/"transform.npz").read_bytes())
    (out/"wideview_selection.json").write_bytes((source/"selection.json").read_bytes())
    (out/"wideview_plan.json").write_bytes((source/"selection_plan.json").read_bytes())
    names=("wideview_episodes.json","wideview_configuration.json","wideview_transform.npz","wideview_selection.json","wideview_plan.json")
    write_json(out/"wideview_results.json",{"status":"complete","completed_utc":datetime.now(timezone.utc).isoformat(),"rows":rows,
        "base_weights_sha256":plan["checkpoint_sha256"],"selection_lock_sha256":digest_file(source/"selection_lock.json"),"evaluation_plan":plan,
        "files":{name:{"path":name,"sha256":digest_file(out/name)} for name in names},
        "scope":"Frozen same-wide-backbone multiview/residual-transform study. No additional training; two encodings per image; balanced transductive protocol.",
        "limitations":["Same repeatedly used public benchmark, one training seed.","Selected on reused validation classes; confirmation is selection, not independent test.","Extra encoding cost; not a cost-matched comparison."]})
    log("wideview_test_complete",rows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--candidate",default=".local/lab04-wide-views-20261125")
    p.add_argument("--artifacts",default="outputs/lab04");p.add_argument("--output",default=".local/lab04-wideview-test-20261125")
    p.add_argument("--data-root",default=".local/mini-imagenet");p.add_argument("--threads",type=int,default=2)
    run(p.parse_args())


if __name__=="__main__":main()
