"""Pre-registered validation-only comparison of cached wide/residual/fused features.

Never loads test data; checkpoints, caches, source and grid are fingerprinted
before scoring. Resuming rejects changed inputs, keeps every negative result.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .calibration import unit_features
from .data import digest_file
from .experiment import log, summarize, write_json
from .graph_calibration import GraphConfig, apply_transform, predict
from .refine import read_cache, tasks_for, validation_partition
from .supervised_whitening import fit


def paired_stats(values, reference):
    delta = np.asarray(values) - np.asarray(reference)
    return {"mean": float(delta.mean()), "ci95": float(1.96 * delta.std(ddof=1) / np.sqrt(len(delta))),
            "wins": int((delta > 0).sum()), "ties": int((delta == 0).sum()), "losses": int((delta < 0).sum())}


def load_wide(directory, split):
    if split not in ("train", "val"):
        raise ValueError("Selection is train/validation only")
    cache = directory / f"{split}_features.npz"
    stamp = json.loads((directory / f"{split}_cache.json").read_text(encoding="utf-8"))
    if stamp["checkpoint_sha256"] != digest_file(directory / "best.pt") or stamp["cache_sha256"] != digest_file(cache):
        raise ValueError("Unverified wider feature cache")
    with np.load(cache, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def compose(wide, light, mix):
    if not 0 <= mix < 1:
        raise ValueError("Light feature weight must lie in [0, 1)")
    if len(wide) != len(light):
        raise ValueError("Feature rows differ")
    if mix == 0:
        return unit_features(wide)
    return np.concatenate((unit_features(wide) * np.sqrt(1-mix), unit_features(light) * np.sqrt(mix)), axis=1)


def run(args):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    wide = Path(args.wide); light = Path(args.light); artifacts = Path(args.artifacts)
    source_names = ("study_cached_refinement.py", "graph_calibration.py", "supervised_whitening.py", "calibration.py", "refine.py", "data.py")
    inputs = [wide/"best.pt", light/"aligned.pt", artifacts/"wide_configuration.json", artifacts/"wide_transform.npz"]
    inputs += [wide/f"{split}_{kind}" for split in ("train", "val") for kind in ("features.npz", "cache.json")]
    inputs += [light/f"aligned_{split}_features.npz" for split in ("train", "val")]
    inputs += [light/"feature_cache.json"]
    candidates = [{"light_weight": mix, "residual_mix": residual, "exponent": exponent}
                  for mix in (0., .25, .5) for residual in (0., .5, 1.) for exponent in (.25, .5)]
    plan = {"schema_version": 1, "seed": args.seed, "search_episodes": args.search, "confirmation_episodes": args.confirm,
            "shortlist": 4, "candidates": candidates, "test_accessed": False,
            "source_hashes": {name: digest_file(Path(__file__).with_name(name)) for name in source_names},
            "inputs": {str(p): digest_file(p) for p in inputs},
            "selection_rule": "search top 4, select on confirmation by paired lower bound vs frozen published wide reference; no candidate published automatically",
            "limitations": "Same 16 repeatedly used validation classes; image-disjoint pools are not an independent generalization estimate. Fusion adds a second encoder."}
    path = out/"selection_plan.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Frozen selection plan or input identity changed")
    if not path.exists(): write_json(path, plan)
    result_path = out/"selection.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous.get("status") == "complete":
            log("cached_refinement_already_complete"); return
    training = load_wide(wide, "train"); validation = load_wide(wide, "val")
    train_light = read_cache(light, "aligned", "train"); val_light = read_cache(light, "aligned", "val")
    for first, second in ((training, train_light), (validation, val_light)):
        if not np.array_equal(first["ids"], second["ids"]) or not np.array_equal(first["labels"], second["labels"]):
            raise ValueError("Wide/light cache rows are not exactly paired")
    train_classes = {str(x).split("/")[1] for x in training["ids"]}
    val_classes = {str(x).split("/")[1] for x in validation["ids"]}
    if train_classes & val_classes or len(train_classes) != 64 or len(val_classes) != 16:
        raise ValueError("Train/validation class isolation failed")
    search, confirmation = validation_partition(validation["labels"], args.seed)
    if set(search) & set(confirmation): raise ValueError("Validation image pools overlap")
    write_json(out/"data_identity.json", {"training_ids_sha256": hashlib.sha256("\n".join(training["ids"]).encode()).hexdigest(),
        "validation_ids_sha256": hashlib.sha256("\n".join(validation["ids"]).encode()).hexdigest(),
        "train_classes": sorted(train_classes), "validation_classes": sorted(val_classes), "test_accessed": False,
        "search_images": len(search), "confirmation_images": len(confirmation)})
    frozen = json.loads((artifacts/"wide_configuration.json").read_text(encoding="utf-8"))
    journal = out/"scores.jsonl"; scores = {}
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            record = json.loads(line); scores[record["key"]] = record["values"]
    def evaluate(key, features, tasks, config):
        if key in scores: return scores[key]
        values=[]; start=time.perf_counter()
        for task in tasks:
            prediction=predict(features[task["support_indices"]],task["support_labels"],features[task["query_indices"]],config)
            values.append(float(np.mean(prediction["predictions"] == task["query_labels"])))
        with journal.open("a",encoding="utf-8",newline="\n") as stream:
            stream.write(json.dumps({"key":key,"values":values},allow_nan=False)+"\n")
        scores[key]=values; log("cached_candidate",key=key,accuracy=np.mean(values),seconds=time.perf_counter()-start)
        return values
    result={"status":"selecting","plan_sha256":digest_file(path),"test_accessed":False,"shots":{}}
    transforms={}
    for shot in (1,5):
        config=GraphConfig(**frozen["selected"][str(shot)]["config"])
        tasks=tasks_for(validation["labels"][search],args.seed+100+shot,shot,args.search)
        checks=tasks_for(validation["labels"][confirmation],args.seed+200+shot,shot,args.confirm)
        with np.load(artifacts/"wide_transform.npz",allow_pickle=False) as src:
            prefix=f"shot{shot}_"; ref_transform={k[len(prefix):]:src[k] for k in src.files if k.startswith(prefix)}
        ref=apply_transform(validation["features"],ref_transform)
        ref_search=evaluate(f"{shot}/reference/search",ref[search],tasks,config)
        ref_confirm=evaluate(f"{shot}/reference/confirmation",ref[confirmation],checks,config)
        records=[]
        for index,candidate in enumerate(candidates):
            mix=candidate["light_weight"]
            train=compose(training["features"],train_light["features"],mix)
            val=compose(validation["features"],val_light["features"],mix)
            transform=fit(train,training["labels"],strength=1.,exponent=candidate["exponent"],residual_mix=candidate["residual_mix"])
            transformed=apply_transform(val,transform)
            values=evaluate(f"{shot}/{index}/search",transformed[search],tasks,config)
            records.append({"index":index,"candidate":candidate,"config":asdict(config),"search":summarize(values),"search_paired":paired_stats(values,ref_search)})
        shortlist=sorted(records,key=lambda row:(-row["search"]["accuracy"],row["index"]))[:plan["shortlist"]]
        for row in shortlist:
            c=row["candidate"]; train=compose(training["features"],train_light["features"],c["light_weight"])
            val=compose(validation["features"],val_light["features"],c["light_weight"])
            transform=fit(train,training["labels"],strength=1.,exponent=c["exponent"],residual_mix=c["residual_mix"])
            values=evaluate(f"{shot}/{row['index']}/confirmation",apply_transform(val,transform)[confirmation],checks,config)
            row["confirmation"]=summarize(values); row["confirmation_paired"]=paired_stats(values,ref_confirm)
        winner=max(shortlist,key=lambda row:(row["confirmation_paired"]["mean"]-row["confirmation_paired"]["ci95"],-row["index"]))
        c=winner["candidate"]; train=compose(training["features"],train_light["features"],c["light_weight"])
        transform=fit(train,training["labels"],strength=1.,exponent=c["exponent"],residual_mix=c["residual_mix"])
        transforms.update({f"shot{shot}_{k}":v for k,v in transform.items()})
        result["shots"][str(shot)]={"selected":winner,"candidates":records,"reference":{"search":summarize(ref_search),"confirmation":summarize(ref_confirm)},
            "promising":winner["confirmation_paired"]["mean"]-winner["confirmation_paired"]["ci95"]>0}
        write_json(result_path,result); log("cached_refinement_shot",shot=shot,selected=winner)
    np.savez_compressed(out/"transform.npz",**transforms)
    result.update({"status":"complete","completed_utc":datetime.now(timezone.utc).isoformat(),"transform_sha256":digest_file(out/"transform.npz")})
    write_json(result_path,result)
    write_json(out/"selection_lock.json",{"status":"locked","selection_sha256":digest_file(result_path),"plan_sha256":digest_file(path),
        "transform_sha256":digest_file(out/"transform.npz"),"locked_utc":datetime.now(timezone.utc).isoformat(),"test_accessed":False})
    log("cached_refinement_complete")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",default=".local/lab04-cached-refinement-20261125")
    p.add_argument("--wide",default=".local/lab04-v3-wide")
    p.add_argument("--light",default=".local/lab04-v2")
    p.add_argument("--artifacts",default="outputs/lab04")
    p.add_argument("--seed",type=int,default=20261127)
    p.add_argument("--search",type=int,default=160)
    p.add_argument("--confirm",type=int,default=400)
    run(p.parse_args())


if __name__=="__main__": main()
