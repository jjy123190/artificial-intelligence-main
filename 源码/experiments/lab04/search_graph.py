"""Validation-only feature/classifier search for the 95% aspiration.

This study does not access the test split and cannot publish a test score.
Each search record is retained, including failures to approach the target.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
import json
from datetime import datetime,timezone
import numpy as np
from .data import digest_file
from .experiment import write_json,log,summarize
from .refine import read_cache, validation_partition, tasks_for
from .graph_calibration import GraphConfig, centered_whitening, apply_transform, predict
from .feature_space import select_prior, transform_features
from .calibration import calibrated_predict,CalibrationConfig


def score(x,tasks,config):
    values=[]
    for task in tasks:
        prediction=predict(x[task["support_indices"]],task["support_labels"],x[task["query_indices"]],config)
        values.append(float(np.mean(prediction["predictions"]==task["query_labels"])))
    return summarize(values)


def run(args):
    source,out=Path(args.features),Path(args.output);out.mkdir(parents=True,exist_ok=True)
    train=read_cache(source,"aligned","train");val=read_cache(source,"aligned","val")
    selection,confirmation=validation_partition(val["labels"],args.seed)
    cfg=GraphConfig(neighbors=10,alpha=.9,power=3.,temperature=.2,prototype_weight=.15,prototype_steps=8,graph_weight=.5,balanced=True)
    plan={"seed":args.seed,"train_cache_sha256":digest_file(source/"aligned_train_features.npz"),
          "validation_cache_sha256":digest_file(source/"aligned_val_features.npz"),"test_accessed":False,
          "search_episodes":120,"confirmation_episodes":300,"target":.95,
          "center_strengths":[0.,1.],"whitening_exponents":[0.,.25,.5],
          "neighborhoods":[5,10,20],"graph_weights":[0.,.5,1.],"alpha":[.8,.95],
          "refine_temperatures":[.1,.2,.4],"refine_prototype_weights":[.1,.4,.8]}
    write_json(out/"plan.json",plan)
    result={"plan":plan,"shots":{}}
    for shots in (1,5):
        selection_labels=val["labels"][selection];confirmation_labels=val["labels"][confirmation]
        tasks=tasks_for(selection_labels,args.seed+10+shots,shots,120)
        confirm_tasks=tasks_for(confirmation_labels,args.seed+20+shots,shots,300)
        spaces=[]
        for center in plan["center_strengths"]:
            for exponent in plan["whitening_exponents"]:
                transform=centered_whitening(train["features"],center,exponent)
                x=apply_transform(val["features"][selection],transform)
                item={"space":{"strength":center,"exponent":exponent},"config":cfg.to_dict(),**score(x,tasks,cfg)}
                spaces.append(item)
        finalists=sorted(spaces,key=lambda x:-x["accuracy"])[:2];records=[]
        for item in finalists:
            transform=centered_whitening(train["features"],**item["space"])
            x=apply_transform(val["features"][selection],transform)
            for neighbors in plan["neighborhoods"]:
                for graph_weight in plan["graph_weights"]:
                    for alpha in plan["alpha"]:
                        config=replace(cfg,neighbors=neighbors,graph_weight=graph_weight,alpha=alpha)
                        records.append({"space":item["space"],"config":config.to_dict(),**score(x,tasks,config)})
        winner=max(records,key=lambda x:x["accuracy"])
        transform=centered_whitening(train["features"],**winner["space"])
        x=apply_transform(val["features"][selection],transform)
        for temperature in plan["refine_temperatures"]:
            for weight in plan["refine_prototype_weights"]:
                config=replace(GraphConfig(**winner["config"]),temperature=temperature,prototype_weight=weight)
                records.append({"space":winner["space"],"config":config.to_dict(),**score(x,tasks,config)})
        unique={json.dumps({"space":row["space"],"config":row["config"]},sort_keys=True):row for row in records}
        shortlist=sorted(unique.values(),key=lambda x:-x["accuracy"])[:5]
        for item in shortlist:
            transform=centered_whitening(train["features"],**item["space"])
            x=apply_transform(val["features"][confirmation],transform)
            item["confirmation"]=score(x,confirm_tasks,GraphConfig(**item["config"]))
        best=max(shortlist,key=lambda x:x["confirmation"]["accuracy"])
        # Paired existing v2 reference on these same validation tasks.
        with np.load(source/"aligned_prior.npz",allow_pickle=False) as p:prior={k:p[k] for k in p.files}
        prior=select_prior(prior,shots,transformed=True)
        x=transform_features(val["features"][confirmation],prior)
        metadata=json.loads((source/"results.json").read_text(encoding="utf-8"))
        existing_cfg=CalibrationConfig(**metadata["calibration"][str(shots)])
        scores=[]
        for task in confirm_tasks:
            pred=calibrated_predict(x[task["support_indices"]],task["support_labels"],x[task["query_indices"]],prior,mean=True,variance=True,prototype=True,config=existing_cfg)
            scores.append(float(np.mean(pred["predictions"]==task["query_labels"])))
        result["shots"][str(shots)]={"spaces":spaces,"candidates":records,"shortlist":shortlist,"selected":best,"v2_reference":summarize(scores),"target_reached_on_validation":best["confirmation"]["accuracy"]>=.95}
        write_json(out/"validation.json",result)
        log("graph_validation",shots=shots,selected=best,v2_reference=summarize(scores))
    result["status"]="complete";result["completed_utc"]=datetime.now(timezone.utc).isoformat()
    write_json(out/"validation.json",result);log("graph_search_complete")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--features",default=".local/lab04-v2")
    p.add_argument("--output",default=".local/lab04-v3-graph");p.add_argument("--seed",type=int,default=20261125)
    run(p.parse_args())

if __name__=="__main__":main()
