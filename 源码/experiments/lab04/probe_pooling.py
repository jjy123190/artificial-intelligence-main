"""Validation-only pooling probe; retains the already trained v2 backbone.

GAP+STD, signed generalized-mean and spatial 2x2 pool features are tested
without any external weights or test-set access. This is not a test score.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .experiment import load_checkpoint,write_json,log,summarize
from .data import image_records,normalized_tensor,digest_file,sample_episode
from .calibration import base_statistics,calibrated_predict,CalibrationConfig,unit_features
from .feature_space import fit_feature_space,transform_features
from .refine import tasks_for,validation_partition
from PIL import Image


def pooling_features(maps):
    mean=maps.mean((-2,-1))
    std=maps.std((-2,-1),unbiased=False)
    gem=torch.sign(maps)*torch.abs(maps).pow(3)
    gem=gem.mean((-2,-1));gem=torch.sign(gem)*gem.abs().pow(1/3)
    grid=F.adaptive_avg_pool2d(maps,(2,2)).flatten(1)
    return {"gap":mean,"moments":torch.cat((mean,std),1),"gem":gem,"grid":grid}


def extract(args,split):
    out=Path(args.output);cache=out/f"{split}_pooling.npz"
    digest=digest_file(args.checkpoint)
    stamp=out/f"{split}_pooling.json"
    if cache.exists() and stamp.exists() and json.loads(stamp.read_text())["checkpoint_sha256"]==digest:
        with np.load(cache,allow_pickle=False) as x:return {k:x[k] for k in x.files}
    if split not in ("train","val"):raise ValueError("Only training and validation pools allowed")
    records=image_records(args.data_root,split)
    names=sorted({row[0] for row in records});mapping={name:i for i,name in enumerate(names)}
    model,_=load_checkpoint(args.checkpoint);model.eval()
    buffers={name:[] for name in ("gap","moments","gem","grid")};labels=[]
    with torch.inference_mode():
        for start in range(0,len(records),64):
            rows=records[start:start+64];images=[]
            for name,path,_ in rows:
                with Image.open(path) as image:images.append(np.asarray(image.convert("RGB")))
                labels.append(mapping[name])
            maps=model.encoder(normalized_tensor(np.stack(images)))
            for name,features in pooling_features(maps).items():buffers[name].append(features.numpy())
            if (start//64+1)%150==0:log("pooling_extract",split=split,images=start+len(rows))
    result={name:np.concatenate(values) for name,values in buffers.items()};result["labels"]=np.array(labels)
    np.savez_compressed(cache,**result);write_json(stamp,{"checkpoint_sha256":digest,"split":split,"images":len(labels),"test_accessed":False})
    return result


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(args.threads)
    train=extract(args,"train");val=extract(args,"val")
    a,b=validation_partition(val["labels"],args.seed)
    result={"test_accessed":False,"checkpoint_sha256":digest_file(args.checkpoint),"shots":{}}
    for shots in (1,5):
        tasks=tasks_for(val["labels"][a],args.seed+shots,shots,120)
        confirm_tasks=tasks_for(val["labels"][b],args.seed+20+shots,shots,300)
        config=CalibrationConfig(mean_weight=.05,variance_blend=.1,temperature=.15,prototype_weight=.15,
            prototype_steps=4,confidence_threshold=0,balanced_queries=True)
        records=[]
        for name in ("gap","moments","gem","grid"):
            for centered in (0.,1.):
                prior=fit_feature_space(train[name],train["labels"],center_strength=centered)
                features=transform_features(val[name][a],prior);scores=[]
                for task in tasks:
                    pred=calibrated_predict(features[task["support_indices"]],task["support_labels"],features[task["query_indices"]],prior,
                        mean=True,variance=True,prototype=True,config=config)
                    scores.append(float(np.mean(pred["predictions"]==task["query_labels"])))
                records.append({"pooling":name,"center_strength":centered,**summarize(scores)})
        shortlist=sorted(records,key=lambda x:-x["accuracy"])[:3]
        for candidate in shortlist:
            name=candidate["pooling"];prior=fit_feature_space(train[name],train["labels"],center_strength=candidate["center_strength"])
            features=transform_features(val[name][b],prior);scores=[]
            for task in confirm_tasks:
                pred=calibrated_predict(features[task["support_indices"]],task["support_labels"],features[task["query_indices"]],prior,
                    mean=True,variance=True,prototype=True,config=config)
                scores.append(float(np.mean(pred["predictions"]==task["query_labels"])))
            candidate["confirmation"]=summarize(scores)
        winner=max(shortlist,key=lambda x:x["confirmation"]["accuracy"])
        result["shots"][str(shots)]={"candidates":records,"shortlist":shortlist,"selected":winner,"config":config.to_dict(),
            "target_reached_on_validation":winner["confirmation"]["accuracy"]>=.95}
        write_json(out/"validation.json",result);log("pooling_validation",shots=shots,selected=winner)
    result["status"]="complete";write_json(out/"validation.json",result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data-root",default=".local/mini-imagenet")
    p.add_argument("--output",default=".local/lab04-v3-pooling");p.add_argument("--checkpoint",default="outputs/lab04/aligned.pt")
    p.add_argument("--threads",type=int,default=1);p.add_argument("--seed",type=int,default=20261125)
    run(p.parse_args())

if __name__=="__main__":main()
