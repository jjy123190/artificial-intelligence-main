"""Validation-only wider ResNet12 candidate training, no test access.

Fixed predeclared optimizer/augmentation/metric objective. Checkpoints resume
atomically. Validation chooses checkpoints and early stopping, never test data.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime, timezone
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from .data import load_split, digest_file, selection_digest
from .experiment import load_checkpoint, save_checkpoint, write_json, extract_features, log
from .calibration import base_statistics, calibrated_predict, CalibrationConfig
from .metric_training import widen_resnet12, break_channel_symmetry, balanced_batches, augment_images, episode_loss
from .refine import tasks_for


def score_validation(network, validation, tasks):
    features=extract_features(network,validation.images,batch_size=64)
    # Baseline diagnostic: no query batch statistics or labels in prediction.
    prior={"means":np.zeros((1,features.shape[1])),"variances":np.ones((1,features.shape[1]))}
    scores={}
    for shots in (1,5):
        values=[]
        for task in tasks[shots]:
            pred=calibrated_predict(features[task["support_indices"]],task["support_labels"],features[task["query_indices"]],prior)
            values.append(float(np.mean(pred["predictions"]==task["query_labels"])))
        scores[str(shots)]=float(np.mean(values))
    return {"1shot":scores["1"],"5shot":scores["5"],"mean":(scores["1"]+scores["5"])/2}


def train(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    if root.resolve()==Path(args.initial).resolve().parent:
        raise ValueError("Never train into the published model directory")
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed)
    plan={**vars(args),"initial_sha256":digest_file(args.initial),"test_accessed":False,
          "objective":"CE label smoothing .05 + metric coefficient; train classes only"}
    plan_file=root/"plan.json"
    if plan_file.exists() and json.loads(plan_file.read_text())!=plan:
        raise ValueError("Candidate plan differs from existing immutable plan")
    write_json(plan_file,plan)
    original,_=load_checkpoint(args.initial)
    network=widen_resnet12(original,factor=2);del original
    break_channel_symmetry(network,scale=.015,generator=torch.Generator().manual_seed(args.seed))
    training=load_split(args.data_root,"train",max_per_class=600,seed=args.seed)
    validation=load_split(args.data_root,"val",max_per_class=200,seed=args.seed+1)
    if set(training.class_names)&set(validation.class_names):raise ValueError("Class leakage")
    write_json(root/"data.json",{"training_images":len(training.images),"validation_images":len(validation.images),
        "training_sha256":selection_digest(training),"validation_sha256":selection_digest(validation),"test_accessed":False})
    tasks={n:tasks_for(validation.labels,args.seed+30+n,n,100) for n in (1,5)}
    optimizer=torch.optim.AdamW(network.parameters(),lr=args.learning_rate,weight_decay=.0005)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=.00001)
    rng=np.random.default_rng(args.seed);generator=torch.Generator().manual_seed(args.seed)
    history=[];best=-1.;best_epoch=0;resume_file=root/"resume.pt"
    if resume_file.exists():
        saved=torch.load(resume_file,map_location="cpu",weights_only=True)
        if saved["plan"]!=plan:raise ValueError("Resume plan mismatch")
        network.load_state_dict(saved["state"]);optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"]);rng.bit_generator.state=saved["rng"]
        generator.set_state(saved["generator"]);torch.set_rng_state(saved["torch_rng"])
        history,best,best_epoch=saved["history"],saved["best"],saved["best_epoch"]
        log("resume_v3",epochs=len(history))
    elif not history:
        score=score_validation(network,validation,tasks)
        best=score["mean"]
        save_checkpoint(root/"best.pt",network,purpose="widened metric candidate",metadata={**plan,"selected_epoch":0,"validation":score})
        write_json(root/"initial_validation.json",score);log("initial_validation",**score)
    for epoch in range(len(history),args.epochs):
        network.train();started=time.perf_counter();total_ce=total_metric=0.;correct=count=0
        for batch_index,ids in enumerate(balanced_batches(training.labels,rng)):
            images=augment_images(training.images[ids],generator)
            labels=torch.from_numpy(training.labels[ids])
            optimizer.zero_grad(set_to_none=True)
            logits,features=network(images,return_features=True)
            ce=F.cross_entropy(logits,labels,label_smoothing=.05)
            metric=episode_loss(features,labels)
            loss=ce+args.metric_weight*metric
            loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),10.)
            optimizer.step();count+=len(ids)
            total_ce+=float(ce.detach())*len(ids);total_metric+=float(metric.detach())*len(ids)
            correct+=int((logits.detach().argmax(1)==labels).sum())
            if (batch_index+1)%200==0:log("train_v3_batch",epoch=epoch+1,batches=batch_index+1,seconds=time.perf_counter()-started)
        scheduler.step()
        score=score_validation(network,validation,tasks)
        row={"epoch":epoch+1,"ce_loss":total_ce/count,"metric_loss":total_metric/count,
             "training_accuracy":correct/count,"validation":score,"seconds":time.perf_counter()-started}
        history.append(row)
        if score["mean"]>best:
            best,best_epoch=score["mean"],epoch+1
            save_checkpoint(root/"best.pt",network,purpose="widened metric candidate",metadata={**plan,"selected_epoch":best_epoch,"validation":score})
        temporary=resume_file.with_suffix(".tmp")
        torch.save({"plan":plan,"state":network.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
            "rng":rng.bit_generator.state,"generator":generator.get_state(),"torch_rng":torch.get_rng_state(),
            "history":history,"best":best,"best_epoch":best_epoch},temporary);temporary.replace(resume_file)
        write_json(root/"training.json",{"status":"running","rows":history,"best_epoch":best_epoch,"best_mean":best})
        log("train_v3_epoch",**row)
        if epoch+1>=8 and epoch+1-best_epoch>=args.patience:
            log("early_stop",epoch=epoch+1,best_epoch=best_epoch);break
    write_json(root/"training.json",{"status":"complete","rows":history,"best_epoch":best_epoch,"best_mean":best,
        "test_accessed":False,"completed_utc":datetime.now(timezone.utc).isoformat()})
    log("v3_training_complete",best_epoch=best_epoch,best_mean=best)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",required=True);p.add_argument("--output",required=True)
    p.add_argument("--initial",default="outputs/lab04/baseline.pt")
    p.add_argument("--epochs",type=int,default=24);p.add_argument("--patience",type=int,default=8)
    p.add_argument("--learning-rate",type=float,default=.0005);p.add_argument("--metric-weight",type=float,default=.25)
    p.add_argument("--threads",type=int,default=4);p.add_argument("--seed",type=int,default=20261125)
    train(p.parse_args())

if __name__=="__main__":main()
