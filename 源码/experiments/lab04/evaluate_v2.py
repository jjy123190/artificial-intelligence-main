"""Evaluate a locked v2 study, saving complete paired task records.

No hyperparameter choice is made here. Archived v1 weights are evaluated on
identical tasks. A resumable JSONL journal is bound to all input hashes.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .calibration import CalibrationConfig, METHODS, base_statistics, calibrated_predict
from .data import digest_file, load_split, normalized_tensor, sample_episode, selection_digest, verify_splits
from .experiment import extract_features, load_checkpoint, log, summarize, write_json
from .feature_space import fit_feature_space, transform_features
from .models import maml_logits
from .refine import read_cache


METHOD_FLAGS = {**METHODS,
    "transform_only": {"mean":False,"variance":False,"prototype":False},
    "no_transform_full": {"mean":True,"variance":True,"prototype":True},
    "unbalanced_full": {"mean":True,"variance":True,"prototype":True},
    "original_baseline": {"mean":False,"variance":False,"prototype":False},
    "original_full": {"mean":True,"variance":True,"prototype":True},
}


def load_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def run(args):
    directory, previous = Path(args.output), Path(args.previous)
    torch.set_num_threads(args.threads)
    locked_path = directory / "v2_calibration_locked.json"
    locked = json.loads(locked_path.read_text(encoding="utf-8"))
    if locked.get("status") != "locked" or locked.get("test_accessed") is not False:
        raise ValueError("Validation configuration has not been locked before test access")
    if args.episodes < 100:
        raise ValueError("At least 100 episodes are required")
    for name in ("baseline", "aligned"):
        if digest_file(directory / f"{name}.pt") != locked["weights"][name]:
            raise ValueError("Locked weights were changed")
    maml_history = json.loads((directory / "maml_training.json").read_text(encoding="utf-8"))
    if maml_history["status"] != "complete":
        raise ValueError("MAML selection is incomplete")
    old = json.loads((previous / "results.json").read_text(encoding="utf-8"))
    old_configs = {n: CalibrationConfig(**old["calibration"][str(n)]) for n in (1, 5)}
    configs = {n:CalibrationConfig(**locked["configurations"][str(n)]["config"]) for n in (1,5)}
    spaces = {n:locked["configurations"][str(n)]["space"] for n in (1,5)}
    priors, transforms = {}, {}
    for name in ("baseline", "aligned"):
        train = read_cache(directory, name, "train")
        raw = base_statistics(train["features"], train["labels"])
        priors[name] = raw
        bundle = dict(raw)
        for shots in (1, 5):
            transformed = fit_feature_space(train["features"], train["labels"], **spaces[shots])
            transforms[(name, shots)] = transformed
            bundle.update({f"shot{shots}_{key}": value for key,value in transformed.items()})
        np.savez_compressed(directory / f"{name}_prior.npz", **bundle)
        del train
    for name in ("baseline", "aligned"):
        path = previous / f"{name}_prior.npz"
        if digest_file(path) != old["priors"][name]["sha256"]:
            raise ValueError("Original prior hash mismatch")
        priors["original_" + name] = load_npz(path)
    plan = {"seed":args.seed,"episodes":args.episodes,"pool_per_class":600,
            "lock_sha256":digest_file(locked_path),"maml_sha256":digest_file(directory/"maml.pt"),
            "adaptation":maml_history["adaptation"], "old_results_sha256":digest_file(previous/"results.json"),
            "source_sha256":{name:digest_file(Path(__file__).with_name(name)) for name in ("calibration.py","feature_space.py","models.py","data.py","evaluate_v2.py")},
            "method_names":[*METHOD_FLAGS,"maml"]}
    plan_file = directory / "evaluation_plan.json"
    if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
        raise ValueError("Existing evaluation journal has a different plan; use a fresh output directory")
    write_json(plan_file, plan)
    integrity = verify_splits(args.data_root)
    test = load_split(args.data_root,"test",max_per_class=600,seed=args.seed+4)
    log("test_opened_after_lock", images=len(test.images), selection_sha256=selection_digest(test))
    networks, features = {}, {}
    for key, path in [("baseline",directory/"baseline.pt"),("aligned",directory/"aligned.pt"),
                      ("original_baseline",previous/"baseline.pt"),("original_aligned",previous/"aligned.pt")]:
        net,_=load_checkpoint(path)
        if key.startswith("original_") and digest_file(path) != old["weights"][key.removeprefix("original_")]["sha256"]:
            raise ValueError("Original weight hash mismatch")
        log("test_encode",model=key)
        features[key]=extract_features(net,test.images,batch_size=64)
        if not key.startswith("original_"):networks[key]=net
        else:del net
    transformed_features = {(name,n):transform_features(features[name],transforms[(name,n)])
                            for name in ("baseline","aligned") for n in (1,5)}
    maml,_=load_checkpoint(directory/"maml.pt")
    journal_path=directory/"evaluation_episodes.jsonl"
    records=[]
    if journal_path.exists():
        with journal_path.open(encoding="utf-8") as stream:
            records=[json.loads(line) for line in stream if line.strip()]
    # Every completed record is an atomic line. Refuse malformed or duplicate
    # journals rather than silently skipping or evaluating different tasks.
    completed={(r["shots"],r["episode"]):r for r in records}
    if len(completed)!=len(records):raise ValueError("Duplicate journal records")
    for shots in (1,5):
        rng=np.random.default_rng(args.seed+200+shots)
        for index in range(args.episodes):
            task=sample_episode(test.labels,rng,shots=shots,queries=15)
            support_ids=[test.ids[i] for i in task["support_indices"]]
            query_ids=[test.ids[i] for i in task["query_indices"]]
            if (shots,index) in completed:
                row=completed[(shots,index)]
                if row["support_ids"]!=support_ids or row["query_ids"]!=query_ids:
                    raise ValueError("Journal task does not match seeded sampling")
                continue
            record={"episode":index,"shots":shots,"classes":task["classes"].tolist(),
                    "support_ids":support_ids,"query_ids":query_ids,
                    "query_labels":task["query_labels"].tolist(),"methods":{}}
            for method,flags in METHOD_FLAGS.items():
                if method.startswith("original_"):
                    source="original_baseline" if method=="original_baseline" else "original_aligned"
                    embedded, prior, config = features[source],priors[source],old_configs[shots]
                else:
                    source="baseline" if method in ("baseline","no_alignment_full") else "aligned"
                    raw=method in ("baseline","alignment_only","no_transform_full")
                    embedded=features[source] if raw else transformed_features[(source,shots)]
                    prior=priors[source] if raw else transforms[(source,shots)]
                    config=configs[shots]
                    if method=="unbalanced_full":config=replace(config,balanced_queries=False)
                start=time.perf_counter()
                prediction=calibrated_predict(embedded[task["support_indices"]],task["support_labels"],
                    embedded[task["query_indices"]],prior,config=config,**flags)
                elapsed=time.perf_counter()-start
                target=task["query_labels"]
                correct=int(np.sum(prediction["predictions"]==target))
                p=prediction["probabilities"]
                record["methods"][method]={"accuracy":correct/len(target),"correct":correct,"seconds":elapsed,
                    "predictions":prediction["predictions"].tolist(),
                    "negative_log_likelihood":float(-np.log(np.maximum(p[np.arange(len(target)),target],1e-15)).mean())}
            start=time.perf_counter()
            adaptation=maml_history["adaptation"][str(shots)]
            logits=maml_logits(maml,normalized_tensor(test.images[task["support_indices"]]),
                torch.from_numpy(task["support_labels"]),normalized_tensor(test.images[task["query_indices"]]),
                steps=adaptation["steps"],learning_rate=adaptation["learning_rate"])
            elapsed=time.perf_counter()-start
            prediction=logits.argmax(1).numpy(); target=task["query_labels"]
            correct=int(np.sum(prediction==target))
            record["methods"]["maml"]={"accuracy":correct/len(target),"correct":correct,"seconds":elapsed,
                "predictions":prediction.tolist(),"negative_log_likelihood":float(torch.nn.functional.cross_entropy(logits,torch.from_numpy(target)))}
            with journal_path.open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(record,ensure_ascii=False)+"\n");stream.flush()
            records.append(record)
            if (index+1)%50==0:log("evaluate_v2",shots=shots,episodes=index+1)
    rows=[]
    for shots in (1,5):
        selected=sorted((r for r in records if r["shots"]==shots),key=lambda r:r["episode"])
        if len(selected)!=args.episodes:raise ValueError("Unexpected episode count")
        baseline=np.array([r["methods"]["baseline"]["accuracy"] for r in selected])
        for method in (*METHOD_FLAGS,"maml"):
            values=np.array([r["methods"][method]["accuracy"] for r in selected]);delta=values-baseline
            flags=METHOD_FLAGS.get(method,{})
            cfg=old_configs[shots] if method.startswith("original_") else configs[shots]
            rows.append({"method":method,"shots":shots,**summarize(values),
                "paired_delta":float(delta.mean()),"paired_ci95":float(1.96*delta.std(ddof=1)/np.sqrt(len(delta))),
                "mean_episode_seconds":float(np.mean([r["methods"][method]["seconds"] for r in selected])),
                "negative_log_likelihood":float(np.mean([r["methods"][method]["negative_log_likelihood"] for r in selected])),
                "timing_scope":"image_encoding_and_support_adaptation" if method=="maml" else "inference_from_cached_features",
                "transductive":bool(flags.get("prototype")),
                "balanced_queries":bool(flags.get("prototype") and cfg.balanced_queries and method!="unbalanced_full")})
    # Use a seeded non-cherry-picked pool for portable live inference.
    pool_rng=np.random.default_rng(args.seed+300)
    demo_indices=np.concatenate([np.sort(pool_rng.choice(np.flatnonzero(test.labels==label),40,replace=False)) for label in np.unique(test.labels)])
    names=json.loads((Path(args.data_root)/"class_names.json").read_text(encoding="utf-8"))
    np.savez_compressed(directory/"demo_pool.npz",images=test.images[demo_indices],labels=test.labels[demo_indices],
        ids=np.array(test.ids)[demo_indices],class_wnids=np.array(test.class_names),
        class_names=np.array([names.get(w,w).replace("_"," ") for w in test.class_names]))
    history=json.loads((directory/"training.json").read_text(encoding="utf-8"));history["maml"]=maml_history
    train_data=json.loads((directory/"training_data.json").read_text(encoding="utf-8"))
    protocol={"dataset":"mini-ImageNet","split":"Ravi & Larochelle 64/16/20", "image_size":[84,84],
        "training_classes":64,"validation_classes":16,"test_classes":20,"train_per_class":600,
        "evaluation_pool_per_class":600,"training_images":38400,"validation_images":9600,"test_images":12000,
        "ways":5,"shots":[1,5],"queries_per_class":15,"test_episodes_per_shot":args.episodes,"seed":args.seed,
        "full_training_split":True,"full_evaluation_split":True,"no_test_tuning":True,
        "train_selection_sha256":train_data["train_sha256"],"test_selection_sha256":selection_digest(test),
        "maml_train_inner_steps":1,"maml_train_queries_per_class":5,"maml_test_adaptation":maml_history["adaptation"],
        "scope":"CPU lightweight ResNet12 follow-up; prior public test results exist; no test-based parameter selection"}
    baseline=networks["baseline"]
    weights={name:{"path":f"{name}.pt","sha256":digest_file(directory/f"{name}.pt")} for name in ("baseline","aligned","maml")}
    limitations=["特征中心只由训练类计算；超参数在验证类选定，测试前锁定。原测试基准已有公开结果，属于后续实验而非全新外部验证。",
                 "校准搜索与复核图片互斥，但仍来自相同验证类别，复核结果参与选型，不能视为独立最终成绩。",
                 "均衡转导假定每类查询数量相同，适用于本评测协议，不保证适用于真实不均衡业务。",
                 "扩大轻量骨干并增加训练预算也会带来改善；不能把相对原版的全部提升归因于校准模块。"]
    result={"schema_version":2,"status":"complete","completed_utc":datetime.now(timezone.utc).isoformat(),
        "protocol":protocol,"dataset_integrity":integrity,
        "model":{"name":"ResNet12-Lite","widths":list(baseline.widths),"embedding_dim":baseline.widths[-1],
                 "parameters_baseline":sum(p.numel() for p in baseline.parameters()),"parameters_maml":sum(p.numel() for p in maml.parameters()),
                 "normalization":"GroupNorm(4)","convolution_layers":12,"standard_width_benchmark":False},
        "runtime":{"torch":str(torch.__version__),"device":"cpu","threads":args.threads},"training":history,
        "weights":weights,"rows":rows,"calibration":{str(n):configs[n].to_dict() for n in (1,5)},
        "maml_adaptation":maml_history["adaptation"],
        "priors":{name:{"path":f"{name}_prior.npz","sha256":digest_file(directory/f"{name}_prior.npz")} for name in ("baseline","aligned")},
        "demo_pool":{"path":"demo_pool.npz","sha256":digest_file(directory/"demo_pool.npz"),"classes":20,"images_per_class":40,"images":800,
                     "source":"seeded random sample of held-out test images; not selected for accuracy"},
        "study":{"version":"v2","name":"验证集选型的轻量模型与稳健校准","selection":locked["selection"],
                 "feature_spaces":{str(n):spaces[n] for n in (1,5)},"limitations":limitations},
        "caveats":limitations+["95% CIs are conditional on one training seed, not independent repeated training.",
                  "Cached-feature timing excludes image encoding. Softmax scores are not calibrated probabilities."],
        "selection_lock_sha256":digest_file(locked_path), "episode_journal_sha256":digest_file(journal_path)}
    write_json(directory/"protocol.json",protocol)
    write_json(directory/"calibration_config.json",result["calibration"])
    write_json(directory/"training.json",history)
    write_json(directory/"episodes.json",{"schema_version":2,"seed":args.seed+200,"episodes":records})
    shutil.copy2(Path(args.data_root)/"dataset_manifest.json",directory/"dataset_manifest.json")
    write_json(directory/"results.json",result)
    log("evaluation_complete",rows=[{k:r[k] for k in ("method","shots","accuracy","ci95","paired_delta")} for r in rows])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",required=True);p.add_argument("--output",required=True)
    p.add_argument("--previous",default="outputs/lab04")
    p.add_argument("--seed",type=int,default=20261124);p.add_argument("--episodes",type=int,default=600)
    p.add_argument("--threads",type=int,default=4)
    run(p.parse_args())


if __name__=="__main__":main()
