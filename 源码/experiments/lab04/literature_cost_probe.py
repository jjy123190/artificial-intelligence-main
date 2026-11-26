"""Synthetic CPU cost probe only: no image data, tuning, training or accuracy.

A recipe is locked before timing. Median batch time is a feasibility indicator,
not an end-to-end deployment benchmark or a reliable full training ETA.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import time

import numpy as np
import psutil
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from .data import digest_file
from .feature_distillation import FeatureProjector,training_loss
from .models import ResNet12
from .reference_model import load_reference
from .study_literature import read,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='outputs/lab04/next_study/literature_20261125/cost_probe')
    p.add_argument('--teacher',default='.local/lab04-reference-feat/feat-1-shot.pth')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    plan={'kind':'synthetic_cpu_feasibility_only','batch':8,'warmups':1,'repetitions':3,'threads':2,'seed':20261201,
          'test_accessed':False,'source_sha256':digest_file(__file__),'teacher_sha256':digest_file(args.teacher),
          'methods':{'small':{'widths':[16,32,64,128],'work':'one-view CE forward+backward'},
                     'medium':{'widths':[32,64,128,256],'work':'one-view CE forward+backward'},
                     'small_teacher':{'widths':[16,32,64,128],'work':'two-view student+frozen teacher forward, full proposed KD loss backward'}},
          'limitations':['Random normalized tensors, no dataset decoding/augmentation/validation/optimizer step.',
                         'Teacher CPU cost included in KD, training projector is not deployment cost.',
                         'Tiny batch and three repeats under variable host load: planning estimate only.']}
    if (out/'plan.json').exists():
        if read(out/'plan.json')!=plan:raise ValueError('Cost probe plan changed')
        if (out/'result.json').exists():print(json.dumps(read(out/'result.json')));return
    else:write(out/'plan.json',plan)
    torch.manual_seed(plan['seed']);torch.set_num_threads(plan['threads'])
    x=torch.randn(8,3,84,84);y=torch.arange(8);rows=[]
    with threadpool_limits(limits=2):
        for name,spec in plan['methods'].items():
            model=ResNet12(widths=spec['widths']);model.train()
            teacher=load_reference(args.teacher) if name=='small_teacher' else None
            if teacher is not None:
                teacher.requires_grad_(False);teacher.eval();projector=FeatureProjector(128,640)
            samples=[];rss_start=psutil.Process().memory_info().rss
            for repeat in range(4):
                model.zero_grad(set_to_none=True)
                if teacher is not None:projector.zero_grad(set_to_none=True)
                start=time.perf_counter()
                logits,z=model(x,return_features=True)
                if teacher is None:loss=F.cross_entropy(logits,y)
                else:
                    with torch.no_grad():ta=teacher(x);tb=teacher(x.flip(-1))
                    zb=model.features(x.flip(-1))
                    loss=training_loss(logits,y,z,zb,ta,tb,projector,z.sum()*0)['total']
                loss.backward()
                seconds=time.perf_counter()-start
                if repeat:samples.append(seconds)
            rows.append({'method':name,'parameters':sum(p.numel() for p in model.parameters()),
                         'seconds':samples,'median_batch_seconds':statistics.median(samples),
                         'rough_38400_image_epoch_minutes':statistics.median(samples)*(38400/8)/60,
                         'rss_after_mib':psutil.Process().memory_info().rss/(1024**2),
                         'rss_delta_mib':(psutil.Process().memory_info().rss-rss_start)/(1024**2)})
            del model,teacher
    result={'status':'complete','test_accessed':False,'plan_sha256':digest_file(out/'plan.json'),
            'cpu_only':not torch.cuda.is_available(),'torch_version':torch.__version__,
            'host_available_mib':psutil.virtual_memory().available/(1024**2),'rows':rows,
            'completed_utc':datetime.now(timezone.utc).isoformat(),'limitations':plan['limitations']}
    write(out/'result.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
