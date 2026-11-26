"""Optional larger self-trained model; deliberately separate from v2 calibration ablations."""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import json
import numpy as np
from .data import digest_file
from .experiment import load_checkpoint
from .graph_calibration import GraphConfig


FILES=("wide.pt","wide_transform.npz","wide_configuration.json","wide_episodes.json",
       "wide_validation.json","wide_training_plan.json","wide_training.json")


def read_extension(directory):
    directory=Path(directory).resolve();path=directory/'wide_results.json'
    if not path.is_file():return None
    raw=path.read_bytes();result=json.loads(raw)
    if result.get('status')!='complete' or set(result.get('files',{}))!=set(FILES):
        raise ValueError('Incomplete wider-model publication')
    observations=[]
    for name in FILES:
        p=directory/name
        if p.is_symlink() or not p.resolve().is_relative_to(directory):raise ValueError('Unsafe wider-model path')
        stat=p.stat();observations.append((name,stat.st_ino,stat.st_size,stat.st_mtime_ns))
    _verify(str(directory),raw,tuple(observations))
    return result


@lru_cache(maxsize=2)
def _verify(directory,raw,observations):
    directory=Path(directory);result=json.loads(raw)
    for name in FILES:
        entry=result['files'][name]
        if entry['path']!=name or digest_file(directory/name)!=entry['sha256']:
            raise ValueError('Wider-model checksum mismatch: '+name)
    if result['candidate_checkpoint_sha256']!=result['files']['wide.pt']['sha256']:
        raise ValueError('Wider backbone differs from tested weights')


@lru_cache(maxsize=2)
def _load_model(directory,fingerprint):
    model,_=load_checkpoint(Path(directory)/'wide.pt');model.eval();return model


def inference_inputs(directory,shots):
    result=read_extension(directory)
    if result is None:raise ValueError('Larger-model study is not published')
    directory=Path(directory).resolve()
    model=_load_model(str(directory),result['candidate_checkpoint_sha256'])
    cfg=json.loads((directory/'wide_configuration.json').read_text(encoding='utf-8'))
    prefix=f'shot{shots}_'
    with np.load(directory/'wide_transform.npz',allow_pickle=False) as source:
        transform={k[len(prefix):]:source[k] for k in source.files if k.startswith(prefix)}
    return model,transform,GraphConfig(**cfg['selected'][str(shots)]['config']),result
