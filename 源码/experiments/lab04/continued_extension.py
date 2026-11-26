"""Optional fixed-capacity continued ResNet12 with independently audited evidence."""
from __future__ import annotations
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from .data import digest_file
from .experiment import load_checkpoint
from .graph_calibration import GraphConfig
from .continued_audit import FILES


def read_extension(directory):
    directory=Path(directory).resolve();path=directory/'continued_results.json'
    if not path.is_file():return None
    raw=path.read_bytes();value=json.loads(raw)
    if value.get('status')!='complete' or set(value.get('files',{}))!=set(FILES):raise ValueError('Incomplete continued-model release')
    observed=[]
    for name in FILES:
        item=directory/name
        if item.is_symlink() or not item.resolve().is_relative_to(directory):raise ValueError('Unsafe continued-model path')
        info=item.stat();observed.append((name,info.st_ino,info.st_size,info.st_mtime_ns))
    _verify(str(directory),raw,tuple(observed));return value


@lru_cache(maxsize=3)
def _verify(directory,raw,observed):
    directory=Path(directory);value=json.loads(raw)
    for name in FILES:
        if value['files'][name]['path']!=name or digest_file(directory/name)!=value['files'][name]['sha256']:raise ValueError('Continued-model checksum mismatch: '+name)
    if value['checkpoint_sha256']!=value['files']['continued.pt']['sha256']:raise ValueError('Continued weights differ from tested checkpoint')


@lru_cache(maxsize=3)
def _load_model(directory,checkpoint_sha256):
    model,_=load_checkpoint(Path(directory)/'continued.pt');model.eval();return model


def inference_inputs(directory,shots):
    value=read_extension(directory)
    if value is None:raise ValueError('Continued model has not been published')
    directory=Path(directory).resolve();model=_load_model(str(directory),value['checkpoint_sha256'])
    cfg=json.loads((directory/'continued_configuration.json').read_text(encoding='utf-8'))['selected'][str(shots)]
    with np.load(directory/'continued_transform.npz',allow_pickle=False) as src:
        prefix=f'shot{shots}_';transform={k[len(prefix):]:src[k] for k in src.files if k.startswith(prefix)}
    return model,transform,GraphConfig(**cfg['config']),cfg['views'],value
