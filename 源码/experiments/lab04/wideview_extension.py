"""Portable optional wider two-view strategy; no new weights or training."""
from __future__ import annotations
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from .data import digest_file
from .graph_calibration import GraphConfig
from .wide_extension import inference_inputs as wide_inputs
from .wideview_audit import FILES


def read_extension(directory):
    directory=Path(directory).resolve();path=directory/'wideview_results.json'
    if not path.is_file():return None
    raw=path.read_bytes();result=json.loads(raw)
    if result.get('status')!='complete' or set(result.get('files',{}))!=set(FILES):raise ValueError('Incomplete wide-view publication')
    stamps=[]
    for name in (*FILES,'wide.pt'):
        p=directory/name
        if p.is_symlink() or not p.resolve().is_relative_to(directory):raise ValueError('Unsafe wide-view path')
        stat=p.stat();stamps.append((name,stat.st_ino,stat.st_size,stat.st_mtime_ns))
    _verify(str(directory),raw,tuple(stamps));return result


@lru_cache(maxsize=3)
def _verify(directory,raw,stamps):
    directory=Path(directory);result=json.loads(raw)
    for name in FILES:
        entry=result['files'][name]
        if entry['path']!=name or digest_file(directory/name)!=entry['sha256']:raise ValueError('Wide-view checksum mismatch: '+name)
    if digest_file(directory/'wide.pt')!=result['base_weights_sha256']:raise ValueError('Wide-view backbone changed')
    cfg=json.loads((directory/'wideview_configuration.json').read_text(encoding='utf-8'))
    if cfg.get('status')!='locked' or cfg.get('test_accessed') is not False:raise ValueError('Wide-view configuration not locked')


def inference_inputs(directory,shots):
    result=read_extension(directory)
    if result is None:raise ValueError('Wider two-view strategy is not published')
    directory=Path(directory).resolve()
    model,_,_,base=wide_inputs(directory,shots)
    if base['candidate_checkpoint_sha256']!=result['base_weights_sha256']:raise ValueError('Wide-view/wide backbone mismatch')
    cfg=json.loads((directory/'wideview_configuration.json').read_text(encoding='utf-8'))['selected'][str(shots)]
    prefix=f'shot{shots}_'
    with np.load(directory/'wideview_transform.npz',allow_pickle=False) as src:transform={k[len(prefix):]:src[k] for k in src.files if k.startswith(prefix)}
    return model,transform,GraphConfig(**cfg['config']),cfg['views'],result
