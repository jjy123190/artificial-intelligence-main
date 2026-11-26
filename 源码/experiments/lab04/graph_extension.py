"""Optional verified graph calibration release, preserving the original v2 evidence."""
from __future__ import annotations
from pathlib import Path
import json,hashlib
from functools import lru_cache
import numpy as np
from .data import digest_file
from .graph_calibration import GraphConfig,predict,apply_transform


NAMES = ("graph_transform.npz", "graph_configuration.json", "graph_episodes.json", "graph_lock.json")


def read_extension(directory, verify=True):
    directory = Path(directory).resolve()
    path = directory / "graph_results.json"
    if not path.is_file():
        return None
    raw = path.read_bytes()
    result = json.loads(raw)
    if result.get("status") != "complete":
        raise ValueError("Graph extension is not complete")
    if set(result.get("validated_files", {})) != set(NAMES):
        raise ValueError("Unexpected graph artifact set")
    if verify:
        observations = []
        for name in (*NAMES, "aligned.pt"):
            candidate = directory / name
            if candidate.is_symlink() or not candidate.resolve().is_relative_to(directory):
                raise ValueError("Unsafe graph artifact path")
            stat = candidate.stat()
            observations.append((name, stat.st_ino, stat.st_size, stat.st_mtime_ns))
        _verify_hashes(str(directory), raw, tuple(observations))
    return result


@lru_cache(maxsize=3)
def _verify_hashes(directory, raw, observations):
    directory = Path(directory)
    result = json.loads(raw)
    if digest_file(directory / "aligned.pt") != result["base_weights_sha256"]:
        raise ValueError("Graph extension backbone changed")
    for name in NAMES:
        entry = result["validated_files"][name]
        if entry["path"] != name or digest_file(directory / name) != entry["sha256"]:
            raise ValueError("Graph extension checksum mismatch: " + name)


def revision(directory, core_bytes):
    extension=Path(directory)/'graph_results.json'
    multi=Path(directory)/'multiview_results.json'
    wide=Path(directory)/'wide_results.json'
    wideview=Path(directory)/'wideview_results.json'
    continued=Path(directory)/'continued_results.json'
    return hashlib.sha256(core_bytes+(extension.read_bytes() if extension.exists() else b'')+
                          (multi.read_bytes() if multi.exists() else b'')+(wide.read_bytes() if wide.exists() else b'')+(wideview.read_bytes() if wideview.exists() else b'')+(continued.read_bytes() if continued.exists() else b'')).hexdigest()


def graph_inputs(directory,shots):
    extension=read_extension(directory)
    if extension is None:raise ValueError('Graph calibration is not installed')
    configs=json.loads((Path(directory)/'graph_configuration.json').read_text(encoding='utf-8'))
    prefix=f'shot{shots}_'
    with np.load(Path(directory)/'graph_transform.npz',allow_pickle=False) as source:
        transform={key[len(prefix):]:source[key] for key in source.files if key.startswith(prefix)}
    return transform,GraphConfig(**configs['configurations'][str(shots)]['config'])


def verify_extension(directory):
    import statistics, math
    from .verify import require, close
    directory=Path(directory)
    result=read_extension(directory)
    if result is None:return None
    records=json.loads((directory/'graph_episodes.json').read_text(encoding='utf-8'))['episodes']
    core=json.loads((directory/'episodes.json').read_text(encoding='utf-8'))['episodes']
    reference={(r['shots'],r['episode']):r for r in core}
    require(len(records)==1200,'Incomplete graph episodes')
    for shots in (1,5):
        selected=sorted((r for r in records if r['shots']==shots),key=lambda r:r['episode'])
        require([r['episode'] for r in selected]==list(range(600)),'Graph episode IDs invalid')
        values=[];deltas=[];vs_full=[]
        for r in selected:
            old=reference[(shots,r['episode'])]
            require(r['support_ids']==old['support_ids'] and r['query_ids']==old['query_ids'],'Unpaired graph samples')
            require(r['query_labels']==old['query_labels'],'Graph score labels changed')
            require(len(r['predictions'])==75,'Wrong graph query count')
            accuracy=sum(a==b for a,b in zip(r['predictions'],r['query_labels']))/75
            close(accuracy,r['accuracy'],'graph per-query accuracy')
            values.append(accuracy);deltas.append(accuracy-old['methods']['baseline']['accuracy'])
            vs_full.append(accuracy-old['methods']['full']['accuracy'])
        row=next(r for r in result['rows'] if r['shots']==shots)
        close(row['accuracy'],statistics.mean(values),'graph accuracy')
        close(row['ci95'],1.96*statistics.stdev(values)/600**.5,'graph interval')
        close(row['paired_delta'],statistics.mean(deltas),'graph paired delta')
        close(row['paired_ci95'],1.96*statistics.stdev(deltas)/600**.5,'graph paired interval')
        close(row['paired_vs_full'],statistics.mean(vs_full),'graph delta vs full')
        close(row['paired_vs_full_ci95'],1.96*statistics.stdev(vs_full)/600**.5,'graph interval vs full')
    return {'verified':True,'episodes_recomputed':1200,'rows':2}
