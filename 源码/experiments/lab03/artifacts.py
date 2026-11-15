"""Provenance for generated artifacts; stale data must not masquerade as current."""
from __future__ import annotations
import hashlib
from .dataset import ROOT

CORE_SOURCES = ('artifacts.py', 'domain.py', 'dataset.py', 'search.py', 'learning.py', 'genetic.py', 'evaluate.py', 'llm.py', 'calibration.py', 'diagnostics.py')
MODEL_SOURCES = ('artifacts.py', 'domain.py', 'dataset.py', 'search.py', 'learning.py', 'genetic.py')

def source_hashes(names=CORE_SOURCES):
    hashes = {'experiments/lab03/' + name: hashlib.sha256((ROOT / 'experiments/lab03' / name).read_bytes()).hexdigest()
            for name in names}
    hashes['data/lab03/ga_warm_start.json'] = hashlib.sha256((ROOT/'data/lab03/ga_warm_start.json').read_bytes()).hexdigest()
    hashes['data/lab03/llm_model_manifest.json'] = hashlib.sha256((ROOT/'data/lab03/llm_model_manifest.json').read_bytes()).hexdigest()
    return hashes

def provenance():
    return {'dataset_sha256': hashlib.sha256((ROOT / 'data/lab03/benchmark.json').read_bytes()).hexdigest(),
            'source_sha256': source_hashes(MODEL_SOURCES)}

def current_artifact(record):
    return isinstance(record, dict) and record.get('provenance') == provenance()

def current_manifest(record):
    if not isinstance(record, dict):
        return False
    expected = provenance()
    hashes = record.get('source_sha256')
    return (record.get('dataset_sha256') == expected['dataset_sha256'] and isinstance(hashes, dict)
            and all(hashes.get(p) == h for p, h in source_hashes().items()))
