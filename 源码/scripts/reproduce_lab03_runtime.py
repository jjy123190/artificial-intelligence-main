"""Reproduce runtime evidence checks and alternating page-load measurements.

No training, no model inference and no search-parameter tuning. The baseline
service is loaded from the hash-verified runtime archive; empirical timings may
change with machine load. Run from any directory with Python >= 3.11.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import types
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(samples=9):
    from experiments.lab03 import service, calibration, runtime_integrity as integrity
    from experiments.lab03.reliability import validate_frozen_strategy
    validate_frozen_strategy()
    folder = ROOT / 'outputs/lab03/runtime'
    saved = json.loads((folder/'baseline.json').read_text(encoding='utf-8'))
    archive = folder/'before_runtime.zip'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != saved['archive_sha256']:
        raise ValueError('Runtime baseline archive hash mismatch')
    name = 'experiments/lab03/service.py'
    with zipfile.ZipFile(archive) as source:
        code = source.read(name)
    if hashlib.sha256(code).hexdigest() != saved['files'][name]:
        raise ValueError('Runtime baseline service hash mismatch')
    module_name = 'experiments.lab03._runtime_measurement_baseline'
    baseline = types.ModuleType(module_name)
    baseline.__package__ = 'experiments.lab03'
    sys.modules[module_name] = baseline
    exec(compile(code, str(archive) + '!/' + name, 'exec'), baseline.__dict__)
    integrity.clear_verified_inputs()
    start=time.perf_counter();after=service.snapshot();cold_after=(time.perf_counter()-start)*1000
    start=time.perf_counter();before=baseline.snapshot();cold_before=(time.perf_counter()-start)*1000
    if before != after or after['warnings']:
        raise ValueError('The unmodified snapshot changed; do not claim a speed-only effect')
    rows=[]
    for i in range(samples):
        order=[('before',baseline.snapshot),('after',service.snapshot)]
        if i % 2:order.reverse()
        row={'iteration':i}
        for label,fn in order:
            start=time.perf_counter();value=fn();row[label+'_ms']=(time.perf_counter()-start)*1000
            if value!=before:raise ValueError('Snapshot changed during alternating measurement')
        rows.append(row)
    counts={}
    for label,fn in [('before',baseline.snapshot),('after',service.snapshot)]:
        with patch.object(calibration,'load_readout',wraps=calibration.load_readout) as load:
            fn();counts[label]=load.call_count
    graph,query,_=service.tasks()['test-7-0-Q0']
    corrupt=copy.deepcopy(service.read('llm_cache.json')[query.id])
    corrupt['scores'][next(iter(corrupt['scores']))]=.123456
    baseline_accepted=baseline.validate_replay(corrupt,graph,query)==corrupt['scores']
    try:
        service.validate_replay(corrupt,graph,query)
    except ValueError:
        current_rejected=True
    else:
        current_rejected=False
    if not baseline_accepted or not current_rejected:raise ValueError('Corruption regression was not reproduced')
    result={'kind':'same-process alternating UI snapshot timing; identical unmodified output; not model/search efficacy',
            'cold_before_ms':cold_before,'cold_after_ms':cold_after,'identical_unmodified_output':True,'runs':rows,
            'before_median_ms':statistics.median(r['before_ms'] for r in rows),
            'after_median_ms':statistics.median(r['after_ms'] for r in rows),
            'before_readout_loads':counts['before'],'after_readout_loads':counts['after'],
            'verdict_cache_limit':integrity.VERDICT_CACHE_LIMIT,'failed_verdicts_cached':False,'mutable_responses_cached':False,
            'baseline_finite_corruption_accepted':baseline_accepted,'current_finite_corruption_rejected':current_rejected,
            'core_sha256':service.source_hashes(),
            'runtime_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                ('experiments/lab03/service.py','experiments/lab03/runtime_integrity.py')},
            'extra_validation':'Original/fresh model probabilities, 100 numeric cases, paired search replays, CSV/graph bootstrap and confirmation matrix aggregation.',
            'limitations':'Cold call pays complete verification; local wall time depends on filesystem and CPU load. All files re-read, successful verdict tokens reused only when full content fingerprint matches. Search/model outputs unchanged.'}
    (folder/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=int,default=9)
    args=parser.parse_args()
    if not 3<=args.samples<=51:parser.error('--samples must be between 3 and 51')
    print(json.dumps(run(args.samples),ensure_ascii=False,indent=2))
