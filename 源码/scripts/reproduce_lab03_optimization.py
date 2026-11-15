"""Re-run validation exploration in an isolated copy of the preserved baseline.

Usage: python -B -X utf8 scripts/reproduce_lab03_optimization.py [--llm]
The default replays search, ML and GA sweeps; --llm also redoes real local Qwen
inference. Never runs test/OOD for configuration selection or overwrites frozen
results. Outputs go to a new directory under outputs/qa.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/'outputs/lab03/optimization'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llm', action='store_true', help='Also run the 0.5B local model validation sweeps (several minutes).')
    args = parser.parse_args()
    protocol = json.loads((FOLDER/'protocol.json').read_text(encoding='utf-8'))
    archive = FOLDER/'baseline_sources.zip'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == protocol['baseline_evidence_sha256'][archive.name]
    assert hashlib.sha256((ROOT/'data/lab03/benchmark.json').read_bytes()).hexdigest() == protocol['baseline_dataset_sha256']
    qa = (ROOT/'outputs/qa').resolve(); qa.mkdir(exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix='lab03-validation-reproduction-', dir=qa)).resolve()
    assert workspace.is_relative_to(qa)
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            target = (workspace/member.filename).resolve()
            if not target.is_relative_to(workspace):
                raise ValueError('Baseline archive entry leaves the isolated workspace')
        package.extractall(workspace)
    dataset = workspace/'data/lab03/benchmark.json'; dataset.parent.mkdir(parents=True)
    shutil.copy2(ROOT/'data/lab03/benchmark.json', dataset)
    output = workspace/'outputs/qa/lab03-optimize-20261112'; output.mkdir(parents=True)
    scripts = ['search_sweep.py', 'hybrid_sweep.py', 'classifier_sweep.py']
    if args.llm:
        scripts += ['llm_sweep.py', 'llm_values.py']
    for name in scripts:
        source = FOLDER/'validation_scripts'/name
        # These are the original exploration scripts; their original imports now
        # resolve only to the archived baseline, not the optimized working tree.
        shutil.copy2(source, output/name)
        subprocess.run([sys.executable, '-B', '-X', 'utf8', str(output/name)], cwd=workspace, check=True)
    print('Validation reproduction saved to:', output)
    print('Frozen project outputs were not changed; no test/OOD metrics were used.')


if __name__ == '__main__':
    main()
