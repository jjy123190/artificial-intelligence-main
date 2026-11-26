"""Publish one independently verified wider-model study as an optional comparison.

Does not replace v2 weights, MAML or matched ablations. The website must keep
capacity and training-budget differences visible; no automatic default change.
"""
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
from .data import digest_file
from .experiment import write_json
from .verify_wide import verify
from .artifact_audit import audit_publication


def publish(candidate,study,target):
    candidate,study,target=Path(candidate).resolve(),Path(study).resolve(),Path(target).resolve()
    if target==candidate or target==study or study.is_relative_to(target):raise ValueError('Keep study and publication separate')
    verify(study,target,candidate)
    files={'wide.pt':candidate/'best.pt','wide_transform.npz':study/'wide_transform.npz',
           'wide_configuration.json':study/'wide_configuration.json','wide_episodes.json':study/'wide_episodes.json',
           'wide_validation.json':candidate/'validation_selection.json',
           'wide_training_plan.json':candidate/'plan.json','wide_training.json':candidate/'training.json'}
    for name,source in files.items():
        if not source.is_file() or source.is_symlink():raise ValueError('Missing/unsafe wider-study source')
    result=json.loads((study/'wide_results.json').read_text(encoding='utf-8'))
    result['files']={name:{'path':name,'sha256':digest_file(source)} for name,source in files.items()}
    result['publication_kind']='Separate larger-backbone/training-budget comparison; not matched v2 ablation'
    for name,source in files.items():
        temp=target/(name+'.publish.tmp');shutil.copy2(source,temp)
        if digest_file(temp)!=result['files'][name]['sha256']:raise ValueError('Wider artifact staging mismatch')
        temp.replace(target/name)
    write_json(target/'wide_results.json',result)
    return audit_publication(target)['wide']['rows']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',default='.local/lab04-v3-wide')
    p.add_argument('--study',default='.local/lab04-wide-test');p.add_argument('--target',default='outputs/lab04')
    args=p.parse_args();print(json.dumps(publish(args.candidate,args.study,args.target),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
