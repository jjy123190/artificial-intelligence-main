"""Publish a verified candidate while retaining the full original experiment.

Run only when inference services are idle/stopped. Each file is staged and
atomically replaced; results.json is published last. Concurrent consumers fail
closed on checksum mismatch rather than mixing checkpoint generations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .data import digest_file
from .verify import verify_artifacts


FILES = ("baseline.pt","aligned.pt","maml.pt","warm.pt","baseline_prior.npz","aligned_prior.npz",
         "demo_pool.npz","episodes.json","protocol.json","training.json","calibration_config.json",
         "dataset_manifest.json","v2_calibration_locked.json","v2_validation_tuning.json","maml_training.json",
         "evaluation_plan.json","evaluation_episodes.jsonl","edge_benchmark.json",
         "training_plan.json","training_data.json")


def publish(candidate, target):
    candidate, target = Path(candidate).resolve(), Path(target).resolve()
    if candidate == target or candidate.is_relative_to(target) or target.is_relative_to(candidate):
        raise ValueError("Candidate and publication directory must be disjoint")
    verify_artifacts(candidate)
    # Existing extensions are bound to an exact aligned backbone and protocol.
    # Do not replace a core study beneath them, then leave stale scores in the UI.
    candidate_result=json.loads((candidate/"results.json").read_text(encoding="utf-8"))
    for extension_name in ("graph_results.json", "multiview_results.json"):
        extension_path=target/extension_name
        if extension_path.exists():
            current_extension=json.loads(extension_path.read_text(encoding="utf-8"))
            if current_extension["base_weights_sha256"] != candidate_result["weights"]["aligned"]["sha256"]:
                raise ValueError("Existing " + extension_name + " is bound to another backbone; publish into a new directory")
            current_core=json.loads((target/"results.json").read_text(encoding="utf-8"))
            if current_core["protocol"] != candidate_result["protocol"]:
                raise ValueError("Cannot replace the paired protocol beneath published extensions")
    for name in FILES:
        if name != "edge_benchmark.json" and not (candidate / name).is_file():
            raise FileNotFoundError(candidate / name)
    target.mkdir(parents=True,exist_ok=True)
    history = target / "history" / "v1"
    if (target / "results.json").exists():
        current=json.loads((target/"results.json").read_text(encoding="utf-8"))
        if current.get("schema_version")==1:
            if history.exists():
                archived=json.loads((history/"results.json").read_text(encoding="utf-8"))
                if current != archived:raise ValueError("Existing v1 archive does not match current result")
            else:
                history.mkdir(parents=True)
                for path in target.iterdir():
                    if path.is_file():shutil.copy2(path,history/path.name)
                verify_artifacts(history)
        elif current.get("schema_version")!=2:
            raise ValueError("Unknown existing result version")
    for name in (*FILES,"results.json"):
        source=candidate/name
        if not source.is_file():
            if name=="edge_benchmark.json":continue
            raise FileNotFoundError(source)
        temporary=target/(name+".publish.tmp")
        shutil.copy2(source,temporary)
        if digest_file(source)!=digest_file(temporary):raise ValueError("Staging checksum mismatch")
        temporary.replace(target/name)
    return verify_artifacts(target)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--candidate",required=True)
    p.add_argument("--target",default="outputs/lab04")
    args=p.parse_args();print(json.dumps(publish(args.candidate,args.target),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
