"""Explicit unit fixtures and optional artifact prerequisites, never saved results."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from functools import lru_cache
from unittest import skipUnless

from experiments.lab03.dataset import ROOT
from experiments.lab03.artifacts import provenance, source_hashes


@lru_cache(maxsize=1)
def _learned_fixtures():
    """Small real in-memory fits; no fabricated training IDs and no files written."""
    from experiments.lab03.dataset import load_dataset
    from experiments.lab03.learning import fit_classifier
    from experiments.lab03.genetic import optimize
    records = load_dataset()
    return {
        "models.json": {str(seed): fit_classifier(records, seed) for seed in (7, 19, 43)},
        "ga_history.json": {str(seed): optimize(records, seed, population=3, generations=0)
                            for seed in (7, 19, 43)},
    }


def requires_artifacts(*names):
    ready = all((ROOT / "outputs/lab03" / name).is_file() for name in names)
    return skipUnless(ready, "需先生成或恢复相应实验结果；纯源码不包含历史实测产物：" + ", ".join(names))


def artifact_fixture(name, default=None):
    """Declared in-memory test fits, never persisted or labelled benchmark results."""
    if name in {"models.json", "ga_history.json"}:
        # Serialization removes list aliasing exactly as saved JSON artifacts do.
        value = json.loads(json.dumps(_learned_fixtures()[name]))
    elif name == "manifest.json":
        value = {"dataset_sha256": provenance()["dataset_sha256"], "source_sha256": source_hashes()}
    else:
        value = default
    return copy.deepcopy(value)
