"""Measure end-to-end CPU inference with warmed weights, reproducibly."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import platform
import hashlib
import statistics
import time
from pathlib import Path

import numpy as np

from .experiment import write_json
from .inference import clear_cache, run_episode


def measure(directory, repetitions=12):
    directory = Path(directory)
    rows = []
    configurations=[("baseline","none"),("maml","none"),("calibrated","full")]
    for mode in ("graph","multiview","wide","wideview","continued"):
        if (directory/f"{mode}_results.json").is_file():configurations.append(("calibrated",mode))
    for shots in (1, 5):
        for method,calibration in configurations:
            clear_cache()
            started = time.perf_counter()
            first = run_episode(directory, shots=shots, seed=20261124, method=method, calibration=calibration)
            cold_ms = (time.perf_counter() - started) * 1000
            compute, wall = [], []
            for i in range(repetitions):
                started = time.perf_counter()
                result = run_episode(directory, shots=shots, seed=20261125+i, method=method, calibration=calibration)
                wall.append((time.perf_counter() - started) * 1000)
                compute.append(result["inference_ms"])
            rows.append({"method":method if calibration in ("none","full") else calibration,
                "calibration":calibration,"shots":shots,"repetitions":repetitions,
                "cold_wall_ms":cold_ms,"warm_wall_median_ms":statistics.median(wall),
                "warm_wall_p95_ms":float(np.percentile(wall,95)),
                "encoding_and_classification_median_ms":statistics.median(compute),
                "encoding_and_classification_p95_ms":float(np.percentile(compute,95)),
                "weights_sha256":first["checkpoint_sha256"],"artifact_revision":first["artifact_revision"],"wall_samples_ms":wall,"compute_samples_ms":compute})
    return {"generated_utc":datetime.now(timezone.utc).isoformat(),"device":"CPU",
        "platform":platform.platform(),"processor":platform.processor(),
        "compute_scope":"All modes: actual frozen image encoding, fixed feature transform, prediction/support adaptation; excludes model loading and checksum verification.",
        "source_sha256":hashlib.sha256(Path(__file__).with_name("inference.py").read_bytes()).hexdigest(),
        "scope":"Local CPU proxy for edge deployment, NOT a measurement on mobile/ARM hardware; no HTTP or thumbnail encoding.",
        "cold_scope":"First load, checksum verification, sampling, decoding normalized tensors, encoding and inference.",
        "warm_scope":"Cached model and pool, but actual image encoding, no cached feature predictions.",
        "rows":rows}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--artifacts",default="outputs/lab04")
    p.add_argument("--repetitions",type=int,default=12);p.add_argument("--output")
    args=p.parse_args()
    if args.repetitions<3:raise ValueError("At least 3 timing repeats are needed")
    output=Path(args.output) if args.output else Path(args.artifacts)/"edge_benchmark.json"
    write_json(output,measure(args.artifacts,args.repetitions));print(output)


if __name__=="__main__":main()
