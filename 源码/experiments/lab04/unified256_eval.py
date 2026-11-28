"""Select only on released validation tasks; replay all locked methods on identical test images."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .accuracy90_data import json_from, load_archive
from .data import digest_file
from .experiment import summarize
from .study_cached_refinement import paired_stats
from .train_gpu_metric import reference_rows, save_json
from .train_medium_gpu import medium_model
from .unified256_methods import (METHODS, RULES, descriptors, fit_prior, index_episodes,
                                 maml_predictions, metric_predictions, protocol_config)
from .unified256_train import (ARTIFACTS, DATA, DEFAULT_WORK, EPOCHS, META_CANDIDATES, META_EPOCHS,
                               WARM, checkpoint, init_maml, persist_plan, setup)

RELEASE = Path("outputs/lab04_unified256")
LOCK_NAME = "validation_lock.json"
EXTENSIONS = ("full_flip", "full_local", "full_flip_local")


def setup_plan(artifacts):
    base = Path(__file__).parent
    saved = json_from(artifacts / "continued_results.json")
    names = ("continued_validation_tasks.jsonl", "continued_episodes.json")
    return {
        **protocol_config(), "main_methods": ["baseline", "maml", "full"],
        "val_tasks_per_shot_calibration": 600, "val_tasks_per_shot_meta_checkpoint": 60,
        "rule_selection": "max combined validation correct, same choice both shots; tie first",
        "testing": "all 600 released episodes/shot; 5-way, 15 queries/class; no test selection",
        "metric_baseline": "CE encoder, train-centered unit features, support-only cosine prototype",
        "maml_reference": "same chosen MAML weights with no inner update",
        "extensions": "2x2 horizontal flip and local patch matching; NOT inputs to the formal full method",
        "episode_sha256": {n: saved["files"][n]["sha256"] for n in names},
        "data_manifest_sha256": digest_file(artifacts / "dataset_manifest.json"),
        "source_sha256": {f: digest_file(base / f) for f in ("unified256_train.py", "unified256_methods.py",
                          "unified256_eval.py", "evaluate_gpu_tracks.py", "experiment.py",
                          "study_cached_refinement.py", "train_gpu_metric.py", "models.py", "data.py",
                          "train_medium_gpu.py", "train_gpu_backbone.py", "accuracy90_data.py")},
    }


def verify_files(base, mapping):
    for name, value in mapping.items():
        if digest_file(base / name) != value:
            raise ValueError(f"Digest mismatch: {base / name}")


def get_metric_model(directory, name):
    model = medium_model()
    model.load_state_dict(checkpoint(directory / (name + ".pt")))
    return model.eval().to("cuda", memory_format=torch.channels_last)


def get_meta_model(directory, file):
    # Restore the entire 5-way model, not just its CE-initialized encoder.
    model = init_maml(checkpoint(directory / "ce.pt"))
    model.load_state_dict(checkpoint(file))
    return model.eval().to("cuda", memory_format=torch.channels_last)


def load_tasks(args, split):
    data = load_archive(args.data_root, args.artifacts, split)
    rows = reference_rows(args.artifacts, split, json_from(args.artifacts / "continued_results.json"))
    tasks = {shot: index_episodes(data, rows, shot) for shot in (1, 5)}
    # Scoring-only data, kept out of both the episode indexer and all predictors.
    answers = {shot: np.asarray([r["query_labels"] for r in sorted(
        (r for r in rows if r["shots"] == shot), key=lambda r: r["episode"])], dtype=np.int64)
               for shot in (1, 5)}
    return data, tasks, answers


def fit_priors(args):
    data = load_archive(args.data_root, args.artifacts, "train")
    arrays = {}
    for arm in ("ce", "da"):
        raw = descriptors(get_metric_model(args.bundle, arm), data.images, "cuda", flip=(arm == "da"))
        for view, x in raw.items():
            for key, val in fit_prior(x, data.labels).items():
                arrays[f"{arm}_{view}_{key}"] = val.numpy()
    np.savez_compressed(args.bundle / "priors.npz", **arrays)


def priors(directory):
    with np.load(directory / "priors.npz", allow_pickle=False) as arrays:
        return {(arm, view): {key: torch.from_numpy(arrays[f"{arm}_{view}_{key}"])
                              for key in ("center", "means", "variances")}
                for arm, view in (("ce", "single"), ("da", "single"), ("da", "flip"))}


def matched_logs(work):
    a, b = [json_from(work / (arm + "_history.json")) for arm in ("ce", "da")]
    if (len(a) != EPOCHS or len(b) != EPOCHS or a[:WARM] != b[:WARM] or
            any(x[k] != y[k] for x, y in zip(a, b) for k in ("draw_sha256", "pixels", "lr", "steps", "epoch"))
            or any(x["alignment_weight"] != 0 for x in a)
            or any(x["alignment_weight"] <= 0 for x in b[WARM:])):
        raise ValueError("The matched training intervention is not complete")
    m = json_from(work / "maml_history.json")
    if len(m["epochs"]) != META_EPOCHS or m["ce_sha256"] != digest_file(work / "ce.pt"):
        raise ValueError("Meta-training does not match the released CE initialization")


def dump_rows(path, rows):
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as out:
        for row in rows:
            out.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    temp.replace(path)


def make_row(shot, idx, tasks, pred, gold, evidence=None):
    queries = len(pred)
    correct = int(np.count_nonzero(pred == gold))
    row = {"shots": shot, "episode": idx, "task_sha256": tasks.hashes[idx],
           "predictions": pred.tolist(), "correct": correct, "accuracy": correct / queries}
    if evidence is not None:
        row.update(evidence)
    return row


def select(args):
    here = args.bundle
    here.mkdir(parents=True, exist_ok=True)
    cfg = setup_plan(args.artifacts)
    persist_plan(here / "evaluation_plan.json", cfg)
    if (here / LOCK_NAME).exists():
        lock = json_from(here / LOCK_NAME)
        verify_files(here, lock["files"])
        if digest_file(args.work / "training_plan.json") != lock["files"]["training_plan.json"]:
            raise ValueError("Different training plan. Use a fresh --bundle for another experiment.")
        for arm in ("ce", "da"):
            if digest_file(args.work / (arm + ".pt")) != lock["files"][arm + ".pt"]:
                raise ValueError("Different run. Use a fresh --bundle.")
        if digest_file(args.work / f"maml_e{lock['meta_epoch']}.pt") != lock["files"]["maml.pt"]:
            raise ValueError("Different meta-training run. Use a fresh --bundle.")
        print("Validation already locked:", here / LOCK_NAME, flush=True)
        return
    matched_logs(args.work)
    keep = ["training_plan.json", "ce.pt", "da.pt", "ce_history.json", "da_history.json", "maml_history.json"]
    for n in keep:
        shutil.copyfile(args.work / n, here / n)
    print("Fitting class priors using only the 64 base classes", flush=True)
    fit_priors(args)
    val, tasks, gold = load_tasks(args, "val")
    raw = {"da": descriptors(get_metric_model(here, "da"), val.images, "cuda")}
    prior = priors(here)
    trials, metric_rows = [], []
    for i, rule in enumerate(RULES):
        counts = {}
        for shot in (1, 5):
            p = metric_predictions(raw, prior, tasks[shot], rule, {"full": METHODS["full"]}, "cuda")["full"]
            counts[str(shot)] = int(np.count_nonzero(p == gold[shot]))
            for j in range(len(p)):
                metric_rows.append({"candidate": i, **make_row(shot, j, tasks[shot], p[j], gold[shot][j])})
        trials.append({"rule_index": i, "correct_by_shot": counts, "total": sum(counts.values())})
    best = max(trials, key=lambda t: t["total"])["rule_index"]
    dump_rows(here / "val_calibration.jsonl", metric_rows)
    del raw, metric_rows
    meta_rows, meta_trials = [], []
    for epoch in META_CANDIDATES:
        path = args.work / f"maml_e{epoch}.pt"
        model = get_meta_model(here, path)
        counts = {}
        for shot in (1, 5):
            pred, _, evidence = maml_predictions(model, val.images, tasks[shot], "cuda", limit=60)
            counts[str(shot)] = int(np.count_nonzero(pred == gold[shot][:len(pred)]))
            for j in range(len(pred)):
                meta_rows.append({"epoch": epoch, **make_row(shot, j, tasks[shot], pred[j], gold[shot][j], evidence[j])})
        meta_trials.append({"epoch": epoch, "correct_by_shot": counts, "total": sum(counts.values()),
                            "weights_sha256": digest_file(path)})
        print("MAML validation", meta_trials[-1], flush=True)
    selected = max(meta_trials, key=lambda t: t["total"])["epoch"]
    dump_rows(here / "val_maml.jsonl", meta_rows)
    shutil.copyfile(args.work / f"maml_e{selected}.pt", here / "maml.pt")
    keep += ["evaluation_plan.json", "priors.npz", "maml.pt", "val_calibration.jsonl", "val_maml.jsonl"]
    save_json(here / LOCK_NAME, {"rule_index": best, "rule": RULES[best], "meta_epoch": selected,
              "calibration_validation": trials, "meta_validation": meta_trials,
              "files": {n: digest_file(here / n) for n in keep}})
    print("Locked before test:", here / LOCK_NAME, "rule", best, "meta epoch", selected, flush=True)


def test(args):
    here, out = args.bundle, args.output or args.bundle
    lock = json_from(here / LOCK_NAME)
    verify_files(here, lock["files"])
    if json_from(here / "evaluation_plan.json") != setup_plan(args.artifacts):
        raise ValueError("The protocol/implementation differs from the locked validation run")
    out.mkdir(parents=True, exist_ok=True)
    test_split, tasks, gold = load_tasks(args, "test")
    print("Test: extracting full images with the two locked 256-D encoders", flush=True)
    raw = {arm: descriptors(get_metric_model(here, arm), test_split.images, "cuda",
                            flip=(arm == "da"), local=(arm == "da")) for arm in ("ce", "da")}
    prior = priors(here)
    model = get_meta_model(here, here / "maml.pt")
    stats, contrasts = {}, {}
    comparison = [("full", "baseline"), ("full", "maml"), ("maml", "maml_no_adapt"),
                  *(("full", k) for k in ("no_alignment", "no_mean", "no_variance", "no_update", "no_balance")),
                  *((k, "full") for k in EXTENSIONS),
                  ("full_flip_local", "full_flip"), ("full_flip_local", "full_local")]
    files = {}
    for shot in (1, 5):
        task = tasks[shot]
        preds = metric_predictions(raw, prior, task, lock["rule"], METHODS, "cuda")
        print(f"Test {shot}-shot: adapting full MAML network independently on 600 support sets", flush=True)
        preds["maml"], preds["maml_no_adapt"], evidence = maml_predictions(model, test_split.images, task, "cuda")
        rates = {}
        for name, pred in preds.items():
            rows = [make_row(shot, j, task, pred[j], gold[shot][j], evidence[j] if name == "maml" else None)
                    for j in range(len(pred))]
            path = out / f"test_{name}_{shot}shot.jsonl"
            dump_rows(path, rows)
            files[path.name] = digest_file(path)
            right = np.array([r["correct"] for r in rows], dtype=np.int64)
            rates[name] = right / 75
            stats.setdefault(name, {})[str(shot)] = {
                **summarize(rates[name]), "correct": int(right.sum()), "queries": int(pred.size)}
        for a, b in comparison:
            contrasts.setdefault(a + "_minus_" + b, {})[str(shot)] = paired_stats(rates[a], rates[b])
    for name in stats:
        print(name, *(round(100 * stats[name][s]["accuracy"], 3) for s in ("1", "5")), flush=True)
    save_json(out / "results.json", {"split": "test", "device": torch.cuda.get_device_name(),
              "torch": str(torch.__version__), "numpy": np.__version__,
              "selection_lock_sha256": digest_file(here / LOCK_NAME), "methods": stats,
              "paired": contrasts, "files": files})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("select", "test"))
    p.add_argument("--work", type=Path, default=DEFAULT_WORK)
    p.add_argument("--bundle", type=Path, default=RELEASE)
    p.add_argument("--output", type=Path, help="Test output (use an empty directory for replay)")
    p.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    p.add_argument("--data-root", type=Path, default=DATA)
    args = p.parse_args()
    setup()
    (select if args.action == "select" else test)(args)


if __name__ == "__main__":
    main()

