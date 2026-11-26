"""Concise read-only current release and candidate status; never starts model work."""
from __future__ import annotations
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def rows(value):
    return [{key:row[key] for key in ("method","shots","accuracy","ci95","paired_delta","paired_ci95",
                                     "paired_vs_graph","paired_vs_graph_ci95","paired_vs_multiview","paired_vs_multiview_ci95","paired_vs_wide","paired_vs_wide_ci95","paired_vs_wideview","paired_vs_wideview_ci95") if key in row}
            for row in value.get("rows",[])]


def main():
    root=Path(__file__).resolve().parents[2];local=root/".local";artifacts=root/"outputs/lab04"
    core=read(artifacts/"results.json")
    result={"objective":"Improve fixed-protocol accuracy; no numerical target", "published":{},"candidates":{}}
    if core:result["published"]["core"]={"version":core.get("study",{}).get("version"),"rows":[r for r in rows(core) if r["method"] in ("baseline","full","maml")]}
    for name in ("graph","multiview","wide","wideview","continued"):
        value=read(artifacts/f"{name}_results.json")
        if value:result["published"][name]={"status":value.get("status"),"rows":rows(value)}
    training=read(local/"lab04-v3-wide/training.json")
    if training:
        result["candidates"]["wide_training"]={key:training[key] for key in ("status","best_epoch","best_mean") if key in training}
        result["candidates"]["wide_training"].update({"completed_epochs":len(training.get("rows",[])),"latest":training.get("rows",[])[-1] if training.get("rows") else None})
    selection=read(local/"lab04-v3-wide/validation_selection.json")
    result["candidates"]["wide_validation"]={"status":selection.get("status"),"selected":{shot:{"validation":entry["selected"].get("confirmation"),"references":entry.get("references")} for shot,entry in selection.get("shots",{}).items()}} if selection else {"status":"not_yet_recorded"}
    wide=read(local/"lab04-wide-test/wide_results.json")
    result["candidates"]["wide_test"]={"status":wide.get("status"),"rows":rows(wide)} if wide else {"status":"not_yet_recorded"}
    for name in ("cached-refinement", "wide-views"):
        value=read(local/f"lab04-{name}-20261125/selection.json")
        if value:result["candidates"][name]={"status":value.get("status"),"test_accessed":value.get("test_accessed"),"shots":{shot:{"candidate":r["selected"]["candidate"],"paired":r["selected"].get("confirmation_paired"),"promising":r.get("promising")} for shot,r in value.get("shots",{}).items()}}
    continued=read(local/"lab04-continuation-20261125/training.json")
    if continued:
        result["candidates"]["continued_training"]={k:continued[k] for k in ("status","best_epoch","best_mean","test_accessed") if k in continued}
        result["candidates"]["continued_training"].update({"completed_epochs":len(continued.get("rows",[])),"latest":continued.get("rows",[])[-1] if continued.get("rows") else None,
            "planned_epochs":16,"public_result_unchanged":True})
    continued_validation=read(local/"lab04-continuation-validation-20261125/validation.json")
    result["candidates"]["continued_validation"]={k:continued_validation[k] for k in ("status","promising","shots","selected_original","test_accessed") if k in continued_validation} if continued_validation else {"status":"not_yet_recorded"}
    result["note"]="Running states are checkpoint metadata, not proof a process is still live. Inspect PID, creation time and command before resuming. External-reference validation is not a published self-trained test result."
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
