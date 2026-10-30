"""Run mathematical reference checks against the isolated submission copy."""
import json
from pathlib import Path
import sys

source = Path(__file__).resolve().parents[1] / "lab02-review/independent_review.py"
code = source.read_text(encoding="utf-8").split("def main():")[0]
code = code.replace("ROOT = Path(__file__).resolve().parents[3]", "ROOT = Path(sys.argv[1]).resolve()")
scope = {"__name__": "submission_oracles", "__file__": str(__file__)}
exec(compile(code, str(source), "exec"), scope)
checks = []
for name in ("datalog_ground_oracle", "resolution_truth_table", "bayes_joint_oracle",
             "fuzzy_integration_oracle", "ds_set_intersection_oracle", "metrics_pairwise_oracle",
             "equity_iteration_oracle", "training_records_audit"):
    details = scope[name]()
    checks.append({"name": name, "passed": True, "details": details})
    print(json.dumps(checks[-1], ensure_ascii=False), flush=True)
Path(__file__).with_name("independent-checks.json").write_text(
    json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
