"""Run the project suite once and retain scoped, fingerprinted test evidence."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.lab02.eval.artifacts import execution_manifest, write_json


class RecordedResult(unittest.TextTestResult):
    def startTest(self, test):
        super().startTest(test)
        self.executed_ids.append(test.id())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executed_ids = []


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    log = Path(__file__).with_name("tests.log")
    with log.open("w", encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordedResult).run(suite)
    metadata = execution_manifest(ROOT)
    report = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "skipped": len(result.skipped), "successful": result.wasSuccessful(), **metadata}
    write_json(ROOT / "outputs/qa/test-summary.json", report)
    write_json(Path(__file__).with_name("test-summary.json"), {**report, "executed_test_ids": result.executed_ids})
    prefix = "tests.lab02."
    lab_ids = [name for name in result.executed_ids if name.startswith(prefix)]
    lab_failures = sum(test.id().startswith(prefix) for test, _ in result.failures)
    lab_errors = sum(test.id().startswith(prefix) for test, _ in result.errors)
    write_json(ROOT / "outputs/lab02/qa/test_summary.json", {
        **metadata, "tests_run": len(lab_ids), "failures": lab_failures, "errors": lab_errors,
        "skipped": sum(test.id().startswith(prefix) for test, _ in result.skipped),
        "successful": lab_failures == 0 and lab_errors == 0,
        "scope": "lab02 cases executed as part of the complete project suite",
    })
    if result.wasSuccessful():
        (ROOT / "outputs/lab02/qa/test_results.txt").write_text(
            f"Full project run: {result.testsRun} tests, including {len(lab_ids)} lab02 tests.\n\n"
            + log.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("tests_run", "failures", "errors", "skipped", "successful")}))
    print(json.dumps({"lab02_tests": len(lab_ids), "log": str(log)}))
    if not result.wasSuccessful():
        for test, details in result.failures + result.errors:
            print(test.id(), details)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
