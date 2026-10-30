import json
from pathlib import Path
import sys
import unittest

def main():
    root = Path(sys.argv[1]).resolve()
    qa = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    suite = unittest.defaultTestLoader.discover(str(root / "tests"), top_level_dir=str(root))
    with (qa / "unit-tests.log").open("w", encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    modules = {}
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] in {"core", "backend", "experiments", "tests", "scripts"}:
            location = getattr(module, "__file__", None)
            if location:
                path = Path(location).resolve()
                if not path.is_relative_to(root):
                    raise AssertionError(f"External project import: {name}: {path}")
                modules[name] = str(path.relative_to(root))
    report = {"root": str(root), "python": sys.executable, "tests": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors),
              "skipped": len(result.skipped), "passed": result.wasSuccessful(),
              "all_project_imports_local": True, "project_modules": modules}
    (qa / "unit-tests.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "project_modules"}, ensure_ascii=False))
    for test, detail in result.failures + result.errors:
        print(test.id(), detail)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
