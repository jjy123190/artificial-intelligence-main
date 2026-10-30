"""Compatibility entry: submit the complete project with lab01 entry instructions."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.package_project import main
if __name__=='__main__':raise SystemExit(main(['--experiment','lab01']))
