import sys
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

runpy.run_path(
    str(SRC / "fairness_fr" / "gui" / "app.py"),
    run_name="__main__"
)
