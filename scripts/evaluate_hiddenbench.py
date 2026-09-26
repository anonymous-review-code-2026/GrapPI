from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphpi.benchmarks.cli import hiddenbench_main, safe_main

if __name__ == "__main__":
    raise SystemExit(safe_main(hiddenbench_main))
