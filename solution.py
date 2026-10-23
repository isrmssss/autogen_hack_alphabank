"""Точка входа платформы: python solution.py path/to/task.md"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness import run  # noqa: E402


_STOP = {"flag": False}


def _handle(signum, frame) -> None:
    _STOP["flag"] = True


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python solution.py path/to/task.md", file=sys.stderr)
        sys.exit(2)
    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)
    run(sys.argv[1], lambda: _STOP["flag"])


if __name__ == "__main__":
    main()
