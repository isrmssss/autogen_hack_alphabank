"""Локальный запуск. Ключ читается из .env и не печатается.

По умолчанию бюджет как на платформе. Для короткой проверки передайте --budget.
Предсказания пишутся в копию задачи внутри runs/, исходная папка задачи не меняется.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
FILE_NAMES = ("train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv")


def load_env(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    if not os.environ.get("OPENAI_API_KEY") and os.environ.get("API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["API_KEY"]
    os.environ.setdefault("OPENAI_BASE_URL", "https://api.alfahack-gateway26.ru/v1")
    os.environ.setdefault("OPENAI_MODEL", "qwen/qwen3.8-27b")


def stage(task_md: Path, run_dir: Path) -> Path:
    source = task_md.resolve().parent
    staged = run_dir / "task"
    staged.mkdir(parents=True, exist_ok=True)
    shutil.copy2(task_md, staged / "task.md")
    for name in FILE_NAMES:
        origin = source / name
        if origin.is_file():
            os.symlink(origin, staged / name)
    return staged / "task.md"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--budget", type=float, default=10800)
    args = parser.parse_args()
    load_env(WORKSPACE / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("нет OPENAI_API_KEY")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^\w.-]+", "_", args.task.resolve().parent.name, flags=re.UNICODE)[:40]
    run_dir = ROOT / "runs" / f"{stamp}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    staged = stage(args.task, run_dir)
    os.environ["AGENT_RUN_DIR"] = str(run_dir)
    os.environ["AGENT_TIME_BUDGET_SEC"] = str(args.budget)
    os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
    print(f"run {run_dir}")
    os.execv(sys.executable, [sys.executable, str(ROOT / "solution.py"), str(staged)])


if __name__ == "__main__":
    main()
