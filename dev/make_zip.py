"""Архив для платформы: корневые .py и выданный requirements.txt."""

from __future__ import annotations

import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    destination = ROOT / "agent.zip"
    files = sorted(path for path in ROOT.glob("*.py") if path.is_file())
    requirements = ROOT / "requirements.txt"
    if not requirements.is_file():
        raise SystemExit("нет requirements.txt")
    if not any(path.name == "solution.py" for path in files):
        raise SystemExit("нет solution.py")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.name)
        archive.write(requirements, "requirements.txt")
    names = zipfile.ZipFile(destination).namelist()
    extra = [name for name in names if "/" in name or name.endswith((".yaml", ".yml", ".md"))]
    if extra:
        raise SystemExit(f"в архиве лишние пути: {extra}")
    size = destination.stat().st_size
    if size > 100 * 1024 * 1024:
        raise SystemExit(f"архив больше 100 МБ: {size}")
    print(f"{destination} {size} bytes {len(names)} files")


if __name__ == "__main__":
    main()
