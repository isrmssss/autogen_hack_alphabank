"""Собрать knowledge_data.py из yaml-черновика main-base.

Новый домен — новый файл в interventions/ с теми же полями карточки.
После правки yaml запустите этот скрипт заново. Платформа yaml не копирует,
поэтому в архив входит только получившийся модуль.
"""

from __future__ import annotations

import argparse
import pprint
from pathlib import Path

import yaml


def first_sentence(text: str, limit: int = 220) -> str:
    clean = " ".join((text or "").split())
    if not clean:
        return ""
    stop = len(clean)
    for mark in (". ", "。"):
        at = clean.find(mark)
        if at != -1:
            stop = min(stop, at + 1)
    return clean[: min(stop, limit)].strip()


def as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def card_from_raw(raw: dict, domain: str) -> dict:
    change = raw.get("change") or {}
    mechanism = str(raw.get("mechanism") or "").strip()
    return {
        "id": raw["id"],
        "domain": domain,
        "summary": first_sentence(mechanism),
        "mechanism": mechanism,
        "applies_when": [str(item) for item in as_list(raw.get("applies_when"))],
        "avoid_when": [str(item) for item in as_list(raw.get("avoid_when"))],
        "try": as_list(change.get("try")),
        "expect_if_helps": as_list(raw.get("expect_if_helps")),
        "expect_if_hurts": as_list(raw.get("expect_if_hurts")),
        "sources": [str(item) for item in as_list(raw.get("sources"))],
        "note": str(raw.get("note") or "").strip(),
    }


def load_cards(base: Path) -> list[dict]:
    cards = []
    folder = base / "interventions"
    for path in sorted(folder.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for raw in data.get("cards") or []:
            cards.append(card_from_raw(raw, path.stem))
    cards.sort(key=lambda item: (item["domain"], item["id"]))
    return cards


def main() -> None:
    here = Path(__file__).resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=here.parents[3] / "main-base")
    parser.add_argument("--out", type=Path, default=here.parents[1] / "knowledge_data.py")
    args = parser.parse_args()
    cards = load_cards(args.base)
    body = pprint.pformat(cards, width=100, sort_dicts=False)
    args.out.write_text(
        "# Собрано из main-base скриптом dev/compile_knowledge.py. Не править руками.\n"
        "CARDS = "
        + body
        + "\n",
        encoding="utf-8",
    )
    print(f"{len(cards)} cards -> {args.out}")


if __name__ == "__main__":
    main()
