"""Индекс карточек и решение, открывать ли одну из них.

Карточка с пустым applies_when сама не всплывает: её можно достать только вопросом.
Сигналы семейства задачи (task_tabular и похожие) видны модели, но сами по себе
базу не открывают — иначе каждая таблица начиналась бы с одной и той же карточки.
"""

from __future__ import annotations

import re

from knowledge_data import CARDS


# Сигналы, которые описывают задачу, а не проблему. Они участвуют в совпадении
# карточки, но не являются поводом открыть базу.
TASK_LABELS = {
    "task_tabular",
    "task_multiclass",
    "task_binary",
    "single_output",
}

TRIGGER_SIGNALS = {
    "labels_imbalanced",
    "high_cardinality_categories",
    "non_iid",
    "rare_class_metric_noisy",
    "too_many_features",
    "split_is_random_only",
    "metric_is_accuracy",
    "gap_persists",
    "train_error_high",
    "val_error_higher",
    "errors_not_sliced",
    "features_not_scaled",
}

CARD_FIELDS = (
    "id",
    "domain",
    "summary",
    "mechanism",
    "applies_when",
    "avoid_when",
    "try",
    "expect_if_helps",
    "expect_if_hurts",
    "sources",
    "note",
)


def _flag(value) -> bool:
    return bool(value)


def detect_signals(profile: dict, journal: list[dict], champion: dict | None) -> set[str]:
    """Реестр детекторов. Новый сигнал — новая функция и имя в TRIGGER_SIGNALS, если он должен открывать базу."""
    signals = {"task_tabular"}
    if profile.get("task_type") == "classification":
        classes = profile.get("n_classes") or 0
        if classes > 2:
            signals.add("task_multiclass")
        elif classes == 2:
            signals.add("task_binary")
    else:
        signals.add("single_output")
    shares = list((profile.get("class_share") or {}).values())
    if shares and (min(shares) < 0.05 or (min(shares) > 0 and max(shares) / min(shares) > 10)):
        signals.add("labels_imbalanced")
    if profile.get("high_cardinality"):
        signals.add("high_cardinality_categories")
    if profile.get("time_candidates"):
        signals.add("non_iid")
    if profile.get("task_type") == "classification" and (profile.get("class_min_count") or 0) < 100:
        signals.add("rare_class_metric_noisy")
    if (profile.get("n_features") or 0) >= 400:
        signals.add("too_many_features")
    if profile.get("metric") == "accuracy" and "labels_imbalanced" in signals:
        signals.add("metric_is_accuracy")
    split_names = [item.get("split_name") for item in journal if item.get("action") == "run" and item.get("ok")]
    if not any(name in {"group", "time"} for name in split_names):
        signals.add("split_is_random_only")
    runs = [item for item in journal if item.get("action") == "run" and item.get("ok")]
    if runs and not any(item.get("action") == "analyze" and item.get("ok") for item in journal):
        signals.add("errors_not_sliced")
    if champion and champion.get("estimator") == "linear" and "scale" not in (champion.get("features") or []):
        signals.add("features_not_scaled")
    last = runs[-1] if runs else None
    if last and last.get("train_metric") is not None and last.get("val_metric") is not None:
        train_metric = float(last["train_metric"])
        val_metric = float(last["val_metric"])
        scale = max(abs(train_metric), abs(val_metric), 1e-9)
        train_better = train_metric > val_metric if profile.get("higher_is_better") else train_metric < val_metric
        if train_better and abs(train_metric - val_metric) / scale >= 0.05:
            signals.add("gap_persists")
        if train_better and abs(train_metric - val_metric) / scale >= 0.02:
            signals.add("val_error_higher")
        constant = profile.get("constant_metric")
        if constant is not None:
            margin = abs(train_metric - float(constant)) / max(abs(float(constant)), 1e-9)
            beats = train_metric > float(constant) if profile.get("higher_is_better") else train_metric < float(constant)
            if not beats or margin < 0.01:
                signals.add("train_error_high")
    return {name for name in signals if _flag(name)}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Zа-яА-Я0-9_]{3,}", (text or "").lower()))


def _view(card: dict) -> dict:
    shown = {key: card.get(key) for key in CARD_FIELDS}
    if not shown.get("note"):
        shown.pop("note", None)
    return shown


class KnowledgeBase:
    def __init__(self, cards: list[dict] | None = None) -> None:
        self.cards = list(cards if cards is not None else CARDS)

    def decide(
        self,
        signals: set[str],
        shown: set[tuple],
        last_triggers: set[str],
        question: str | None,
        repeat_hurt: bool,
    ) -> dict | None:
        if question is not None:
            return self._by_question(question)
        if repeat_hurt:
            opened = self._by_avoid(signals, shown)
            if opened:
                return opened
        triggers = signals & TRIGGER_SIGNALS
        if not (triggers - last_triggers):
            return None
        opened = self._by_signals(signals, shown)
        if opened:
            opened["advance_triggers"] = True
            return opened
        return {"advance_triggers": True, "view": None}

    def _by_signals(self, signals: set[str], shown: set[tuple]) -> dict | None:
        key = tuple(sorted(signals))
        matches = []
        for card in self.cards:
            need = set(card.get("applies_when") or [])
            if not need or not need <= signals:
                continue
            if set(card.get("avoid_when") or []) & signals:
                continue
            if (card["id"], key) in shown:
                continue
            matches.append(card)
        if not matches:
            return None
        matches.sort(key=lambda card: (-len(card.get("applies_when") or []), card["id"]))
        best = matches[0]
        shown.add((best["id"], key))
        return {
            "advance_triggers": False,
            "view": {
                "reason": "новый сигнал: " + ", ".join(sorted(signals & TRIGGER_SIGNALS)),
                "card": _view(best),
                "also": [_title(card) for card in matches[1:3]],
            },
        }

    def _by_avoid(self, signals: set[str], shown: set[tuple]) -> dict | None:
        matches = []
        for card in self.cards:
            avoided = set(card.get("avoid_when") or [])
            if not avoided & signals:
                continue
            mark = ("hurt", card["id"])
            if mark in shown:
                continue
            matches.append((len(avoided & signals), card))
        if not matches:
            return None
        matches.sort(key=lambda item: (-item[0], item[1]["id"]))
        best = matches[0][1]
        shown.add(("hurt", best["id"]))
        return {
            "advance_triggers": False,
            "view": {
                "reason": "повтор механизма, который уже ухудшал результат",
                "card": _view(best),
                "also": [_title(item[1]) for item in matches[1:3]],
            },
        }

    def _by_question(self, question: str) -> dict:
        asked = _tokens(question)
        ranked = []
        for card in self.cards:
            blob = _tokens(
                " ".join(
                    [
                        card.get("summary") or "",
                        card.get("mechanism") or "",
                        card.get("id") or "",
                        " ".join(card.get("applies_when") or []),
                    ]
                )
            )
            score = len(asked & blob)
            for signal in card.get("applies_when") or []:
                if signal.lower() in question.lower():
                    score += 3
            if score:
                ranked.append((score, card))
        if not ranked:
            return {
                "advance_triggers": False,
                "view": {
                    "reason": "запрос не совпал ни с одной карточкой",
                    "card": None,
                    "also": [],
                },
            }
        ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
        best = ranked[0][1]
        return {
            "advance_triggers": False,
            "view": {
                "reason": "модель запросила базу",
                "card": _view(best),
                "also": [_title(item[1]) for item in ranked[1:3]],
            },
        }


def _title(card: dict) -> dict:
    return {"id": card["id"], "summary": card.get("summary") or ""}
