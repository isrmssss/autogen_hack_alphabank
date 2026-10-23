"""Промпты и разбор ответа. Система не выбирает алгоритм за модель."""

from __future__ import annotations

import json
import re


SYSTEM_PROMPT = """Ты ведёшь одну ML-задачу до файла submission.csv. Харнес не выбирает модель и не задаёт порядок алгоритмов. На каждом шаге ты видишь факты и сам называешь одно следующее действие.

Доступные действия:
- run — обучить и оценить спецификацию. Качество посчитает код на валидации.
- analyze — посчитать срез ошибок текущего чемпиона. Ты называешь срез, код считает.
- consult_knowledge — попросить одну карточку из базы. Проси, только когда для следующего приёма не хватает знания. Если карточка уже лежит в состоянии, не проси её снова.
- save — обучить текущую лучшую спецификацию на всём train и обновить submission.csv.
- stop — остановиться, если следующий опыт не стоит оставшегося времени.

Имена оценщиков, которые среда умеет запустить, перечислены в состоянии в поле toolbox. Это возможности, а не последовательность. Пустые params означают штатные значения библиотеки.

Правила ответа:
- Одно основное действие. Второе добавляй только в parallel и только если оно не зависит от результата первого.
- Не повторяй спецификацию, которая уже есть в журнале.
- Не используй test для выбора.
- Не записывай качество числом, которого нет в журнале.
- Если timeout_sec больше remaining_sec, это действие не запустится. Сразу выбери более короткое, save или stop.
- Ответ — один JSON без текста вокруг.

Форма:
{
  "why": "одно-два предложения",
  "action": "run | analyze | consult_knowledge | save | stop",
  "spec": {},
  "parallel": null
}

Для run в spec поля: estimator, params, features, code, split, fidelity, timeout_sec.
Для analyze в spec поле slice: by_category | by_quantile | per_class | missing_vs_error и необязательное column.
Для consult_knowledge в spec поле question: какие сигналы или какой вопрос.
Для save и stop spec может быть пустым.
parallel — либо null, либо такой же объект action/spec.
"""

REPAIR_PROMPT = """Код transform(train, test) упал. Верни только исправленную функцию.
Не добавляй импорты вне pandas и numpy. Не читай файлы и не обращайся к сети.
Не используй целевую колонку как признак. Ответ — тело функции, без пояснений.

Ошибка:
{stderr}

Код:
{code}
"""

ACTIONS = {"run", "analyze", "consult_knowledge", "save", "stop"}


def parse_json_object(text: str) -> dict:
    if not text or not str(text).strip():
        raise ValueError("пустой ответ")
    raw = str(text)
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("в ответе нет JSON")
    data = json.loads(raw[start: end + 1])
    if not isinstance(data, dict):
        raise ValueError("JSON должен быть объектом")
    return data


def _coerce(value):
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if re.fullmatch(r"-?\d+", value.strip()):
            return int(value.strip())
        if re.fullmatch(r"-?\d+\.\d+", value.strip()):
            return float(value.strip())
        return value
    if isinstance(value, dict):
        return {str(key): _coerce(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_coerce(item) for item in value]
    return value


def parse_decision(text: str) -> dict:
    data = _coerce(parse_json_object(text))
    action = data.get("action")
    if action not in ACTIONS:
        raise ValueError(f"неизвестное действие: {action}")
    spec = data.get("spec") or {}
    if not isinstance(spec, dict):
        raise ValueError("spec должен быть объектом")
    parallel = data.get("parallel")
    if parallel is not None:
        if not isinstance(parallel, dict) or parallel.get("action") not in ACTIONS:
            raise ValueError("parallel должен быть таким же объектом action/spec или null")
        parallel_spec = parallel.get("spec") or {}
        if not isinstance(parallel_spec, dict):
            raise ValueError("spec второго действия должен быть объектом")
        parallel = {"action": parallel["action"], "spec": parallel_spec, "why": parallel.get("why") or ""}
    return {
        "why": str(data.get("why") or ""),
        "action": action,
        "spec": spec,
        "parallel": parallel,
    }


def user_message(state: dict) -> str:
    note = ""
    if not state.get("knowledge"):
        note = "База знаний в этот раз не открывалась.\n\n"
    return note + json.dumps(state, ensure_ascii=False) + "\n\nВыбери следующее действие."


def repair_message(code: str, stderr: str) -> str:
    return REPAIR_PROMPT.format(stderr=stderr[-2000:], code=code[-4000:])


def feature_names(features) -> list[str]:
    names = []
    for step in features or []:
        if isinstance(step, str):
            names.append(step)
        elif isinstance(step, dict):
            names.append(str(step.get("name") or step.get("step") or "step"))
    return names


def mechanism_key(spec: dict) -> tuple:
    return (
        spec.get("estimator"),
        tuple(feature_names(spec.get("features"))),
        bool(spec.get("code")),
    )


def fingerprint(spec: dict) -> str:
    material = {
        "estimator": spec.get("estimator"),
        "params": spec.get("params") or {},
        "features": spec.get("features") or [],
        "code": spec.get("code") or "",
        "split": spec.get("split"),
        "fidelity": spec.get("fidelity"),
    }
    return json.dumps(material, ensure_ascii=False, sort_keys=True, default=str)
