"""Проверка разбора задачи, карточки, обучения и дочернего процесса. В исходные папки задач не пишет."""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, str(ROOT))

from knowledge import TRIGGER_SIGNALS, KnowledgeBase, detect_signals  # noqa: E402
from task_io import constant_frame, load_task, verify_submission, write_submission  # noqa: E402
from toolbox import apply_code, child_main, run_experiment  # noqa: E402


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)
    print(f"ok {message}")


def tiny_task(folder: Path, metric_line: str, target_values, kind: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    train = pd.DataFrame({"id": [1, 2, 3, 4], "x": [0.1, 0.2, 0.3, 0.4], "y": target_values})
    test = pd.DataFrame({"id": [5, 6], "x": [0.5, 0.6]})
    train.to_csv(folder / "train.csv", index=False)
    test.to_csv(folder / "test.csv", index=False)
    pd.DataFrame({
        "feature": ["id", "x", "y"],
        "role": ["id", "feature", "target"],
        "description_ru": ["идентификатор", "признак", "цель"],
    }).to_csv(folder / "data_dictionary.csv", index=False)
    pd.DataFrame({"id": [5, 6], "y": [0, 0]}).to_csv(folder / "sample_submission.csv", index=False)
    (folder / "task.md").write_text(
        f"Файлы лежат рядом.\nОсновная метрика: {metric_line}\n",
        encoding="utf-8",
    )


def main() -> None:
    regression = load_task(WORKSPACE / "Регрессия" / "task.md")
    classification = load_task(WORKSPACE / "Многоклассовая классификация" / "task.md")
    check(regression.target == "site_eui", "регрессия: цель site_eui")
    check(regression.id_column == "id", "регрессия: id")
    check(regression.metric == "rmse" and regression.higher_is_better is False, "регрессия: rmse")
    check(regression.train_path.parent == regression.path.parent, "регрессия: csv рядом с task.md")
    check(classification.target == "attack_cat", "классификация: цель attack_cat")
    check(classification.metric == "f1_macro" and classification.higher_is_better is True, "классификация: macro f1")
    check(classification.train_path.parent == classification.path.parent, "классификация: csv рядом с task.md")
    before = regression.path.parent / "submission.csv"
    stamp = before.stat().st_mtime if before.is_file() else None
    load_task(regression.path)
    after = before.stat().st_mtime if before.is_file() else None
    check(stamp == after, "разбор регрессии не переписывает submission.csv")

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        nested = root / "case" / "nested"
        tiny_task(nested, "RMSE", [1.0, 2.0, 3.0, 4.0], "reg")
        task_md = root / "case" / "task.md"
        task_md.write_text(
            "Данные: nested/train.csv nested/test.csv nested/data_dictionary.csv nested/sample_submission.csv\n"
            "Основная метрика: RMSE\n",
            encoding="utf-8",
        )
        loaded = load_task(task_md)
        check(loaded.train_path == nested / "train.csv", "относительный путь из текста")
        check(loaded.metric == "rmse" and loaded.task_type == "regression", "вложенная задача — регрессия")

        beside = root / "beside"
        tiny_task(beside, "Macro F1", ["a", "b", "a", "c"], "cls")
        text = (beside / "task.md").read_text(encoding="utf-8")
        (beside / "task.md").write_text(
            "Смотри missing/train.csv missing/test.csv missing/data_dictionary.csv missing/sample_submission.csv\n"
            + text,
            encoding="utf-8",
        )
        loaded = load_task(beside / "task.md")
        check(loaded.train_path == beside / "train.csv", "файлы рядом, если путь из текста пуст")
        check(loaded.metric == "f1_macro" and loaded.task_type == "classification", "задача рядом — классификация")

        train = pd.read_csv(nested / "train.csv")
        test = pd.read_csv(nested / "test.csv")
        task = load_task(nested / "task.md")
        write_submission(task, constant_frame(task, train, test))
        check(verify_submission(task) == [], "константный файл проходит проверку")
        bad = pd.read_csv(task.directory / "submission.csv")
        bad["y"] = [1, None]
        bad.to_csv(task.directory / "submission.csv", index=False)
        check(bool(verify_submission(task)), "битый файл не проходит проверку")

    fresh = KnowledgeBase()
    check(fresh.decide({"task_tabular", "single_output"}, set(), set(), None, False) is None, "метки задачи сами базу не открывают")
    signals = {"task_tabular", "single_output", "non_iid", "split_is_random_only"}
    shown: set = set()
    opened = fresh.decide(signals, shown, set(), None, False)
    check(opened is not None and opened.get("view") and opened["view"].get("card"), "регрессионные сигналы открывают карточку")
    card = opened["view"]["card"]
    check(set(card["applies_when"]) <= signals and bool(card["applies_when"]), f"карточка {card['id']} совпала по сигналам")
    again = fresh.decide(signals, shown, signals & TRIGGER_SIGNALS, None, False)
    check(again is None, "тот же набор сигналов карточку второй раз не открывает")
    asked = fresh.decide(set(), set(), set(), "rare class metric", False)
    check(asked.get("advance_triggers") is False and "reason" in asked.get("view", {}), "вопрос возвращает причину и не двигает сигналы")

    cls_signals = {"task_tabular", "task_multiclass", "labels_imbalanced", "rare_class_metric_noisy", "split_is_random_only"}
    cls_open = KnowledgeBase().decide(cls_signals, set(), set(), None, False)
    check(cls_open and set(cls_open["view"]["card"]["applies_when"]) <= cls_signals, "классовые сигналы открывают карточку")
    profile = {
        "task_type": "classification",
        "n_classes": 3,
        "class_share": {"a": 0.9, "b": 0.09, "c": 0.01},
        "class_min_count": 10,
        "n_features": 4,
        "metric": "f1_macro",
        "higher_is_better": True,
        "high_cardinality": [],
        "time_candidates": [],
        "constant_metric": 0.1,
    }
    detected = detect_signals(profile, [], None)
    check("labels_imbalanced" in detected and "rare_class_metric_noisy" in detected, "детектор редкого класса")

    with tempfile.TemporaryDirectory() as raw:
        folder = Path(raw)
        frame = pd.DataFrame({
            "id": list(range(80)),
            "x": [i / 10 for i in range(80)],
            "g": [i % 5 for i in range(80)],
            "year": list(range(80)),
            "y": [i / 5 for i in range(80)],
        })
        test = pd.DataFrame({"id": list(range(80, 100)), "x": [1.0] * 20, "g": [0] * 20, "year": [80] * 20})
        frame.to_csv(folder / "train.csv", index=False)
        test.to_csv(folder / "test.csv", index=False)
        spec = {
            "estimator": "linear",
            "params": {},
            "features": [],
            "code": "",
            "split": {"name": "kfold"},
            "fidelity": {"mode": "cv", "folds": 3},
            "timeout_sec": 60,
        }
        result = run_experiment(_payload(folder, spec, "y_reg.csv", "val_reg.csv"))
        check(result.get("ok") is True and result["val_metric"] < 1, f"линейная регрессия на синтетике: {result.get('val_metric')}")
        group_spec = dict(spec)
        group_spec["split"] = {"name": "group", "column": "g"}
        group_spec["fidelity"] = {"mode": "holdout", "fraction": 0.2}
        grouped = run_experiment(_payload(folder, group_spec, "g.csv", "gv.csv"))
        check(grouped.get("ok") is True, f"групповой сплит: {grouped.get('error')}")
        time_spec = dict(spec)
        time_spec["split"] = {"name": "time", "column": "year"}
        time_spec["fidelity"] = {"mode": "holdout", "fraction": 0.2}
        timed = run_experiment(_payload(folder, time_spec, "t.csv", "tv.csv"))
        check(timed.get("ok") is True, f"временной сплит: {timed.get('error')}")
        code_spec = dict(spec)
        code_spec["code"] = (
            "def transform(train, test):\n"
            "    train = train.copy()\n"
            "    test = test.copy()\n"
            "    train['z'] = train['x'] * 2\n"
            "    test['z'] = test['x'] * 2\n"
            "    return train, test\n"
        )
        coded = run_experiment(_payload(folder, code_spec, "c.csv", "cv.csv"))
        check(coded.get("ok") is True, f"transform внутри фолда: {coded.get('error')}")
        try:
            apply_code(frame, test, "import os\ndef transform(train, test):\n    return train, test\n", "y", "id")
        except ValueError as error:
            check("импорт" in str(error), "import os отклонён")
        else:
            raise SystemExit("import os не был отклонён")
        payload_path = folder / "payload.json"
        spawn_spec = dict(spec)
        spawn_spec["fidelity"] = {"mode": "holdout", "fraction": 0.2}
        payload = _payload(folder, spawn_spec, "spawn.csv", "spawn_val.csv")
        payload["result_path"] = str(folder / "spawn_result.json")
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        process = multiprocessing.get_context("spawn").Process(target=child_main, args=(str(payload_path),))
        process.start()
        process.join(90)
        check(process.exitcode == 0 and (folder / "spawn_result.json").is_file(), "дочерний процесс завершился")
        spawned = json.loads((folder / "spawn_result.json").read_text(encoding="utf-8"))
        check(spawned.get("ok") is True, "дочерний процесс импортирует обучение")

    print("all checks passed")


def _payload(folder: Path, spec: dict, pred_name: str, val_name: str) -> dict:
    return {
        "train_path": str(folder / "train.csv"),
        "test_path": str(folder / "test.csv"),
        "spec": spec,
        "targets": ["y"],
        "id_column": "id",
        "task_type": "regression",
        "metric": "rmse",
        "n_jobs": 1,
        "result_path": str(folder / "spawn_result.json"),
        "test_pred_path": str(folder / pred_name),
        "val_path": str(folder / val_name),
        "final_only": False,
    }


if __name__ == "__main__":
    main()
