"""Чтение условия и запись submission.csv.

Пути ищутся от каталога task.md. Имена id, цели и метрики берутся из текста,
словаря и шаблона, а не из конкретного соревнования.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


FILE_NAMES = {
    "train": "train.csv",
    "test": "test.csv",
    "dictionary": "data_dictionary.csv",
    "sample": "sample_submission.csv",
}

HIGHER_IS_BETTER = {
    "f1_macro": True,
    "f1_weighted": True,
    "f1_micro": True,
    "accuracy": True,
    "roc_auc": True,
    "pr_auc": True,
    "r2": True,
    "kappa": True,
    "rmse": False,
    "mae": False,
    "rmsle": False,
    "mape": False,
    "logloss": False,
}

METRIC_PATTERNS = [
    (r"macro[\s-]*f1|f1[\s-]*macro|макро[\s-]*f1", "f1_macro"),
    (r"weighted[\s-]*f1|f1[\s-]*weighted", "f1_weighted"),
    (r"micro[\s-]*f1|f1[\s-]*micro", "f1_micro"),
    (r"\brmsle\b|root mean squared logarithmic", "rmsle"),
    (r"\brmse\b|root mean squared|среднеквадратич", "rmse"),
    (r"\bmae\b|mean absolute", "mae"),
    (r"\bmape\b", "mape"),
    (r"log[\s-]*loss|logloss|кросс-энтроп", "logloss"),
    (r"pr[\s-]*auc|average precision", "pr_auc"),
    (r"roc[\s-]*auc", "roc_auc"),
    (r"quadratic weighted kappa|\bkappa\b", "kappa"),
    (r"\br2\b|r\^2|коэффициент детерминации", "r2"),
    (r"\baccuracy\b|точност", "accuracy"),
]


@dataclass
class Task:
    path: Path
    directory: Path
    text: str
    train_path: Path
    test_path: Path
    dictionary_path: Path
    sample_path: Path
    id_column: str
    targets: list[str]
    submission_columns: list[str]
    metric: str
    higher_is_better: bool
    metric_snippet: str
    task_type: str
    descriptions: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def target(self) -> str:
        return self.targets[0]


def _directory_complete(folder: Path) -> bool:
    return all((folder / name).is_file() for name in FILE_NAMES.values())


def _mentioned_paths(text: str, directory: Path) -> dict[str, Path]:
    found = {}
    for key, name in FILE_NAMES.items():
        pattern = r"[^\s`'\"]*" + re.escape(name)
        for raw in re.findall(pattern, text):
            candidate = (directory / raw.strip("`\"'")).resolve()
            if candidate.is_file():
                found[key] = candidate
                break
    return found


def _search_directory(directory: Path) -> Path | None:
    """Ищет полный набор csv только внутри каталога task.md, без подъёма к чужим задачам."""
    if _directory_complete(directory):
        return directory
    inside = [item for item in directory.rglob("*") if item.is_dir() and _directory_complete(item)]
    if not inside:
        return None
    inside.sort(key=lambda item: (len(item.parts), str(item)))
    return inside[0]


def resolve_files(directory: Path, text: str) -> dict[str, Path]:
    mentioned = _mentioned_paths(text, directory)
    if len(mentioned) == len(FILE_NAMES):
        return mentioned
    folder = _search_directory(directory)
    if folder is None:
        missing = ", ".join(FILE_NAMES.values())
        raise FileNotFoundError(f"Не найдены файлы задачи ({missing}) от {directory}")
    return {key: folder / name for key, name in FILE_NAMES.items()}


def detect_metric(text: str) -> tuple[str, bool, str]:
    anchor = re.search(r"основн\w*\s+метрик\w*|метрик\w*\s+оценк\w*", text, flags=re.IGNORECASE)
    window = text[anchor.start(): anchor.start() + 700] if anchor else text
    for source in (window, text):
        for pattern, name in METRIC_PATTERNS:
            match = re.search(pattern, source, flags=re.IGNORECASE)
            if match:
                start = max(0, match.start() - 80)
                snippet = " ".join(source[start: match.end() + 80].split())
                return name, HIGHER_IS_BETTER[name], snippet
    return "", True, ""


def _role_map(dictionary: pd.DataFrame) -> tuple[str | None, list[str], dict[str, str]]:
    columns = {str(column).lower(): column for column in dictionary.columns}
    feature_col = columns.get("feature") or dictionary.columns[0]
    role_col = columns.get("role")
    description_col = columns.get("description_ru") or columns.get("description")
    descriptions = {}
    if description_col is not None:
        for _, row in dictionary.iterrows():
            name = str(row[feature_col])
            value = row[description_col]
            if pd.notna(value):
                descriptions[name] = str(value)
    id_name = None
    targets = []
    if role_col is not None:
        for _, row in dictionary.iterrows():
            role = str(row[role_col]).strip().lower()
            name = str(row[feature_col])
            if role == "id":
                id_name = name
            elif role == "target":
                targets.append(name)
    return id_name, targets, descriptions


def _infer_task_type(metric: str, train_path: Path, target: str) -> str:
    if metric in {"rmse", "mae", "rmsle", "mape", "r2"}:
        return "regression"
    if metric in {"f1_macro", "f1_weighted", "f1_micro", "accuracy", "roc_auc", "pr_auc", "logloss", "kappa"}:
        return "classification"
    series = pd.read_csv(train_path, usecols=[target])[target]
    if series.dtype == object or str(series.dtype).startswith("string"):
        return "classification"
    if series.nunique(dropna=True) <= 20:
        return "classification"
    return "regression"


def load_task(path: Path) -> Task:
    path = path.resolve()
    text = path.read_text(encoding="utf-8")
    files = resolve_files(path.parent, text)
    dictionary = pd.read_csv(files["dictionary"], encoding="utf-8")
    sample_columns = list(pd.read_csv(files["sample"], nrows=0, encoding="utf-8").columns)
    dict_id, dict_targets, descriptions = _role_map(dictionary)
    notes = []
    if dict_id and dict_id in sample_columns:
        id_column = dict_id
    else:
        id_column = sample_columns[0]
        notes.append("id взят из первого столбца sample_submission")
    sample_targets = [column for column in sample_columns if column != id_column]
    if dict_targets:
        targets = [column for column in sample_targets if column in dict_targets] or list(dict_targets)
        if set(targets) != set(sample_targets):
            notes.append("столбцы цели в словаре и шаблоне различаются, оставлен шаблон")
            targets = sample_targets
    else:
        train_header = set(pd.read_csv(files["train"], nrows=0, encoding="utf-8").columns)
        test_header = set(pd.read_csv(files["test"], nrows=0, encoding="utf-8").columns)
        only_train = [column for column in sample_targets if column in train_header and column not in test_header]
        targets = only_train or sample_targets
        notes.append("роль target в словаре не найдена, цель выведена из шаблона")
    metric, higher, snippet = detect_metric(text)
    if not metric:
        metric = "rmse"
        higher = False
        snippet = ""
        notes.append("метрика в тексте не найдена, для внутренней проверки взята rmse")
    task_type = _infer_task_type(metric, files["train"], targets[0])
    return Task(
        path=path,
        directory=path.parent,
        text=text,
        train_path=files["train"],
        test_path=files["test"],
        dictionary_path=files["dictionary"],
        sample_path=files["sample"],
        id_column=id_column,
        targets=targets,
        submission_columns=sample_columns,
        metric=metric,
        higher_is_better=higher,
        metric_snippet=snippet,
        task_type=task_type,
        descriptions=descriptions,
        notes=notes,
    )


def constant_frame(task: Task, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame({task.id_column: test[task.id_column].values})
    for target in task.targets:
        column = train[target]
        if task.task_type == "classification":
            value = column.mode(dropna=True).iloc[0]
        else:
            value = float(column.mean())
        frame[target] = value
    return frame[task.submission_columns]


def write_submission(task: Task, frame: pd.DataFrame) -> Path:
    destination = task.directory / "submission.csv"
    temporary = task.directory / ".submission.csv.tmp"
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    return destination


def verify_submission(task: Task) -> list[str]:
    path = task.directory / "submission.csv"
    errors = []
    if not path.is_file():
        return ["submission.csv не создан"]
    if path.stat().st_size == 0:
        return ["submission.csv пуст"]
    submission = pd.read_csv(path, encoding="utf-8")
    test = pd.read_csv(task.test_path, encoding="utf-8")
    train = pd.read_csv(task.train_path, usecols=task.targets, encoding="utf-8")
    if list(submission.columns) != task.submission_columns:
        errors.append(
            "колонки submission.csv не совпадают с шаблоном: "
            f"{list(submission.columns)} != {task.submission_columns}"
        )
    if len(submission) != len(test):
        errors.append(f"число строк {len(submission)} вместо {len(test)}")
    if task.id_column in submission.columns and task.id_column in test.columns:
        left = submission[task.id_column].astype(str).tolist()
        right = test[task.id_column].astype(str).tolist()
        if left != right:
            errors.append("id не совпадают с test.csv или изменён порядок строк")
        if submission[task.id_column].duplicated().any():
            errors.append("в submission.csv есть повторяющиеся id")
    if submission.isna().any().any():
        errors.append("в submission.csv есть пропуски")
    numeric = submission.select_dtypes(include="number")
    if not numeric.empty and np.isinf(numeric.to_numpy(dtype=float)).any():
        errors.append("в submission.csv есть бесконечные значения")
    if task.task_type == "classification":
        for target in task.targets:
            if target not in submission.columns:
                continue
            allowed = set(train[target].astype(str))
            got = set(submission[target].astype(str))
            extra = got - allowed
            if extra:
                errors.append(f"метки вне обучающих классов {target}: {sorted(extra)[:8]}")
    return errors
