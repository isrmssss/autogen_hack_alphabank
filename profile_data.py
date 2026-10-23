"""Фактический профиль таблицы. Здесь нет советов, какую модель взять."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from task_io import Task


TIME_NAME = re.compile(r"(^|_)(year|date|time|month|hour|timestamp)(_|$)", re.IGNORECASE)
PERIOD_TEXT = re.compile(r"год|время|период|дата", re.IGNORECASE)


def _kind(series: pd.Series) -> str:
    if series.dtype == object or str(series.dtype).startswith("string") or str(series.dtype) == "category":
        return "categorical"
    return "numeric"


def _constant_metric(task: Task, train: pd.DataFrame) -> float | None:
    target = train[task.target]
    if task.task_type == "classification":
        mode = target.mode(dropna=True).iloc[0]
        labels = list(pd.unique(target.dropna()))
        f1s = []
        for label in labels:
            support = int((target == label).sum())
            if label == mode:
                precision = support / len(target)
                recall = 1.0
                f1s.append(0.0 if precision == 0 else 2 * precision * recall / (precision + recall))
            else:
                f1s.append(0.0)
        if task.metric == "accuracy":
            return float((target == mode).mean())
        if task.metric in {"f1_macro", "f1_weighted", "f1_micro"}:
            return float(np.mean(f1s)) if f1s else 0.0
        return None
    prediction = float(target.mean())
    error = target.astype(float) - prediction
    if task.metric == "rmse":
        return float(np.sqrt(np.mean(error ** 2)))
    if task.metric == "mae":
        return float(np.mean(np.abs(error)))
    if task.metric == "rmsle":
        clipped = np.clip(target.astype(float), 0, None)
        return float(np.sqrt(np.mean((np.log1p(clipped) - np.log1p(max(prediction, 0))) ** 2)))
    return None


def build_profile(train: pd.DataFrame, test: pd.DataFrame, task: Task) -> dict:
    target = train[task.target]
    columns = []
    missing = []
    categoricals = []
    high_skew = []
    time_candidates = []
    group_candidates = []
    for name in train.columns:
        if name == task.id_column or name in task.targets:
            continue
        series = train[name]
        kind = _kind(series)
        missing_share = float(series.isna().mean())
        description = task.descriptions.get(name, "")
        item = {
            "name": name,
            "kind": kind,
            "missing": round(missing_share, 4),
            "description": description[:160],
        }
        if kind == "categorical":
            count = int(series.nunique(dropna=True))
            item["nunique"] = count
            categoricals.append({"name": name, "nunique": count, "missing": round(missing_share, 4)})
            if 2 <= count <= 200:
                group_candidates.append(name)
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            count = int(numeric.nunique(dropna=True))
            item["nunique"] = count
            if numeric.notna().sum() > 5:
                skew = float(numeric.skew())
                if abs(skew) > 1 and numeric.min() >= 0:
                    high_skew.append(name)
            if 2 <= count <= 50:
                group_candidates.append(name)
        if TIME_NAME.search(name):
            time_candidates.append(name)
        if missing_share > 0:
            missing.append({"column": name, "share": round(missing_share, 4)})
        columns.append(item)
    missing.sort(key=lambda item: item["share"], reverse=True)
    class_share = {}
    class_min_count = None
    target_skew = None
    if task.task_type == "classification":
        shares = target.astype(str).value_counts(normalize=True)
        counts = target.astype(str).value_counts()
        class_share = {str(key): round(float(value), 4) for key, value in shares.items()}
        class_min_count = int(counts.min()) if len(counts) else 0
    else:
        numeric_target = pd.to_numeric(target, errors="coerce")
        target_skew = float(numeric_target.skew()) if numeric_target.notna().any() else None
    high_card = [item["name"] for item in categoricals if item["nunique"] >= 30]
    return {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": len(columns),
        "task_type": task.task_type,
        "metric": task.metric,
        "higher_is_better": task.higher_is_better,
        "n_classes": int(target.nunique(dropna=True)) if task.task_type == "classification" else None,
        "class_share": class_share,
        "class_min_count": class_min_count,
        "target_skew": None if target_skew is None else round(target_skew, 3),
        "target_min": None if task.task_type == "classification" else float(pd.to_numeric(target, errors="coerce").min()),
        "target_max": None if task.task_type == "classification" else float(pd.to_numeric(target, errors="coerce").max()),
        "constant_metric": _constant_metric(task, train),
        "missing": missing[:20],
        "categoricals": categoricals[:30],
        "high_cardinality": high_card[:15],
        "high_skew_columns": high_skew[:15],
        "time_candidates": time_candidates[:15],
        "group_candidates": group_candidates[:15],
        "task_mentions_period": bool(PERIOD_TEXT.search(task.text)),
        "duplicate_rows": int(train.duplicated().sum()),
        "columns": columns,
    }
