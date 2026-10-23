"""Исполнение одной спецификации. Порядок оценщиков здесь ни на что не влияет."""

from __future__ import annotations

import ast
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


ESTIMATORS = ("linear", "lightgbm", "catboost", "xgboost", "torch_mlp", "flaml", "lightautoml")

TOOLBOX = [
    {
        "name": "linear",
        "note": "Ridge или логистическая регрессия со штатными параметрами. Пропуски заполняются медианой внутри фолда. Категории до 100 уровней — one-hot, более широкие — порядковый код.",
    },
    {
        "name": "lightgbm",
        "note": "Штатные параметры LightGBM. Категории передаются как categorical.",
    },
    {
        "name": "catboost",
        "note": "Штатные параметры CatBoost. Категории передаются строками. Журнал CatBoost на диск не пишется.",
    },
    {
        "name": "xgboost",
        "note": "Штатные параметры XGBoost. Категории внутри фолда заменяются порядковым кодом.",
    },
    {
        "name": "torch_mlp",
        "note": "Небольшая сеть на CPU. Пустые params значат hidden=64, epochs=5, lr=0.001, batch=256. Пропуски — медиана, категории — порядковый код. Масштаб сам не включается.",
    },
    {
        "name": "flaml",
        "note": "Поиск FLAML только при fidelity.mode=holdout. Свой перекрёстный поиск поверх него не запускается. time_budget равен timeout_sec.",
    },
    {
        "name": "lightautoml",
        "note": "TabularAutoML только при fidelity.mode=holdout. Таймаут равен timeout_sec.",
    },
]

SPEC_HELP = {
    "split": {"name": "kfold | stratified | group | time", "column": "для group и time", "folds": "для cv можно продублировать сюда"},
    "fidelity": {"mode": "holdout | cv", "fraction": 0.2, "folds": 3, "row_fraction": 1},
    "features": [
        "missing_indicator",
        "log1p_skew",
        "sincos_degrees",
        "frequency",
        "target_encoding",
        "scale",
        "log_target",
        {"name": "ratio", "numerator": "col_a", "denominator": "col_b"},
        {"name": "groupby", "group": "col", "columns": ["col"], "agg": "mean"},
    ],
}

ANGLE_NAME = ("direction", "angle", "azimuth", "heading")
BANNED_CALLS = {
    "open", "exec", "eval", "compile", "__import__", "getattr", "setattr", "globals",
    "locals", "vars", "input", "breakpoint", "help",
}
BANNED_ATTRS = {
    "read_csv", "read_parquet", "read_pickle", "to_csv", "to_pickle", "to_json",
    "load", "save", "system", "popen", "remove", "unlink", "mkdir",
}
BANNED_NAMES = {"os", "sys", "subprocess", "socket", "pathlib", "shutil", "requests", "builtins", "pickle"}
SAFE_BUILTINS = {
    "range": range, "len": len, "min": min, "max": max, "sum": sum, "abs": abs,
    "float": float, "int": int, "str": str, "bool": bool, "list": list, "dict": dict,
    "set": set, "tuple": tuple, "enumerate": enumerate, "zip": zip, "map": map,
    "sorted": sorted, "round": round, "print": print, "pow": pow,
}


def child_main(payload_path: str) -> None:
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    try:
        result = run_experiment(payload)
    except Exception as error:
        result = {
            "ok": False,
            "error": f"{type(error).__name__}: {error}"[:800],
            "trace": traceback.format_exc()[-2000:],
        }
    Path(payload["result_path"]).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


def run_experiment(payload: dict) -> dict:
    problems = validate_spec(payload["spec"])
    if problems:
        return {"ok": False, "error": "; ".join(problems)}
    if payload.get("final_only"):
        return _fit_full(payload)
    targets = list(payload["targets"])
    if len(targets) == 1:
        return _run_single(payload, targets[0], payload["test_pred_path"], payload["val_path"])
    metrics = []
    stds = []
    pred_frames = []
    val_path = None
    for index, target in enumerate(targets):
        pred_path = str(Path(payload["test_pred_path"]).with_name(f"target_{index}.csv"))
        one_val = str(Path(payload["val_path"]).with_name(f"val_{index}.csv"))
        result = _run_single(payload, target, pred_path, one_val)
        if not result.get("ok"):
            result["error"] = f"{target}: {result.get('error')}"
            return result
        metrics.append(result["val_metric"])
        if result.get("val_std") is not None:
            stds.append(result["val_std"])
        pred_frames.append(pd.read_csv(pred_path))
        if index == 0:
            val_path = one_val
    merged = pd.concat(pred_frames, axis=1)
    merged.to_csv(payload["test_pred_path"], index=False)
    return {
        "ok": True,
        "val_metric": float(np.mean(metrics)),
        "train_metric": None,
        "val_std": float(np.mean(stds)) if stds else None,
        "n_splits": result["n_splits"],
        "test_pred_path": payload["test_pred_path"],
        "val_path": val_path,
    }


def _fit_full(payload: dict) -> dict:
    """Дообучение чемпиона на всём train. Валидацию заново не выбирает."""
    spec = payload["spec"]
    train = pd.read_csv(payload["train_path"])
    test = pd.read_csv(payload["test_path"])
    frames = []
    for target in payload["targets"]:
        fitted = _fit_fold(train, test, spec, target, payload, final=True)
        prediction = fitted["other_pred"]
        frames.append(pd.DataFrame({target: _as_column(prediction)}))
    merged = pd.concat(frames, axis=1)
    if payload["task_type"] == "regression" and not np.isfinite(merged.to_numpy(dtype=float)).all():
        return {"ok": False, "error": "дообучение дало бесконечные предсказания"}
    merged.to_csv(payload["test_pred_path"], index=False)
    return {"ok": True, "test_pred_path": payload["test_pred_path"]}


def _run_single(payload: dict, target: str, pred_path: str, val_path: str) -> dict:
    spec = payload["spec"]
    train = pd.read_csv(payload["train_path"])
    test = pd.read_csv(payload["test_path"])
    work = _subsample(train, spec, payload["task_type"], target)
    positions = make_splits(work, spec, payload["task_type"], target)
    val_metrics = []
    train_metrics = []
    val_parts = []
    best_iters = []
    for train_pos, valid_pos in positions:
        fold_train = work.iloc[train_pos]
        fold_valid = work.iloc[valid_pos]
        fitted = _fit_fold(fold_train, fold_valid, spec, target, payload, final=False)
        best_iters.extend(fitted["best_iters"])
        valid_pred = fitted["other_pred"]
        train_pred = fitted["train_pred"]
        val_metrics.append(_metric(fold_valid[target], valid_pred, spec, payload))
        train_metrics.append(_metric(fold_train[target], train_pred, spec, payload))
        val_parts.append(pd.DataFrame({
            "row_index": work.index.to_numpy()[valid_pos],
            "y_true": fold_valid[target].to_numpy(),
            "y_pred": _as_column(valid_pred),
        }))
    final_spec = dict(spec)
    final_params = dict(spec.get("params") or {})
    if best_iters:
        final_params["n_estimators"] = max(1, int(np.median(best_iters)))
        final_params.pop("early_stopping_rounds", None)
        final_spec["params"] = final_params
    fitted = _fit_fold(work, test, final_spec, target, payload, final=True)
    test_pred = fitted["other_pred"]
    if payload["task_type"] == "regression" and not np.isfinite(np.asarray(test_pred, dtype=float)).all():
        return {"ok": False, "error": "предсказания содержат бесконечности"}
    pd.DataFrame({target: _as_column(test_pred)}).to_csv(pred_path, index=False)
    pd.concat(val_parts, ignore_index=True).to_csv(val_path, index=False)
    std = float(np.std(val_metrics)) if len(val_metrics) > 1 else None
    return {
        "ok": True,
        "val_metric": float(np.mean(val_metrics)),
        "train_metric": float(np.mean(train_metrics)),
        "val_std": std,
        "n_splits": len(val_metrics),
        "test_pred_path": pred_path,
        "val_path": val_path,
    }


def validate_spec(spec: dict) -> list[str]:
    errors = []
    if spec.get("estimator") not in ESTIMATORS:
        errors.append(f"неизвестный estimator, доступны: {', '.join(ESTIMATORS)}")
    params = spec.get("params") or {}
    if not isinstance(params, dict):
        errors.append("params должен быть объектом")
    features = spec.get("features") or []
    if not isinstance(features, list):
        errors.append("features должен быть списком")
    split = spec.get("split")
    if isinstance(split, str):
        spec["split"] = {"name": split}
        split = spec["split"]
    if not isinstance(split, dict) or split.get("name") not in {"kfold", "stratified", "group", "time"}:
        errors.append("split.name должен быть kfold, stratified, group или time")
    elif split["name"] in {"group", "time"} and not split.get("column"):
        errors.append("для group и time нужна split.column")
    fidelity = spec.get("fidelity")
    if not isinstance(fidelity, dict) or fidelity.get("mode") not in {"holdout", "cv"}:
        errors.append("fidelity.mode должен быть holdout или cv")
    elif fidelity["mode"] == "holdout" and "fraction" not in fidelity:
        errors.append("для holdout нужен fidelity.fraction")
    elif fidelity["mode"] == "cv" and "folds" not in fidelity:
        errors.append("для cv нужен fidelity.folds")
    if spec.get("estimator") in {"flaml", "lightautoml"} and isinstance(fidelity, dict) and fidelity.get("mode") != "holdout":
        errors.append(f"{spec.get('estimator')} запускается только с fidelity.mode=holdout")
    try:
        if float(spec.get("timeout_sec")) <= 0:
            errors.append("timeout_sec должен быть больше нуля")
    except (TypeError, ValueError):
        errors.append("нужен положительный timeout_sec")
    code = spec.get("code") or ""
    if code and len(str(code)) > 4000:
        errors.append("code длиннее 4000 символов")
    return errors


def _subsample(train: pd.DataFrame, spec: dict, task_type: str, target: str) -> pd.DataFrame:
    if spec["split"]["name"] == "time":
        return train
    fraction = float(spec["fidelity"].get("row_fraction", 1) or 1)
    if fraction >= 0.999:
        return train
    if task_type == "classification":
        parts = []
        for _, group in train.groupby(target, sort=False):
            take = max(1, int(round(len(group) * fraction)))
            parts.append(group.sample(n=min(take, len(group)), random_state=0))
        return pd.concat(parts).sort_index()
    size = max(30, int(round(len(train) * fraction)))
    return train.sample(n=min(size, len(train)), random_state=0)


def make_splits(work: pd.DataFrame, spec: dict, task_type: str, target: str) -> list[tuple[np.ndarray, np.ndarray]]:
    name = spec["split"]["name"]
    fidelity = spec["fidelity"]
    y = work[target]
    if name in {"group", "time"}:
        column = spec["split"]["column"]
        if column not in work.columns:
            raise ValueError(f"колонки сплита нет: {column}")
        if work[column].isna().any():
            raise ValueError(f"в колонке сплита {column} есть пропуски")
        work.attrs["split_column"] = column
    if name == "stratified" and task_type != "classification":
        raise ValueError("stratified применим к классификации")
    if fidelity["mode"] == "holdout":
        fraction = float(fidelity["fraction"])
        if not 0.05 <= fraction <= 0.5:
            raise ValueError("fidelity.fraction должен быть от 0.05 до 0.5")
        return [_holdout(work, name, fraction, task_type, target)]
    folds = int(fidelity["folds"])
    if folds < 2 or folds > 10:
        raise ValueError("fidelity.folds должен быть от 2 до 10")
    return _cv(work, name, folds, task_type, target)


def _holdout(work, name, fraction, task_type, target):
    index = np.arange(len(work))
    if name == "time":
        return _time_holdout(work, fraction)
    if name == "group":
        return _group_holdout(work, fraction)
    from sklearn.model_selection import train_test_split
    stratify = work[target] if name == "stratified" else None
    train_pos, valid_pos = train_test_split(
        index, test_size=fraction, random_state=0, stratify=stratify
    )
    return np.asarray(train_pos), np.asarray(valid_pos)


def _time_holdout(work, fraction):
    column = _split_column(work)
    order = np.argsort(work[column].to_numpy(), kind="mergesort")
    cut = int(round(len(order) * (1 - fraction)))
    cut = min(max(cut, 1), len(order) - 1)
    return order[:cut], order[cut:]


def _group_holdout(work, fraction):
    column = _split_column(work)
    values = work[column].to_numpy()
    unique = pd.unique(values)
    if len(unique) < 2:
        raise ValueError("для group нужно хотя бы две группы")
    count = min(len(unique) - 1, max(1, int(round(len(unique) * fraction))))
    chosen = set(pd.Series(unique).sample(n=count, random_state=0).tolist())
    valid = np.flatnonzero(pd.Series(values).isin(chosen).to_numpy())
    train = np.flatnonzero(~pd.Series(values).isin(chosen).to_numpy())
    if len(train) == 0 or len(valid) == 0:
        raise ValueError("групповой holdout оставил пустую часть")
    return train, valid


def _split_column(work: pd.DataFrame) -> str:
    column = work.attrs.get("split_column")
    if not column:
        raise ValueError("не задана колонка сплита")
    return column


def _cv(work, name, folds, task_type, target):
    y = work[target]
    if name == "time":
        from sklearn.model_selection import TimeSeriesSplit
        order = np.argsort(work[_split_column(work)].to_numpy(), kind="mergesort")
        splitter = TimeSeriesSplit(n_splits=folds)
        return [(order[train], order[valid]) for train, valid in splitter.split(np.arange(len(work)))]
    if name == "group":
        from sklearn.model_selection import GroupKFold
        groups = work[_split_column(work)]
        n_groups = int(groups.nunique(dropna=True))
        if n_groups < 2:
            raise ValueError("для group нужно хотя бы две группы")
        splitter = GroupKFold(n_splits=min(folds, n_groups))
        return [(train, valid) for train, valid in splitter.split(work, y, groups)]
    if name == "stratified":
        from sklearn.model_selection import StratifiedKFold
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
        return [(train, valid) for train, valid in splitter.split(work, y)]
    from sklearn.model_selection import KFold
    splitter = KFold(n_splits=folds, shuffle=True, random_state=0)
    return [(train, valid) for train, valid in splitter.split(work)]


def _fit_fold(train_df, other_df, spec, target, payload, final: bool) -> dict:
    train_df = train_df.copy()
    other_df = other_df.copy()
    if spec["split"]["name"] in {"group", "time"}:
        train_df.attrs["split_column"] = spec["split"]["column"]
        other_df.attrs["split_column"] = spec["split"]["column"]
    steps = list(spec.get("features") or [])
    code = spec.get("code") or ""
    train_df, other_df, log_target = apply_features(
        train_df, other_df, steps, target, payload["task_type"], payload["id_column"]
    )
    if code:
        train_df, other_df = apply_code(train_df, other_df, code, target, payload["id_column"])
    y_fit, classes = _fit_target(train_df[target], payload["task_type"], log_target)
    valid_for_fit = None if final else other_df
    model = _fit_model(
        train_df, valid_for_fit, y_fit, classes, spec, payload, log_target, target,
    )
    return {
        "bundle": model,
        "classes": classes,
        "log_target": log_target,
        "task_type": payload["task_type"],
        "best_iters": _best_iteration(model),
        "train_pred": _predict_original(model, train_df, classes, payload, log_target, target),
        "other_pred": _predict_original(model, other_df, classes, payload, log_target, target),
    }


def _fit_target(y, task_type, log_target):
    if task_type == "classification":
        classes = sorted(pd.unique(y.dropna()), key=str)
        mapping = {label: index for index, label in enumerate(classes)}
        encoded = y.map(mapping)
        if encoded.isna().any():
            raise ValueError("в цели классификации есть пропуски")
        return encoded.astype(int), classes
    numeric = pd.to_numeric(y, errors="coerce")
    if numeric.isna().any():
        raise ValueError("в числовой цели есть пропуски")
    if log_target:
        if (numeric < 0).any():
            raise ValueError("log_target нельзя применить к отрицательной цели")
        numeric = np.log1p(numeric)
    return numeric.astype(float), None


def _encode_y(y, task_type, log_target, classes):
    if task_type != "classification":
        numeric = pd.to_numeric(y, errors="coerce")
        if numeric.isna().any():
            return None
        if log_target:
            if (numeric < 0).any():
                return None
            return np.log1p(numeric.astype(float))
        return numeric.astype(float)
    if not classes:
        return None
    encoded = y.map({label: index for index, label in enumerate(classes)})
    if encoded.isna().any():
        return None
    return encoded.astype(int)


def _predict_original(bundle, frame, classes, payload, log_target, target):
    feature_frame = _model_frame(frame, payload["id_column"], target)
    raw = _predict_raw(bundle, feature_frame, frame, payload["task_type"])
    if payload["task_type"] == "classification":
        flat = np.asarray(raw).reshape(-1)
        if np.issubdtype(flat.dtype, np.number):
            indexes = np.rint(flat.astype(float)).astype(int)
            return np.array([classes[index] if 0 <= index < len(classes) else classes[0] for index in indexes])
        return flat
    pred = np.asarray(raw, dtype=float).reshape(-1)
    if log_target:
        pred = np.expm1(pred)
    return pred


def _predict_raw(bundle, feature_frame, full_frame, task_type):
    kind = bundle.get("kind")
    model = bundle["model"]
    if kind == "sklearn":
        return model.predict(feature_frame)
    if kind == "lgb":
        frame = feature_frame if _has_category(feature_frame) else _as_lgb_categories(feature_frame, bundle.get("cats") or [])
        return model.predict(frame)
    if kind == "cat":
        return model.predict(_as_strings(feature_frame, bundle.get("cats") or []))
    if kind == "matrix":
        return model.predict(bundle["transformer"].transform(feature_frame))
    if kind == "torch":
        return _torch_predict(bundle, feature_frame)
    if kind == "lama":
        present = full_frame.drop(columns=[bundle.get("target")], errors="ignore")
        raw = model.predict(present)
        data = raw.data if hasattr(raw, "data") else np.asarray(raw)
        if getattr(data, "ndim", 1) == 2:
            return data[:, 0] if task_type == "regression" else data.argmax(axis=1)
        return np.asarray(data).reshape(-1)
    raise ValueError(f"нет предсказания для {kind}")


def _fit_model(train_df, valid_df, y_fit, classes, spec, payload, log_target, target):
    estimator = spec["estimator"]
    params = dict(spec.get("params") or {})
    n_jobs = int(payload.get("n_jobs") or 1)
    x_train = _model_frame(train_df, payload["id_column"], target)
    x_valid = _model_frame(valid_df, payload["id_column"], target) if valid_df is not None else None
    y_valid_fit = _encode_y(valid_df[target], payload["task_type"], log_target, classes) if valid_df is not None else None
    cats = [column for column in x_train.columns if _is_cat(x_train[column])]
    if estimator == "linear":
        return _fit_linear(x_train, y_fit, params, payload["task_type"], classes, n_jobs, cats)
    if estimator == "lightgbm":
        return _fit_lightgbm(x_train, y_fit, x_valid, y_valid_fit, params, payload["task_type"], cats, n_jobs)
    if estimator == "catboost":
        return _fit_catboost(x_train, y_fit, x_valid, y_valid_fit, params, payload["task_type"], cats, n_jobs)
    if estimator == "xgboost":
        return _fit_xgboost(x_train, y_fit, x_valid, y_valid_fit, params, payload["task_type"], cats, n_jobs)
    if estimator == "torch_mlp":
        return _fit_torch(x_train, y_fit, params, payload["task_type"], classes, cats, n_jobs)
    if estimator == "flaml":
        return _fit_flaml(x_train, y_fit, params, payload, cats)
    if estimator == "lightautoml":
        return _fit_lightautoml(train_df, valid_df, target, payload)
    raise ValueError(f"неизвестный estimator {estimator}")


def _model_frame(frame, id_column, target):
    if frame is None:
        return None
    columns = [column for column in frame.columns if column != id_column and column != target]
    return frame[columns].copy()


def _is_cat(series: pd.Series) -> bool:
    return series.dtype == object or str(series.dtype).startswith("string") or str(series.dtype) == "category"


def _fit_linear(x_train, y_fit, params, task_type, classes, n_jobs, cats):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

    wide = [column for column in cats if x_train[column].nunique(dropna=True) > 100]
    narrow = [column for column in cats if column not in wide]
    numbers = [column for column in x_train.columns if column not in cats]
    transformers = []
    if numbers:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", "passthrough"),
        ]), numbers))
    if narrow:
        transformers.append(("onehot", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), narrow))
    if wide:
        transformers.append(("ordinal", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), wide))
    matrix = ColumnTransformer(transformers, remainder="drop")
    if task_type == "classification":
        clf_params = dict(params)
        clf_params.setdefault("max_iter", clf_params.get("max_iter", 200))
        clf_params.setdefault("n_jobs", n_jobs)
        model = LogisticRegression(**{key: value for key, value in clf_params.items() if key != "early_stopping_rounds"})
    else:
        model = Ridge(**{key: value for key, value in params.items() if key not in {"n_jobs", "early_stopping_rounds"}})
    pipe = Pipeline([("matrix", matrix), ("model", model)])
    pipe.fit(x_train, y_fit)
    return {"kind": "sklearn", "model": pipe, "classes": classes, "cats": cats}


def _fit_lightgbm(x_train, y_fit, x_valid, y_valid, params, task_type, cats, n_jobs):
    import lightgbm as lgb

    clean = dict(params)
    early = clean.pop("early_stopping_rounds", None)
    clean.setdefault("n_jobs", n_jobs)
    clean.setdefault("random_state", 0)
    clean.setdefault("verbose", -1)
    model = (lgb.LGBMRegressor if task_type == "regression" else lgb.LGBMClassifier)(**clean)
    x_train = _as_lgb_categories(x_train, cats)
    fit_args = {}
    if early and x_valid is not None and y_valid is not None:
        x_valid = _as_lgb_categories(x_valid, cats)
        fit_args["eval_set"] = [(x_valid, y_valid)]
        fit_args["callbacks"] = [lgb.early_stopping(int(early), verbose=False)]
    model.fit(x_train, y_fit, categorical_feature=[column for column in cats if column in x_train.columns], **fit_args)
    return {"kind": "lgb", "model": model, "classes": None if task_type == "regression" else _classes_from_fit(y_fit), "cats": cats, "task_type": task_type}


def _classes_from_fit(y_fit):
    return list(range(int(np.max(y_fit)) + 1))


def _as_lgb_categories(frame, cats):
    frame = frame.copy()
    for column in cats:
        if column in frame.columns:
            frame[column] = frame[column].astype("category")
    return frame


def _fit_catboost(x_train, y_fit, x_valid, y_valid, params, task_type, cats, n_jobs):
    from catboost import CatBoostClassifier, CatBoostRegressor

    clean = dict(params)
    early = clean.pop("early_stopping_rounds", None)
    clean.setdefault("thread_count", n_jobs)
    clean.setdefault("random_seed", 0)
    clean.setdefault("logging_level", "Silent")
    clean["allow_writing_files"] = False
    model = (CatBoostRegressor if task_type == "regression" else CatBoostClassifier)(**clean)
    x_train = _as_strings(x_train, cats)
    cat_idx = [x_train.columns.get_loc(column) for column in cats if column in x_train.columns]
    fit_args = {"cat_features": cat_idx}
    if early and x_valid is not None and y_valid is not None:
        fit_args["eval_set"] = (_as_strings(x_valid, cats), y_valid)
        fit_args["early_stopping_rounds"] = int(early)
        fit_args["use_best_model"] = True
    model.fit(x_train, y_fit, **fit_args)
    return {"kind": "cat", "model": model, "classes": None if task_type == "regression" else list(getattr(model, "classes_", [])), "cats": cats}


def _as_strings(frame, cats):
    frame = frame.copy()
    for column in cats:
        if column in frame.columns:
            frame[column] = frame[column].astype("string").fillna("__missing__").astype(str)
    return frame


def _fit_xgboost(x_train, y_fit, x_valid, y_valid, params, task_type, cats, n_jobs):
    from xgboost import XGBClassifier, XGBRegressor

    transformer = _OrdinalMatrix(cats)
    matrix = transformer.fit_transform(x_train)
    clean = {key: value for key, value in params.items() if key != "early_stopping_rounds"}
    early = params.get("early_stopping_rounds")
    clean.setdefault("n_jobs", n_jobs)
    clean.setdefault("random_state", 0)
    clean.setdefault("verbosity", 0)
    model = (XGBRegressor if task_type == "regression" else XGBClassifier)(**clean)
    fit_args = {}
    if early and x_valid is not None and y_valid is not None:
        fit_args["eval_set"] = [(transformer.transform(x_valid), y_valid)]
        fit_args["verbose"] = False
        model.set_params(early_stopping_rounds=int(early))
    model.fit(matrix, y_fit, **fit_args)
    return {"kind": "matrix", "model": model, "transformer": transformer, "classes": None if task_type == "regression" else list(range(int(np.max(y_fit)) + 1))}


def _fit_torch(x_train, y_fit, params, task_type, classes, cats, n_jobs):
    import torch
    from torch import nn

    torch.set_num_threads(max(1, n_jobs))
    transformer = _OrdinalMatrix(cats)
    matrix = transformer.fit_transform(x_train).astype(np.float32)
    hidden = int(params.get("hidden", 64))
    epochs = int(params.get("epochs", 5))
    lr = float(params.get("lr", 1e-3))
    batch = int(params.get("batch", 256))
    output = 1 if task_type == "regression" else len(classes)
    network = nn.Sequential(nn.Linear(matrix.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, output))
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    loss_fn = nn.MSELoss() if task_type == "regression" else nn.CrossEntropyLoss()
    features = torch.tensor(matrix)
    target = torch.tensor(np.asarray(y_fit), dtype=torch.float32 if task_type == "regression" else torch.long)
    network.train()
    for _ in range(max(1, epochs)):
        permutation = torch.randperm(len(features))
        for start in range(0, len(features), max(1, batch)):
            batch_index = permutation[start: start + batch]
            optimizer.zero_grad()
            pred = network(features[batch_index])
            truth = target[batch_index]
            loss = loss_fn(pred.squeeze(-1), truth) if task_type == "regression" else loss_fn(pred, truth)
            loss.backward()
            optimizer.step()
    return {"kind": "torch", "model": network, "transformer": transformer, "classes": classes, "task_type": task_type}


def _fit_flaml(x_train, y_fit, params, payload, cats):
    from flaml import AutoML

    transformer = _OrdinalMatrix(cats)
    matrix = transformer.fit_transform(x_train)
    budget = max(5, int(float(payload["spec"]["timeout_sec"]) - 15))
    metric = {
        "rmse": "rmse",
        "mae": "mae",
        "r2": "r2",
        "rmsle": "rmse",
        "accuracy": "accuracy",
        "f1_macro": "macro_f1",
        "f1_weighted": "f1",
        "logloss": "log_loss",
        "roc_auc": "roc_auc",
    }.get(payload["metric"], "auto")
    model = AutoML()
    model.fit(
        matrix,
        np.asarray(y_fit),
        task="regression" if payload["task_type"] == "regression" else "classification",
        time_budget=budget,
        metric=metric,
        n_jobs=int(payload.get("n_jobs") or 1),
        verbose=0,
        **{key: value for key, value in params.items() if key not in {"early_stopping_rounds"}},
    )
    return {"kind": "matrix", "model": model, "transformer": transformer, "classes": None}


def _fit_lightautoml(train_df, valid_df, target, payload):
    from lightautoml.automl.presets.tabular_presets import TabularAutoML
    from lightautoml.tasks import Task

    task_name = "reg" if payload["task_type"] == "regression" else "multiclass"
    if payload["task_type"] == "classification" and train_df[target].nunique() == 2:
        task_name = "binary"
    timeout = max(5, int(float(payload["spec"]["timeout_sec"]) - 15))
    model = TabularAutoML(task=Task(task_name), timeout=timeout, cpu_limit=int(payload.get("n_jobs") or 1))
    model.fit_predict(train_df, roles={"target": target})
    return {"kind": "lama", "model": model, "target": target, "classes": None}


class _OrdinalMatrix:
    def __init__(self, cats: list[str]) -> None:
        self.cats = list(cats)
        self.maps = {}
        self.columns = []
        self.medians = {}

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        self.columns = list(frame.columns)
        self.maps = {}
        self.medians = {}
        for column in self.cats:
            if column not in frame.columns:
                continue
            values = frame[column].astype("string").fillna("__missing__")
            categories = pd.unique(values)
            self.maps[column] = {value: index for index, value in enumerate(categories)}
        for column in self.columns:
            if column in self.maps:
                continue
            numeric = pd.to_numeric(frame[column], errors="coerce")
            self.medians[column] = float(numeric.median()) if numeric.notna().any() else 0.0
        return self.transform(frame)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        columns = []
        for column in self.columns:
            if column in self.maps:
                values = frame[column].astype("string").fillna("__missing__")
                mapping = self.maps[column]
                columns.append(values.map(lambda value: mapping.get(value, -1)).to_numpy(dtype=float))
            else:
                numeric = pd.to_numeric(frame[column], errors="coerce")
                columns.append(numeric.fillna(self.medians.get(column, 0.0)).to_numpy(dtype=float))
        if not columns:
            raise ValueError("после подготовки не осталось признаков")
        return np.column_stack(columns)


def _best_iteration(model_bundle) -> list[int]:
    model = model_bundle.get("model") if isinstance(model_bundle, dict) else model_bundle
    for attr in ("best_iteration_", "best_iteration"):
        value = getattr(model, attr, None)
        if isinstance(value, int) and value > 0:
            return [value]
    getter = getattr(model, "get_best_iteration", None)
    if callable(getter):
        try:
            value = int(getter())
        except Exception:
            value = 0
        if value > 0:
            return [value]
    return []


def _has_category(frame) -> bool:
    return any(str(dtype) == "category" for dtype in frame.dtypes)


def _torch_predict(bundle, frame):
    import torch

    network = bundle["model"]
    network.eval()
    matrix = bundle["transformer"].transform(frame).astype(np.float32)
    with torch.no_grad():
        pred = network(torch.tensor(matrix)).numpy()
    if bundle["task_type"] == "regression":
        return pred.reshape(-1)
    return pred.argmax(axis=1)


def _metric(y_true, y_pred, spec, payload) -> float:
    metric = payload["metric"]
    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if payload["task_type"] == "regression":
        truth = truth.astype(float)
        pred = pred.astype(float)
        if not np.isfinite(pred).all():
            raise ValueError("валидационные предсказания содержат бесконечности")
        if metric == "rmse":
            return float(np.sqrt(np.mean((truth - pred) ** 2)))
        if metric == "mae":
            return float(np.mean(np.abs(truth - pred)))
        if metric == "rmsle":
            return float(np.sqrt(np.mean((np.log1p(np.clip(truth, 0, None)) - np.log1p(np.clip(pred, 0, None))) ** 2)))
        if metric == "mape":
            return float(np.mean(np.abs(truth - pred) / np.clip(np.abs(truth), 1e-8, None)))
        if metric == "r2":
            center = np.mean(truth)
            denom = np.sum((truth - center) ** 2)
            return float(1 - np.sum((truth - pred) ** 2) / denom) if denom else 0.0
        raise ValueError(f"метрика {metric} не считается для регрессии")
    from sklearn.metrics import accuracy_score, f1_score

    truth_s = pd.Series(truth).astype(str)
    pred_s = pd.Series(pred).astype(str)
    if metric == "f1_macro":
        return float(f1_score(truth_s, pred_s, average="macro"))
    if metric == "f1_weighted":
        return float(f1_score(truth_s, pred_s, average="weighted"))
    if metric == "f1_micro":
        return float(f1_score(truth_s, pred_s, average="micro"))
    if metric == "accuracy":
        return float(accuracy_score(truth_s, pred_s))
    raise ValueError(f"метрика {metric} для этого оценщика не посчитана: нужны вероятности, которых спецификация не сохраняет")


def _as_column(values) -> np.ndarray:
    return np.asarray(values).reshape(-1)


def apply_features(train_df, other_df, steps, target, task_type, id_column):
    log_target = False
    for step in steps:
        name, options = _normalize_step(step)
        if name == "log_target":
            log_target = True
            continue
        handler = FEATURES.get(name)
        if handler is None:
            known = ", ".join(sorted(FEATURES))
            raise ValueError(f"неизвестный шаг признаков {name}. Доступны: {known}, log_target")
        train_df, other_df = handler(train_df, other_df, options, target, task_type, id_column)
    return train_df, other_df, log_target


def _normalize_step(step):
    if isinstance(step, str):
        return step, {}
    if isinstance(step, dict):
        if step.get("fit_on") in {"all", "test"} or step.get("include_test") is True:
            raise ValueError("шаг не может учиться на test или на всех строках сразу")
        name = step.get("name") or step.get("step")
        if not name:
            raise ValueError("у шага признаков нет name")
        return str(name), step
    raise ValueError("шаг признаков должен быть строкой или объектом")


def _missing_indicator(train_df, other_df, options, target, task_type, id_column):
    columns = [column for column in train_df.columns if column not in {target, id_column} and train_df[column].isna().any()]
    for column in columns[:30]:
        name = f"{column}__isna"
        train_df[name] = train_df[column].isna().astype(np.int8)
        other_df[name] = other_df[column].isna().astype(np.int8) if column in other_df.columns else 0
    return train_df, other_df


def _log1p_skew(train_df, other_df, options, target, task_type, id_column):
    for column in list(train_df.columns):
        if column in {target, id_column} or _is_cat(train_df[column]):
            continue
        numeric = pd.to_numeric(train_df[column], errors="coerce")
        if numeric.notna().sum() < 5 or numeric.min() < 0 or abs(float(numeric.skew())) <= 1:
            continue
        name = f"{column}__log1p"
        train_df[name] = np.log1p(numeric.clip(lower=0))
        other_numeric = pd.to_numeric(other_df[column], errors="coerce") if column in other_df.columns else 0
        other_df[name] = np.log1p(pd.Series(other_numeric).clip(lower=0))
    return train_df, other_df


def _sincos(train_df, other_df, options, target, task_type, id_column):
    for column in list(train_df.columns):
        if column in {target, id_column} or not any(token in column.lower() for token in ANGLE_NAME):
            continue
        for frame in (train_df, other_df):
            if column not in frame.columns:
                frame[f"{column}__sin"] = np.nan
                frame[f"{column}__cos"] = np.nan
                continue
            radians = np.deg2rad(pd.to_numeric(frame[column], errors="coerce"))
            frame[f"{column}__sin"] = np.sin(radians)
            frame[f"{column}__cos"] = np.cos(radians)
    return train_df, other_df


def _frequency(train_df, other_df, options, target, task_type, id_column):
    for column in list(train_df.columns):
        if column in {target, id_column} or not _is_cat(train_df[column]):
            continue
        share = train_df[column].astype("string").fillna("__missing__").value_counts(normalize=True)
        name = f"{column}__freq"
        train_df[name] = train_df[column].astype("string").fillna("__missing__").map(share).fillna(0).astype(float)
        if column in other_df.columns:
            other_df[name] = other_df[column].astype("string").fillna("__missing__").map(share).fillna(0).astype(float)
        else:
            other_df[name] = 0.0
    return train_df, other_df


def _target_encoding(train_df, other_df, options, target, task_type, id_column):
    smooth = 20.0
    for column in list(train_df.columns):
        if column in {target, id_column} or not _is_cat(train_df[column]) or train_df[column].nunique(dropna=True) < 10:
            continue
        keys = train_df[column].astype("string").fillna("__missing__")
        if column in other_df.columns:
            other_keys = other_df[column].astype("string").fillna("__missing__")
        else:
            other_keys = pd.Series("__missing__", index=other_df.index)
        if task_type == "regression":
            stats = pd.DataFrame({"key": keys, "y": pd.to_numeric(train_df[target], errors="coerce")})
            grouped = stats.groupby("key")["y"].agg(["mean", "count"])
            global_mean = float(stats["y"].mean())
            encoded = (grouped["mean"] * grouped["count"] + global_mean * smooth) / (grouped["count"] + smooth)
            name = f"{column}__te"
            train_df[name] = keys.map(encoded).fillna(global_mean).astype(float)
            other_df[name] = other_keys.map(encoded).fillna(global_mean).astype(float)
        else:
            labels = train_df[target].astype(str)
            global_share = labels.value_counts(normalize=True)
            for label, share in list(global_share.items())[:15]:
                stats = pd.DataFrame({"key": keys, "hit": (labels == label).astype(float)})
                grouped = stats.groupby("key")["hit"].agg(["mean", "count"])
                encoded = (grouped["mean"] * grouped["count"] + share * smooth) / (grouped["count"] + smooth)
                name = f"{column}__te_{label}"
                train_df[name] = keys.map(encoded).fillna(share).astype(float)
                other_df[name] = other_keys.map(encoded).fillna(share).astype(float)
    return train_df, other_df


def _scale(train_df, other_df, options, target, task_type, id_column):
    for column in list(train_df.columns):
        if column in {target, id_column} or _is_cat(train_df[column]):
            continue
        numeric = pd.to_numeric(train_df[column], errors="coerce")
        center = float(numeric.median()) if numeric.notna().any() else 0.0
        filled = numeric.fillna(center)
        std = float(filled.std()) or 1.0
        train_df[column] = (filled - center) / std
        other_numeric = pd.to_numeric(other_df[column], errors="coerce") if column in other_df.columns else center
        other_df[column] = (pd.Series(other_numeric).fillna(center) - center) / std
    return train_df, other_df


def _ratio(train_df, other_df, options, target, task_type, id_column):
    numerator = options.get("numerator")
    denominator = options.get("denominator")
    if numerator not in train_df.columns or denominator not in train_df.columns:
        raise ValueError("для ratio нужны существующие numerator и denominator")
    if target in {numerator, denominator}:
        raise ValueError("ratio не может использовать цель")
    name = f"{numerator}__div__{denominator}"
    for frame in (train_df, other_df):
        left = pd.to_numeric(frame[numerator], errors="coerce")
        right = pd.to_numeric(frame[denominator], errors="coerce").replace(0, np.nan)
        frame[name] = left / right
    return train_df, other_df


def _groupby(train_df, other_df, options, target, task_type, id_column):
    group = options.get("group")
    columns = options.get("columns") or []
    agg = options.get("agg") or "mean"
    if group not in train_df.columns:
        raise ValueError(f"нет групповой колонки {group}")
    if target in columns or group == target:
        raise ValueError("groupby не агрегирует цель")
    if agg not in {"mean", "median", "min", "max", "std"}:
        raise ValueError("agg должен быть mean, median, min, max или std")
    for column in columns:
        if column not in train_df.columns:
            raise ValueError(f"нет колонки {column}")
        stats = pd.to_numeric(train_df[column], errors="coerce").groupby(train_df[group]).agg(agg)
        global_value = float(pd.to_numeric(train_df[column], errors="coerce").mean())
        name = f"{group}__{agg}__{column}"
        train_df[name] = train_df[group].map(stats).fillna(global_value).astype(float)
        other_df[name] = other_df[group].map(stats).fillna(global_value).astype(float)
    return train_df, other_df


FEATURES = {
    "missing_indicator": _missing_indicator,
    "log1p_skew": _log1p_skew,
    "sincos_degrees": _sincos,
    "frequency": _frequency,
    "target_encoding": _target_encoding,
    "scale": _scale,
    "ratio": _ratio,
    "groupby": _groupby,
}


def apply_code(train_df, other_df, code: str, target: str, id_column: str):
    tree = ast.parse(code)
    _check_tree(tree)
    namespace = {"pd": pd, "np": np, "__builtins__": SAFE_BUILTINS}
    exec(compile(tree, "<transform>", "exec"), namespace, namespace)
    transform = namespace.get("transform")
    if not callable(transform):
        raise ValueError("в code нет функции transform(train, test)")
    other_in = other_df.drop(columns=[target], errors="ignore").copy()
    out_train, out_other = transform(train_df.copy(), other_in)
    if not isinstance(out_train, pd.DataFrame) or not isinstance(out_other, pd.DataFrame):
        raise ValueError("transform должен вернуть два DataFrame")
    if len(out_train) != len(train_df) or len(out_other) != len(other_df):
        raise ValueError("transform изменил число строк")
    out_train = out_train.reset_index(drop=True)
    out_other = out_other.reset_index(drop=True)
    if target not in out_train.columns:
        out_train[target] = train_df[target].to_numpy()
    if target in other_df.columns and target not in out_other.columns:
        out_other[target] = other_df[target].to_numpy()
    features = [column for column in out_train.columns if column != target]
    for column in features:
        if column not in out_other.columns:
            out_other[column] = np.nan
    train_columns = features + [target]
    other_columns = features + ([target] if target in out_other.columns else [])
    return out_train.reindex(columns=train_columns).copy(), out_other.reindex(columns=other_columns).copy()


def _check_tree(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("в transform нельзя импортировать модули: pandas и numpy уже доступны как pd и np")
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise ValueError(f"нельзя использовать {node.id}")
        if isinstance(node, ast.Attribute) and (node.attr in BANNED_ATTRS or node.attr.startswith("__")):
            raise ValueError(f"нельзя вызывать .{node.attr}")
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            if name in BANNED_CALLS or name in BANNED_ATTRS:
                raise ValueError(f"нельзя вызывать {name}")


def analyze(spec: dict, champion: dict | None, train: pd.DataFrame, task) -> dict:
    if not champion or not champion.get("val_path"):
        return {"ok": False, "error": "чемпиона ещё нет"}
    slice_name = spec.get("slice")
    if slice_name not in {"by_category", "by_quantile", "per_class", "missing_vs_error"}:
        return {"ok": False, "error": "slice должен быть by_category, by_quantile, per_class или missing_vs_error"}
    if slice_name == "per_class" and task.task_type != "classification":
        return {"ok": False, "error": "per_class только для классификации"}
    if slice_name == "by_quantile" and task.task_type != "regression":
        return {"ok": False, "error": "by_quantile для классификации не считается, возьми per_class"}
    val = pd.read_csv(champion["val_path"])
    column = spec.get("column")
    if slice_name == "per_class":
        rows = []
        truth = val["y_true"].astype(str)
        pred = val["y_pred"].astype(str)
        for label in sorted(truth.unique()):
            support = int((truth == label).sum())
            tp = int(((truth == label) & (pred == label)).sum())
            fp = int(((truth != label) & (pred == label)).sum())
            fn = support - tp
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows.append({"label": label, "support": support, "f1": round(f1, 4), "recall": round(recall, 4)})
        rows.sort(key=lambda item: item["f1"])
        return {"ok": True, "summary": rows[:20]}
    if slice_name == "by_quantile":
        truth = pd.to_numeric(val["y_true"], errors="coerce")
        pred = pd.to_numeric(val["y_pred"], errors="coerce")
        try:
            bins = pd.qcut(pred, q=min(5, max(2, int(pred.nunique(dropna=True)))), duplicates="drop")
        except ValueError as error:
            return {"ok": False, "error": f"квантили не посчитались: {error}"}
        table = []
        for bucket, part in pred.groupby(bins, observed=True):
            idx = part.index
            residual = float((truth.loc[idx] - pred.loc[idx]).mean())
            table.append({"bin": str(bucket), "n": int(len(idx)), "mean_residual": round(residual, 4)})
        return {"ok": True, "summary": table}
    if slice_name == "missing_vs_error":
        if column is None:
            missing = train.drop(columns=[task.id_column, *task.targets], errors="ignore").isna().mean()
            missing = missing[missing > 0]
            if missing.empty:
                return {"ok": True, "summary": "пропусков в признаках нет"}
            column = str(missing.sort_values(ascending=False).index[0])
        if column not in train.columns:
            return {"ok": False, "error": f"нет колонки {column}"}
        aligned = train.loc[val["row_index"], column]
        mask = aligned.isna().to_numpy()
        return {"ok": True, "summary": {
            "column": column,
            "missing": _error_brief(val, mask, task),
            "present": _error_brief(val, ~mask, task),
        }}
    if column is None:
        cats = [name for name in train.columns if name not in {task.id_column, *task.targets} and _is_cat(train[name])]
        if not cats:
            return {"ok": False, "error": "для by_category нужна колонка или категориальный признак"}
        column = max(cats, key=lambda name: train[name].nunique(dropna=True))
    if column not in train.columns:
        return {"ok": False, "error": f"нет колонки {column}"}
    labels = train.loc[val["row_index"], column].astype("string").fillna("__missing__").to_numpy()
    rows = []
    for label in pd.unique(labels):
        mask = labels == label
        if int(mask.sum()) < 5:
            continue
        rows.append({"value": str(label), "n": int(mask.sum()), "score": _error_brief(val, mask, task)})
    rows.sort(key=lambda item: item["score"] if task.higher_is_better else -item["score"])
    return {"ok": True, "summary": {"column": column, "worst": rows[:12]}}


def _error_brief(val: pd.DataFrame, mask: np.ndarray, task) -> float:
    part = val.loc[mask]
    if part.empty:
        return 0.0
    if task.task_type == "classification":
        return float((part["y_true"].astype(str) == part["y_pred"].astype(str)).mean())
    truth = pd.to_numeric(part["y_true"], errors="coerce")
    pred = pd.to_numeric(part["y_pred"], errors="coerce")
    return float(np.sqrt(np.mean((truth - pred) ** 2)))
