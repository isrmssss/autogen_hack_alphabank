"""Цикл задачи. Модель выбирает действие, этот модуль его исполняет и хранит файл."""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import time
import traceback
from pathlib import Path

import pandas as pd

from knowledge import TRIGGER_SIGNALS, KnowledgeBase, detect_signals
from llm_client import LLM, LLMError
from profile_data import build_profile
from research import (
    SYSTEM_PROMPT,
    feature_names,
    fingerprint,
    mechanism_key,
    parse_decision,
    repair_message,
    user_message,
)
from task_io import constant_frame, load_task, verify_submission, write_submission
from toolbox import SPEC_HELP, TOOLBOX, analyze, child_main, validate_spec


IDLE_LIMIT = 6
CODE_LIMIT = 3
PARALLEL_PAIRS = {
    ("run", "run"),
    ("run", "analyze"),
    ("analyze", "run"),
    ("analyze", "analyze"),
}
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def run(task_path: str, stop_flag) -> None:
    started = time.monotonic()
    budget = float(os.environ.get("AGENT_TIME_BUDGET_SEC") or 10800)
    deadline = started + budget
    task = None
    run_dir = None
    try:
        task = load_task(Path(task_path))
        run_dir = _run_dir(task)
        train = pd.read_csv(task.train_path)
        test = pd.read_csv(task.test_path)
        profile = build_profile(train, test, task)
        write_submission(task, constant_frame(task, train, test))
        problems = verify_submission(task)
        if problems:
            raise RuntimeError("; ".join(problems))
        _remember(task, run_dir)
        _event(run_dir, {
            "type": "start",
            "task": str(task.path),
            "metric": task.metric,
            "higher_is_better": task.higher_is_better,
            "id": task.id_column,
            "targets": task.targets,
            "task_type": task.task_type,
            "notes": task.notes,
            "profile": profile,
        })
        print("constant submission", flush=True)
        if any(not os.environ.get(name) for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL")):
            _event(run_dir, {"type": "stop", "reason": "нет переменных OPENAI"})
            print("нет переменных OPENAI, оставлен константный файл", flush=True)
            return
        _loop(task, train, test, profile, run_dir, started, deadline, stop_flag)
    except Exception as error:
        _event(run_dir, {"type": "crash", "error": f"{type(error).__name__}: {error}"[:500]})
        print(f"сбой: {type(error).__name__}", flush=True)
        traceback.print_exc()
    finally:
        if task is not None:
            _finish(task, run_dir)


def _loop(task, train, test, profile, run_dir, started, deadline, stop_flag) -> None:
    kb = KnowledgeBase()
    llm = LLM()
    journal: list[dict] = []
    fingerprints: set[str] = set()
    save_seen: set[str] = set()
    shown: set[tuple] = set()
    last_triggers: set[str] = set()
    champion = None
    pending_question = None
    repeat_hurt = False
    last_error = None
    idle = 0
    step = 0
    while time.monotonic() < deadline - 5 and not stop_flag():
        step += 1
        signals = detect_signals(profile, journal, champion)
        opened = kb.decide(signals, shown, last_triggers, pending_question, repeat_hurt)
        pending_question = None
        repeat_hurt = False
        if opened and opened.get("advance_triggers"):
            last_triggers = set(signals & TRIGGER_SIGNALS)
        view = opened.get("view") if opened else None
        remaining = max(0.0, deadline - time.monotonic())
        state = _state(task, profile, signals, journal, champion, started, remaining, view, last_error)
        try:
            decision = parse_decision(llm.complete(SYSTEM_PROMPT, user_message(state)))
        except (LLMError, ValueError) as error:
            last_error = str(error)[:500]
            idle += 1
            _event(run_dir, {"type": "model_error", "step": step, "error": last_error, "idle": idle})
            print(f"step {step} model_error", flush=True)
            if idle >= IDLE_LIMIT:
                break
            _sleep(min(30, 2 ** idle), deadline, stop_flag)
            continue
        action = decision["action"]
        second = decision.get("parallel")
        label = action if not second else f"{action}+{second.get('action')}"
        print(f"step {step} {label}", flush=True)
        if action == "stop" or stop_flag():
            _event(run_dir, {"type": "decision", "step": step, "action": "stop", "why": decision.get("why")})
            break
        if action == "consult_knowledge":
            pending_question = str((decision.get("spec") or {}).get("question") or "")
            idle = 0
            last_error = "parallel с consult_knowledge не запускается" if second else None
            journal.append({"step": step, "action": "consult_knowledge", "ok": True, "question": pending_question})
            _event(run_dir, {
                "type": "consult",
                "step": step,
                "question": pending_question,
                "why": decision.get("why"),
                "knowledge": None if not view else view.get("reason"),
            })
            continue
        jobs = _jobs(decision, journal, fingerprints, save_seen, champion, remaining, step)
        if not jobs:
            break
        _execute(jobs, task, train, test, champion, llm, run_dir, deadline, stop_flag)
        worked = False
        for job in jobs:
            if job["status"] == "reject":
                last_error = job["error"]
                journal.append({
                    "step": step,
                    "action": job["action"],
                    "ok": False,
                    "error": job["error"],
                    "hurt": False,
                    "adopted": False,
                })
                continue
            worked = True
            outcome = _consume(job, task, test, champion, fingerprints, save_seen, run_dir)
            job["entry"] = outcome["entry"]
            champion = outcome["champion"]
            last_error = outcome.get("error")
            if outcome.get("repeat"):
                repeat_hurt = True
            journal.append(outcome["entry"])
        idle = 0 if worked else idle + 1
        _event(run_dir, {
            "type": "step",
            "step": step,
            "why": decision.get("why"),
            "knowledge": None if not view else view.get("reason"),
            "knowledge_card": None if not view or not view.get("card") else view["card"].get("id"),
            "signals": sorted(signals),
            "remaining_sec": round(remaining, 1),
            "jobs": [_job_log(job) for job in jobs],
            "champion": None if champion is None else champion.get("val_metric"),
            "idle": idle,
        })
        if idle >= IDLE_LIMIT:
            break
    _event(run_dir, {"type": "loop_end", "steps": step, "idle": idle})


def _jobs(decision, journal, fingerprints, save_seen, champion, remaining, step) -> list[dict]:
    pairs = [(decision["action"], decision.get("spec") or {}, decision.get("why") or "", "a")]
    second = decision.get("parallel")
    if second:
        if (decision["action"], second.get("action")) in PARALLEL_PAIRS:
            pairs.append((second["action"], second.get("spec") or {}, second.get("why") or "", "b"))
        else:
            pairs.append(("reject", {}, "параллельное действие не независимое", "b"))
    code_slots = CODE_LIMIT - sum(1 for item in journal if item.get("had_code"))
    jobs = []
    for action, spec, why, suffix in pairs:
        if action == "reject":
            jobs.append({"status": "reject", "action": "parallel", "error": why, "step": step})
            continue
        if action == "save":
            jobs.append(_prepare_save(champion, remaining, save_seen, why, step))
            continue
        if action == "analyze":
            jobs.append(_prepare_analyze(spec, champion, why, step))
            continue
        if action != "run":
            jobs.append({"status": "reject", "action": action, "error": f"действие {action} не исполняется", "step": step})
            continue
        spec = _normalize_spec(spec)
        prepared = _prepare_run(spec, journal, fingerprints, remaining, code_slots, why, step, suffix)
        if prepared["status"] == "run" and spec.get("code"):
            code_slots -= 1
        jobs.append(prepared)
    return jobs


def _prepare_run(spec, journal, fingerprints, remaining, code_slots, why, step, suffix) -> dict:
    timeout = _timeout_value(spec)
    if timeout is not None and timeout > remaining:
        return {"status": "reject", "action": "run", "error": "timeout_sec больше оставшегося времени", "step": step, "why": why}
    if spec.get("code") and code_slots <= 0:
        return {"status": "reject", "action": "run", "error": "код transform уже пробовался три раза", "step": step, "why": why}
    problems = validate_spec(spec)
    if problems:
        return {"status": "reject", "action": "run", "error": "; ".join(problems), "step": step, "why": why}
    mark = fingerprint(spec)
    if mark in fingerprints:
        return {"status": "reject", "action": "run", "error": "такая спецификация уже есть в журнале", "step": step, "why": why}
    return {
        "status": "run",
        "action": "run",
        "spec": spec,
        "fingerprint": mark,
        "repeat": _seen_hurt(journal, spec),
        "why": why,
        "step": step,
        "suffix": suffix,
        "final_only": False,
    }


def _prepare_analyze(spec, champion, why, step) -> dict:
    if not champion:
        return {"status": "reject", "action": "analyze", "error": "чемпиона ещё нет", "step": step, "why": why}
    if spec.get("slice") not in {"by_category", "by_quantile", "per_class", "missing_vs_error"}:
        return {"status": "reject", "action": "analyze", "error": "slice должен быть by_category, by_quantile, per_class или missing_vs_error", "step": step, "why": why}
    return {"status": "analyze", "action": "analyze", "spec": spec, "why": why, "step": step}


def _prepare_save(champion, remaining, save_seen, why, step) -> dict:
    if not champion:
        return {"status": "reject", "action": "save", "error": "чемпиона ещё нет", "step": step, "why": why}
    if remaining < 30:
        return {"status": "reject", "action": "save", "error": "на save осталось меньше 30 секунд", "step": step, "why": why}
    spec = json.loads(json.dumps(champion["spec"]))
    spec["timeout_sec"] = max(5, int(remaining - 20))
    spec = _normalize_spec(spec)
    mark = fingerprint(spec)
    if mark in save_seen:
        return {"status": "reject", "action": "save", "error": "save этой спецификации уже выполнялся", "step": step, "why": why}
    return {
        "status": "save",
        "action": "save",
        "spec": spec,
        "why": why,
        "step": step,
        "suffix": "save",
        "final_only": True,
        "fingerprint": mark,
    }


def _execute(jobs, task, train, test, champion, llm, run_dir, deadline, stop_flag) -> None:
    runnable = [job for job in jobs if job["status"] in {"run", "save"}]
    share = max(1, _cpus() // max(1, len(runnable)))
    for job in runnable:
        if stop_flag() or time.monotonic() > deadline - 5:
            job["result"] = {"ok": False, "error": "процесс останавливается"}
            continue
        job["launch"] = _launch(job, task, run_dir, share)
    for job in jobs:
        if job["status"] != "analyze":
            continue
        job["result"] = analyze(job["spec"], champion, train, task)
    for job in runnable:
        if job.get("result"):
            continue
        job["result"] = _join(job["launch"], deadline, stop_flag)
        if job["status"] == "run" and job["spec"].get("code") and not job["result"].get("ok"):
            if "таймаут" not in (job["result"].get("error") or ""):
                repaired, result = _repair(job, task, run_dir, share, llm, deadline, stop_flag)
                job["spec"] = repaired
                job["result"] = result
                job["repaired"] = repaired.get("code") != job.get("original_code")


def _launch(job, task, run_dir, n_jobs) -> dict:
    folder = run_dir / "experiments" / f"step{job['step']}_{job['suffix']}"
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_path": str(task.train_path),
        "test_path": str(task.test_path),
        "spec": job["spec"],
        "targets": list(task.targets),
        "id_column": task.id_column,
        "task_type": task.task_type,
        "metric": task.metric,
        "n_jobs": n_jobs,
        "result_path": str(folder / "result.json"),
        "test_pred_path": str(folder / "pred.csv"),
        "val_path": str(folder / "val.csv"),
        "final_only": bool(job.get("final_only")),
    }
    path = folder / "payload.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    job["original_code"] = job["spec"].get("code") or ""
    return {"proc": _start(str(path), n_jobs), "payload": payload, "timeout": float(job["spec"].get("timeout_sec") or 60)}


def _start(payload_path: str, n_jobs: int):
    old = {key: os.environ.get(key) for key in THREAD_ENV}
    try:
        for key in THREAD_ENV:
            os.environ[key] = str(max(1, int(n_jobs)))
        process = multiprocessing.get_context("spawn").Process(target=child_main, args=(payload_path,))
        process.start()
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return process


def _join(launch, deadline, stop_flag) -> dict:
    process = launch["proc"]
    limit = min(time.monotonic() + float(launch["timeout"]) + 60, deadline)
    while process.is_alive() and time.monotonic() < limit and not stop_flag():
        process.join(2)
    if process.is_alive():
        process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join(5)
        return {"ok": False, "error": "таймаут опыта"}
    path = Path(launch["payload"]["result_path"])
    if not path.is_file():
        return {"ok": False, "error": f"процесс завершился без результата, код {process.exitcode}"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"ok": False, "error": "результат опыта не читается"}


def _repair(job, task, run_dir, n_jobs, llm, deadline, stop_flag):
    error = job["result"].get("error") or ""
    try:
        text = llm.complete(
            "Верни только исправленную функцию transform. Без пояснений.",
            repair_message(job["spec"].get("code") or "", error),
        )
    except (LLMError, ValueError) as exc:
        return job["spec"], {"ok": False, "error": f"ремонт не выполнен: {exc}"[:500]}
    new_code = _function_text(text)
    if "def transform" not in new_code:
        return job["spec"], {"ok": False, "error": "ремонт не вернул функцию transform"}
    repaired = dict(job["spec"])
    repaired["code"] = new_code
    follow = dict(job)
    follow["spec"] = repaired
    follow["suffix"] = f"{job['suffix']}_repair"
    follow["final_only"] = False
    launch = _launch(follow, task, run_dir, n_jobs)
    return repaired, _join(launch, deadline, stop_flag)


def _consume(job, task, test, champion, fingerprints, save_seen, run_dir) -> dict:
    result = job.get("result") or {"ok": False, "error": "нет результата"}
    spec = job.get("spec") or {}
    if job["status"] == "analyze":
        return {
            "champion": champion,
            "error": None if result.get("ok") else result.get("error"),
            "entry": {
                "step": job["step"],
                "action": "analyze",
                "ok": bool(result.get("ok")),
                "error": None if result.get("ok") else str(result.get("error") or "")[:500],
                "summary": result.get("summary"),
                "hurt": False,
                "adopted": False,
                "why": job.get("why"),
            },
        }
    if job.get("fingerprint"):
        fingerprints.add(job["fingerprint"])
    if job.get("repaired"):
        fingerprints.add(fingerprint(spec))
    if job["status"] == "save":
        if job.get("fingerprint"):
            save_seen.add(job["fingerprint"])
        error = None
        if result.get("ok"):
            error = _publish(task, test, result.get("test_pred_path"), run_dir)
            if error:
                result = {"ok": False, "error": "; ".join(error)}
        return {
            "champion": champion,
            "error": None if result.get("ok") else result.get("error"),
            "entry": {
                "step": job["step"],
                "action": "save",
                "ok": bool(result.get("ok")),
                "error": None if result.get("ok") else str(result.get("error") or "")[:500],
                "hurt": not result.get("ok"),
                "adopted": bool(result.get("ok")),
                "why": job.get("why"),
            },
        }
    adopted = False
    outcome = "error"
    publish_error = None
    if result.get("ok") and result.get("val_metric") is not None:
        outcome = _judge(result.get("val_metric"), result.get("val_std"), champion, task.higher_is_better)
        if outcome == "adopted":
            publish_error = _publish(task, test, result.get("test_pred_path"), run_dir)
            if publish_error:
                outcome = "error"
                result = {"ok": False, "error": "; ".join(publish_error)}
            else:
                adopted = True
                champion = {
                    "spec": spec,
                    "val_metric": result.get("val_metric"),
                    "val_std": result.get("val_std"),
                    "val_path": result.get("val_path"),
                    "estimator": spec.get("estimator"),
                    "features": feature_names(spec.get("features")),
                    "split_name": _split_name(spec),
                }
    hurt = (not result.get("ok")) or outcome == "hurt"
    return {
        "champion": champion,
        "repeat": bool(job.get("repeat")),
        "error": None if result.get("ok") else result.get("error"),
        "entry": {
            "step": job["step"],
            "action": "run",
            "ok": bool(result.get("ok")),
            "estimator": spec.get("estimator"),
            "features": feature_names(spec.get("features")),
            "split_name": _split_name(spec),
            "train_metric": result.get("train_metric"),
            "val_metric": result.get("val_metric"),
            "val_std": result.get("val_std"),
            "adopted": adopted,
            "error": None if result.get("ok") else str(result.get("error") or "")[:500],
            "hurt": hurt,
            "fingerprint": job.get("fingerprint"),
            "mechanism": list(mechanism_key(spec)),
            "why": job.get("why"),
            "had_code": bool(spec.get("code")),
            "outcome": outcome,
        },
    }


def _publish(task, test, pred_path, run_dir) -> list[str]:
    if not pred_path or not Path(pred_path).is_file():
        return ["файл предсказаний не создан"]
    preds = pd.read_csv(pred_path)
    if len(preds) != len(test):
        return [f"длина предсказаний {len(preds)} вместо {len(test)}"]
    frame = pd.DataFrame({task.id_column: test[task.id_column].to_numpy()})
    for target in task.targets:
        if target not in preds.columns:
            return [f"в предсказаниях нет колонки {target}"]
        frame[target] = preds[target].to_numpy()
    write_submission(task, frame[task.submission_columns])
    errors = verify_submission(task)
    if errors:
        _restore(task, run_dir)
        return errors
    _remember(task, run_dir)
    return []


def _judge(val, std, champion, higher: bool) -> str:
    if champion is None or champion.get("val_metric") is None:
        return "adopted"
    old = float(champion["val_metric"])
    new = float(val)
    delta = new - old if higher else old - new
    if delta < 0:
        return "hurt"
    if delta == 0:
        return "no_change"
    if std is not None and float(std) > 0 and delta <= 0.5 * float(std):
        return "no_change"
    return "adopted"


def _state(task, profile, signals, journal, champion, started, remaining, view, last_error) -> dict:
    text = task.text if len(task.text) <= 12000 else task.text[:12000]
    return {
        "task": {
            "summary": text,
            "metric_text": task.metric_snippet,
            "id": task.id_column,
            "targets": task.targets,
            "task_type": task.task_type,
            "metric": task.metric,
            "higher_is_better": task.higher_is_better,
            "notes": task.notes,
        },
        "profile": profile,
        "signals": sorted(signals),
        "journal": _journal_view(journal),
        "champion": _champion_view(champion),
        "elapsed_sec": round(time.monotonic() - started, 1),
        "remaining_sec": round(remaining, 1),
        "knowledge": view,
        "last_tool_error": last_error,
        "toolbox": TOOLBOX,
        "contracts": SPEC_HELP,
    }


def _journal_view(journal: list[dict]) -> list[dict]:
    keys = (
        "step", "action", "ok", "estimator", "features", "split_name", "train_metric",
        "val_metric", "val_std", "adopted", "error", "why", "summary", "outcome", "question",
    )
    view = []
    for item in journal[-15:]:
        compact = {key: item.get(key) for key in keys if item.get(key) is not None}
        if isinstance(compact.get("why"), str):
            compact["why"] = compact["why"][:400]
        if isinstance(compact.get("error"), str):
            compact["error"] = compact["error"][:500]
        view.append(compact)
    return view


def _champion_view(champion) -> dict | None:
    if not champion:
        return None
    spec = dict(champion["spec"])
    if spec.get("code"):
        spec = dict(spec)
        spec["code"] = spec["code"][:400]
    return {
        "val_metric": champion.get("val_metric"),
        "val_std": champion.get("val_std"),
        "estimator": champion.get("estimator"),
        "features": champion.get("features"),
        "split_name": champion.get("split_name"),
        "spec": spec,
    }


def _normalize_spec(spec: dict) -> dict:
    spec = dict(spec or {})
    if isinstance(spec.get("split"), str):
        spec["split"] = {"name": spec["split"]}
    spec["params"] = spec.get("params") or {}
    spec["features"] = spec.get("features") or []
    spec["code"] = spec.get("code") or ""
    return spec


def _timeout_value(spec) -> float | None:
    try:
        return float(spec.get("timeout_sec"))
    except (TypeError, ValueError):
        return None


def _seen_hurt(journal, spec) -> bool:
    key = list(mechanism_key(spec))
    return any(item.get("hurt") and item.get("mechanism") == key for item in journal)


def _split_name(spec) -> str | None:
    split = spec.get("split")
    if isinstance(split, dict):
        return split.get("name")
    if isinstance(split, str):
        return split
    return None


def _function_text(text: str) -> str:
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = [line for line in raw.splitlines() if not line.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    start = raw.find("def transform")
    if start >= 0:
        raw = raw[start:]
    return raw.strip()


def _job_log(job: dict) -> dict:
    result = job.get("result") or {}
    return {
        "action": job.get("action"),
        "status": job.get("status"),
        "ok": result.get("ok"),
        "error": None if not result.get("error") else str(result.get("error"))[:500],
        "val_metric": result.get("val_metric"),
        "train_metric": result.get("train_metric"),
        "adopted": (job.get("entry") or {}).get("adopted"),
        "outcome": (job.get("entry") or {}).get("outcome"),
        "estimator": (job.get("spec") or {}).get("estimator"),
    }


def _cpus() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def _sleep(seconds: float, deadline: float, stop_flag) -> None:
    end = min(time.monotonic() + seconds, deadline)
    while time.monotonic() < end and not stop_flag():
        time.sleep(0.5)


def _run_dir(task) -> Path:
    raw = os.environ.get("AGENT_RUN_DIR")
    path = Path(raw) if raw else task.directory / "agent_run"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _event(run_dir, record: dict) -> None:
    if run_dir is None:
        return
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _remember(task, run_dir: Path) -> None:
    shutil.copy2(task.directory / "submission.csv", run_dir / "submission.csv")


def _restore(task, run_dir: Path) -> None:
    backup = run_dir / "submission.csv"
    if backup.is_file():
        write_submission(task, pd.read_csv(backup))


def _finish(task, run_dir) -> None:
    errors = verify_submission(task)
    if errors and run_dir is not None:
        _restore(task, run_dir)
        errors = verify_submission(task)
    if errors:
        _event(run_dir, {"type": "verify", "ok": False, "errors": errors})
        print("submission check failed", flush=True)
        return
    _event(run_dir, {"type": "verify", "ok": True})
    print("submission ok", flush=True)
