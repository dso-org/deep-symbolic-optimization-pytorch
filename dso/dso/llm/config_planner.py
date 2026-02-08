"""LLM-assisted planner for generating DSO configs."""

import json
import os
import re
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple
from urllib import error, request

import numpy as np
import pandas as pd

from dso.config import load_config
from dso.utils import safe_merge_dicts


_INVALID = object()


def _resolve_regression_dataset(dataset: str) -> Optional[str]:
    if dataset is None:
        return None
    if dataset.endswith(".csv") and os.path.exists(dataset):
        return dataset

    candidate = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "task",
        "regression",
        "data",
        f"{dataset}.csv",
    )
    return candidate if os.path.exists(candidate) else None


def _profile_regression_dataset(dataset: Optional[str]) -> Dict[str, Any]:
    profile: Dict[str, Any] = {"dataset": dataset, "source": "unknown"}

    path = _resolve_regression_dataset(dataset) if dataset else None
    if path is None:
        profile["available"] = False
        return profile

    df = pd.read_csv(path, header=None)
    if df.shape[1] < 2:
        profile["available"] = False
        profile["error"] = "Dataset must have at least one feature and one target column."
        return profile

    values = df.values
    X = values[:, :-1].astype(np.float64)
    y = values[:, -1].astype(np.float64)

    feature_std = np.std(X, axis=0)
    target_std = float(np.std(y))
    profile |= {
        "available": True,
        "source": path,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "nan_fraction": float(np.isnan(values).mean()),
        "x_min": float(np.min(X)),
        "x_max": float(np.max(X)),
        "x_mean_std": float(np.mean(feature_std)),
        "x_max_std": float(np.max(feature_std)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "y_std": target_std,
        "has_negative_inputs": np.min(X) < 0,
        "likely_periodic_signal": _estimate_periodicity(y) > 0.2,
    }
    return profile


def _estimate_periodicity(signal: np.ndarray) -> float:
    if signal.size < 16:
        return 0.0
    centered = signal - np.mean(signal)
    spectrum = np.abs(np.fft.rfft(centered))
    if spectrum.size <= 2:
        return 0.0
    energy = float(np.sum(spectrum[1:]))
    return 0.0 if energy <= 1e-12 else float(np.max(spectrum[1:]) / energy)


def _deterministic_patch(
    task_type: str, profile: Dict[str, Any], user_goal: str
) -> Dict[str, Any]:
    goal = (user_goal or "").lower()

    if task_type == "control":
        return {
            "training": {
                "n_samples": 20000,
                "batch_size": 100,
                "n_cores_batch": 1,
            },
            "policy": {"initializer": "var_scale"},
        }

    n_rows = int(profile.get("n_rows", 1000))
    likely_periodic = bool(profile.get("likely_periodic_signal", False))
    fast_mode = any(word in goal for word in ["fast", "quick", "debug", "small"])
    robust_mode = any(word in goal for word in ["robust", "noisy", "stable"])

    function_set = ["add", "sub", "mul", "div", "poly"]
    if likely_periodic or "periodic" in goal:
        function_set.extend(["sin", "cos"])
    if not fast_mode:
        function_set.extend(["exp", "log"])
    if "constant" in goal or "intercept" in goal:
        function_set.append("const")

    n_samples = 20000 if fast_mode else 100000
    if n_rows > 5000 and not fast_mode:
        n_samples = 200000
    batch_size = 200 if fast_mode else 1000

    patch: Dict[str, Any] = {
        "task": {"protected": robust_mode, "function_set": function_set},
        "training": {
            "n_samples": n_samples,
            "batch_size": batch_size,
            "epsilon": 0.05,
            "baseline": "R_e",
        },
        "policy": {
            "initializer": "var_scale",
            "num_units": 32 if fast_mode else 64,
        },
        "prior": {
            "length": {"on": True, "min_": 4, "max_": 25 if fast_mode else 35}
        },
    }
    return patch


def _build_messages(
    base_config: Dict[str, Any], profile: Dict[str, Any], user_goal: str
) -> list:
    system_text = (
        "You are a configuration planner for Deep Symbolic Optimization (DSO). "
        "Return strict JSON only. Keep changes conservative and runnable."
    )
    user_text = (
        "Create a JSON object with keys: 'patch' and 'rationale'. "
        "'patch' must be a partial config update compatible with this base config. "
        "Only modify fields that materially improve the run for the stated goal.\n\n"
        f"Goal:\n{user_goal}\n\n"
        f"Dataset/profile:\n{json.dumps(profile, indent=2)}\n\n"
        f"Base config:\n{json.dumps(base_config, indent=2)}\n"
    )
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


def _call_llm_json(
    *,
    messages: list,
    model: str,
    api_key: str,
    base_url: str,
    timeout_s: int = 60,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = request.Request(url=url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

    data = json.loads(raw)
    content = data["choices"][0]["message"]["content"]
    return _extract_json_object(content)


def _extract_json_object(text: str) -> Dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        if match := re.search(
            r"```(?:json)?\s*(\{.*\})\s*```", candidate, flags=re.S
        ):
            candidate = match[1]
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object.")
    return parsed


def _sanitize_scalar(value: Any, template: Any) -> Any:
    if template is None:
        return value
    if isinstance(template, bool):
        return value if isinstance(value, bool) else _INVALID
    if isinstance(template, int):
        return value if isinstance(value, int) else _INVALID
    if isinstance(template, float):
        return float(value) if isinstance(value, (int, float)) else _INVALID
    if isinstance(template, str):
        return value if isinstance(value, str) else _INVALID
    if isinstance(template, list):
        return value if isinstance(value, list) else _INVALID
    return _INVALID


def _sanitize_patch_value(value: Any, template: Any) -> Any:
    if not isinstance(template, dict):
        return _sanitize_scalar(value, template)
    if not isinstance(value, dict):
        return _INVALID
    sanitized: Dict[str, Any] = {}
    for key, sub_value in value.items():
        if key not in template:
            continue
        clean = _sanitize_patch_value(sub_value, template[key])
        if clean is not _INVALID:
            sanitized[key] = clean
    return sanitized


def _sanitize_patch(patch: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, Any]:
    clean = _sanitize_patch_value(patch, template)
    return {} if clean is _INVALID or not isinstance(clean, dict) else clean


def plan_config(
    *,
    task_type: str,
    dataset: Optional[str] = None,
    env_name: Optional[str] = None,
    user_goal: str = "",
    base_config_path: Optional[str] = None,
    dry_run: bool = False,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: str = "https://api.openai.com/v1",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if task_type not in {"regression", "control"}:
        raise ValueError("task_type must be 'regression' or 'control'.")

    if task_type == "regression" and not dataset:
        raise ValueError("dataset is required for regression planning.")
    if task_type == "control" and not env_name:
        raise ValueError("env_name is required for control planning.")

    if base_config_path:
        base_config = load_config(base_config_path)
    else:
        base_config = load_config({"task": {"task_type": task_type}})

    base_config = deepcopy(base_config)
    base_config["task"]["task_type"] = task_type
    if task_type == "regression":
        base_config["task"]["dataset"] = dataset
    else:
        base_config["task"]["env"] = env_name

    profile = (
        _profile_regression_dataset(dataset)
        if task_type == "regression"
        else {"task_type": "control", "env": env_name}
    )

    deterministic_patch = _deterministic_patch(task_type, profile, user_goal)
    patch_candidate = deterministic_patch
    rationale = "Deterministic dry-run planner used."
    planner_mode = "dry_run"

    if not dry_run:
        if not api_key:
            raise ValueError(
                "api_key is required when dry_run is False. "
                "Set OPENAI_API_KEY or pass it explicitly."
            )
        try:
            response = _call_llm_json(
                messages=_build_messages(base_config, profile, user_goal),
                model=model,
                api_key=api_key,
                base_url=base_url,
            )
            patch_candidate = response.get("patch", response)
            rationale = response.get(
                "rationale", "LLM-generated config patch without rationale."
            )
            planner_mode = "llm"
        except Exception as exc:
            rationale = (
                "LLM call failed, fell back to deterministic planner. "
                f"Failure: {exc}"
            )
            patch_candidate = deterministic_patch
            planner_mode = "fallback"

    sanitized_patch = _sanitize_patch(patch_candidate, base_config)
    merged_config = safe_merge_dicts(base_config, sanitized_patch)
    final_config = load_config(merged_config)

    report = {
        "planner_mode": planner_mode,
        "task_type": task_type,
        "dataset": dataset,
        "env_name": env_name,
        "model": None if dry_run else model,
        "base_url": None if dry_run else base_url,
        "user_goal": user_goal,
        "profile": profile,
        "sanitized_patch": sanitized_patch,
        "rationale": rationale,
    }
    return final_config, report
