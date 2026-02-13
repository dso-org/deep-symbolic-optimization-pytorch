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
    periodicity_fft = _estimate_periodicity(y)
    feature_diagnostics = _compute_feature_diagnostics(X, y)
    interaction_diagnostics = _compute_interaction_diagnostics(X, y)
    periodic_basis_max = (
        float(max(d["periodic_score"] for d in feature_diagnostics))
        if feature_diagnostics
        else 0.0
    )
    family_scores = _compute_family_scores(
        feature_diagnostics=feature_diagnostics,
        interaction_diagnostics=interaction_diagnostics,
        periodicity_fft=periodicity_fft,
    )
    top_family_hypotheses = _top_family_hypotheses(family_scores)
    likely_periodic_signal = bool(
        (periodicity_fft > 0.25 and periodic_basis_max > 0.20)
        or periodic_basis_max > 0.45
    )
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
        "has_negative_inputs": bool(np.min(X) < 0),
        "periodicity_fft": periodicity_fft,
        "periodic_basis_max_corr": periodic_basis_max,
        "likely_periodic_signal": likely_periodic_signal,
        "max_feature_monotonicity": (
            float(max(d["abs_spearman_y"] for d in feature_diagnostics))
            if feature_diagnostics
            else 0.0
        ),
        "feature_diagnostics": feature_diagnostics,
        "interaction_diagnostics": interaction_diagnostics,
        "family_scores": family_scores,
        "top_family_hypotheses": top_family_hypotheses,
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


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(valid)) < 3:
        return 0.0
    a = a[valid]
    b = b[valid]
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    corr = np.corrcoef(a, b)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(np.clip(corr, -1.0, 1.0))


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(valid)) < 3:
        return 0.0
    a_rank = pd.Series(a[valid]).rank(method="average").to_numpy(dtype=np.float64)
    b_rank = pd.Series(b[valid]).rank(method="average").to_numpy(dtype=np.float64)
    return _safe_corr(a_rank, b_rank)


def _safe_minmax_scale(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    valid = np.isfinite(x)
    if int(np.sum(valid)) == 0:
        return np.zeros_like(x, dtype=np.float64)
    x_valid = x[valid]
    x_min = float(np.min(x_valid))
    x_range = float(np.max(x_valid) - x_min)
    out = np.zeros_like(x, dtype=np.float64)
    if x_range < 1e-12:
        return out
    out[valid] = (x_valid - x_min) / x_range
    return out


def _compute_feature_diagnostics(X: np.ndarray, y: np.ndarray) -> list:
    diagnostics = []
    for i in range(X.shape[1]):
        xi = X[:, i]
        x01 = _safe_minmax_scale(xi)
        x_centered = xi - np.mean(xi)

        pearson_y = _safe_corr(xi, y)
        spearman_y = _safe_spearman(xi, y)
        abs_linear = abs(pearson_y)
        quadratic_score = abs(_safe_corr(np.square(xi), y))
        cubic_score = abs(_safe_corr(np.power(xi, 3), y))
        polynomial_score = float(max(abs_linear, quadratic_score, cubic_score))

        trig_candidates = [
            np.sin(2.0 * np.pi * x01),
            np.cos(2.0 * np.pi * x01),
            np.sin(4.0 * np.pi * x01),
            np.cos(4.0 * np.pi * x01),
        ]
        periodic_score = float(max(abs(_safe_corr(t, y)) for t in trig_candidates))

        exp_input = np.clip(x_centered, -6.0, 6.0)
        exp_score = abs(_safe_corr(np.exp(exp_input), y))

        # Shift to non-negative so log1p stays defined and stable.
        shift = max(0.0, -float(np.min(xi)))
        log_score = abs(_safe_corr(np.log1p(np.maximum(xi + shift, 0.0)), y))

        diagnostics.append(
            {
                "feature": f"x{i + 1}",
                "min": float(np.min(xi)),
                "max": float(np.max(xi)),
                "std": float(np.std(xi)),
                "pearson_y": float(pearson_y),
                "spearman_y": float(spearman_y),
                "abs_spearman_y": float(abs(spearman_y)),
                "linear_score": float(abs_linear),
                "polynomial_score": polynomial_score,
                "periodic_score": periodic_score,
                "exp_score": float(exp_score),
                "log_score": float(log_score),
            }
        )
    return diagnostics


def _compute_interaction_diagnostics(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    n_features = int(X.shape[1])
    if n_features < 2:
        return {"max_abs_corr": 0.0, "top_terms": []}

    pair_scores = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            term = X[:, i] * X[:, j]
            score = abs(_safe_corr(term, y))
            pair_scores.append(
                {
                    "term": f"x{i + 1}*x{j + 1}",
                    "abs_corr_with_y": float(score),
                }
            )

    pair_scores.sort(key=lambda d: d["abs_corr_with_y"], reverse=True)
    max_abs_corr = pair_scores[0]["abs_corr_with_y"] if pair_scores else 0.0
    return {"max_abs_corr": float(max_abs_corr), "top_terms": pair_scores[:3]}


def _compute_family_scores(
    *,
    feature_diagnostics: list,
    interaction_diagnostics: Dict[str, Any],
    periodicity_fft: float,
) -> Dict[str, float]:
    if not feature_diagnostics:
        return {
            "polynomial": 0.0,
            "periodic": float(periodicity_fft),
            "exponential": 0.0,
            "logarithmic": 0.0,
            "interaction": 0.0,
        }

    polynomial_score = max(d["polynomial_score"] for d in feature_diagnostics)
    periodic_score = max(
        float(periodicity_fft),
        max(d["periodic_score"] for d in feature_diagnostics),
    )
    exponential_score = max(d["exp_score"] for d in feature_diagnostics)
    logarithmic_score = max(d["log_score"] for d in feature_diagnostics)
    interaction_score = float(interaction_diagnostics.get("max_abs_corr", 0.0))
    return {
        "polynomial": float(polynomial_score),
        "periodic": float(periodic_score),
        "exponential": float(exponential_score),
        "logarithmic": float(logarithmic_score),
        "interaction": interaction_score,
    }


def _top_family_hypotheses(
    family_scores: Dict[str, float], top_k: int = 3
) -> list:
    ranked = sorted(family_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [
        {"family": name, "score": float(score)}
        for name, score in ranked[:top_k]
    ]


def _suggest_skeletons_deterministic(
    profile: Dict[str, Any], user_goal: str
) -> Optional[list]:
    """Suggest skeleton expressions based on dataset profile and user goal."""
    goal = (user_goal or "").lower()
    skeletons = []
    n_features = int(profile.get("n_features", 1))

    # Build variable list: x1, x2, ..., xN
    var_names = [f"x{i+1}" for i in range(n_features)]

    # Check for skeleton-related keywords in goal
    if "skeleton" not in goal:
        # Only suggest skeletons if explicitly requested or dataset strongly suggests it
        if not profile.get("likely_periodic_signal", False):
            return None

    # Fourier basis for periodic signals
    if profile.get("likely_periodic_signal", False) or "periodic" in goal or "fourier" in goal:
        # For multivariate: add Fourier basis for each variable
        if n_features == 1:
            skeletons.append({
                "name": "fourier_basis",
                "expression": "a*cos(x1) + b*sin(x1) + c",
                "description": "Fourier basis for periodic signal"
            })
        else:
            # Combined Fourier basis across all variables
            terms = []
            coef_idx = 0
            coef_letters = "abcdefghijklmnopqrstuvwyz"
            for v in var_names:
                c1 = coef_letters[coef_idx % len(coef_letters)]
                c2 = coef_letters[(coef_idx + 1) % len(coef_letters)]
                terms.append(f"{c1}*cos({v}) + {c2}*sin({v})")
                coef_idx += 2
            # Add offset
            offset = coef_letters[coef_idx % len(coef_letters)]
            expr = " + ".join(terms) + f" + {offset}"
            skeletons.append({
                "name": "fourier_basis",
                "expression": expr,
                "description": f"Fourier basis across {n_features} variables"
            })

    # Exponential for wide y range (orders of magnitude)
    y_min = profile.get("y_min", 0)
    y_max = profile.get("y_max", 1)
    y_std = profile.get("y_std", 1)
    y_range = y_max - y_min
    if y_range > 100 * y_std or "exponential" in goal or "exp" in goal:
        skeletons.append({
            "name": "exp_decay",
            "expression": "a*exp(-b*x1) + c",
            "description": "Exponential decay for wide value range"
        })

    # Linear basis as fallback if requested — uses all variables
    if "linear" in goal or "robust" in goal:
        coef_letters = "abcdefghijklmnopqrstuvwyz"
        terms = [f"{coef_letters[i]}*{v}" for i, v in enumerate(var_names)]
        offset = coef_letters[len(var_names)]
        expr = " + ".join(terms) + f" + {offset}"
        skeletons.insert(0, {
            "name": "linear_basis",
            "expression": expr,
            "description": f"Linear basis across {n_features} variable(s)"
        })

    return skeletons[:3] if skeletons else None  # Limit to 3 suggestions


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

    # Add skeleton expression suggestions if appropriate
    if profile.get("available"):
        skeleton_suggestions = _suggest_skeletons_deterministic(profile, user_goal)
        if skeleton_suggestions:
            patch["task"]["skeleton_expressions"] = skeleton_suggestions

    return patch


def _build_messages(
    base_config: Dict[str, Any], profile: Dict[str, Any], user_goal: str
) -> list:
    n_features = int(profile.get("n_features", 1))
    var_hint = ", ".join(f"x{i+1}" for i in range(n_features))

    system_text = (
        "You are a configuration planner for Deep Symbolic Optimization (DSO). "
        "Return strict JSON only. Keep changes conservative and runnable.\n\n"
        "When task_type is regression, you may suggest skeleton_expressions "
        "based on dataset characteristics:\n"
        "- Periodic signal → Fourier basis: \"a*cos(x1) + b*sin(x1) + c\"\n"
        "- Exponential growth/decay → \"a*exp(b*x1) + c\" or \"a*exp(-b*x1) + c\"\n"
        "- Power law → \"a*x1**b + c\" (nonlinear, slower optimization)\n"
        "- Polynomial → Use built-in \"poly\" token instead\n\n"
        "Skeleton expressions are optimized automatically. Prefer linear "
        "skeletons (faster) over nonlinear when possible.\n\n"
        "Dataset diagnostics include `family_scores`, `top_family_hypotheses`, "
        "`feature_diagnostics`, and `interaction_diagnostics`. Base your "
        "skeleton proposals on these signals instead of generic templates.\n"
        "Do NOT return bare token names as expressions (for example: poly, "
        "const, sin, exp). Expressions must be symbolic formulas containing "
        "coefficients and variables, e.g. 'a*x1**2 + b*x1 + c'.\n"
        "Traversal length controls are set through config fields "
        "`prior.length.max_` (hard cap) and `policy.max_length` (controller cap). "
        "If the goal asks for concise expressions, lower these values "
        "(for example, around 16-28) and keep skeleton suggestions short.\n"
        "If `interaction_diagnostics.max_abs_corr` is high, include at least one "
        "interaction skeleton using products like x1*x2.\n\n"
        f"This dataset has {n_features} input variable(s): {var_hint}. "
        "When the dataset is multivariate, skeleton expressions SHOULD reference "
        "the relevant variables (x1, x2, ...). Examples for multivariate data:\n"
        "- Linear: \"a*x1 + b*x2 + c\"\n"
        "- Interaction: \"a*x1*x2 + b*x1 + c*x2 + d\"\n"
        "- Mixed: \"a*cos(x1) + b*x2**2 + c\"\n\n"
        "Return skeleton_expressions as array of objects with:\n"
        "- name: unique identifier\n"
        "- expression: math expression with coefficients (a,b,c,...) and vars (x1,x2,...)\n"
        "- description: why this skeleton fits the data"
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


def _sanitize_skeleton_expressions(
    raw_skeletons: Any, n_input_vars: int
) -> Dict[str, Any]:
    """Keep only well-formed, parseable skeleton specs."""
    from dso.skeleton import parse_skeleton

    if not isinstance(raw_skeletons, list):
        return {"kept": [], "requested": 0, "dropped": 0}

    reserved_expr = {"poly", "const"}
    cleaned = []
    used_names = set()
    dropped = 0

    for i, spec in enumerate(raw_skeletons):
        if not isinstance(spec, dict):
            dropped += 1
            continue

        expr = spec.get("expression")
        if not isinstance(expr, str) or not expr.strip():
            dropped += 1
            continue
        expr = expr.strip()
        if expr.lower() in reserved_expr:
            dropped += 1
            continue

        name = spec.get("name")
        if not isinstance(name, str) or not name.strip():
            name = f"skeleton_{i + 1}"
        name = name.strip()
        base_name = name
        suffix = 2
        while name in used_names:
            name = f"{base_name}_{suffix}"
            suffix += 1

        try:
            parse_skeleton(expr, n_input_vars, name)
        except Exception:
            dropped += 1
            continue

        clean_spec: Dict[str, Any] = {"name": name, "expression": expr}
        description = spec.get("description")
        if isinstance(description, str) and description.strip():
            clean_spec["description"] = description.strip()
        max_count = spec.get("max_count")
        if isinstance(max_count, int) and max_count > 0:
            clean_spec["max_count"] = max_count

        cleaned.append(clean_spec)
        used_names.add(name)

    return {
        "kept": cleaned,
        "requested": len(raw_skeletons),
        "dropped": dropped,
    }


def _wants_short_traversals(user_goal: str) -> bool:
    goal = (user_goal or "").lower()
    hints = [
        "short traversal",
        "short traversals",
        "shorter traversal",
        "shorter traversals",
        "concise",
        "compact",
        "simple expression",
        "simple expressions",
        "short expression",
        "short expressions",
        "interpretable",
    ]
    return any(hint in goal for hint in hints)


def _apply_goal_based_length_controls(
    sanitized_patch: Dict[str, Any], user_goal: str
) -> Optional[Dict[str, Any]]:
    """Apply conservative traversal-length limits when requested by the goal."""
    if not _wants_short_traversals(user_goal):
        return None

    applied: Dict[str, Any] = {}
    prior_patch = sanitized_patch.setdefault("prior", {})
    length_patch = prior_patch.setdefault("length", {})
    length_patch["on"] = True

    if not isinstance(length_patch.get("max_"), int) or length_patch["max_"] > 24:
        length_patch["max_"] = 24
        applied["prior.length.max_"] = 24

    min_length = length_patch.get("min_")
    if not isinstance(min_length, int) or min_length < 2:
        length_patch["min_"] = 4
        applied["prior.length.min_"] = 4
    elif isinstance(length_patch.get("max_"), int) and min_length > length_patch["max_"]:
        length_patch["min_"] = 4
        applied["prior.length.min_"] = 4

    policy_patch = sanitized_patch.setdefault("policy", {})
    if not isinstance(policy_patch.get("max_length"), int) or policy_patch["max_length"] > 32:
        policy_patch["max_length"] = 32
        applied["policy.max_length"] = 32

    task_patch = sanitized_patch.get("task", {})
    skeletons = task_patch.get("skeleton_expressions")
    if isinstance(skeletons, list) and len(skeletons) > 3:
        task_patch["skeleton_expressions"] = skeletons[:3]
        applied["task.skeleton_expressions.count"] = 3

    return applied or {"applied": True}


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

    # skeleton_expressions and skeleton_optimizer_params are lists/dicts
    # not present in the base config template, so _sanitize_patch drops them.
    # Preserve them from the original patch if present.
    task_patch = patch_candidate.get("task", {})
    skeleton_sanitize_stats = None
    if task_type == "regression" and "skeleton_expressions" in task_patch:
        n_input_vars = int(profile.get("n_features", 1))
        skeleton_sanitize_stats = _sanitize_skeleton_expressions(
            task_patch["skeleton_expressions"], n_input_vars
        )
        if skeleton_sanitize_stats["kept"]:
            sanitized_patch.setdefault("task", {})["skeleton_expressions"] = (
                skeleton_sanitize_stats["kept"]
            )
    if "skeleton_optimizer_params" in task_patch:
        sanitized_patch.setdefault("task", {})["skeleton_optimizer_params"] = task_patch[
            "skeleton_optimizer_params"
        ]
    length_controls_applied = _apply_goal_based_length_controls(
        sanitized_patch, user_goal
    )

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
        "skeleton_sanitize": skeleton_sanitize_stats,
        "length_controls": length_controls_applied,
    }
    return final_config, report
