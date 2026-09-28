"""Materialize unlisted generalization cases at evaluation time.

Guide §4.3 says the generalization probe is a random shape that appears in no
public or hidden list.  A manifest therefore stores only a private sampling
policy.  This module realizes the shape after evaluation starts, avoids every
listed shape and seed, and returns the runtime seed so the single run remains
reproducible without turning the result into a published lookup table.
"""

from __future__ import annotations

import copy
import json
import random
import secrets
from typing import Dict, List, Optional, Tuple


GENERATOR_ID = "musa_operator_eval/evaluator/generalization_probe.py"
PATHS = {"fused_library", "library_composition", "custom_fallback"}
DTYPES = {"float16", "bfloat16", "float32"}


def validate_policy(policy: dict, tier: str) -> List[str]:
    problems: List[str] = []
    if not isinstance(policy, dict):
        return ["generalization_probe must be an object"]
    if policy.get("generator") != GENERATOR_ID:
        problems.append(f"generalization_probe.generator must be {GENERATOR_ID}")
    if policy.get("seed_source") != "runtime_entropy":
        problems.append("generalization_probe.seed_source must be runtime_entropy")
    count = policy.get("count")
    if not isinstance(count, int) or not 1 <= count <= 16:
        problems.append("generalization_probe.count must be an integer from 1 to 16")
    dtype_choices = policy.get("dtype_choices")
    if not isinstance(dtype_choices, list) or not dtype_choices:
        problems.append("generalization_probe declares no dtype choices")
    elif any(dtype not in DTYPES for dtype in dtype_choices):
        problems.append("generalization_probe declares an unknown dtype choice")
    shape_space = policy.get("shape_space")
    if not isinstance(shape_space, dict) or not shape_space:
        problems.append("generalization_probe declares no shape space")
    else:
        for axis, spec in shape_space.items():
            if not isinstance(spec, dict):
                problems.append(f"shape axis {axis} must be an object")
                continue
            choices = spec.get("choices")
            bounded = "minimum" in spec and "maximum" in spec
            if bool(choices) == bool(bounded):
                problems.append(f"shape axis {axis} must declare choices or minimum/maximum, exclusively")
            if choices and (not all(isinstance(value, int) and value > 0 for value in choices)):
                problems.append(f"shape axis {axis} has a non-positive or non-integer choice")
            if bounded:
                minimum, maximum = spec["minimum"], spec["maximum"]
                step = spec.get("step", 1)
                if not all(isinstance(value, int) for value in (minimum, maximum, step)):
                    problems.append(f"shape axis {axis} bounds and step must be integers")
                elif minimum < 1 or maximum < minimum or step < 1:
                    problems.append(f"shape axis {axis} has an invalid range")
    template = policy.get("template")
    if not isinstance(template, dict):
        problems.append("generalization_probe.template must be an object")
        template = {}
    if "shape" in template or "seed" in template:
        problems.append("the runtime probe template may not predeclare a shape or seed")
    rules = policy.get("expected_path_rules") or []
    if tier == "B_library" and not rules:
        problems.append("a B-tier runtime probe needs expected_path_rules")
    if rules and not isinstance(rules, list):
        problems.append("expected_path_rules must be a list")
    elif rules:
        for index, rule in enumerate(rules):
            if not isinstance(rule, dict):
                problems.append(f"expected_path_rules[{index}] must be an object")
                continue
            if rule.get("path") not in PATHS:
                problems.append("a runtime expected-path rule names an unknown path")
            conditions = rule.get("when")
            if conditions is not None:
                if not isinstance(conditions, dict):
                    problems.append(f"expected_path_rules[{index}].when must be an object")
                else:
                    for axis, condition in conditions.items():
                        if not isinstance(condition, dict) or not condition:
                            problems.append(
                                f"expected_path_rules[{index}].when.{axis} must be a non-empty object"
                            )
        if not any(isinstance(rule, dict) and "when" not in rule for rule in rules):
            problems.append("expected_path_rules need a final default rule without `when`")
    return problems


def _sample_axis(rng: random.Random, spec: dict) -> int:
    if spec.get("choices"):
        return rng.choice(spec["choices"])
    values = range(spec["minimum"], spec["maximum"] + 1, spec.get("step", 1))
    return rng.choice(values)


def _condition_matches(value: int, condition: dict) -> bool:
    if "equals" in condition and value != condition["equals"]:
        return False
    if "choices" in condition and value not in condition["choices"]:
        return False
    if "minimum" in condition and value < condition["minimum"]:
        return False
    if "maximum" in condition and value > condition["maximum"]:
        return False
    return True


def expected_path(shape: dict, rules: List[dict]) -> Optional[str]:
    for rule in rules:
        conditions = rule.get("when")
        if conditions is None:
            return rule["path"]
        if all(axis in shape and _condition_matches(shape[axis], condition)
               for axis, condition in conditions.items()):
            return rule["path"]
    return None


def materialize_probes(
    task: dict,
    manifest: dict,
    runtime_seed: Optional[int] = None,
) -> Tuple[List[dict], int]:
    policy = manifest.get("generalization_probe")
    if not policy:
        return [], runtime_seed if runtime_seed is not None else 0
    problems = validate_policy(policy, task.get("tier"))
    if problems:
        raise ValueError("; ".join(problems))

    seed = runtime_seed if runtime_seed is not None else secrets.randbits(63)
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("the runtime generalization seed must be a non-negative integer")
    rng = random.Random(seed)
    listed_shapes = {json.dumps(case["shape"], sort_keys=True) for case in manifest.get("cases", [])}
    listed_seeds = {case["seed"] for case in manifest.get("cases", [])}
    probes: List[dict] = []

    for index in range(policy["count"]):
        for _attempt in range(1000):
            shape = {axis: _sample_axis(rng, spec) for axis, spec in policy["shape_space"].items()}
            shape_key = json.dumps(shape, sort_keys=True)
            case_seed = rng.getrandbits(63)
            if shape_key not in listed_shapes and case_seed not in listed_seeds:
                break
        else:
            raise ValueError("the generalization shape space cannot produce an unlisted case")

        case = copy.deepcopy(policy.get("template") or {})
        case.update({
            "case_id": f"generalization_{seed:016x}_{index + 1:02d}",
            "tag": "generalization",
            "seed": case_seed,
            "dtype": rng.choice(policy["dtype_choices"]),
            "shape": shape,
        })
        case.setdefault("distribution", "normal")
        case["golden"] = f"{case['case_id']}/golden/tensors.json"
        if task.get("tier") == "B_library":
            selected = expected_path(shape, policy["expected_path_rules"])
            if selected is None:
                raise ValueError(f"no expected-path rule covers runtime shape {shape}")
            case["expected_path"] = selected
        probes.append(case)
        listed_shapes.add(shape_key)
        listed_seeds.add(case_seed)

    return probes, seed

