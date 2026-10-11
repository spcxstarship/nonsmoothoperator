"""Provider-independent contract for selecting the next actuation experiment.

The model can choose only a candidate ID. This module does not simulate a
system, train a PINN, or treat model-written text as executable code.
"""

import json
from dataclasses import dataclass
from typing import Any


PROMPT_VERSION = "select-v1"


SYSTEM_INSTRUCTIONS = """You select one next actuation experiment for a hybrid-system identification study.
Choose exactly one ID from the supplied candidates. Base the choice only on the
given objective, completed experiments, and diagnostics. Diagnostics are
imperfect estimates, not proof of model error. Do not design or alter the
PINN, its loss, or its training procedure. Return only JSON with keys
candidate_id, hypothesis, and rationale. The hypothesis should state what
the selected experiment may reveal. Keep each text field under 500 characters.
"""


@dataclass(frozen=True)
class Decision:
    candidate_id: str
    hypothesis: str
    rationale: str


def validate_context(context: dict[str, Any]) -> set[str]:
    """Validate the small public interface before sending it to a model."""
    if not isinstance(context, dict):
        raise ValueError("context must be a JSON object")
    if not isinstance(context.get("objective"), str) or not context["objective"].strip():
        raise ValueError("objective must be a nonempty string")
    candidates = context.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a nonempty list")
    ids = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("each candidate must be an object")
        candidate_id = candidate.get("id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("each candidate needs a nonempty id")
        ids.append(candidate_id)
    if len(ids) != len(set(ids)):
        raise ValueError("candidate ids must be unique")
    for field in ("history", "diagnostics"):
        if field in context and not isinstance(context[field], (list if field == "history" else dict)):
            raise ValueError(f"{field} has the wrong type")
    return set(ids)


def build_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    validate_context(context)
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": json.dumps(context, sort_keys=True)},
    ]


def parse_decision(response: str, allowed_ids: set[str]) -> Decision:
    """Reject malformed choices and choices outside the supplied set."""
    try:
        result = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError("model response is not JSON") from error
    if not isinstance(result, dict) or set(result) != {
        "candidate_id", "hypothesis", "rationale"
    }:
        raise ValueError("model response has the wrong fields")
    candidate_id = result["candidate_id"]
    if not isinstance(candidate_id, str) or candidate_id not in allowed_ids:
        raise ValueError("model selected an unknown candidate")
    for field in ("hypothesis", "rationale"):
        value = result[field]
        if not isinstance(value, str) or not value.strip() or len(value) > 500:
            raise ValueError(f"{field} must be a nonempty string under 500 characters")
    return Decision(candidate_id, result["hypothesis"], result["rationale"])
