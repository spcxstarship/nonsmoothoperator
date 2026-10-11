"""Store one experiment-selection round as local, provider-independent files."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from selector import parse_decision, validate_context


SCHEMA_VERSION = 1
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "runs"
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
USAGE_FIELDS = ("elapsed_ms", "input_tokens", "cached_input_tokens",
                "output_tokens", "reasoning_tokens")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def valid_id(value: str) -> str:
    if value in {".", ".."} or not ID_PATTERN.fullmatch(value):
        raise ValueError(f"invalid identifier: {value!r}")
    return value


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_new(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)


def write_manifest(path: Path, manifest: dict) -> None:
    data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(".manifest.tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
    os.replace(temporary, path)


def start(args: argparse.Namespace) -> Path:
    study, trial, round_id = map(valid_id, (args.study, args.trial, args.round))
    if not args.selector.strip():
        raise ValueError("selector ID must be nonempty")
    context_data = args.context.read_bytes()
    validate_context(json.loads(context_data))
    instructions_data = args.instructions.read_bytes() if args.instructions else None
    target = args.root / study / trial / round_id
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    write_new(target / "context.json", context_data)
    if instructions_data is not None:
        write_new(target / "instructions.txt", instructions_data)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "open",
        "study_id": study,
        "trial_id": trial,
        "round_id": round_id,
        "started_at_utc": utc_now(),
        "completed_at_utc": None,
        "selector": {
            "id": args.selector,
            "model_id": args.model_id,
            "prompt_version": args.prompt_version,
        },
        "selected_candidate_id": None,
        "files_sha256": {"context.json": digest(context_data)},
        "usage": None,
    }
    if instructions_data is not None:
        manifest["files_sha256"]["instructions.txt"] = digest(instructions_data)
    write_manifest(target / "manifest.json", manifest)
    return target


def validate_usage(usage: dict) -> dict:
    if set(usage) != {"source", *USAGE_FIELDS}:
        raise ValueError("usage must have source and the five standard fields")
    if not isinstance(usage["source"], str) or not usage["source"].strip():
        raise ValueError("usage source must be a nonempty string")
    for field in USAGE_FIELDS:
        value = usage[field]
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"usage {field} must be a nonnegative integer or null")
    return usage


def finish(args: argparse.Namespace) -> Path:
    target = args.directory
    manifest_path = target / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "open":
        raise ValueError("round is not open with the supported schema version")
    for name, expected in manifest["files_sha256"].items():
        if digest((target / name).read_bytes()) != expected:
            raise ValueError(f"frozen {name} has changed")
    context_data = (target / "context.json").read_bytes()
    allowed_ids = validate_context(json.loads(context_data))
    decision_data = args.decision.read_bytes()
    decision = parse_decision(decision_data.decode("utf-8"), allowed_ids)
    result_data = args.result.read_bytes()
    result = json.loads(result_data)
    if not isinstance(result, dict) or result.get("candidate_id") != decision.candidate_id:
        raise ValueError("result candidate_id must match the validated decision")
    usage = validate_usage(read_json(args.usage)) if args.usage else None
    if (target / "decision.json").exists() or (target / "result.json").exists():
        raise ValueError("round already has decision or result files")

    write_new(target / "decision.json", decision_data)
    write_new(target / "result.json", result_data)
    manifest["selected_candidate_id"] = decision.candidate_id
    manifest["completed_at_utc"] = utc_now()
    manifest["status"] = "complete"
    manifest["usage"] = usage
    manifest["files_sha256"].update({
        "decision.json": digest(decision_data),
        "result.json": digest(result_data),
    })
    write_manifest(manifest_path, manifest)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    begin = subcommands.add_parser("start", help="freeze the before-choice context")
    begin.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    begin.add_argument("--study", required=True)
    begin.add_argument("--trial", required=True)
    begin.add_argument("--round", required=True)
    begin.add_argument("--context", required=True, type=Path)
    begin.add_argument("--instructions", type=Path,
                       help="optional exact selector instructions to preserve")
    begin.add_argument("--selector", required=True, help="policy ID, model or heuristic")
    begin.add_argument("--model-id", help="model ID if the selector uses a model")
    begin.add_argument("--prompt-version", help="prompt version if applicable")
    end = subcommands.add_parser("finish", help="save a choice and matching result")
    end.add_argument("directory", type=Path)
    end.add_argument("--decision", required=True, type=Path)
    end.add_argument("--result", required=True, type=Path)
    end.add_argument("--usage", type=Path, help="optional standard usage JSON")
    args = parser.parse_args()
    try:
        target = start(args) if args.command == "start" else finish(args)
        print(target)
        return 0
    except (OSError, ValueError, KeyError, TypeError, UnicodeDecodeError) as error:
        print(f"Round store failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
