"""Sequential research controller harness: init, step, replay, or an external LLM loop.

Model outputs are data in a restricted formula language. Only the explicitly
configured controller executable is launched; never execute commands from its
response. Generated artifacts remain in ignored data/ directories.
"""

import argparse
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from packages.research.proposals import prompt, validate

SOURCES = ("experiments/research_loop.py", "experiments/contact_selection_pilot.py",
           "experiments/research-proposal.example.json", "packages/research/__init__.py",
           "packages/research/proposals.py", "packages/research/ball_design.py",
           "packages/dynamics/__init__.py", "packages/dynamics/bouncing_ball.py",
           "requirements-research.txt")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def locked(directory):
    with (directory/".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def initialize(directory, grid_size=41, seed=20261007):
    if grid_size not in (41, 81, 161) or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("grid must be 41, 81, or 161; seed must be a 32-bit unsigned integer")
    directory.mkdir(parents=True, exist_ok=False)
    snapshots = {}
    for name in SOURCES:
        dest = directory/"source"/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT/name).read_bytes())
        snapshots[str(dest.relative_to(directory))] = sha(dest)
    config = {"schema_version": 1, "seed": seed, "grid_size": grid_size, "world_count": 4,
              "refinement_rmse_tolerance_m": 0.0005,
              "max_jobs": 30, "max_proposals": 2, "job_timeout_s": 120,
              "total_execution_s": 1800, "controller_timeout_s": 60,
              "max_controller_calls": 2, "stage": "development"}
    write(directory/"config.json", config)
    environment = {"python": platform.python_version(), "platform": platform.system(), "machine": platform.machine()}
    from importlib.metadata import version
    environment["dependencies"] = {name: version(name) for name in ("numpy", "scipy")}
    write(directory/"environment.json", environment)
    manifest = {"schema_version": 1, "created_at": now(), "status": "ready",
                "execution_s": 0., "controller_calls": [], "jobs": [], "proposals": [],
                "files": {**snapshots, "config.json": sha(directory/"config.json"),
                          "environment.json": sha(directory/"environment.json")}}
    write(directory/"manifest.json", manifest)
    refresh_prompt(directory, manifest)


def verify(directory, manifest, restarting=True):
    for name, expected in manifest["files"].items():
        if sha(directory/name) != expected:
            raise ValueError(f"frozen artifact changed: {name}")
    # Changes to numerical behavior require a new campaign. Run the old snapshot
    # explicitly if the working tree has moved on.
    for name in SOURCES:
        if sha(ROOT/name) != manifest["files"]["source/"+name]:
            raise ValueError("source differs from campaign snapshot; invoke its source/experiments/research_loop.py")
    from importlib.metadata import version
    environment = read(directory/"environment.json")
    if environment["python"] != platform.python_version() or any(
        version(name) != expected for name, expected in environment["dependencies"].items()
    ):
        raise ValueError("Python or scientific dependency versions changed; recreate the recorded environment")
    if restarting and any(j["status"] == "running" for j in manifest["jobs"] + manifest["controller_calls"]):
        raise ValueError("unresolved running attempt: inspect its recorded PID; do not automatically retry; start a new campaign after reconciling the process")


def refresh_prompt(directory, manifest):
    config = read(directory/"config.json")
    history = [read(directory/p["summary"]) for p in manifest["proposals"] if "summary" in p]
    remaining = {"jobs": config["max_jobs"]-len(manifest["jobs"]),
                 "proposals": config["max_proposals"]-len(manifest["proposals"]),
                 "execution_s": max(0., config["total_execution_s"]-manifest["execution_s"])}
    example = read(directory/"source/experiments/research-proposal.example.json")
    write(directory/"next-prompt.json", prompt(history, remaining, example))


def execute(directory, manifest, record, argv, timeout, stdin=None, controller=False):
    """Reserve before launch, enforce wall time externally, and retain failed attempts."""
    config = read(directory/"config.json")
    remaining = config["total_execution_s"]-manifest["execution_s"]
    if remaining <= 0:
        raise ValueError("execution budget exhausted")
    records = manifest["controller_calls" if controller else "jobs"]
    limit = config["max_controller_calls"] if controller else config["max_jobs"]
    if len(records) >= limit:
        raise ValueError("attempt budget exhausted")
    record.update(status="running", started_at=now(), pid=None)
    records.append(record)
    write(directory/"manifest.json", manifest)
    started = time.monotonic()
    process = None
    try:
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        with (directory/record["stdout"]).open("wb") as out, (directory/record["stderr"]).open("wb") as err:
            process = subprocess.Popen(argv, cwd=directory/"source", stdin=subprocess.PIPE,
                                       stdout=out, stderr=err, start_new_session=True, env=env)
            record["pid"] = process.pid
            write(directory/"manifest.json", manifest)
            process.communicate(input=stdin, timeout=min(timeout, remaining))
            record["returncode"] = process.returncode
            record["status"] = "completed" if process.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        record["status"] = "timed_out"
    except (KeyboardInterrupt, SystemExit):
        record["status"] = "interrupted"
        raise
    except OSError as error:
        record.update(status="failed", error=type(error).__name__)
    finally:
        if process is not None:
            # Descendants can outlive an exited group leader. Always clean up
            # our process group, including the exit-between-poll-and-kill race.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        record["elapsed_s"] = time.monotonic()-started
        record["finished_at"] = now()
        manifest["execution_s"] += record["elapsed_s"]
        for key in ("stdout", "stderr"):
            path = directory/record[key]
            if path.exists():
                manifest["files"][record[key]] = sha(path)
        write(directory/"manifest.json", manifest)
    return record["status"] == "completed"


def scientific(value):
    """Exclude timing/environment fields for deterministic replay comparisons."""
    if isinstance(value, dict):
        return {k: scientific(v) for k, v in value.items() if k not in ("selection_s", "inference_s")}
    if isinstance(value, list):
        return [scientific(v) for v in value]
    return value


def summarize(directory, manifest, entry, expected):
    results = [read(directory/j["output"]) for j in manifest["jobs"]
               if j.get("proposal") == entry["id"] and j.get("kind") == "case" and j["status"] == "completed"]
    summary = {"proposal": read(directory/entry["proposal"]), "completed_cases": len(results),
               "expected_cases": expected, "outcome": "inconclusive", "stage": "development",
               "validity": "Development evidence only; independent scientific replication and confirmation are pending.",
               "evidence": [j["output"] for j in manifest["jobs"] if j.get("proposal") == entry["id"] and j["status"] == "completed"],
               "job_statuses": [j["status"] for j in manifest["jobs"] if j.get("proposal") == entry["id"]],
               "results": []}
    # Never rank methods on a selectively completed subset.
    if len(results) == expected:
        comparisons = [comparison for result in results for comparison in result["refinement"]["comparison"].values()]
        summary["numerical_audit"] = {
            "all_rmse_within_refinement_tolerance": all(c["within_tolerance"] for c in comparisons),
            "all_choices_identical_on_refinement": all(c["choices_identical"] for c in comparisons),
            "max_rmse_change_m": max(c["max_rmse_change_m"] for c in comparisons),
            "max_individual_rmse_change_m": max(c["max_individual_rmse_change_m"] for c in comparisons),
        }
        if not all(c["within_tolerance"] and c["choices_identical"] for c in comparisons):
            summary["validity"] = "Numerical refinement changed a choice or exceeded the RMSE tolerance. Resolve numerical sensitivity before interpreting policy differences."
        for t in sorted({r["measurement_time_s"] for r in results}):
            cases = [r for r in results if r["measurement_time_s"] == t]
            means = {policy: [sum(c["policies"][policy][step]["mean_heldout_rmse_m"] for c in cases)/len(cases)
                              for step in range(6)] for policy in cases[0]["policies"]}
            paired = [c["policies"]["proposed"][-1]["mean_heldout_rmse_m"]-
                      c["policies"]["local_d"][-1]["mean_heldout_rmse_m"] for c in cases]
            costs = {p: sum(sum(row.get("selection_s", 0) for row in c["policies"][p]) for c in cases)/len(cases)
                     for p in cases[0]["policies"]}
            summary["results"].append({"measurement_time_s": t, "mean_rmse_curve_m": means,
                                       "paired_final_difference_m": paired,
                                       "mean_total_selection_s": costs})
    else:
        summary["failure_note"] = "Incomplete batch: no method ranking; inspect retained job statuses."
    if entry.get("replicates"):
        previous = { (j.get("world"), j.get("regime")): j for j in manifest["jobs"]
                    if j.get("proposal") == entry["replicates"] and j.get("kind") == "case" and j["status"] == "completed"}
        comparisons = []
        for job in manifest["jobs"]:
            if job.get("proposal") == entry["id"] and job.get("kind") == "case" and job["status"] == "completed":
                old = previous.get((job["world"], job["regime"]))
                comparisons.append(bool(old) and scientific(read(directory/old["output"])) == scientific(read(directory/job["output"])))
        summary["repeatability"] = {"all_scientific_fields_identical": len(comparisons) == expected and all(comparisons),
                                    "compared_cases": len(comparisons), "independent_replication": False}
    dest = Path(entry["proposal"]).parent/"summary.json"
    write(directory/dest, summary)
    entry["summary"] = str(dest)
    manifest["files"][str(dest)] = sha(directory/dest)
    lines = [f"# {entry['id']}", "", "Development evidence; scientific outcome remains inconclusive.",
             f"Completed cases: {len(results)}/{expected}.", "", "| Measurement time | Policy | Final mean RMSE (m) |", "| --- | --- | --- |"]
    for row in summary["results"]:
        for policy, curve in row["mean_rmse_curve_m"].items():
            lines.append(f"| {row['measurement_time_s']:.2f} | {policy} | {curve[-1]:.6f} |")
    lines.extend(["", summary["validity"], "", "Exact proposals, per-round choices, scores, measurements and timing are in the JSON records."])
    report = dest.with_suffix(".md")
    (directory/report).write_text("\n".join(lines)+"\n")
    manifest["files"][str(report)] = sha(directory/report)


def step(directory, response_bytes, origin):
    manifest = read(directory/"manifest.json")
    verify(directory, manifest)
    config = read(directory/"config.json")
    if manifest["status"] != "ready" or len(manifest["proposals"]) >= config["max_proposals"]:
        raise ValueError("campaign is stopped or proposal budget exhausted")
    # Preserve malformed responses too; validation occurs before any numerical job.
    response_name = f"response-{len(list(directory.glob('response-*.json'))):03d}.json"
    (directory/response_name).write_bytes(response_bytes)
    manifest["files"][response_name] = sha(directory/response_name)
    write(directory/"manifest.json", manifest)
    try:
        if len(response_bytes) > 65536:
            raise ValueError("proposal response exceeds 64 KiB")
        proposal = validate(json.loads(response_bytes))
    except (ValueError, TypeError) as error:
        manifest.setdefault("rejections", []).append({"response": response_name, "reason": str(error)})
        write(directory/"manifest.json", manifest)
        raise
    if any(p["id"] == proposal["id"] for p in manifest["proposals"]):
        raise ValueError("duplicate proposal id; completed proposals are never rerun implicitly")
    if proposal["action"] == "stop":
        manifest["status"] = "stopped_by_controller"
        write(directory/"manifest.json", manifest)
        return
    if proposal["replicates"]:
        previous = next((p for p in manifest["proposals"] if p["id"] == proposal["replicates"]), None)
        if previous is None:
            raise ValueError("replication target does not exist")
        original = read(directory/previous["proposal"])
        if any(proposal[key] != original[key] for key in ("formula", "measurement_times_s", "noise_std_m")):
            raise ValueError("repeatability proposal must preserve formula and experiment settings")
    expected = config["world_count"]*len(proposal["measurement_times_s"])
    if len(manifest["jobs"])+expected+1 > config["max_jobs"]:
        raise ValueError("insufficient remaining job budget for the complete batch and checks")
    folder = Path("proposals")/proposal["id"]
    (directory/folder).mkdir(parents=True)
    write(directory/folder/"proposal.json", proposal)
    write(directory/folder/"prompt.json", read(directory/"next-prompt.json"))
    entry = {"id": proposal["id"], "proposal": str(folder/"proposal.json"),
             "origin": origin, "replicates": proposal["replicates"], "status": "running"}
    manifest["proposals"].append(entry)
    for name in (folder/"proposal.json", folder/"prompt.json"):
        manifest["files"][str(name)] = sha(directory/name)
    write(directory/"manifest.json", manifest)
    tasks = [("check", None, None)] + [("case", world, regime)
                for regime in range(len(proposal["measurement_times_s"])) for world in range(config["world_count"])]
    try:
        for index, (kind, world, regime) in enumerate(tasks):
            if manifest["execution_s"] >= config["total_execution_s"]:
                break
            name = folder/f"job-{index:03d}"
            output = str(name)+".json"
            record = {"proposal": proposal["id"], "kind": kind, "world": world, "regime": regime,
                      "stdout": str(name)+".stdout.txt", "stderr": str(name)+".stderr.txt", "output": output}
            argv = [sys.executable, str(directory/"source/experiments/contact_selection_pilot.py"),
                    "--config", str(directory/"config.json"), "--proposal", str(directory/entry["proposal"]),
                    "--out", str(directory/output)]
            argv += ["--check"] if kind == "check" else ["--world", str(world), "--regime", str(regime)]
            ok = execute(directory, manifest, record, argv, config["job_timeout_s"])
            if ok:
                try:
                    read(directory/output)
                    manifest["files"][output] = sha(directory/output)
                except (OSError, ValueError):
                    record["status"] = "invalid_output"
                    ok = False
            write(directory/"manifest.json", manifest)
            print(f"{proposal['id']}: {index+1}/{len(tasks)} {record['status']}", flush=True)
            if kind == "check" and not ok:
                break
        entry["status"] = "finished"
    finally:
        if entry["status"] == "running":
            entry["status"] = "interrupted"
        summarize(directory, manifest, entry, expected)
        if (len(manifest["proposals"]) >= config["max_proposals"]
                or manifest["execution_s"] >= config["total_execution_s"]
                or len(manifest["jobs"]) >= config["max_jobs"]):
            manifest["status"] = "budget_exhausted"
        write(directory/"manifest.json", manifest)
        refresh_prompt(directory, manifest)


def automatic(directory, command):
    if not isinstance(command, list) or not command or any(not isinstance(v, str) for v in command):
        raise ValueError("controller command must be a nonempty JSON argv array")
    while True:
        manifest = read(directory/"manifest.json")
        verify(directory, manifest)
        config = read(directory/"config.json")
        if (manifest["status"] != "ready" or len(manifest["proposals"]) >= config["max_proposals"]
                or len(manifest["controller_calls"]) >= config["max_controller_calls"]
                or manifest["execution_s"] >= config["total_execution_s"]):
            break
        index = len(manifest["controller_calls"])
        record = {"stdout": f"controller-{index}.stdout.json", "stderr": f"controller-{index}.stderr.txt"}
        ok = execute(directory, manifest, record, command, config["controller_timeout_s"],
                     stdin=(directory/"next-prompt.json").read_bytes(), controller=True)
        if not ok:
            break
        step(directory, (directory/record["stdout"]).read_bytes(), "external-controller")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    init = sub.add_parser("init"); init.add_argument("directory", type=Path)
    init.add_argument("--grid-size", type=int, choices=(41, 81, 161), default=41)
    init.add_argument("--seed", type=int, default=20261007)
    single = sub.add_parser("step"); single.add_argument("directory", type=Path)
    single.add_argument("--proposal", required=True, type=Path)
    single.add_argument("--origin", default="file-controller", help="honest provenance label, not model authentication")
    auto = sub.add_parser("run"); auto.add_argument("directory", type=Path)
    auto.add_argument("--controller-command-json", required=True)
    status = sub.add_parser("status"); status.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    try:
        if args.action == "init":
            initialize(directory, args.grid_size, args.seed)
        elif args.action == "status":
            manifest = read(directory/"manifest.json")
            verify(directory, manifest, restarting=False)
            print(json.dumps({key: manifest[key] for key in ("status", "execution_s", "proposals", "jobs", "controller_calls")}, indent=2))
        else:
            with locked(directory):
                if args.action == "step":
                    step(directory, args.proposal.read_bytes(), args.origin)
                else:
                    automatic(directory, json.loads(args.controller_command_json))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"Research loop stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
