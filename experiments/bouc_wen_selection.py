"""Prospective session-LLM comparison on eight fresh normalized Bouc-Wen systems.

Stages: prepare --out NEW; context --out NEW; record --out NEW < decisions.json;
evaluate --out NEW. The caller must make every LLM decision before evaluation.
The LLM is the current assistant, with development conversation context, not a
fresh independent API call. Ground truth and future observations are stored in
an evaluator directory and must not be read by the caller before recording.
This is procedural blinding, not a security boundary or an external-model test.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from packages.dynamics import bouc_wen as bw
import bouc_wen_pilot as pilot

SEED = 20261010
COUNT = 8
POLICIES = ("session_llm", "prediction_variance", "fisher_d", "fixed_mixed", "random")
EVAL_TIMES = np.linspace(0, bw.DURATION, 321)
SOURCE_NAMES = ("experiments/bouc_wen_selection.py", "experiments/bouc_wen_pilot.py",
                "packages/dynamics/bouc_wen.py", "tests/test_bouc_wen_selection.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def source_check(out):
    protocol = read(out / "protocol.json")
    for name, expected in protocol["source_hashes"].items():
        if digest(ROOT / name) != expected or digest(out / "sources" / name) != expected:
            raise RuntimeError(f"Source changed after freeze: {name}")


def prepared_check(out):
    source_check(out)
    if (out / "failure.json").exists():
        raise RuntimeError("Failed campaign cannot continue")
    for name, expected in read(out / "prepared.json")["hashes"].items():
        if digest(out / name) != expected:
            raise RuntimeError(f"Prepared artifact changed: {name}")


def jacobian(parameters, profile, times, step):
    """Normalized-parameter sensitivities using accurate reference integration."""
    q = (np.asarray(parameters)-bw.LOWER)/(bw.UPPER-bw.LOWER)
    columns = []
    for j in range(4):
        a, b = q.copy(), q.copy()
        a[j], b[j] = max(0., q[j]-step), min(1., q[j]+step)
        ya = bw.simulate(bw.LOWER+a*(bw.UPPER-bw.LOWER), profile, times, reference=True)[:, 0]
        yb = bw.simulate(bw.LOWER+b*(bw.UPPER-bw.LOWER), profile, times, reference=True)[:, 0]
        columns.append((yb-ya)/(b[j]-a[j]))
    return np.column_stack(columns)


def numerical_scores(parameters, step=.001):
    initial = jacobian(parameters, "baseline", bw.OBS_TIMES, step)
    # 12 I is moment-matched precision for independent normalized uniform priors.
    # Both selectors are local linear approximations, not posterior calculations.
    precision = 12*np.eye(4) + initial.T@initial/pilot.NOISE**2
    covariance = np.linalg.inv(precision)
    target = np.zeros((4, 4))
    for name in bw.RESERVED:
        j = jacobian(parameters, name, EVAL_TIMES, step)
        target += j.T@j/(len(EVAL_TIMES)*len(bw.RESERVED))
    scores = {}
    for name in bw.CANDIDATES:
        j = jacobian(parameters, name, bw.OBS_TIMES, step)
        updated = precision + j.T@j/pilot.NOISE**2
        scores[name] = {
            "fisher_d": float(np.linalg.slogdet(updated)[1]-np.linalg.slogdet(precision)[1]),
            "prediction_variance": float(np.trace(target@(covariance-np.linalg.inv(updated)))),
        }
    choices = {policy: max(bw.CANDIDATES, key=lambda a: scores[a][policy])
               for policy in ("fisher_d", "prediction_variance")}
    return {"scores": scores, "choices": choices}


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    protocol = pilot.protocol()
    protocol.update({
        "study": "normalized_bouc_wen_session_llm_v1", "seed": SEED,
        "worlds": COUNT, "stage": "prospective_development_selector_comparison",
        "policies": POLICIES, "primary_comparison": "session_llm minus fixed_mixed",
        "llm_execution": "Current assistant session; all eight decisions in one batch, development context retained; no independent API calls or model override",
        "llm_model_id": None, "llm_token_usage": None,
        "blinding": "Caller reads only context.json before decisions; evaluator files and numerical recommendations are withheld. Same filesystem is not a security boundary.",
        "prior_context_disclosure": "Session knows original four development cases' aggregate results, simulator/fitter source, candidate and target definitions; no fresh truths or outcomes shown before selection.",
        "selector_information": "Noisy baseline positions, known equations and bounds, three training-only fits and diagnostics, candidate inputs, deployment input definitions",
        "target_input_visibility": "Both LLM and numerical selectors know chirp/nested input definitions; their true responses remain unobserved until final evaluation",
        "selection_rounds": 1, "llm_decisions": COUNT,
        "numerical_rule": "Reference finite differences in normalized coordinates; P=12I+J_baseline.T J_baseline/sigma^2. D maximizes logdet gain; prediction selector maximizes trace(W*(inv(P)-inv(P+F))). W averages target-output Jacobian Gram matrices. Tie order: slow, fast, mixed.",
        "derivative_steps": [.001, .0005],
        "numerical_gate": "Both numerical choices must agree at both steps; no outcome-based retuning",
        "random_seed": SEED+2, "noise_seed": SEED+1,
        "statistics": "Paired case differences and descriptive 95% percentile bootstrap over eight systems, 10000 resamples, seed 20261013; no significance or equivalence claim",
        "cache": "Identical action histories share the same deterministic fit and paired observation noise",
        "go_rule": "No automatic follow-on campaign",
        "limits": ["Eight generated systems, one noise stream each",
                   "One session-conditioned LLM policy, not independent fresh LLM calls",
                   "Known model family and normalized exponent-one adaptation",
                   "Equal data/duration/force RMS; compute and representations differ",
                   "Local numerical uncertainty is uncalibrated",
                   "No LLM or PINN weights are trained"],
    })
    hashes = {}
    for name in SOURCE_NAMES:
        source, target = ROOT/name, out/"sources"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[name] = digest(source)
    protocol["source_hashes"] = hashes
    write(out/"protocol.json", protocol)
    deadline = started+protocol["runtime_limit_s"]
    truth_rng, noise_rng, random_rng = (np.random.default_rng(SEED+i) for i in range(3))
    truths = bw.LOWER+truth_rng.uniform(.15, .85, (COUNT, 4))*(bw.UPPER-bw.LOWER)
    public_cases = []
    for index, truth in enumerate(truths, 1):
        case_id = f"case-{index}"
        private = out/"evaluator"/case_id
        observations = {}
        checks = []
        for name in bw.PROFILES:
            regular = bw.simulate(truth, name, EVAL_TIMES)
            accurate = bw.simulate(truth, name, EVAL_TIMES, reference=True)
            delta = float(np.max(np.abs(regular[:, 0]-accurate[:, 0])))
            checks.append({"profile": name, "position_difference": delta})
            if delta > protocol["numerical_position_tolerance"] or np.max(abs(accurate[:, 2])) > 1+1e-7:
                write(private/"numerical-failure.json", checks)
                raise RuntimeError("New-system numerical preflight failed")
            if name not in bw.RESERVED:
                y = bw.simulate(truth, name, reference=True)[:, 0]
                y += noise_rng.normal(0, pilot.NOISE, len(y))
                observations[name] = {"profile": name, "positions": y.tolist()}
        write(private/"world.json", {"parameters": truth.tolist(), "observations": observations,
                                    "random_choice": str(random_rng.choice(bw.CANDIDATES))})
        write(private/"preflight.json", checks)
        fitted = pilot.fit([observations["baseline"]], deadline=deadline)
        write(out/"baseline-fits"/f"{case_id}.json", fitted)
        if not fitted["best"]["converged"]:
            raise RuntimeError("Baseline selected fit did not converge")
        coarse = numerical_scores(fitted["best"]["parameters"], .001)
        fine = numerical_scores(fitted["best"]["parameters"], .0005)
        write(private/"numerical-selection.json", {"coarse": coarse, "fine": fine})
        if coarse["choices"] != fine["choices"]:
            raise RuntimeError("Numerical selector ranking failed derivative refinement")
        public_cases.append({"case_id": case_id, "observed": observations["baseline"],
                             "fit": {"best": fitted["best"], "starts": fitted["starts"]}})
        print(json.dumps({"prepared": case_id}), flush=True)
    context = {
        "instruction": "Choose ONE candidate per case before any additional data. Minimize expected prediction error on the deployment inputs after the shared continuous fitter uses 128 positions. Use physical reasoning from the baseline and diagnostic fits; do not access evaluator files or compute numerical selector utilities. Return a JSON array with case_id,candidate_id,rationale for every case, one short sentence per rationale. No tools except reading this context. No outcome feedback or retries.",
        "model": {"parameters": bw.PARAMETERS, "bounds": [bw.LOWER.tolist(), bw.UPPER.tolist()],
                  "equations": protocol["equations"], "known": protocol["known"]},
        "noise_std": pilot.NOISE, "times": bw.OBS_TIMES.tolist(),
        "baseline_input": "0.08*sin(2*pi*t/8)",
        "candidate_inputs": {"slow_cycles": "0.8*sin(2*pi*t/8)",
                             "fast_cycles": "0.8*sin(2*pi*t/4)",
                             "mixed_cycles": "0.8/sqrt(3)*(sin(2*pi*t/8)+sin(2*pi*t/4)+sin(2*pi*t/2))"},
        "deployment_inputs": {"chirp": "0.8*sin(2*pi*(0.06*t+0.003*t*t))",
                              "nested_cycles": "0.6*sin(2*pi*t/10)+0.3*sin(2*pi*t/3.3)"},
        "equal_budgets": "All candidates: 16 time units, 64 position observations, force RMS sqrt(0.32), peak below 1.4; reset x=v=w=0",
        "diagnostic_caution": "Fits are based only on gentle baseline, may sit on parameter bounds, and are not a calibrated posterior. Start convergence is optimizer termination, not identification certainty.",
        "cases": public_cases,
    }
    write(out/"context.json", context)
    manifest = {str(f.relative_to(out)): digest(f) for f in sorted(out.rglob("*")) if f.is_file()}
    write(out/"prepared.json", {"hashes": manifest, "elapsed_s": time.monotonic()-started})


def validate_decisions(raw):
    decisions = json.loads(raw)
    if not isinstance(decisions, list) or len(decisions) != COUNT:
        raise ValueError("Exactly eight decisions required")
    if {d.get("case_id") for d in decisions} != {f"case-{i}" for i in range(1, COUNT+1)}:
        raise ValueError("Every case must occur exactly once")
    for d in decisions:
        if set(d) != {"case_id", "candidate_id", "rationale"}:
            raise ValueError("Unexpected decision fields")
        if d["candidate_id"] not in bw.CANDIDATES:
            raise ValueError("Unknown candidate")
        if not isinstance(d["rationale"], str) or not 1 <= len(d["rationale"]) <= 700:
            raise ValueError("A short visible rationale is required")
    return decisions


def record(out, raw):
    prepared_check(out)
    if (out/"evaluation-start.json").exists():
        raise RuntimeError("Cannot record decisions after evaluation starts")
    if not (out/"context-read.json").exists():
        raise RuntimeError("Read the allowed context first")
    decisions = validate_decisions(raw)
    response = out/"llm-response.txt"
    with response.open("x") as f:
        f.write(raw)
    write(out/"decisions.json", {"decisions": decisions, "recorded_at": now(),
                               "raw_response_sha256": digest(response),
                               "context_sha256": digest(out/"context.json"),
                               "execution": "Current conversation assistant authored this exact visible response; no external API or subagent call",
                               "observed_model_id": None, "usage": None,
                               "fresh_context": False, "outcome_feedback": False})


def evaluate(out):
    prepared_check(out)
    recorded = read(out/"decisions.json")
    if digest(out/"llm-response.txt") != recorded["raw_response_sha256"]:
        raise RuntimeError("Raw LLM response changed")
    decisions = validate_decisions((out/"llm-response.txt").read_text())
    if decisions != recorded["decisions"] or digest(out/"context.json") != recorded["context_sha256"]:
        raise RuntimeError("Decision/context integrity failed")
    write(out/"evaluation-start.json", {"started_at": now(), "decision_sha256": digest(out/"decisions.json")})
    started = time.monotonic()
    protocol = read(out/"protocol.json")
    deadline = started+protocol["runtime_limit_s"]-read(out/"prepared.json")["elapsed_s"]
    rows = []
    for decision in decisions:
        case_id = decision["case_id"]
        private = out/"evaluator"/case_id
        world = read(private/"world.json")
        numerical = read(private/"numerical-selection.json")["fine"]["choices"]
        actions = {"session_llm": decision["candidate_id"], **numerical,
                   "fixed_mixed": "mixed_cycles", "random": world["random_choice"]}
        reserved = {name: bw.simulate(world["parameters"], name, EVAL_TIMES, reference=True)[:, 0]
                    for name in bw.RESERVED}
        outcomes = {}
        for action in bw.CANDIDATES:
            if action not in actions.values():
                continue
            history = [world["observations"]["baseline"], world["observations"][action]]
            fitted = pilot.fit(history, deadline=deadline)
            base = out/"fits"/case_id/action
            write(base/"history.json", history)
            write(base/"fit.json", fitted)
            if not fitted["best"]["converged"]:
                raise RuntimeError("Final selected fit did not converge; no cohort ranking")
            best = fitted["best"]["parameters"]
            per_profile = {}
            prediction_record = {}
            for name, truth in reserved.items():
                y = bw.simulate(best, name, EVAL_TIMES, reference=True)[:, 0]
                regular = bw.simulate(best, name, EVAL_TIMES)[:, 0]
                if np.max(abs(regular-y)) > protocol["numerical_position_tolerance"]:
                    raise RuntimeError("Fitted reference agreement failed")
                per_profile[name] = float(np.sqrt(np.mean((y-truth)**2)))
                prediction_record[name] = {"truth": truth.tolist(), "predicted": y.tolist()}
            outcomes[action] = {"reserved_rmse": per_profile,
                                "mean_reserved_rmse": float(np.mean(list(per_profile.values())))}
            write(base/"predictions.json", prediction_record)
        row = {"case_id": case_id, "actions": actions,
               "outcomes": {policy: outcomes[action] for policy, action in actions.items()},
               "positions_per_policy": 128}
        write(out/"case-results"/f"{case_id}.json", row)
        rows.append(row)
        print(json.dumps({"evaluated": case_id}), flush=True)
    means = {policy: float(np.mean([r["outcomes"][policy]["mean_reserved_rmse"] for r in rows]))
             for policy in POLICIES}
    bootstrap = np.random.default_rng(20261013).integers(0, COUNT, (10000, COUNT))
    paired = {}
    for policy in POLICIES[1:]:
        diff = np.array([r["outcomes"]["session_llm"]["mean_reserved_rmse"]
                        -r["outcomes"][policy]["mean_reserved_rmse"] for r in rows])
        paired[policy] = {"case_differences": diff.tolist(), "mean_difference": float(diff.mean()),
                          "descriptive_bootstrap_95pct": np.quantile(diff[bootstrap].mean(axis=1), [.025, .975]).tolist(),
                          "llm_lower": int(np.sum(diff < -1e-12)), "equal": int(np.sum(abs(diff) <= 1e-12)),
                          "llm_higher": int(np.sum(diff > 1e-12))}
    write(out/"results.json", {"status": "complete", "rows": rows, "mean_reserved_rmse": means,
                               "paired_llm_minus_control": paired,
                               "numerical_runtime_s": time.monotonic()-started+read(out/"prepared.json")["elapsed_s"],
                               "interpretation": "Eight-system descriptive session-LLM comparison; no independent fresh model calls, weight training, significance or equivalence claim"})
    print(json.dumps({"status": "complete", "means": means}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "context", "record", "evaluate"))
    parser.add_argument("--out", required=True, type=Path)
    a = parser.parse_args()
    # Refuse accidental re-use before entering failure-recording code, so an
    # already completed campaign cannot be modified by a duplicate invocation.
    if a.command == "prepare" and a.out.exists():
        raise FileExistsError("Prepare requires a fresh directory")
    if a.command == "evaluate" and (a.out/"evaluation-start.json").exists():
        raise FileExistsError("Evaluation is single-use")
    try:
        if a.command == "prepare":
            prepare(a.out)
        elif a.command == "context":
            prepared_check(a.out)
            if (a.out/"evaluation-start.json").exists():
                raise RuntimeError("Evaluation already started")
            write(a.out/"context-read.json", {"read_at": now(), "sha256": digest(a.out/"context.json")})
            print((a.out/"context.json").read_text())
        elif a.command == "record":
            record(a.out, sys.stdin.read())
        else:
            evaluate(a.out)
    except Exception as exc:
        # Preserve failure; never resume a partially evaluated cohort as though
        # the successfully processed subset were the whole experiment.
        if a.command in ("prepare", "evaluate") and a.out.exists() and not (a.out/"failure.json").exists():
            write(a.out/"failure.json", {"type": type(exc).__name__, "message": str(exc),
                                       "stage": a.command, "aggregate_ranking": "withheld"})
        raise


if __name__ == "__main__":
    main()
