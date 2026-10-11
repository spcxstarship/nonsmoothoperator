"""Small declarative language for research proposals; never evaluates model code."""

import math
import re


FEATURES = ("measurement_snr", "contact_entropy", "mean_fisher_trace", "local_trace")
TRANSFORMS = ("identity", "log1p", "sqrt")
TIMES = (0.38, 0.46, 0.60)
NOISE = (0.01, 0.02, 0.05)
PROMPT_VERSION = "ball-research-v1"
INSTRUCTIONS = """You are the research controller for a DEVELOPMENT study of hidden
impact identification. Propose one bounded numerical investigation or stop.
You may choose measurement regimes, sensor noise, and a selection formula built
from the declared features. Physics, inference, baselines, evaluator and resource
limits are fixed. Formula novelty and scientific superiority are not presumed.
Use previous evidence, seek disconfirming outcomes, and replicate useful results.
Return only one JSON object matching the supplied schema. Do not return Python,
shell commands, Markdown fences, secrets, or private information. A successful
run is not proof of the hypothesis. All cases and failures will be retained.
"""


def validate(proposal):
    fields = {"schema_version", "id", "action", "hypothesis", "mechanism",
              "disconfirming_observation", "measurement_times_s", "noise_std_m",
              "formula", "replicates"}
    if not isinstance(proposal, dict) or set(proposal) != fields:
        raise ValueError("proposal must have exactly the documented fields")
    if type(proposal["schema_version"]) is not int or proposal["schema_version"] != 1:
        raise ValueError("unsupported proposal schema")
    if not isinstance(proposal["id"], str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", proposal["id"]):
        raise ValueError("invalid proposal id")
    if proposal["action"] not in ("experiment", "stop"):
        raise ValueError("action must be experiment or stop")
    for field in ("hypothesis", "mechanism", "disconfirming_observation"):
        if not isinstance(proposal[field], str) or not 1 <= len(proposal[field].strip()) <= 1200:
            raise ValueError(f"invalid {field}")
    times = proposal["measurement_times_s"]
    if not isinstance(times, list) or not times or any(type(t) not in (int, float) or t not in TIMES for t in times):
        raise ValueError("measurement times must come from the allowed set")
    if times != sorted(set(times)):
        raise ValueError("measurement times must be unique and increasing")
    if type(proposal["noise_std_m"]) not in (int, float) or proposal["noise_std_m"] not in NOISE:
        raise ValueError("noise must come from the allowed set")
    if proposal["replicates"] is not None and (
        not isinstance(proposal["replicates"], str)
        or not re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", proposal["replicates"])
    ):
        raise ValueError("replicates must be a proposal id or null")
    formula = proposal["formula"]
    if not isinstance(formula, list) or not 1 <= len(formula) <= 4:
        raise ValueError("formula needs one to four terms")
    for term in formula:
        if not isinstance(term, dict) or set(term) != {"weight", "feature", "transform"}:
            raise ValueError("invalid formula term")
        w = term["weight"]
        if type(w) not in (int, float) or not math.isfinite(w) or abs(w) > 10:
            raise ValueError("formula weight must be finite and within [-10,10]")
        if term["feature"] not in FEATURES or term["transform"] not in TRANSFORMS:
            raise ValueError("unknown formula feature or transform")
    if not any(term["weight"] != 0 for term in formula):
        raise ValueError("formula cannot have only zero weights")
    return proposal


def score(formula, features):
    transforms = {"identity": lambda x: x, "log1p": math.log1p, "sqrt": math.sqrt}
    value = sum(term["weight"] * transforms[term["transform"]](features[term["feature"]])
                for term in formula)
    if not math.isfinite(value):
        raise ValueError("nonfinite formula score")
    return value


def prompt(history, remaining, example):
    return {
        "prompt_version": PROMPT_VERSION, "instructions": INSTRUCTIONS,
        "proposal_example": example,
        "features": {
            "measurement_snr": "posterior variance of predicted height / sensor variance",
            "contact_entropy": "binary entropy in nats of posterior contact probability at measurement",
            "mean_fisher_trace": "posterior mean of trace of normalized, noise-scaled candidate Fisher matrix",
            "local_trace": "local candidate Fisher trace weighted by inverse accumulated diagonal precision",
        },
        "formula_language": "sum of 1–4 weighted terms; each transforms one feature",
        "transforms": list(TRANSFORMS), "weight_bounds": [-10, 10],
        "allowed_measurement_times_s": list(TIMES), "allowed_noise_std_m": list(NOISE),
        "fixed_protocol": "4 shared prior-drawn worlds per regime, 3 initial observations, 5 new experiments; six fixed numerical baselines; shared grid inference; held-out position RMSE",
        "replication": "Set replicates to an earlier proposal ID and preserve its settings/formula. Reuses data seeds to verify computational repeatability, not independent scientific replication.",
        "remaining_budget": remaining, "history": history,
        "stop": "Set action to stop (other schema fields still required).",
    }
