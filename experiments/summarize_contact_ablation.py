"""Verify and summarize paired contact-entropy ablations from frozen campaigns."""

import argparse
import hashlib
import json
from pathlib import Path

from research_loop import scientific


def read(path):
    return json.loads(path.read_text())


def verified_campaign(root):
    manifest = read(root/"manifest.json")
    for name, expected in manifest["files"].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"artifact changed: {root.name}/{name}")
    entries = manifest["proposals"]
    if len(entries) != 1:
        raise ValueError("comparison requires exactly one experiment proposal per campaign")
    entry = entries[0]
    proposal = read(root/entry["proposal"])
    summary = read(root/entry["summary"])
    if summary["completed_cases"] != summary["expected_cases"]:
        raise ValueError("incomplete campaign: preserve failure ledger; do not rank its completed subset")
    cases = {}
    for job in manifest["jobs"]:
        if job.get("kind") == "case":
            if job["status"] != "completed":
                raise ValueError("failed case cannot be dropped from an ablation")
            result = read(root/job["output"])
            key = (result["world"], result["measurement_time_s"])
            if key in cases:
                raise ValueError("duplicate case")
            cases[key] = result
    if len(cases) != summary["expected_cases"]:
        raise ValueError("case ledger does not match summary")
    return manifest, read(root/"config.json"), proposal, summary, cases


def compare(original, ablated):
    om, oc, op, os, old = verified_campaign(original)
    am, ac, ap, ass, new = verified_campaign(ablated)
    if oc != ac or read(original/"environment.json") != read(ablated/"environment.json"):
        raise ValueError("numerical configurations and environments must match")
    sources = lambda manifest: {k: v for k, v in manifest["files"].items() if k.startswith("source/")}
    if sources(om) != sources(am):
        raise ValueError("source snapshots differ")
    retained = [t for t in op["formula"] if t["feature"] != "contact_entropy"]
    removed = [t for t in op["formula"] if t["feature"] == "contact_entropy"]
    if ap["formula"] != retained or removed != [{"weight": .25, "feature": "contact_entropy", "transform": "identity"}]:
        raise ValueError("proposal must remove only the declared contact-entropy term")
    if any(op[key] != ap[key] for key in ("measurement_times_s", "noise_std_m")) or old.keys() != new.keys():
        raise ValueError("experiment conditions or cases differ")
    results = []
    for (world, t), before in sorted(old.items()):
        after = new[(world, t)]
        for level in ("base", "refined"):
            a = before if level == "base" else before["refinement"]
            b = after if level == "base" else after["refinement"]
            if a["policies"].keys() != b["policies"].keys():
                raise ValueError("policies differ")
            for policy in a["policies"]:
                if policy != "proposed" and scientific(a["policies"][policy]) != scientific(b["policies"][policy]):
                    raise ValueError(f"baseline changed: world {world}, time {t}, {level}, {policy}")
        a, b = before["policies"]["proposed"], after["policies"]["proposed"]
        changed = [i for i in range(1, 6) if a[i]["u"] != b[i]["u"]]
        first = None
        if changed:
            i = changed[0]
            if (scientific(a[0]) != scientific(b[0]) or a[i]["diagnostics"] != b[i]["diagnostics"]
                    or any(a[j]["measurement_m"] != b[j]["measurement_m"] for j in range(1, i))):
                raise ValueError("first differing action does not share the same prior observations and diagnostics")
            first = {"round": i, "original_u": a[i]["u"], "ablated_u": b[i]["u"],
                     "original_scores": a[i]["scores"], "ablated_scores": b[i]["scores"],
                     "diagnostics": a[i]["diagnostics"]}
        entropy = [max(d["features"]["contact_entropy"] for d in r["diagnostics"]) for r in a[1:]]
        results.append({"seed": oc["seed"], "world": world, "measurement_time_s": t,
                        "original_rmse_m": a[-1]["mean_heldout_rmse_m"],
                        "ablated_rmse_m": b[-1]["mean_heldout_rmse_m"],
                        "difference_m": b[-1]["mean_heldout_rmse_m"]-a[-1]["mean_heldout_rmse_m"],
                        "original_curve_m": [r["mean_heldout_rmse_m"] for r in a],
                        "ablated_curve_m": [r["mean_heldout_rmse_m"] for r in b],
                        "original_choices": [r["u"] for r in a[1:]],
                        "ablated_choices": [r["u"] for r in b[1:]],
                        "first_difference": first,
                        "decisions_with_nonzero_contact_bonus": sum(e > 1e-12 for e in entropy),
                        "max_candidate_contact_entropy": max(entropy),
                        "local_d_rmse_m": before["policies"]["local_d"][-1]["mean_heldout_rmse_m"],
                        "eig_rmse_m": before["policies"]["eig"][-1]["mean_heldout_rmse_m"]})
    audits = [os["numerical_audit"], ass["numerical_audit"]]
    stable = all(a["all_choices_identical_on_refinement"] and a["all_rmse_within_refinement_tolerance"] for a in audits)
    return {"original_campaign": original.name, "ablated_campaign": ablated.name,
            "seed": oc["seed"], "baselines_identical": True, "source_and_configuration_match": True,
            "numerically_comparable": stable, "audits": audits, "cases": results,
            "original_manifest_sha256": hashlib.sha256((original/"manifest.json").read_bytes()).hexdigest(),
            "ablated_manifest_sha256": hashlib.sha256((ablated/"manifest.json").read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    pairs = [compare(*pair) for pair in args.pair]
    cases = [c for pair in pairs for c in pair["cases"]]
    if len({(c["seed"], c["world"], c["measurement_time_s"]) for c in cases}) != len(cases):
        raise ValueError("duplicate paired scenarios across campaigns")
    groups = []
    for seed in (*sorted({c["seed"] for c in cases}), "pooled"):
        for t in sorted({c["measurement_time_s"] for c in cases}):
            selected = [c for c in cases if c["measurement_time_s"] == t and (seed == "pooled" or c["seed"] == seed)]
            means = {key: sum(c[key] for c in selected)/len(selected)
                     for key in ("original_rmse_m", "ablated_rmse_m", "difference_m", "local_d_rmse_m", "eig_rmse_m")}
            groups.append({"seed": seed, "measurement_time_s": t, "worlds": len(selected), **means,
                           "changed_sequences": sum(c["original_choices"] != c["ablated_choices"] for c in selected),
                           "decisions_with_nonzero_contact_bonus": sum(c["decisions_with_nonzero_contact_bonus"] for c in selected),
                           "removal_helps_cases": sum(c["difference_m"] < -1e-12 for c in selected),
                           "removal_hurts_cases": sum(c["difference_m"] > 1e-12 for c in selected)})
    result = {"stage": "development", "pairs": pairs, "groups": groups,
              "numerically_comparable": all(p["numerically_comparable"] for p in pairs),
              "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "changed_case_count": sum(c["original_choices"] != c["ablated_choices"] for c in cases),
              "case_count": len(cases)}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/"analysis-source.py").write_bytes(Path(__file__).read_bytes())
    (args.out/"comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    lines = ["# Contact-entropy ablation", "", "Development evidence; negative difference means removal helped.",
             "", f"Numerically comparable: {result['numerically_comparable']}. Changed sequences: {result['changed_case_count']}/{len(cases)}.",
             "", "| Seed | Time (s) | With entropy (mm) | Without entropy (mm) | Difference (mm) | Changed sequences |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in groups:
        lines.append(f"| {row['seed']} | {row['measurement_time_s']:.2f} | {row['original_rmse_m']*1000:.6f} | {row['ablated_rmse_m']*1000:.6f} | {row['difference_m']*1000:.6f} | {row['changed_sequences']}/{row['worlds']} |")
    (args.out/"comparison.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
