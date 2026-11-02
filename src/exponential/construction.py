"""Reproducible h=3 construction using the existing exact rejection predicate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
from time import perf_counter

import numpy as np

from core import ring8
from verification.rejection import self_test, verify_current_algorithm

ROOT = Path(__file__).resolve().parents[2]
TARGETS = (("q48", 16, 4, 5), ("q90", 30, 8, 7))
DEFAULT_SEED = 202610011


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_generated(path, value, overwrite=False, allow_manifest=False):
    path = Path(path).resolve()
    if path.is_relative_to((ROOT / "data").resolve()):
        raise ValueError("The deposited data directory is read-only; choose an output elsewhere.")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}; choose another path or use --overwrite.")
    if allow_manifest and path != ROOT / "MANIFEST.json":
        raise ValueError("The release manifest is written to MANIFEST.json at the repository root.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def accepted_candidate(m, k, d, rng, max_trials):
    # finite_bound also checks the target parameter range before sampling.
    ring8.finite_bound(m, k, d)
    if max_trials < 1:
        raise ValueError("max_trials must be positive")
    for trial in range(1, max_trials + 1):
        b, a = ring8.sample_candidate(m, rng)
        check = verify_current_algorithm(b, a, k, d)
        if check["passed"]:
            # Also check the inverse-column logical phase identities.
            ring8.make_instance(b, a, k, d, check)
            return b, a, trial
    raise RuntimeError(f"No candidate passed after {max_trials} trials for M={m}, K={k}, d={d}.")


def published_record(name, m, k, d, b, a, seed, max_trials, trial):
    # D records the manuscript parameter. Rejection proves the lower bound;
    # this command does not perform a minimum-weight logical-operator search.
    return {
        "schema_version": 1, "name": name.upper(), "h": 3,
        "N": 3*m, "M": m, "r": m-k, "K": k, "D": d,
        "B_mod8": b, "A_mod8": a,
        "construction": {
            "seed": seed, "example_order": [x[0] for x in TARGETS],
            "max_trials_per_example": max_trials, "accepted_trial": trial,
            "target_distance": d,
        },
    }


def reproduce(output_dir, seed=DEFAULT_SEED, max_trials=10000, overwrite=False):
    rng = random.Random(seed)
    summary = []
    for name, m, k, d in TARGETS:
        start = perf_counter()
        b, a, trial = accepted_candidate(m, k, d, rng, max_trials)
        record = published_record(name, m, k, d, b, a, seed, max_trials, trial)
        saved = read_json(ROOT / "data" / "finite" / name / "code.json")
        matrices_match = b == saved["B_mod8"] and a == saved["A_mod8"]
        if seed == DEFAULT_SEED and not matrices_match:
            raise AssertionError(f"The recorded seed did not reproduce {name}'s matrices.")
        write_generated(Path(output_dir) / name / "code.json", record, overwrite)
        summary.append({
            "name": name, "accepted_trial": trial,
            "distance_lower_bound_checked": d,
            "matches_deposited_matrices": matrices_match,
            "seconds": round(perf_counter()-start, 6),
        })
    return {"passed": True, "seed": seed, "examples": summary}


def verify(input_dir):
    summary = []
    for name, m, k, d in TARGETS:
        record = read_json(Path(input_dir) / name / "code.json")
        expected = {"h": 3, "N": 3*m, "M": m, "r": m-k, "K": k, "D": d}
        if any(record[key] != value for key, value in expected.items()):
            raise ValueError(f"Unexpected parameters for {name}")
        start = perf_counter()
        b, a = record["B_mod8"], record["A_mod8"]
        check = verify_current_algorithm(b, a, k, d)
        if not check["passed"]:
            raise AssertionError(f"{name} failed the rejection test: {check}")
        ring8.make_instance(b, a, k, d, check)
        bound = ring8.finite_bound(m, k, d)
        summary.append({
            "name": name, "passed": True, "distance_lower_bound_checked": d,
            "matrix_and_addressing_identities_checked": True,
            "ensemble_failure_bound_exact": str(bound),
            "ensemble_sufficient_condition_holds": bound < 1,
            "seconds": round(perf_counter()-start, 6),
        })
    return {"passed": True, "python": platform.python_version(),
            "numpy": np.__version__, "examples": summary}


def custom_record(m, k, d, b, a, sampling):
    return {"schema_version": 1, "h": 3, "N": 3*m, "M": m,
            "r": m-k, "K": k, "distance_lower_bound": d,
            "B_mod8": b, "A_mod8": a, "construction": sampling}


def manifest():
    excluded = {".git", "__pycache__", ".venv", "generated"}
    files = {}
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if not path.is_file() or excluded.intersection(rel.parts) or rel.as_posix() == "MANIFEST.json":
            continue
        files[rel.as_posix()] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "bytes": path.stat().st_size}
    return {"algorithm": "sha256", "excludes": sorted(excluded) + ["MANIFEST.json"],
            "files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("reproduce", help="Regenerate q48, then q90, using one shared seeded PRNG.")
    p.add_argument("--output-dir", type=Path, default=ROOT / "generated" / "reproduced")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-trials", type=int, default=10000)
    p.add_argument("--overwrite", action="store_true")
    p = commands.add_parser("verify", help="Run the full construction acceptance predicate on both saved examples.")
    p.add_argument("--input-dir", type=Path, default=ROOT / "data" / "finite")
    p.add_argument("--output", type=Path)
    p.add_argument("--overwrite", action="store_true")
    for command in ("search", "exhaustive"):
        p = commands.add_parser(command, help="Bounded random search." if command == "search" else "Literal deterministic enumeration; use tiny targets.")
        p.add_argument("--M", type=int, required=True)
        p.add_argument("--K", type=int, required=True)
        p.add_argument("--distance", type=int, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--overwrite", action="store_true")
        if command == "search":
            p.add_argument("--seed", type=int)
            p.add_argument("--max-trials", type=int, default=100)
    commands.add_parser("self-test", help="Cross-check exact ring reductions and the accelerated test against literal enumeration.")
    commands.add_parser("manifest", help="Refresh the release SHA-256 manifest.")
    args = parser.parse_args()
    if args.command == "reproduce":
        result = reproduce(args.output_dir, args.seed, args.max_trials, args.overwrite)
    elif args.command == "verify":
        result = verify(args.input_dir)
        if args.output is not None:
            write_generated(args.output, result, args.overwrite)
    elif args.command == "search":
        rng = random.SystemRandom() if args.seed is None else random.Random(args.seed)
        b, a, trial = accepted_candidate(args.M, args.K, args.distance, rng, args.max_trials)
        result = custom_record(args.M, args.K, args.distance, b, a,
                               {"mode": "random", "seed": args.seed,
                                "max_trials": args.max_trials, "accepted_trial": trial})
        write_generated(args.output, result, args.overwrite)
    elif args.command == "exhaustive":
        ring8.finite_bound(args.M, args.K, args.distance)
        obj = ring8.exhaustive(args.M, args.K, args.distance)
        result = custom_record(args.M, args.K, args.distance, obj["U_mod8"], obj["A_mod8"],
                               {"mode": "lexicographic exhaustive enumeration"})
        write_generated(args.output, result, args.overwrite)
    elif args.command == "self-test":
        result = self_test()
    else:
        result = manifest()
        write_generated(ROOT / "MANIFEST.json", result, overwrite=True, allow_manifest=True)
    print(json.dumps(result, indent=2))
