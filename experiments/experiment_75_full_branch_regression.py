"""Experiment 75: full branch regression - revision 2.

Fixes Experiment 69 discovery ambiguity by preferring canonical regression
experiments and excluding helper/diagnostic scripts from automatic selection.

No Groq/network calls are intentionally made by this runner.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

REQUIRED_NUMBERS = [
    60,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
]

OPTIONAL_NUMBERS = [61]

# Canonical names for experiments where helper/diagnostic files may share
# the same numeric prefix.
CANONICAL_FILENAMES = {
    60: "experiment_60_retrieval_regression.py",
    62: "experiment_62_answer_type_inference.py",
    63: "experiment_63_candidate_type_validation.py",
    64: "experiment_64_confidence_scoring.py",
    65: "experiment_65_service_type_confidence.py",
    66: "experiment_66_typed_confidence_regression.py",
    67: "experiment_67_layout_analysis.py",
    68: "experiment_68_row_segmentation_refinement.py",
    69: "experiment_69_table_reconstruction.py",
    70: "experiment_70_form_reconstruction.py",
    71: "experiment_71_structure_classification.py",
    72: "experiment_72_unified_structured_document.py",
    73: "experiment_73_structured_qa_integration.py",
    74: "experiment_74_production_integration.py",
}

HELPER_MARKERS = {
    "diagnostic",
    "debug",
    "probe",
    "inspect",
    "inspection",
    "helper",
    "scratch",
    "testdata",
}


def is_helper(path: Path) -> bool:
    name = path.stem.casefold()
    return any(marker in name for marker in HELPER_MARKERS)


def find_experiment(number: int) -> Path | None:
    canonical_name = CANONICAL_FILENAMES.get(number)

    if canonical_name:
        canonical = EXPERIMENTS_DIR / canonical_name
        if canonical.is_file():
            return canonical

    matches = sorted(
        path
        for path in EXPERIMENTS_DIR.glob(f"experiment_{number}_*.py")
        if path.name != Path(__file__).name
    )

    if not matches:
        return None

    non_helpers = [
        path for path in matches
        if not is_helper(path)
    ]

    if len(non_helpers) == 1:
        return non_helpers[0]

    if not non_helpers:
        non_helpers = matches

    # Prefer production/regression-oriented names over revision/archive copies.
    preferred_markers = (
        "regression",
        "integration",
        "reconstruction",
        "classification",
        "analysis",
        "refinement",
        "inference",
        "validation",
        "confidence",
        "structured",
    )

    scored = []
    for path in non_helpers:
        name = path.stem.casefold()
        score = sum(
            1 for marker in preferred_markers
            if marker in name
        )
        if "revision" in name:
            score -= 1
        if "old" in name or "backup" in name:
            score -= 2
        scored.append((score, path))

    scored.sort(
        key=lambda item: (
            -item[0],
            item[1].name,
        )
    )

    if len(scored) == 1:
        return scored[0][1]

    if scored[0][0] > scored[1][0]:
        return scored[0][1]

    names = ", ".join(path.name for _, path in scored)
    raise RuntimeError(
        f"Experiment {number} is still ambiguous. Found: {names}"
    )


def module_name(path: Path) -> str:
    relative = path.relative_to(PROJECT_ROOT).with_suffix("")
    return ".".join(relative.parts)


def run_one(number: int, path: Path) -> tuple[bool, int]:
    module = module_name(path)

    print()
    print("=" * 118)
    print(f"RUNNING EXPERIMENT {number}: {path.name}")
    print("=" * 118)

    completed = subprocess.run(
        [sys.executable, "-m", module],
        cwd=PROJECT_ROOT,
        check=False,
    )

    passed = completed.returncode == 0

    print()
    print(
        f"[{'PASS' if passed else 'FAIL'}] "
        f"Experiment {number} | exit code {completed.returncode}"
    )

    return passed, completed.returncode


def main() -> int:
    print("=" * 118)
    print("EXPERIMENT 75: FULL TYPED-CONFIDENCE + AUTOMATIC-STRUCTURE REGRESSION - REVISION 2")
    print("=" * 118)
    print(f"Python: {sys.executable}")
    print(f"Project: {PROJECT_ROOT}")

    required_paths: dict[int, Path] = {}

    for number in REQUIRED_NUMBERS:
        path = find_experiment(number)
        if path is None:
            print(f"[MISSING] Required experiment {number}")
            return 1
        required_paths[number] = path

    optional_paths: dict[int, Path] = {}

    for number in OPTIONAL_NUMBERS:
        path = find_experiment(number)
        if path is not None:
            optional_paths[number] = path

    print()
    print("-" * 118)
    print("DISCOVERED REGRESSION SUITES")
    print("-" * 118)

    for number in sorted(required_paths):
        print(
            f"Exp {number:02d}: "
            f"{required_paths[number].name}"
        )

    for number in sorted(optional_paths):
        print(
            f"Exp {number:02d}: "
            f"{optional_paths[number].name} "
            f"(optional)"
        )

    run_order = [
        (
            number,
            required_paths.get(number)
            or optional_paths[number],
        )
        for number in sorted(
            set(required_paths)
            | set(optional_paths)
        )
    ]

    results = []

    for number, path in run_order:
        passed, returncode = run_one(number, path)
        results.append(
            {
                "number": number,
                "file": path.name,
                "passed": passed,
                "returncode": returncode,
                "required": number in REQUIRED_NUMBERS,
            }
        )

    print()
    print("=" * 118)
    print("FULL REGRESSION SUMMARY")
    print("=" * 118)

    for item in results:
        requirement = (
            "required"
            if item["required"]
            else "optional"
        )
        print(
            f"[{'PASS' if item['passed'] else 'FAIL'}] "
            f"Exp {item['number']:02d} | "
            f"{item['file']} | "
            f"{requirement}"
        )

    required_results = [
        item for item in results
        if item["required"]
    ]

    required_passed = sum(
        1
        for item in required_results
        if item["passed"]
    )

    all_required_passed = (
        required_passed
        == len(required_results)
    )

    print()
    print(
        f"Required suites passed: "
        f"{required_passed}/{len(required_results)}"
    )

    if optional_paths:
        optional_results = [
            item for item in results
            if not item["required"]
        ]
        optional_passed = sum(
            1
            for item in optional_results
            if item["passed"]
        )
        print(
            f"Optional suites passed: "
            f"{optional_passed}/{len(optional_results)}"
        )

    print(
        "No Groq requests are intentionally made by this regression runner."
    )

    print()
    print("=" * 118)
    print(
        "OVERALL: "
        + (
            "PASS - all required regression suites passed"
            if all_required_passed
            else "FAIL - one or more required regression suites failed"
        )
    )
    print("=" * 118)

    return 0 if all_required_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
