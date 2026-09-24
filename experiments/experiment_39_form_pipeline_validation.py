"""
Experiment 39: pic13 Form Pipeline Validation

Validates the unseen-form pipeline for pic13.

Checks:
- required artifacts exist
- OCR region count
- Experiment 37 checkbox reconstruction
- Experiment 38 structured form fields
- checkbox states
- free-text association
- semantic role counts

This experiment does not rerun OCR or modify previous results.
"""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PIC13 = (
    PROJECT_ROOT
    / "results"
    / "parsing"
    / "pic13"
)

GUI13 = (
    PROJECT_ROOT
    / "results"
    / "gui_v4"
    / "single"
    / "pic13"
)


FILES = {
    "scan": (
        GUI13
        / "scan"
        / "scan.png"
    ),

    "ocr_csv": (
        GUI13
        / "ocr"
        / "results.csv"
    ),

    "ocr_text": (
        GUI13
        / "ocr"
        / "text.txt"
    ),

    "experiment_36": (
        PIC13
        / "form_layout_baseline"
        / "structured_form_baseline.json"
    ),

    "experiment_37": (
        PIC13
        / "refined_form"
        / "refined_form_structure.json"
    ),

    "experiment_38": (
        PIC13
        / "semantic_form"
        / "semantic_form_structure.json"
    ),

    "experiment_38_report": (
        PIC13
        / "semantic_form"
        / "semantic_form_report.txt"
    ),
}


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def check_files():
    print("=" * 72)
    print("ARTIFACT CHECK")
    print("=" * 72)

    missing = []

    for name, path in FILES.items():
        exists = path.is_file()

        print(
            f"{'[OK]' if exists else '[MISSING]':<10}"
            f"{name:<24}"
            f"{path}"
        )

        if not exists:
            missing.append(name)

    return missing


def validate_ocr():
    exp36 = load_json(
        FILES["experiment_36"]
    )

    errors = []

    if (
        exp36["statistics"][
            "ocr_regions"
        ]
        != 37
    ):
        errors.append(
            "Expected 37 OCR regions."
        )

    if (
        exp36["image"]["width"]
        != 1144
    ):
        errors.append(
            "Unexpected image width."
        )

    if (
        exp36["image"]["height"]
        != 1649
    ):
        errors.append(
            "Unexpected image height."
        )

    return errors


def validate_exp37_controls():
    document = load_json(
        FILES["experiment_37"]
    )

    controls = document.get(
        "checkboxes",
        [],
    )

    errors = []

    if len(controls) != 7:
        errors.append(
            "Expected 7 refined checkbox controls."
        )

    checked = [
        item
        for item in controls
        if item["checked"]
    ]

    unchecked = [
        item
        for item in controls
        if not item["checked"]
    ]

    if len(checked) != 3:
        errors.append(
            "Expected 3 checked controls."
        )

    if len(unchecked) != 4:
        errors.append(
            "Expected 4 unchecked controls."
        )

    expected_states = {
        "Limited Information (go to question 2B)": True,
        "Any Information (go to question 3)": False,
        "Medicare eligibility": False,
        "Medicare claims": True,
        "Plan enrollment (e.g. drug or MA Plan)": False,
        "Premium payments": False,
        (
            "Other (Write any other information you want "
            "shared below. For example, payment information)"
        ): True,
    }

    actual = {
        item["label"]:
        bool(item["checked"])
        for item in controls
    }

    for label, expected in (
        expected_states.items()
    ):
        if label not in actual:
            errors.append(
                f"Missing checkbox: {label}"
            )

            continue

        if actual[label] != expected:
            errors.append(
                f"Incorrect checkbox state for: {label}"
            )

    return errors


def validate_fields():
    document = load_json(
        FILES["experiment_38"]
    )

    fields = document.get(
        "fields",
        [],
    )

    errors = []

    if len(fields) != 7:
        errors.append(
            "Expected 7 structured fields."
        )

    actual = {
        field["label"]:
        field["value"]
        for field in fields
    }

    expected = {
        (
            "Name "
            "(First, Middle, Last, Suffix)"
        ): "Alex Carter",

        (
            "Date of Birth "
            "(mm/dd/yyyy)"
        ): "03/14/1992",

        (
            "Medicare Identification "
            "Number"
        ): "1EG4-TE5-MK72",

        "Street Address": (
            "101 Sample Ridge Drive, "
            "Suite 8"
        ),

        "Zip code": "78701",

        "State": "TX",

        "City": "Austin",
    }

    for label, expected_value in (
        expected.items()
    ):
        if label not in actual:
            errors.append(
                f"Missing field: {label}"
            )

            continue

        if (
            actual[label]
            != expected_value
        ):
            errors.append(
                f"Unexpected value for "
                f"{label}: "
                f"{actual[label]!r}"
            )

    return errors


def validate_followup():
    document = load_json(
        FILES["experiment_38"]
    )

    followups = document.get(
        "followup_text",
        [],
    )

    errors = []

    if len(followups) != 1:
        errors.append(
            "Expected exactly one "
            "follow-up text region."
        )

        return errors

    item = followups[0]

    if (
        item["text_region_id"]
        != 36
    ):
        errors.append(
            "Expected free-text response "
            "to come from region 36."
        )

    expected_text = (
        "Coverage questions and claims "
        "history for document handling demo"
    )

    if item["text"] != expected_text:
        errors.append(
            "Unexpected free-text response."
        )

    if not item[
        "control_label"
    ].startswith("Other"):
        errors.append(
            "Free-text response is not "
            "associated with Other."
        )

    return errors


def validate_role_counts():
    document = load_json(
        FILES["experiment_38"]
    )

    roles = document.get(
        "region_roles",
        [],
    )

    counts = {}

    for item in roles:
        role = item["role"]

        counts[role] = (
            counts.get(
                role,
                0,
            )
            + 1
        )

    expected_counts = {
        "checkbox_label": 7,
        "field_label": 7,
        "field_value": 7,
        "free_text_response": 1,
        "instruction_or_section": 12,
        "unclassified_text": 3,
    }

    errors = []

    for role, expected in (
        expected_counts.items()
    ):
        actual = counts.get(
            role,
            0,
        )

        if actual != expected:
            errors.append(
                f"Role count mismatch for "
                f"{role}: "
                f"{actual} != {expected}"
            )

    total = sum(
        counts.values()
    )

    if total != 37:
        errors.append(
            f"Region role total is "
            f"{total}, expected 37."
        )

    return errors


def validate_statistics():
    document = load_json(
        FILES["experiment_38"]
    )

    stats = document[
        "statistics"
    ]

    errors = []

    expected = {
        "ocr_regions": 37,
        "fields": 7,
        "checkboxes": 7,
        "checked_checkboxes": 3,
        "unchecked_checkboxes": 4,
        "followup_text_regions": 1,
    }

    for key, expected_value in (
        expected.items()
    ):
        actual = stats.get(key)

        if actual != expected_value:
            errors.append(
                f"{key}: "
                f"{actual} != "
                f"{expected_value}"
            )

    return errors


def print_validation(
    title,
    errors,
):
    print()
    print("-" * 72)
    print(title)
    print("-" * 72)

    if not errors:
        print("[PASS]")
        return True

    print("[FAIL]")

    for error in errors:
        print(
            f"  - {error}"
        )

    return False


def main():
    print("=" * 72)
    print(
        "EXPERIMENT 39: "
        "PIC13 FORM PIPELINE VALIDATION"
    )
    print("=" * 72)

    missing = check_files()

    if missing:
        print()
        print(
            "Cannot continue because "
            "required artifacts are missing:"
        )

        for name in missing:
            print(
                f"  - {name}"
            )

        raise SystemExit(1)

    checks = [
        (
            "OCR baseline",
            validate_ocr(),
        ),
        (
            "Refined checkbox controls",
            validate_exp37_controls(),
        ),
        (
            "Structured form fields",
            validate_fields(),
        ),
        (
            "Follow-up free text",
            validate_followup(),
        ),
        (
            "Semantic region roles",
            validate_role_counts(),
        ),
        (
            "Final form statistics",
            validate_statistics(),
        ),
    ]

    passed = 0

    for title, errors in checks:
        if print_validation(
            title,
            errors,
        ):
            passed += 1

    print()
    print("=" * 72)
    print("FORM PIPELINE SUMMARY")
    print("=" * 72)

    print(
        f"Validation groups passed: "
        f"{passed}/{len(checks)}"
    )

    if passed == len(checks):
        print(
            "STATUS: PIC13 FORM "
            "PIPELINE CHECKPOINT PASSED"
        )

    else:
        print(
            "STATUS: PIC13 FORM "
            "PIPELINE CHECKPOINT FAILED"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()