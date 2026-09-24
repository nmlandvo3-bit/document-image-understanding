"""
Experiment 38: Form Semantic Role Cleanup

Refines Experiment 37 by separating:
- ordinary form labels
- ordinary form values
- checkbox labels
- section/question text
- follow-up free text

Goals:
- Recover Name -> Alex Carter
- Preserve Date of Birth, Medicare ID, Address, City, State, ZIP
- Prevent checkbox options from becoming fake key/value fields
- Prevent section/question lines from becoming values
- Preserve Experiment 37 checkbox detections and states

No OCR is rerun.
"""

import csv
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OCR_CSV_PATH = (
    PROJECT_ROOT
    / "results"
    / "gui_v4"
    / "single"
    / "pic13"
    / "ocr"
    / "results.csv"
)

EXP37_JSON = (
    PROJECT_ROOT
    / "results"
    / "parsing"
    / "pic13"
    / "refined_form"
    / "refined_form_structure.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "parsing"
    / "pic13"
    / "semantic_form"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "semantic_form_structure.json"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "semantic_form_report.txt"
)


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_ocr_regions():
    if not OCR_CSV_PATH.is_file():
        raise FileNotFoundError(
            OCR_CSV_PATH
        )

    regions = []

    with OCR_CSV_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            x = int(row["x"])
            y = int(row["y"])
            w = int(row["w"])
            h = int(row["h"])

            regions.append(
                {
                    "region_id": int(
                        row["region"]
                    ),
                    "text": row["text"],
                    "confidence": float(
                        row["confidence"]
                    ),
                    "bbox": {
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "x2": x + w,
                        "y2": y + h,
                    },
                }
            )

    return regions


def strip_leading_numbering(text):
    """
    Remove leading form question numbering for semantic tests.

    Examples:
        "1. Name (...)" -> "Name (...)"
        "2. Choose ..."  -> "Choose ..."

    Raw OCR text is NOT modified.
    """

    return re.sub(
        r"^\s*\d+\.\s*",
        "",
        text.strip(),
    )


def is_question_or_section(text):
    """
    Detect obvious section/question/instruction lines.
    """

    value = text.strip()

    if not value:
        return True

    lowered = value.lower()

    if re.match(
        r"^\s*\d+[a-z]?:",
        value,
        flags=re.IGNORECASE,
    ):
        return True

    if re.match(
        r"^\s*\d+\.\s+",
        value,
    ):
        cleaned = strip_leading_numbering(
            value
        ).lower()

        # Numbered labels such as "1. Name" are still
        # valid field labels.
        if cleaned.startswith(
            (
                "name",
                "date",
                "street",
                "city",
                "state",
                "zip",
                "medicare identification",
            )
        ):
            return False

        return True

    section_phrases = (
        "choose the information",
        "check only one box",
        "check all that apply",
        "i want to share",
        "information about the person",
        "use this form",
        "authorization to disclose",
        "department of",
        "centers for",
        "form approved",
        "omb no.",
    )

    return any(
        phrase in lowered
        for phrase
        in section_phrases
    )


def is_value_like(text):
    """
    Estimate whether OCR text resembles entered form data.

    Unlike Experiment 37, leading question numbers such as
    "1. Name" are not treated as value evidence.
    """

    value = text.strip()

    if not value:
        return False

    if is_question_or_section(value):
        return False

    semantic = strip_leading_numbering(
        value
    )

    # Date/numeric-like.
    if re.fullmatch(
        r"[\d\s/.,()-]+",
        semantic,
    ):
        return True

    # Alphanumeric IDs.
    if (
        re.search(
            r"[A-Za-z]",
            semantic,
        )
        and re.search(
            r"\d",
            semantic,
        )
    ):
        return True

    # Short uppercase code, e.g. TX.
    if re.fullmatch(
        r"[A-Z]{2,4}",
        semantic,
    ):
        return True

    words = semantic.split()

    # Short ordinary entered values such as:
    # Alex Carter
    # Austin
    if (
        1 <= len(words) <= 6
        and len(semantic) <= 50
    ):
        if not semantic.endswith(
            ("?", ":")
        ):
            return True

    return False


def value_score(text):
    value = text.strip()

    if not is_value_like(value):
        return -100

    semantic = strip_leading_numbering(
        value
    )

    score = 0

    if re.fullmatch(
        r"[\d\s/.,()-]+",
        semantic,
    ):
        score += 8

    if (
        re.search(
            r"[A-Za-z]",
            semantic,
        )
        and re.search(
            r"\d",
            semantic,
        )
    ):
        score += 8

    if re.fullmatch(
        r"[A-Z]{2,4}",
        semantic,
    ):
        score += 8

    words = semantic.split()

    if 1 <= len(words) <= 3:
        score += 5

    if len(semantic) <= 40:
        score += 2

    if len(semantic) > 70:
        score -= 10

    return score


def is_candidate_label(
    region,
    excluded_region_ids,
):
    if (
        region["region_id"]
        in excluded_region_ids
    ):
        return False

    text = region["text"].strip()

    if not text:
        return False

    if is_question_or_section(text):
        return False

    # A strong value should not become a field label,
    # unless the text itself looks explicitly like a label.
    label_markers = (
        "name",
        "date of birth",
        "identification number",
        "street address",
        "city",
        "state",
        "zip code",
    )

    lowered = strip_leading_numbering(
        text
    ).lower()

    if any(
        marker in lowered
        for marker in label_markers
    ):
        return True

    if is_value_like(text):
        return False

    return True


def reconstruct_fields(
    regions,
    excluded_region_ids,
):
    candidates = []

    for label in regions:
        if not is_candidate_label(
            label,
            excluded_region_ids,
        ):
            continue

        label_box = label["bbox"]

        possible_values = []

        for value in regions:
            if (
                value["region_id"]
                == label["region_id"]
            ):
                continue

            if (
                value["region_id"]
                in excluded_region_ids
            ):
                continue

            if is_question_or_section(
                value["text"]
            ):
                continue

            likelihood = value_score(
                value["text"]
            )

            if likelihood < 0:
                continue

            value_box = value["bbox"]

            vertical_gap = (
                value_box["y"]
                - label_box["y2"]
            )

            # OCR boxes can overlap slightly.
            if not (
                -15
                <= vertical_gap
                <= 35
            ):
                continue

            x_difference = abs(
                value_box["x"]
                - label_box["x"]
            )

            if x_difference > 45:
                continue

            geometry_score = (
                abs(vertical_gap)
                + x_difference
                - likelihood * 4
            )

            possible_values.append(
                (
                    geometry_score,
                    value,
                    vertical_gap,
                    x_difference,
                    likelihood,
                )
            )

        possible_values.sort(
            key=lambda item: item[0]
        )

        if not possible_values:
            continue

        (
            pair_score,
            value,
            vertical_gap,
            x_difference,
            likelihood,
        ) = possible_values[0]

        candidates.append(
            {
                "label_region_id": (
                    label[
                        "region_id"
                    ]
                ),
                "label_raw_text": (
                    label["text"]
                ),
                "label": (
                    strip_leading_numbering(
                        label["text"]
                    )
                ),
                "value_region_id": (
                    value[
                        "region_id"
                    ]
                ),
                "value": (
                    value["text"]
                ),
                "label_confidence": (
                    label[
                        "confidence"
                    ]
                ),
                "value_confidence": (
                    value[
                        "confidence"
                    ]
                ),
                "vertical_gap": (
                    vertical_gap
                ),
                "left_alignment_difference": (
                    x_difference
                ),
                "value_score": (
                    likelihood
                ),
                "pair_score": (
                    pair_score
                ),
            }
        )

    candidates.sort(
        key=lambda item: (
            item["pair_score"],
            item["label_region_id"],
        )
    )

    # One value should only belong to one label.
    used_values = set()
    fields = []

    for candidate in candidates:
        value_id = candidate[
            "value_region_id"
        ]

        if value_id in used_values:
            continue

        fields.append(candidate)
        used_values.add(
            value_id
        )

    fields.sort(
        key=lambda item:
        item["label_region_id"]
    )

    for index, field in enumerate(
        fields,
        start=1,
    ):
        field[
            "field_id"
        ] = index

    return fields


def classify_regions(
    regions,
    checkboxes,
    followups,
    fields,
):
    checkbox_region_ids = {
        item[
            "label_region_id"
        ]
        for item in checkboxes
    }

    followup_region_ids = {
        item[
            "text_region_id"
        ]
        for item in followups
    }

    field_label_ids = {
        item[
            "label_region_id"
        ]
        for item in fields
    }

    field_value_ids = {
        item[
            "value_region_id"
        ]
        for item in fields
    }

    classifications = []

    for region in regions:
        region_id = region[
            "region_id"
        ]

        if region_id in (
            checkbox_region_ids
        ):
            role = (
                "checkbox_label"
            )

        elif region_id in (
            followup_region_ids
        ):
            role = (
                "free_text_response"
            )

        elif region_id in (
            field_label_ids
        ):
            role = (
                "field_label"
            )

        elif region_id in (
            field_value_ids
        ):
            role = (
                "field_value"
            )

        elif is_question_or_section(
            region["text"]
        ):
            role = (
                "instruction_or_section"
            )

        else:
            role = (
                "unclassified_text"
            )

        classifications.append(
            {
                "region_id": (
                    region_id
                ),
                "text": (
                    region["text"]
                ),
                "role": role,
            }
        )

    return classifications


def make_report(
    fields,
    checkboxes,
    followups,
    classifications,
):
    lines = [
        "=" * 78,
        "EXPERIMENT 38: FORM SEMANTIC ROLE CLEANUP",
        "=" * 78,
        (
            f"Structured fields: "
            f"{len(fields)}"
        ),
        (
            f"Checkbox controls: "
            f"{len(checkboxes)}"
        ),
        (
            f"Follow-up text: "
            f"{len(followups)}"
        ),
        "=" * 78,
        "",
        "STRUCTURED FORM FIELDS",
        "-" * 78,
    ]

    for field in fields:
        lines.append(
            (
                f"F{field['field_id']:02d} "
                f"R{field['label_region_id']:02d} "
                f"{field['label']!r}"
                "  ->  "
                f"R{field['value_region_id']:02d} "
                f"{field['value']!r}"
            )
        )

    lines.extend(
        [
            "",
            "CHECKBOX CONTROLS",
            "-" * 78,
        ]
    )

    for checkbox in checkboxes:
        lines.append(
            (
                f"C{checkbox['control_id']:02d} "
                f"{checkbox['state']:<9} "
                f"{checkbox['label']!r} "
                f"(ratio="
                f"{checkbox['interior_dark_ratio']:.3f})"
            )
        )

    lines.extend(
        [
            "",
            "FOLLOW-UP TEXT",
            "-" * 78,
        ]
    )

    if not followups:
        lines.append("None")

    for item in followups:
        lines.append(
            (
                f"C{item['control_id']:02d} "
                f"{item['control_label']!r}"
                "  ->  "
                f"R{item['text_region_id']:02d} "
                f"{item['text']!r}"
            )
        )

    lines.extend(
        [
            "",
            "REGION ROLE SUMMARY",
            "-" * 78,
        ]
    )

    role_counts = {}

    for item in classifications:
        role = item["role"]

        role_counts[role] = (
            role_counts.get(
                role,
                0,
            )
            + 1
        )

    for role, count in sorted(
        role_counts.items()
    ):
        lines.append(
            f"{role}: {count}"
        )

    return "\n".join(
        lines
    ) + "\n"


def main():
    exp37 = load_json(
        EXP37_JSON
    )

    regions = (
        load_ocr_regions()
    )

    checkboxes = exp37.get(
        "checkboxes",
        [],
    )

    followups = exp37.get(
        "followup_text",
        [],
    )

    excluded_region_ids = {
        checkbox[
            "label_region_id"
        ]
        for checkbox in checkboxes
    }

    excluded_region_ids.update(
        followup[
            "text_region_id"
        ]
        for followup in followups
    )

    fields = reconstruct_fields(
        regions,
        excluded_region_ids,
    )

    classifications = (
        classify_regions(
            regions,
            checkboxes,
            followups,
            fields,
        )
    )

    result = {
        "experiment": 38,
        "schema_version": (
            "0.3-form-experimental"
        ),
        "document_type": (
            "structured_form"
        ),
        "source": {
            "experiment_37": str(
                EXP37_JSON
            ),
            "ocr_csv": str(
                OCR_CSV_PATH
            ),
        },
        "statistics": {
            "ocr_regions": (
                len(regions)
            ),
            "fields": (
                len(fields)
            ),
            "checkboxes": (
                len(checkboxes)
            ),
            "checked_checkboxes": (
                sum(
                    bool(
                        item[
                            "checked"
                        ]
                    )
                    for item
                    in checkboxes
                )
            ),
            "unchecked_checkboxes": (
                sum(
                    not bool(
                        item[
                            "checked"
                        ]
                    )
                    for item
                    in checkboxes
                )
            ),
            "followup_text_regions": (
                len(followups)
            ),
        },
        "fields": fields,
        "checkboxes": checkboxes,
        "followup_text": followups,
        "region_roles": (
            classifications
        ),
        "notes": [
            (
                "Checkbox labels are excluded from "
                "ordinary key/value reconstruction."
            ),
            (
                "Follow-up response regions are excluded "
                "from ordinary key/value reconstruction."
            ),
            (
                "Leading question numbers are removed "
                "only from semantic label text."
            ),
            (
                "Raw OCR text remains preserved through "
                "source region references."
            ),
            (
                "No actual field values or checkbox "
                "labels were hard-coded."
            ),
        ],
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    report = make_report(
        fields,
        checkboxes,
        followups,
        classifications,
    )

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print("=" * 72)
    print(
        "EXPERIMENT 38: "
        "FORM SEMANTIC ROLE CLEANUP"
    )
    print("=" * 72)

    print(
        "OCR regions:",
        len(regions),
    )

    print(
        "Structured fields:",
        len(fields),
    )

    print(
        "Checkbox controls:",
        len(checkboxes),
    )

    print(
        "Checked:",
        sum(
            bool(
                item["checked"]
            )
            for item
            in checkboxes
        ),
    )

    print(
        "Unchecked:",
        sum(
            not bool(
                item["checked"]
            )
            for item
            in checkboxes
        ),
    )

    print(
        "Follow-up text:",
        len(followups),
    )

    print()

    print(
        f"JSON: {OUTPUT_JSON}"
    )

    print(
        f"Report: {REPORT_PATH}"
    )

    print("=" * 72)
    print(
        "EXPERIMENT 38 COMPLETE"
    )


if __name__ == "__main__":
    main()