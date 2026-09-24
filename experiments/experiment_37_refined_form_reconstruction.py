"""
Experiment 37: Refined Form Reconstruction

Refines the unseen-form baseline from Experiment 36.

Goals:
- Reuse existing OCR and scan image.
- Detect actual checkbox controls while rejecting character glyphs.
- Infer checked/unchecked state from visual evidence.
- Reconstruct likely key/value form fields geometrically.
- Preserve OCR evidence and source region IDs.
- Detect possible follow-up free-text associated with controls.

No OCR is rerun.
No form values or checkbox labels are hard-coded.
"""

import csv
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = (
    PROJECT_ROOT
    / "results"
    / "gui_v4"
    / "single"
    / "pic13"
    / "scan"
    / "scan.png"
)

OCR_CSV_PATH = (
    PROJECT_ROOT
    / "results"
    / "gui_v4"
    / "single"
    / "pic13"
    / "ocr"
    / "results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "parsing"
    / "pic13"
    / "refined_form"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "refined_form_structure.json"
)

CHECKBOX_OVERLAY_PATH = (
    OUTPUT_DIR
    / "refined_checkbox_overlay.png"
)

FIELD_OVERLAY_PATH = (
    OUTPUT_DIR
    / "refined_field_overlay.png"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "refined_form_report.txt"
)


def load_ocr_regions():
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
                    "meaningful": (
                        str(
                            row["meaningful"]
                        )
                        .strip()
                        .lower()
                        == "true"
                    ),
                    "bbox": {
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "x2": x + w,
                        "y2": y + h,
                    },
                    "center": {
                        "x": x + w / 2.0,
                        "y": y + h / 2.0,
                    },
                }
            )

    return regions


def clean_control_label(text):
    """
    Remove OCR checkbox symbols from the searchable/display
    label while keeping raw OCR separately.
    """

    if not text:
        return ""

    cleaned = text.strip()

    cleaned = re.sub(
        r"^[☐☑☒✓✔□■▣]+\s*",
        "",
        cleaned,
    )

    return cleaned.strip()


def get_binary_image(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    return cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU,
    )[1]


def detect_square_candidates(
    image,
    binary,
):
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    height, width = binary.shape

    candidates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(
            contour
        )

        # Experiment 36 showed that the genuine
        # controls on this scan are approximately
        # small square objects. These are geometric
        # limits, not label-specific rules.
        if not (
            18 <= w <= 24
            and 18 <= h <= 24
        ):
            continue

        if (
            x <= 5
            or y <= 5
            or x + w >= width - 5
            or y + h >= height - 5
        ):
            continue

        aspect = w / float(h)

        if not 0.78 <= aspect <= 1.28:
            continue

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(
            contour,
            0.07 * perimeter,
            True,
        )

        if not 4 <= len(approx) <= 8:
            continue

        margin_x = max(
            3,
            int(round(w * 0.23)),
        )

        margin_y = max(
            3,
            int(round(h * 0.23)),
        )

        inner = binary[
            y + margin_y:
            y + h - margin_y,
            x + margin_x:
            x + w - margin_x,
        ]

        if inner.size == 0:
            continue

        interior_dark_ratio = (
            np.count_nonzero(inner)
            / inner.size
        )

        candidates.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "x2": x + w,
                "y2": y + h,
                "center_x": x + w / 2.0,
                "center_y": y + h / 2.0,
                "aspect_ratio": aspect,
                "interior_dark_ratio": (
                    interior_dark_ratio
                ),
            }
        )

    return deduplicate_candidates(
        candidates
    )


def deduplicate_candidates(
    candidates,
):
    candidates = sorted(
        candidates,
        key=lambda item: (
            item["y"],
            item["x"],
            item["w"] * item["h"],
        ),
    )

    kept = []

    for candidate in candidates:
        duplicate = False

        for existing in kept:
            distance = math.hypot(
                candidate["center_x"]
                - existing["center_x"],
                candidate["center_y"]
                - existing["center_y"],
            )

            if distance <= 6:
                duplicate = True
                break

        if not duplicate:
            kept.append(candidate)

    return kept


def find_control_label(
    checkbox,
    regions,
):
    """
    Find OCR text on approximately the same horizontal
    line as the square control.

    Important:
    PaddleOCR may include the square inside the OCR
    region bbox, so overlap between the checkbox and
    text bbox is allowed.
    """

    possibilities = []

    for region in regions:
        bbox = region["bbox"]

        vertical_distance = abs(
            checkbox["center_y"]
            - region["center"]["y"]
        )

        if vertical_distance > 18:
            continue

        # Genuine controls should be close to the
        # beginning of their associated OCR region.
        left_edge_difference = abs(
            checkbox["x"]
            - bbox["x"]
        )

        gap_from_control = (
            bbox["x"]
            - checkbox["x2"]
        )

        # Handles both:
        #
        # [box] Label
        #
        # and OCR bboxes that include the checkbox:
        #
        # [box Label........]
        if (
            left_edge_difference > 42
            and not (
                -15
                <= gap_from_control
                <= 42
            )
        ):
            continue

        # The OCR region should extend substantially
        # beyond the checkbox itself.
        if bbox["x2"] <= (
            checkbox["x2"] + 20
        ):
            continue

        score = (
            vertical_distance
            + min(
                left_edge_difference,
                abs(gap_from_control),
            )
        )

        possibilities.append(
            (
                score,
                region,
            )
        )

    possibilities.sort(
        key=lambda item: item[0]
    )

    if not possibilities:
        return None

    return possibilities[0][1]


def filter_real_controls(
    raw_candidates,
    regions,
):
    controls = []

    used_region_ids = set()

    for candidate in raw_candidates:
        region = find_control_label(
            candidate,
            regions,
        )

        if region is None:
            continue

        # One OCR option line should normally map to
        # one square control.
        if (
            region["region_id"]
            in used_region_ids
        ):
            continue

        control = dict(candidate)

        control["label_region_id"] = (
            region["region_id"]
        )

        control["label_raw_text"] = (
            region["text"]
        )

        control["label"] = (
            clean_control_label(
                region["text"]
            )
        )

        controls.append(control)

        used_region_ids.add(
            region["region_id"]
        )

    controls.sort(
        key=lambda item: (
            item["y"],
            item["x"],
        )
    )

    for index, control in enumerate(
        controls,
        start=1,
    ):
        control[
            "control_id"
        ] = index

    return controls


def infer_state_threshold(
    controls,
):
    """
    Determine the checked/unchecked split from the controls
    themselves.

    We use the largest separation between sorted interior
    darkness values. This avoids hard-coding a threshold
    based on a particular label.
    """

    ratios = sorted(
        control[
            "interior_dark_ratio"
        ]
        for control in controls
    )

    if len(ratios) < 2:
        return 0.30

    best_gap = -1.0
    threshold = 0.30

    for left, right in zip(
        ratios[:-1],
        ratios[1:],
    ):
        gap = right - left

        if gap > best_gap:
            best_gap = gap
            threshold = (
                left + right
            ) / 2.0

    # Avoid nonsensical automatic splits if the
    # distribution has no useful separation.
    if best_gap < 0.08:
        return 0.30

    return threshold


def assign_control_states(
    controls,
):
    threshold = infer_state_threshold(
        controls
    )

    for control in controls:
        ratio = float(
            control[
                "interior_dark_ratio"
            ]
        )

        control[
            "interior_dark_ratio"
        ] = ratio

        control["checked"] = bool(
            ratio >= threshold
        )

        control[
            "state"
        ] = (
            "checked"
            if control["checked"]
            else "unchecked"
        )

    return float(threshold)


def is_obvious_value(text):
    """
    Strong indicators that text is probably a value rather
    than a field label.
    """

    value = text.strip()

    if not value:
        return False

    # Numeric or date-like value.
    if re.fullmatch(
        r"[\d\s/.,()-]+",
        value,
    ):
        return True

    # Contains both letters and digits, common for IDs.
    if (
        re.search(r"[A-Za-z]", value)
        and re.search(r"\d", value)
    ):
        return True

    # Very short uppercase code such as a state code.
    if re.fullmatch(
        r"[A-Z]{2,4}",
        value,
    ):
        return True

    return False


def value_likelihood(text):
    """
    Generic value-likelihood score.

    Higher = more likely to be form-entered data.
    """

    value = text.strip()

    if not value:
        return 0

    score = 0

    if is_obvious_value(value):
        score += 5

    words = value.split()

    # Short alphabetic entries often represent names,
    # cities, simple values, etc.
    if (
        1 <= len(words) <= 2
        and all(
            re.search(
                r"[A-Za-z]",
                word,
            )
            for word in words
        )
    ):
        score += 3

    if len(value) <= 40:
        score += 1

    # Long instructions/questions are poor values.
    if len(value) > 70:
        score -= 5

    if value.endswith("?"):
        score -= 5

    return score


def find_first_form_body_y(
    regions,
):
    """
    Numbered forms often begin with a numbered field or
    question. Use that as a generic body boundary when
    available, preventing page headers from becoming
    field/value pairs.
    """

    numbered = []

    for region in regions:
        if re.match(
            r"^\s*\d+\.",
            region["text"],
        ):
            numbered.append(
                region["bbox"]["y"]
            )

    if numbered:
        return min(numbered)

    return 0


def reconstruct_fields(
    regions,
):
    body_y = find_first_form_body_y(
        regions
    )

    candidates = []

    for label_region in regions:
        label_box = (
            label_region["bbox"]
        )

        if label_box["y"] < body_y:
            continue

        # Strong values should not themselves become
        # labels for another field.
        if is_obvious_value(
            label_region["text"]
        ):
            continue

        possible_values = []

        for value_region in regions:
            if (
                value_region["region_id"]
                == label_region["region_id"]
            ):
                continue

            value_box = (
                value_region["bbox"]
            )

            vertical_gap = (
                value_box["y"]
                - label_box["y2"]
            )

            # Allow slightly overlapping OCR boxes.
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

            likelihood = (
                value_likelihood(
                    value_region["text"]
                )
            )

            if likelihood < 3:
                continue

            # Lower is better.
            geometry_score = (
                abs(vertical_gap)
                + x_difference
                - (likelihood * 4)
            )

            possible_values.append(
                (
                    geometry_score,
                    value_region,
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
            score,
            value_region,
            vertical_gap,
            x_difference,
            likelihood,
        ) = possible_values[0]

        candidates.append(
            {
                "label_region_id": (
                    label_region[
                        "region_id"
                    ]
                ),
                "label": (
                    label_region[
                        "text"
                    ]
                ),
                "value_region_id": (
                    value_region[
                        "region_id"
                    ]
                ),
                "value": (
                    value_region[
                        "text"
                    ]
                ),
                "label_confidence": (
                    label_region[
                        "confidence"
                    ]
                ),
                "value_confidence": (
                    value_region[
                        "confidence"
                    ]
                ),
                "vertical_gap": (
                    vertical_gap
                ),
                "left_alignment_difference": (
                    x_difference
                ),
                "value_likelihood": (
                    likelihood
                ),
                "pair_score": score,
            }
        )

    # Keep the strongest mapping to each value region.
    candidates.sort(
        key=lambda item: (
            item["pair_score"],
            item["label_region_id"],
        )
    )

    used_values = set()
    fields = []

    for candidate in candidates:
        value_id = candidate[
            "value_region_id"
        ]

        if value_id in used_values:
            continue

        fields.append(candidate)
        used_values.add(value_id)

    fields.sort(
        key=lambda item:
        item["label_region_id"]
    )

    for index, field in enumerate(
        fields,
        start=1,
    ):
        field["field_id"] = index

    return fields


def find_followup_text(
    controls,
    regions,
):
    """
    Find possible free-text continuation beneath checked
    options.

    A candidate must:
    - be below the option,
    - align with its text,
    - not itself be another checkbox label.
    """

    checkbox_region_ids = {
        control["label_region_id"]
        for control in controls
    }

    followups = []

    region_by_id = {
        region["region_id"]: region
        for region in regions
    }

    for control in controls:
        if not control["checked"]:
            continue

        label_region = region_by_id[
            control["label_region_id"]
        ]

        label_box = (
            label_region["bbox"]
        )

        possible = []

        for region in regions:
            if (
                region["region_id"]
                in checkbox_region_ids
            ):
                continue

            bbox = region["bbox"]

            gap = (
                bbox["y"]
                - label_box["y2"]
            )

            if not (
                5 <= gap <= 75
            ):
                continue

            x_difference = abs(
                bbox["x"]
                - label_box["x"]
            )

            if x_difference > 45:
                continue

            if len(
                region["text"].strip()
            ) < 15:
                continue

            score = (
                gap + x_difference
            )

            possible.append(
                (
                    score,
                    region,
                    gap,
                    x_difference,
                )
            )

        possible.sort(
            key=lambda item: item[0]
        )

        if possible:
            (
                score,
                region,
                gap,
                x_difference,
            ) = possible[0]

            followups.append(
                {
                    "control_id": (
                        control[
                            "control_id"
                        ]
                    ),
                    "control_label": (
                        control[
                            "label"
                        ]
                    ),
                    "text_region_id": (
                        region[
                            "region_id"
                        ]
                    ),
                    "text": (
                        region["text"]
                    ),
                    "vertical_gap": gap,
                    "left_alignment_difference": (
                        x_difference
                    ),
                    "association_score": (
                        score
                    ),
                }
            )

    return followups


def draw_checkbox_overlay(
    image,
    controls,
    threshold,
):
    overlay = image.copy()

    for control in controls:
        x = control["x"]
        y = control["y"]

        x2 = control["x2"]
        y2 = control["y2"]

        cv2.rectangle(
            overlay,
            (x, y),
            (x2, y2),
            (0, 0, 255),
            2,
        )

        text = (
            f"C{control['control_id']} "
            f"{control['state']} "
            f"{control['interior_dark_ratio']:.3f}"
        )

        cv2.putText(
            overlay,
            text,
            (
                x,
                max(
                    15,
                    y - 7,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        overlay,
        f"Automatic state threshold: {threshold:.3f}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return overlay


def draw_field_overlay(
    image,
    fields,
    regions,
):
    overlay = image.copy()

    region_by_id = {
        region["region_id"]: region
        for region in regions
    }

    for field in fields:
        label = region_by_id[
            field["label_region_id"]
        ]

        value = region_by_id[
            field["value_region_id"]
        ]

        lb = label["bbox"]
        vb = value["bbox"]

        cv2.rectangle(
            overlay,
            (
                lb["x"],
                lb["y"],
            ),
            (
                lb["x2"],
                lb["y2"],
            ),
            (255, 0, 0),
            2,
        )

        cv2.rectangle(
            overlay,
            (
                vb["x"],
                vb["y"],
            ),
            (
                vb["x2"],
                vb["y2"],
            ),
            (0, 128, 0),
            2,
        )

        start = (
            int(label["center"]["x"]),
            int(label["center"]["y"]),
        )

        end = (
            int(value["center"]["x"]),
            int(value["center"]["y"]),
        )

        cv2.line(
            overlay,
            start,
            end,
            (0, 0, 255),
            1,
        )

        cv2.putText(
            overlay,
            f"F{field['field_id']}",
            (
                lb["x"],
                max(
                    15,
                    lb["y"] - 5,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return overlay


def make_report(
    image,
    raw_square_candidates,
    controls,
    threshold,
    fields,
    followups,
):
    height, width = image.shape[:2]

    lines = [
        "=" * 78,
        "EXPERIMENT 37: REFINED FORM RECONSTRUCTION",
        "=" * 78,
        f"Image: {width}x{height}",
        (
            "Raw geometric square candidates: "
            f"{len(raw_square_candidates)}"
        ),
        (
            "Refined checkbox controls: "
            f"{len(controls)}"
        ),
        (
            "Automatic state threshold: "
            f"{threshold:.4f}"
        ),
        (
            "Reconstructed fields: "
            f"{len(fields)}"
        ),
        (
            "Follow-up text candidates: "
            f"{len(followups)}"
        ),
        "=" * 78,
        "",
        "REFINED CHECKBOX CONTROLS",
        "-" * 78,
    ]

    for control in controls:
        lines.append(
            (
                f"C{control['control_id']:02d} "
                f"{control['state']:<9} "
                f"ratio="
                f"{control['interior_dark_ratio']:.3f} "
                f"R{control['label_region_id']:02d} "
                f"{control['label']!r}"
            )
        )

    lines.extend(
        [
            "",
            "RECONSTRUCTED FORM FIELDS",
            "-" * 78,
        ]
    )

    for field in fields:
        lines.append(
            (
                f"F{field['field_id']:02d} "
                f"R{field['label_region_id']:02d} "
                f"{field['label']!r}"
                "  ->  "
                f"R{field['value_region_id']:02d} "
                f"{field['value']!r} "
                f"[gap={field['vertical_gap']}, "
                f"dx="
                f"{field['left_alignment_difference']}, "
                f"score={field['pair_score']}]"
            )
        )

    lines.extend(
        [
            "",
            "FOLLOW-UP TEXT CANDIDATES",
            "-" * 78,
        ]
    )

    if not followups:
        lines.append(
            "None"
        )

    for followup in followups:
        lines.append(
            (
                f"C{followup['control_id']:02d} "
                f"{followup['control_label']!r}"
                "  ->  "
                f"R{followup['text_region_id']:02d} "
                f"{followup['text']!r}"
            )
        )

    return "\n".join(
        lines
    ) + "\n"


def main():
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(
            IMAGE_PATH
        )

    if not OCR_CSV_PATH.is_file():
        raise FileNotFoundError(
            OCR_CSV_PATH
        )

    image = cv2.imread(
        str(IMAGE_PATH)
    )

    if image is None:
        raise RuntimeError(
            f"Unable to load image: "
            f"{IMAGE_PATH}"
        )

    regions = load_ocr_regions()

    binary = get_binary_image(
        image
    )

    raw_square_candidates = (
        detect_square_candidates(
            image,
            binary,
        )
    )

    controls = filter_real_controls(
        raw_square_candidates,
        regions,
    )

    threshold = (
        assign_control_states(
            controls
        )
    )

    fields = reconstruct_fields(
        regions
    )

    followups = find_followup_text(
        controls,
        regions,
    )

    height, width = (
        image.shape[:2]
    )

    result = {
        "experiment": 37,
        "schema_version": (
            "0.2-form-experimental"
        ),
        "document_type": (
            "structured_form_candidate"
        ),
        "source": {
            "image": str(
                IMAGE_PATH
            ),
            "ocr_csv": str(
                OCR_CSV_PATH
            ),
        },
        "image": {
            "width": width,
            "height": height,
        },
        "statistics": {
            "ocr_regions": (
                len(regions)
            ),
            "raw_square_candidates": (
                len(
                    raw_square_candidates
                )
            ),
            "refined_checkbox_controls": (
                len(controls)
            ),
            "reconstructed_fields": (
                len(fields)
            ),
            "followup_text_candidates": (
                len(followups)
            ),
        },
        "checkbox_state_threshold": (
            threshold
        ),
        "fields": fields,
        "checkboxes": controls,
        "followup_text": followups,
        "notes": [
            (
                "Existing production OCR was reused."
            ),
            (
                "No OCR was rerun."
            ),
            (
                "No field values or checkbox option "
                "labels were hard-coded."
            ),
            (
                "Checkbox state threshold was inferred "
                "from the detected control distribution."
            ),
            (
                "Small OCR bounding-box overlaps are "
                "allowed during field reconstruction."
            ),
            (
                "Raw OCR text remains unchanged in its "
                "source artifact."
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

    checkbox_overlay = (
        draw_checkbox_overlay(
            image,
            controls,
            threshold,
        )
    )

    field_overlay = (
        draw_field_overlay(
            image,
            fields,
            regions,
        )
    )

    cv2.imwrite(
        str(
            CHECKBOX_OVERLAY_PATH
        ),
        checkbox_overlay,
    )

    cv2.imwrite(
        str(
            FIELD_OVERLAY_PATH
        ),
        field_overlay,
    )

    report = make_report(
        image,
        raw_square_candidates,
        controls,
        threshold,
        fields,
        followups,
    )

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print("=" * 72)
    print(
        "EXPERIMENT 37: "
        "REFINED FORM RECONSTRUCTION"
    )
    print("=" * 72)

    print(
        f"OCR regions: "
        f"{len(regions)}"
    )

    print(
        "Raw square candidates:",
        len(
            raw_square_candidates
        ),
    )

    print(
        "Refined checkbox controls:",
        len(controls),
    )

    print(
        "Automatic state threshold:",
        f"{threshold:.4f}",
    )

    print(
        "Reconstructed fields:",
        len(fields),
    )

    print(
        "Follow-up text candidates:",
        len(followups),
    )

    print()

    print(
        f"JSON: {OUTPUT_JSON}"
    )

    print(
        "Checkbox overlay:",
        CHECKBOX_OVERLAY_PATH,
    )

    print(
        "Field overlay:",
        FIELD_OVERLAY_PATH,
    )

    print(
        f"Report: {REPORT_PATH}"
    )

    print("=" * 72)
    print(
        "EXPERIMENT 37 COMPLETE"
    )


if __name__ == "__main__":
    main()