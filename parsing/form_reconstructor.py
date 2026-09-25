"""Conservative automatic form reconstruction.

Experiment 70 revision 4.

Changes from revision 3:
- fixes enumerated-label detection so address values such as "101 Sample..."
  are not mistaken for question labels;
- recognizes time-shaped values explicitly;
- keeps generic label words such as CASH from being treated as short uppercase
  values;
- adds horizontal-overlap support for vertically stacked label/value fields,
  helping aligned below-values beat unrelated same-row values.

No OCR, LLM, API, or document-specific field names are used.
"""

from __future__ import annotations

import re
from typing import Any

from parsing.layout_analyzer_refined import analyze_document_layout_refined


DATEISH_RE = re.compile(r"^\d{1,4}[/-]\d{1,2}[/-]\d{1,4}$")
TIMEISH_RE = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?$")
NUMBERISH_RE = re.compile(r"^[\s$€£¥NTnt0-9.,/%+\-()]+$")
IDENTIFIERISH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")

# Requires punctuation after the leading item number. This accepts "1. Name"
# and "2A:" but does not misclassify an address beginning "101 Sample...".
ENUMERATED_LABEL_RE = re.compile(r"^\d{1,2}[A-Za-z]?[.):]\s*")

CHECKBOX_PREFIXES = ("☐", "☑", "□", "■", "✓", "✔")

GENERIC_LABEL_WORDS = {
    "name", "date", "birth", "number", "id", "identification",
    "address", "street", "city", "state", "zip", "code",
    "phone", "email", "amount", "total", "tax", "cash",
    "reference", "status", "payment",
}


def _normalize(text: str) -> str:
    return " ".join(str(text).split()).strip()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+", text.casefold())


def _strong_value(text: str) -> bool:
    text = _normalize(text)
    if not text:
        return False

    if DATEISH_RE.match(text) or TIMEISH_RE.match(text):
        return True

    if NUMBERISH_RE.match(text):
        return True

    if IDENTIFIERISH_RE.match(text) and any(ch.isdigit() for ch in text):
        return True

    # Short uppercase codes such as TX are values, but known generic form
    # labels such as CASH/TAX/ID should remain labels.
    if (
        text.isalpha()
        and text.isupper()
        and 1 <= len(text) <= 4
        and text.casefold() not in GENERIC_LABEL_WORDS
    ):
        return True

    # Address/identifier-like text containing digits, but not enumerated labels.
    if (
        any(ch.isdigit() for ch in text)
        and len(text) <= 100
        and not ENUMERATED_LABEL_RE.match(text)
    ):
        return True

    return False


def _probable_label(text: str) -> bool:
    text = _normalize(text)
    if not text:
        return False

    if text.startswith(CHECKBOX_PREFIXES):
        return False

    if len(text) > 100:
        return False

    # Atomic values must not become labels.
    if DATEISH_RE.match(text) or TIMEISH_RE.match(text) or NUMBERISH_RE.match(text):
        return False

    if ENUMERATED_LABEL_RE.match(text):
        return True

    if text.endswith((".", "?", "!")):
        return False

    if text.endswith((":", "：")):
        return True

    words = _tokens(text)
    if not words or len(words) > 12:
        return False

    # Generic form vocabulary is considered before the fallback strong-value
    # check, so labels such as CASH remain labels.
    if any(word in GENERIC_LABEL_WORDS for word in words):
        return True

    if _strong_value(text):
        return False

    if "(" in text and ")" in text:
        return True

    return False


def _textual_value(text: str) -> bool:
    text = _normalize(text)
    if not text:
        return False

    if text.startswith(CHECKBOX_PREFIXES):
        return False

    if _probable_label(text):
        return False

    if _strong_value(text):
        return True

    words = text.split()
    if 1 <= len(words) <= 10 and len(text) <= 100:
        if not text.endswith((".", "?", "!")):
            return True

    return False


def _center(region: dict[str, Any]) -> tuple[float, float]:
    box = region["bbox"]
    return (
        float(box["x"]) + float(box["width"]) / 2.0,
        float(box["y"]) + float(box["height"]) / 2.0,
    )


def _horizontal_overlap_ratio(
    a: dict[str, Any],
    b: dict[str, Any],
) -> float:
    abox = a["bbox"]
    bbox = b["bbox"]

    a_left = float(abox["x"])
    a_right = a_left + float(abox["width"])
    b_left = float(bbox["x"])
    b_right = b_left + float(bbox["width"])

    overlap = max(
        0.0,
        min(a_right, b_right) - max(a_left, b_left),
    )

    denom = max(
        1.0,
        min(float(abox["width"]), float(bbox["width"])),
    )

    return overlap / denom


def _rows(layout: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}

    for region in layout.get("regions", []):
        grouped.setdefault(int(region["row_id"]), []).append(region)

    for row_id in grouped:
        grouped[row_id].sort(key=lambda r: float(r["bbox"]["x"]))

    return grouped


def _candidate_score(
    label: dict[str, Any],
    value: dict[str, Any],
    *,
    relationship: str,
    row_distance: int,
) -> float:
    """Lower is better."""
    lx, ly = _center(label)
    vx, vy = _center(value)

    label_width = max(1.0, float(label["bbox"]["width"]))
    label_height = max(1.0, float(label["bbox"]["height"]))

    dx = abs(vx - lx)
    dy = abs(vy - ly)

    if relationship == "same_row_right":
        label_right = (
            float(label["bbox"]["x"])
            + float(label["bbox"]["width"])
        )
        horizontal_gap = max(
            0.0,
            float(value["bbox"]["x"]) - label_right,
        )

        score = (
            horizontal_gap / label_width * 22.0
            + dy / label_height * 8.0
            - 18.0
        )
    else:
        label_bottom = (
            float(label["bbox"]["y"])
            + float(label["bbox"]["height"])
        )
        vertical_gap = max(
            0.0,
            float(value["bbox"]["y"]) - label_bottom,
        )

        overlap_ratio = _horizontal_overlap_ratio(label, value)

        score = (
            dx / label_width * 30.0
            + vertical_gap / label_height * 20.0
            + (row_distance - 1) * 18.0
            - overlap_ratio * 35.0
        )

    text = _normalize(value["text"])

    if _strong_value(text):
        score -= 28.0
    else:
        score += 5.0

    return score


def _collect_candidates(
    label: dict[str, Any],
    row_id: int,
    grouped_rows: dict[int, list[dict[str, Any]]],
    *,
    lookahead_rows: int,
) -> list[tuple[float, str, dict[str, Any]]]:
    candidates: list[tuple[float, str, dict[str, Any]]] = []

    label_right = (
        float(label["bbox"]["x"])
        + float(label["bbox"]["width"])
    )

    for region in grouped_rows.get(row_id, []):
        if region["region_id"] == label["region_id"]:
            continue

        if float(region["bbox"]["x"]) < label_right:
            continue

        if not _textual_value(region["text"]):
            continue

        score = _candidate_score(
            label,
            region,
            relationship="same_row_right",
            row_distance=0,
        )
        candidates.append((score, "same_row_right", region))

    for distance in range(1, lookahead_rows + 1):
        for region in grouped_rows.get(row_id + distance, []):
            if not _textual_value(region["text"]):
                continue

            score = _candidate_score(
                label,
                region,
                relationship="next_row_below",
                row_distance=distance,
            )
            candidates.append((score, "next_row_below", region))

    candidates.sort(
        key=lambda item: (
            item[0],
            int(item[2]["row_id"]),
            float(item[2]["bbox"]["x"]),
        )
    )
    return candidates


def reconstruct_form_fields(
    document: dict[str, Any],
    *,
    lookahead_rows: int = 2,
) -> dict[str, Any]:
    layout = analyze_document_layout_refined(document)
    grouped_rows = _rows(layout)

    fields = []

    for row_id in sorted(grouped_rows):
        for label in grouped_rows[row_id]:
            label_text = _normalize(label["text"])

            if not _probable_label(label_text):
                continue

            candidates = _collect_candidates(
                label,
                row_id,
                grouped_rows,
                lookahead_rows=lookahead_rows,
            )

            if not candidates:
                continue

            best_score, relationship, value = candidates[0]

            if best_score > 115.0:
                continue

            fields.append(
                {
                    "field_id": f"field_{len(fields)+1:03d}",
                    "label": label_text,
                    "value": _normalize(value["text"]),
                    "label_region_id": label["region_id"],
                    "value_region_id": value["region_id"],
                    "relationship": relationship,
                    "label_row_id": label["row_id"],
                    "value_row_id": value["row_id"],
                    "geometry_score": round(best_score, 3),
                    "inference_status": "automatic_geometric_candidate",
                    "semantic_verified": False,
                }
            )

    return {
        "document_id": layout.get("document_id"),
        "region_count": layout.get("region_count", 0),
        "row_count": layout.get("row_count", 0),
        "field_candidate_count": len(fields),
        "fields": fields,
    }
