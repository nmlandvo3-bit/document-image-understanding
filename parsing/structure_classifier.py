"""Mixed document structure classification.

Experiment 71 revision 4.

Priority refinement:
- controls first;
- strong table evidence before heading detection, so uppercase table labels
  such as TOTAL remain table cells;
- headings then override accidental form assignments;
- form evidence may override only weak/short table candidates.

No OCR, LLM, API, or document-specific field names are used.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from parsing.layout_analyzer_refined import analyze_document_layout_refined
from parsing.table_reconstructor import reconstruct_candidate_tables
from parsing.form_reconstructor import reconstruct_form_fields


CHECKBOX_PREFIXES = ("☐", "☑", "□", "■", "✓", "✔")


def _normalize(text: str) -> str:
    return " ".join(str(text).split()).strip()


def _is_control(text: str) -> bool:
    text = _normalize(text)
    return bool(text) and text.startswith(CHECKBOX_PREFIXES)


def _is_heading(text: str) -> bool:
    text = _normalize(text)
    if not text or len(text) > 90:
        return False

    words = text.split()
    if not 1 <= len(words) <= 12:
        return False

    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False

    uppercase_ratio = sum(ch.isupper() for ch in letters) / len(letters)
    return uppercase_ratio >= 0.85


def _is_paragraph(text: str) -> bool:
    text = _normalize(text)
    if not text:
        return False

    return len(text.split()) >= 12 or len(text) >= 100


def _collect_table_region_strength(
    table_result: dict[str, Any],
) -> dict[int | str, str]:
    """Map table region IDs to strong/weak evidence.

    Candidates with >=4 reconstructed rows are strong.
    Shorter candidates are weak and may be overridden by form evidence.
    """
    strength: dict[int | str, str] = {}

    for table in table_result.get("tables", []):
        rows = table.get("rows", [])
        table_strength = "strong" if len(rows) >= 4 else "weak"

        for row in rows:
            for cell in row.get("cells", []):
                for region_id in cell.get("region_ids", []):
                    if strength.get(region_id) == "strong":
                        continue
                    strength[region_id] = table_strength

    return strength


def _collect_form_region_ids(
    form_result: dict[str, Any],
) -> tuple[set[int | str], set[int | str]]:
    label_ids: set[int | str] = set()
    value_ids: set[int | str] = set()

    for field in form_result.get("fields", []):
        label_ids.add(field["label_region_id"])
        value_ids.add(field["value_region_id"])

    return label_ids, value_ids


def _region_class(
    region: dict[str, Any],
    *,
    table_strength: dict[int | str, str],
    form_label_ids: set[int | str],
    form_value_ids: set[int | str],
) -> tuple[str, str]:
    rid = region["region_id"]
    text = _normalize(region["text"])
    strength = table_strength.get(rid)
    in_form = rid in form_label_ids or rid in form_value_ids

    if _is_control(text):
        return "control", "checkbox_or_control_marker"

    # Strong repeated table geometry is stronger than typography.
    # This keeps uppercase labels such as TOTAL/CASH inside real tables.
    if strength == "strong":
        return "table", "strong_reconstructed_table_cell"

    # Heading detection may override accidental form-candidate assignments.
    if _is_heading(text):
        return "heading", "uppercase_title_like_text"

    # Form evidence outranks only weak table candidates.
    if in_form:
        return "form", "reconstructed_form_field"

    if strength == "weak":
        return "table", "weak_reconstructed_table_cell"

    if _is_paragraph(text):
        return "paragraph", "long_prose_region"

    return "free_text", "unclassified_residual_text"


def _group_sections(
    regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: dict[int, list[dict[str, Any]]] = {}

    for region in regions:
        rows.setdefault(int(region["row_id"]), []).append(region)

    row_classes = []

    for row_id in sorted(rows):
        classes = [region["structure_type"] for region in rows[row_id]]
        counts = Counter(classes)

        priority = {
            "control": 0,
            "table": 1,
            "heading": 2,
            "form": 3,
            "paragraph": 4,
            "free_text": 5,
        }

        dominant = sorted(
            counts.items(),
            key=lambda item: (-item[1], priority[item[0]]),
        )[0][0]

        row_classes.append((row_id, dominant))

    sections = []
    current_rows = []
    current_type = None

    for row_id, structure_type in row_classes:
        if current_type is None:
            current_type = structure_type
            current_rows = [row_id]
            continue

        if structure_type == current_type and row_id == current_rows[-1] + 1:
            current_rows.append(row_id)
        else:
            sections.append(
                {
                    "section_id": f"section_{len(sections)+1:03d}",
                    "structure_type": current_type,
                    "row_start": current_rows[0],
                    "row_end": current_rows[-1],
                    "row_ids": list(current_rows),
                }
            )
            current_type = structure_type
            current_rows = [row_id]

    if current_rows:
        sections.append(
            {
                "section_id": f"section_{len(sections)+1:03d}",
                "structure_type": current_type,
                "row_start": current_rows[0],
                "row_end": current_rows[-1],
                "row_ids": list(current_rows),
            }
        )

    return sections


def classify_document_structure(
    document: dict[str, Any],
) -> dict[str, Any]:
    layout = analyze_document_layout_refined(document)
    table_result = reconstruct_candidate_tables(document)
    form_result = reconstruct_form_fields(document)

    table_strength = _collect_table_region_strength(table_result)
    form_label_ids, form_value_ids = _collect_form_region_ids(form_result)

    regions = []

    for region in layout.get("regions", []):
        structure_type, reason = _region_class(
            region,
            table_strength=table_strength,
            form_label_ids=form_label_ids,
            form_value_ids=form_value_ids,
        )

        enriched = dict(region)
        enriched["structure_type"] = structure_type
        enriched["structure_reason"] = reason
        enriched["table_strength"] = table_strength.get(region["region_id"])
        regions.append(enriched)

    counts = Counter(region["structure_type"] for region in regions)
    sections = _group_sections(regions)

    present_types = [
        structure_type
        for structure_type in (
            "heading",
            "table",
            "form",
            "paragraph",
            "control",
            "free_text",
        )
        if counts.get(structure_type, 0) > 0
    ]

    return {
        "document_id": layout.get("document_id"),
        "region_count": layout.get("region_count", 0),
        "row_count": layout.get("row_count", 0),
        "structure_types_present": present_types,
        "structure_counts": dict(counts),
        "candidate_table_count": table_result.get("candidate_table_count", 0),
        "field_candidate_count": form_result.get("field_candidate_count", 0),
        "section_count": len(sections),
        "sections": sections,
        "regions": regions,
    }
