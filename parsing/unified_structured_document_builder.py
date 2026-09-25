"""Unified structured-document builder.

Experiment 72 combines the validated outputs of Experiments 68-71 into one
traceable document representation.

The builder:
- preserves the original OCR evidence unchanged;
- adds generic form-field candidates;
- adds generic table records/cells;
- adds mixed-layout sections;
- preserves controls and residual/free-text regions;
- never resolves ambiguity by inventing a semantic answer.

No OCR, LLM, API, or document-specific field names are used.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from parsing.layout_analyzer_refined import analyze_document_layout_refined
from parsing.table_reconstructor import reconstruct_candidate_tables
from parsing.form_reconstructor import reconstruct_form_fields
from parsing.structure_classifier import classify_document_structure


def _evidence_id(document_id: str | None, region_id: int | str) -> str:
    if document_id:
        return f"{document_id}:ocr_region:{region_id}"
    return f"ocr_region:{region_id}"


def _build_fields(
    document_id: str | None,
    form_result: dict[str, Any],
) -> list[dict[str, Any]]:
    fields = []

    for field in form_result.get("fields", []):
        fields.append(
            {
                "field_id": field["field_id"],
                "label": field["label"],
                "value": field["value"],
                "label_region_id": field["label_region_id"],
                "value_region_id": field["value_region_id"],
                "relationship": field["relationship"],
                "inference_status": field.get(
                    "inference_status",
                    "automatic_geometric_candidate",
                ),
                "semantic_verified": bool(
                    field.get("semantic_verified", False)
                ),
                "geometry_score": field.get("geometry_score"),
                "evidence_item_ids": list(
                    dict.fromkeys(
                        [
                            _evidence_id(
                                document_id,
                                field["label_region_id"],
                            ),
                            _evidence_id(
                                document_id,
                                field["value_region_id"],
                            ),
                        ]
                    )
                ),
            }
        )

    return fields


def _build_records(
    document_id: str | None,
    table_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert reconstructed tables into generic row records."""
    records = []

    for table in table_result.get("tables", []):
        table_id = table["table_id"]

        for row_index, row in enumerate(
            table.get("rows", []),
            start=1,
        ):
            cells = []

            for cell in row.get("cells", []):
                region_ids = list(cell.get("region_ids", []))

                cells.append(
                    {
                        "column_index": cell["column_index"],
                        "column_position": cell.get("column_position"),
                        "text": cell.get("text", ""),
                        "confidence": cell.get("confidence"),
                        "region_ids": region_ids,
                        "evidence_item_ids": [
                            _evidence_id(document_id, region_id)
                            for region_id in region_ids
                        ],
                    }
                )

            records.append(
                {
                    "record_id": f"{table_id}:row_{row_index:03d}",
                    "record_type": "table_row",
                    "table_id": table_id,
                    "source_row_id": row["row_id"],
                    "column_count": table.get("column_count", 0),
                    "cells": cells,
                    "semantic_labels_assigned": False,
                    "classification": table.get(
                        "classification",
                        "geometric_candidate",
                    ),
                }
            )

    return records


def _build_sections(
    structure_result: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "section_id": section["section_id"],
            "structure_type": section["structure_type"],
            "row_start": section["row_start"],
            "row_end": section["row_end"],
            "row_ids": list(section["row_ids"]),
        }
        for section in structure_result.get("sections", [])
    ]


def _build_controls(
    document_id: str | None,
    structure_result: dict[str, Any],
) -> list[dict[str, Any]]:
    controls = []

    for region in structure_result.get("regions", []):
        if region.get("structure_type") != "control":
            continue

        controls.append(
            {
                "control_id": f"control_region_{region['region_id']}",
                "control_type": "checkbox_or_control_marker",
                "text": region.get("text", ""),
                "region_id": region["region_id"],
                "state": "unknown",
                "semantic_verified": False,
                "evidence_item_ids": [
                    _evidence_id(document_id, region["region_id"])
                ],
            }
        )

    return controls


def _build_free_text(
    document_id: str | None,
    structure_result: dict[str, Any],
) -> list[dict[str, Any]]:
    free_text = []

    allowed = {"heading", "paragraph", "free_text"}

    for region in structure_result.get("regions", []):
        structure_type = region.get("structure_type")

        if structure_type not in allowed:
            continue

        free_text.append(
            {
                "text_item_id": f"text_region_{region['region_id']}",
                "text": region.get("text", ""),
                "structure_type": structure_type,
                "region_id": region["region_id"],
                "row_id": region.get("row_id"),
                "block_id": region.get("block_id"),
                "evidence_item_ids": [
                    _evidence_id(document_id, region["region_id"])
                ],
            }
        )

    return free_text


def build_unified_structured_document(
    document: dict[str, Any],
) -> dict[str, Any]:
    """Build one structured document while preserving raw OCR provenance."""
    document_id = document.get("document_id")

    layout = analyze_document_layout_refined(document)
    table_result = reconstruct_candidate_tables(document)
    form_result = reconstruct_form_fields(document)
    structure_result = classify_document_structure(document)

    fields = _build_fields(document_id, form_result)
    records = _build_records(document_id, table_result)
    sections = _build_sections(structure_result)
    controls = _build_controls(document_id, structure_result)
    free_text = _build_free_text(document_id, structure_result)

    # Evidence is deliberately copied rather than transformed so the new
    # structured layer remains traceable to the exact OCR evidence.
    evidence = deepcopy(document.get("evidence", []))

    return {
        "document_id": document_id,
        "document_type": "mixed_structured_ocr",
        "source_document_type": document.get("document_type"),
        "structure_recognition": "automatic_geometric",
        "semantic_verification": "not_performed",
        "region_count": layout.get("region_count", 0),
        "row_count": layout.get("row_count", 0),
        "structure_types_present": list(
            structure_result.get("structure_types_present", [])
        ),
        "fields": fields,
        "records": records,
        "controls": controls,
        "sections": sections,
        "free_text": free_text,
        "evidence": evidence,
        "parsing_metadata": {
            "candidate_table_count": table_result.get(
                "candidate_table_count",
                0,
            ),
            "field_candidate_count": form_result.get(
                "field_candidate_count",
                0,
            ),
            "section_count": structure_result.get(
                "section_count",
                0,
            ),
            "structure_counts": deepcopy(
                structure_result.get("structure_counts", {})
            ),
            "ambiguity_policy": (
                "preserve_observed_candidates_without_semantic_resolution"
            ),
        },
    }
