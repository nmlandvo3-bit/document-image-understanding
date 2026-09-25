"""Refined OCR row segmentation for Experiment 68.

This module builds on Experiment 67 but uses stricter row membership to reduce
vertical chain-merging. It does not infer tables, forms, or semantic fields.

No OCR, LLM, or API calls are performed here.
"""

from __future__ import annotations

from statistics import median
from typing import Any

from parsing.layout_analyzer import (
    _bbox,
    _horizontal_overlap,
    extract_ocr_regions,
)


def _pair_row_compatibility(
    a: dict[str, float],
    b: dict[str, float],
    *,
    center_factor: float = 0.38,
    overlap_threshold: float = 0.55,
) -> tuple[bool, float, float]:
    """Return whether two boxes are plausibly on the same visual text row."""
    center_distance = abs(a["cy"] - b["cy"])
    min_height = max(1.0, min(a["height"], b["height"]))

    overlap = max(
        0.0,
        min(a["y2"], b["y2"]) - max(a["y"], b["y"]),
    )
    vertical_overlap = overlap / min_height

    compatible = (
        center_distance <= center_factor * min_height
        or vertical_overlap >= overlap_threshold
    )

    return compatible, center_distance, vertical_overlap


def assign_rows_refined(
    regions: list[dict[str, Any]],
    *,
    center_factor: float = 0.38,
    overlap_threshold: float = 0.55,
) -> dict[int | str, int]:
    """Group regions using row representatives instead of overlap chaining.

    Important difference from Experiment 67:
    a new region must be compatible with the row's robust representative
    geometry, not merely overlap one member of an already-growing row.
    """
    if not regions:
        return {}

    ordered = sorted(
        regions,
        key=lambda r: (
            r["_layout_bbox"]["cy"],
            r["_layout_bbox"]["x"],
        ),
    )

    rows: list[list[dict[str, Any]]] = []

    for region in ordered:
        rb = region["_layout_bbox"]

        best_index = None
        best_score = None

        for idx, row in enumerate(rows):
            centers = [item["_layout_bbox"]["cy"] for item in row]
            heights = [item["_layout_bbox"]["height"] for item in row]

            representative = {
                "cy": median(centers),
                "height": median(heights),
            }
            representative["y"] = (
                representative["cy"] - representative["height"] / 2.0
            )
            representative["y2"] = (
                representative["cy"] + representative["height"] / 2.0
            )

            compatible, center_distance, overlap = _pair_row_compatibility(
                rb,
                representative,
                center_factor=center_factor,
                overlap_threshold=overlap_threshold,
            )

            if not compatible:
                continue

            # Prefer the closest vertical-center row, with overlap as tie-breaker.
            score = (
                center_distance,
                -overlap,
            )

            if best_score is None or score < best_score:
                best_score = score
                best_index = idx

        if best_index is None:
            rows.append([region])
        else:
            rows[best_index].append(region)

    rows.sort(
        key=lambda row: median(
            item["_layout_bbox"]["cy"]
            for item in row
        )
    )

    mapping: dict[int | str, int] = {}

    for row_id, row in enumerate(rows, start=1):
        row.sort(key=lambda item: item["_layout_bbox"]["x"])

        for region in row:
            mapping[region["region_id"]] = row_id

    return mapping


def assign_blocks_refined(
    regions: list[dict[str, Any]],
    row_map: dict[int | str, int],
    *,
    gap_factor: float = 2.2,
) -> dict[int | str, int]:
    """Create coarse blocks from refined rows."""
    if not regions:
        return {}

    grouped: dict[int, list[dict[str, Any]]] = {}

    for region in regions:
        grouped.setdefault(
            row_map[region["region_id"]],
            [],
        ).append(region)

    row_bounds = []
    heights = []

    for row_id, items in sorted(grouped.items()):
        top = min(item["_layout_bbox"]["y"] for item in items)
        bottom = max(item["_layout_bbox"]["y2"] for item in items)

        row_bounds.append(
            (
                row_id,
                top,
                bottom,
            )
        )

        heights.extend(
            item["_layout_bbox"]["height"]
            for item in items
        )

    typical_height = median(heights) if heights else 1.0
    max_gap = max(
        4.0,
        typical_height * gap_factor,
    )

    row_to_block = {}
    block_id = 1
    previous_bottom = None

    for row_id, top, bottom in row_bounds:
        if (
            previous_bottom is not None
            and top - previous_bottom > max_gap
        ):
            block_id += 1

        row_to_block[row_id] = block_id
        previous_bottom = bottom

    return {
        region["region_id"]:
            row_to_block[
                row_map[
                    region["region_id"]
                ]
            ]
        for region in regions
    }


def _same_row_neighbors(
    region: dict[str, Any],
    regions: list[dict[str, Any]],
    row_map: dict[int | str, int],
) -> tuple[list[int | str], list[int | str]]:
    rid = region["region_id"]
    rb = region["_layout_bbox"]
    row_id = row_map[rid]

    peers = [
        other
        for other in regions
        if (
            other["region_id"] != rid
            and row_map[other["region_id"]] == row_id
        )
    ]

    left = sorted(
        (
            (
                rb["x"] - other["_layout_bbox"]["x2"],
                other["region_id"],
            )
            for other in peers
            if other["_layout_bbox"]["x2"] <= rb["x"]
        ),
        key=lambda item: item[0],
    )

    right = sorted(
        (
            (
                other["_layout_bbox"]["x"] - rb["x2"],
                other["region_id"],
            )
            for other in peers
            if other["_layout_bbox"]["x"] >= rb["x2"]
        ),
        key=lambda item: item[0],
    )

    return (
        [item[1] for item in left[:2]],
        [item[1] for item in right[:2]],
    )


def _vertical_neighbors(
    region: dict[str, Any],
    regions: list[dict[str, Any]],
    row_map: dict[int | str, int],
) -> tuple[list[int | str], list[int | str]]:
    rid = region["region_id"]
    rb = region["_layout_bbox"]

    above = []
    below = []

    for other in regions:
        if other["region_id"] == rid:
            continue

        ob = other["_layout_bbox"]

        aligned = (
            _horizontal_overlap(rb, ob) >= 0.30
            or rb["x"] <= ob["cx"] <= rb["x2"]
            or ob["x"] <= rb["cx"] <= ob["x2"]
        )

        if not aligned:
            continue

        other_row = row_map[other["region_id"]]
        current_row = row_map[rid]

        if other_row < current_row:
            gap = max(
                0.0,
                rb["y"] - ob["y2"],
            )

            above.append(
                (
                    gap,
                    abs(rb["cx"] - ob["cx"]),
                    other["region_id"],
                )
            )

        elif other_row > current_row:
            gap = max(
                0.0,
                ob["y"] - rb["y2"],
            )

            below.append(
                (
                    gap,
                    abs(rb["cx"] - ob["cx"]),
                    other["region_id"],
                )
            )

    above.sort()
    below.sort()

    return (
        [item[2] for item in above[:2]],
        [item[2] for item in below[:2]],
    )


def analyze_document_layout_refined(
    document: dict[str, Any],
) -> dict[str, Any]:
    """Return refined row/block layout without semantic classification."""
    regions = extract_ocr_regions(document)

    if not regions:
        return {
            "document_id": document.get("document_id"),
            "region_count": 0,
            "row_count": 0,
            "block_count": 0,
            "regions": [],
        }

    row_map = assign_rows_refined(regions)
    block_map = assign_blocks_refined(
        regions,
        row_map,
    )

    min_x = min(
        region["_layout_bbox"]["x"]
        for region in regions
    )

    max_x2 = max(
        region["_layout_bbox"]["x2"]
        for region in regions
    )

    page_width = max(
        1.0,
        max_x2 - min_x,
    )

    enriched = []

    for region in regions:
        rid = region["region_id"]
        rb = region["_layout_bbox"]

        left, right = _same_row_neighbors(
            region,
            regions,
            row_map,
        )

        above, below = _vertical_neighbors(
            region,
            regions,
            row_map,
        )

        enriched.append(
            {
                "region_id": rid,
                "text": str(
                    region.get(
                        "text",
                        "",
                    )
                ).strip(),
                "bbox": {
                    "x": rb["x"],
                    "y": rb["y"],
                    "width": rb["width"],
                    "height": rb["height"],
                },
                "confidence": region.get("confidence"),
                "row_id": row_map[rid],
                "block_id": block_map[rid],
                "column_position": round(
                    (
                        rb["cx"] - min_x
                    )
                    / page_width,
                    3,
                ),
                "neighbors_left": left,
                "neighbors_right": right,
                "neighbors_above": above,
                "neighbors_below": below,
            }
        )

    return {
        "document_id": document.get("document_id"),
        "region_count": len(enriched),
        "row_count": len(set(row_map.values())),
        "block_count": len(set(block_map.values())),
        "regions": enriched,
    }
