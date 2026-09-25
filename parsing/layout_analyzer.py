"""Generic OCR layout analysis.

This module is intentionally conservative. It does NOT classify a page as a
table or form and does NOT create semantic label/value relationships.

Input:
    A unified raw-OCR document such as the output of Experiment 49.

Output:
    OCR regions enriched with:
    - row_id
    - column_position
    - local neighbors
    - block_id

No OCR, LLM, or API calls are performed here.
"""

from __future__ import annotations

from statistics import median
from typing import Any


def _bbox(region: dict[str, Any]) -> dict[str, float] | None:
    raw = region.get("bbox")
    if not isinstance(raw, dict):
        return None
    try:
        x = float(raw["x"])
        y = float(raw["y"])
        width = float(raw.get("width", raw.get("w")))
        height = float(raw.get("height", raw.get("h")))
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "x2": x + width,
        "y2": y + height,
        "cx": x + width / 2.0,
        "cy": y + height / 2.0,
    }


def _vertical_overlap(a: dict[str, float], b: dict[str, float]) -> float:
    overlap = max(0.0, min(a["y2"], b["y2"]) - max(a["y"], b["y"]))
    denom = min(a["height"], b["height"])
    return overlap / denom if denom > 0 else 0.0


def _horizontal_overlap(a: dict[str, float], b: dict[str, float]) -> float:
    overlap = max(0.0, min(a["x2"], b["x2"]) - max(a["x"], b["x"]))
    denom = min(a["width"], b["width"])
    return overlap / denom if denom > 0 else 0.0


def extract_ocr_regions(document: dict[str, Any]) -> list[dict[str, Any]]:
    regions = []
    for item in document.get("evidence", []):
        if not isinstance(item, dict):
            continue
        if item.get("evidence_type") != "ocr_region":
            continue
        if item.get("meaningful") is False:
            continue
        if not str(item.get("text", "")).strip():
            continue
        box = _bbox(item)
        if box is None:
            continue
        region = dict(item)
        region["_layout_bbox"] = box
        regions.append(region)

    regions.sort(key=lambda r: (r["_layout_bbox"]["cy"], r["_layout_bbox"]["x"]))
    return regions


def assign_rows(
    regions: list[dict[str, Any]],
    *,
    center_tolerance_factor: float = 0.55,
    min_vertical_overlap: float = 0.35,
) -> dict[int | str, int]:
    if not regions:
        return {}

    typical_height = median(r["_layout_bbox"]["height"] for r in regions)
    center_tolerance = max(2.0, typical_height * center_tolerance_factor)
    rows: list[list[dict[str, Any]]] = []

    for region in regions:
        rb = region["_layout_bbox"]
        best_row_index = None
        best_distance = None

        for idx, row in enumerate(rows):
            row_center = sum(x["_layout_bbox"]["cy"] for x in row) / len(row)
            center_distance = abs(rb["cy"] - row_center)
            max_overlap = max(
                (_vertical_overlap(rb, other["_layout_bbox"]) for other in row),
                default=0.0,
            )

            if center_distance <= center_tolerance or max_overlap >= min_vertical_overlap:
                if best_distance is None or center_distance < best_distance:
                    best_row_index = idx
                    best_distance = center_distance

        if best_row_index is None:
            rows.append([region])
        else:
            rows[best_row_index].append(region)

    rows.sort(key=lambda row: sum(r["_layout_bbox"]["cy"] for r in row) / len(row))

    mapping = {}
    for row_id, row in enumerate(rows, start=1):
        row.sort(key=lambda r: r["_layout_bbox"]["x"])
        for region in row:
            mapping[region["region_id"]] = row_id
    return mapping


def assign_blocks(
    regions: list[dict[str, Any]],
    row_map: dict[int | str, int],
    *,
    gap_factor: float = 2.2,
) -> dict[int | str, int]:
    if not regions:
        return {}

    rows: dict[int, list[dict[str, Any]]] = {}
    for region in regions:
        rows.setdefault(row_map[region["region_id"]], []).append(region)

    row_bounds = []
    heights = []
    for row_id, items in sorted(rows.items()):
        top = min(i["_layout_bbox"]["y"] for i in items)
        bottom = max(i["_layout_bbox"]["y2"] for i in items)
        row_bounds.append((row_id, top, bottom))
        heights.extend(i["_layout_bbox"]["height"] for i in items)

    typical_height = median(heights) if heights else 1.0
    max_gap = max(4.0, typical_height * gap_factor)

    row_to_block = {}
    block_id = 1
    previous_bottom = None
    for row_id, top, bottom in row_bounds:
        if previous_bottom is not None and (top - previous_bottom) > max_gap:
            block_id += 1
        row_to_block[row_id] = block_id
        previous_bottom = bottom

    return {
        region["region_id"]: row_to_block[row_map[region["region_id"]]]
        for region in regions
    }


def _nearest_same_row(region, regions, row_map):
    rid = region["region_id"]
    rb = region["_layout_bbox"]
    row_id = row_map[rid]
    peers = [
        other for other in regions
        if other["region_id"] != rid
        and row_map[other["region_id"]] == row_id
    ]

    left = sorted(
        (
            (rb["x"] - other["_layout_bbox"]["x2"], other["region_id"])
            for other in peers
            if other["_layout_bbox"]["x2"] <= rb["x"]
        ),
        key=lambda x: x[0],
    )
    right = sorted(
        (
            (other["_layout_bbox"]["x"] - rb["x2"], other["region_id"])
            for other in peers
            if other["_layout_bbox"]["x"] >= rb["x2"]
        ),
        key=lambda x: x[0],
    )
    return [x[1] for x in left[:2]], [x[1] for x in right[:2]]


def _nearest_vertical(region, regions, row_map):
    rid = region["region_id"]
    rb = region["_layout_bbox"]
    above, below = [], []

    for other in regions:
        if other["region_id"] == rid:
            continue
        ob = other["_layout_bbox"]
        h_overlap = _horizontal_overlap(rb, ob)
        aligned = (
            h_overlap >= 0.30
            or rb["x"] <= ob["cx"] <= rb["x2"]
            or ob["x"] <= rb["cx"] <= ob["x2"]
        )
        if not aligned:
            continue

        if row_map[other["region_id"]] < row_map[rid]:
            gap = max(0.0, rb["y"] - ob["y2"])
            above.append((gap, abs(rb["cx"] - ob["cx"]), other["region_id"]))
        elif row_map[other["region_id"]] > row_map[rid]:
            gap = max(0.0, ob["y"] - rb["y2"])
            below.append((gap, abs(rb["cx"] - ob["cx"]), other["region_id"]))

    above.sort()
    below.sort()
    return [x[2] for x in above[:2]], [x[2] for x in below[:2]]


def analyze_document_layout(document: dict[str, Any]) -> dict[str, Any]:
    regions = extract_ocr_regions(document)
    if not regions:
        return {
            "document_id": document.get("document_id"),
            "region_count": 0,
            "row_count": 0,
            "block_count": 0,
            "regions": [],
        }

    row_map = assign_rows(regions)
    block_map = assign_blocks(regions, row_map)

    min_x = min(r["_layout_bbox"]["x"] for r in regions)
    max_x2 = max(r["_layout_bbox"]["x2"] for r in regions)
    page_width = max(1.0, max_x2 - min_x)

    enriched = []
    for region in regions:
        rid = region["region_id"]
        rb = region["_layout_bbox"]
        left, right = _nearest_same_row(region, regions, row_map)
        above, below = _nearest_vertical(region, regions, row_map)

        enriched.append({
            "region_id": rid,
            "text": str(region.get("text", "")).strip(),
            "bbox": {
                "x": rb["x"],
                "y": rb["y"],
                "width": rb["width"],
                "height": rb["height"],
            },
            "confidence": region.get("confidence"),
            "row_id": row_map[rid],
            "block_id": block_map[rid],
            "column_position": round((rb["cx"] - min_x) / page_width, 3),
            "neighbors_left": left,
            "neighbors_right": right,
            "neighbors_above": above,
            "neighbors_below": below,
        })

    return {
        "document_id": document.get("document_id"),
        "region_count": len(enriched),
        "row_count": len(set(row_map.values())),
        "block_count": len(set(block_map.values())),
        "regions": enriched,
    }
