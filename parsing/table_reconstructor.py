"""Conservative automatic table reconstruction.

Experiment 69 revision 3.

Changes from revision 2:
- permits up to two sparse bridge rows after a strict table seed;
- requires the eventual continuation row to strongly cover the established
  columns (at least 3 aligned columns when the table has 3+ columns);
- performs only one bridge extension beyond the seed, preventing unrelated
  later sections from being absorbed into the same table candidate.

No OCR, LLM, API, or business-semantic inference is performed here.
"""

from __future__ import annotations

from statistics import median
from typing import Any

from parsing.layout_analyzer_refined import analyze_document_layout_refined


def _group_rows(layout: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    rows: dict[int, list[dict[str, Any]]] = {}
    for region in layout.get("regions", []):
        rows.setdefault(int(region["row_id"]), []).append(region)

    for row_id in rows:
        rows[row_id].sort(key=lambda r: float(r["column_position"]))

    return rows


def _cluster_positions(
    positions: list[float],
    tolerance: float = 0.09,
) -> list[float]:
    if not positions:
        return []

    clusters: list[list[float]] = []

    for position in sorted(positions):
        best_index = None
        best_distance = None

        for idx, cluster in enumerate(clusters):
            center = median(cluster)
            distance = abs(position - center)

            if distance <= tolerance:
                if best_distance is None or distance < best_distance:
                    best_index = idx
                    best_distance = distance

        if best_index is None:
            clusters.append([position])
        else:
            clusters[best_index].append(position)

    return [round(float(median(cluster)), 3) for cluster in clusters]


def _nearest_column(
    position: float,
    centers: list[float],
) -> tuple[int, float]:
    if not centers:
        return 0, 1.0

    ranked = sorted(
        (
            (abs(position - center), index)
            for index, center in enumerate(centers)
        ),
        key=lambda item: item[0],
    )

    distance, index = ranked[0]
    return index, distance


def _aligned_columns(
    row: list[dict[str, Any]],
    centers: list[float],
    *,
    max_assignment_distance: float = 0.12,
) -> set[int]:
    aligned: set[int] = set()

    for region in row:
        column_index, distance = _nearest_column(
            float(region["column_position"]),
            centers,
        )

        if distance <= max_assignment_distance:
            aligned.add(column_index)

    return aligned


def _row_signature(
    row: list[dict[str, Any]],
    centers: list[float],
    *,
    max_assignment_distance: float = 0.12,
) -> tuple[int, ...]:
    return tuple(
        sorted(
            _aligned_columns(
                row,
                centers,
                max_assignment_distance=max_assignment_distance,
            )
        )
    )


def _seed_runs(
    rows: dict[int, list[dict[str, Any]]],
    centers: list[float],
    *,
    min_columns: int = 2,
    min_rows: int = 3,
) -> list[list[int]]:
    signatures = {
        row_id: _row_signature(row, centers)
        for row_id, row in rows.items()
    }

    eligible = {
        row_id
        for row_id, signature in signatures.items()
        if len(signature) >= min_columns
    }

    runs: list[list[int]] = []
    current: list[int] = []
    previous = None

    for row_id in sorted(rows):
        if row_id not in eligible:
            if len(current) >= min_rows:
                runs.append(current)
            current = []
            previous = None
            continue

        if previous is None or row_id == previous + 1:
            current.append(row_id)
        else:
            if len(current) >= min_rows:
                runs.append(current)
            current = [row_id]

        previous = row_id

    if len(current) >= min_rows:
        runs.append(current)

    return runs


def _centers_for_run(
    rows: dict[int, list[dict[str, Any]]],
    run: list[int],
    *,
    tolerance: float = 0.09,
) -> list[float]:
    positions = [
        float(region["column_position"])
        for row_id in run
        for region in rows[row_id]
    ]

    return _cluster_positions(
        positions,
        tolerance=tolerance,
    )


def _required_strong_columns(column_count: int) -> int:
    if column_count <= 2:
        return 2
    return 3


def _extend_seed_once(
    rows: dict[int, list[dict[str, Any]]],
    seed: list[int],
    *,
    column_tolerance: float = 0.09,
    max_assignment_distance: float = 0.12,
    max_sparse_bridge_rows: int = 2,
) -> list[int]:
    """Bridge from a strict seed to one later strongly aligned row.

    This intentionally performs only one bridge extension. It prevents a
    valid table from recursively swallowing a later unrelated table/section.
    """
    centers = _centers_for_run(
        rows,
        seed,
        tolerance=column_tolerance,
    )

    required_strong = _required_strong_columns(
        len(centers)
    )

    start_row = seed[-1] + 1
    bridge_rows: list[int] = []

    for offset in range(max_sparse_bridge_rows + 1):
        row_id = start_row + offset
        row = rows.get(row_id)

        if row is None:
            return list(seed)

        aligned_count = len(
            _aligned_columns(
                row,
                centers,
                max_assignment_distance=max_assignment_distance,
            )
        )

        if aligned_count >= required_strong:
            return list(seed) + bridge_rows + [row_id]

        # Sparse bridge rows are allowed only if they are genuinely sparse.
        if aligned_count <= 1 and len(bridge_rows) < max_sparse_bridge_rows:
            bridge_rows.append(row_id)
            continue

        return list(seed)

    return list(seed)


def _reconstruct_run(
    rows: dict[int, list[dict[str, Any]]],
    run: list[int],
    *,
    column_tolerance: float = 0.09,
    max_assignment_distance: float = 0.12,
) -> dict[str, Any]:
    centers = _centers_for_run(
        rows,
        run,
        tolerance=column_tolerance,
    )

    table_rows = []

    for row_id in run:
        cells_by_column: dict[int, list[dict[str, Any]]] = {}

        for region in rows[row_id]:
            column_index, distance = _nearest_column(
                float(region["column_position"]),
                centers,
            )

            if distance > max_assignment_distance:
                continue

            cells_by_column.setdefault(
                column_index,
                [],
            ).append(region)

        cells = []

        for column_index in range(len(centers)):
            regions = cells_by_column.get(
                column_index,
                [],
            )

            if not regions:
                cells.append(
                    {
                        "column_index": column_index,
                        "column_position": centers[column_index],
                        "region_ids": [],
                        "text": "",
                        "confidence": None,
                    }
                )
                continue

            regions.sort(
                key=lambda region: float(
                    region["bbox"]["x"]
                )
            )

            text = " ".join(
                str(region["text"]).strip()
                for region in regions
                if str(region["text"]).strip()
            )

            confidences = [
                float(region["confidence"])
                for region in regions
                if isinstance(
                    region.get("confidence"),
                    (int, float),
                )
            ]

            cells.append(
                {
                    "column_index": column_index,
                    "column_position": centers[column_index],
                    "region_ids": [
                        region["region_id"]
                        for region in regions
                    ],
                    "text": text,
                    "confidence": (
                        round(
                            sum(confidences) / len(confidences),
                            4,
                        )
                        if confidences
                        else None
                    ),
                }
            )

        table_rows.append(
            {
                "row_id": row_id,
                "cells": cells,
                "nonempty_cell_count": sum(
                    bool(cell["text"])
                    for cell in cells
                ),
            }
        )

    return {
        "row_ids": run,
        "column_count": len(centers),
        "column_positions": centers,
        "rows": table_rows,
    }


def reconstruct_candidate_tables(
    document: dict[str, Any],
    *,
    min_columns: int = 2,
    min_rows: int = 3,
    column_tolerance: float = 0.09,
    max_assignment_distance: float = 0.12,
) -> dict[str, Any]:
    layout = analyze_document_layout_refined(
        document
    )

    rows = _group_rows(layout)

    all_positions = [
        float(region["column_position"])
        for region in layout.get("regions", [])
    ]

    global_centers = _cluster_positions(
        all_positions,
        tolerance=column_tolerance,
    )

    seeds = _seed_runs(
        rows,
        global_centers,
        min_columns=min_columns,
        min_rows=min_rows,
    )

    tables = []

    for index, seed in enumerate(
        seeds,
        start=1,
    ):
        run = _extend_seed_once(
            rows,
            seed,
            column_tolerance=column_tolerance,
            max_assignment_distance=max_assignment_distance,
            max_sparse_bridge_rows=2,
        )

        reconstructed = _reconstruct_run(
            rows,
            run,
            column_tolerance=column_tolerance,
            max_assignment_distance=max_assignment_distance,
        )

        if reconstructed["column_count"] < min_columns:
            continue

        reconstructed["table_id"] = (
            f"candidate_table_{index}"
        )
        reconstructed["classification"] = (
            "geometric_candidate"
        )
        reconstructed[
            "semantic_labels_assigned"
        ] = False
        reconstructed["seed_row_ids"] = seed

        tables.append(reconstructed)

    return {
        "document_id": layout.get("document_id"),
        "region_count": layout.get(
            "region_count",
            0,
        ),
        "row_count": layout.get(
            "row_count",
            0,
        ),
        "candidate_table_count": len(tables),
        "tables": tables,
    }
