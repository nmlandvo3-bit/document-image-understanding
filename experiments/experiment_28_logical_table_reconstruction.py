
"""
Experiment 28: Logical Table Reconstruction

Converts the candidate cells from Experiment 27 into a logical
table with row indices, column indices, and merged-cell spans.

Uses cell geometry rather than hard-coded timetable row positions.

Input:
    results/parsing/pic1/consolidated_table/
        consolidated_table_cells.json

Outputs:
    results/parsing/pic1/logical_table/
        logical_table.json
        table_structure.txt

Does not modify the original OCR results or previous experiments.
"""

import json
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "parsing"
    / "pic1"
    / "consolidated_table"
    / "consolidated_table_cells.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "parsing"
    / "pic1"
    / "logical_table"
)

# Detected borders vary slightly between neighboring cells.
# Endpoints within this distance will be grouped together.
BOUNDARY_CLUSTER_TOLERANCE = 5

# After grouping, an individual cell edge must be sufficiently
# close to the inferred logical boundary.
EDGE_SNAP_TOLERANCE = 7


def load_document() -> dict:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(INPUT_PATH)

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        document = json.load(file)

    if not document.get("cells"):
        raise ValueError("The input JSON contains no cells.")

    return document


def cluster_boundaries(
    positions: list[int],
    tolerance: int,
) -> list[int]:
    """
    Group nearby coordinate values into logical boundaries.

    Example:
        [329, 330, 331, 332, 333] -> one boundary

    Uses the median of each group as its representative
    position to reduce the influence of slightly displaced edges.
    """

    if not positions:
        raise ValueError("Cannot cluster an empty boundary list.")

    sorted_positions = sorted(positions)

    groups = [[sorted_positions[0]]]

    for position in sorted_positions[1:]:
        if position - groups[-1][-1] <= tolerance:
            groups[-1].append(position)
        else:
            groups.append([position])

    boundaries = []

    for group in groups:
        midpoint = len(group) // 2

        if len(group) % 2 == 1:
            representative = group[midpoint]
        else:
            representative = round(
                (group[midpoint - 1] + group[midpoint]) / 2
            )

        boundaries.append(int(representative))

    return boundaries


def infer_table_boundaries(cells: list[dict]) -> tuple[list[int], list[int]]:
    """Infer column and row boundaries from all detected cells."""

    x_positions = []
    y_positions = []

    for cell in cells:
        box = cell["bounding_box"]

        x_positions.extend([box["x"], box["x2"]])
        y_positions.extend([box["y"], box["y2"]])

    x_boundaries = cluster_boundaries(
        x_positions,
        BOUNDARY_CLUSTER_TOLERANCE,
    )

    y_boundaries = cluster_boundaries(
        y_positions,
        BOUNDARY_CLUSTER_TOLERANCE,
    )

    if len(x_boundaries) < 2 or len(y_boundaries) < 2:
        raise ValueError(
            "Could not infer enough row or column boundaries."
        )

    return x_boundaries, y_boundaries


def nearest_boundary_index(
    coordinate: int,
    boundaries: list[int],
    cell_id: int,
    edge_name: str,
) -> int:
    """
    Find the closest inferred boundary to a cell edge.

    Reject an edge that is too far away rather than silently
    assigning it to an incorrect row or column.
    """

    index = min(
        range(len(boundaries)),
        key=lambda i: abs(boundaries[i] - coordinate),
    )

    distance = abs(boundaries[index] - coordinate)

    if distance > EDGE_SNAP_TOLERANCE:
        raise ValueError(
            f"Cell {cell_id}: {edge_name}={coordinate} is "
            f"{distance} pixels from its nearest logical boundary "
            f"at {boundaries[index]}."
        )

    return index


def assign_logical_positions(
    cells: list[dict],
    x_boundaries: list[int],
    y_boundaries: list[int],
) -> list[dict]:
    """Assign each detected cell a row, column, and span."""

    positioned_cells = []

    for original_cell in cells:
        cell = deepcopy(original_cell)

        cell_id = cell["cell_id"]
        box = cell["bounding_box"]

        column_start = nearest_boundary_index(
            box["x"],
            x_boundaries,
            cell_id,
            "x",
        )

        column_end = nearest_boundary_index(
            box["x2"],
            x_boundaries,
            cell_id,
            "x2",
        )

        row_start = nearest_boundary_index(
            box["y"],
            y_boundaries,
            cell_id,
            "y",
        )

        row_end = nearest_boundary_index(
            box["y2"],
            y_boundaries,
            cell_id,
            "y2",
        )

        column_span = column_end - column_start
        row_span = row_end - row_start

        if column_span < 1 or row_span < 1:
            raise ValueError(
                f"Cell {cell_id} has an invalid logical span: "
                f"row_span={row_span}, column_span={column_span}."
            )

        cell["logical_position"] = {
            "row": row_start,
            "column": column_start,
            "row_span": row_span,
            "column_span": column_span,
        }

        positioned_cells.append(cell)

    positioned_cells.sort(
        key=lambda cell: (
            cell["logical_position"]["row"],
            cell["logical_position"]["column"],
        )
    )

    return positioned_cells


def validate_table(
    cells: list[dict],
    row_count: int,
    column_count: int,
) -> dict:
    """
    Check logical table coverage.

    Each logical position should belong to exactly one cell.
    A merged cell legitimately occupies multiple positions.
    """

    occupancy = [
        [None for _ in range(column_count)]
        for _ in range(row_count)
    ]

    collisions = []
    merged_cells = []

    for cell in cells:
        position = cell["logical_position"]

        row = position["row"]
        column = position["column"]
        row_span = position["row_span"]
        column_span = position["column_span"]

        if row_span > 1 or column_span > 1:
            merged_cells.append(cell["cell_id"])

        for occupied_row in range(row, row + row_span):
            for occupied_column in range(
                column,
                column + column_span,
            ):
                previous_cell = occupancy[
                    occupied_row
                ][occupied_column]

                if previous_cell is not None:
                    collisions.append(
                        {
                            "row": occupied_row,
                            "column": occupied_column,
                            "cell_ids": [
                                previous_cell,
                                cell["cell_id"],
                            ],
                        }
                    )
                else:
                    occupancy[
                        occupied_row
                    ][occupied_column] = cell["cell_id"]

    missing_positions = []

    for row in range(row_count):
        for column in range(column_count):
            if occupancy[row][column] is None:
                missing_positions.append(
                    {
                        "row": row,
                        "column": column,
                    }
                )

    return {
        "merged_cell_ids": merged_cells,
        "merged_cell_count": len(merged_cells),
        "collisions": collisions,
        "collision_count": len(collisions),
        "missing_positions": missing_positions,
        "missing_position_count": len(missing_positions),
        "occupancy": occupancy,
    }


def make_structure_report(
    cells: list[dict],
    row_count: int,
    column_count: int,
    validation: dict,
) -> str:
    """
    Create a readable table-structure report.

    Text previews are diagnostic only. The full text remains
    available in the JSON output.
    """

    lines = [
        "=" * 78,
        "EXPERIMENT 28: LOGICAL TABLE STRUCTURE",
        "=" * 78,
        f"Logical rows: {row_count}",
        f"Logical columns: {column_count}",
        f"Detected cells: {len(cells)}",
        f"Merged cells: {validation['merged_cell_count']}",
        f"Overlapping positions: {validation['collision_count']}",
        f"Missing positions: {validation['missing_position_count']}",
        "=" * 78,
    ]

    for row in range(row_count):
        lines.append(f"\nROW {row}")

        row_cells = [
            cell
            for cell in cells
            if cell["logical_position"]["row"] == row
        ]

        for cell in row_cells:
            position = cell["logical_position"]

            text_preview = (
                cell.get("text", "")
                .replace("\n", " ")
                .strip()
            )[:65]

            lines.append(
                f"  Cell {cell['cell_id']:>2} | "
                f"column={position['column']} | "
                f"row_span={position['row_span']} | "
                f"column_span={position['column_span']} | "
                f"{text_preview}"
            )

    lines.extend(
        [
            "",
            "=" * 78,
            "VALIDATION",
            "=" * 78,
            "Overlapping logical positions:",
            json.dumps(
                validation["collisions"],
                indent=2,
            ),
            "",
            "Missing logical positions:",
            json.dumps(
                validation["missing_positions"],
                indent=2,
            ),
            "=" * 78,
        ]
    )

    return "\n".join(lines)


def main() -> None:
    document = load_document()

    x_boundaries, y_boundaries = infer_table_boundaries(
        document["cells"]
    )

    positioned_cells = assign_logical_positions(
        document["cells"],
        x_boundaries,
        y_boundaries,
    )

    column_count = len(x_boundaries) - 1
    row_count = len(y_boundaries) - 1

    validation = validate_table(
        positioned_cells,
        row_count,
        column_count,
    )

    result = {
        "schema_version": "0.4-experimental",
        "document_type": "structured_table_candidate",
        "source": deepcopy(document["source"]),
        "image_dimensions": deepcopy(
            document["image_dimensions"]
        ),
        "table_geometry": {
            "row_count": row_count,
            "column_count": column_count,
            "x_boundaries": x_boundaries,
            "y_boundaries": y_boundaries,
            "row_and_column_indexing": "zero_based",
        },
        "statistics": {
            **deepcopy(document["statistics"]),
            "logical_rows": row_count,
            "logical_columns": column_count,
            "merged_cells": validation["merged_cell_count"],
            "overlapping_positions": validation["collision_count"],
            "missing_positions": validation[
                "missing_position_count"
            ],
        },
        "cells": positioned_cells,
        "logical_occupancy": validation["occupancy"],
        "validation": {
            "merged_cell_ids": validation["merged_cell_ids"],
            "collisions": validation["collisions"],
            "missing_positions": validation["missing_positions"],
        },
        "unassigned_regions": deepcopy(
            document["unassigned_regions"]
        ),
        "recovery_summary": deepcopy(
            document["recovery_summary"]
        ),
        "limitations": [
            "Logical positions are inferred from detected cell geometry.",
            "A cell spanning several rows or columns is represented "
            "by row_span or column_span.",
            "Geometric coverage does not guarantee that the text "
            "or table semantics are correct.",
            "Original cross-boundary OCR regions remain unassigned.",
            "This result has not been validated against a manually "
            "annotated ground-truth table.",
        ],
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = OUTPUT_DIR / "logical_table.json"
    report_path = OUTPUT_DIR / "table_structure.txt"

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    report = make_structure_report(
        positioned_cells,
        row_count,
        column_count,
        validation,
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    print("=" * 65)
    print("EXPERIMENT 28: LOGICAL TABLE RECONSTRUCTION")
    print("=" * 65)
    print(f"Logical rows:          {row_count}")
    print(f"Logical columns:       {column_count}")
    print(f"Candidate cells:       {len(positioned_cells)}")
    print(
        "Merged cells:          "
        f"{validation['merged_cell_count']}"
    )
    print(
        "Overlapping positions: "
        f"{validation['collision_count']}"
    )
    print(
        "Missing positions:     "
        f"{validation['missing_position_count']}"
    )
    print(f"JSON:   {json_path}")
    print(f"Report: {report_path}")
    print("=" * 65)

    if validation["collisions"] or validation["missing_positions"]:
        print(
            "WARNING: The logical table has overlapping or missing "
            "positions. Inspect the report before using the result."
        )
    else:
        print(
            "All logical positions are covered exactly once. "
            "Visual and text verification are still required."
        )

    print("EXPERIMENT 28 COMPLETE")


if __name__ == "__main__":
    main()