"""Load previously saved OCR scans into the local structured document corpus.

This reads JSON and builds structures in memory. It does not run OCR, call an
LLM, access the network, or rewrite the processed scans.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from parsing.unified_structured_document_builder import build_unified_structured_document
from services.structured_document_corpus import index_structured_documents


def load_processed_scan_corpus(root: str | Path) -> dict[str, Any]:
    """Index all unified_raw_ocr.json files beneath an existing directory.

    Fail on an empty directory or duplicate document IDs. A corrupt JSON scan
    raises an error naming its path, so no partial corpus is returned.
    """
    directory = Path(root)
    if not directory.is_dir():
        raise FileNotFoundError(f"Processed scan directory does not exist: {directory}")
    paths = sorted(directory.rglob("unified_raw_ocr.json"), key=lambda p: str(p).casefold())
    if not paths:
        raise ValueError(f"No unified_raw_ocr.json files found under: {directory}")

    documents = []
    sources = {}
    for path in paths:
        try:
            raw_document = json.loads(path.read_text(encoding="utf-8"))
            structured = build_unified_structured_document(raw_document)
        except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"Could not build structured document from {path}: {exc}") from exc
        documents.append(structured)
        sources[structured.get("document_id")] = str(path)

    index = index_structured_documents(documents)
    index["source_paths"] = sources
    return index
