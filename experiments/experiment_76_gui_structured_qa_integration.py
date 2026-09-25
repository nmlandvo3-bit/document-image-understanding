"""Experiment 76: GUI structured-first Q&A integration regression.

Static/source-level checks only. No Tk mainloop, OCR, Groq, or network calls.

Validates that:
- scans still generate unified_raw_ocr.json;
- the current scan remains the Q&A document scope;
- DocumentQAService is still lazy-loaded from the GUI;
- the GUI now communicates structured-first retrieval with raw OCR fallback;
- the existing request-id protection remains present.
"""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = PROJECT_ROOT / "app.py"


def main() -> int:
    print("=" * 108)
    print("EXPERIMENT 76: GUI STRUCTURED-FIRST Q&A INTEGRATION")
    print("=" * 108)

    if not APP_PATH.is_file():
        print(f"[MISSING] {APP_PATH}")
        return 1

    source = APP_PATH.read_text(encoding="utf-8")
    ast.parse(source)

    checks = [
        (
            "app.py parses successfully",
            True,
        ),
        (
            "single scan still creates unified_raw_ocr.json",
            'qa_path = output_dir / "unified_raw_ocr.json"' in source,
        ),
        (
            "single scan still stores Q&A JSON path",
            "self.single_qa_json_path = qa_path.resolve()" in source,
        ),
        (
            "Ask About This Scan still uses current scan path",
            "path = self.single_qa_json_path" in source,
        ),
        (
            "current scan becomes active Q&A scope",
            "self.qa_document_path = path" in source,
        ),
        (
            "Q&A service still lazy-loads DocumentQAService",
            "from services.document_qa_service import DocumentQAService" in source,
        ),
        (
            "Q&A passes selected path to service",
            "DocumentQAService(document_paths=paths)" in source,
        ),
        (
            "GUI communicates automatic structured retrieval first",
            "automatic structured retrieval first" in source,
        ),
        (
            "GUI communicates raw OCR fallback",
            "raw OCR fallback when needed" in source,
        ),
        (
            "question status reflects document-structure analysis",
            "Analyzing document structure and retrieving grounded evidence..." in source,
        ),
        (
            "request counter protection remains",
            "self.qa_request_counter += 1" in source,
        ),
        (
            "active request binds question and scope",
            "self.qa_active_request = (request_id, question, self.qa_document_path)" in source,
        ),
    ]

    passed = 0

    print()
    print("-" * 108)
    print("GUI INTEGRATION CHECKS")
    print("-" * 108)

    for index, (label, ok) in enumerate(checks, start=1):
        print(f"[{'PASS' if ok else 'FAIL'}] {index:02d}. {label}")
        if ok:
            passed += 1

    print()
    print("=" * 108)
    print(f"RESULT: {passed}/{len(checks)} GUI integration checks passed")
    print("=" * 108)
    print("No OCR, Groq, or network requests were made.")

    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
