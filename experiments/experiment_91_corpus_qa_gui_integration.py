"""Experiment 91: local GUI corpus answer presentation and wiring."""
from __future__ import annotations

import ast
from pathlib import Path

from services.corpus_qa_presentation import present_corpus_answer
from services.corpus_question_router import answer_corpus_question
from services.processed_scan_corpus import load_processed_scan_corpus

PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "results/gui_v4/single"


def main() -> int:
    if not (ROOT / "receipt/unified_raw_ocr.json").is_file():
        print(f"[MISSING] Processed receipt scan under {ROOT}")
        return 1
    index = load_processed_scan_corpus(ROOT)
    ask = lambda question: present_corpus_answer(answer_corpus_question(index, question))
    field_answer, field_evidence = ask("What is the total for FRESH MART SUPERMARKET?")
    listing_answer, listing_evidence = ask("List the totals of all receipts")
    highest_answer, highest_evidence = ask("Which receipt has the highest total?")
    unsupported_answer, _ = ask("Who bought the coffee?")
    conflict_answer, conflict_evidence = present_corpus_answer({
        "status": "ambiguous_field", "document_id": "synthetic",
        "candidates": [
            {"observed_value": "$10", "evidence_item_ids": ["synthetic:ocr_region:1"]},
            {"observed_value": "$12", "evidence_item_ids": ["synthetic:ocr_region:2"]},
        ],
    })

    gui_path = PROJECT / "app.py"
    gui_source = gui_path.read_text(encoding="utf-8")
    app = next(n for n in ast.parse(gui_source).body
               if isinstance(n, ast.ClassDef) and n.name == "DocumentUnderstandingApp")
    methods = {n.name: n for n in app.body if isinstance(n, ast.FunctionDef)}
    local_worker = ast.get_source_segment(gui_source, methods["process_corpus_question"])
    checks = [
        ("GUI includes local corpus tab", "Corpus Q&A (local)" in gui_source and "create_corpus_qa_mode" in methods),
        ("existing document QA remains", "create_document_qa_mode" in methods and "start_document_question" in methods),
        ("local worker loads saved scans and routes question", "load_processed_scan_corpus" in local_worker and "answer_corpus_question" in local_worker),
        ("local worker does not call Groq", "Groq" not in local_worker and "DocumentQAService" not in local_worker),
        ("field answer is clearly a candidate", "Candidate: $20.09" in field_answer and "scan_supermarket_" in field_answer),
        ("field evidence includes both OCR IDs", field_evidence.count("scan_supermarket_b79744e73844:ocr_region:") == 2),
        ("listing names both receipts", "scan_receipt_4c095626a778" in listing_answer and "scan_supermarket_b79744e73844" in listing_answer),
        ("listing retains OCR evidence", "scan_receipt_4c095626a778:ocr_region:" in listing_evidence and "scan_supermarket_b79744e73844:ocr_region:" in listing_evidence),
        ("unknown currency blocks highest total", "Cannot compare" in highest_answer and "no known currency" in highest_answer and "scan_receipt_4c095626a778:ocr_region:" in highest_evidence),
        ("unsupported question is explained", "Unsupported question" in unsupported_answer),
        ("ambiguous values are not selected", "No value was selected" in conflict_answer and all(x in conflict_evidence for x in ("synthetic:ocr_region:1", "synthetic:ocr_region:2"))),
        ("display states generated structure is unverified", "unverified" in field_evidence and "unverified" in highest_evidence),
    ]
    print("EXPERIMENT 91: LOCAL CORPUS GUI")
    for number, (label, passed) in enumerate(checks, 1):
        print(f"[{'PASS' if passed else 'FAIL'}] {number:02d}. {label}")
    count = sum(passed for _, passed in checks)
    print(f"RESULT: {count}/{len(checks)} checks passed")
    print("No OCR, Groq, or network requests were made.")
    return 0 if count == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
