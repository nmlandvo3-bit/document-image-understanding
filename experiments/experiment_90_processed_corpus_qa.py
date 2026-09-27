"""Experiment 90: load saved scans and answer corpus questions end to end.

Run without arguments for regression checks, or add --question for JSON answers:
python -m experiments.experiment_90_processed_corpus_qa --question "Which receipt has receipt number 004512?"
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from services.corpus_question_router import answer_corpus_question
from services.processed_scan_corpus import load_processed_scan_corpus

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "results/gui_v4/single"


def file_hashes(root: Path) -> dict[str, str]:
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("unified_raw_ocr.json")}


def regression(root: Path) -> int:
    before = file_hashes(root)
    corpus = load_processed_scan_corpus(root)
    ask = lambda question: answer_corpus_question(corpus, question)
    campus = ask("Which receipt has receipt number 004512?")
    market = ask("Which document has heading FRESH MART SUPERMARKET?")
    date = ask("What is the date on receipt number 004512?")
    total = ask("What is the total for FRESH MART SUPERMARKET?")
    listing = ask("List the totals of all receipts")
    highest = ask("Which receipt has the highest total?")
    unknown = ask("What is the total on receipt number 999999?")
    campus_id = "scan_receipt_4c095626a778"
    market_id = "scan_supermarket_b79744e73844"

    checks = [
        ("all six processed scans loaded", len(corpus["documents"]) == 6 and len(corpus["by_id"]) == 6),
        ("six source paths retained", len(corpus["source_paths"]) == 6 and all(Path(path).is_file() for path in corpus["source_paths"].values())),
        ("campus receipt found", campus["status"] == "one_match" and campus["answer"] == campus_id),
        ("campus receipt evidence preserved", bool(campus.get("matches")) and
         any(item.startswith(campus_id + ":ocr_region:")
             for item in campus["matches"][0]["evidence_item_ids"])),
        ("supermarket heading found", market["status"] == "one_match" and market["answer"] == market_id),
        ("campus date answered", date["status"] == "answered" and date["answer"] == "2026-09-14" and date["document_id"] == campus_id),
        ("supermarket total answered with provenance", total["status"] == "answered" and total["answer"] == "$20.09" and total["document_id"] == market_id and len(total["evidence_item_ids"]) == 2),
        ("two receipt totals listed", listing["status"] == "answered" and len(listing["totals"]) == 2),
        ("unknown currency blocks ranking", highest["status"] == "not_comparable" and highest.get("reason") == "currency_unknown" and highest.get("answer") is None),
        ("missing receipt does not select a value", unknown["status"] == "document_not_found" and unknown["answer"] is None),
        ("source OCR JSON files unchanged", before == file_hashes(root)),
    ]
    print("EXPERIMENT 90: PROCESSED CORPUS QA")
    for number, (label, passed) in enumerate(checks, 1):
        print(f"[{'PASS' if passed else 'FAIL'}] {number:02d}. {label}")
    count = sum(passed for _, passed in checks)
    print(f"RESULT: {count}/{len(checks)} checks passed")
    print("No OCR, Groq, or network requests were made.")
    return 0 if count == len(checks) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Directory containing saved unified_raw_ocr.json scans")
    parser.add_argument("--question", action="append", default=[],
                        help="Ask a supported corpus question; repeat for more questions")
    args = parser.parse_args()
    if args.question:
        corpus = load_processed_scan_corpus(args.root)
        for question in args.question:
            print(json.dumps({"question": question,
                              "result": answer_corpus_question(corpus, question)},
                             ensure_ascii=False))
        return 0
    return regression(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
