"""Experiment 77: real-world receipt money OCR-noise regression."""
from __future__ import annotations
import json
from pathlib import Path
from parsing.unified_structured_document_builder import build_unified_structured_document
from services.structured_document_retrieval import retrieve_structured_answer

ROOT = Path(__file__).resolve().parent.parent
SUPERMARKET = ROOT / "results/gui_v4/single/supermarket/unified_raw_ocr.json"
SYNTHETIC = ROOT / "results/gui_v4/single/ChatGPT Image Sep 24, 2026, 06_13_58 PM/unified_raw_ocr.json"
NHI = ROOT / "results/gui_v4/single/WhatsApp Image 2026-09-22 at 22.21.24/unified_raw_ocr.json"

def load(p):
    return json.loads(p.read_text(encoding="utf-8"))

def query(doc, q):
    return retrieve_structured_answer(build_unified_structured_document(doc), q)

def main() -> int:
    print("=" * 112)
    print("EXPERIMENT 77: REAL-WORLD RECEIPT MONEY OCR-NOISE REGRESSION")
    print("=" * 112)
    for p in (SUPERMARKET, SYNTHETIC, NHI):
        if not p.is_file():
            print(f"[MISSING] {p}")
            return 1

    sm, syn, nhi = load(SUPERMARKET), load(SYNTHETIC), load(NHI)
    beef = query(sm, "What is the price of Ground Beef?")
    eggs = query(sm, "What is the price of Grade A Large Eggs?")
    bread = query(sm, "What is the price of whole wheat bread?")
    total = query(sm, "What is the total?")
    tax = query(sm, "What is the tax?")
    amb = query(syn, "What is the total?")
    date = query(nhi, "What is the payment date?")

    b = beef.get("selected_candidate") or {}
    e = eggs.get("selected_candidate") or {}
    w = bread.get("selected_candidate") or {}
    t = total.get("selected_candidate") or {}
    tx = tax.get("selected_candidate") or {}
    d = date.get("selected_candidate") or {}

    struct = build_unified_structured_document(sm)
    rows = [" | ".join(str(c.get("text", "")) for c in r.get("cells", [])) for r in struct.get("records", [])]

    checks = [
        ("Ground Beef becomes structured answer candidate", beef.get("status") == "structured_answer_candidate"),
        ("Ground Beef normalized value is $5.99", b.get("value") == "$5.99"),
        ("Ground Beef preserves observed value", b.get("observed_value") == ".$ 5.99"),
        ("Ground Beef normalization is marked", b.get("value_normalized_for_type") is True),
        ("Eggs become structured answer candidate", eggs.get("status") == "structured_answer_candidate"),
        ("Eggs normalized value is $3.99", e.get("value") == "$3.99"),
        ("Eggs preserve observed value", e.get("observed_value") == ".$ 3.99"),
        ("Eggs normalization is marked", e.get("value_normalized_for_type") is True),
        ("Original Ground Beef OCR remains unchanged", any(".$ 5.99" in row for row in rows)),
        ("Original Eggs OCR remains unchanged", any(".$ 3.99" in row for row in rows)),
        ("Whole Wheat Bread remains $2.49", w.get("value") == "$2.49"),
        ("Whole Wheat Bread needs no normalization", w.get("value_normalized_for_type") is False),
        ("Total remains $20.09", t.get("value") == "$20.09"),
        ("Tax remains $1.14", tx.get("value") == "$1.14"),
        ("Synthetic total remains ambiguous", amb.get("status") == "ambiguous_structured_evidence"),
        ("Synthetic ambiguity selects no value", amb.get("selected_candidate") is None),
        ("NHI payment date remains structured", date.get("status") == "structured_answer_candidate"),
        ("NHI payment date remains 115/09/21", d.get("value") == "115/09/21"),
    ]

    print("\nGround Beef:\n", beef)
    print("\nGrade A Large Eggs:\n", eggs)
    print("\n" + "-" * 112)
    passed = 0
    for i, (label, ok) in enumerate(checks, 1):
        print(f"[{'PASS' if ok else 'FAIL'}] {i:02d}. {label}")
        passed += int(ok)
    print("=" * 112)
    print(f"RESULT: {passed}/{len(checks)} checks passed")
    print("No OCR, Groq, or network requests were made.")
    return 0 if passed == len(checks) else 1

if __name__ == "__main__":
    raise SystemExit(main())
