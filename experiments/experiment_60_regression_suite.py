from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from services.layout_aware_ocr_retrieval import build_raw_ocr_packet

ROOT = Path("results")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_raw_doc(folder_name: str) -> tuple[Path, dict[str, Any]]:
    matches = [
        p for p in ROOT.rglob("unified_raw_ocr.json")
        if p.parent.name == folder_name
    ]
    if not matches:
        raise FileNotFoundError(
            f"Could not find unified_raw_ocr.json under a folder named {folder_name!r}"
        )
    path = matches[0]
    return path, load_json(path)


def norm(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def evidence_texts(packet: dict[str, Any]) -> list[str]:
    out = []
    for item in packet.get("evidence", []):
        content = item.get("content", {})
        out.append(str(content.get("text", "")).strip())
    return out


def contains_text(texts: list[str], expected: str) -> bool:
    e = norm(expected)
    return any(e == norm(t) or e in norm(t) for t in texts)


def run_case(
    *,
    name: str,
    document: dict[str, Any],
    question: str,
    must_include: list[str],
    ambiguous: bool,
) -> tuple[bool, str]:
    packet = build_raw_ocr_packet(question, [document])
    texts = evidence_texts(packet)
    ambiguities = packet.get("ambiguities", []) or []

    missing = [x for x in must_include if not contains_text(texts, x)]
    ambiguity_ok = bool(ambiguities) == ambiguous
    passed = not missing and ambiguity_ok

    detail = (
        f"question={question!r}\n"
        f"  answer_mode={packet.get('answer_mode')}\n"
        f"  evidence={texts}\n"
        f"  ambiguities={ambiguities}\n"
    )
    if missing:
        detail += f"  MISSING EXPECTED TEXT: {missing}\n"
    if not ambiguity_ok:
        detail += (
            f"  AMBIGUITY MISMATCH: expected ambiguous={ambiguous}, "
            f"got {bool(ambiguities)}\n"
        )

    return passed, detail


def main() -> int:
    print("=" * 88)
    print("EXPERIMENT 60: RAW-OCR RETRIEVAL REGRESSION SUITE")
    print("=" * 88)

    docs: dict[str, dict[str, Any]] = {}

    sources = {
        "receipt": "receipt",
        "pic13": "pic13",
        "nhi": "WhatsApp Image 2026-09-22 at 22.21.24",
        "synthetic": "ChatGPT Image Sep 24, 2026, 06_13_58 PM",
    }

    for key, folder in sources.items():
        path, doc = find_raw_doc(folder)
        docs[key] = doc
        print(f"[FOUND] {key:9s} -> {path}")

    cases = [
        {
            "name": "Receipt total",
            "document": docs["receipt"],
            "question": "what is the total?",
            "must_include": ["TOTAL", "310"],
            "ambiguous": False,
        },
        {
            "name": "Receipt tax",
            "document": docs["receipt"],
            "question": "what is the tax?",
            "must_include": ["TAX", "15"],
            "ambiguous": False,
        },
        {
            "name": "Form DOB",
            "document": docs["pic13"],
            "question": "what is the date of birth?",
            "must_include": ["Date of Birth", "03/14/1992"],
            "ambiguous": False,
        },
        {
            "name": "NHI payment date",
            "document": docs["nhi"],
            "question": "what is the payment date?",
            "must_include": ["Payment Date", "115/09/21"],
            "ambiguous": False,
        },
        {
            "name": "NHI card pickup",
            "document": docs["nhi"],
            "question": "what is the card pick-up number?",
            "must_include": ["Card Pick-up", "0055"],
            "ambiguous": False,
        },
        {
            "name": "Synthetic total ambiguity",
            "document": docs["synthetic"],
            "question": "what is the total?",
            "must_include": ["TOTAL", "310", "500"],
            "ambiguous": True,
        },
        {
            "name": "Synthetic reference ambiguity",
            "document": docs["synthetic"],
            "question": "what is the reference number?",
            "must_include": ["Reference Number", "0055", "0066"],
            "ambiguous": True,
        },
        {
            "name": "Synthetic customer name",
            "document": docs["synthetic"],
            "question": "what is the customer name?",
            "must_include": ["Name", "Alex Carter"],
            "ambiguous": False,
        },
    ]

    passed_count = 0

    print()
    for i, case in enumerate(cases, start=1):
        passed, detail = run_case(**case)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {i:02d}. {case['name']}")
        print(detail)
        if passed:
            passed_count += 1

    print("=" * 88)
    print(f"RESULT: {passed_count}/{len(cases)} tests passed")
    print("=" * 88)

    return 0 if passed_count == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
