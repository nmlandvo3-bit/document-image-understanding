"""Offline regression: cashier and location questions on Trend Lounge OCR."""

import json
from copy import deepcopy
from pathlib import Path

from services.document_qa_service import DocumentQAService


FIXTURE = Path("results/gui_v4/single/Trend Lounge/unified_raw_ocr.json")


def check(condition, label):
    if not condition:
        raise AssertionError(label)
    print(f"[PASS] {label}")


def main():
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    before = deepcopy(raw["evidence"])
    service = DocumentQAService.__new__(DocumentQAService)
    service.documents = [raw]
    service.structured_documents = {}

    cashier = service.build_llm_packet("what is the cashier name?")
    check(cashier["answer_mode"] == "structured_answer_candidate",
          "cashier is selected from a structured field")
    check(cashier["structured_candidate"]["value"] == "Sarah M.",
          "cashier answer preserves OCR text")

    question = "what is the location of the lounge"
    location = service.build_llm_packet(question)
    check(location["answer_mode"] == "structured_answer_candidate",
          "location question selects printed address")
    check(location["question"] == question,
          "original location question retained")
    check(location["structured_candidate"]["value"] ==
          "123 Fashion Blvd, New York, NY 10001",
          "selected address is exactly the printed OCR region")
    check(len(location["evidence"]) == 1 and
          location["evidence"][0]["evidence_item_id"].endswith(":ocr_region:2"),
          "address cites the actual OCR region")

    check(service._location_packet("cashier?") is None,
          "cashier question cannot select an address")
    check(service._location_packet("what is the customer's address?") is None,
          "customer address is not mistaken for store address")

    duplicate = deepcopy(raw["evidence"][1])
    duplicate["region_id"] = 999
    duplicate["text"] = "456 Market Road, Boston, MA 02101"
    raw["evidence"].append(duplicate)
    ambiguous = service._location_packet(question)
    check(ambiguous["answer_mode"] == "structured_ambiguous_candidates" and
          len(ambiguous["ambiguities"]) == 2,
          "two printed addresses cannot be silently selected")
    raw["evidence"].pop()
    check(raw["evidence"] == before, "raw OCR evidence unchanged")
    print("RESULT: 10/10 checks passed")
    print("No OCR, Groq, or network requests were made.")


if __name__ == "__main__":
    main()
