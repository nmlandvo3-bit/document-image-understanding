"""Offline regression for natural cashier and location corpus questions."""

from copy import deepcopy

from services.corpus_question_router import answer_corpus_question
from services.corpus_qa_presentation import present_corpus_answer
from services.structured_document_corpus import index_structured_documents


def document(doc_id, merchant, number, cashier, address):
    fields = {
        "merchant": {"status": "candidate", "observed_value": merchant,
                     "evidence_item_ids": [f"{doc_id}:ocr_region:1"]},
        "receipt_number": {"status": "candidate", "observed_value": number,
                           "evidence_item_ids": [f"{doc_id}:ocr_region:3"]},
        "cashier": {"status": "candidate", "observed_value": cashier,
                    "display_value": cashier,
                    "evidence_item_ids": [f"{doc_id}:ocr_region:4"]},
    }
    return {
        "document_id": doc_id,
        "receipt_semantics": {"status": "receipt_candidate", "fields": fields},
        "evidence": [
            {"evidence_type": "ocr_region", "region_id": 2, "text": address},
        ],
    }


def check(condition, label):
    if not condition:
        raise AssertionError(label)
    print(f"[PASS] {label}")


def main():
    lounge = document("trend", "THE TREND LOUNGE", "TL-98432", "Sarah M.",
                      "123 Fashion Blvd, New York, NY 10001")
    campus = document("campus", "NDHU CAMPUS STORE", "004512", "Jane Smith",
                      "10 University Road, Hualien 974")
    index = index_structured_documents([lounge, campus])
    ask = lambda q: answer_corpus_question(index, q)

    cashier = ask("What is the cashier name for THE TREND LOUNGE?")
    check(cashier["status"] == "answered" and cashier["answer"] == "Sarah M.",
          "cashier name maps to a grounded receipt field")
    check(cashier["document_id"] == "trend" and
          cashier["evidence_item_ids"] == ["trend:ocr_region:4"],
          "cashier answer retains its document and OCR ID")

    location = ask("What is the location of THE TREND LOUNGE?")
    check(location["status"] == "answered" and
          location["answer"] == "123 Fashion Blvd, New York, NY 10001",
          "location maps to exactly printed address")
    check(location["document_id"] == "trend" and
          location["evidence_item_ids"] == ["trend:ocr_region:2"],
          "address cites the printed OCR region")
    check("trend:ocr_region:2" in present_corpus_answer(location)[1],
          "GUI presentation retains address evidence")

    by_number = ask("What is the address on receipt number TL-98432?")
    check(by_number["answer"] == location["answer"],
          "receipt number can select the same address")
    check(ask("Where is THE TREND LOUNGE located?")["answer"] == location["answer"],
          "natural location phrasing resolves exact merchant")

    missing = ask("What is the location of an unknown store?")
    check(missing["status"] == "document_not_found" and missing["answer"] is None,
          "unknown merchant cannot select a location")
    check(ask("Who bought the coffee?")["status"] == "unsupported_question",
          "unsupported questions still fail explicitly")

    different = deepcopy(lounge)
    different["evidence"].append({"evidence_type": "ocr_region", "region_id": 99,
                                  "text": "456 Market Road, Boston, MA 02101"})
    ambiguous = answer_corpus_question(index_structured_documents([different]),
        "What is the location of THE TREND LOUNGE?")
    check(ambiguous["status"] == "ambiguous_field" and ambiguous["answer"] is None
          and len(ambiguous["candidates"]) == 2,
          "multiple addresses remain ambiguous")
    duplicate = deepcopy(lounge)
    duplicate["document_id"] = "trend2"
    conflict = answer_corpus_question(index_structured_documents([lounge, duplicate]),
        "What is the cashier name for THE TREND LOUNGE?")
    check(conflict["status"] == "ambiguous_document" and conflict["answer"] is None,
          "duplicate merchants do not select one document")
    print("RESULT: 11/11 checks passed")
    print("No OCR, Groq, or network requests were made.")


if __name__ == "__main__":
    main()
