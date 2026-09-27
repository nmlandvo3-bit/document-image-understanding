"""Offline regression for Groq question interpretation and local retrieval.

Run from the project root: python -m experiments.experiment_94_groq_question_planning
"""

import json
from pathlib import Path
from types import SimpleNamespace

from services.document_qa_service import DocumentQAService


FIXTURE = Path("results/gui_v4/single/Trend Lounge/unified_raw_ocr.json")


class FakeCompletions:
    def __init__(self):
        self.focus = "price Classic White Tee"
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps({"focused_question": self.focus}))
        )])


def check(condition, label):
    if not condition:
        raise AssertionError(label)
    print(f"[PASS] {label}")


def main():
    original = json.loads(FIXTURE.read_text(encoding="utf-8"))
    before = json.dumps(original.get("evidence"), sort_keys=True)
    service = DocumentQAService.__new__(DocumentQAService)
    service.model = "offline-fake"
    service.documents = [original]
    service.structured_documents = {}
    fake = FakeCompletions()
    service.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    question = "What is the price for Classic White Tee?"

    focus = service._focused_question(question)
    context = json.loads(fake.calls[0]["messages"][1]["content"])["document_context"][0]
    check(bool(context["fields"]), "planner sees reconstructed fields")
    check(bool(context["headings"]), "planner sees reconstructed headings")
    check(bool(context["rows"]), "planner sees reconstructed table rows")
    check(bool(context["ocr_regions"]), "planner sees OCR text and geometry")
    check(focus == fake.focus, "valid focus accepted")
    packet = service.build_llm_packet(question, focus)
    check(packet["question"] == question, "original question retained for answer")
    check(packet.get("structured_candidate", {}).get("value") == "$40.00",
          "local retrieval selects the tee price")
    check(len(packet["structured_candidate"]["evidence_item_ids"]) >= 2,
          "selected price preserves OCR provenance")

    fake.focus = "price"
    check(service._focused_question(question) == question,
          "planner cannot drop the named item")
    fake.focus = "price Classic White Tee $9.99"
    check(service._focused_question(question) == question,
          "planner cannot introduce a document value")
    fake.focus = "quantity Classic White Tee"
    check(service._focused_question(question) == question,
          "planner cannot change the requested attribute")
    fake.focus = "not json"  # A malformed plan is ignored.
    check(service._focused_question(question) == question,
          "unusable planner output falls back safely")
    check(json.dumps(original.get("evidence"), sort_keys=True) == before,
          "raw OCR evidence unchanged")
    print("RESULT: 13/13 checks passed")
    print("No OCR, Groq, or network requests were made.")


if __name__ == "__main__":
    main()
