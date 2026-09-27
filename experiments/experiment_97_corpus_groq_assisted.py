"""Offline stub regression for optional Groq-assisted corpus questions."""

import json
from types import SimpleNamespace

from experiments.experiment_96_local_corpus_semantics import document
from services.corpus_groq_assisted import answer_corpus_question_assisted
from services.corpus_question_router import answer_corpus_question
from services.structured_document_corpus import index_structured_documents


class FakeCompletions:
    def __init__(self, focused_question, answer, evidence_ids):
        self.focused_question = focused_question
        self.answer = answer
        self.evidence_ids = evidence_ids
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = (
            {"focused_question": self.focused_question}
            if len(self.calls) % 2 else
            {"answer": self.answer, "evidence_item_ids": self.evidence_ids}
        )
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(content))
        )])


def check(condition, name):
    if not condition:
        raise AssertionError(name)
    print(f"[PASS] {name}")


def main():
    raw = document("trend", "THE TREND LOUNGE", "TL-98432", "Sarah M.",
                   "123 Fashion Blvd, New York, NY 10001")
    index = index_structured_documents([raw])
    question = "Where is THE TREND LOUNGE located?"
    canonical = "What is the location of THE TREND LOUNGE?"
    address = "123 Fashion Blvd, New York, NY 10001"
    fake = FakeCompletions(canonical, address, ["trend:ocr_region:2"])
    client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    result = answer_corpus_question_assisted(index, question, client)
    check(result["status"] == "answered" and result["answer"] == address,
          "Groq wording maps to local address evidence")
    check(result["interpreted_question"] == canonical,
          "accepted interpretation is visible")
    check(result["evidence_item_ids"] == ["trend:ocr_region:2"],
          "final answer cites supplied OCR ID")
    check(len(fake.calls) == 2 and
          "ocr_regions" in fake.calls[0]["messages"][1]["content"],
          "planner sees bounded OCR and grounded answer uses second call")

    fake = FakeCompletions("What is the location of OTHER STORE?", address,
                           ["trend:ocr_region:2"])
    result = answer_corpus_question_assisted(index, question,
        SimpleNamespace(chat=SimpleNamespace(completions=fake)))
    check(result["interpreted_question"] == question and result["answer"] == address,
          "invented merchant is rejected")

    fake = FakeCompletions(canonical, "456 Wrong Road", ["trend:ocr_region:2"])
    result = answer_corpus_question_assisted(index, question,
        SimpleNamespace(chat=SimpleNamespace(completions=fake)))
    check(result["answer"] == address and
          result["provider"] == "local_retrieval_with_groq_interpretation",
          "invented final value falls back to local result")

    raw["evidence"].append({"evidence_type": "ocr_region", "region_id": 99,
                            "text": "456 Market Road, Boston, MA 02101"})
    fake = FakeCompletions(canonical, address, ["trend:ocr_region:2"])
    result = answer_corpus_question_assisted(index, question,
        SimpleNamespace(chat=SimpleNamespace(completions=fake)))
    check(result["status"] == "ambiguous_field" and result["answer"] is None
          and len(fake.calls) == 1,
          "ambiguous addresses never reach final answer call")
    check(answer_corpus_question(index, question)["status"] == "ambiguous_field",
          "offline route preserves ambiguity")
    print("RESULT: 8/8 checks passed")
    print("No OCR, Groq, or network requests were made.")


if __name__ == "__main__":
    main()
