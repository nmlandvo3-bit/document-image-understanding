"""
Document QA Service

Reusable application service built from the validated
Experiments 40-45.

Pipeline
--------
Unified documents
    -> generic retrieval
    -> evidence compression
    -> grounded Groq request
    -> provenance validation
    -> application response

This module does NOT rerun OCR or document parsing.
"""

import json
import os
import re
import sys
from pathlib import Path

from groq import Groq

from services.layout_aware_ocr_retrieval import build_raw_ocr_packet

from experiments.experiment_41_generic_cross_document_retrieval import (
    PIC1_PATH,
    PIC13_PATH,
    build_search_corpus,
    load_json,
)

from experiments.experiment_44_grounding_evidence_compression import (
    build_compressed_packet,
)


if hasattr(
    sys.stdout,
    "reconfigure",
):
    try:
        sys.stdout.reconfigure(
            encoding="utf-8"
        )
    except Exception:
        pass


DEFAULT_MODEL = (
    "openai/gpt-oss-20b"
)


SYSTEM_PROMPT = """
You answer questions about documents.

Rules:
- Use only the supplied document evidence.
- Do not use outside knowledge.
- Do not invent missing facts.
- Do not silently correct OCR.
- If evidence is insufficient, say so; set evidence_item_ids to []
  and uncertainty to "insufficient_document_evidence".
- For a factual answer, cite only the evidence IDs that actually support it.
- Do not cite unrelated evidence merely because it was supplied.
- Use only supplied evidence_item_ids.
- For raw OCR: spatial neighbors are candidate associations, not verified fields.
- Cite BOTH the label and the candidate value if you use a spatial association.
- When multiple candidates are plausible or the relationship is ambiguous, say so.
- Keep the answer concise.

Return JSON only:

{
  "answer": "natural-language answer",
  "evidence_item_ids": ["ids actually used"],
  "uncertainty": null
}
""".strip()


class DocumentQAService:
    def __init__(
        self,
        model=DEFAULT_MODEL,
        api_key=None,
        document_paths=None,
    ):
        self.model = model

        self.api_key = (
            api_key
            or os.getenv(
                "GROQ_API_KEY"
            )
        )

        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set."
            )

        self.client = Groq(
            api_key=self.api_key
        )

        self.documents = []
        self.corpus = []

        self._load_documents(document_paths)

    def _load_documents(self, document_paths=None):
        """Index only the explicitly selected unified documents.

        With no paths provided, load the two validated demo documents.
        This never converts raw OCR results into a table or form.
        """
        paths = (
            [PIC1_PATH, PIC13_PATH]
            if document_paths is None
            else list(document_paths)
        )
        if not paths:
            raise ValueError("At least one unified document path is required.")

        documents = []
        seen_ids = set()
        required = {
            "document_id", "schema_version", "document_type",
            "records", "fields", "controls", "free_text", "evidence",
        }
        for raw_path in paths:
            path = Path(raw_path)
            if not path.is_file():
                raise FileNotFoundError(path)
            document = load_json(path)
            if not isinstance(document, dict):
                raise ValueError(f"Not a unified document object: {path}")
            missing = required - document.keys()
            if missing:
                raise ValueError(
                    f"Not a unified document ({path}); missing: {sorted(missing)}"
                )
            document_id = document["document_id"]
            if not isinstance(document_id, str) or not document_id.strip():
                raise ValueError(f"Invalid document_id: {path}")
            if document_id in seen_ids:
                raise ValueError(f"Duplicate document_id: {document_id}")
            seen_ids.add(document_id)
            for key in ("records", "fields", "controls", "free_text", "evidence"):
                if not isinstance(document[key], list):
                    raise ValueError(f"Expected list for {key}: {path}")
            if not any(document[key] for key in ("records", "fields", "controls", "free_text")):
                raise ValueError(
                    f"No searchable semantic items in {path}. "
                    "Raw OCR JSON needs a separate structure parser."
                )
            documents.append(document)

        corpus = build_search_corpus(documents)
        if not corpus:
            raise ValueError("Selected documents have no searchable items.")
        self.documents = documents
        self.corpus = corpus

    def get_document_summary(
        self,
    ):
        """
        Lightweight information about indexed documents.
        """

        summary = []

        for document in self.documents:
            summary.append(
                {
                    "document_id": (
                        document.get(
                            "document_id"
                        )
                    ),
                    "document_type": (
                        document.get(
                            "document_type"
                        )
                    ),
                    "records": len(
                        document.get(
                            "records",
                            [],
                        )
                    ),
                    "fields": len(
                        document.get(
                            "fields",
                            [],
                        )
                    ),
                    "controls": len(
                        document.get(
                            "controls",
                            [],
                        )
                    ),
                    "free_text": len(
                        document.get(
                            "free_text",
                            [],
                        )
                    ),
                }
            )

        return summary

    @staticmethod
    def _compact_content(
        evidence_item,
    ):
        item_type = (
            evidence_item[
                "item_type"
            ]
        )

        content = (
            evidence_item[
                "content"
            ]
        )

        if item_type == "record":
            return {
                "record_id": (
                    content.get(
                        "record_id"
                    )
                ),
                "category": (
                    content.get(
                        "category"
                    )
                ),
                "event": (
                    content.get(
                        "event"
                    )
                ),
                "time": (
                    content.get(
                        "time"
                    )
                ),
                "detail": (
                    content.get(
                        "detail"
                    )
                ),
                "explanation": (
                    content.get(
                        "explanation"
                    )
                ),
                "review_flags": (
                    content.get(
                        "review_flags",
                        [],
                    )
                ),
            }

        if item_type == "field":
            return {
                "label": (
                    content.get(
                        "label"
                    )
                ),
                "value": (
                    content.get(
                        "value"
                    )
                ),
            }

        if item_type == "control":
            return {
                "label": (
                    content.get(
                        "label"
                    )
                ),
                "checked": bool(
                    content.get(
                        "checked",
                        False,
                    )
                ),
                "state": (
                    content.get(
                        "state"
                    )
                ),
            }

        if item_type == "free_text":
            return {
                "source_control_label": (
                    content.get(
                        "source_control_label"
                    )
                ),
                "text": (
                    content.get(
                        "text"
                    )
                ),
            }

        return content

    def build_llm_packet(
        self,
        question,
    ):
        """
        Build compressed evidence for the LLM.
        """

        # Raw OCR has separate label and value observations. The semantic
        # compressor only indexes isolated text, so use conservative spatial
        # expansion for unclassified OCR-only documents.
        if all(d.get("document_type") == "unclassified" for d in self.documents):
            return build_raw_ocr_packet(question, self.documents)

        packet = build_compressed_packet(
            question,
            self.corpus,
            top_k=15,
        )

        compact_evidence = []

        for item in packet.get(
            "evidence",
            [],
        ):
            compact_evidence.append(
                {
                    "evidence_item_id": (
                        item[
                            "item_id"
                        ]
                    ),
                    "document_id": (
                        item[
                            "document_id"
                        ]
                    ),
                    "item_type": (
                        item[
                            "item_type"
                        ]
                    ),
                    "content": (
                        self._compact_content(
                            item
                        )
                    ),
                }
            )

        return {
            "question": question,
            "answer_mode": (
                packet.get(
                    "answer_mode"
                )
            ),
            "evidence": (
                compact_evidence
            ),
        }

    @staticmethod
    def _parse_json_response(
        content,
    ):
        content = (
            content.strip()
        )

        if content.startswith(
            "```"
        ):
            lines = (
                content.splitlines()
            )

            if lines:
                lines = lines[1:]

            if (
                lines
                and lines[-1]
                .strip()
                .startswith(
                    "```"
                )
            ):
                lines = (
                    lines[:-1]
                )

            content = "\n".join(
                lines
            ).strip()

        return json.loads(
            content
        )

    def _call_groq(
        self,
        packet,
    ):
        response = (
            self.client
            .chat
            .completions
            .create(
                model=self.model,
                messages=[
                    {
                        "role": (
                            "system"
                        ),
                        "content": (
                            SYSTEM_PROMPT
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            json.dumps(
                                packet,
                                ensure_ascii=False,
                                separators=(
                                    ",",
                                    ":",
                                ),
                            )
                        ),
                    },
                ],
                temperature=0.0,
                reasoning_effort="low",
                include_reasoning=False,
                response_format={
                    "type": (
                        "json_object"
                    ),
                },
                max_completion_tokens=400,
                stream=False,
            )
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if not content:
            raise RuntimeError(
                "Groq returned an empty response."
            )

        parsed = (
            self._parse_json_response(
                content
            )
        )

        required = {
            "answer",
            "evidence_item_ids",
            "uncertainty",
        }

        missing = (
            required
            - set(
                parsed.keys()
            )
        )

        if missing:
            raise RuntimeError(
                "LLM response missing: "
                + ", ".join(
                    sorted(
                        missing
                    )
                )
            )

        usage = {}

        if getattr(
            response,
            "usage",
            None,
        ):
            usage = {
                "prompt_tokens": getattr(
                    response.usage,
                    "prompt_tokens",
                    None,
                ),
                "completion_tokens": getattr(
                    response.usage,
                    "completion_tokens",
                    None,
                ),
                "total_tokens": getattr(
                    response.usage,
                    "total_tokens",
                    None,
                ),
            }

        return {
            "answer": str(
                parsed[
                    "answer"
                ]
            ),
            "evidence_item_ids": list(
                parsed[
                    "evidence_item_ids"
                ]
            ),
            "uncertainty": (
                parsed[
                    "uncertainty"
                ]
            ),
            "usage": usage,
        }

    @staticmethod
    def _validate_evidence(
        packet,
        llm_result,
    ):
        supplied_ids = {
            item[
                "evidence_item_id"
            ]
            for item in packet[
                "evidence"
            ]
        }

        used_ids = set(
            llm_result[
                "evidence_item_ids"
            ]
        )

        invalid_ids = (
            used_ids
            - supplied_ids
        )

        return {
            "valid": (
                len(
                    invalid_ids
                )
                == 0
            ),
            "supplied_ids": (
                sorted(
                    supplied_ids
                )
            ),
            "used_ids": (
                sorted(
                    used_ids
                )
            ),
            "invalid_ids": (
                sorted(
                    invalid_ids
                )
            ),
        }

    @staticmethod
    def _additional_spatial_context(packet, cited_evidence, grounded):
        """Show related packet evidence without rewriting model citations.

        Relations are heuristic spatial candidates, never verified fields.
        Return an empty list for refusals or non-spatial evidence packets.
        """
        if not grounded:
            return []

        evidence_by_id = {
            item["evidence_item_id"]: item
            for item in packet.get("evidence", [])
        }
        cited_ids = {
            item["evidence_item_id"] for item in cited_evidence
        }
        context = []
        seen = set()
        for relation in packet.get("spatial_candidates", []):
            if relation.get("relationship") != "spatial_candidate_not_verified":
                continue
            doc_id = relation.get("document_id")
            label_region = relation.get("label_region_id")
            value_region = relation.get("candidate_region_id")
            if doc_id is None or label_region is None or value_region is None:
                continue
            label_id = f"{doc_id}:ocr_region:{label_region}"
            value_id = f"{doc_id}:ocr_region:{value_region}"
            # Display the other half ONLY when the model cited one half.
            # All displayed context must already be in the actual LLM packet.
            for cited_id, context_id in ((value_id, label_id), (label_id, value_id)):
                if cited_id not in cited_ids or context_id in cited_ids:
                    continue
                if context_id not in evidence_by_id or context_id in seen:
                    continue
                seen.add(context_id)
                context.append({
                    "evidence": evidence_by_id[context_id],
                    "related_model_citation_id": cited_id,
                    "relationship": "spatial_candidate_not_verified",
                    "spatial_direction": relation.get("spatial_direction"),
                })
        return context

    @staticmethod
    def _raw_ocr_answer_check(packet, model_result):
        """Conservative literal-value check, not a semantic correctness proof.

        For spatial pairs, the model answer must contain a value from the
        supplied packet, with that value region among the model citations.
        Header-only citations and values copied from other questions fail.
        """
        if packet.get("answer_mode") != "raw_ocr_spatial_candidates":
            return True, "not_applicable"
        cited = set(model_result.get("evidence_item_ids", []))
        answer = str(model_result.get("answer", "")).strip().casefold()
        if not cited or not answer:
            return False, "no_cited_answer"
        indexed = {i["evidence_item_id"]: i for i in packet.get("evidence", [])}
        question = str(packet.get("question", "")).casefold()
        relations = packet.get("spatial_candidates", [])
        # The spatial relation defines the pool of possible answer values.
        if relations:
            for relation in relations:
                value_id = (f"{relation['document_id']}:ocr_region:"
                            f"{relation['candidate_region_id']}")
                if value_id not in cited or value_id not in indexed:
                    continue
                raw = str(indexed[value_id].get("content", {}).get("text", "")).strip()
                if not raw:
                    continue
                # A date question must not accept the nearby clock time, or
                # vice versa, when both occupy the same table column.
                if "date" in question and "time" not in question:
                    if not re.fullmatch(r"\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}", raw):
                        continue
                if "time" in question and "date" not in question:
                    if not re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", raw):
                        continue
                # Avoid accepting a number as a substring of a longer number.
                if re.fullmatch(r"\d+(?:\.\d+)?", raw):
                    if re.search(r"(?<![\d.])" + re.escape(raw) + r"(?![\d.])", answer):
                        return True, "literal_candidate_value"
                elif raw.casefold() in answer:
                    return True, "literal_candidate_value"
            return False, "answer_not_supported_by_cited_spatial_value"
        # Standalone OCR regions (e.g. the shop name) have no pair.
        for evidence_id in cited:
            item = indexed.get(evidence_id)
            if item is None:
                continue
            text = str(item.get("content", {}).get("text", "")).strip().casefold()
            if text and (answer in text or text in answer):
                return True, "literal_single_region"
        return False, "answer_not_found_in_cited_ocr"

    def answer(
        self,
        question,
    ):
        """
        Main application-facing method.
        """

        if not isinstance(
            question,
            str,
        ):
            raise TypeError(
                "question must be a string."
            )

        question = (
            question.strip()
        )

        if not question:
            raise ValueError(
                "question cannot be empty."
            )

        packet = (
            self.build_llm_packet(
                question
            )
        )

        # Experiment 59: do not ask the model to choose among multiple
        # equally plausible spatial values for the same label. The ambiguity
        # is detected locally from the current question packet.
        ambiguities = packet.get("ambiguities", [])
        if ambiguities:
            return {
                "question": question,
                "answer": (
                    "Ambiguous document evidence: multiple plausible values "
                    "were found for this question."
                ),
                "grounded": False,
                "answer_consistency": {
                    "passed": False,
                    "reason": "ambiguous_spatial_candidates",
                },
                "provider": "local_retrieval",
                "model": self.model,
                "answer_mode": packet.get("answer_mode"),
                "evidence": [],
                "additional_spatial_context": [],
                "evidence_validation": {
                    "valid": True,
                    "supplied_ids": [
                        item["evidence_item_id"]
                        for item in packet.get("evidence", [])
                    ],
                    "used_ids": [],
                    "invalid_ids": [],
                },
                "uncertainty": "ambiguous_document_evidence",
                "ambiguities": ambiguities,
                "usage": {},
            }

        if not packet[
            "evidence"
        ]:
            return {
                "question": question,
                "answer": (
                    "I could not find sufficient "
                    "document evidence to answer "
                    "that question."
                ),
                "grounded": False,
                "answer_consistency": {"passed": None, "reason": "no_evidence"},
                "provider": "groq",
                "model": self.model,
                "answer_mode": (
                    packet.get(
                        "answer_mode"
                    )
                ),
                "evidence": [],
                "additional_spatial_context": [],
                "evidence_validation": {
                    "valid": True,
                    "supplied_ids": [],
                    "used_ids": [],
                    "invalid_ids": [],
                },
                "uncertainty": (
                    "insufficient_document_evidence"
                ),
                "usage": {},
            }

        llm_result = (
            self._call_groq(
                packet
            )
        )

        validation = (
            self._validate_evidence(
                packet,
                llm_result,
            )
        )

        if not validation["valid"]:
            # A previous question's IDs (or fabricated IDs) must never appear
            # as a successful answer for the current evidence packet.
            return {
                "question": question,
                "answer": "The response cited evidence outside this question's document packet. Please retry or review the OCR regions.",
                "grounded": False,
                "answer_consistency": {"passed": False, "reason": "invalid_evidence_ids"},
                "provider": "groq",
                "model": self.model,
                "answer_mode": packet.get("answer_mode"),
                "evidence": [],
                "additional_spatial_context": [],
                "evidence_validation": validation,
                "uncertainty": "answer_evidence_mismatch",
                "usage": llm_result["usage"],
            }

        evidence_by_id = {
            item[
                "evidence_item_id"
            ]: item
            for item in packet[
                "evidence"
            ]
        }

        used_evidence = []

        for evidence_id in (
            llm_result[
                "evidence_item_ids"
            ]
        ):
            item = (
                evidence_by_id.get(
                    evidence_id
                )
            )

            if item is not None:
                used_evidence.append(
                    item
                )

        # Evidence-ID validation checks provenance, not factual support.
        # A model refusal with no cited evidence must never be shown as a
        # grounded factual answer, even if candidate evidence was retrieved.
        uncertainty = llm_result["uncertainty"]
        if not used_evidence and not uncertainty:
            uncertainty = "insufficient_document_evidence"

        grounded = bool(used_evidence) and not bool(uncertainty)
        answer_check_passed, answer_check_reason = self._raw_ocr_answer_check(
            packet, llm_result
        ) if grounded else (None, "not_checked_unanswered")
        answer_text = llm_result["answer"]
        if grounded and not answer_check_passed:
            # Do not show the unsupported model output as a factual answer.
            grounded = False
            uncertainty = "answer_evidence_mismatch"
            answer_text = (
                "The response could not be verified against the evidence "
                "for this question. Please review the OCR regions."
            )
            used_evidence = []
        additional_spatial_context = self._additional_spatial_context(
            packet, used_evidence, grounded
        )

        return {
            "question": question,
            "answer": answer_text,
            "answer_consistency": {
                "passed": answer_check_passed,
                "reason": answer_check_reason,
            },
            "grounded": grounded,
            "provider": "groq",
            "model": self.model,
            "answer_mode": (
                packet.get(
                    "answer_mode"
                )
            ),
            "evidence": (
                used_evidence
            ),
            "additional_spatial_context": additional_spatial_context,
            "evidence_validation": (
                validation
            ),
            "uncertainty": uncertainty,
            "usage": (
                llm_result[
                    "usage"
                ]
            ),
        }


def interactive_cli():
    print(
        "=" * 72
    )

    print(
        "DOCUMENT QA SERVICE"
    )

    print(
        "=" * 72
    )

    service = (
        DocumentQAService()
    )

    print(
        "Indexed documents:"
    )

    for document in (
        service
        .get_document_summary()
    ):
        print(
            (
                f"  {document['document_id']} "
                f"({document['document_type']})"
            )
        )

    print()

    print(
        "Type a question."
    )

    print(
        "Type 'exit' to quit."
    )

    print()

    while True:
        try:
            question = input(
                "Question> "
            ).strip()

        except (
            EOFError,
            KeyboardInterrupt,
        ):
            print()
            break

        if question.lower() in {
            "exit",
            "quit",
            "q",
        }:
            break

        if not question:
            continue

        try:
            result = (
                service.answer(
                    question
                )
            )

            print()
            print(
                "Answer:"
            )

            print(
                result[
                    "answer"
                ]
            )

            print()

            print(
                "Grounded:",
                result[
                    "grounded"
                ],
            )

            print(
                "Evidence:"
            )

            if not result[
                "evidence"
            ]:
                print(
                    "  None"
                )

            for item in result[
                "evidence"
            ]:
                print(
                    " ",
                    item[
                        "evidence_item_id"
                    ],
                )

            print()

        except Exception as exc:
            print()
            print(
                (
                    f"ERROR: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )
            )

            print()


if __name__ == "__main__":
    interactive_cli()