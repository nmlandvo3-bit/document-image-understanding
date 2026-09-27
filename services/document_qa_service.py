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

OCR is not rerun. Local structure is reconstructed once when a scan is loaded.
"""

import json
import os
import re
import sys
from pathlib import Path

from groq import Groq

from services.layout_aware_ocr_retrieval import build_raw_ocr_packet
from parsing.unified_structured_document_builder import build_unified_structured_document
from services.structured_document_retrieval import retrieve_structured_answer
from services.answer_type_confidence import (
    infer_answer_type,
    score_candidate,
)

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

QUESTION_PLAN_PROMPT = """
Interpret a question about the selected OCR document for local retrieval.
The supplied OCR and structure are unverified candidates. Do not answer.
Return JSON only: {"focused_question": "a short search question"}.
Keep only words from the original question; do not add values seen in the
document. Preserve every important word from the original question,
including the target and requested attribute (such as price versus quantity).
If the question cannot be narrowed safely, copy it unchanged.
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
        self.structured_documents = {}

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
        # Reconstruct once on load; the OCR evidence itself stays unchanged.
        self.structured_documents = {
            document["document_id"]: build_unified_structured_document(document)
            for document in documents
            if document.get("document_type") == "unclassified"
        }

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

    def _raw_region_packet_item(self, evidence_id):
        for document in self.documents:
            did = document.get("document_id")
            prefix = f"{did}:ocr_region:"
            if not str(evidence_id).startswith(prefix):
                continue

            raw_region_id = str(evidence_id)[len(prefix):]
            try:
                region_id = int(raw_region_id)
            except ValueError:
                region_id = raw_region_id

            for region in document.get("evidence", []):
                if region.get("region_id") != region_id:
                    continue
                return {
                    "evidence_item_id": evidence_id,
                    "document_id": did,
                    "item_type": "ocr_region",
                    "content": {
                        "region_id": region.get("region_id"),
                        "text": region.get("text"),
                        "bbox": region.get("bbox"),
                        "confidence": region.get("confidence"),
                    },
                }
        return None

    def _question_context(self):
        """Bounded OCR and locally reconstructed structure for interpretation."""
        result = []
        for document in self.documents:
            if document.get("document_type") != "unclassified":
                continue
            structured = getattr(self, "structured_documents", {}).get(
                document["document_id"]
            ) or build_unified_structured_document(document)
            result.append({
                "document_id": document["document_id"],
                "fields": [{
                    "label": field.get("label"),
                    "value": field.get("value"),
                    "evidence_item_ids": field.get("evidence_item_ids", []),
                } for field in structured.get("fields", [])[:40]],
                "headings": [{
                    "text": item.get("text"),
                    "evidence_item_ids": item.get("evidence_item_ids", []),
                } for item in structured.get("free_text", [])
                    if item.get("structure_type") == "heading" ][:20],
                "rows": [{
                    "source_row_id": record.get("source_row_id"),
                    "cells": [{
                        "column_index": cell.get("column_index"),
                        "text": cell.get("text"),
                        "evidence_item_ids": cell.get("evidence_item_ids", []),
                    } for cell in record.get("cells", []) if str(cell.get("text", "")).strip()],
                } for record in structured.get("records", [])[:40]],
                "ocr_regions": [{
                    "region_id": region.get("region_id"),
                    "text": region.get("text"),
                    "bbox": region.get("bbox"),
                } for region in document.get("evidence", [])[:80]
                    if region.get("evidence_type") == "ocr_region"],
            })
        return result

    def _focused_question(self, question):
        """Ask Groq to narrow wording; reject new facts or a changed type."""
        context = self._question_context()
        if not context:
            return question
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": QUESTION_PLAN_PROMPT},
                    {"role": "user", "content": json.dumps({
                        "question": question,
                        "document_context": context,
                    }, ensure_ascii=False, separators=(",", ":"))},
                ],
                temperature=0.0,
                reasoning_effort="low",
                include_reasoning=False,
                response_format={"type": "json_object"},
                max_completion_tokens=120,
                stream=False,
            )
            parsed = self._parse_json_response(response.choices[0].message.content or "")
            focus = parsed.get("focused_question")
            if not isinstance(focus, str) or not focus.strip():
                return question
            original = set(re.findall(r"[\w]+", question.casefold()))
            proposed = set(re.findall(r"[\w]+", focus.casefold()))
            if not proposed or not proposed.issubset(original):
                return question
            question_words = {
                "a", "an", "the", "what", "which", "who", "where", "when",
                "how", "is", "are", "was", "were", "does", "did", "do",
                "can", "could", "please", "tell", "me", "give", "show",
                "of", "on", "in", "at", "by", "for", "from", "to", "this",
                "that", "it", "its",
            }
            if not (original - question_words).issubset(proposed):
                return question
            if infer_answer_type(focus) != infer_answer_type(question):
                return question
            return focus.strip()
        except Exception:
            # A failed interpretation request cannot block local retrieval.
            return question

    def _location_packet(self, question):
        """Select explicitly printed street addresses for location questions."""
        wording = question.casefold()
        if not re.search(r"\b(?:address|location|located)\b", wording):
            return None
        # An item location or customer's address is a different request.
        if re.search(r"\b(?:customer|cashier|item|product|shipping|billing)\b", wording):
            return None

        street = re.compile(
            r"\b\d{1,6}\s+[^\n,]{2,80}?\s+"
            r"(?:street|st\.?|avenue|ave\.?|road|rd\.?|"
            r"boulevard|blvd\.?|lane|ln\.?|drive|dr\.?)\b",
            re.IGNORECASE,
        )
        found = []
        for document in self.documents:
            if document.get("document_type") != "unclassified":
                continue
            for region in document.get("evidence", []):
                if region.get("evidence_type") != "ocr_region":
                    continue
                value = " ".join(str(region.get("text", "")).split())
                if not street.search(value):
                    continue
                evidence_id = (
                    f"{document['document_id']}:ocr_region:{region['region_id']}"
                )
                found.append((value, evidence_id))

        if not found:
            return None
        distinct = {value.casefold() for value, _ in found}
        if len(distinct) > 1:
            evidence = [
                item for _, evidence_id in found
                if (item := self._raw_region_packet_item(evidence_id)) is not None
            ]
            return {
                "question": question,
                "answer_mode": "structured_ambiguous_candidates",
                "evidence": evidence,
                "ambiguities": [{
                    "value": value,
                    "label": "printed_address",
                    "source": "ocr_address_region",
                    "evidence_item_ids": [evidence_id],
                    "reason": "multiple_printed_addresses",
                } for value, evidence_id in found],
                "structured_candidate": None,
                "evidence_level": "automatic_structured_candidates",
            }

        value = found[0][0]
        ids = list(dict.fromkeys(evidence_id for _, evidence_id in found))
        evidence = [
            item for evidence_id in ids
            if (item := self._raw_region_packet_item(evidence_id)) is not None
        ]
        if not evidence:
            return None
        return {
            "question": question,
            "answer_mode": "structured_answer_candidate",
            "evidence": evidence,
            "ambiguities": [],
            "structured_candidate": {
                "label": "printed_address",
                "value": value,
                "source": "ocr_address_region",
                "relationship": "single_ocr_region",
                "evidence_item_ids": ids,
                "value_evidence_item_ids": ids,
            },
            "evidence_level": "automatic_structured_candidates",
        }

    def _structured_packet(self, question, retrieval_question=None):
        structured_results = []

        for document in self.documents:
            if document.get("document_type") != "unclassified":
                continue

            structured_document = getattr(self, "structured_documents", {}).get(
                document["document_id"]
            ) or build_unified_structured_document(document)
            retrieval = retrieve_structured_answer(
                structured_document, retrieval_question or question
            )
            structured_results.append(retrieval)

        if not structured_results:
            return None

        ambiguous = [
            result for result in structured_results
            if result.get("status") == "ambiguous_structured_evidence"
        ]

        if ambiguous:
            ambiguity_items = []
            evidence_ids = []
            for result in ambiguous:
                for candidate in result.get("ambiguities", []):
                    ambiguity_items.append({
                        "value": candidate.get("value"),
                        "label": candidate.get("label"),
                        "source": candidate.get("source"),
                        "evidence_item_ids": list(candidate.get("evidence_item_ids", [])),
                        "reason": "multiple_plausible_structured_values",
                    })
                    evidence_ids.extend(candidate.get("evidence_item_ids", []))

            packet_evidence = []
            seen = set()
            for evidence_id in evidence_ids:
                if evidence_id in seen:
                    continue
                seen.add(evidence_id)
                item = self._raw_region_packet_item(evidence_id)
                if item is not None:
                    packet_evidence.append(item)

            return {
                "question": question,
                "answer_mode": "structured_ambiguous_candidates",
                "evidence": packet_evidence,
                "ambiguities": ambiguity_items,
                "structured_candidate": None,
                "evidence_level": "automatic_structured_candidates",
            }

        selected = []
        for result in structured_results:
            if result.get("status") != "structured_answer_candidate":
                continue
            candidate = result.get("selected_candidate")
            if candidate:
                selected.append(candidate)

        if not selected:
            return None

        distinct_values = {
            str(candidate.get("value", "")).strip().casefold()
            for candidate in selected
            if str(candidate.get("value", "")).strip()
        }

        if len(distinct_values) > 1:
            ambiguity_items = []
            evidence_ids = []
            for candidate in selected:
                ambiguity_items.append({
                    "value": candidate.get("value"),
                    "label": candidate.get("label"),
                    "source": candidate.get("source"),
                    "evidence_item_ids": list(candidate.get("evidence_item_ids", [])),
                    "reason": "conflicting_structured_documents",
                })
                evidence_ids.extend(candidate.get("evidence_item_ids", []))

            packet_evidence = []
            seen = set()
            for evidence_id in evidence_ids:
                if evidence_id in seen:
                    continue
                seen.add(evidence_id)
                item = self._raw_region_packet_item(evidence_id)
                if item is not None:
                    packet_evidence.append(item)

            return {
                "question": question,
                "answer_mode": "structured_ambiguous_candidates",
                "evidence": packet_evidence,
                "ambiguities": ambiguity_items,
                "structured_candidate": None,
                "evidence_level": "automatic_structured_candidates",
            }

        candidate = selected[0]
        candidate_ids = list(candidate.get("evidence_item_ids", []))
        packet_evidence = []
        for evidence_id in candidate_ids:
            item = self._raw_region_packet_item(evidence_id)
            if item is not None:
                packet_evidence.append(item)

        if not packet_evidence:
            return None

        value = str(candidate.get("value", "")).strip()
        value_ids = []
        for item in packet_evidence:
            text = str((item.get("content") or {}).get("text", "")).strip()
            if text.casefold() == value.casefold():
                value_ids.append(item["evidence_item_id"])
        if not value_ids and candidate_ids:
            value_ids = [candidate_ids[-1]]

        return {
            "question": question,
            "answer_mode": "structured_answer_candidate",
            "evidence": packet_evidence,
            "ambiguities": [],
            "structured_candidate": {
                "label": candidate.get("label"),
                "value": candidate.get("value"),
                "source": candidate.get("source"),
                "relationship": candidate.get("relationship"),
                "evidence_item_ids": candidate_ids,
                "value_evidence_item_ids": value_ids,
            },
            "evidence_level": "automatic_structured_candidates",
        }

    def build_llm_packet(self, question, retrieval_question=None):
        if all(d.get("document_type") == "unclassified" for d in self.documents):
            location_packet = self._location_packet(question)
            if location_packet is not None:
                return location_packet
            structured_packet = self._structured_packet(question, retrieval_question)
            if structured_packet is not None:
                return structured_packet
            return build_raw_ocr_packet(question, self.documents)

        packet = build_compressed_packet(question, self.corpus, top_k=15)
        compact_evidence = []
        for item in packet.get("evidence", []):
            compact_evidence.append({
                "evidence_item_id": item["item_id"],
                "document_id": item["document_id"],
                "item_type": item["item_type"],
                "content": self._compact_content(item),
            })

        return {
            "question": question,
            "answer_mode": packet.get("answer_mode"),
            "evidence": compact_evidence,
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

    @staticmethod
    def _empty_confidence(question, reason, score=0.0):
        """Typed metadata for unanswered, rejected, or locally blocked results."""
        return {
            "answer_type": infer_answer_type(question),
            "confidence_score": round(float(score), 3),
            "confidence_level": (
                "high" if score >= 0.90
                else "medium" if score >= 0.70
                else "low"
            ),
            "confidence_reason": reason,
            "confidence_is_calibrated_probability": False,
        }

    @staticmethod
    def _raw_ocr_confidence_metadata(packet, model_result):
        """Score the spatial value actually cited and supported by the answer."""
        question = str(packet.get("question", ""))
        expected_type = infer_answer_type(question)

        if packet.get("answer_mode") != "raw_ocr_spatial_candidates":
            return {
                "answer_type": expected_type,
                "confidence_score": None,
                "confidence_level": "not_applicable",
                "confidence_reason": (
                    "confidence_scoring_currently_applies_to_raw_ocr_spatial_answers"
                ),
                "confidence_is_calibrated_probability": False,
            }

        cited = set(model_result.get("evidence_item_ids", []))
        answer = str(model_result.get("answer", "")).strip().casefold()
        indexed = {
            item["evidence_item_id"]: item
            for item in packet.get("evidence", [])
        }

        relations = packet.get("spatial_candidates", [])

        grouped = {}
        for relation in relations:
            grouped.setdefault(
                (
                    relation.get("document_id"),
                    relation.get("label_region_id"),
                ),
                [],
            ).append(relation)

        for relation in relations:
            doc_id = relation.get("document_id")
            candidate_region_id = relation.get("candidate_region_id")

            if doc_id is None or candidate_region_id is None:
                continue

            value_id = f"{doc_id}:ocr_region:{candidate_region_id}"

            if value_id not in cited:
                continue

            item = indexed.get(value_id)
            if item is None:
                continue

            content = item.get("content", {})
            raw = str(content.get("text", "")).strip()

            if not raw:
                continue

            raw_folded = raw.casefold()
            if raw_folded not in answer and answer not in raw_folded:
                if re.fullmatch(r"\d+(?:\.\d+)?", raw):
                    if not re.search(
                        r"(?<![\d.])" + re.escape(raw) + r"(?![\d.])",
                        answer,
                    ):
                        continue
                else:
                    continue

            group_key = (
                relation.get("document_id"),
                relation.get("label_region_id"),
            )
            candidate_count = max(1, len(grouped.get(group_key, [])))

            result = score_candidate(
                question=question,
                candidate=raw,
                relation=relation.get("spatial_direction") or "unknown",
                ocr_confidence=content.get("confidence"),
                ambiguous=False,
                candidate_count=candidate_count,
            )

            return {
                "answer_type": result.expected_type,
                "confidence_score": result.score,
                "confidence_level": result.level,
                "confidence_reason": result.reason,
                "confidence_is_calibrated_probability": False,
            }

        # Grounded raw OCR with no scoreable cited spatial value is intentionally
        # not assigned a made-up numeric confidence.
        return {
            "answer_type": expected_type,
            "confidence_score": None,
            "confidence_level": "not_available",
            "confidence_reason": "no_scoreable_cited_spatial_value",
            "confidence_is_calibrated_probability": False,
        }

    @staticmethod
    def _structured_answer_check(packet, model_result):
        candidate = packet.get("structured_candidate") or {}
        expected = str(candidate.get("value", "")).strip()
        answer = str(model_result.get("answer", "")).strip().casefold()
        cited = set(model_result.get("evidence_item_ids", []))
        value_ids = set(candidate.get("value_evidence_item_ids", []))

        if not expected:
            return False, "missing_structured_candidate_value"
        if not answer or not cited:
            return False, "no_cited_answer"
        if value_ids and not (cited & value_ids):
            return False, "structured_value_evidence_not_cited"
        if expected.casefold() in answer:
            return True, "literal_structured_candidate_value"
        return False, "answer_not_supported_by_structured_value"

    def _answer_check(self, packet, model_result):
        if packet.get("answer_mode") == "structured_answer_candidate":
            return self._structured_answer_check(packet, model_result)
        return self._raw_ocr_answer_check(packet, model_result)

    def _structured_confidence_metadata(self, packet):
        candidate = packet.get("structured_candidate") or {}
        value = str(candidate.get("value", "")).strip()
        source = candidate.get("source")
        relationship = candidate.get("relationship")

        if source == "structured_table_same_row":
            relation = "same_row_right"
        elif source == "structured_table_same_column":
            relation = "table_column_below"
        elif relationship == "same_row_right":
            relation = "same_row_right"
        elif relationship in {"next_row_below", "below"}:
            relation = "below"
        else:
            relation = "unknown"

        value_ids = set(candidate.get("value_evidence_item_ids", []))
        ocr_confidence = None
        for item in packet.get("evidence", []):
            if item.get("evidence_item_id") not in value_ids:
                continue
            raw = (item.get("content") or {}).get("confidence")
            if raw is not None:
                try:
                    ocr_confidence = float(raw)
                except (TypeError, ValueError):
                    ocr_confidence = None
                break

        result = score_candidate(
            question=packet.get("question", ""),
            candidate=value,
            relation=relation,
            ocr_confidence=ocr_confidence,
            ambiguous=False,
            candidate_count=1,
        )

        return {
            "answer_type": getattr(result, "expected_type", None),
            "confidence_score": float(getattr(result, "score", 0.0)),
            "confidence_level": getattr(result, "level", "low"),
            "confidence_reason": getattr(result, "reason", "structured_candidate"),
            "confidence_is_calibrated_probability": False,
        }

    def _confidence_metadata(self, packet, llm_result):
        if packet.get("answer_mode") == "structured_answer_candidate":
            return self._structured_confidence_metadata(packet)
        return self._raw_ocr_confidence_metadata(packet, llm_result)

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

        retrieval_question = self._focused_question(question)
        packet = self.build_llm_packet(question, retrieval_question)

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
                **self._empty_confidence(
                    question,
                    "ambiguous_document_evidence",
                    score=0.49,
                ),
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
                **self._empty_confidence(
                    question,
                    "insufficient_document_evidence",
                    score=0.0,
                ),
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
                **self._empty_confidence(
                    question,
                    "invalid_evidence_ids",
                    score=0.0,
                ),
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
        answer_check_passed, answer_check_reason = self._answer_check(
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

        if grounded:
            confidence_metadata = self._confidence_metadata(
                packet,
                llm_result,
            )
        else:
            confidence_metadata = self._empty_confidence(
                question,
                uncertainty or "answer_not_grounded",
                score=0.0,
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
            **confidence_metadata,
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
