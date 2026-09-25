"""Patch current DocumentQAService for structured-first retrieval.

Run from project root:
    python -m experiments.patch_document_qa_service_structured
"""
from __future__ import annotations

from pathlib import Path
import inspect
import shutil

from services.document_qa_service import DocumentQAService

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVICE_PATH = PROJECT_ROOT / "services" / "document_qa_service.py"
BACKUP_PATH = PROJECT_ROOT / "services" / "document_qa_service.pre_structured_exp74.py"

IMPORT_ANCHOR = "from services.layout_aware_ocr_retrieval import build_raw_ocr_packet\n"
IMPORT_BLOCK = '''from services.layout_aware_ocr_retrieval import build_raw_ocr_packet
from parsing.unified_structured_document_builder import build_unified_structured_document
from services.structured_document_retrieval import retrieve_structured_answer
from services.answer_type_confidence import score_candidate
'''

NEW_BUILD = '''    def _raw_region_packet_item(self, evidence_id):
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

    def _structured_packet(self, question):
        structured_results = []

        for document in self.documents:
            if document.get("document_type") != "unclassified":
                continue

            structured_document = build_unified_structured_document(document)
            retrieval = retrieve_structured_answer(structured_document, question)
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

    def build_llm_packet(self, question):
        if all(d.get("document_type") == "unclassified" for d in self.documents):
            structured_packet = self._structured_packet(question)
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
'''

STRUCTURED_METHODS = '''    @staticmethod
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

'''

OLD_CHECK = '''        answer_check_passed, answer_check_reason = self._raw_ocr_answer_check(
            packet, llm_result
        ) if grounded else (None, "not_checked_unanswered")'''
NEW_CHECK = '''        answer_check_passed, answer_check_reason = self._answer_check(
            packet, llm_result
        ) if grounded else (None, "not_checked_unanswered")'''

OLD_CONF = '''        if grounded:
            confidence_metadata = self._raw_ocr_confidence_metadata(
                packet,
                llm_result,
            )
        else:'''
NEW_CONF = '''        if grounded:
            confidence_metadata = self._confidence_metadata(
                packet,
                llm_result,
            )
        else:'''


def main() -> int:
    if not SERVICE_PATH.is_file():
        raise FileNotFoundError(SERVICE_PATH)

    runtime_path = Path(inspect.getfile(DocumentQAService)).resolve()
    if runtime_path != SERVICE_PATH.resolve():
        raise RuntimeError(f"Unexpected runtime service path: {runtime_path}")

    source = SERVICE_PATH.read_text(encoding="utf-8")

    if "def _structured_packet(" in source:
        print("Structured integration already installed. No changes made.")
        return 0

    if not BACKUP_PATH.exists():
        shutil.copy2(SERVICE_PATH, BACKUP_PATH)
        print(f"[BACKUP] {BACKUP_PATH}")
    else:
        print(f"[BACKUP EXISTS] {BACKUP_PATH}")

    if "from services.structured_document_retrieval import" not in source:
        if IMPORT_ANCHOR not in source:
            raise RuntimeError("Import anchor not found; service not modified.")
        source = source.replace(IMPORT_ANCHOR, IMPORT_BLOCK, 1)

    old_build = inspect.getsource(DocumentQAService.build_llm_packet)
    if old_build.rstrip() not in source:
        raise RuntimeError("Exact build_llm_packet source not found; service not modified.")
    source = source.replace(old_build.rstrip(), NEW_BUILD.rstrip(), 1)

    answer_anchor = "    def answer(\n"
    if answer_anchor not in source:
        raise RuntimeError("answer() anchor not found; service not modified.")
    source = source.replace(answer_anchor, STRUCTURED_METHODS + answer_anchor, 1)

    if OLD_CHECK not in source:
        raise RuntimeError("Answer-check block not found; service not modified.")
    source = source.replace(OLD_CHECK, NEW_CHECK, 1)

    if OLD_CONF not in source:
        raise RuntimeError("Confidence block not found; service not modified.")
    source = source.replace(OLD_CONF, NEW_CONF, 1)

    compile(source, str(SERVICE_PATH), "exec")
    SERVICE_PATH.write_text(source, encoding="utf-8")

    print(f"[PATCHED] {SERVICE_PATH}")
    print("Syntax compilation passed.")
    print("Raw-OCR fallback and existing typed-confidence logic were preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
