"""Groq analysis restricted to the explicitly selected unified document."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from services.corpus_groq_assisted import answer_corpus_question_analysis
from services.corpus_qa_presentation import present_corpus_answer


def answer_selected_document(path: Path, question: str, client: Any = None, provider: str = "groq") -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(document, dict) or not isinstance(document.get('document_id'), str):
        raise ValueError('Select a unified document JSON with a document_id.')
    if not isinstance(document.get('evidence'), list):
        raise ValueError('Selected document has no OCR evidence list.')
    result = answer_corpus_question_analysis({'documents': [document]}, question, client=client, provider=provider)
    answer, _ = present_corpus_answer(result)
    grounded = result.get('status') == 'answered'
    lookup = {
        f"{document['document_id']}:ocr_region:{item['region_id']}": item
        for item in document['evidence']
        if item.get('evidence_type') == 'ocr_region'
    }
    evidence = [{
        'evidence_item_id': eid, 'document_id': document['document_id'],
        'item_type': 'ocr_region', 'content': {'text': lookup[eid]['text']},
    } for eid in result.get('evidence_item_ids', [])]
    return {
        **result, 'question': question, 'answer': answer,
        'provider': f'{provider}_selected_document_analysis', 'model': result.get('model'),
        'grounded': grounded, 'evidence': evidence,
        'evidence_validation': {'valid': grounded},
        'answer_mode': 'extractive_ocr_analysis',
        'answer_consistency': {'reason': 'Exact OCR quotes and citations checked' if grounded else result['status']},
        'confidence_reason': 'Citation and quote validation only; relevance and completeness are not verified.',
        'uncertainty': 'OCR and model selection may be incorrect; no semantic verification.',
    }
