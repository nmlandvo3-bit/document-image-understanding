"""Conservative receipt-total candidates from unchanged OCR geometry."""
import re


def asks_receipt_total(question):
    words = set(re.findall(r'\w+', question.casefold()))
    return bool(words & {'total', 'totals'}) and not bool(words & {
        'subtotal', 'subtotals', 'tax', 'change', 'cash', 'compare', 'sum',
        'combined', 'average', 'highest', 'lowest', 'difference'})


def receipt_total_context(document):
    regions = [e for e in document.get('evidence', []) if e.get('evidence_type') == 'ocr_region']
    texts = [' '.join(str(e.get('text', '')).split()) for e in regions]
    semantics = document.get('receipt_semantics') or {}
    # A payment form can contain every summary label. Require receipt identity
    # or transaction/item metadata as well; summary labels alone are insufficient.
    identity = any(re.search(r'\breceipt\s*(?:no\.?|number|#)', t, re.I) for t in texts)
    metadata = any(re.search(r'\b(?:cashier|items\s+sold)\s*:', t, re.I) for t in texts)
    summaries = sum(any(re.fullmatch(pattern, t, re.I) for t in texts)
                    for pattern in [r'SUBTOTAL\s*:?', r'TOTAL\s*:?', r'(?:CHANGE|PAID|CASH)\s*:?.*'])
    is_receipt = bool(identity or metadata) and summaries >= 2
    if semantics.get('status') == 'receipt_candidate' and not any(t.upper() == 'PAYMENT INFORMATION' for t in texts):
        is_receipt = True
    result = {'receipt_candidate': is_receipt, 'total_candidates': []}
    if not is_receipt:
        return result
    doc_id = document['document_id']
    def eid(e):
        return f"{doc_id}:ocr_region:{e['region_id']}"
    for label, text in zip(regions, texts):
        if not re.fullmatch(r'TOTAL\s*:?', text, re.I):
            continue
        lb = label.get('bbox') or {}
        for amount, value in zip(regions, texts):
            if not re.fullmatch(r'[$€£¥]?\s*\d+(?:,\d{3})*(?:\.\d{1,2})?', value):
                continue
            ab = amount.get('bbox') or {}
            try:
                lx, ly, lw, lh = [float(lb[k]) for k in ('x', 'y', 'width', 'height')]
                ax, ay, aw, ah = [float(ab[k]) for k in ('x', 'y', 'width', 'height')]
            except (KeyError, TypeError, ValueError):
                continue
            overlap = max(0, min(ly + lh, ay + ah) - max(ly, ay))
            if min(lh, ah) <= 0 or ax < lx + lw or overlap < .7 * min(lh, ah):
                continue
            result['total_candidates'].append({'quote': amount['text'],
                'evidence_item_ids': [eid(label), eid(amount)],
                'relationship': 'same_line_ocr_geometry'})
    return result


def receipt_total_listing(index, question):
    """Prepare a simple per-receipt list; other questions keep model routing."""
    words = set(re.findall(r'\w+', question.casefold()))
    if not asks_receipt_total(question) or not words.intersection({'list', 'show'}):
        return None
    if not words.intersection({'receipt', 'receipts'}) or words.intersection({
        'cashier', 'address', 'items', 'item', 'date', 'where', 'why', 'how'}):
        return None
    rows, reports, claims = [], [], []
    for document in index.get('documents', []):
        context = receipt_total_context(document)
        doc_id = document['document_id']
        if not context['receipt_candidate']:
            reports.append({'document_id':doc_id, 'status':'not_receipt_candidate'})
            continue
        candidates = context['total_candidates']
        values = list(dict.fromkeys(c['quote'] for c in candidates))
        if len(values) != 1:
            status = 'ambiguous_total' if values else 'missing_total'
            reports.append({'document_id':doc_id, 'status':status})
            rows.append({'document_id':doc_id, 'status':status, 'quote':None})
            continue
        ids = list(dict.fromkeys(eid for c in candidates for eid in c['evidence_item_ids']))
        claim = {'document_id':doc_id, 'quote':values[0], 'evidence_item_ids':ids}
        claims.append(claim)
        rows.append({**claim, 'status':'candidate'})
        reports.append({'document_id':doc_id, 'status':'candidate_total'})
    if not rows:
        return None
    complete = all(r['status'] in {'candidate_total','not_receipt_candidate'} for r in reports)
    return {'rows':rows, 'claims':claims, 'reports':reports, 'complete':complete,
            'status':'answered' if complete else 'partial_analysis'}


def apply_receipt_total_listing(result, listing):
    """Render original OCR totals without an error-prone second model rewrite."""
    if not listing:
        return result
    # This is a separately disclosed local result, not acceptance of a bad
    # model response. Provider/network/context errors stay visible.
    if result.get('status') not in {'answered','model_not_found','model_ambiguous','partial_analysis','analysis_failed'}:
        return result
    claims = listing['claims']
    rows = listing['rows']
    lines = [f"{r['document_id']}: {r['quote'] or '[' + r['status'] + ']'}" for r in rows]
    corrected = {**result, 'status':listing['status'], 'answer_type':'corpus_analysis',
        'answer':'\n'.join(lines), 'claims':claims,
        'evidence_item_ids':list(dict.fromkeys(eid for c in claims for eid in c['evidence_item_ids'])),
        'selection_method':'local_receipt_total_listing',
        'model_status':result['status'], 'model_failure_reason':result.get('failure_reason'),
        'receipt_total_listing':listing, 'local_coverage_complete':listing['complete']}
    corrected.pop('failure_reason',None)
    # Model inspection reports retain their meaning and are shown separately
    # from local extraction coverage, including model failures.
    return corrected
