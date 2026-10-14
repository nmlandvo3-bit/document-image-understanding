"""Receipt item candidates and transparent comparison of printed row prices.

No OCR edits, currency conversion, quantity multiplication or model execution.
Geometry is candidate evidence, not proof that OCR or receipt meaning is true.
"""
import re
from decimal import Decimal
from services.receipt_qa_context import receipt_total_context

SUMMARY = re.compile(r'^(?:SUB\s*TOTAL|TOTAL|(?:SALES\s+)?TAX|CASH|CHANGE|PAID)\b', re.I)
AMOUNT = re.compile(r'^\s*[.·]*\s*([$€£¥]?)\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)\s*$')
METADATA = re.compile(r'\b(?:date|time|cashier|customer|receipt|register|reg|trn|tel)\s*[:#]|\breceipt\s+no\b', re.I)


def _box(item):
    try:
        box = item['bbox']
        x, y, w, h = (float(box[k]) for k in ('x', 'y', 'width', 'height'))
        if w > 0 and h > 0:
            return x, y, w, h
    except (KeyError, TypeError, ValueError):
        pass
    return None


def _overlap(a, b):
    return max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])) / min(a[3], b[3])


def receipt_item_context(document):
    regions = [e for e in document.get('evidence', []) if e.get('evidence_type') == 'ocr_region']
    boxed = [(e, _box(e)) for e in regions if _box(e)]
    boxed.sort(key=lambda pair: (pair[1][1], pair[1][0]))
    headers = [(e, b) for e, b in boxed if str(e.get('text', '')).strip().casefold() in {'item', 'description', 'product'}]
    price_headers = [(e, b) for e, b in boxed if str(e.get('text', '')).strip().casefold() in {'price', 'amount'}]
    table_headers = [(e, b, pb) for e, b in headers for pe, pb in price_headers if _overlap(b, pb) >= .7 and pb[0] > b[0]]
    is_receipt = receipt_total_context(document)['receipt_candidate']
    if any(str(e.get('text', '')).strip().upper() == 'PAYMENT INFORMATION' for e in regions):
        is_receipt = False
    result = {'receipt_candidate':is_receipt, 'items':[], 'coverage':'not_receipt_candidate',
              'price_basis':'printed item-row amount; not inferred unit price'}
    if not is_receipt:
        return result
    result['coverage'] = 'unsupported_layout'
    summaries = [(e, b) for e, b in boxed if SUMMARY.match(str(e.get('text', '')).strip())]
    if not summaries:
        return result
    stop = min(b[1] for e, b in summaries)
    start = max((b[1] + b[3] for e, b, pb in table_headers), default=-1)
    if not table_headers:
        # Metadata establishes the top of a receipt's item area.
        start = max((b[1] + b[3] for e, b in boxed if b[1] < stop and
                     METADATA.search(str(e.get('text', '')))), default=-1)
        if start < 0:
            return result
    body = [(e, b) for e, b in boxed if b[1] >= start and b[1] < stop]
    amounts = [(e, b, AMOUNT.fullmatch(str(e.get('text', '')))) for e, b in body]
    amounts = [(e, b, m) for e, b, m in amounts if m]
    names = [(e, b) for e, b in body if re.search(r'[A-Za-z\u4e00-\u9fff]', str(e.get('text', '')))
             and not SUMMARY.match(str(e.get('text', '')).strip())]
    unresolved = 0
    for name, nb in names:
        candidates = [(amount, ab, m) for amount, ab, m in amounts
                      if ab[0] >= nb[0] + nb[2] and _overlap(nb, ab) >= .7
                      and (not table_headers or any(
                          ab[0] + ab[2] >= pb[0] and ab[0] >= pb[0] - pb[2]
                          for he, hb, pb in table_headers))]
        # Rightmost amount column is the printed row price, not quantity.
        if candidates:
            rightmost = max(ab[0] + ab[2] for a, ab, m in candidates)
            candidates = [(a, ab, m) for a, ab, m in candidates if ab[0] + ab[2] == rightmost]
        if len(candidates) != 1:
            unresolved += 1
            continue
        amount, ab, match = candidates[0]
        if sum(_overlap(b, ab) >= .7 for e, b in names) != 1:
            unresolved += 1
            continue
        def eid(e):
            return f"{document['document_id']}:ocr_region:{e['region_id']}"
        result['items'].append({'name':str(name['text']), 'price':str(amount['text']),
            'numeric_price':str(Decimal(match[2].replace(',', ''))),
            'currency_symbol':match[1] or None,
            'name_evidence_item_ids':[eid(name)], 'price_evidence_item_ids':[eid(amount)],
            'relationship':'same_line_ocr_geometry'})
    # Every amount in the item area must belong to a row (quantity cells excepted).
    used = {eid for item in result['items'] for eid in item['price_evidence_item_ids']}
    qty_headers = [(e, b) for e, b in boxed if str(e.get('text', '')).strip().casefold() in {'qty', 'quantity'}]
    for amount, ab, match in amounts:
        eid = f"{document['document_id']}:ocr_region:{amount['region_id']}"
        is_qty = any(qb[0] - qb[2] <= ab[0] <= qb[0] + 2 * qb[2] for e, qb in qty_headers) and not match[1]
        if eid not in used and not is_qty:
            unresolved += 1
    result['coverage'] = 'candidate_rows_complete' if result['items'] and not unresolved else 'incomplete_rows'
    result['unresolved_rows_or_amounts'] = unresolved
    return result


def _words(text):
    return re.findall(r'[a-z0-9]+', text.casefold())


def named_receipt_scope(index, question):
    """Resolve only an explicitly named receipt; unrecognized wording stays general."""
    if len(re.findall(r'\breceipts?\b', question, re.I)) > 1:
        return None
    scope = re.search(r'\b(?:on|in|from|at)\s+(?:the\s+)?(.+?)\s+receipts?\b', question.casefold())
    if not scope:
        return None
    scope_words = [w for w in _words(scope[1]) if w not in {'the', 'all', 'of', 'ther', 'these', 'this', 'selected', 'saved', 'scanned'}]
    if not scope_words:
        return None
    selected = []
    for d in index.get('documents', []):
        first = [e for e in d.get('evidence', []) if e.get('evidence_type') == 'ocr_region'][:3]
        identity = set(_words(d['document_id'] + ' ' + ' '.join(str(e.get('text','')) for e in first)))
        if set(scope_words).issubset(identity):
            selected.append(d)
    return {'description':' '.join(scope_words), 'documents':selected} if selected else None


def item_comparison(index, question):
    """Recognize simple item extrema; other questions retain general model routing."""
    q = question.casefold()
    if not re.search(r'\b(?:item|items|product|products)\b', q):
        return None
    mode = 'max' if re.search(r'most expensive|highest[ -](?:priced|price)|\bthe expensive (?:item|product)\b', q) else (
        'min' if re.search(r'cheapest|least expensive|lowest[ -](?:priced|price)', q) else None)
    if not mode or re.search(r'\b(?:cashier|address|where|total|quantity|unit|average|sum)\b', q):
        return None
    scope = named_receipt_scope(index, question)
    selected = scope['documents'] if scope else index.get('documents', [])
    if not selected:
        return None
    contexts = [(d, receipt_item_context(d)) for d in selected]
    receipts = [(d, c) for d, c in contexts if c['receipt_candidate']]
    if not receipts:
        return None
    if any(c['coverage'] != 'candidate_rows_complete' for d, c in receipts):
        return {'status':'comparison_incomplete', 'items':[], 'scope':scope['description'] if scope else 'all receipts',
                'reason':'At least one receipt has incomplete or unsupported item rows.'}
    items = [{**item, 'document_id':d['document_id']} for d, c in receipts for item in c['items']]
    groups = {}
    for item in items:
        groups.setdefault(item['currency_symbol'] or 'unspecified', []).append(item)
    winners = []
    for group in groups.values():
        extremum = (max if mode == 'max' else min)(Decimal(item['numeric_price']) for item in group)
        winners.extend(item for item in group if Decimal(item['numeric_price']) == extremum)
    same_document = len(receipts) == 1
    ambiguous = len(groups) > 1 or ('unspecified' in groups and not same_document)
    return {'status':'comparison_ambiguous' if ambiguous else 'answered', 'items':winners,
            'scope':scope['description'] if scope else 'all receipts', 'mode':mode,
            'reason':'Different or unspecified currencies prevent one global monetary comparison.' if ambiguous else
                     'Extremum checked against all paired item rows in the selected receipt candidates.',
            'receipt_count':len(receipts), 'price_basis':'printed item-row price; no currency conversion or inferred unit price'}


def apply_item_comparison(result, comparison):
    """Repair only with locally paired evidence, and disclose its origin."""
    if not comparison or result.get('status') not in {'answered','model_not_found','model_ambiguous','partial_analysis','analysis_failed'}:
        return result
    expected = []
    for item in comparison['items']:
        expected.extend([{'document_id':item['document_id'],'quote':item['name'],
                          'evidence_item_ids':item['name_evidence_item_ids']},
                         {'document_id':item['document_id'],'quote':item['price'],
                          'evidence_item_ids':item['price_evidence_item_ids']}])
    def keys(claims):
        return {(c['document_id'],c['quote'],tuple(c['evidence_item_ids'])) for c in claims}
    matched = result.get('status') == 'answered' and keys(result.get('claims',[])) == keys(expected)
    lines = [f"{i['document_id']}: {i['name']} — {i['price']}" for i in comparison['items']]
    return {**result, 'status':comparison['status'], 'model_status':result['status'],
        'model_failure_reason':result.get('failure_reason'),
        'local_coverage_complete':comparison['status']!='comparison_incomplete',
        'selection_method':'model_selection_checked_against_item_rows' if matched else 'local_item_price_comparison',
        'item_comparison':comparison, 'claims':expected,
        'answer':'\n'.join(lines) or None,
        'evidence_item_ids':list(dict.fromkeys(eid for c in expected for eid in c['evidence_item_ids']))}
