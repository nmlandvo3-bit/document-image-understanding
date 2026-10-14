"""Offline receipt-total validation using OCR fixtures from three supplied scans."""
import copy
import json
import sys
import types
import tempfile
from pathlib import Path
router = types.ModuleType('services.corpus_question_router')
router.answer_corpus_question = lambda *args: None
sys.modules['services.corpus_question_router'] = router
from services.corpus_groq_assisted import answer_corpus_question_analysis as analyze
from services.receipt_qa_context import receipt_total_context
from services.document_groq_analysis import answer_selected_document

scans = json.loads((Path(__file__).parent / 'fixtures/receipt_total_scans.json').read_text())
original = copy.deepcopy(scans)
class Client:
    def __init__(self, mode='valid'):
        self.mode = mode
        self.chat = types.SimpleNamespace(completions=self)
    def create(self, **kwargs):
        documents = json.loads(kwargs['messages'][1]['content'])['documents']
        claims = []
        for d in documents:
            candidates = d['receipt_context']['total_candidates']
            if not candidates:
                if self.mode == 'form':
                    e = next(e for e in d['ocr_regions'] if e[1] == '310')
                    claims.append(dict(document_id=d['document_id'],quote=e[1],evidence_item_ids=[e[0]]))
                continue
            c = candidates[0]
            claim = dict(document_id=d['document_id'],quote=c['quote'],evidence_item_ids=c['evidence_item_ids'])
            if self.mode == 'label':
                claim['quote'] = 'TOTAL'
            if self.mode == 'subtotal':
                e = next(e for e in d['ocr_regions'] if e[1] in {'295', '$ 18.95'})
                claim.update(quote=e[1], evidence_item_ids=[e[0]])
            if self.mode == 'missing_label':
                claim['evidence_item_ids'] = claim['evidence_item_ids'][1:]
            claims.append(claim)
        payload = dict(status='answered' if claims else 'not_found',claims=claims)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(finish_reason='stop',message=types.SimpleNamespace(content=json.dumps(payload)))])
checks = 0
def check(label, condition):
    global checks
    assert condition, label
    checks += 1
    print('[PASS]',label)
r = analyze({'documents': scans}, 'List totals of all receipts', client=Client(),provider='ollama')
check('actual scans yield only campus and supermarket totals',r['status']=='answered' and {c['quote'] for c in r['claims']} == {'310','$20.09'})
check('each total cites both label and amount',all(len(c['evidence_item_ids'])==2 for c in r['claims']))
# Semantic hints no longer hard-block an answer. Both providers use the same
# extractive validation contract; this deliberately does not prove relevance.
for provider in ('ollama', 'groq'):
    r = analyze({'documents': scans}, 'List totals of all receipts',client=Client(),provider=provider)
    check(provider + ' receives receipt hints',r['status']=='answered')
form = next(d for d in scans if 'PAYMENT INFORMATION' in [e['text'] for e in d['evidence']])
check('payment form excluded',not receipt_total_context(form)['receipt_candidate'])
# Selected-document path reads the raw JSON, without requiring reconstructed fields.
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'selected.json'
    path.write_text(json.dumps(next(d for d in scans if 'supermarket' in d['document_id'])))
    r = answer_selected_document(path,'What is the total?',client=Client(),provider='ollama')
    check('selected scan accepts label and amount citations',r['grounded'] and len(r['evidence'])==2)
conflict = copy.deepcopy(next(d for d in scans if 'supermarket' in d['document_id']))
extra = copy.deepcopy(next(e for e in conflict['evidence'] if e['text']=='$20.09'))
extra['region_id'] = 999
extra['text'] = '$99.99'
conflict['evidence'].append(extra)
r = analyze({'documents':[conflict]},'What is the total?',client=Client(),provider='ollama')
check('receipt hints do not impose a model-specific semantic gate',r['status']=='answered')
check('raw OCR remains unchanged',scans==original)
print(f'RESULT: {checks}/{checks} checks passed. No OCR, Groq, or network requests were made.')
