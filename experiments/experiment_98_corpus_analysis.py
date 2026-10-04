"""Offline corpus analysis checks; no OCR or network calls."""
import json
import sys
import types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# The legacy router is not needed by this independent analysis test.
router = types.ModuleType('services.corpus_question_router')
router.answer_corpus_question = lambda *a: None
sys.modules['services.corpus_question_router'] = router
from services.corpus_groq_assisted import answer_corpus_question_analysis as analyze
from services.corpus_qa_presentation import present_corpus_answer
class Client:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []
        self.chat = types.SimpleNamespace(completions=self)
    def create(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=json.dumps(self.payload)), finish_reason='stop')])
index = {'documents': [{'document_id': 'd1', 'evidence': [
    {'evidence_type': 'ocr_region', 'region_id': 1, 'text': 'Cashier: Sarah M.'},
    {'evidence_type': 'ocr_region', 'region_id': 2, 'text': '123 Fashion Blvd, New York'}]}]}
claims = [{'document_id': 'd1', 'quote': 'Sarah M.', 'evidence_item_ids': ['d1:ocr_region:1']},
          {'document_id': 'd1', 'quote': '123 Fashion Blvd, New York', 'evidence_item_ids': ['d1:ocr_region:2']}]
def check(label, ok):
    assert ok, label
    print('[PASS]', label)
c = Client({'status': 'answered', 'claims': claims})
r = analyze(index, 'Who is the cashier and where is it?', client=c)
check('multi-part answer selected by model', r['status'] == 'answered' and len(r['claims']) == 2)
check('one model call contains complete corpus OCR', len(c.calls) == 1 and 'Cashier: Sarah M.' in c.calls[0]['messages'][1]['content'])
check('GUI shows answer and OCR provenance', 'Sarah M.' in present_corpus_answer(r)[0] and 'd1:ocr_region:2' in present_corpus_answer(r)[1])
for label, bad in [('invented quote', {**claims[0], 'quote': 'John'}),
                   ('unknown ID', {**claims[0], 'evidence_item_ids': ['x']}),
                   ('wrong document', {**claims[0], 'document_id': 'd2'}),
                   ('wrong region', {**claims[0], 'evidence_item_ids': ['d1:ocr_region:2']})]:
    check(label + ' rejected', analyze(index, 'cashier?', client=Client({'status': 'answered', 'claims': [bad]}))['status'] == 'analysis_failed')
c = Client({})
check('oversized corpus refused before request', analyze(index, '?', client=c, max_context_bytes=1)['status'] == 'context_limit' and not c.calls)
for status in ['not_found', 'ambiguous']:
    check(status + ' preserved', analyze(index, '?', client=Client({'status': status, 'claims': []}))['status'] == 'model_' + status)
check('source OCR unchanged', index['documents'][0]['evidence'][0]['text'] == 'Cashier: Sarah M.')
short_claim = {**claims[0], 'evidence_item_ids': ['e0']}
short_result = analyze(index, 'cashier?', client=Client({'status': 'answered', 'claims': [short_claim]}))
check('short citation restored to original OCR ID', short_result['evidence_item_ids'] == ['d1:ocr_region:1'])
class LimitError(Exception):
    status_code = 413
class LimitedClient(Client):
    def create(self, **kwargs):
        raise LimitError('request too large')
check('provider limit reported explicitly', analyze(index, '?', client=LimitedClient({}))['status'] == 'request_limit')
print('RESULT: 13/13 checks passed. No OCR, Groq, or network requests were made.')
