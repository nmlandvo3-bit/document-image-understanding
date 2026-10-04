"""Offline selected scan scope and citation checks."""
import json
import sys
import tempfile
import types
from pathlib import Path
router = types.ModuleType('services.corpus_question_router')
router.answer_corpus_question = lambda *a: None
sys.modules['services.corpus_question_router'] = router
from services.document_groq_analysis import answer_selected_document
class Client:
    def __init__(self, quote='Sarah M.'):
        self.quote = quote
        self.chat = types.SimpleNamespace(completions=self)
        self.packet = None
    def create(self, **kwargs):
        self.packet = json.loads(kwargs['messages'][1]['content'])
        payload = {'status': 'answered', 'claims': [{'document_id': 'selected',
            'quote': self.quote, 'evidence_item_ids': ['e0']}]}
        return types.SimpleNamespace(choices=[types.SimpleNamespace(finish_reason='stop',
            message=types.SimpleNamespace(content=json.dumps(payload)))])
def check(label, ok):
    assert ok, label
    print('[PASS]', label)
with tempfile.TemporaryDirectory() as folder:
    p=Path(folder)/'scan.json'
    p.write_text(json.dumps({'document_id': 'selected', 'evidence': [
        {'evidence_type': 'ocr_region', 'region_id': 4, 'text': 'Cashier: Sarah M.'}]}))
    c=Client()
    r=answer_selected_document(p, 'Who is the cashier?', client=c)
    check('only selected document sent', len(c.packet['documents']) == 1 and c.packet['documents'][0]['document_id'] == 'selected')
    check('question retained for GUI request checks', r['question'] == 'Who is the cashier?')
    check('answer and original OCR citation retained', 'Sarah M.' in r['answer'] and r['evidence'][0]['evidence_item_id'] == 'selected:ocr_region:4')
    check('GUI receives cited OCR text', r['evidence'][0]['content']['text'] == 'Cashier: Sarah M.')
    check('accepted quote marked grounded with semantic limitation', r['grounded'] and r['semantic_verified'] is False)
    bad=answer_selected_document(p, '?', client=Client('John'))
    check('invented answer rejected', not bad['grounded'] and not bad['evidence'] and bad['status'] == 'analysis_failed')
print('RESULT: 6/6 checks passed. No OCR, Groq, or network requests were made.')
