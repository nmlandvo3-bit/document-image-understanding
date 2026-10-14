"""Offline coverage and provenance regression for corpus batching."""
import json
import sys
import types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
router=types.ModuleType('services.corpus_question_router');router.answer_corpus_question=lambda *a:None
sys.modules['services.corpus_question_router']=router
from services.corpus_groq_assisted import answer_corpus_question_analysis as analyze
from services.corpus_qa_presentation import present_corpus_answer
class LimitError(Exception):
 status_code=413
class Client:
 def __init__(self, failed=None):
  self.failed=failed; self.documents_seen=[]
  self.chat=types.SimpleNamespace(completions=self)
 def create(self, **kwargs):
  packet=json.loads(kwargs['messages'][1]['content']); docs=packet['documents']
  if len(docs)>1 and packet['analysis_stage']!='corpus_synthesis': raise LimitError()
  if packet['analysis_stage']=='corpus_synthesis':
   payload={'status':'answered','claims':[{'document_id':d['document_id'],'quote':'$10.00','evidence_item_ids':[d['ocr_regions'][0][0]]} for d in docs]}
   return types.SimpleNamespace(choices=[types.SimpleNamespace(finish_reason='stop',message=types.SimpleNamespace(content=json.dumps(payload)))])
  d=docs[0];self.documents_seen.append(d['document_id'])
  payload={'status':'answered','claims':[{'document_id':d['document_id'],'quote':'$10.00','evidence_item_ids':[d['ocr_regions'][0][0]]}]}
  if d['document_id']==self.failed: payload['claims'][0]['quote']='invented'
  return types.SimpleNamespace(choices=[types.SimpleNamespace(finish_reason='stop',message=types.SimpleNamespace(content=json.dumps(payload)))])
def document(name):
 return {'document_id':name,'evidence':[{'evidence_type':'ocr_region','region_id':1,'text':'Total $10.00'}]}
index={'documents':[document('a'),document('b'),document('c')]}
checks=0
def check(label,ok):
 global checks
 assert ok,label
 checks+=1;print('[PASS]',label)
c=Client();r=analyze(index,'List printed amounts',client=c,provider='ollama')
check('oversized request becomes bounded document calls',c.documents_seen==['a','b','c'])
check('all documents have explicit coverage',r['coverage_complete'] and r['documents_analyzed']==3 and len(r['batch_reports'])==3)
check('all accepted totals retained',len(r['claims'])==3 and r['status']=='answered')
check('same amount from different documents preserved',r['evidence_item_ids']==['a:ocr_region:1','b:ocr_region:1','c:ocr_region:1'])
check('coverage visible in GUI','3/3' in present_corpus_answer(r)[1])
c=Client('b');r=analyze(index,'List printed amounts',client=c,provider='ollama')
check('failed document prevents complete result',r['status']=='partial_analysis' and not r['coverage_complete'])
check('failed document identified',r['batch_reports'][1]['status']=='analysis_failed' and bool(r['batch_reports'][1]['failure_reason']))
check('only validated partial quotes retained',len(r['claims'])==2 and 'b:ocr_region:1' not in r['evidence_item_ids'])
check('GUI prominently marks incomplete results',present_corpus_answer(r)[0].startswith('INCOMPLETE ANALYSIS'))
c=Client();r=analyze(index,'Which receipt has highest total?',client=c,provider='ollama')
check('comparison reaches extraction and final synthesis',r['status']=='answered' and c.documents_seen==['a','b','c'] and r['synthesis_status']=='answered')
c=Client();r=analyze(index,'List printed amounts',client=c,provider='ollama',max_context_bytes=1)
check('oversized individual documents remain explicit failures',r['status']=='partial_analysis' and r['documents_analyzed']==0 and not c.documents_seen)
check('original evidence unchanged',index['documents'][0]['evidence'][0]['text']=='Total $10.00')
print(f'RESULT: {checks}/{checks} checks passed. No OCR, Groq, or network requests were made.')
