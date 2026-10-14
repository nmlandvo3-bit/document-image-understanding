"""Offline routing, context, synthesis and citation regressions; no live model."""
import copy
import json
import os
import sys
import types
from unittest.mock import patch
router = types.ModuleType('services.corpus_question_router')
router.answer_corpus_question = lambda *args: None
sys.modules['services.corpus_question_router'] = router
from services.corpus_groq_assisted import answer_corpus_question_analysis as analyze
from services.ollama_client import OllamaClient, OllamaError


def document(name, item, price):
    return {'document_id':name, 'evidence':[
        {'evidence_type':'ocr_region','region_id':1,'text':item,
         'bbox':{'x':10,'y':10,'width':120,'height':20}},
        {'evidence_type':'ocr_region','region_id':2,'text':price,
         'bbox':{'x':200,'y':10,'width':60,'height':20}}]}
index = {'documents':[document('fresh','Ground Beef','$5.99'),
                       document('trend','Leather Bomber Jacket','$129.99')]}
original = copy.deepcopy(index)
class Client:
    def __init__(self, batched=False, fail=None, synthesis='answered'):
        self.batched, self.fail, self.synthesis = batched, fail, synthesis
        self.calls = []
        self.chat = types.SimpleNamespace(completions=self)
    def create(self, **kwargs):
        p = json.loads(kwargs['messages'][1]['content'])
        self.calls.append((p,kwargs))
        docs = p['documents']
        if self.batched and p['analysis_stage']=='answer':
            raise OllamaError('force bounded requests',413)
        status = 'answered'
        if p['analysis_stage']=='corpus_synthesis':
            if self.synthesis=='error':
                raise RuntimeError('test failure')
            status = self.synthesis
        if p['analysis_stage']=='answer' or p['analysis_stage']=='corpus_synthesis':
            docs = [d for d in docs if d['document_id']=='trend']
        claims = [{'document_id':d['document_id'],'quote':text,'evidence_item_ids':[eid]}
                  for d in docs for eid,text in d['ocr_regions']]
        if self.fail and any(d['document_id']==self.fail for d in docs):
            claims[0]['quote']='invented'
        if status!='answered':
            claims=[]
        return types.SimpleNamespace(choices=[types.SimpleNamespace(finish_reason='stop',
            message=types.SimpleNamespace(content=json.dumps({'status':status,'claims':claims})))])

count=0
def check(label,condition):
    global count
    assert condition,label
    count+=1
    print('[PASS]',label)
question='What is the most expensive item on the receipts?'
a,b=Client(),Client()
ra=analyze(index,question,client=a,provider='ollama')
rb=analyze(index,question,client=b,provider='groq')
check('same complete corpus and original question reach both providers',a.calls[0][0]==b.calls[0][0])
check('same evidence instructions reach both providers',a.calls[0][1]['messages']==b.calls[0][1]['messages'])
check('local answer retains item name and price',[c['quote'] for c in ra['claims']]==['Leather Bomber Jacket','$129.99'])
check('layout accompanies OCR without replacing its text',len(a.calls[0][0]['documents'][0]['ocr_layout'])==2)
check('model results preserve original citation IDs',ra['evidence_item_ids']==['trend:ocr_region:1','trend:ocr_region:2'])
for q in [question,'Who was the cashier and where is the store?', 'Which receipt has the lowest total?']:
    c=Client(batched=True)
    r=analyze(index,q,client=c,provider='ollama')
    check('no keyword gate: '+q,r['status']=='answered' and c.calls[-1][0]['analysis_stage']=='corpus_synthesis')
    check('original question retained through every stage',all(p['question']==q for p,k in c.calls))
c=Client(batched=True,fail='fresh');r=analyze(index,question,client=c,provider='ollama')
check('failed document prevents a global answer',r['status']=='partial_analysis' and not r['coverage_complete'])
check('no final synthesis after incomplete extraction',all(p['analysis_stage']!='corpus_synthesis' for p,k in c.calls))
c=Client(batched=True,synthesis='error');r=analyze(index,question,client=c,provider='ollama')
check('synthesis failure is visible with complete extraction',r['status']=='partial_analysis' and r['coverage_complete'] and r['failure_reason'])
for status in ('ambiguous','not_found'):
    c=Client(batched=True,synthesis=status);r=analyze(index,question,client=c,provider='ollama')
    check('final '+status+' preserved',r['status']=='model_'+status and not r.get('claims'))
c=Client(batched=True,fail='trend');r=analyze(index,question,client=c,provider='ollama')
check('invented per-document quote rejected',r['status']=='partial_analysis' and not any(c['document_id']=='trend' for c in r['claims']))
check('source OCR unchanged',index==original)

payloads=[]
def transport(payload):
    payloads.append(payload)
    return {'done':True,'message':{'content':'{"status":"not_found","claims":[]}'}}
messages=[{'role':'user','content':'x'*9000}]
with patch.dict(os.environ,{},clear=True):
    c=OllamaClient(transport=transport)
    c.create(model='llama3.1:8b',messages=messages)
check('default context grows for a complete request',payloads[-1]['options']['num_ctx']==16384)
check('context growth does not truncate messages',payloads[-1]['messages']==messages)
with patch.dict(os.environ,{'DOCUMENT_QA_OLLAMA_NUM_CTX':'8192'},clear=True):
    c=OllamaClient(transport=transport)
    try:
        c.create(model='llama3.1:8b',messages=messages)
        fixed=False
    except OllamaError as e:
        fixed=e.status_code==413
check('explicit context configuration respected',fixed)
with patch.dict(os.environ,{},clear=True):
    c=OllamaClient(transport=transport)
    before=len(payloads)
    try:
        c.create(model='llama3.1:8b',messages=[{'role':'user','content':'x'*40000}])
        bounded=False
    except OllamaError as e:
        bounded=e.status_code==413
check('automatic context has a finite ceiling',bounded and len(payloads)==before)
print(f'RESULT: {count}/{count} checks passed. No OCR, Groq, Ollama or network requests were made.')
