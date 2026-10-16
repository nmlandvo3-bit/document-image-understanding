"""Offline configurable-model regressions. No model or network requests."""
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch
router=types.ModuleType('services.corpus_question_router')
router.answer_corpus_question=lambda *a:None
sys.modules['services.corpus_question_router']=router
from services.ollama_client import OllamaClient, OllamaError, resolve_ollama_model
from services.corpus_groq_assisted import answer_corpus_question_analysis as analyze
from services.document_groq_analysis import answer_selected_document
from services.corpus_qa_presentation import present_corpus_answer
QWEN='qwen3.8:27b'; LLAMA='llama3.1:8b'
index={'documents':[{'document_id':name,'evidence':[{'evidence_type':'ocr_region','region_id':1,'text':'Cashier: Sarah M.'}]} for name in ('one','two')]}
count=0
def check(label,ok):
    global count
    assert ok,label
    count+=1;print('[PASS]',label)
class Client:
    def __init__(self,batched=False):
        self.batched=batched;self.calls=[];self.chat=types.SimpleNamespace(completions=self)
    def create(self,**kwargs):
        p=json.loads(kwargs['messages'][1]['content']);self.calls.append((kwargs['model'],p))
        if self.batched and p['analysis_stage']=='answer':raise OllamaError('force batches',413)
        claims=[{'document_id':d['document_id'],'quote':'Sarah M.','evidence_item_ids':[d['ocr_regions'][0][0]]} for d in p['documents']]
        return types.SimpleNamespace(choices=[types.SimpleNamespace(finish_reason='stop',message=types.SimpleNamespace(content=json.dumps({'status':'answered','claims':claims})))])
with patch.dict(os.environ,{},clear=True):
    check('legacy default preserved',resolve_ollama_model()==LLAMA)
for model in (QWEN,LLAMA):
    with patch.dict(os.environ,{'DOCUMENT_QA_OLLAMA_MODEL':model},clear=True):
        c=Client();r=analyze(index,'Who was the cashier?',provider='ollama',client=c)
        check(model+' selected for corpus',r['model']==model and c.calls[0][0]==model and r['status']=='answered')
        check(model+' visible in corpus evidence',model in present_corpus_answer(r)[1])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'scan.json';path.write_text(json.dumps(index['documents'][0]))
            c=Client();r=answer_selected_document(path,'Who was the cashier?',provider='ollama',client=c)
            check(model+' selected for document',r['model']==model and c.calls[0][0]==model and r['grounded'])
            check(model+' selected document does not leak corpus',len(c.calls[0][1]['documents'])==1)
        c=Client(True);r=analyze(index,'Who was the cashier?',provider='ollama',client=c)
        check(model+' retained across batches and synthesis',r['model']==model and len(c.calls)==4 and all(m==model for m,p in c.calls))
with patch.dict(os.environ,{'DOCUMENT_QA_OLLAMA_MODEL':QWEN},clear=True):
    c=Client();r=analyze(index,'Who was the cashier?',provider='ollama',client=c,ollama_model=LLAMA)
    check('explicit model overrides environment',r['model']==LLAMA and c.calls[0][0]==LLAMA)
    c=Client();r=analyze(index,'Who was the cashier?',provider='groq',client=c)
    check('local setting does not change Groq model',r['model']=='openai/gpt-oss-20b' and c.calls[0][0]=='openai/gpt-oss-20b')
for invalid in ('','bad tag','bad\nmodel'):
    try:resolve_ollama_model(invalid);ok=False
    except ValueError:ok=True
    check('invalid model setting rejected '+repr(invalid),ok)
payloads=[]
def transport(payload):
    payloads.append(payload)
    return {'done':True,'message':{'content':'{"status":"not_found","claims":[]}'}}
with patch.dict(os.environ,{'DOCUMENT_QA_OLLAMA_TIMEOUT_SEC':'300'},clear=True):
    c=OllamaClient(transport=transport)
    c.create(model=QWEN,messages=[{'role':'user','content':'Return JSON.'}])
    check('selected tag reaches native Ollama payload',payloads[-1]['model']==QWEN)
    check('structured output retained',payloads[-1]['format']['properties']['status']['enum']==['answered','not_found','ambiguous'])
    check('longer timeout configurable',c.timeout==300)
with patch.dict(os.environ,{'DOCUMENT_QA_OLLAMA_TIMEOUT_SEC':'300'},clear=True):
    check('explicit timeout overrides environment',OllamaClient(timeout=30,transport=transport).timeout==30)
check('model thinking default preserved','think' not in payloads[-1])
for setting in ('true','false'):
    with patch.dict(os.environ,{'DOCUMENT_QA_OLLAMA_THINK':setting},clear=True):
        c=OllamaClient(transport=transport)
        c.create(model=QWEN,messages=[{'role':'user','content':'Return JSON.'}])
        check('explicit thinking setting forwarded '+setting,payloads[-1]['think'] is (setting=='true'))
with patch.dict(os.environ,{'DOCUMENT_QA_OLLAMA_THINK':'invalid'},clear=True):
    try:OllamaClient(transport=transport);ok=False
    except ValueError:ok=True
    check('invalid thinking setting rejected',ok)
print(f'RESULT: {count}/{count} checks passed. No OCR, Groq, Ollama or network requests were made.')
