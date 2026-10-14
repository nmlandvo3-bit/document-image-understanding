"""Offline receipt-list regressions for full scan coverage and failed synthesis."""
import copy
import json
import sys
import types
from pathlib import Path
router=types.ModuleType('services.corpus_question_router')
router.answer_corpus_question=lambda *a:None
sys.modules['services.corpus_question_router']=router
from services.corpus_groq_assisted import answer_corpus_question_analysis as analyze
from services.corpus_qa_presentation import present_corpus_answer
from services.ollama_client import OllamaError
scans=json.loads((Path(__file__).parent/'fixtures/receipt_total_scans.json').read_text())
original=copy.deepcopy(scans)
class Client:
    def __init__(self,batch=False):
        self.batch=batch;self.calls=[];self.chat=types.SimpleNamespace(completions=self)
    def create(self,**kwargs):
        p=json.loads(kwargs['messages'][1]['content']);self.calls.append(p)
        if self.batch and p['analysis_stage']=='answer':
            raise OllamaError('Force batches',413)
        d=p['documents'][0]
        # Deliberately invalid model selection; local total evidence is separate.
        payload={'status':'answered','claims':[{'document_id':d['document_id'],'quote':'invented','evidence_item_ids':['e0']}]}
        return types.SimpleNamespace(choices=[types.SimpleNamespace(finish_reason='stop',message=types.SimpleNamespace(content=json.dumps(payload)))])
count=0
def check(label,ok):
    global count
    assert ok,label
    count+=1;print('[PASS]',label)
q='list the total of all receipts'
for provider in ('ollama','groq'):
    c=Client();r=analyze({'documents':scans},q,client=c,provider=provider)
    check(provider+' uses separately disclosed local selection',r['status']=='answered' and r['selection_method']=='local_receipt_total_listing' and r['model_status']=='analysis_failed')
    check(provider+' lists one total per receipt',{x['quote'] for x in r['claims']}=={'310','$20.09'} and len(r['claims'])==2)
    check(provider+' excludes payment form',not any('ChatGPT' in x['document_id'] for x in r['claims']))
    check(provider+' includes label and amount citations',all(len(x['evidence_item_ids'])==2 for x in r['claims']))
    check(provider+' does not hide model validation error',r['model_failure_reason'].startswith('Unsupported quote or citation') and 'Model path failure' in present_corpus_answer(r)[1])
    check(provider+' local result is labelled',present_corpus_answer(r)[0].startswith('LOCAL RECEIPT TOTALS'))
c=Client(True);r=analyze({'documents':scans},q,client=c,provider='ollama')
check('no second model synthesis for receipt listing',len(c.calls)==4 and not any(p['analysis_stage']=='corpus_synthesis' for p in c.calls))
check('local coverage separate from model coverage',r['local_coverage_complete'] and not r['coverage_complete'])
check('failed model batches do not invent local totals',r['status']=='answered' and len(r['claims'])==2)
supermarket=next(d for d in scans if 'supermarket' in d['document_id'])
duplicate=copy.deepcopy(supermarket)
label=copy.deepcopy(next(e for e in duplicate['evidence'] if e['text']=='TOTAL:'))
label['region_id']=999
# Duplicate labels are one local total with combined citations.
duplicate['evidence'].append(label)
r=analyze({'documents':[duplicate]},q,client=Client(),provider='ollama')
check('same-document total repeats collapse into one row',len(r['claims'])==1 and r['claims'][0]['quote']=='$20.09')
conflict=copy.deepcopy(supermarket)
amount=copy.deepcopy(next(e for e in conflict['evidence'] if e['text']=='$20.09'))
amount['region_id']=999;amount['text']='$99.99';conflict['evidence'].append(amount)
r=analyze({'documents':[conflict]},q,client=Client(),provider='ollama')
check('conflicting totals remain explicit',r['status']=='partial_analysis' and '[ambiguous_total]' in r['answer'] and not r['claims'])
missing=copy.deepcopy(supermarket)
missing['evidence']=[e for e in missing['evidence'] if e['text']!='$20.09']
r=analyze({'documents':[missing]},q,client=Client(),provider='ollama')
check('missing amount is reported rather than inferred',r['status']=='partial_analysis' and '[missing_total]' in r['answer'])
class Offline(Client):
    def create(self,**kwargs):raise OllamaError('offline',503)
r=analyze({'documents':scans},q,client=Offline(),provider='ollama')
check('direct provider unavailability stays visible',r['status']=='local_model_unavailable')
check('raw OCR unchanged',scans==original)
print(f'RESULT: {count}/{count} checks passed. No OCR, Groq, Ollama or network requests were made.')
