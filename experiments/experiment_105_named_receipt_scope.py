"""Offline source-scope regression over uploaded OCR fixtures."""
import copy
import json
import sys
import types
from pathlib import Path
router=types.ModuleType('services.corpus_question_router')
router.answer_corpus_question=lambda *a:None
sys.modules['services.corpus_question_router']=router
from services.corpus_groq_assisted import answer_corpus_question_analysis as analyze
from services.receipt_item_qa import named_receipt_scope
from services.corpus_qa_presentation import present_corpus_answer
scans=json.loads((Path(__file__).parent/'fixtures/receipt_total_scans.json').read_text())
index={'documents':scans};original=copy.deepcopy(index)
fresh=next(d for d in scans if 'supermarket' in d['document_id'])
class Client:
    def __init__(self, fail=False):
        self.fail=fail;self.packets=[];self.chat=types.SimpleNamespace(completions=self)
    def create(self,**kwargs):
        p=json.loads(kwargs['messages'][1]['content']);self.packets.append(p)
        if self.fail:
            d=p['documents'][0]
            payload={'status':'answered','claims':[{'document_id':d['document_id'], 'quote':'invented', 'evidence_item_ids':[d['ocr_regions'][0][0]]}]}
        else:
            payload={'status':'not_found','claims':[]}
        return types.SimpleNamespace(choices=[types.SimpleNamespace(finish_reason='stop',message=types.SimpleNamespace(content=json.dumps(payload)))])
count=0
def check(label,condition):
    global count
    assert condition,label
    count+=1;print('[PASS]',label)
q='What is the most expensive item on the Fresh Mart receipt? Include its name and price.'
for provider in ('ollama','groq'):
    c=Client();r=analyze(index,q,provider=provider,client=c)
    check(provider+' sends only explicitly named receipt',[d['document_id'] for d in c.packets[0]['documents']]==[fresh['document_id']])
    check(provider+' original question retained',c.packets[0]['question']==q)
    check(provider+' source scope disclosed',r['corpus_documents_total']==3 and r['documents_in_scope']==1 and 'in requested scope: 1' in present_corpus_answer(r)[1])
    check(provider+' named comparison completes independently',r['status']=='answered' and 'Ground Beef' in r['answer'] and r['documents_analyzed']==1)
c=Client();r=analyze(index,'What is the most expensive item on the receipts?',provider='ollama',client=c)
check('all-receipt question still inspects full corpus',len(c.packets[0]['documents'])==3 and not r.get('scope_description'))
check('question with multiple named receipts stays general',named_receipt_scope(index,'Compare the Fresh Mart receipt and the campus receipt') is None)
check('unmatched source does not silently select an arbitrary document',named_receipt_scope(index,'What is on the Missing Shop receipt?') is None)
duplicate=copy.deepcopy(fresh);duplicate['document_id']='second_fresh_scan'
scope=named_receipt_scope({'documents':scans+[duplicate]},q)
check('multiple scans matching merchant remain in scope',len(scope['documents'])==2)
c=Client(True);r=analyze(index,q,provider='ollama',client=c)
check('citation failure inside scope is still rejected',r['model_status']=='analysis_failed' and r['selection_method']=='local_item_price_comparison')
check('scope filtering does not mutate OCR corpus',index==original)
print(f'RESULT: {count}/{count} checks passed. No OCR, Groq, Ollama or network requests were made.')
