"""Eight screenshot phrasings over actual OCR plus a synthetic Trend table.

Mock responses deliberately fail validation. This tests independent local
answers and diagnostics, not Llama's live accuracy.
"""
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
from services.ollama_client import OllamaError
scans=json.loads((Path(__file__).parent/'fixtures/receipt_total_scans.json').read_text())
trend={'document_id':'synthetic_trend_lounge','evidence':[]}
def add(rid,text,x,y,w=100):
    trend['evidence'].append({'evidence_type':'ocr_region','region_id':rid,'text':text,
        'bbox':{'x':x,'y':y,'width':w,'height':12}})
add(1,'THE TREND LOUNGE',10,0,200);add(2,'Receipt #: TEST',10,20,180)
add(3,'Item',10,50);add(4,'Qty',220,50,25);add(5,'Price',300,50,60)
add(6,'Leather Bomber Jacket',10,80,180);add(7,'1',220,80,20);add(8,'$129.99',300,80,60)
add(9,'Athletic Crew Socks',10,105,180);add(10,'3',220,105,20);add(11,'$18.00',300,105,60)
add(12,'Subtotal',10,130);add(13,'$147.99',300,130,60)
add(14,'Total',10,155);add(15,'$269.68',300,155,60)
others=[{'document_id':f'other_{i}','evidence':[{'evidence_type':'ocr_region','region_id':1,'text':'of'}]} for i in range(3)]
index={'documents':scans+[trend]+others};original=copy.deepcopy(index)
class Client:
    def __init__(self):
        self.calls=[];self.chat=types.SimpleNamespace(completions=self)
    def create(self,**kwargs):
        p=json.loads(kwargs['messages'][1]['content']);self.calls.append(p)
        if p['analysis_stage']=='answer':raise OllamaError('force batches',413)
        d=p['documents'][0]
        payload={'status':'answered','claims':[{'document_id':d['document_id'],'quote':'invented','evidence_item_ids':['e0']}]}
        return types.SimpleNamespace(choices=[types.SimpleNamespace(finish_reason='stop',message=types.SimpleNamespace(content=json.dumps(payload)))])
count=0
def check(label,condition):
    global count
    assert condition,label
    count+=1;print('[PASS]',label)
questions=[('which item is the cheapest?','min'),('list all ther receipts total?','totals'),
 ('list all the receipts total?','totals'),('whats the expensive item?','max'),
 ('whats the expensive item all of the receipts?','max'),('list all the total of the receipts?','totals'),
 ('whats the cheapest item in all of the receipts?','min'),('whats the expensive item in all of the receipts?','max')]
for number,(q,task) in enumerate(questions,1):
    c=Client();r=analyze(index,q,provider='ollama',client=c,max_context_bytes=50000)
    check(f'picture {number}: all seven sources retained',named_receipt_scope(index,q) is None and len(c.calls[0]['documents'])==7)
    check(f'picture {number}: no unnecessary final synthesis',not any(p['analysis_stage']=='corpus_synthesis' for p in c.calls))
    if task=='totals':
        check(f'picture {number}: three unique receipt totals',r['status']=='answered' and {x['quote'] for x in r['claims']}=={'310','$20.09','$269.68'} and len(r['claims'])==3)
    else:
        expected={'Pen Blue 0.5','Whole Wheat Bread.'} if task=='min' else {'Notebook A4','Leather Bomber Jacket'}
        check(f'picture {number}: correct candidates and currency ambiguity',r['status']=='comparison_ambiguous' and {x['name'] for x in r['item_comparison']['items']}==expected)
    display=present_corpus_answer(r)[1]
    check(f'picture {number}: processing and model success separated','Documents processed: 7/7' in display and 'Successful model results: 0/7' in display)
    check(f'picture {number}: active revision visible','Q&A revision: receipt_qa_107' in display)
check('named scope still works',len(named_receipt_scope(index,'What is the expensive item on the Trend Lounge receipt?')['documents'])==1)
check('raw OCR unchanged',index==original)
print(f'RESULT: {count}/{count} checks passed. No OCR, Groq, Ollama or network requests were made.')
