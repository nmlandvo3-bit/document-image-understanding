"""Offline actual OCR and synthetic table regressions; not a live-model benchmark."""
import copy
import json
import sys
import types
from decimal import Decimal
from pathlib import Path
router = types.ModuleType('services.corpus_question_router')
router.answer_corpus_question = lambda *a: None
sys.modules['services.corpus_question_router'] = router
from services.receipt_item_qa import receipt_item_context, item_comparison
from services.corpus_groq_assisted import answer_corpus_question_analysis as analyze
from services.corpus_qa_presentation import present_corpus_answer
from services.ollama_client import OllamaError

scans = json.loads((Path(__file__).parent/'fixtures/receipt_total_scans.json').read_text())
# Synthetic OCR table: its coordinates and IDs are NOT uploaded Trend OCR.
trend = {'document_id':'synthetic_trend_lounge','evidence':[]}
def add(rid,text,x,y,w=100):
    trend['evidence'].append({'evidence_type':'ocr_region','region_id':rid,'text':text,
        'bbox':{'x':x,'y':y,'width':w,'height':12}})
add(1,'THE TREND LOUNGE',10,0,200)
add(2,'Receipt #: TEST-1',10,20,180)
add(3,'Item',10,50); add(4,'Qty',220,50,25); add(5,'Price',300,50,60)
for row,(name,qty,price) in enumerate([('Slim Fit Denim Jeans','1','$59.99'),('Classic White Tee','2','$40.00'),('Leather Bomber Jacket','1','$129.99'),('Athletic Crew Socks','3','$18.00')]):
    y=80+row*25
    add(10+row*3,name,10,y,180); add(11+row*3,qty,220,y,20); add(12+row*3,price,300,y,60)
add(30,'Subtotal',10,190); add(31,'$247.98',300,190,60)
add(32,'Total',10,215); add(33,'$269.68',300,215,60)
index={'documents':scans+[trend]}; original=copy.deepcopy(index)
count=0
def check(label,condition):
    global count
    assert condition,label
    count+=1;print('[PASS]',label)
for d in scans:
    c=receipt_item_context(d)
    if 'PAYMENT INFORMATION' in [e['text'] for e in d['evidence']]:
        check('payment form is not an item receipt',not c['receipt_candidate'] and not c['items'])
    else:
        check('actual scan has five complete item pairs',c['coverage']=='candidate_rows_complete' and len(c['items'])==5)
c=receipt_item_context(trend)
check('table pairs names with price rather than quantity',len(c['items'])==4 and [i['numeric_price'] for i in c['items']]==['59.99','40.00','129.99','18.00'])
check('summary amounts excluded',all(i['price'] not in {'$247.98','$269.68'} for i in c['items']))
q='What is the most expensive item on the Trend Lounge receipt? Include its name and price.'
comparison=item_comparison(index,q)
check('named receipt scoped before comparison',comparison['receipt_count']==1 and comparison['items'][0]['name']=='Leather Bomber Jacket')
check('minimum selects socks',item_comparison(index,'What is the cheapest item on the Trend Lounge receipt?')['items'][0]['name']=='Athletic Crew Socks')
check('mixed or unspecified currencies have no global winner',item_comparison(index,'What is the most expensive item on the receipts?')['status']=='comparison_ambiguous')
check('unrelated questions keep general routing',item_comparison(index,'Who was the cashier?') is None)
check('multipart questions keep general routing',item_comparison(index,'What is the most expensive item and who was the cashier?') is None)

class Client:
    def __init__(self,mode='price_only',batched=False,fail_doc=None):
        self.mode,self.batched,self.fail_doc=mode,batched,fail_doc
        self.chat=types.SimpleNamespace(completions=self)
        self.packets=[]
    def create(self,**kwargs):
        p=json.loads(kwargs['messages'][1]['content']);self.packets.append(p)
        if self.batched and p['analysis_stage']=='answer':
            raise OllamaError('Force batching',413)
        docs=[d for d in p['documents'] if d['document_id']=='synthetic_trend_lounge']
        claims=[]
        for d in docs:
            for eid,text in d['ocr_regions']:
                if (self.mode=='correct' and text in {'Leather Bomber Jacket','$129.99'}) or (
                    self.mode=='price_only' and text in {'$59.99','$129.99','$18.00'}):
                    claims.append({'document_id':d['document_id'],'quote':text,'evidence_item_ids':[eid]})
        if self.fail_doc and any(d['document_id']==self.fail_doc for d in p['documents']):
            claims=[{'document_id':self.fail_doc,'quote':'invented','evidence_item_ids':['e0']}]
        payload={'status':'answered' if claims else 'not_found','claims':claims}
        return types.SimpleNamespace(choices=[types.SimpleNamespace(finish_reason='stop',message=types.SimpleNamespace(content=json.dumps(payload)))])
for provider in ('ollama','groq'):
    client=Client();r=analyze(index,q,provider=provider,client=client,max_context_bytes=50000)
    check(provider+' repairs incomplete price-only model answer',r['status']=='answered' and r['selection_method']=='local_item_price_comparison' and 'Leather Bomber Jacket' in r['answer'] and '$129.99' in r['answer'] and '$59.99' not in r['answer'])
    check(provider+' receives explicit item pairs',any(d['item_context']['items'] for d in client.packets[0]['documents']))
    check(provider+' repair has name and price provenance',len(r['claims'])==2 and len(r['evidence_item_ids'])==2)
    check(provider+' UI discloses local correction',present_corpus_answer(r)[0].startswith('LOCAL COMPARISON') and 'local_item_price_comparison' in present_corpus_answer(r)[1])
r=analyze(index,q,provider='ollama',client=Client('not_found'),max_context_bytes=50000)
check('not_found can fall back to paired evidence with disclosure',r['status']=='answered' and r['model_status']=='model_not_found' and r['selection_method']=='local_item_price_comparison')
r=analyze(index,q,provider='ollama',client=Client('correct'),max_context_bytes=50000)
check('correct model answer passes comparison checking',r['selection_method']=='model_selection_checked_against_item_rows')
r=analyze(index,'What is the most expensive item on the receipts?',provider='ollama',client=Client(batched=True),max_context_bytes=50000)
check('batched final selection checked against original corpus',r['status']=='comparison_ambiguous' and r['coverage_complete'] and 'Leather Bomber Jacket' in r['answer'])
r=analyze(index,q,provider='ollama',client=Client(fail_doc=scans[0]['document_id']),max_context_bytes=50000)
check('failure outside named receipt scope does not block answer',r['status']=='answered' and r['documents_in_scope']==1)
r=analyze(index,'What is the most expensive item on the receipts?',provider='ollama',client=Client(),max_context_bytes=50000)
check('currency ambiguity visible with group candidates',r['status']=='comparison_ambiguous' and present_corpus_answer(r)[0].startswith('CURRENCY AMBIGUITY'))
broken=copy.deepcopy(trend)
next(e for e in broken['evidence'] if e['text']=='$129.99')['bbox']['y']=400
r=analyze({'documents':[broken]},q,provider='ollama',client=Client(),max_context_bytes=50000)
check('unpaired name prevents a false maximum',r['status']=='comparison_incomplete' and not r['claims'])
tied=copy.deepcopy(trend)
next(e for e in tied['evidence'] if e['text']=='$59.99')['text']='$129.99'
check('ties retained',len(item_comparison({'documents':[tied]},q)['items'])==2)
check('OCR sources unchanged',index==original)
print(f'RESULT: {count}/{count} checks passed. No OCR, Groq, Ollama or network requests were made.')
