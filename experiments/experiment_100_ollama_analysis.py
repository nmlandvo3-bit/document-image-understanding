"""Offline tests of local provider requests and OCR validation."""
import json
import sys
import tempfile
import types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
router=types.ModuleType('services.corpus_question_router')
router.answer_corpus_question=lambda *a: None
sys.modules['services.corpus_question_router']=router
from services.ollama_client import OllamaClient
from services.corpus_groq_assisted import answer_corpus_question_analysis
from services.document_groq_analysis import answer_selected_document
from services.corpus_qa_presentation import present_corpus_answer
checks=0
def check(label, ok):
 global checks
 assert ok, label
 checks+=1
 print('[PASS]',label)
doc={'document_id':'scan', 'evidence':[{'evidence_type':'ocr_region','region_id':4,'text':'Cashier: Sarah M.'}]}
requests=[]
def transport(body):
 requests.append(body)
 return {'done':True,'done_reason':'stop','message':{'content':json.dumps({'status':'answered','claims':[{'document_id':'scan','quote':'Sarah M.','evidence_item_ids':['e0']}]})}}
client=OllamaClient(transport=transport)
r=answer_corpus_question_analysis({'documents':[doc]},'Cashier?',client=client,provider='ollama')
check('local provider returns cited answer',r['status']=='answered' and r['provider']=='ollama_corpus_analysis')
check('installed model selected',requests[0]['model']=='llama3.1:8b')
check('JSON schema supplied',requests[0]['format']['required']==['status','claims'])
check('Groq options removed',not any(k in requests[0] for k in ['reasoning_effort','include_reasoning','response_format']))
check('context and temperature explicit',requests[0]['options']['num_ctx']==8192 and requests[0]['options']['temperature']==0)
check('original OCR IDs restored',r['evidence_item_ids']==['scan:ocr_region:4'])
check('local result shown by presenter','Sarah M.' in present_corpus_answer(r)[0])
with tempfile.TemporaryDirectory() as folder:
 p=Path(folder)/'scan.json';p.write_text(json.dumps(doc))
 r=answer_selected_document(p,'Who is cashier?',client=client,provider='ollama')
 check('document provider and model accurate',r['provider']=='ollama_selected_document_analysis' and r['model']=='llama3.1:8b')
 check('only selected document sent',len(json.loads(requests[-1]['messages'][1]['content'])['documents'])==1)
 check('question and evidence reach GUI',r['question']=='Who is cashier?' and r['evidence'][0]['content']['text']=='Cashier: Sarah M.')
def invented(body):
 data=transport(body);data['message']['content']=data['message']['content'].replace('Sarah M.','John');return data
r=answer_corpus_question_analysis({'documents':[doc]},'?',client=OllamaClient(transport=invented),provider='ollama')
check('invented local answer rejected',r['status']=='analysis_failed')
def truncated(body):
 data=transport(body);data['done_reason']='length';return data
r=answer_corpus_question_analysis({'documents':[doc]},'?',client=OllamaClient(transport=truncated),provider='ollama')
check('truncated output rejected',r['status']=='analysis_failed')
count=len(requests)
r=answer_corpus_question_analysis({'documents':[doc]},'?',client=OllamaClient(num_ctx=512,transport=transport),provider='ollama')
check('oversized context refused without request',r['status']=='local_context_limit' and len(requests)==count)
print(f'RESULT: {checks}/{checks} checks passed. No OCR, Groq, or network requests were made.')
