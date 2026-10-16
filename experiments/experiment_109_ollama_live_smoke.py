"""LIVE check: one local Ollama request with a synthetic cashier OCR region.

Run explicitly after setting DOCUMENT_QA_OLLAMA_MODEL. No OCR/Groq requests.
"""
import time
from services.ollama_client import resolve_ollama_model
from services.corpus_groq_assisted import answer_corpus_question_analysis


def main():
    model=resolve_ollama_model()
    print('LIVE local Ollama request using:',model,flush=True)
    index={'documents':[{'document_id':'model_selection_smoke','evidence':[
        {'evidence_type':'ocr_region','region_id':1,'text':'Cashier: Sarah M.'}]}]}
    start=time.perf_counter()
    result=answer_corpus_question_analysis(index,'What is the cashier name?',provider='ollama')
    print('Requested model:',result.get('model'))
    print('Status:',result['status'])
    print('Answer:',result.get('answer'))
    print('Elapsed seconds:',round(time.perf_counter()-start,2))
    if result.get('failure_reason'):
        print('Failure:',result['failure_reason'])
    if result['status']!='answered' or not any('Sarah M.' in c['quote'] for c in result.get('claims',[])):
        raise SystemExit('FAIL: no accepted cashier answer. Check Ollama/model output before the GUI.')
    print('PASS: selected local model returned an OCR-cited cashier answer.')


if __name__=='__main__':
    main()
