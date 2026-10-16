# Receipt Q&A revision 107

Close the application before extracting this ZIP into your project root.
This bundle contains the latest complete Q&A service files and experiments
98–107, including previous Ollama updates. It does not replace app.py or OCR
results. Your existing Ollama-enabled app.py is required.

Run python -m experiments.experiment_107_screenshot_questions, then
python -m experiments.experiment_106_receipt_total_listing, and git diff --check.
Restart with python app.py. The evidence panel must show:
Q&A revision: receipt_qa_107

All/all of the receipts keep the full corpus; named merchants narrow scope.
Simple cheapest/most expensive/the expensive item questions use linked OCR
item rows. Complete local evidence can provide an explicitly labelled local
answer even when model extraction fails. Model failures remain in diagnostics.
Different or unspecified printed currencies produce grouped candidates, not
currency conversion. Symbol matches alone do not verify denomination.
Receipt-total lists contain one printed total per receipt candidate. Missing
or conflicting totals are reported. General questions still use model analysis.

176 offline checks passed with mock model responses. Campus, supermarket and
payment fixtures come from supplied OCR. Trend tables in comparison tests are
synthetic; tests do not certify actual Trend OCR or live Llama accuracy.
