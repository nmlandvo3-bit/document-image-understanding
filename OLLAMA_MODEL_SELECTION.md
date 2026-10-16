# Select an Ollama model for both Q&A tabs

This update removes the hardcoded local model from the shared Q&A path and
batch/synthesis calls. It retains llama3.1:8b as the default. To select Qwen,
close the application and launch it from the same PowerShell terminal:

```powershell
$env:DOCUMENT_QA_OLLAMA_MODEL = "qwen3.8:27b"
$env:DOCUMENT_QA_OLLAMA_TIMEOUT_SEC = "300"
python -m experiments.experiment_109_ollama_live_smoke
python .\app.py
```

Select Ollama (local) in either Q&A tab. Enable model analysis in Corpus Q&A.
The corpus evidence panel displays Requested model: qwen3.8:27b. The selected
model is also retained in Document Q&A response metadata. Model selection in
Ollama's own chat window does not set this application configuration.

To switch back, close the application and relaunch:

```powershell
$env:DOCUMENT_QA_OLLAMA_MODEL = "llama3.1:8b"
python .\app.py
```

Environment settings apply to the terminal session and child processes.
The timeout defaults to 180 seconds and may be set between 5 and 1800.
Explicit local model arguments override the environment for Python callers.
Groq's model configuration is unchanged. No new Python dependencies are needed.

Run the offline regression with:

```powershell
python -m experiments.experiment_108_ollama_model_selection
```

Experiments 98–108 passed 200 offline checks. Experiment 109 is a LIVE request
to the user's local Ollama; it has not been run against the installed Qwen here.
Receipt checks, currency handling and OCR citation validation are retained.
Model output quality and runtime must be evaluated on actual scans. This update
supports optional DOCUMENT_QA_OLLAMA_THINK=true or false, using the Ollama
chat API's think field. Leave it unset (or set default) to preserve model defaults.
For a model that supports disabling thinking, false can avoid spending the output
budget on thinking during this extractive task. Check model support if Ollama
rejects that option. See https://docs.ollama.com/api/chat.
