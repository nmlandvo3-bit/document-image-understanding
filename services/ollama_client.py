"""Loopback-only Ollama adapter for the existing evidence analysis contract."""
from __future__ import annotations
import json
import os
import re
from types import SimpleNamespace
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler
from urllib.error import HTTPError, URLError

DEFAULT_OLLAMA_MODEL = "llama3.1:8b"


def resolve_ollama_model(model=None):
    selected = model if model is not None else os.getenv("DOCUMENT_QA_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    if not isinstance(selected, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,254}", selected.strip()):
        raise ValueError("Set DOCUMENT_QA_OLLAMA_MODEL to an installed Ollama model tag.")
    return selected.strip()


class OllamaError(RuntimeError):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise OllamaError('Ollama redirected a local request; request rejected.')

SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'status': {'type': 'string', 'enum': ['answered', 'not_found', 'ambiguous']},
        'claims': {'type': 'array', 'maxItems': 30, 'items': {
            'type': 'object', 'additionalProperties': False,
            'properties': {'document_id': {'type': 'string'}, 'quote': {'type': 'string'},
                'evidence_item_ids': {'type': 'array', 'minItems': 1, 'items': {'type': 'string'}}},
            'required': ['document_id', 'quote', 'evidence_item_ids']}},
    }, 'required': ['status', 'claims'],
}

class OllamaClient:
    def __init__(self, num_ctx=None, timeout=None, transport=None):
        if timeout is None:
            timeout = float(os.getenv("DOCUMENT_QA_OLLAMA_TIMEOUT_SEC", "180"))
        if not 5 <= timeout <= 1800:
            raise ValueError("DOCUMENT_QA_OLLAMA_TIMEOUT_SEC must be between 5 and 1800")
        thinking = os.getenv("DOCUMENT_QA_OLLAMA_THINK", "default").strip().lower()
        if thinking not in {"default", "true", "false"}:
            raise ValueError("DOCUMENT_QA_OLLAMA_THINK must be default, true or false")
        self.think = None if thinking == "default" else thinking == "true"
        configured = os.getenv("DOCUMENT_QA_OLLAMA_NUM_CTX")
        self.adaptive_context = num_ctx is None and configured is None
        if num_ctx is None:
            num_ctx = int(configured or "8192")
        if not 512 <= num_ctx <= 131072:
            raise ValueError("DOCUMENT_QA_OLLAMA_NUM_CTX must be between 512 and 131072")
        self.num_ctx, self.timeout = num_ctx, timeout
        self.transport = transport or self._post
        self.chat = SimpleNamespace(completions=self)

    def _post(self, payload):
        request = Request('http://127.0.0.1:11434/api/chat',
            data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
            headers={'Content-Type': 'application/json'}, method='POST')
        # No environment proxies or redirects: OCR stays on loopback.
        opener = build_opener(ProxyHandler({}), NoRedirect())
        try:
            with opener.open(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code == 404:
                raise OllamaError(f"Local model missing: {payload.get('model', 'unknown')}. Check ollama list.", 404) from error
            raise OllamaError('Ollama HTTP request failed.', error.code) from error
        except (URLError, TimeoutError) as error:
            raise OllamaError('Ollama unavailable or timed out. Check the Ollama application.', 503) from error

    def create(self, *, model, messages, max_completion_tokens=1000, **kwargs):
        # UTF-8 bytes provide a conservative token upper bound for Llama's
        # byte-level tokenizer. Reserve room for template and generated output.
        input_bound = sum(len(m['content'].encode('utf-8')) for m in messages) + 512
        request_ctx = self.num_ctx
        if self.adaptive_context:
            # Grow for this request only. Explicit user configuration stays fixed.
            needed = input_bound + max_completion_tokens
            while request_ctx < needed and request_ctx < 32768:
                request_ctx *= 2
        if input_bound + max_completion_tokens > request_ctx:
            raise OllamaError('Input exceeds the conservative local context budget.', 413)
        payload = {'model': model, 'messages': messages, 'stream': False,
            'format': SCHEMA, 'options': {'temperature': 0, 'num_ctx': request_ctx,
                'num_predict': max_completion_tokens}, 'keep_alive': '5m'}
        if self.think is not None:
            payload["think"] = self.think
        result = self.transport(payload)
        if result.get('done') is not True:
            raise OllamaError('Incomplete Ollama response.')
        content = result.get('message', {}).get('content')
        if not isinstance(content, str):
            raise OllamaError('Ollama response has no text.')
        finish = 'length' if result.get('done_reason') == 'length' else 'stop'
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish,
            message=SimpleNamespace(content=content))])
