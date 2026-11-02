"""Cache-friendly routing and credential-free usage records for OpenRouter."""
import hashlib
import json
import math
from pathlib import Path
import time
from uuid import uuid4
import httpx


class OpenRouterTransport(httpx.BaseTransport):
    """Keep a routing session; never cache completions or replay simulator actions.

    Argon uses provider-managed implicit caching, not explicit cache_control.
    This wrapper records only usage/IDs/hashes, never keys, images or messages.
    """
    def __init__(self, inner=None, *, cache_friendly=True, read_timeout_s=300.):
        if not math.isfinite(read_timeout_s) or read_timeout_s <= 0:
            raise ValueError('Request read timeout must be positive and finite')
        self.inner = inner if inner is not None else httpx.HTTPTransport()
        self.session_id = 'robocurve-' + uuid4().hex if cache_friendly else None
        self.read_timeout_s = float(read_timeout_s)
        self.output = None
        self.records = []
        self._message_hashes = []

    def handle_request(self, request):
        if request.url.host != 'openrouter.ai':
            raise ValueError('OpenRouter transport only accepts openrouter.ai requests')
        body = json.loads(request.content)
        effort = body.pop('reasoning_effort', None)
        if effort is not None:
            body['reasoning'] = {'effort': effort}
        headers = request.headers.copy()
        headers.pop('content-length', None)
        if self.session_id:
            headers['x-session-id'] = self.session_id
        extensions = dict(request.extensions)
        extensions['timeout'] = {**extensions.get('timeout', {}), 'read': self.read_timeout_s}
        outgoing = httpx.Request(request.method, request.url, headers=headers,
                                 content=json.dumps(body).encode(), extensions=extensions)
        hashes = [hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()
                  for m in body.get('messages', [])]
        previous = self._message_hashes
        record = {'attempt': len(self.records)+1, 'requested_model': body.get('model'),
                  'reasoning_effort': body.get('reasoning', {}).get('effort'),
                  'session_id': self.session_id, 'message_count': len(hashes),
                  'conversation_prefix_preserved': None if not previous else hashes[:len(previous)] == previous,
                  'cached_tokens': None, 'cache_write_tokens': None}
        self._message_hashes = hashes
        started = time.monotonic()
        try:
            response = self.inner.handle_request(outgoing)
            response.read()
        except httpx.TransportError as exc:
            record.update(error_type=type(exc).__name__, elapsed_s=time.monotonic()-started)
            self._record(record)
            raise
        record.update(status_code=response.status_code, elapsed_s=time.monotonic()-started)
        try:
            payload = response.json()
        except (ValueError, UnicodeDecodeError):
            payload = {}
        if isinstance(payload, dict):
            usage = payload.get('usage') or {}
            if isinstance(usage, dict):
                for key in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'cost'):
                    value = usage.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                        record[key] = value
                details = usage.get('prompt_tokens_details') or {}
                if isinstance(details, dict):
                    for key in ('cached_tokens', 'cache_write_tokens'):
                        value = details.get(key)
                        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                            record[key] = value
                completion = usage.get('completion_tokens_details') or {}
                if isinstance(completion, dict) and isinstance(completion.get('reasoning_tokens'), int):
                    record['reasoning_tokens'] = completion['reasoning_tokens']
            for source, target in (('id','generation_id'),('model','returned_model'),('provider','provider')):
                if isinstance(payload.get(source), str): record[target] = payload[source]
        self._record(record)
        return response

    def _record(self, record):
        self.records.append(record)
        if self.output is not None:
            directory = Path(self.output)
            with (directory/'openrouter_requests.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False)+'\n')
            summary = {'strategy': 'gemini_implicit', 'session_id': self.session_id,
                       'http_attempts': len(self.records),
                       'successful_responses': sum(r.get('status_code') == 200 for r in self.records),
                       'cache_usage_reported': any(r['cached_tokens'] is not None for r in self.records),
                       'cache_hit_confirmed': any((r['cached_tokens'] or 0)>0 for r in self.records),
                       'cached_tokens': sum(r['cached_tokens'] or 0 for r in self.records),
                       'reported_cost': sum(r.get('cost',0) for r in self.records)}
            temporary = directory/'openrouter_usage.partial.json'
            temporary.write_text(json.dumps(summary, indent=2)+'\n')
            temporary.replace(directory/'openrouter_usage.json')
        if record.get('status_code') == 200:
            cached = record['cached_tokens'] if record['cached_tokens'] is not None else 'unreported'
            print(f"[openrouter] cached_tokens={cached}; reasoning_effort={record['reasoning_effort']}", flush=True)

    def close(self):
        self.inner.close()
