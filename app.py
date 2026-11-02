#!/usr/bin/env python3
"""
Live 10 Fastest Free LLM 3D Evaluator v4
===================================

A localhost application that:
  * accepts one user question or prompt
  * asks OpenRouter for models ordered by recent median latency
  * runs one safe preflight, then the remaining fastest usable models in parallel
  * caps replacement attempts to prevent quota-draining retry storms
  * optionally enables OpenRouter web search and citation annotations
  * compares all 45 answer pairs mathematically
  * estimates answer quality, correctness/grounding, agreement, confidence, and speed
  * identifies the best response without using another judge LLM
  * displays the models as a complete interactive 3D graph
  * includes a fully local no-API sample mode

Important limitation:
The winner is the strongest answer under the transparent scoring formula. When
web citations are available, correctness is estimated against retrieved evidence.
Without citations, correctness is a cross-model consensus estimate, not proof of truth.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import secrets
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

APP_DIR = Path(__file__).resolve().parent
RUNS_DIR = APP_DIR / "runs"
ENV_FILE = APP_DIR / ".env"
HOST = "127.0.0.1"
DEFAULT_PORT = 8020
APP_BUILD = "v4-free-no-web-2026-07-23"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
ACTIVE_MODEL_COUNT = 10
MAX_WORKERS = 10
REQUEST_TIMEOUT_SECONDS = 180
MAX_TASK_CHARS = 6000
MAX_REPLACEMENT_ATTEMPTS = 3

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "to", "of", "in", "on", "for",
    "with", "without", "at", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "this", "that", "these", "those", "they", "their", "them", "he", "she", "his", "her",
    "we", "our", "you", "your", "i", "me", "my", "do", "does", "did", "doing", "have", "has", "had",
    "can", "could", "would", "should", "may", "might", "must", "will", "shall", "not", "no", "yes",
    "about", "into", "through", "during", "before", "after", "above", "below", "between", "because", "so",
    "very", "more", "most", "less", "least", "also", "such", "any", "all", "some", "each", "other",
}

PREFERRED_FAMILIES = [
    "qwen", "llama", "deepseek", "mistral", "gemma", "phi", "nemotron", "glm", "command", "hermes",
    "trinity", "molmo", "yi", "dolphin", "nous", "openchat", "zephyr", "solar", "granite",
]

# These are candidates only. The live OpenRouter catalog is preferred.
FALLBACK_FREE_MODEL_IDS = [
    "qwen/qwen3-4b:free",
    "qwen/qwen3-8b:free",
    "meta-llama/llama-3.3-8b-instruct:free",
    "google/gemma-3-12b-it:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "deepseek/deepseek-r1-0528-qwen3-8b:free",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
    "microsoft/phi-4-reasoning-plus:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "openrouter/free",
]

# Transparent, editable domain-quality heuristics. These are project rules,
# not universal truth labels.
HIGH_AUTHORITY_DOMAINS = {
    "sec.gov", "federalreserve.gov", "bls.gov", "census.gov", "nih.gov", "cdc.gov", "who.int",
    "worldbank.org", "imf.org", "oecd.org", "un.org", "europa.eu", "data.gov", "nasa.gov",
    "nature.com", "science.org", "nejm.org", "thelancet.com", "arxiv.org", "pubmed.ncbi.nlm.nih.gov",
}
REPUTABLE_NEWS_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "ft.com", "wsj.com", "bloomberg.com",
    "nytimes.com", "washingtonpost.com", "economist.com", "npr.org", "cnbc.com", "marketwatch.com",
}
LOW_AUTHORITY_PATTERNS = {"reddit.com", "quora.com", "medium.com", "blogspot.com", "wordpress.com", "facebook.com", "x.com"}

JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


class OpenRouterAPIError(RuntimeError):
    """HTTP error with status and headers preserved for safe retry decisions."""

    def __init__(self, status: int, detail: Any, headers: Optional[Dict[str, str]] = None):
        self.status = int(status)
        self.detail = detail
        self.headers = headers or {}
        super().__init__(f"HTTP {self.status}: {self.detail}")

    @property
    def message(self) -> str:
        if isinstance(self.detail, dict):
            return str((self.detail.get("error") or {}).get("message") or self.detail)
        return str(self.detail)

    def is_free_daily_quota(self) -> bool:
        text = (self.message + " " + str(self.detail)).lower()
        return self.status == 429 and (
            "free-models-per-day" in text or
            "x-ratelimit-remaining': '0" in text or
            'x-ratelimit-remaining": "0' in text
        )

    def reset_utc_text(self) -> str:
        reset = self.headers.get("X-RateLimit-Reset") or self.headers.get("x-ratelimit-reset")
        if not reset and isinstance(self.detail, dict):
            meta = (self.detail.get("error") or {}).get("metadata") or {}
            hdrs = meta.get("headers") or {}
            reset = hdrs.get("X-RateLimit-Reset") or hdrs.get("x-ratelimit-reset")
        try:
            value = int(str(reset))
            if value > 10_000_000_000:
                value //= 1000
            return datetime.fromtimestamp(value, tz=__import__("datetime").timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            return "the reset time shown by OpenRouter"


BLOCKED_MODEL_TERMS = (
    "content safety", "content-safety", "content_safety", "moderation",
    "rerank", "reranker", "embedding", "embed model", "embed-vl",
    "image generation", "text-to-image", "seedream", "music generation", "lyria",
    "text-to-speech", "speech-to-text", "tts", "audio generation",
)


@dataclass
class WebSource:
    url: str
    canonical_url: str
    domain: str
    title: str
    content: str
    start_index: Optional[int] = None
    end_index: Optional[int] = None


@dataclass
class ModelAnswer:
    model_id: str
    model_name: str
    provider: str
    answer: str
    key_points: List[str]
    claims: List[str]
    confidence: float
    sources: List[WebSource]
    latency_seconds: float
    web_search_requests: int = 0
    catalog_speed_rank: int = 0
    raw_response: str = ""
    error: str = ""


# -----------------------------------------------------------------------------
# Environment and HTTP
# -----------------------------------------------------------------------------

def load_env_file() -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not ENV_FILE.exists():
        return values
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def save_api_key(api_key: str) -> None:
    ENV_FILE.write_text(f"OPENROUTER_API_KEY={api_key}\n", encoding="utf-8")


def current_api_key() -> str:
    return os.environ.get("OPENROUTER_API_KEY", "") or load_env_file().get("OPENROUTER_API_KEY", "")


def request_json(url: str, method: str = "GET", headers: Optional[Dict[str, str]] = None,
                 payload: Optional[Dict[str, Any]] = None, timeout: int = REQUEST_TIMEOUT_SECONDS) -> Dict[str, Any]:
    data = None
    request_headers = {"User-Agent": "Live-10-Fastest-LLM-3D-Evaluator/6.0"}
    if headers:
        request_headers.update(headers)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(raw)
        except Exception:
            detail = raw
        raise OpenRouterAPIError(exc.code, detail, dict(exc.headers.items())) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach OpenRouter: {exc.reason}") from exc


# -----------------------------------------------------------------------------
# Text math
# -----------------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?%?", str(text).lower())
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def cosine_counter(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b.get(k, 0.0) for k in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def lexical_cosine(a: str, b: str) -> float:
    return cosine_counter(Counter(tokenize(a)), Counter(tokenize(b)))


def build_tfidf_vectors(documents: Sequence[str]) -> List[Counter]:
    tokenized = [tokenize(d) for d in documents]
    n = len(documents)
    df = Counter()
    for tokens in tokenized:
        df.update(set(tokens))
    vectors: List[Counter] = []
    for tokens in tokenized:
        counts = Counter(tokens)
        total = max(1, sum(counts.values()))
        vec = Counter()
        for token, count in counts.items():
            tf = count / total
            idf = math.log((1 + n) / (1 + df[token])) + 1
            vec[token] = tf * idf
        vectors.append(vec)
    return vectors


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", clean_text(text))
    return [p.strip() for p in parts if len(p.strip()) >= 20]


def extract_numbers(text: str) -> List[str]:
    return re.findall(r"(?<![A-Za-z])[-+]?\$?\d[\d,]*(?:\.\d+)?%?", str(text))


def normalize_number(token: str) -> str:
    return token.replace("$", "").replace(",", "").strip().lower()


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def canonicalize_url(url: str) -> str:
    try:
        p = urllib.parse.urlparse(url.strip())
        host = (p.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        path = re.sub(r"/+", "/", p.path or "/").rstrip("/") or "/"
        return urllib.parse.urlunparse((p.scheme.lower() or "https", host, path, "", "", ""))
    except Exception:
        return url.strip()


def domain_from_url(url: str) -> str:
    try:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def domain_authority(domain: str) -> float:
    d = domain.lower().strip()
    if not d:
        return 0.0
    if d in HIGH_AUTHORITY_DOMAINS or d.endswith(".gov"):
        return 1.0
    if d.endswith(".edu") or ".ac." in d:
        return 0.95
    if d in REPUTABLE_NEWS_DOMAINS:
        return 0.84
    if any(d == x or d.endswith("." + x) for x in LOW_AUTHORITY_PATTERNS):
        return 0.35
    if d.endswith(".org"):
        return 0.72
    if d.endswith(".gov.uk") or d.endswith(".gc.ca"):
        return 0.95
    # Official corporate pages and other established domains are unknown to a
    # static heuristic, so they receive a neutral score instead of a penalty.
    return 0.60


def best_evidence_support(claim: str, sources: Sequence[WebSource]) -> Tuple[float, Optional[WebSource], float, float]:
    """Return combined support, best source, lexical support, numeric support."""
    if not claim or not sources:
        return 0.0, None, 0.0, 0.0
    claim_numbers = {normalize_number(n) for n in extract_numbers(claim)}
    best = (0.0, None, 0.0, 0.0)
    for source in sources:
        evidence = f"{source.title} {source.content}"
        lexical = lexical_cosine(claim, evidence)
        if claim_numbers:
            evidence_numbers = {normalize_number(n) for n in extract_numbers(evidence)}
            numeric = len(claim_numbers & evidence_numbers) / len(claim_numbers)
        else:
            numeric = 1.0
        # A source cannot strongly support a numerical claim if the number is absent.
        support = 0.72 * lexical + 0.28 * numeric
        if support > best[0]:
            best = (support, source, lexical, numeric)
    return best


# -----------------------------------------------------------------------------
# Model discovery and OpenRouter calls
# -----------------------------------------------------------------------------

def number_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def is_free_model(model: Dict[str, Any]) -> bool:
    model_id = str(model.get("id", ""))
    pricing = model.get("pricing") or {}
    return model_id.endswith(":free") or (
        number_value(pricing.get("prompt")) == 0.0 and number_value(pricing.get("completion")) == 0.0
    )


def supports_text(model: Dict[str, Any]) -> bool:
    outputs = (model.get("architecture") or {}).get("output_modalities") or []
    return not outputs or "text" in outputs


def supports_tools(model: Dict[str, Any]) -> bool:
    return "tools" in (model.get("supported_parameters") or [])


def is_general_chat_model(model: Dict[str, Any]) -> bool:
    """Accept any free model that can generate text, excluding utility/safety endpoints.

    Multimodal and coding-capable models are allowed because the application sends
    text-only prompts and compares their text answers. This intentionally avoids the
    older overly strict family-name filter that reduced the catalog to seven models.
    """
    mid = str(model.get("id") or "").lower()
    name = str(model.get("name") or "").lower()
    description = str(model.get("description") or "").lower()
    text = f" {mid} {name} {description} "
    if any(term in text for term in BLOCKED_MODEL_TERMS):
        return False
    arch = model.get("architecture") or {}
    outputs = arch.get("output_modalities") or []
    return not outputs or "text" in outputs


def model_rank(model: Dict[str, Any]) -> Tuple[int, int, int, str]:
    mid = str(model.get("id", "")).lower()
    family = len(PREFERRED_FAMILIES)
    for i, name in enumerate(PREFERRED_FAMILIES):
        if name in mid:
            family = i
            break
    tool_penalty = 0 if supports_tools(model) else 1
    context = int(model.get("context_length") or 0)
    return tool_penalty, family, -context, mid


def discover_fastest_models(api_key: str, manual_ids: Sequence[str], free_only: bool = True) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return free text-generating models in OpenRouter latency order.

    The application does not request tools or web search. It accepts text-capable
    multimodal and coding models, while excluding safety, embedding, reranking, and
    media-generation utility endpoints. If fewer than ten explicit free model IDs are
    visible, OpenRouter's free router fills the missing slots and reports the actual
    selected model in each response.
    """
    if manual_ids:
        models = [
            {"id": m, "name": m, "context_length": 0, "supported_parameters": [],
             "_catalog_speed_rank": i + 1, "_slot_key": f"manual-{i+1}"}
            for i, m in enumerate(manual_ids)
        ]
        return models[:ACTIVE_MODEL_COUNT], models[ACTIVE_MODEL_COUNT:]

    params = {
        "output_modalities": "text",
        "sort": "latency-low-to-high",
        "limit": "1000",
    }
    url = OPENROUTER_MODELS_URL + "?" + urllib.parse.urlencode(params)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    catalog = request_json(url, headers=headers, timeout=60)
    raw_models = catalog.get("data") or []

    models: List[Dict[str, Any]] = []
    seen = set()
    for catalog_rank, model in enumerate(raw_models, 1):
        model_id = str(model.get("id") or "").strip()
        if not model_id or model_id in seen or model_id in {"openrouter/auto", "openrouter/free"}:
            continue
        if not supports_text(model) or not is_general_chat_model(model):
            continue
        if free_only and not is_free_model(model):
            continue
        item = dict(model)
        item["_catalog_speed_rank"] = catalog_rank
        item["_slot_key"] = model_id
        models.append(item)
        seen.add(model_id)

    # Fill a rare catalog shortfall with independent free-router slots. OpenRouter
    # returns the actual routed model in the completion response.
    missing = max(0, ACTIVE_MODEL_COUNT - len(models))
    for i in range(missing):
        models.append({
            "id": "openrouter/free",
            "name": f"OpenRouter Free Router Slot {i+1}",
            "context_length": 0,
            "supported_parameters": [],
            "_catalog_speed_rank": len(raw_models) + i + 1,
            "_slot_key": f"free-router-slot-{i+1}",
            "_router_slot": i + 1,
        })

    if len(models) < 2:
        raise RuntimeError(
            "OpenRouter returned fewer than two free text-generating candidates. "
            "This is usually a temporary availability or account-routing issue."
        )
    return models[:ACTIVE_MODEL_COUNT], models[ACTIVE_MODEL_COUNT:80]

def build_messages(task: str) -> List[Dict[str, str]]:
    system = (
        "You are one member of a ten-model evaluation panel. Answer only from your "
        "own knowledge and reasoning. Do not browse the web, call tools, or invent "
        "citations. Return one valid JSON object only, without markdown fences. "
        "The application compares your response mathematically with the other models."
    )
    user = f"""USER QUESTION, PROMPT, TOPIC, OR TEXT
{task}

RETURN EXACTLY THIS JSON SHAPE
{{
  "answer": "A direct, clear, useful answer to the user's input.",
  "key_points": ["main point one", "main point two", "main point three"],
  "claims": ["one short checkable claim", "another short checkable claim"],
  "confidence": 0.0
}}

RULES
1. Answer the exact user input.
2. confidence must be from 0 to 1.
3. claims must be short and individually checkable.
4. Clearly state uncertainty when you are not sure.
5. Do not include URLs, citations, or pretend that you searched the web.
6. Prefer concise but complete reasoning.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def extract_message(response: Dict[str, Any]) -> Dict[str, Any]:
    choices = response.get("choices") or []
    return (choices[0].get("message") or {}) if choices else {}


def extract_content(message: Dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                out.append(str(item.get("text", "")))
            elif isinstance(item, str):
                out.append(item)
        return "\n".join(out)
    return str(content or "")


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(cleaned[start:i + 1])
                        return obj if isinstance(obj, dict) else None
                    except Exception:
                        return None
    return None


def parse_sources(message: Dict[str, Any]) -> List[WebSource]:
    sources: List[WebSource] = []
    seen = set()
    for ann in message.get("annotations") or []:
        if not isinstance(ann, dict) or ann.get("type") != "url_citation":
            continue
        uc = ann.get("url_citation") or {}
        url = str(uc.get("url") or "").strip()
        if not url:
            continue
        canonical = canonicalize_url(url)
        if canonical in seen:
            continue
        seen.add(canonical)
        sources.append(WebSource(
            url=url,
            canonical_url=canonical,
            domain=domain_from_url(url),
            title=str(uc.get("title") or url).strip(),
            content=clean_text(str(uc.get("content") or "")),
            start_index=uc.get("start_index") if isinstance(uc.get("start_index"), int) else None,
            end_index=uc.get("end_index") if isinstance(uc.get("end_index"), int) else None,
        ))
    return sources


def normalize_model_answer(model: Dict[str, Any], response: Dict[str, Any], latency: float, expected_sources: bool = False) -> ModelAnswer:
    message = extract_message(response)
    raw_content = extract_content(message)
    parsed = extract_json_object(raw_content)
    requested_model_id = str(model.get("id", "unknown"))
    actual_model_id = str(response.get("model") or requested_model_id)
    router_slot = int(model.get("_router_slot") or 0)
    model_id = f"{actual_model_id}#slot-{router_slot}" if router_slot else actual_model_id
    model_name = actual_model_id + (f" (free router slot {router_slot})" if router_slot else "")
    if not router_slot:
        model_name = str(model.get("name") or actual_model_id)
    provider = actual_model_id.split("/", 1)[0] if "/" in actual_model_id else actual_model_id
    sources: List[WebSource] = []

    if parsed:
        answer = clean_text(str(parsed.get("answer") or raw_content))
        kp = parsed.get("key_points") or []
        claims_raw = parsed.get("claims") or []
        key_points = [clean_text(str(x)) for x in kp if clean_text(str(x))] if isinstance(kp, list) else []
        claims = [clean_text(str(x)) for x in claims_raw if clean_text(str(x))] if isinstance(claims_raw, list) else []
        try:
            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
        except Exception:
            confidence = 0.5
    else:
        answer = clean_text(raw_content) or "No usable answer."
        key_points = split_sentences(answer)[:4]
        claims = split_sentences(answer)[:8]
        confidence = 0.5

    if not claims:
        claims = split_sentences(answer)[:8]
    usage = response.get("usage") or {}
    tool_use = usage.get("server_tool_use") or {}
    searches = int(tool_use.get("web_search_requests") or 0)
    error = "" if parsed else "Response was not valid JSON; fallback sentence parsing was used."
    if expected_sources and not sources:
        error = (error + " No OpenRouter URL citation annotations were returned.").strip()

    return ModelAnswer(
        model_id=model_id, model_name=model_name, provider=provider, answer=answer,
        key_points=key_points, claims=claims, confidence=confidence, sources=sources,
        latency_seconds=latency, web_search_requests=searches, catalog_speed_rank=int(model.get("_catalog_speed_rank") or 0), raw_response=raw_content, error=error,
    )


def call_model(api_key: str, model: Dict[str, Any], task: str) -> ModelAnswer:
    payload: Dict[str, Any] = {
        "model": model.get("id"),
        "messages": build_messages(task),
        "provider": {"sort": "latency"},
        "temperature": 0.2,
        "max_tokens": 1500,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": f"http://{HOST}:{DEFAULT_PORT}",
        "X-OpenRouter-Title": "Live 10 Fastest Free LLM 3D Evaluator v4",
    }
    started = time.perf_counter()
    try:
        response = request_json(OPENROUTER_CHAT_URL, method="POST", headers=headers, payload=payload)
        return normalize_model_answer(model, response, time.perf_counter() - started, expected_sources=False)
    except Exception as exc:
        if isinstance(exc, OpenRouterAPIError) and exc.is_free_daily_quota():
            error = f"FREE_DAILY_QUOTA: {exc}; reset {exc.reset_utc_text()}"
        else:
            error = str(exc)
        mid = str(model.get("id", "unknown"))
        slot = int(model.get("_router_slot") or 0)
        internal_id = f"{mid}#slot-{slot}" if slot else mid
        name = str(model.get("name") or mid)
        return ModelAnswer(
            internal_id, name, mid.split("/", 1)[0], "", [], [], 0.0, [],
            time.perf_counter() - started,
            catalog_speed_rank=int(model.get("_catalog_speed_rank") or 0),
            error=error,
        )


# -----------------------------------------------------------------------------
# Demo data
# -----------------------------------------------------------------------------

def demo_answers(task: str) -> List[ModelAnswer]:
    patterns = [
        ("Bigfoot is a legendary ape-like creature reported mainly in North American forests. Eyewitness stories and footprint claims are common, but no verified body, DNA sample, or specimen has established its existence.", .91),
        ("Bigfoot, also called Sasquatch, is part of North American folklore. The evidence is mostly testimony, photographs, videos, and alleged tracks, none of which has produced scientific confirmation.", .87),
        ("The best-supported conclusion is that Bigfoot remains unproven. Reports may involve bears, hoaxes, misidentification, and cultural storytelling.", .84),
        ("Bigfoot is described as a large, hairy, humanlike animal. Many people report sightings, although reproducible physical evidence is lacking.", .82),
        ("There is no accepted scientific proof of Bigfoot, but the legend remains influential in folklore, tourism, and popular culture.", .80),
        ("Bigfoot definitely exists because thousands of people have seen it.", .93),
        ("Bigfoot claims are interesting but inconclusive. Eyewitness quantity alone cannot replace independently testable evidence.", .78),
        ("Sasquatch reports deserve investigation, yet current evidence does not justify a confident biological conclusion.", .75),
        ("Bigfoot was scientifically discovered in 1967 and is now classified as a primate species.", .88),
        ("A careful answer separates the cultural legend from the scientific question: the legend is real, while the animal's existence has not been demonstrated.", .90),
    ]
    out = []
    for i, (answer, conf) in enumerate(patterns, 1):
        out.append(ModelAnswer(
            model_id=f"demo/model-{i}", model_name=f"Demo LLM {i:02d}", provider="demo",
            answer=answer, key_points=split_sentences(answer), claims=split_sentences(answer),
            confidence=conf, sources=[], latency_seconds=.25 + i*.04,
            web_search_requests=0, catalog_speed_rank=i, raw_response=answer,
        ))
    return out


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------

def source_pool(answers: Sequence[ModelAnswer]) -> List[WebSource]:
    by_url: Dict[str, WebSource] = {}
    for answer in answers:
        for source in answer.sources:
            old = by_url.get(source.canonical_url)
            if old is None or len(source.content) > len(old.content):
                by_url[source.canonical_url] = source
    return list(by_url.values())


def pairwise_scores(answers: Sequence[ModelAnswer]) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
    answer_vecs = build_tfidf_vectors([a.answer for a in answers])
    key_vecs = build_tfidf_vectors([" ".join(a.key_points) for a in answers])
    claim_vecs = build_tfidf_vectors([" ".join(a.claims) for a in answers])
    evidence_vecs = build_tfidf_vectors([" ".join(s.content for s in a.sources) for a in answers])
    n = len(answers)
    matrix = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            text_sim = cosine_counter(answer_vecs[i], answer_vecs[j])
            key_sim = cosine_counter(key_vecs[i], key_vecs[j])
            claim_sim = cosine_counter(claim_vecs[i], claim_vecs[j])
            urls_a = [s.canonical_url for s in answers[i].sources]
            urls_b = [s.canonical_url for s in answers[j].sources]
            domains_a = [s.domain for s in answers[i].sources]
            domains_b = [s.domain for s in answers[j].sources]
            url_overlap = jaccard(urls_a, urls_b)
            domain_overlap = jaccard(domains_a, domains_b)
            source_overlap = 0.70 * url_overlap + 0.30 * domain_overlap
            evidence_sim = cosine_counter(evidence_vecs[i], evidence_vecs[j])
            if not answers[i].sources and not answers[j].sources:
                # In no-source mode, renormalize the answer-only components so
                # identical answers can still reach a pair score of 1.
                combined = (0.35*text_sim + 0.20*key_sim + 0.20*claim_sim) / 0.75
            else:
                combined = 0.35*text_sim + 0.20*key_sim + 0.20*claim_sim + 0.15*source_overlap + 0.10*evidence_sim
            combined = max(0.0, min(1.0, combined))
            matrix[i][j] = matrix[j][i] = combined
            rows.append({
                "model_a": answers[i].model_id, "name_a": answers[i].model_name,
                "model_b": answers[j].model_id, "name_b": answers[j].model_name,
                "text_similarity": text_sim, "key_point_similarity": key_sim,
                "claim_similarity": claim_sim, "url_overlap": url_overlap,
                "domain_overlap": domain_overlap, "source_overlap": source_overlap,
                "evidence_similarity": evidence_sim, "combined_similarity": combined,
                "sources_a": urls_a, "sources_b": urls_b,
            })
    return rows, matrix


def clarity_score(answer: str) -> float:
    words = answer.split()
    if not words:
        return 0.0
    length_score = min(1.0, len(words) / 110)
    sentence_count = max(1, len(split_sentences(answer)))
    avg = len(words) / sentence_count
    readability = 1.0 if 8 <= avg <= 28 else max(0.35, 1.0 - abs(avg - 18)/35)
    return 0.45*length_score + 0.55*readability


def completeness_score(answer: ModelAnswer) -> float:
    return min(1.0, 0.50 * min(len(answer.key_points) / 3, 1) + 0.50 * min(len(answer.claims) / 4, 1))


def cross_model_claim_support(index: int, answers: Sequence[ModelAnswer]) -> float:
    own_claims = answers[index].claims or split_sentences(answers[index].answer)[:8]
    other_claims = [
        claim for j, answer in enumerate(answers) if j != index
        for claim in (answer.claims or split_sentences(answer.answer)[:8])
    ]
    if not own_claims or not other_claims:
        return 0.0
    scores = [max(lexical_cosine(claim, other) for other in other_claims) for claim in own_claims]
    return sum(scores) / len(scores)


def cross_model_numeric_support(index: int, answers: Sequence[ModelAnswer]) -> float:
    own_numbers = {normalize_number(n) for n in extract_numbers(answers[index].answer)}
    if not own_numbers:
        return 1.0
    counts = Counter()
    for j, answer in enumerate(answers):
        if j == index:
            continue
        counts.update({normalize_number(n) for n in extract_numbers(answer.answer)})
    return sum(1 for n in own_numbers if counts[n] >= 1) / len(own_numbers)


def calculate_results(answers: Sequence[ModelAnswer], weights: Dict[str, float], task: str) -> Dict[str, Any]:
    pairs, matrix = pairwise_scores(answers)
    pooled = source_pool(answers)

    confidences = [a.confidence for a in answers]
    temp = 0.25
    scaled = [c / temp for c in confidences]
    mx = max(scaled) if scaled else 0
    exps = [math.exp(x - mx) for x in scaled]
    probs = [x / sum(exps) for x in exps] if exps else []
    max_prob = max(probs) if probs else 1

    latencies = [max(0.001, a.latency_seconds) for a in answers]
    lo_latency = min(latencies) if latencies else 0.0
    hi_latency = max(latencies) if latencies else 1.0
    latency_span = hi_latency - lo_latency
    speed_scores = [1.0 if latency_span < 1e-9 else (hi_latency - x) / latency_span for x in latencies]

    rows: List[Dict[str, Any]] = []
    verification_rows: List[Dict[str, Any]] = []
    for i, answer in enumerate(answers):
        agreement = sum(matrix[i][j] for j in range(len(answers)) if j != i) / max(1, len(answers) - 1)
        citation_presence = min(1.0, len(answer.sources) / 3)
        authorities = [domain_authority(s.domain) for s in answer.sources]
        authority = sum(authorities) / len(authorities) if authorities else 0.0
        unique_domains = len({s.domain for s in answer.sources if s.domain})
        diversity = min(1.0, unique_domains / 3)
        source_quality = 0.45 * authority + 0.25 * diversity + 0.30 * citation_presence

        supports: List[float] = []
        numeric_scores: List[float] = []
        corroborations: List[float] = []
        claim_details: List[Dict[str, Any]] = []
        for claim in answer.claims:
            own_support, own_source, lexical, numeric = best_evidence_support(claim, answer.sources)
            supports.append(own_support)
            numeric_scores.append(numeric)
            supporting_domains = set()
            pooled_matches = []
            for source in pooled:
                support, _, _, _ = best_evidence_support(claim, [source])
                if support >= 0.33:
                    supporting_domains.add(source.domain)
                    pooled_matches.append((support, source))
            corroboration = min(1.0, len(supporting_domains) / 2)
            corroborations.append(corroboration)
            pooled_matches.sort(key=lambda x: x[0], reverse=True)
            detail = {
                "claim": claim,
                "own_support": own_support,
                "lexical_support": lexical,
                "numeric_support": numeric,
                "best_source_url": own_source.url if own_source else "",
                "best_source_title": own_source.title if own_source else "",
                "best_source_domain": own_source.domain if own_source else "",
                "corroborating_domain_count": len(supporting_domains),
                "corroborating_domains": sorted(supporting_domains),
                "corroboration": corroboration,
                "pooled_best_sources": [
                    {"support": sc, "url": src.url, "title": src.title, "domain": src.domain}
                    for sc, src in pooled_matches[:3]
                ],
            }
            claim_details.append(detail)
            verification_rows.append({"model_id": answer.model_id, "model_name": answer.model_name, **detail})

        claim_support = sum(supports) / len(supports) if supports else 0.0
        numeric_support = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.0
        corroboration = sum(corroborations) / len(corroborations) if corroborations else 0.0
        claim_consensus = cross_model_claim_support(i, answers)
        number_consensus = cross_model_numeric_support(i, answers)

        if answer.sources:
            evidence_correctness = 0.50 * claim_support + 0.25 * numeric_support + 0.25 * corroboration
            factual_correctness = 0.80 * evidence_correctness + 0.20 * (0.70 * claim_consensus + 0.30 * number_consensus)
            verification_mode = "web-evidence plus cross-model support"
        else:
            factual_correctness = 0.70 * claim_consensus + 0.30 * number_consensus
            verification_mode = "cross-model consensus estimate; no web search"

        completeness = completeness_score(answer)
        clarity = clarity_score(answer.answer)
        if answer.sources:
            quality = 0.50 * factual_correctness + 0.20 * source_quality + 0.17 * completeness + 0.13 * clarity
        else:
            quality = 0.62 * factual_correctness + 0.23 * completeness + 0.15 * clarity

        confidence_w = probs[i] / max_prob if probs else 0.0
        speed_v = speed_scores[i]
        final = (
            weights["agreement"] * agreement
            + weights["quality"] * quality
            + weights["confidence"] * confidence_w
            + weights["speed"] * speed_v
        )
        contributions = {
            "agreement": weights["agreement"] * agreement,
            "quality": weights["quality"] * quality,
            "confidence": weights["confidence"] * confidence_w,
            "speed": weights["speed"] * speed_v,
        }

        rows.append({
            "model_id": answer.model_id,
            "model_name": answer.model_name,
            "provider": answer.provider,
            "answer": answer.answer,
            "key_points": answer.key_points,
            "claims": answer.claims,
            "confidence_raw": answer.confidence,
            "confidence_W": confidence_w,
            "agreement_C": agreement,
            "source_quality": source_quality,
            "source_authority": authority,
            "source_diversity": diversity,
            "citation_presence": citation_presence,
            "claim_support": claim_support,
            "numeric_support": numeric_support,
            "corroboration": corroboration,
            "claim_consensus": claim_consensus,
            "number_consensus": number_consensus,
            "factual_correctness": factual_correctness,
            "verification_mode": verification_mode,
            "completeness": completeness,
            "clarity": clarity,
            "quality_Q": quality,
            "speed_V": speed_v,
            "final_S": final,
            "score_contributions": contributions,
            "sources": [asdict(s) | {"authority_score": domain_authority(s.domain)} for s in answer.sources],
            "claim_details": claim_details,
            "latency_seconds": answer.latency_seconds,
            "catalog_speed_rank": answer.catalog_speed_rank,
            "web_search_requests": answer.web_search_requests,
            "error": answer.error,
        })

    rows.sort(key=lambda r: r["final_S"], reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank

    source_usage: Dict[str, Dict[str, Any]] = {}
    for answer in answers:
        for source in answer.sources:
            item = source_usage.setdefault(source.canonical_url, {
                "url": source.url,
                "canonical_url": source.canonical_url,
                "domain": source.domain,
                "title": source.title,
                "content": source.content,
                "authority_score": domain_authority(source.domain),
                "used_by": [],
            })
            item["used_by"].append(answer.model_id)
    source_rows = list(source_usage.values())
    for source in source_rows:
        source["model_count"] = len(set(source["used_by"]))
    source_rows.sort(key=lambda source: (source["model_count"], source["authority_score"]), reverse=True)

    winner = rows[0] if rows else {}
    contribution_labels = {
        "quality": "quality and correctness",
        "agreement": "agreement with the other models",
        "speed": "observed response speed",
        "confidence": "normalized self-confidence",
    }
    ordered_contributions = sorted((winner.get("score_contributions") or {}).items(), key=lambda item: item[1], reverse=True)
    winner_reasons = [
        f"Its final weighted score was {winner.get('final_S', 0):.3f}, the highest of the ten responses.",
        f"Its quality score was {winner.get('quality_Q', 0):.3f} and its estimated correctness score was {winner.get('factual_correctness', 0):.3f}.",
        f"It agreed with the rest of the panel at {winner.get('agreement_C', 0):.3f} and answered in {winner.get('latency_seconds', 0):.2f} seconds.",
    ]
    if ordered_contributions:
        top_names = [contribution_labels.get(name, name) for name, _ in ordered_contributions[:2]]
        winner_reasons.append("Its two largest score contributions came from " + " and ".join(top_names) + ".")
    if winner.get("sources"):
        winner_reasons.append(f"It returned {len(winner['sources'])} source citation(s); citation support is included in its correctness estimate.")
    else:
        winner_reasons.append("Its correctness estimate comes from agreement of claims and numbers across the ten model responses. This is not independent proof of truth.")

    labels = [a.model_name for a in answers]
    return {
        "models": rows,
        "pairs": pairs,
        "similarity_matrix": matrix,
        "matrix_labels": labels,
        "sources": source_rows,
        "claim_verification": verification_rows,
        "weights": weights,
        "model_count": len(answers),
        "pair_count": len(pairs),
        "unique_source_count": len(source_rows),
        "unique_domain_count": len({s["domain"] for s in source_rows}),
        "winner": winner,
        "winner_reasons": winner_reasons,
        "method_note": (
            "No web search or source retrieval is used. The application does not ask an eleventh LLM to judge the answers. "
            "It ranks the responses using transparent answer similarity, cross-model claim and number agreement, completeness, clarity, confidence, and observed speed. "
            "The correctness value is a consensus estimate, not independent proof of truth."
        ),
    }


# -----------------------------------------------------------------------------
# Jobs and files
# -----------------------------------------------------------------------------

def update_job(job_id: str, **changes: Any) -> None:
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(changes)


def add_progress(job_id: str, message: str, progress: Optional[int] = None) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.setdefault("messages", []).append(message)
        if progress is not None:
            job["progress"] = progress


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serial = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()}
            writer.writerow(serial)


def write_run_files(job_id: str, input_data: Dict[str, Any], result: Dict[str, Any]) -> str:
    RUNS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{stamp}_{job_id[:6]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "input.json").write_text(json.dumps(input_data, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(run_dir / "model_scores.csv", result["models"], [
        "rank","model_id","model_name","provider","agreement_C","factual_correctness","source_quality",
        "source_authority","source_diversity","claim_support","numeric_support","corroboration","completeness",
        "clarity","quality_Q","confidence_raw","confidence_W","speed_V","final_S","latency_seconds","catalog_speed_rank","web_search_requests","verification_mode","error"
    ])
    write_csv(run_dir / "pairwise_comparisons.csv", result["pairs"], [
        "name_a","name_b","text_similarity","key_point_similarity","claim_similarity","url_overlap",
        "domain_overlap","source_overlap","evidence_similarity","combined_similarity","sources_a","sources_b"
    ])
    write_csv(run_dir / "sources.csv", result["sources"], [
        "title","url","domain","authority_score","model_count","used_by","content"
    ])
    write_csv(run_dir / "claim_verification.csv", result["claim_verification"], [
        "model_name","claim","own_support","lexical_support","numeric_support","best_source_title",
        "best_source_url","best_source_domain","corroborating_domain_count","corroborating_domains","corroboration"
    ])
    return str(run_dir)


def normalize_weights(raw: Dict[str, Any]) -> Dict[str, float]:
    defaults = {"agreement": 0.25, "quality": 0.55, "confidence": 0.05, "speed": 0.15}
    values: Dict[str, float] = {}
    for key, default in defaults.items():
        try:
            values[key] = max(0.0, float(raw.get(key, default)))
        except Exception:
            values[key] = default
    total = sum(values.values())
    if total <= 0:
        return defaults
    return {key: value / total for key, value in values.items()}


def run_job(job_id: str, payload: Dict[str, Any]) -> None:
    try:
        task = clean_text(payload.get("task", ""))
        demo = bool(payload.get("demo_mode"))
        if not task:
            raise ValueError("Enter a question, prompt, topic, or text.")
        if len(task) > MAX_TASK_CHARS:
            raise ValueError(f"Input is too long. Maximum is {MAX_TASK_CHARS:,} characters.")

        weights = normalize_weights(payload.get("weights") or {})
        use_web_search = False
        allow_plain_fallback = False
        free_only = True
        engine = "disabled"
        max_results = 0
        excluded: List[str] = []
        manual = [x.strip() for x in str(payload.get("manual_models") or "").splitlines() if x.strip()]

        add_progress(job_id, "Preparing the ten-model evaluation.", 4)
        if demo:
            answers = demo_answers(task)
            add_progress(job_id, "Sample mode: generated ten local demonstration responses. No API key or network request was used.", 45)
        else:
            api_key = str(payload.get("api_key") or "").strip() or current_api_key()
            if not api_key:
                raise ValueError("An OpenRouter API key is required for live mode. Add it to the included .env file.")
            if payload.get("save_key"):
                save_api_key(api_key)

            mode_text = "free text models" if free_only else "text models, including paid models"
            add_progress(job_id, f"BUILD {APP_BUILD}: requesting OpenRouter's live catalog of {mode_text}, sorted by recent median latency.", 8)
            selected, backups = discover_fastest_models(api_key, manual, free_only=True)

            # Quota-safe preflight: one candidate is called first. Its successful
            # answer becomes model #1, so this does not add an extra successful
            # request. If the account is already at zero free requests, the app
            # stops after this single failed request instead of launching ten.
            candidate_queue = list(selected) + list(backups)
            answers: List[ModelAnswer] = []
            failed: List[ModelAnswer] = []
            used_ids = set()
            probe_attempts = 0
            max_probe_attempts = min(3, len(candidate_queue))
            add_progress(job_id, "SAFE PREFLIGHT: testing one free text-generating model before launching the remaining nine requests.", 12)
            while candidate_queue and not answers and probe_attempts < max_probe_attempts:
                model = candidate_queue.pop(0)
                probe_attempts += 1
                used_ids.add(str(model.get("_slot_key") or model.get("id")))
                answer = call_model(api_key, model, task)
                if "FREE_DAILY_QUOTA" in answer.error:
                    reset = answer.error.split("reset ", 1)[-1]
                    raise RuntimeError(
                        "OpenRouter reports that this account has zero free-model requests remaining. "
                        f"The v3 SAFE preflight stopped after one failed request. Wait until {reset}, "
                        "add OpenRouter credits, or use the no-key sample mode."
                    )
                if answer.answer:
                    answers.append(answer)
                    source_note = f" with {len(answer.sources)} source(s)" if answer.sources else ""
                    add_progress(job_id, f"Preflight passed: {answer.model_name} returned a usable answer in {answer.latency_seconds:.2f}s{source_note}.", 18)
                else:
                    failed.append(answer)
                    add_progress(job_id, f"Preflight candidate {probe_attempts}/{max_probe_attempts} failed: {answer.model_name}: {answer.error[:220]}", 14 + probe_attempts)

            if not answers:
                detail = "; ".join(a.error for a in failed[:3] if a.error)
                raise RuntimeError(
                    "No free text-generating OpenRouter candidate passed the safe preflight. "
                    "No bulk request was launched. " + detail
                )

            # Fill the remaining slots from the same latency-ordered catalog,
            # excluding models already attempted during preflight.
            remaining_models: List[Dict[str, Any]] = []
            for model in list(selected) + list(backups):
                mid = str(model.get("_slot_key") or model.get("id"))
                if mid in used_ids:
                    continue
                remaining_models.append(model)
                used_ids.add(mid)
                if len(remaining_models) >= ACTIVE_MODEL_COUNT - len(answers):
                    break

            add_progress(job_id, f"Calling the remaining {len(remaining_models)} fastest free text-generating candidates in parallel. No web search is used.", 22)
            with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, max(1, len(remaining_models)))) as pool:
                future_map = {
                    pool.submit(call_model, api_key, model, task): model
                    for model in remaining_models
                }
                completed = 0
                for future in as_completed(future_map):
                    answer = future.result()
                    completed += 1
                    if "FREE_DAILY_QUOTA" in answer.error:
                        failed.append(answer)
                    elif answer.answer:
                        answers.append(answer)
                    else:
                        failed.append(answer)
                    if answer.answer:
                        status = f"succeeded in {answer.latency_seconds:.2f}s"
                        if answer.sources:
                            status += f" with {len(answer.sources)} source(s)"
                        elif use_web_search:
                            status += " but returned no source annotations"
                    else:
                        status = f"failed in {answer.latency_seconds:.2f}s: {answer.error[:220]}"
                    add_progress(job_id, f"{completed}/{len(remaining_models)} remaining: {answer.model_name} {status}.", 22 + int(32 * completed / max(1, len(remaining_models))))

            quota_failures = [a for a in failed if "FREE_DAILY_QUOTA" in a.error]
            if quota_failures:
                reset = quota_failures[0].error.split("reset ", 1)[-1]
                raise RuntimeError(
                    "OpenRouter exhausted the account's free-model daily quota during the run. "
                    f"The application stopped all replacement attempts. Wait until {reset}, add credits, "
                    "or use sample mode."
                )

            # At most two controlled replacements. No hidden retry storm.
            attempts = 0
            replacement_pool = [m for m in candidate_queue if str(m.get("_slot_key") or m.get("id")) not in used_ids]
            while len(answers) < ACTIVE_MODEL_COUNT and replacement_pool and attempts < MAX_REPLACEMENT_ATTEMPTS:
                model = replacement_pool.pop(0)
                attempts += 1
                add_progress(job_id, f"Controlled replacement {attempts}/{MAX_REPLACEMENT_ATTEMPTS}: {model.get('id')}.")
                answer = call_model(api_key, model, task)
                if "FREE_DAILY_QUOTA" in answer.error:
                    reset = answer.error.split("reset ", 1)[-1]
                    raise RuntimeError(
                        "OpenRouter's free-model quota was exhausted during a controlled replacement. "
                        f"No more requests were sent. Wait until {reset}."
                    )
                if answer.answer:
                    answers.append(answer)
                else:
                    failed.append(answer)

            answers = answers[:ACTIVE_MODEL_COUNT]
            if len(answers) < 2:
                detail = "; ".join(a.error for a in failed[:3] if a.error)
                raise RuntimeError("Fewer than two usable LLM responses were returned. " + detail)
            if len(answers) < ACTIVE_MODEL_COUNT:
                add_progress(job_id, f"Continuing safely with {len(answers)} usable responses. Replacement attempts were capped at {MAX_REPLACEMENT_ATTEMPTS}.")

        add_progress(job_id, "Building all answer-to-answer similarities and the complete 3D network.", 64)
        result = calculate_results(answers, weights, task)
        add_progress(job_id, f"Calculated {result['pair_count']} unique answer-to-answer comparisons.", 78)
        add_progress(job_id, "Calculating agreement, cross-model correctness, quality, confidence, and observed speed.", 88)
        input_data = {
            "task": task,
            "demo_mode": demo,
            "weights": weights,
            "use_web_search": False,
            "free_only": True,
            "selected_models": [answer.model_id for answer in answers],
            "created_at": datetime.now().isoformat(),
        }
        result["task"] = task
        result["created_at"] = datetime.now().isoformat()
        result["selection_note"] = (
            f"Live candidates were taken from OpenRouter's latency-low-to-high catalog order. The app used free text-generating models only and sent no web-search tools. "
            f"{len(answers)} usable responses were scored. A one-model quota-safe preflight ran first, and replacements were capped at {MAX_REPLACEMENT_ATTEMPTS}."
            if not demo else
            "Sample mode uses ten local demonstration responses and simulated timings."
        )
        result["run_directory"] = write_run_files(job_id, input_data, result)
        update_job(job_id, status="complete", progress=100, result=result)
        add_progress(job_id, f"Complete. Winner: {result['models'][0]['model_name']}. Results saved in {result['run_directory']}.")
    except Exception as exc:
        update_job(job_id, status="error", progress=100, error=str(exc), traceback=traceback.format_exc())
        add_progress(job_id, f"ERROR: {exc}")


# -----------------------------------------------------------------------------
# Browser UI
# -----------------------------------------------------------------------------
PAGE_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Live 10 Fastest Free LLM 3D Evaluator v4</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--bg:#06101d;--panel:#0c1b2d;--panel2:#11263d;--text:#e8f0f9;--muted:#9bb0c5;--accent:#54c8ff;--good:#54dc91;--warn:#ffd166;--bad:#ff6b6b;--line:#29425e}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% 0,#142c49,#06101d 60%);color:var(--text);font-family:Inter,Segoe UI,Arial,sans-serif;min-height:100vh}header{padding:20px 28px;border-bottom:1px solid var(--line);background:#06101dee;position:sticky;top:0;z-index:9;backdrop-filter:blur(10px)}h1{margin:0;font-size:25px}header p{margin:7px 0 0;color:var(--muted)}main{max-width:1580px;margin:auto;padding:22px}.grid{display:grid;grid-template-columns:minmax(330px,440px) 1fr;gap:18px}.card{background:#0c1b2df2;border:1px solid var(--line);border-radius:16px;padding:17px;box-shadow:0 16px 42px #0004}label{display:block;font-weight:750;margin:12px 0 6px}textarea,input,select,button{font:inherit}textarea,input[type=password],input[type=text],input[type=number],select{width:100%;background:#061421;color:var(--text);border:1px solid #325372;border-radius:10px;padding:10px;outline:none}textarea{min-height:130px;resize:vertical}button{border:0;border-radius:10px;padding:12px 15px;font-weight:800;cursor:pointer;color:white;background:linear-gradient(90deg,#1ea9e8,#6b7cff)}button.secondary{background:#1a3654}button:disabled{opacity:.45}.row{display:flex;gap:9px}.row>*{flex:1}.check{display:flex;gap:8px;align-items:center;color:var(--muted);margin-top:9px}.check input{width:auto}.notice{padding:11px;border-radius:10px;background:#102a42;border:1px solid #27516f;color:#d2ecff;margin:11px 0;line-height:1.45}.cost{border-color:#695d2a;background:#302a12;color:#ffe79d}.small{font-size:12px;color:var(--muted);line-height:1.45}.weights{display:grid;grid-template-columns:repeat(4,1fr);gap:7px}.progress{height:9px;background:#061421;border-radius:20px;overflow:hidden;margin:12px 0}.progress>div{height:100%;width:0;background:linear-gradient(90deg,var(--accent),var(--good));transition:.25s}.log{height:125px;overflow:auto;background:#05101b;border-radius:9px;padding:9px;font:12px Consolas,monospace;color:#bed0e2;white-space:pre-wrap}.hidden{display:none}.summary{display:grid;grid-template-columns:repeat(5,1fr);gap:9px;margin-bottom:12px}.metric{background:#081827;border:1px solid var(--line);border-radius:11px;padding:11px}.metric b{display:block;font-size:20px;margin-top:4px}.metric span{font-size:11px;color:var(--muted)}.tabs{display:flex;gap:14px;flex-wrap:wrap;margin:8px 0 18px;padding:18px;border:1px solid #29425e;border-radius:22px;background:#09192a}.tab{background:#173651;padding:16px 24px;border-radius:20px;font-size:18px;box-shadow:none}.tab.active{background:#2d86b5}.tab:hover{background:#22577d}.view{display:none}.view.active{display:block}#graph3d,#sourceGraph,#heatmap{height:650px}.detail{background:#081827;border:1px solid var(--line);border-radius:11px;padding:14px;margin-top:10px}.answer{line-height:1.58}.formula{font:12px Consolas,monospace;background:#05101b;border-radius:9px;padding:10px;color:#bce9ff;overflow:auto}.pill{display:inline-block;padding:3px 8px;margin:2px;border-radius:20px;background:#173957;font-size:11px}.ranking{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:9px}.rank-card{background:#081827;border:1px solid var(--line);border-radius:11px;padding:12px;cursor:pointer}.rank-card:hover{border-color:var(--accent)}.rank-num{font-size:22px;font-weight:900;color:var(--accent)}.bar{height:6px;background:#1a2d42;border-radius:9px;overflow:hidden;margin:8px 0}.bar div{height:100%;background:linear-gradient(90deg,var(--bad),var(--warn),var(--good))}.source-card,.claim-card{border-left:4px solid var(--accent);background:#0d2337;padding:10px;margin:8px 0;border-radius:6px}.source-card.good,.claim-card.good{border-color:var(--good)}.source-card.bad,.claim-card.bad{border-color:var(--bad)}table{width:100%;border-collapse:collapse;font-size:12px}th,td{border:1px solid var(--line);padding:7px;text-align:left;vertical-align:top}th{background:#102a42;position:sticky;top:0}.table-wrap{max-height:640px;overflow:auto}.error{color:#ff9999}.warning{color:#ffe08a}a{color:#71d4ff}.two{display:grid;grid-template-columns:1fr 1fr;gap:10px}@media(max-width:1050px){.grid{grid-template-columns:1fr}.summary{grid-template-columns:repeat(2,1fr)}#graph3d,#sourceGraph,#heatmap{height:520px}.two{grid-template-columns:1fr}}
</style></head><body>
<header><h1>Live 10 Fastest Free LLM 3D Evaluator <span style="font-size:.45em;background:#35a7d6;padding:6px 10px;border-radius:10px;vertical-align:middle"></span></h1><p>Type any question, topic, prompt, or text. Ten fast free OpenRouter models answer in parallel. No web search is used. Transparent mathematics compares the answers and explains the winner.</p></header>
<main>
<div class="notice" style="border:2px solid #35a7d6"><b>RUNNING BUILD v4 FREE NO WEB on port 8020.</b> Free text models only. No tools, citations, or web-search requests are sent.</div><div class="grid"><section class="card"><h2>1. Ask all 10 LLMs</h2>
<div class="notice"><b>API key setup:</b> place your OpenRouter key in the included <code>.env</code> file. The key stays on your localhost server and is never displayed in the browser.</div>
<div id="keyStatus" class="notice">Checking the local .env file...</div>
<div class="notice cost"><b></b></div>
<label>Your question, prompt, topic, or text</label><textarea id="task" placeholder="Example: Big Foot"></textarea>
<details><summary style="cursor:pointer;margin-top:12px">Model selection and scoring</summary>
<div class="notice"><b>Fixed mode:</b> free models only, text responses only, no web search. Multimodal or coding-capable models may participate when they can answer a text prompt.</div>
<label>Manual free model IDs (optional, one per line)</label><textarea id="manualModels" placeholder="Leave blank to choose the fastest free text-generating models from OpenRouter's live catalog." style="min-height:90px"></textarea>
<label>Final-score weights</label><div class="weights"><div><span class="small">Agreement C</span><input id="wAgreement" type="number" step="0.05" value="0.25"></div><div><span class="small">Quality Q</span><input id="wQuality" type="number" step="0.05" value="0.55"></div><div><span class="small">Confidence W</span><input id="wConfidence" type="number" step="0.05" value="0.05"></div><div><span class="small">Speed V</span><input id="wSpeed" type="number" step="0.05" value="0.15"></div></div></details>
<div class="row" style="margin-top:13px"><button id="runBtn">Run 10 free LLMs</button><button id="demoBtn" class="secondary">Run sample without API key</button></div>
<button id="sampleBtn" class="secondary" style="width:100%;margin-top:9px">Load sample text only</button>
<div class="progress"><div id="progressBar"></div></div><div id="statusText" class="small">Ready.</div><div id="log" class="log"></div></section>
<section class="card"><div id="empty"><h2>2. Results</h2><p class="small">Run live or use sample mode to build all 45 comparisons, rank the ten answers, explain the winner, and display the complete 3D graph.</p></div><div id="results" class="hidden"><div id="summary" class="summary"></div><div class="tabs">
<button class="tab active" data-view="graphView">3D complete graph</button><button class="tab" data-view="matrixView">Similarity matrix</button><button class="tab" data-view="rankingView">Ranking</button><button class="tab" data-view="validationView">Correctness analysis</button><button class="tab" data-view="responsesView">All responses</button></div>
<div id="graphView" class="view active"><div id="graph3d"></div><div id="graphDetail" class="detail">Click an LLM node or an edge midpoint to inspect its mathematics.</div></div>
<div id="matrixView" class="view"><div id="heatmap"></div><div class="formula">Pair = (0.35×answer text + 0.20×key points + 0.20×claims) ÷ 0.75</div></div>
<div id="rankingView" class="view"><div id="winnerReason" class="notice"></div><div id="ranking" class="ranking"></div><div id="rankDetail" class="detail"></div></div>
<div id="validationView" class="view"><h3>Cross-model correctness estimate</h3><div class="notice">No web search is used. “Correctness” here means that a model's claims and numbers are supported by the other LLM responses. Ten models can still agree on something wrong, so this is a consensus estimate, not verified truth.</div><div class="table-wrap"><table id="validationTable"></table></div><div id="validationDetail" class="detail"></div></div>
<div id="responsesView" class="view"><div id="answers"></div></div></div></section></div></main>
<script>
const $=id=>document.getElementById(id);let latest=null,pollTimer=null;const esc=s=>String(s??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));const f=(n,d=3)=>Number(n||0).toFixed(d);const color=v=>v>=.75?'#54dc91':v>=.5?'#ffd166':'#ff6b6b';
const SAMPLE_TEXT="Big Foot";
$('sampleBtn').onclick=()=>{$('task').value=SAMPLE_TEXT};
fetch('/api/config').then(r=>r.json()).then(c=>{$('keyStatus').innerHTML=`<b>Build ${esc(c.build)} · port ${c.port}.</b> `+(c.has_saved_key?'OpenRouter key found in .env. Live mode is ready.':'No OpenRouter key found in .env. Add OPENROUTER_API_KEY=your_key. Sample mode still works.')}).catch(()=>{$('keyStatus').textContent='Could not check .env status.'});
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.view').forEach(x=>x.classList.remove('active'));b.classList.add('active');$(b.dataset.view).classList.add('active')});
function buildPayload(demo){return{task:$('task').value,demo_mode:demo,manual_models:$('manualModels').value,weights:{agreement:Number($('wAgreement').value),quality:Number($('wQuality').value),confidence:Number($('wConfidence').value),speed:Number($('wSpeed').value)}}}
function setBusy(v){$('runBtn').disabled=v;$('demoBtn').disabled=v}
async function startRun(demo){if(demo)$('task').value=SAMPLE_TEXT;setBusy(true);$('log').textContent='';$('progressBar').style.width='2%';try{const r=await fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(buildPayload(demo))});const d=await r.json();if(!r.ok)throw new Error(d.error||'Could not start');poll(d.job_id)}catch(e){fail(e.message)}}
$('runBtn').onclick=()=>startRun(false);$('demoBtn').onclick=()=>startRun(true);
function poll(id){if(pollTimer)clearInterval(pollTimer);pollTimer=setInterval(async()=>{try{const r=await fetch('/api/status?job_id='+encodeURIComponent(id));const j=await r.json();$('progressBar').style.width=(j.progress||0)+'%';$('statusText').textContent=j.status;$('log').textContent=(j.messages||[]).join('\n');$('log').scrollTop=$('log').scrollHeight;if(j.status==='complete'){clearInterval(pollTimer);setBusy(false);latest=j.result;renderAll(latest)}else if(j.status==='error'){clearInterval(pollTimer);fail(j.error)}}catch(e){clearInterval(pollTimer);fail(e.message)}},900)}function fail(m){setBusy(false);$('statusText').innerHTML='<span class="error">'+esc(m)+'</span>'}
function renderAll(d){$('empty').classList.add('hidden');$('results').classList.remove('hidden');$('summary').innerHTML=`<div class="metric"><span>Active LLMs</span><b>${d.model_count}</b></div><div class="metric"><span>Pair comparisons</span><b>${d.pair_count}</b></div><div class="metric"><span>Fastest observed</span><b>${f(Math.min(...d.models.map(m=>m.latency_seconds)),2)}s</b></div><div class="metric"><span>Winner score</span><b>${f(d.models[0]?.final_S)}</b></div><div class="metric"><span>Best answer</span><b style="font-size:14px">${esc(d.models[0]?.model_name||'')}</b></div>`;renderGraph(d);renderHeatmap(d);renderRanking(d);renderValidation(d);renderAnswers(d)}
function positions(n,r=9){return Array.from({length:n},(_,i)=>{const a=2*Math.PI*i/n;return{x:r*Math.cos(a),y:r*Math.sin(a)}})}
function renderGraph(d){const ms=d.models,pos=positions(ms.length),by=Object.fromEntries(ms.map((m,i)=>[m.model_id,{...m,...pos[i]}])),tr=[];d.pairs.forEach((p,i)=>{const a=by[p.model_a],b=by[p.model_b];if(!a||!b)return;tr.push({type:'scatter3d',mode:'lines',x:[a.x,b.x],y:[a.y,b.y],z:[a.final_S*10,b.final_S*10],line:{width:1+p.combined_similarity*7,color:color(p.combined_similarity)},opacity:.25+p.combined_similarity*.6,hoverinfo:'skip',showlegend:false})});const mx=[],my=[],mz=[],mt=[],mc=[];d.pairs.forEach((p,i)=>{const a=by[p.model_a],b=by[p.model_b];if(!a||!b)return;mx.push((a.x+b.x)/2);my.push((a.y+b.y)/2);mz.push((a.final_S+b.final_S)*5);mt.push(`${a.model_name} ↔ ${b.model_name}<br>Combined ${f(p.combined_similarity)}<br>Text ${f(p.text_similarity)}<br>Claims ${f(p.claim_similarity)}`);mc.push(i)});tr.push({type:'scatter3d',mode:'markers',x:mx,y:my,z:mz,marker:{size:4,color:d.pairs.map(p=>p.combined_similarity),colorscale:'RdYlGn',cmin:0,cmax:1},text:mt,hoverinfo:'text',customdata:mc,name:'Pairs'});tr.push({type:'scatter3d',mode:'markers+text',x:ms.map((m,i)=>pos[i].x),y:ms.map((m,i)=>pos[i].y),z:ms.map(m=>m.final_S*10),text:ms.map(m=>`#${m.rank} ${m.model_name}`),textposition:'top center',marker:{size:ms.map(m=>10+14*m.final_S),color:ms.map(m=>m.final_S),colorscale:'Viridis',cmin:0,cmax:1,colorbar:{title:'Final S'},line:{color:'#fff',width:1}},customdata:ms.map((m,i)=>i),hovertext:ms.map(m=>`${m.model_name}<br>S ${f(m.final_S)}<br>Correct ${f(m.factual_correctness)}<br>Consensus ${f(m.factual_correctness)}`),hoverinfo:'text',name:'LLMs'});Plotly.newPlot('graph3d',tr,{paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:'#e8f0f9'},margin:{l:0,r:0,t:10,b:0},showlegend:false,scene:{xaxis:{visible:false},yaxis:{visible:false},zaxis:{title:'Final score'},bgcolor:'#071421'}},{responsive:true});$('graph3d').on('plotly_click',e=>{const p=e.points[0];p.data.name==='LLMs'?showModel(ms[p.customdata],$('graphDetail')):showPair(d.pairs[p.customdata],$('graphDetail'))})}
function renderSourceGraph(d){const ms=d.models,mp=positions(ms.length,10),domains=[...new Set(d.sources.map(s=>s.domain))].filter(Boolean),dp=positions(domains.length,4),tr=[];const byD=Object.fromEntries(domains.map((x,i)=>[x,dp[i]]));ms.forEach((m,i)=>(m.sources||[]).forEach(s=>{const q=byD[s.domain];if(q)tr.push({type:'scatter3d',mode:'lines',x:[mp[i].x,q.x],y:[mp[i].y,q.y],z:[m.final_S*8,0],line:{width:1+3*s.authority_score,color:color(s.authority_score)},opacity:.5,hoverinfo:'skip',showlegend:false})}));tr.push({type:'scatter3d',mode:'markers+text',x:ms.map((m,i)=>mp[i].x),y:ms.map((m,i)=>mp[i].y),z:ms.map(m=>m.final_S*8),text:ms.map(m=>m.model_name),textposition:'top center',marker:{size:11,color:ms.map(m=>m.final_S),colorscale:'Viridis'},hoverinfo:'text',hovertext:ms.map(m=>`${m.model_name}<br>${m.sources.length} sources`),name:'LLMs'});tr.push({type:'scatter3d',mode:'markers+text',x:domains.map(x=>byD[x].x),y:domains.map(x=>byD[x].y),z:domains.map(()=>0),text:domains,textposition:'bottom center',marker:{size:domains.map(x=>8+6*Math.max(...d.sources.filter(s=>s.domain===x).map(s=>s.authority_score))),color:domains.map(x=>Math.max(...d.sources.filter(s=>s.domain===x).map(s=>s.authority_score))),colorscale:'RdYlGn',cmin:0,cmax:1},hoverinfo:'text',hovertext:domains.map(x=>x),name:'Source domains'});Plotly.newPlot('sourceGraph',tr,{paper_bgcolor:'transparent',font:{color:'#e8f0f9'},margin:{l:0,r:0,t:10,b:0},showlegend:false,scene:{xaxis:{visible:false},yaxis:{visible:false},zaxis:{visible:false},bgcolor:'#071421'}},{responsive:true})}
function renderHeatmap(d){Plotly.newPlot('heatmap',[{z:d.similarity_matrix,x:d.matrix_labels,y:d.matrix_labels,type:'heatmap',zmin:0,zmax:1,colorscale:'RdYlGn',text:d.similarity_matrix.map(r=>r.map(v=>f(v,2))),texttemplate:'%{text}',hovertemplate:'%{y} ↔ %{x}<br>%{z:.3f}<extra></extra>'}],{paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:'#e8f0f9'},margin:{l:130,r:20,t:25,b:120},xaxis:{tickangle:-40}},{responsive:true})}
function renderRanking(d){$('winnerReason').innerHTML=`<h3>Why ${esc(d.models[0]?.model_name||'')} won</h3><p>${(d.winner_reasons||[]).map(esc).join('<br>')}</p><div class="formula">S = ${f(d.weights.agreement,2)}×Agreement + ${f(d.weights.quality,2)}×Quality + ${f(d.weights.confidence,2)}×Confidence + ${f(d.weights.speed,2)}×Speed</div><p class="small">${esc(d.selection_note||'')}</p>`;$('ranking').innerHTML=d.models.map((m,i)=>`<div class="rank-card" data-i="${i}"><div class="rank-num">#${m.rank}</div><b>${esc(m.model_name)}</b><div class="small">${esc(m.model_id)}</div><div class="bar"><div style="width:${m.final_S*100}%"></div></div><span class="pill">S ${f(m.final_S)}</span><span class="pill">Quality ${f(m.quality_Q)}</span><span class="pill">${f(m.latency_seconds,2)}s</span></div>`).join('');document.querySelectorAll('.rank-card').forEach(x=>x.onclick=()=>showModel(d.models[Number(x.dataset.i)],$('rankDetail')));showModel(d.models[0],$('rankDetail'))}
function showModel(m,t){const claims=(m.claims||[]).map(c=>`<div class="claim-card"><b>${esc(c)}</b></div>`).join('');t.innerHTML=`<h3>#${m.rank} ${esc(m.model_name)}</h3><div class="small">${esc(m.model_id)} · OpenRouter catalog speed position ${m.catalog_speed_rank||'n/a'} · actual completion ${f(m.latency_seconds,2)} sec</div><p class="answer">${esc(m.answer).replace(/\n/g,'<br>')}</p><div class="formula">S = ${f(latest.weights.agreement,2)}×C + ${f(latest.weights.quality,2)}×Q + ${f(latest.weights.confidence,2)}×W + ${f(latest.weights.speed,2)}×V<br>= ${f(m.final_S)}</div><p><span class="pill">Agreement C ${f(m.agreement_C)}</span><span class="pill">Quality Q ${f(m.quality_Q)}</span><span class="pill">Consensus correctness ${f(m.factual_correctness)}</span><span class="pill">Claim consensus ${f(m.claim_consensus)}</span><span class="pill">Number consensus ${f(m.number_consensus)}</span><span class="pill">Confidence W ${f(m.confidence_W)}</span><span class="pill">Speed V ${f(m.speed_V)}</span></p><p class="small"><b>Verification mode:</b> ${esc(m.verification_mode)}</p><h4>Claims extracted from this response</h4>${claims||'<p class="small">No structured claims were returned.</p>'}${m.error?`<p class="warning">${esc(m.error)}</p>`:''}`}
function showPair(p,t){t.innerHTML=`<h3>${esc(p.name_a)} ↔ ${esc(p.name_b)}</h3><div class="formula">Pair = (0.35×Text + 0.20×Key points + 0.20×Claims) ÷ 0.75<br>= ${f(p.combined_similarity)}</div><p><span class="pill">Text ${f(p.text_similarity)}</span><span class="pill">Key points ${f(p.key_point_similarity)}</span><span class="pill">Claims ${f(p.claim_similarity)}</span></p>`}
function renderValidation(d){$('validationTable').innerHTML='<thead><tr><th>Rank</th><th>Model</th><th>Correctness estimate</th><th>Claim consensus</th><th>Number consensus</th><th>Completeness</th><th>Clarity</th></tr></thead><tbody>'+d.models.map((m,i)=>`<tr data-i="${i}"><td>#${m.rank}</td><td>${esc(m.model_name)}</td><td>${f(m.factual_correctness)}</td><td>${f(m.claim_consensus)}</td><td>${f(m.number_consensus)}</td><td>${f(m.completeness)}</td><td>${f(m.clarity)}</td></tr>`).join('')+'</tbody>';document.querySelectorAll('#validationTable tbody tr').forEach(x=>x.onclick=()=>{const m=d.models[Number(x.dataset.i)];$('validationDetail').innerHTML=`<h3>${esc(m.model_name)}</h3><p class="answer">${esc(m.answer).replace(/\n/g,'<br>')}</p><p><span class="pill">Correctness estimate ${f(m.factual_correctness)}</span><span class="pill">Claim consensus ${f(m.claim_consensus)}</span><span class="pill">Number consensus ${f(m.number_consensus)}</span><span class="pill">Agreement ${f(m.agreement_C)}</span></p><p class="small">This score checks whether other models make similar claims and use matching numbers. It does not verify the answer against outside sources.</p>`});if(d.models.length){document.querySelector('#validationTable tbody tr')?.click()}}
function renderAnswers(d){$('answers').innerHTML=`<div class="notice">${esc(d.method_note)}</div>`+d.models.map(m=>`<div class="detail"><h3>#${m.rank} ${esc(m.model_name)}</h3><p class="answer">${esc(m.answer).replace(/\n/g,'<br>')}</p><p class="small">${m.claims.length} claims · ${f(m.latency_seconds,2)} sec · final ${f(m.final_S)}</p></div>`).join('')}
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    server_version = "Live10FastestFreeLLM3D/7.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(data)

    def send_json(self, obj: Any, status: int = 200) -> None:
        self.send_bytes(json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

    def read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1_000_000: raise ValueError("Request too large")
        return json.loads(self.rfile.read(length).decode("utf-8") or "{}")

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            index_file = APP_DIR / "index.html"
            html_content = index_file.read_text(encoding="utf-8") if index_file.exists() else PAGE_HTML
            self.send_bytes(html_content.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/config":
            self.send_json({"has_saved_key": bool(current_api_key()), "active_model_count": ACTIVE_MODEL_COUNT, "build": APP_BUILD, "port": DEFAULT_PORT})
            return
        if parsed.path == "/api/sample_questions":
            samples_file = APP_DIR / "sample_questions.txt"
            questions = [q.strip() for q in samples_file.read_text(encoding="utf-8").split("\n\n") if q.strip()] if samples_file.exists() else []
            self.send_json({"questions": questions})
            return
        if parsed.path == "/api/status":
            job_id = (urllib.parse.parse_qs(parsed.query).get("job_id") or [""])[0]
            with JOBS_LOCK: job = dict(JOBS.get(job_id) or {})
            if not job: self.send_json({"error":"Unknown job"},404)
            else: self.send_json(job)
            return
        self.send_json({"error":"Not found"},404)

    def do_POST(self) -> None:
        if self.path != "/api/start": self.send_json({"error":"Not found"},404); return
        try:
            payload = self.read_json()
            job_id = secrets.token_hex(8)
            with JOBS_LOCK:
                JOBS[job_id] = {"status":"running","progress":1,"messages":[],"result":None,"error":""}
            threading.Thread(target=run_job, args=(job_id,payload), daemon=True).start()
            self.send_json({"job_id":job_id},202)
        except Exception as exc:
            self.send_json({"error":str(exc)},400)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live 10 Fastest Free LLM 3D Evaluator v4")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    RUNS_DIR.mkdir(exist_ok=True)
    server = ThreadingHTTPServer((HOST, args.port), Handler)
    url = f"http://{HOST}:{args.port}/?build={APP_BUILD}"
    print("="*68)
    print(f"Live 10 Fastest Free LLM 3D Evaluator - {APP_BUILD}")
    print(f"Open: {url}")
    print("Keep this terminal open. Press Ctrl+C to stop the server.")
    print("="*68)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try: server.serve_forever()
    except KeyboardInterrupt: print("\nStopping server...")
    finally: server.server_close()


if __name__ == "__main__":
    main()
