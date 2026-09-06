import os
import re
import json
import math
import time
import random
import hashlib
import secrets
import traceback
from datetime import datetime, timezone
from typing import Optional, Any
from urllib.parse import urlencode, urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel

APP_NAME = "ASCEND AI"
APP_VERSION = "4.0.0"

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

PRIVACY_URL = "https://telegra.ph/Politika-konfidencialnosti-09-06-116"
TERMS_URL = "https://telegra.ph/Polzovatelskoe-soglashenie-09-06-54"
SUPPORT_HANDLE = "@lovnff"
SUPPORT_URL = "https://t.me/lovnff"

PLANS = [
    {"id": "start", "name": "Старт", "requests": 50, "price": 59, "per_request": "≈1.18₽ за запрос", "popular": False},
    {"id": "standard", "name": "Стандарт", "requests": 150, "price": 149, "per_request": "≈0.99₽ за запрос", "popular": True},
    {"id": "pro", "name": "Профи", "requests": 400, "price": 349, "per_request": "≈0.87₽ за запрос", "popular": False},
    {"id": "unlimited", "name": "Безлимит на месяц", "requests": None, "price": 499, "per_request": "без ограничений по числу запросов", "popular": False},
]

LLM_ENDPOINTS = {
    "provod": ("https://api.provod.ai/v1/chat/completions", "PROVOD_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
    "deepseek": ("https://api.deepseek.com/chat/completions", "DEEPSEEK_API_KEY"),
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", "QWEN_API_KEY"),
}
OPENROUTER_MODELS = [
    "deepseek/deepseek-chat-v3.1:free",
    "qwen/qwen3-235b-a22b:free",
    "deepseek/deepseek-r1-distill-qwen-14b:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]
MODEL_BY_PROVIDER = {
    "provod": os.getenv("PROVOD_MODEL", "xiaomi/mimo-v2.5"),
    "deepseek": "deepseek-chat",
    "qwen": "qwen-plus",
}
LLM_TIMEOUT = 30

SEARCH_NODES = [
    "https://searx.be", "https://searx.tiekoetter.com", "https://searxng.site",
    "https://search.inetol.net", "https://priv.au", "https://search.bus-hit.me",
    "https://searx.namejeff.xyz", "https://baresearch.org", "https://opnxng.com",
    "https://search.sapti.me",
]
SEARCH_TIMEOUT = 10
PAGE_TIMEOUT = 12
MAX_SEARCH_RESULTS = 6
MAX_SOURCE_TEXT = 3500
MAX_MESSAGE_LENGTH = 5000
MAX_MEMORY = 30
MAX_CHAT_HISTORY = 300
MAX_LLM_HISTORY = 12

SETTINGS = {
    "openrouter_api_key": os.getenv("OPENROUTER_API_KEY", ""),
    "deepseek_api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    "qwen_api_key": os.getenv("QWEN_API_KEY", ""),
    "provod_api_key": os.getenv("PROVOD_API_KEY", ""),
    "provod_model": MODEL_BY_PROVIDER["provod"],
    "llm_direct_mode": os.getenv("LLM_DIRECT_MODE", "true"),
}

STOPWORDS = {
    "и", "а", "но", "или", "да", "в", "во", "на", "за", "из", "к", "ко", "с", "со", "у", "о", "об", "от", "до", "по", "для", "при", "над", "под", "не", "ни", "же", "ли", "бы", "как", "что", "это", "этот", "эта", "эти", "мне", "меня", "моя", "мой", "есть", "можно", "нужно", "надо", "ну", "вот",
}

ALIASES = {
    "жирный": ["жирный", "жирная", "жирную", "себум", "сальная", "жирность"],
    "прыщи": ["прыщи", "прыщ", "акне", "угри", "высыпания"],
    "лицо": ["лицо", "лица", "лицу", "фейс"],
    "волосы": ["волосы", "волос", "волосяной"],
    "питание": ["питание", "еда", "продукты", "рацион", "диета"],
    "сон": ["сон", "спать", "засыпать", "недосып", "бессонница"],
    "тренировки": ["тренировка", "тренировки", "спорт", "мышцы", "зал", "упражнение", "упражнения"],
    "мешки": ["мешки", "отеки", "отек", "под глазами", "глазами"],
    "темные круги": ["темные круги", "темные круги под глазами", "синяки под глазами", "круги под глазами", "синяки"],
    "морщины": ["морщины", "морщина", "складки", "старение"],
    "перхоть": ["перхоть", "перхоти", "себорейный", "шелушение", "кожа головы"],
}

KNOWLEDGE = [
    {
        "title": "Жирная кожа", "category": "skin",
        "question": "Что делать если у меня жирная кожа?",
        "answer": """Если кожа быстро становится жирной, лучше не пытаться постоянно и агрессивно обезжиривать её.\n\nБазовый уход:\n1. Мягкое очищение утром и вечером.\n2. Без необходимости не использовать агрессивное мыло и спиртовые средства.\n3. Можно рассмотреть ниацинамид или салициловую кислоту, если они подходят коже.\n4. Лёгкий увлажняющий крем.\n5. Днём — солнцезащита.\n6. Воспаления лучше не выдавливать.\n\nПри выраженном болезненном акне стоит обратиться к дерматологу.""",
        "tags": ["жирная кожа", "себум", "лицо", "акне", "прыщи"],
    },
    {
        "title": "Акне", "category": "skin",
        "question": "Как избавиться от прыщей и акне?",
        "answer": """При склонности к акне полезнее простой регулярный уход, чем большое количество средств.\n\nУтром: мягкое очищение, увлажнение, солнцезащита.\nВечером: очищение, подходящее средство против акне, увлажнение.\n\nНе стоит одновременно вводить много новых активных средств. При тяжёлом, болезненном акне или рубцах лучше получить консультацию дерматолога.""",
        "tags": ["прыщи", "акне", "угри", "кожа", "лицо"],
    },
    {
        "title": "Внешность", "category": "appearance",
        "question": "Как улучшить внешность?",
        "answer": """На внешний вид влияет сочетание привычек. Хорошая база — стабильный сон, физическая активность, сбалансированное питание, уход за кожей и волосами, гигиена, солнцезащита, а также подходящие одежда и причёска. Лучше улучшать несколько направлений постепенно, а не искать одно чудо-средство.""",
        "tags": ["внешность", "лицо", "красота", "уход"],
    },
    {
        "title": "Питание", "category": "nutrition",
        "question": "Что есть чтобы лучше выглядеть?",
        "answer": """Для внешнего вида обычно важнее сбалансированный рацион, чем экстремальная диета. Старайся регулярно получать достаточно белка, овощей и фруктов, цельных продуктов, полезных жиров и жидкости. Без конкретной причины не стоит полностью исключать целые группы продуктов.""",
        "tags": ["питание", "еда", "рацион", "диета", "внешность"],
    },
    {
        "title": "Сон", "category": "lifestyle",
        "question": "Как сон влияет на внешность?",
        "answer": """Стабильный режим сна важен для общего самочувствия. Полезно ложиться и вставать примерно в одно время, уменьшать яркий экран перед сном, не злоупотреблять кофеином поздно вечером и обеспечить комфортные условия для сна. Главное — регулярность.""",
        "tags": ["сон", "режим", "внешность", "лицо", "недосып"],
    },
    {
        "title": "Тренировки", "category": "fitness",
        "question": "Как тренироваться чтобы улучшить тело?",
        "answer": """Для физической формы можно сочетать силовые тренировки и кардио. Основные принципы: постепенно повышать нагрузку, соблюдать технику, тренировать основные мышечные группы, оставлять время на восстановление и следить за питанием и сном. Тренироваться каждый день необязательно.""",
        "tags": ["тренировки", "спорт", "мышцы", "тело", "зал"],
    },
    {
        "title": "Тёмные круги", "category": "темные круги",
        "question": "Что делать с тёмными кругами и синяками под глазами?",
        "answer": """Тёмные круги могут быть связаны с особенностями тонкой кожи, недосыпом, обезвоживанием, наследственностью, пигментацией или заметными сосудами.\n\nМожет помочь стабильный сон, достаточное употребление воды, холодный компресс утром и солнцезащита. Средства для области вокруг глаз следует подбирать с учётом чувствительности кожи. Если изменения появились резко, есть боль или выраженный отёк, стоит обратиться к врачу.""",
        "tags": ["темные круги", "синяки", "мешки", "под глазами", "глаза", "недосып"],
    },
]

knowledge_cache: list[dict[str, Any]] = []
local_credits: dict[str, dict[str, Any]] = {}

app = FastAPI(title=APP_NAME, version=APP_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])


def setting(name: str) -> str:
    return SETTINGS.get(name, "") or ""


def direct_llm_mode() -> bool:
    return setting("llm_direct_mode").strip().lower() in {"1", "true", "yes", "on"}


def normalize(text: Any) -> str:
    value = str(text or "").lower().replace("ё", "е")
    value = re.sub(r"[^а-яa-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokenize(text: str) -> list[str]:
    return [word for word in normalize(text).split() if len(word) >= 2 and word not in STOPWORDS]


def stable_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def expand_terms(text: str) -> set[str]:
    normalized = normalize(text)
    terms = set(tokenize(text))
    padded = f" {normalized} "
    for canonical, variants in ALIASES.items():
        if any(f" {normalize(v)} " in padded for v in variants):
            terms.add(canonical)
            terms.update(tokenize(" ".join(variants)))
    return terms


def jaccard(a: str, b: str) -> float:
    left, right = expand_terms(a), expand_terms(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class NeuralBrain:
    def __init__(self) -> None:
        self.words: list[str] = []
        self.categories: list[str] = []
        self.word_index: dict[str, int] = {}
        self.category_index: dict[str, int] = {}
        self.w1 = self.b1 = self.w2 = self.b2 = None
        self.ready = False

    def build(self, items: list[dict[str, Any]]) -> None:
        vocabulary: set[str] = set()
        labels: set[str] = set()
        for item in items:
            vocabulary.update(expand_terms(" ".join([item.get("question", ""), item.get("answer", ""), *item.get("tags", [])])))
            if item.get("category"):
                labels.add(item["category"])
        self.words = sorted(vocabulary)
        self.categories = sorted(labels)
        self.word_index = {word: i for i, word in enumerate(self.words)}
        self.category_index = {name: i for i, name in enumerate(self.categories)}
        if not self.words or not self.categories:
            self.ready = False
            return
        hidden = min(128, max(16, len(self.words) // 2))
        rng = np.random.default_rng(2026)
        self.w1 = rng.normal(0, math.sqrt(2 / len(self.words)), (len(self.words), hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, math.sqrt(2 / hidden), (hidden, len(self.categories)))
        self.b2 = np.zeros(len(self.categories))
        self.ready = True

    def vector(self, text: str) -> np.ndarray:
        result = np.zeros(len(self.words))
        for word in expand_terms(text):
            index = self.word_index.get(word)
            if index is not None:
                result[index] += 1
        norm = np.linalg.norm(result)
        return result / norm if norm else result

    @staticmethod
    def softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values)
        exps = np.exp(shifted)
        return exps / (np.sum(exps) + 1e-12)

    def forward(self, vector: np.ndarray):
        z1 = vector @ self.w1 + self.b1
        hidden = np.maximum(z1, 0)
        return z1, hidden, self.softmax(hidden @ self.w2 + self.b2)

    def train(self, items: list[dict[str, Any]], epochs: int = 160, rate: float = 0.035) -> dict[str, Any]:
        self.build(items)
        if not self.ready:
            return {"success": False, "epochs": 0}
        samples = []
        for item in items:
            label = self.category_index.get(item.get("category"))
            if label is not None:
                samples.append((self.vector(item.get("question", "") + " " + " ".join(item.get("tags", []))), label))
        for _ in range(epochs):
            for x, label in samples:
                z1, hidden, prediction = self.forward(x)
                target = np.zeros(len(self.categories)); target[label] = 1
                error = prediction - target
                self.w2 -= rate * np.outer(hidden, error)
                self.b2 -= rate * error
                dz = (error @ self.w2.T) * (z1 > 0)
                self.w1 -= rate * np.outer(x, dz)
                self.b1 -= rate * dz
        return {"success": True, "epochs": epochs, "samples": len(samples), "vocabulary": len(self.words), "categories": len(self.categories)}

    def predict(self, text: str):
        if not self.ready:
            return None, 0.0
        vector = self.vector(text)
        if not np.any(vector):
            return None, 0.0
        _, _, output = self.forward(vector)
        index = int(np.argmax(output))
        return self.categories[index], float(output[index])


brain = NeuralBrain()


def supabase_request(method: str, table: str, data: Optional[dict] = None, params: Optional[dict] = None, prefer: Optional[str] = None):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return []
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    if params:
        try:
            url += "?" + urlencode(params, doseq=True)
        except Exception:
            return []
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json", "Prefer": prefer or "return=representation"}
    try:
        response = requests.request(method, url, headers=headers, json=data, timeout=20)
    except requests.RequestException as exc:
        print("Supabase request failed:", repr(exc))
        return []
    if response.status_code >= 400:
        print("Supabase returned", response.status_code, response.text[:500])
        return []
    if not response.text:
        return []
    try:
        return response.json()
    except ValueError:
        return []


def attach_ids(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in items:
        copy = dict(item)
        copy.setdefault("id", stable_hash(copy.get("title", "") + copy.get("question", "")))
        result.append(copy)
    return result


def load_knowledge() -> None:
    global knowledge_cache
    if SUPABASE_URL and SUPABASE_KEY:
        rows = supabase_request("GET", "knowledge", params={"select": "*", "approved": "eq.true", "order": "created_at.desc"})
        knowledge_cache = rows or attach_ids(KNOWLEDGE)
    else:
        knowledge_cache = attach_ids(KNOWLEDGE)
    brain.train(knowledge_cache)


def search_local_knowledge(query: str) -> list[tuple[float, dict[str, Any]]]:
    predicted, confidence = brain.predict(query)
    scored = []
    qterms = expand_terms(query)
    for item in knowledge_cache:
        question, title = item.get("question", ""), item.get("title", "")
        tags = " ".join(item.get("tags", []))
        score = jaccard(query, question) * 0.5 + jaccard(query, tags) * 0.25 + jaccard(query, title) * 0.15
        if predicted and item.get("category") == predicted and score >= 0.08:
            score += confidence * 0.15
        combined = normalize(question + " " + title + " " + tags)
        score += min(0.15, sum(0.03 for word in qterms if len(word) >= 4 and word in combined))
        scored.append((score, item))
    return [entry for entry in sorted(scored, key=lambda x: x[0], reverse=True) if entry[0] >= 0.10][:5]


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def valid_http_url(value: str) -> bool:
    try:
        return urlparse(value).scheme in {"http", "https"}
    except Exception:
        return False


def parse_search_payload(payload: dict, limit: int) -> list[dict[str, str]]:
    output, seen = [], set()
    for row in payload.get("results", []) if isinstance(payload, dict) else []:
        if not isinstance(row, dict):
            continue
        title = clean_text(row.get("title"))
        url = str(row.get("url") or row.get("link") or "")
        snippet = clean_text(row.get("content") or row.get("snippet"))
        if not title or not valid_http_url(url) or url in seen:
            continue
        seen.add(url)
        output.append({"title": title[:250], "url": url, "snippet": snippet[:1500], "source": "searxng"})
        if len(output) >= limit:
            break
    return output


def searxng_search(query: str, limit: int = MAX_SEARCH_RESULTS) -> list[dict[str, str]]:
    if not query.strip():
        return []
    headers = {"User-Agent": "Mozilla/5.0 ASCEND-AI/4", "Accept": "application/json", "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.7"}
    for host in SEARCH_NODES:
        for attempt in range(2):
            try:
                response = requests.get(host.rstrip("/") + "/search", params={"q": query, "format": "json", "language": "ru-RU", "safesearch": "1", "categories": "general"}, headers=headers, timeout=SEARCH_TIMEOUT)
                if response.status_code == 429 and attempt == 0:
                    time.sleep(1.5)
                    continue
                if response.status_code >= 400 or "json" not in response.headers.get("content-type", "").lower():
                    break
                results = parse_search_payload(response.json(), limit)
                if results:
                    return results
            except Exception:
                break
    return []


def duckduckgo_search(query: str, limit: int = MAX_SEARCH_RESULTS) -> list[dict[str, str]]:
    if not query.strip():
        return []
    try:
        response = requests.post("https://html.duckduckgo.com/html/", data={"q": query, "kl": "ru-ru"}, headers={"User-Agent": "Mozilla/5.0 ASCEND-AI/4"}, timeout=SEARCH_TIMEOUT)
        if response.status_code >= 400:
            return []
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception:
        return []
    result, seen = [], set()
    for block in soup.select(".result"):
        link = block.select_one("a.result__a")
        if not link:
            continue
        url = link.get("href", "")
        title = clean_text(link.get_text(" ", strip=True))
        snippet_node = block.select_one(".result__snippet")
        snippet = clean_text(snippet_node.get_text(" ", strip=True) if snippet_node else "")
        if title and valid_http_url(url) and url not in seen:
            seen.add(url); result.append({"title": title[:250], "url": url, "snippet": snippet[:1500], "source": "duckduckgo"})
        if len(result) >= limit:
            break
    return result


def web_search_with_fallback(query: str, limit: int = MAX_SEARCH_RESULTS):
    for name, function in (("searxng", searxng_search), ("duckduckgo", duckduckgo_search)):
        try:
            results = function(query, limit)
        except Exception:
            results = []
        if results:
            return results, name
    return [], None


def fetch_page_text(url: str) -> str:
    if not valid_http_url(url):
        return ""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ASCEND-AI/4.0)", "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.7"}, timeout=PAGE_TIMEOUT, allow_redirects=True)
        if response.status_code >= 400 or "text/html" not in response.headers.get("content-type", "").lower():
            return ""
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form"]):
            tag.decompose()
        return clean_text(soup.get_text(" ", strip=True))[:MAX_SOURCE_TEXT]
    except Exception:
        return ""


def collect_web_information(query: str) -> list[dict[str, str]]:
    results, _ = web_search_with_fallback(query)
    enriched = []
    for item in results:
        text = fetch_page_text(item["url"])
        if not text and not item.get("snippet"):
            continue
        enriched.append({**item, "page_text": text})
    return enriched


def save_web_sources(session_id: str, query: str, results: list[dict[str, str]]) -> None:
    for item in results:
        supabase_request("POST", "web_sources", {"session_id": session_id, "query": query, "title": item.get("title", ""), "url": item.get("url", ""), "snippet": item.get("snippet", ""), "page_text": item.get("page_text", ""), "source": item.get("source", "web")})


def build_web_context(results: list[dict[str, str]]) -> str:
    blocks = []
    for i, item in enumerate(results, 1):
        text = item.get("page_text") or item.get("snippet") or ""
        if text:
            blocks.append(f"ИСТОЧНИК {i}\nНазвание: {item.get('title','')}\nURL: {item.get('url','')}\nИнформация:\n{text}")
    return "\n\n".join(blocks)


def clean_web_text(results: list[dict[str, str]]) -> str:
    return "\n\n".join(clean_text(x.get("page_text") or x.get("snippet")) for x in results if x.get("page_text") or x.get("snippet"))


def split_sentences(text: str) -> list[str]:
    return [clean_text(x) for x in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if len(clean_text(x)) > 20]


def rank_sentences(query: str, text: str, limit: int = 8) -> list[str]:
    qwords = expand_terms(query)
    scored = []
    for sentence in split_sentences(text):
        words = expand_terms(sentence)
        overlap = len(qwords & words)
        if overlap:
            scored.append((overlap / math.sqrt(max(1, len(words))), sentence))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [sentence for _, sentence in scored[:limit]]


def llm_available() -> bool:
    return any(setting(name) for name in ("openrouter_api_key", "deepseek_api_key", "qwen_api_key", "provod_api_key"))


def request_completion(url: str, key: str, model: str, messages: list[dict[str, str]], extra_headers: Optional[dict] = None):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    return requests.post(url, headers=headers, json={"model": model, "messages": messages, "temperature": 0.4, "max_tokens": 900}, timeout=LLM_TIMEOUT)


def extract_completion(response) -> Optional[str]:
    if response is None or response.status_code >= 400:
        return None
    try:
        content = response.json()["choices"][0]["message"]["content"]
        return str(content).strip() or None
    except Exception:
        return None


def try_provod(messages):
    key = setting("provod_api_key")
    if not key:
        return None
    try:
        return extract_completion(request_completion(LLM_ENDPOINTS["provod"][0], key, setting("provod_model"), messages))
    except Exception:
        return None


def try_openrouter(messages):
    key = setting("openrouter_api_key")
    if not key:
        return None
    for model in OPENROUTER_MODELS:
        try:
            answer = extract_completion(request_completion(LLM_ENDPOINTS["openrouter"][0], key, model, messages, {"HTTP-Referer": "https://ascend-ai.local", "X-Title": APP_NAME}))
            if answer:
                return answer
        except Exception:
            continue
    return None


def try_direct(provider: str, messages):
    setting_name = LLM_ENDPOINTS[provider][1].lower()
    key = setting(setting_name)
    if not key:
        return None
    try:
        return extract_completion(request_completion(LLM_ENDPOINTS[provider][0], key, MODEL_BY_PROVIDER[provider], messages))
    except Exception:
        return None


def call_llm(system_prompt: str, user_prompt: str, history: Optional[list[dict]] = None) -> Optional[str]:
    messages = [{"role": "system", "content": system_prompt}]
    for item in (history or [])[-MAX_LLM_HISTORY:]:
        if item.get("role") in {"user", "assistant"} and str(item.get("content", "")).strip():
            messages.append({"role": item["role"], "content": str(item["content"]).strip()})
    messages.append({"role": "user", "content": user_prompt})
    for provider in ("provod", "openrouter", "deepseek", "qwen"):
        answer = try_provod(messages) if provider == "provod" else try_openrouter(messages) if provider == "openrouter" else try_direct(provider, messages)
        if answer:
            return answer
    return None


def assistant_prompt() -> str:
    return "Ты дружелюбный русскоязычный ассистент по уходу за собой: кожа, внешность, питание, сон и тренировки. Пиши понятно и по делу. Не выдумывай медицинские сведения; при сомнениях советуй обратиться к специалисту. Не вставляй URL в основной текст ответа."


def llm_answer_from_local(query, knowledge_answer, web_results, history=None):
    extra = build_web_context(web_results) if web_results else ""
    prompt = f"Вопрос: {query}\n\nБаза знаний:\n{knowledge_answer}"
    if extra:
        prompt += f"\n\nАктуальные материалы:\n{extra}\nИспользуй их только если они действительно относятся к вопросу и не противоречат базе."
    return call_llm(assistant_prompt(), prompt + "\n\nСформулируй естественный связный ответ на русском.", history)


def llm_answer_from_web(query, web_results, history=None):
    context = build_web_context(web_results)
    if not context:
        return None
    return call_llm(assistant_prompt(), f"Вопрос пользователя:\n{query}\n\nНайденная информация:\n{context}\n\nСделай полезный связный ответ, опираясь на предоставленные данные.", history)


def llm_answer_general(query, history=None):
    return call_llm(assistant_prompt() + " Веб-проверки сейчас нет. В конце коротко укажи, что ответ дан без сверки со свежими интернет-источниками.", f"Вопрос пользователя: {query}", history)


def fallback_web_answer(query, results):
    text = clean_web_text(results)
    if not text:
        return ""
    sentences = rank_sentences(query, text) or split_sentences(text)[:5]
    if not sentences:
        return ""
    return "🌐 Нашёл материалы по вопросу.\n\n" + "\n".join(f"• {x}" for x in sentences) + "\n\n⚠️ Для важных решений проверь первоисточники."


def generate_response(query, memory, local_results, web_results):
    if local_results and local_results[0][0] >= 0.18:
        answer = local_results[0][1].get("answer", "").strip()
        if llm_available():
            generated = llm_answer_from_local(query, answer, web_results, memory)
            if generated:
                return generated
        if web_results:
            additions = rank_sentences(query, clean_web_text(web_results), 4)
            if additions:
                answer += "\n\n🌐 Дополнение из актуального поиска:\n" + "\n".join("• " + x for x in additions)
        return answer
    if web_results:
        generated = llm_answer_from_web(query, web_results, memory) if llm_available() else None
        return generated or fallback_web_answer(query, web_results)
    if llm_available():
        generated = llm_answer_general(query, memory)
        if generated:
            return generated
    return "Не удалось получить ответ прямо сейчас. Попробуй повторить запрос немного позже или сформулировать его подробнее."


def save_message(session_id, role, content):
    rows = supabase_request("POST", "chat_messages", {"session_id": session_id, "role": role, "content": content})
    return rows[0] if rows else None


def get_memory(session_id):
    rows = supabase_request("GET", "chat_messages", params={"select": "role,content,created_at", "session_id": f"eq.{session_id}", "order": "created_at.desc", "limit": str(MAX_MEMORY)})
    rows.reverse()
    return rows


def get_chat_messages(session_id, limit=MAX_CHAT_HISTORY):
    return supabase_request("GET", "chat_messages", params={"select": "role,content,created_at", "session_id": f"eq.{session_id}", "order": "created_at.asc", "limit": str(limit)}) or []


def make_chat_title(message):
    text = clean_text(message)
    return text if len(text) <= 42 else text[:42].rstrip() + "…"


def create_chat_session(session_id, device_id, title="Новый чат"):
    supabase_request("POST", "chat_sessions", {"session_id": session_id, "device_id": device_id or "", "title": title})


def list_chat_sessions(device_id):
    if not device_id:
        return []
    return supabase_request("GET", "chat_sessions", params={"select": "session_id,title,created_at,updated_at", "device_id": f"eq.{device_id}", "order": "updated_at.desc", "limit": "100"}) or []


def touch_chat_session(session_id, title=None):
    payload = {"updated_at": datetime.now(timezone.utc).isoformat()}
    if title:
        payload["title"] = title
    supabase_request("PATCH", "chat_sessions", payload, params={"session_id": f"eq.{session_id}"})


def delete_chat_session(session_id):
    supabase_request("DELETE", "chat_sessions", params={"session_id": f"eq.{session_id}"})
    supabase_request("DELETE", "chat_messages", params={"session_id": f"eq.{session_id}"})


def client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")
    return hashlib.sha256(f"ascend-credits:{ip or 'unknown'}".encode()).hexdigest()[:40]


def get_credit_state(key):
    rows = supabase_request("GET", "user_credits", params={"select": "*", "key": f"eq.{key}"})
    if rows:
        return rows[0]
    return dict(local_credits.get(key, {"key": key, "credits": 0, "free_used": False}))


def save_credit_state(key, credits, free_used):
    state = {"key": key, "credits": int(credits), "free_used": bool(free_used)}
    if SUPABASE_URL and SUPABASE_KEY:
        supabase_request("POST", "user_credits", state, params={"on_conflict": "key"}, prefer="resolution=merge-duplicates,return=representation")
    else:
        local_credits[key] = state


def consume_access(key):
    state = get_credit_state(key)
    credits = int(state.get("credits") or 0)
    used = bool(state.get("free_used"))
    if not used:
        save_credit_state(key, credits, True)
        return {"allowed": True, "mode": "free", "credits": credits}
    if credits > 0:
        credits -= 1
        save_credit_state(key, credits, True)
        return {"allowed": True, "mode": "paid", "credits": credits}
    return {"allowed": False, "mode": "blocked", "credits": 0}


def save_training_log(question, answer, category, source):
    supabase_request("POST", "training_log", {"question": question, "answer": answer, "category": category, "source": source, "approved": True})


class ChatRequest(BaseModel):
    session_id: str
    message: str


class NewChatBody(BaseModel):
    device_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: Optional[str] = None
    rating: int
    comment: Optional[str] = ""


class CreditTopUp(BaseModel):
    ip: str
    credits: int


GREETINGS = {"привет", "здравствуй", "здравствуйте", "приветик", "хай", "хеллоу", "хелло", "йо", "ку", "здарова", "здорово"}
FAREWELLS = {"пока", "прощай", "досвидания", "бывай", "увидимся"}
THANKS = {"спасибо", "благодарю", "спс", "сенкс", "thanks"}
HOW_ARE_YOU = {"как дела", "как ты", "как жизнь", "как оно", "че как", "что нового"}


def detect_small_talk(message):
    normalized = normalize(message)
    for phrase in HOW_ARE_YOU:
        if phrase in normalized:
            return "Спасибо, у меня всё хорошо 🙂 Чем могу помочь?"
    words = normalized.split()
    if not words or len(words) > 4:
        return None
    word_set = set(words)
    if word_set & GREETINGS and word_set <= GREETINGS | {"как", "дела", "там"}:
        return random.choice(["Привет! С чем помочь?", "Привет 👋 Что хочешь узнать?"])
    if word_set & FAREWELLS and word_set <= FAREWELLS:
        return "Пока! Возвращайся, если появятся вопросы 🙂"
    if word_set & THANKS and word_set <= THANKS:
        return "Пожалуйста! Обращайся."
    return None


@app.exception_handler(Exception)
async def on_unhandled(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": f"Внутренняя ошибка сервера: {type(exc).__name__}"})


@app.exception_handler(StarletteHTTPException)
async def on_http_error(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def on_validation(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": "Некорректные данные запроса.", "errors": exc.errors()})


@app.post("/api/chat")
def chat(data: ChatRequest, request: Request):
    message = data.message.strip()
    if not message:
        raise HTTPException(400, "Пустой запрос.")
    if len(message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(400, "Сообщение слишком длинное.")
    memory = get_memory(data.session_id)
    fresh = not memory
    save_message(data.session_id, "user", message)
    touch_chat_session(data.session_id, make_chat_title(message) if fresh else None)
    small_talk = detect_small_talk(message)
    if small_talk:
        saved = save_message(data.session_id, "assistant", small_talk)
        return {"answer": small_talk, "sources": [], "knowledge_found": False, "web_found": False, "memory_used": len(memory), "message_id": saved.get("id") if saved else None, "credits": {"mode": "free", "remaining": None}}
    access = consume_access(client_key(request))
    if not access["allowed"]:
        raise HTTPException(402, "Бесплатный запрос уже использован, а баланс пуст. Пополни баланс через раздел «Инфо».")
    local_results = search_local_knowledge(message)
    web_results = [] if direct_llm_mode() and llm_available() else collect_web_information(message)
    if web_results:
        save_web_sources(data.session_id, message, web_results)
    answer = generate_response(message, memory, local_results, web_results)
    saved = save_message(data.session_id, "assistant", answer)
    category = local_results[0][1].get("category") if local_results else "web"
    save_training_log(message, answer, category, "search")
    return {"answer": answer, "sources": [{"title": x.get("title", ""), "url": x.get("url", "")} for x in web_results], "knowledge_found": bool(local_results), "web_found": bool(web_results), "memory_used": len(memory), "message_id": saved.get("id") if saved else None, "credits": {"mode": access["mode"], "remaining": access["credits"]}}


@app.post("/api/chats/new")
def new_chat(data: NewChatBody):
    session_id = secrets.token_hex(16)
    create_chat_session(session_id, data.device_id)
    return {"session_id": session_id, "title": "Новый чат"}


@app.get("/api/chats")
def chats(device_id: str = ""):
    return list_chat_sessions(device_id)


@app.get("/api/chats/{session_id}/messages")
def messages(session_id: str):
    return get_chat_messages(session_id)


@app.delete("/api/chats/{session_id}")
def remove_chat(session_id: str):
    delete_chat_session(session_id)
    return {"success": True}


@app.get("/api/credits")
def credits(request: Request):
    state = get_credit_state(client_key(request))
    return {"free_used": bool(state.get("free_used")), "credits": int(state.get("credits") or 0)}


@app.get("/api/pricing")
def pricing():
    return PLANS


# Служебное начисление сохраняется как API-only операция. В пользовательском
# интерфейсе нет страницы, кнопки или режима администратора.
def check_manual_password(request: Request):
    provided = request.headers.get("X-Admin-Password", "")
    if not ADMIN_PASSWORD or not provided or not secrets.compare_digest(provided, ADMIN_PASSWORD):
        raise HTTPException(401, "Нет доступа.")


@app.post("/api/manual/credits")
def manual_credit(request: Request, data: CreditTopUp):
    check_manual_password(request)
    ip = data.ip.strip()
    if not ip or data.credits <= 0:
        raise HTTPException(400, "Укажите корректный IP и положительное количество запросов.")
    key = hashlib.sha256(f"ascend-credits:{ip}".encode()).hexdigest()[:40]
    current = get_credit_state(key)
    total = int(current.get("credits") or 0) + data.credits
    save_credit_state(key, total, True)
    return {"success": True, "key": key, "credits": total}


@app.post("/api/feedback")
def feedback(data: FeedbackRequest):
    if not 1 <= data.rating <= 5:
        raise HTTPException(400, "Оценка должна быть от 1 до 5.")
    supabase_request("POST", "ai_feedback", {"session_id": data.session_id, "message_id": data.message_id, "rating": data.rating, "comment": data.comment or ""})
    return {"success": True}


HTML = r'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ASCEND AI</title>
<style>
:root{--bg:#07070b;--panel:#111117;--panel2:#181820;--line:#292933;--text:#f4f4f7;--muted:#9696a2;--gold:#f5d45d;--orange:#ff9d58;--pink:#ff6bd4;--purple:#887bff;--bad:#ff7070}*{box-sizing:border-box}html,body{height:100%;margin:0}body{background:var(--bg);color:var(--text);font-family:Inter,-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;overflow:hidden}button,input,textarea{font:inherit;color:inherit}button{cursor:pointer}.app{height:100%;display:flex;position:relative}.glow{position:fixed;inset:0;overflow:hidden;pointer-events:none}.orb{position:absolute;border-radius:50%;filter:blur(95px);opacity:.24}.o1{width:520px;height:520px;left:-180px;top:-170px;background:var(--gold)}.o2{width:470px;height:470px;right:-170px;bottom:-180px;background:var(--orange)}.o3{width:360px;height:360px;left:60%;top:35%;background:var(--purple);opacity:.14}.side{position:fixed;z-index:20;inset:0 auto 0 0;width:290px;background:rgba(14,14,20,.96);backdrop-filter:blur(18px);border-right:1px solid var(--line);padding:16px 13px;display:flex;flex-direction:column;transform:translateX(-100%);transition:.3s}.side.open{transform:none}.overlay{position:fixed;z-index:15;inset:0;background:#0009;opacity:0;pointer-events:none;transition:.25s}.overlay.open{opacity:1;pointer-events:auto}.new{border:0;border-radius:14px;padding:13px;font-weight:800;background:linear-gradient(135deg,var(--gold),var(--orange),var(--pink));color:#19150f}.chats{flex:1;overflow:auto;margin-top:12px}.chat{padding:11px 10px;border:1px solid transparent;border-radius:11px;display:flex;justify-content:space-between;gap:8px;color:var(--muted)}.chat:hover,.chat.active{background:var(--panel2);color:var(--text);border-color:var(--line)}.chat span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.trash{background:none;border:0;color:var(--muted);opacity:.25}.chat:hover .trash{opacity:1}.foot{border-top:1px solid var(--line);padding-top:10px}.linkbtn{width:100%;border:0;background:none;text-align:left;color:var(--muted);padding:9px 5px;border-radius:9px}.linkbtn:hover{background:var(--panel);color:var(--text)}.main{min-width:0;width:100%;display:flex;flex-direction:column;z-index:2}.bar{height:64px;flex:none;display:flex;align-items:center;justify-content:space-between;padding:0 18px;border-bottom:1px solid var(--line);background:#09090dbb;backdrop-filter:blur(14px)}.left,.right{display:flex;align-items:center;gap:9px}.icon{width:40px;height:40px;border:1px solid var(--line);border-radius:12px;background:var(--panel);display:grid;place-items:center}.brand{font-weight:900;font-size:18px;letter-spacing:.3px}.brand b{background:linear-gradient(90deg,var(--purple),var(--gold),var(--orange),var(--pink));-webkit-background-clip:text;background-clip:text;color:transparent}.pill{border:1px solid var(--line);background:var(--panel);border-radius:999px;padding:8px 12px;font-size:12px;color:var(--muted)}.warn{color:var(--orange);border-color:#ff9d5866}.messages{flex:1;overflow:auto;padding:28px 0 10px}.inner{width:min(760px,100%);margin:auto;padding:0 18px}.msg{display:flex;margin:0 0 18px;animation:in .3s ease}.msg.user{justify-content:flex-end}.avatar{width:32px;height:32px;flex:none;border-radius:10px;margin-right:9px;display:grid;place-items:center;background:linear-gradient(135deg,var(--gold),var(--orange),var(--pink));color:#16120d}.bubble{max-width:82%;padding:14px 17px;border-radius:18px;line-height:1.62;white-space:pre-wrap;font-size:15px}.ai .bubble{background:var(--panel);border:1px solid var(--line);border-top-left-radius:6px}.user .bubble{background:linear-gradient(135deg,var(--gold),var(--orange));color:#18130d;font-weight:600;border-top-right-radius:6px}.sources{margin:-8px 0 20px 41px}.src{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:9px 11px;margin-top:6px}.src a{font-size:12px;color:var(--gold);word-break:break-all;text-decoration:none}.typing{display:flex;align-items:center;margin-bottom:18px}.dots{display:flex;gap:6px;padding:15px 18px;background:var(--panel);border:1px solid var(--line);border-radius:18px;border-top-left-radius:6px}.dots i{width:7px;height:7px;border-radius:50%;background:var(--gold);animation:b .9s infinite}.dots i:nth-child(2){animation-delay:.15s;background:var(--orange)}.dots i:nth-child(3){animation-delay:.3s;background:var(--purple)}@keyframes b{0%,100%{transform:translateY(0);opacity:.4}50%{transform:translateY(-5px);opacity:1}}@keyframes in{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}.composer{padding:13px 18px 20px}.compose{width:min(760px,100%);margin:auto;display:flex;align-items:end;gap:9px;padding:8px 8px 8px 17px;border:1px solid var(--line);background:var(--panel);border-radius:19px}.compose:focus-within{box-shadow:0 0 0 4px #f5d45d18;border-color:#454553}.compose textarea{flex:1;resize:none;background:none;border:0;outline:0;min-height:24px;max-height:150px;padding:10px 0}.send{width:44px;height:44px;border:0;border-radius:13px;background:linear-gradient(135deg,var(--gold),var(--orange));color:#19140e;font-weight:900}.hint{text-align:center;color:#555560;font-size:11px;margin-top:9px}.hint button{border:0;background:none;color:#777783;text-decoration:underline;font-size:11px}.modal{position:fixed;z-index:50;inset:0;display:grid;place-items:center;padding:18px;background:#000b;opacity:0;pointer-events:none;transition:.25s}.modal.open{opacity:1;pointer-events:auto}.card{width:min(460px,100%);max-height:88vh;overflow:auto;background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:25px;position:relative;box-shadow:0 25px 80px #0009}.close{position:absolute;right:15px;top:13px;background:none;border:0;color:var(--muted);font-size:20px}.card h2{margin:0 0 8px}.card p{color:var(--muted);line-height:1.6;font-size:14px}.docs{display:grid;gap:9px;margin:17px 0}.doc{display:flex;justify-content:space-between;padding:12px 14px;border:1px solid var(--line);border-radius:12px;background:var(--panel2);text-decoration:none;color:var(--text)}.primary{width:100%;border:0;border-radius:13px;padding:13px;font-weight:800;background:linear-gradient(135deg,var(--gold),var(--orange))}.plans{display:grid;gap:10px;margin:16px 0}.plan{padding:15px;border:1px solid var(--line);border-radius:15px;background:var(--panel2);position:relative}.plan.pop{border-color:var(--gold)}.badge{position:absolute;right:13px;top:-9px;background:var(--gold);color:#17130d;border-radius:99px;padding:3px 9px;font-size:10px;font-weight:800}.pt{display:flex;justify-content:space-between;font-weight:800}.meta{font-size:12px;color:var(--muted);margin-top:4px}@media(min-width:900px){.side{transform:none;position:relative}.overlay{display:none}.main{margin-left:0}.menu{display:none}}@media(max-width:600px){.pill{display:none}.bubble{max-width:90%}}
</style></head><body>
<div class="glow"><div class="orb o1"></div><div class="orb o2"></div><div class="orb o3"></div></div>
<div id="overlay" class="overlay" onclick="closeSide()"></div>
<aside id="side" class="side"><button class="new" onclick="newChat()">＋ Новый чат</button><div id="chatList" class="chats"><div style="color:#777;padding:12px">Загрузка…</div></div><div class="foot"><button class="linkbtn" onclick="openInfo()">ℹ️ Информация, тарифы и поддержка</button></div></aside>
<main class="main"><header class="bar"><div class="left"><button class="icon menu" onclick="openSide()">☰</button><div class="brand"><b>ASCEND</b> AI</div></div><div class="right"><button id="balance" class="pill" onclick="openPricing()">Проверка…</button><button class="icon" onclick="openInfo()">ⓘ</button></div></header>
<section id="messages" class="messages"><div id="inner" class="inner"><div class="msg ai"><div class="avatar">🧠</div><div class="bubble">Привет! Я ASCEND AI.\n\nПомогу с вопросами о коже, внешности, питании, сне и тренировках. Если ответа нет в моей базе, могу использовать актуальный поиск.\n\nПервый запрос — бесплатно 🎁</div></div></div></section>
<div class="composer"><div class="compose"><textarea id="input" rows="1" placeholder="Напиши свой вопрос…"></textarea><button id="send" class="send" onclick="sendMessage()">➤</button></div><div class="hint">Ответы носят справочный характер · <button onclick="openInfo()">Тарифы и поддержка</button></div></div></main></div>
<div id="welcome" class="modal"><div class="card"><h2>Добро пожаловать 👋</h2><p>ASCEND AI — ассистент по уходу за собой. Перед началом ознакомься с документами.</p><div class="docs"><a class="doc" href="__PRIVACY__" target="_blank">Политика конфиденциальности ↗</a><a class="doc" href="__TERMS__" target="_blank">Пользовательское соглашение ↗</a><a class="doc" href="__SUPPORT__" target="_blank">Поддержка в Telegram ↗</a></div><label style="display:flex;gap:8px;color:#999;font-size:13px;margin:15px 0"><input id="accept" type="checkbox"> Я принимаю документы</label><button class="primary" onclick="acceptTerms()">Продолжить</button></div></div>
<div id="info" class="modal"><div class="card"><button class="close" onclick="closeModal('info')">×</button><h2>Информация</h2><p>Документы, поддержка и доступные тарифы.</p><div class="docs"><a class="doc" href="__PRIVACY__" target="_blank">Политика конфиденциальности ↗</a><a class="doc" href="__TERMS__" target="_blank">Пользовательское соглашение ↗</a><a class="doc" href="__SUPPORT__" target="_blank">Поддержка: __HANDLE__ ↗</a></div><button class="primary" onclick="closeModal('info');openPricing()">💳 Тарифы</button></div></div>
<div id="pricing" class="modal"><div class="card"><button class="close" onclick="closeModal('pricing')">×</button><h2>Тарифы</h2><p>Первый запрос бесплатный. Дальше можно выбрать пакет запросов.</p><div id="plans" class="plans">Загрузка…</div><a class="doc" href="__SUPPORT__" target="_blank">Оплата / вопросы — написать в поддержку ↗</a></div></div>
<div id="paywall" class="modal"><div class="card"><button class="close" onclick="closeModal('paywall')">×</button><h2>Баланс исчерпан</h2><p>Бесплатный запрос уже использован. Чтобы продолжить, открой тарифы и выбери пакет.</p><button class="primary" onclick="closeModal('paywall');openPricing()">Посмотреть тарифы</button></div></div>
<script>
const DKEY='ascend_device_id_v2', CKEY='ascend_chat_v2', TKEY='ascend_terms_v2';
let deviceId=localStorage.getItem(DKEY)||crypto.randomUUID();localStorage.setItem(DKEY,deviceId);let currentChat=localStorage.getItem(CKEY);let busy=false;
const $=id=>document.getElementById(id);
function openModal(id){$(id).classList.add('open')}function closeModal(id){$(id).classList.remove('open')}function openInfo(){openModal('info')}function openPricing(){openModal('pricing');loadPlans()}function openSide(){$('side').classList.add('open');$('overlay').classList.add('open');loadChats()}function closeSide(){$('side').classList.remove('open');$('overlay').classList.remove('open')}
function acceptTerms(){if(!$('accept').checked)return alert('Подтверди ознакомление с документами.');localStorage.setItem(TKEY,'1');closeModal('welcome')}
function esc(v){return String(v??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#039;')}
async function loadChats(){try{let r=await fetch('/api/chats?device_id='+encodeURIComponent(deviceId));let data=await r.json();$('chatList').innerHTML='';if(!data.length){$('chatList').innerHTML='<div style="color:#777;padding:12px">Сохранённых чатов пока нет</div>';return}data.forEach(c=>{let el=document.createElement('div');el.className='chat '+(c.session_id===currentChat?'active':'');el.innerHTML='<span>'+esc(c.title||'Новый чат')+'</span><button class="trash">×</button>';el.onclick=()=>switchChat(c.session_id);el.querySelector('.trash').onclick=e=>{e.stopPropagation();deleteChat(c.session_id)};$('chatList').appendChild(el)})}catch(e){$('chatList').innerHTML='<div style="color:#777;padding:12px">Не удалось загрузить чаты</div>'}}
async function newChat(){try{let r=await fetch('/api/chats/new',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({device_id:deviceId})});let d=await r.json();currentChat=d.session_id;localStorage.setItem(CKEY,currentChat);renderEmpty();closeSide();loadChats()}catch(e){console.error(e)}}
async function switchChat(id){currentChat=id;localStorage.setItem(CKEY,id);closeSide();await loadMessages();loadChats()}
async function deleteChat(id){if(!confirm('Удалить этот чат?'))return;await fetch('/api/chats/'+id,{method:'DELETE'});if(id===currentChat){await newChat()}else loadChats()}
function renderEmpty(){$('inner').innerHTML='<div class="msg ai"><div class="avatar">🧠</div><div class="bubble">Новый чат начат. О чём хочешь спросить?</div></div>'}
async function loadMessages(){if(!currentChat)return;try{let r=await fetch('/api/chats/'+currentChat+'/messages');let data=await r.json();$('inner').innerHTML='';if(!data.length)return renderEmpty();data.forEach(m=>addMessage(m.role==='user'?'user':'ai',m.content))}catch(e){renderEmpty()}}
function bottom(){$('messages').scrollTop=$('messages').scrollHeight}
function addMessage(role,text){let w=document.createElement('div');w.className='msg '+role;w.innerHTML=role==='user'?'<div class="bubble"></div>':'<div class="avatar">🧠</div><div class="bubble"></div>';w.querySelector('.bubble').textContent=text;$('inner').appendChild(w);bottom()}
function addSources(src){if(!src?.length)return;let w=document.createElement('div');w.className='sources';w.innerHTML='<div style="color:#888;font-size:12px">🌐 Источники</div>'+src.map(s=>'<div class="src"><div>'+esc(s.title||s.url)+'</div><a href="'+esc(s.url)+'" target="_blank" rel="noopener">'+esc(s.url)+'</a></div>').join('');$('inner').appendChild(w);bottom()}
function typing(show){let old=document.getElementById('typing');if(!show){old?.remove();return}if(old)return;let w=document.createElement('div');w.id='typing';w.className='typing';w.innerHTML='<div class="avatar">🧠</div><div class="dots"><i></i><i></i><i></i></div>';$('inner').appendChild(w);bottom()}
function resize(){let el=$('input');el.style.height='auto';el.style.height=Math.min(150,el.scrollHeight)+'px'}
$('input').addEventListener('input',resize);$('input').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMessage()}});
async function sendMessage(){if(busy)return;let input=$('input'),msg=input.value.trim();if(!msg)return;if(msg.length>5000)return alert('Сообщение слишком длинное.');if(!currentChat)await newChat();busy=true;$('send').disabled=true;addMessage('user',msg);input.value='';resize();typing(true);try{let r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:currentChat,message:msg})});let d=await r.json();typing(false);if(r.status===402){addMessage('ai',d.detail||'Баланс исчерпан.');openModal('paywall')}else if(!r.ok)addMessage('ai',d.detail||'Ошибка сервера.');else{addMessage('ai',d.answer||'Пустой ответ.');addSources(d.sources);updateBalance(d.credits);loadChats()}}catch(e){typing(false);addMessage('ai','Ошибка соединения с сервером.')}busy=false;$('send').disabled=false}
async function updateBalance(c){if(!c)return loadBalance();if(c.mode==='free'){$('balance').textContent='🎁 Бесплатный запрос использован';$('balance').classList.remove('warn')}else if(c.remaining>0){$('balance').textContent='💳 Баланс: '+c.remaining;$('balance').classList.remove('warn')}else{$('balance').textContent='⚠️ Пополнить баланс';$('balance').classList.add('warn')}}
async function loadBalance(){try{let r=await fetch('/api/credits'),d=await r.json();if(!d.free_used){$('balance').textContent='🎁 Первый запрос бесплатно';$('balance').classList.remove('warn')}else if(d.credits>0){$('balance').textContent='💳 Баланс: '+d.credits;$('balance').classList.remove('warn')}else{$('balance').textContent='⚠️ Пополнить баланс';$('balance').classList.add('warn')}}catch(e){$('balance').textContent='💳 Тарифы'}}
async function loadPlans(){try{let r=await fetch('/api/pricing'),data=await r.json();$('plans').innerHTML=data.map(p=>'<div class="plan '+(p.popular?'pop':'')+'">'+(p.popular?'<div class="badge">Популярный</div>':'')+'<div class="pt"><span>'+esc(p.name)+'</span><span>'+p.price+'₽</span></div><div class="meta">'+(p.requests?p.requests+' запросов':'Без ограничений')+' · '+esc(p.per_request)+'</div></div>').join('')}catch(e){$('plans').textContent='Не удалось загрузить тарифы'}}
(async function init(){if(!localStorage.getItem(TKEY))openModal('welcome');loadBalance();if(!currentChat)await newChat();else await loadMessages()})();
</script></body></html>'''

HTML = HTML.replace("__PRIVACY__", PRIVACY_URL).replace("__TERMS__", TERMS_URL).replace("__SUPPORT__", SUPPORT_URL).replace("__HANDLE__", SUPPORT_HANDLE)

LEGAL_STYLE = "<style>body{margin:0;background:#09090c;color:#f5f5f7;font-family:-apple-system,Inter,Arial;line-height:1.7}.wrap{max-width:720px;margin:auto;padding:40px 20px 80px}h1{font-size:28px}h2{color:#f5d45d;margin-top:30px}p,li{color:#c6c6cd;font-size:14px}a{color:#f5d45d}.back{color:#92929d;text-decoration:none;font-size:13px}.card{background:#141419;border:1px solid #292933;border-radius:16px;padding:18px;margin:14px 0}.amount{font-size:22px;font-weight:900;color:#f5d45d}.meta{color:#92929d;font-size:12px}</style>"


@app.get("/")
async def index():
    return HTMLResponse(HTML)


@app.get("/privacy")
async def privacy():
    return RedirectResponse(PRIVACY_URL)


@app.get("/terms")
async def terms():
    return RedirectResponse(TERMS_URL)


@app.get("/contacts", response_class=HTMLResponse)
async def contacts():
    return f"<!doctype html><html lang='ru'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Поддержка — {APP_NAME}</title>{LEGAL_STYLE}</head><body><div class='wrap'><a class='back' href='/'>← Назад</a><h1>Поддержка</h1><p>По вопросам работы сервиса, оплаты или удаления данных напиши нам.</p><p><a href='{SUPPORT_URL}' target='_blank'>Написать в Telegram: {SUPPORT_HANDLE}</a></p><p>Среднее время ответа — до 24 часов.</p><p><a href='{PRIVACY_URL}' target='_blank'>Политика конфиденциальности</a> · <a href='{TERMS_URL}' target='_blank'>Пользовательское соглашение</a></p></div></body></html>"


@app.get("/pricing", response_class=HTMLResponse)
async def pricing_page():
    cards = "".join(f"<div class='card'><div class='amount'>{p['price']} ₽</div><div>{p['name']} — {p['requests'] if p['requests'] else 'без ограничений'} запросов</div><div class='meta'>{p['per_request']}</div></div>" for p in PLANS)
    return f"<!doctype html><html lang='ru'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Тарифы — {APP_NAME}</title>{LEGAL_STYLE}</head><body><div class='wrap'><a class='back' href='/'>← Назад</a><h1>Тарифы</h1><p>Первый запрос бесплатный. Дальше доступны пакеты запросов.</p>{cards}<p>Для оплаты и вопросов обратись в <a href='/contacts'>поддержку</a>.</p></div></body></html>"


@app.get("/health")
async def health():
    return {"status": "ok", "version": APP_VERSION, "brain_ready": brain.ready, "knowledge": len(knowledge_cache), "llm_enabled": llm_available(), "supabase_configured": bool(SUPABASE_URL and SUPABASE_KEY)}


@app.on_event("startup")
async def startup():
    try:
        load_knowledge()
    except Exception:
        traceback.print_exc()
    print(f"{APP_NAME} {APP_VERSION}: knowledge={len(knowledge_cache)}, brain={brain.ready}, llm={llm_available()}, supabase={bool(SUPABASE_URL and SUPABASE_KEY)}")

