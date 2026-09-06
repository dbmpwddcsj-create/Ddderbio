import os
import re
import json
import math
import hashlib
import secrets
import time
import random

from typing import Optional, List
from urllib.parse import urlencode, urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup

import traceback

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "ASCEND AI"

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "CHANGE_THIS_PASSWORD")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY", "")


# ============================================================
# LLM SETTINGS 
# ============================================================

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
QWEN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
PROVOD_URL = "https://api.provod.ai/v1/chat/completions"

LLM_TIMEOUT = 30

OPENROUTER_FREE_MODELS = [
    "deepseek/deepseek-chat-v3.1:free",
    "qwen/qwen3-235b-a22b:free",
    "deepseek/deepseek-r1-distill-qwen-14b:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]

DEEPSEEK_MODEL = "deepseek-chat"
QWEN_MODEL = "qwen-plus"

PROVOD_DEFAULT_MODEL = os.getenv("PROVOD_MODEL", "xiaomi/mimo-v2.5")

_DEFAULT_SETTINGS = {
    "openrouter_api_key": os.getenv("OPENROUTER_API_KEY", ""),
    "deepseek_api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    "qwen_api_key": os.getenv("QWEN_API_KEY", ""),
    "provod_api_key": os.getenv("PROVOD_API_KEY", ""),
    "provod_model": PROVOD_DEFAULT_MODEL,
    "llm_direct_mode": os.getenv("LLM_DIRECT_MODE", "true"),
}

runtime_settings = dict(_DEFAULT_SETTINGS)

def is_llm_direct_mode():
    return get_setting("llm_direct_mode").strip().lower() in ("1", "true", "yes", "on")

def get_setting(key):
    return runtime_settings.get(key, "") or ""

def mask_key(value):
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "…" + value[-4:]


# ============================================================
# SEARCH ENGINES 
# ============================================================

SEARXNG_INSTANCES = [
    "https://searx.be",
    "https://searx.tiekoetter.com",
    "https://searxng.site",
    "https://search.inetol.net",
    "https://priv.au",
    "https://search.bus-hit.me",
    "https://searx.namejeff.xyz",
    "https://baresearch.org",
    "https://opnxng.com",
    "https://search.sapti.me",
]

SEARXNG_TIMEOUT = 10
SEARXNG_MAX_RETRIES_PER_INSTANCE = 2
SEARXNG_RETRY_BACKOFF_BASE = 1.5


# ============================================================
# LIMITS & TRACKING
# ============================================================

MAX_MEMORY = 30
MAX_SEARCH_RESULTS = 6
MAX_SOURCE_TEXT = 3500
MAX_MESSAGE_LENGTH = 5000
PAGE_TIMEOUT = 12

# Simple in-memory tracking for free requests limit
DEVICE_USAGE = {}

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title=APP_NAME, version="2.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    print("UNHANDLED EXCEPTION on", request.method, request.url.path, flush=True)
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": f"Внутренняя ошибка сервера: {type(exc).__name__}"},
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": "Некорректные данные запроса.", "errors": exc.errors()},
    )


# ============================================================
# SUPABASE REST CLIENT
# ============================================================

def supabase_request(method, table, data=None, params=None):
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        return []
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    if params:
        try:
            url += "?" + urlencode(params, doseq=True)
        except Exception:
            return []
    body = None
    if data is not None:
        try:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        except Exception:
            return []
    headers = {
        "apikey": SUPABASE_SECRET_KEY,
        "Authorization": "Bearer " + SUPABASE_SECRET_KEY,
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    try:
        response = requests.request(method=method, url=url, headers=headers, data=body, timeout=20)
    except Exception:
        return []
    if response.status_code >= 400 or not response.text:
        return []
    try:
        return response.json()
    except Exception:
        return []


# ============================================================
# TEXT UTILS
# ============================================================

RUSSIAN_STOPWORDS = {
    "и", "а", "но", "или", "да", "в", "во", "на", "за", "из", "к", "ко",
    "с", "со", "у", "о", "об", "от", "до", "по", "для", "при", "над",
    "под", "не", "ни", "же", "ли", "бы", "как", "что", "это", "этот",
    "эта", "эти", "мне", "меня", "моя", "мой", "есть", "можно", "нужно",
    "надо", "ну", "вот",
}

def normalize(text):
    if not text:
        return ""
    text = str(text).lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text):
    return [w for w in normalize(text).split() if w not in RUSSIAN_STOPWORDS and len(w) >= 2]

def stable_hash(text):
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


# ============================================================
# SYNONYMS
# ============================================================

SYNONYMS = {
    "жирный": ["жирный", "жирная", "жирную", "сальная", "сальный", "себум", "жирность"],
    "прыщи": ["прыщи", "прыщ", "акне", "угри", "угрей", "высыпания"],
    "лицо": ["лицо", "лица", "лицу", "фейс"],
    "волосы": ["волосы", "волос", "волосяной"],
    "питание": ["питание", "еда", "продукты", "рацион", "диета"],
    "сон": ["сон", "спать", "засыпать", "недосып", "бессонница"],
    "тренировки": ["тренировка", "тренировки", "спорт", "мышцы", "зал", "качаться", "упражнение", "упражнения"],
    "мешки": ["мешки", "отеки", "отек", "под глазами", "глазами"],
    "темные круги": ["темные круги", "темные круги под глазами", "синяки под глазами", "круги под глазами", "синяки"],
    "морщины": ["морщины", "морщина", "складки", "старение"],
    "перхоть": ["перхоть", "перхотью", "перхоти", "себорейный", "шелушение", "шелушится", "кожа головы", "шелушение кожи головы", "шелушится кожа головы"],
}

def expand_query(text):
    normalized_text = normalize(text)
    words = tokenize(text)
    expanded = set(words)
    for canonical, variants in SYNONYMS.items():
        if any(normalize(v) in normalized_text for v in variants if normalize(v)):
            expanded.add(canonical)
            for variant in variants:
                expanded.update(tokenize(variant))
    return list(expanded)


# ============================================================
# NEURAL BRAIN
# ============================================================

class NeuralBrain:
    def __init__(self):
        self.vocabulary = []
        self.word_index = {}
        self.categories = []
        self.category_index = {}
        self.W1, self.b1, self.W2, self.b2 = None, None, None, None
        self.ready = False

    def build(self, knowledge):
        vocabulary = set()
        categories = set()
        for item in knowledge:
            text = f"{item.get('question', '')} {item.get('answer', '')} {' '.join(item.get('tags', []))}"
            vocabulary.update(expand_query(text))
            if item.get("category"):
                categories.add(item.get("category"))

        self.vocabulary = sorted(vocabulary)
        self.word_index = {w: i for i, w in enumerate(self.vocabulary)}
        self.categories = sorted(categories)
        self.category_index = {c: i for i, c in enumerate(self.categories)}

        if not self.vocabulary or not self.categories:
            self.ready = False
            return

        input_size, output_size = len(self.vocabulary), len(self.categories)
        hidden_size = min(128, max(16, input_size // 2))

        rng = np.random.default_rng(42)
        self.W1 = rng.normal(0, np.sqrt(2 / input_size), (input_size, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.W2 = rng.normal(0, np.sqrt(2 / hidden_size), (hidden_size, output_size))
        self.b2 = np.zeros(output_size)
        self.ready = True

    def vectorize(self, text):
        vector = np.zeros(len(self.vocabulary))
        for word in expand_query(text):
            if word in self.word_index:
                vector[self.word_index[word]] += 1
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    @staticmethod
    def relu(x): return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        x = x - np.max(x)
        exp = np.exp(x)
        return exp / (np.sum(exp) + 1e-9)

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        h = self.relu(z1)
        z2 = h @ self.W2 + self.b2
        return z1, h, self.softmax(z2)

    def train(self, knowledge, epochs=180, learning_rate=0.035):
        self.build(knowledge)
        if not self.ready: return {"success": False}

        dataset = []
        for item in knowledge:
            text = item.get("question", "") + " " + " ".join(item.get("tags", []))
            if item.get("category") in self.category_index:
                dataset.append((self.vectorize(text), self.category_index[item["category"]]))

        if not dataset: return {"success": False}

        for _ in range(epochs):
            for x, label in dataset:
                z1, h, prediction = self.forward(x)
                target = np.zeros(len(self.categories))
                target[label] = 1
                error = prediction - target
                self.W2 -= learning_rate * np.outer(h, error)
                self.b2 -= learning_rate * error
                dh = error @ self.W2.T
                dz1 = dh * (z1 > 0)
                self.W1 -= learning_rate * np.outer(x, dz1)
                self.b1 -= learning_rate * dz1
        return {"success": True}

    def predict(self, text):
        if not self.ready: return None, 0.0
        x = self.vectorize(text)
        if not np.any(x): return None, 0.0
        _, _, output = self.forward(x)
        index = int(np.argmax(output))
        return self.categories[index], float(output[index])


# ============================================================
# DEFAULT KNOWLEDGE
# ============================================================

DEFAULT_KNOWLEDGE = [
    {
        "title": "Жирная кожа", "category": "skin",
        "question": "Что делать если у меня жирная кожа?",
        "answer": "Умывай лицо мягким очищающим средством утром и вечером. Не используй агрессивное мыло. Используй лёгкий увлажняющий крем.",
        "tags": ["жирная кожа", "себум", "кожа", "лицо", "акне", "прыщи"],
    },
]

knowledge_cache = []
brain = NeuralBrain()

def load_knowledge():
    global knowledge_cache
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        knowledge_cache = DEFAULT_KNOWLEDGE.copy()
    else:
        rows = supabase_request("GET", "knowledge", params={"select": "*", "approved": "eq.true"})
        knowledge_cache = rows if rows else DEFAULT_KNOWLEDGE.copy()
    brain.train(knowledge_cache)


def similarity(a, b):
    a_words, b_words = set(expand_query(a)), set(expand_query(b))
    if not a_words or not b_words: return 0.0
    return len(a_words & b_words) / max(1, len(a_words | b_words))

def search_local_knowledge(query):
    predicted_category, confidence = brain.predict(query)
    results = []
    query_expanded = set(expand_query(query))

    for item in knowledge_cache:
        question = item.get("question", "")
        tags = " ".join(item.get("tags", []))
        title = item.get("title", "")

        score = max(similarity(query, question), similarity(query, tags), similarity(query, title)) * 0.50
        if predicted_category and item.get("category") == predicted_category and score >= 0.04:
            score += confidence * 0.15

        for word in query_expanded:
            if len(word) >= 4 and word in normalize(question + " " + title + " " + tags):
                score += 0.03

        results.append((score, item))

    results.sort(key=lambda x: x[0], reverse=True)
    return [item for item in results if item[0] >= 0.10][:5]


# ============================================================
# WEB HELPERS
# ============================================================

def clean_text(text): return re.sub(r"\s+", " ", text or "").strip()
def valid_http_url(url): return urlparse(url).scheme in {"http", "https"} if url else False

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}

def searxng_search(query, limit=MAX_SEARCH_RESULTS):
    query = query.strip()
    if not query: return []
    for instance in SEARXNG_INSTANCES:
        url = instance.rstrip("/") + "/search"
        try:
            response = requests.get(url, params={"q": query, "format": "json", "language": "ru-RU"}, headers=BROWSER_HEADERS, timeout=SEARXNG_TIMEOUT)
            if response.status_code == 200 and "json" in response.headers.get("content-type", "").lower():
                payload = response.json()
                results = []
                for raw in payload.get("results", []):
                    title, url_value, snippet = clean_text(raw.get("title")), str(raw.get("url") or raw.get("link")), clean_text(raw.get("content") or raw.get("snippet"))
                    if title and valid_http_url(url_value):
                        results.append({"title": title[:250], "url": url_value, "snippet": snippet[:1500], "source": "searxng"})
                        if len(results) >= limit: break
                if results: return results
        except Exception: continue
    return []

def duckduckgo_html_search(query, limit=MAX_SEARCH_RESULTS):
    if not query.strip(): return []
    try:
        response = requests.post("https://html.duckduckgo.com/html/", data={"q": query, "kl": "ru-ru"}, headers=BROWSER_HEADERS, timeout=SEARXNG_TIMEOUT)
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for result_div in soup.select(".result"):
            link = result_div.select_one("a.result__a")
            snippet_tag = result_div.select_one(".result__snippet")
            if link:
                title, href = clean_text(link.get_text(" ", strip=True)), link.get("href", "")
                snippet = clean_text(snippet_tag.get_text(" ", strip=True)) if snippet_tag else ""
                if title and valid_http_url(href):
                    results.append({"title": title[:250], "url": href, "snippet": snippet[:1500], "source": "duckduckgo"})
                    if len(results) >= limit: break
        return results
    except Exception: return []

SEARCH_ENGINES = [("searxng", searxng_search), ("duckduckgo", duckduckgo_html_search)]

def web_search_with_fallback(query, limit=MAX_SEARCH_RESULTS):
    for name, engine_fn in SEARCH_ENGINES:
        try:
            results = engine_fn(query, limit)
            if results: return results, name
        except Exception: continue
    return [], None

def fetch_page_text(url):
    if not valid_http_url(url): return ""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=PAGE_TIMEOUT)
        if response.status_code < 400 and "text/html" in response.headers.get("content-type", "").lower():
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]): tag.decompose()
            return clean_text(soup.get_text(" ", strip=True))[:MAX_SOURCE_TEXT]
    except Exception: pass
    return ""

def collect_web_information(query):
    search_results, _ = web_search_with_fallback(query)
    enriched = []
    for result in search_results:
        page_text = fetch_page_text(result["url"])
        if page_text or clean_text(result.get("snippet", "")):
            enriched.append({**result, "page_text": page_text})
    return enriched

def build_web_context(results):
    return "\n".join([f"Название: {r.get('title')}\nТекст: {r.get('page_text') or r.get('snippet')}\n" for r in results if r.get('page_text') or r.get('snippet')])

def clean_web_text(results):
    return "\n\n".join([clean_text(r.get('page_text') or r.get('snippet')) for r in results if r.get('page_text') or r.get('snippet')])

def split_sentences(text):
    return [clean_text(x) for x in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if len(clean_text(x)) > 20]

def rank_sentences(query, text, limit=8):
    sentences = split_sentences(text)
    qwords = set(expand_query(query))
    scored = []
    for sentence in sentences:
        swords = set(expand_query(sentence))
        overlap = len(qwords & swords)
        if overlap:
            scored.append((overlap / math.sqrt(max(1, len(swords))), sentence))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:limit]]

# ============================================================
# SETTINGS PERSISTENCE
# ============================================================
def load_settings():
    if SUPABASE_URL and SUPABASE_SECRET_KEY:
        rows = supabase_request("GET", "app_settings", params={"select": "key,value"})
        for row in rows or []:
            if row.get("key") in runtime_settings and row.get("value"):
                runtime_settings[row["key"]] = row["value"]

# ============================================================
# LLM ANSWER SYNTHESIS
# ============================================================
def llm_available():
    return any(get_setting(k) for k in ("openrouter_api_key", "deepseek_api_key", "qwen_api_key", "provod_api_key"))

def _chat_completion_request(url, api_key, model, messages):
    return requests.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                         json={"model": model, "messages": messages, "temperature": 0.4, "max_tokens": 900}, timeout=LLM_TIMEOUT)

def call_llm(system_prompt, user_prompt):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    
    # Try Provod
    if get_setting("provod_api_key"):
        try:
            res = _chat_completion_request(PROVOD_URL, get_setting("provod_api_key"), get_setting("provod_model"), messages)
            if res.status_code == 200: return res.json()["choices"][0]["message"]["content"].strip()
        except Exception: pass
        
    # Try OpenRouter
    if get_setting("openrouter_api_key"):
        for model in OPENROUTER_FREE_MODELS:
            try:
                res = _chat_completion_request(OPENROUTER_URL, get_setting("openrouter_api_key"), model, messages)
                if res.status_code == 200: return res.json()["choices"][0]["message"]["content"].strip()
            except Exception: continue

    # Try DeepSeek
    if get_setting("deepseek_api_key"):
        try:
            res = _chat_completion_request(DEEPSEEK_URL, get_setting("deepseek_api_key"), DEEPSEEK_MODEL, messages)
            if res.status_code == 200: return res.json()["choices"][0]["message"]["content"].strip()
        except Exception: pass

    return None

def llm_answer_from_local(query, knowledge_answer, web_results):
    web_context = build_web_context(web_results) if web_results else ""
    sys_p = "Ты дружелюбный ассистент по уходу за собой. Отвечай простым текстом на русском. Без ссылок."
    usr_p = f"Вопрос: {query}\nБаза знаний:\n{knowledge_answer}\n"
    if web_context: usr_p += f"Новое из веба:\n{web_context}\n"
    return call_llm(sys_p, usr_p)

def fallback_web_answer(query, web_results):
    web_text = clean_web_text(web_results)
    if not web_text: return ""
    sentences = rank_sentences(query, web_text, limit=6) or split_sentences(web_text)[:5]
    if not sentences: return ""
    return "🌐 Информация из интернета:\n\n" + "\n".join("• " + s for s in sentences)

def generate_response(query, local_results, web_results):
    best_item = local_results[0][1] if local_results else None
    
    if best_item:
        ans = best_item.get("answer", "")
        if llm_available():
            llm_ans = llm_answer_from_local(query, ans, web_results)
            if llm_ans: return llm_ans
        return ans + ("\n\n🌐 Дополнение из веба: " + fallback_web_answer(query, web_results) if web_results else "")

    if web_results:
        if llm_available():
            llm_ans = call_llm("Ты дружелюбный ассистент. Ответь по существу без ссылок на основе веба.", f"Вопрос: {query}\nВеб:\n{build_web_context(web_results)}")
            if llm_ans: return llm_ans
        return fallback_web_answer(query, web_results)

    if llm_available():
        llm_ans = call_llm("Ты ассистент. Отвечай из своих знаний.", f"Вопрос: {query}")
        if llm_ans: return llm_ans

    return "Не смог найти информацию прямо сейчас. Попробуй позже."

# ============================================================
# SMALL TALK
# ============================================================
def detect_small_talk(message):
    normalized = normalize(message)
    if any(phrase in normalized for phrase in {"как дела", "как жизнь", "че как"}):
        return "Всё отлично! Готов помочь. Что тебя интересует?"
    words = set(normalized.split())
    if len(words) <= 3 and words & {"привет", "здравствуй", "хай", "ку"}:
        return "Привет 👋 С чем помочь сегодня? (кожа, питание, тренировки...)"
    return None

# ============================================================
# MODELS & ROUTES
# ============================================================

class ChatMessageItem(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: str
    message: str
    device_id: str = "unknown"
    history: Optional[List[ChatMessageItem]] = []

HTML = r"""
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>ASCEND AI</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
    --bg-main: #0B0C10;
    --bg-secondary: #1F2833;
    --accent: #45A29E;
    --accent-hover: #66FCF1;
    --text-main: #C5C6C7;
    --text-bright: #FFFFFF;
    --bubble-user: linear-gradient(135deg, #1f2833 0%, #2b3a4a 100%);
    --bubble-ai: rgba(255, 255, 255, 0.03);
}
* { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
body { background: var(--bg-main); color: var(--text-main); height: 100vh; overflow: hidden; display: flex; }

/* LAYOUT */
.app-container { display: flex; width: 100%; height: 100%; }

/* SIDEBAR */
.sidebar { 
    width: 280px; background: var(--bg-secondary); display: flex; flex-direction: column; 
    transition: transform 0.3s ease; z-index: 100; border-right: 1px solid rgba(255,255,255,0.05);
}
.sidebar-header { padding: 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.05); }
.sidebar-header h2 { font-size: 16px; color: var(--text-bright); font-weight: 600; }
.new-chat-btn { background: transparent; border: 1px solid var(--accent); color: var(--accent); width: 32px; height: 32px; border-radius: 8px; font-size: 20px; display: flex; align-items: center; justify-content: center; cursor: pointer; transition: all 0.2s; }
.new-chat-btn:hover { background: var(--accent); color: var(--bg-main); }
.chat-list { flex: 1; overflow-y: auto; padding: 15px; }
.chat-item { display: flex; justify-content: space-between; align-items: center; padding: 12px 15px; background: rgba(255,255,255,0.02); border-radius: 10px; margin-bottom: 8px; cursor: pointer; transition: background 0.2s; font-size: 14px; color: var(--text-bright); }
.chat-item:hover { background: rgba(255,255,255,0.06); }
.chat-item.active { background: rgba(69, 162, 158, 0.15); border: 1px solid rgba(69, 162, 158, 0.3); }
.chat-item-title { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 170px; }
.delete-chat { color: #ff5555; background: none; border: none; font-size: 16px; cursor: pointer; opacity: 0.5; padding: 5px; }
.delete-chat:hover { opacity: 1; }
.sidebar-footer { padding: 20px; border-top: 1px solid rgba(255,255,255,0.05); font-size: 12px; display: flex; flex-direction: column; gap: 8px; }
.sidebar-footer a { color: var(--text-main); text-decoration: none; opacity: 0.7; transition: opacity 0.2s; }
.sidebar-footer a:hover { opacity: 1; color: var(--accent-hover); }
.mekbuda { font-size: 10px; opacity: 0.3; text-align: right; margin-top: 10px; font-weight: bold; letter-spacing: 1px; }

/* MAIN CONTENT */
.main-content { flex: 1; display: flex; flex-direction: column; position: relative; }
.header { height: 60px; display: flex; align-items: center; justify-content: space-between; padding: 0 20px; border-bottom: 1px solid rgba(255,255,255,0.05); background: rgba(11, 12, 16, 0.8); backdrop-filter: blur(10px); z-index: 50; }
.header-left { display: flex; align-items: center; gap: 15px; }
.hamburger { background: none; border: none; color: var(--text-bright); font-size: 24px; cursor: pointer; display: none; }
.logo { font-size: 18px; font-weight: 700; color: var(--text-bright); letter-spacing: 1px; }
.logo span { color: var(--accent); }
.pay-btn { background: var(--accent); color: var(--bg-main); border: none; padding: 8px 16px; border-radius: 20px; font-weight: 600; font-size: 13px; cursor: pointer; transition: 0.2s; }
.pay-btn:hover { background: var(--accent-hover); box-shadow: 0 0 10px rgba(102,252,241,0.4); }

/* CHAT AREA */
.chat-area { flex: 1; overflow-y: auto; padding: 30px 20px; scroll-behavior: smooth; }
.chat-container { max-width: 800px; margin: 0 auto; display: flex; flex-direction: column; gap: 24px; }
.message { display: flex; animation: slideUp 0.4s ease-out forwards; opacity: 0; transform: translateY(15px); }
.message.user { justify-content: flex-end; }
.bubble { max-width: 85%; padding: 14px 18px; border-radius: 18px; line-height: 1.5; font-size: 15px; word-wrap: break-word; white-space: pre-wrap;}
.message.user .bubble { background: var(--bubble-user); color: var(--text-bright); border-bottom-right-radius: 4px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
.message.ai .bubble { background: var(--bubble-ai); border: 1px solid rgba(255,255,255,0.08); border-bottom-left-radius: 4px; color: var(--text-bright); backdrop-filter: blur(10px); }

/* TYPING ANIMATION */
.typing-indicator { display: flex; gap: 5px; padding: 15px 20px; align-items: center; width: fit-content; background: var(--bubble-ai); border-radius: 18px; border-bottom-left-radius: 4px; border: 1px solid rgba(255,255,255,0.08); animation: fadeIn 0.3s forwards; }
.typing-dot { width: 6px; height: 6px; background: var(--accent); border-radius: 50%; animation: bounce 1.4s infinite ease-in-out both; }
.typing-dot:nth-child(1) { animation-delay: -0.32s; }
.typing-dot:nth-child(2) { animation-delay: -0.16s; }

/* INPUT AREA */
.input-area { padding: 20px; max-width: 840px; margin: 0 auto; width: 100%; background: var(--bg-main); z-index: 10; }
.input-wrapper { display: flex; gap: 12px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1); border-radius: 24px; padding: 8px 12px; align-items: flex-end; transition: border-color 0.3s; }
.input-wrapper:focus-within { border-color: var(--accent); box-shadow: 0 0 15px rgba(69,162,158,0.1); }
textarea { flex: 1; background: transparent; border: none; color: var(--text-bright); font-size: 15px; resize: none; max-height: 120px; padding: 10px 5px; outline: none; line-height: 1.4; }
.send-btn { background: var(--accent); color: var(--bg-main); border: none; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; cursor: pointer; transition: 0.2s; flex-shrink: 0; margin-bottom: 2px; }
.send-btn:hover { background: var(--accent-hover); transform: scale(1.05); }
.send-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
.send-btn svg { width: 18px; height: 18px; fill: currentColor; margin-left: 2px; }

/* MODALS */
.modal-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); backdrop-filter: blur(5px); z-index: 1000; display: flex; align-items: center; justify-content: center; opacity: 1; transition: opacity 0.3s; }
.modal-overlay.hidden { opacity: 0; pointer-events: none; }
.modal-content { background: var(--bg-secondary); padding: 30px; border-radius: 20px; max-width: 420px; width: 90%; border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 15px 30px rgba(0,0,0,0.5); transform: translateY(0); transition: 0.3s; }
.modal-overlay.hidden .modal-content { transform: translateY(20px); }
.modal-content h2 { color: var(--text-bright); margin-bottom: 15px; font-size: 20px; text-align: center; }
.modal-content p { font-size: 14px; margin-bottom: 20px; line-height: 1.5; text-align: center; opacity: 0.8; }
.modal-content ul { list-style: none; margin-bottom: 25px; display: flex; flex-direction: column; gap: 10px; }
.modal-content ul a { color: var(--accent); text-decoration: none; display: block; padding: 12px; background: rgba(0,0,0,0.2); border-radius: 10px; text-align: center; font-weight: 500; border: 1px solid transparent; transition: 0.2s; }
.modal-content ul a:hover { border-color: var(--accent); background: rgba(69,162,158,0.1); }
.modal-btn { width: 100%; padding: 14px; background: var(--accent); color: var(--bg-main); border: none; border-radius: 12px; font-size: 16px; font-weight: 600; cursor: pointer; transition: 0.2s; }
.modal-btn:hover { background: var(--accent-hover); }
.close-modal { background: transparent; color: var(--text-main); border: 1px solid rgba(255,255,255,0.2); margin-top: 10px; }

/* TARIFFS */
.tariffs { display: flex; flex-direction: column; gap: 10px; margin-bottom: 20px; }
.tariff { padding: 15px; background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; cursor: pointer; text-align: center; transition: 0.2s; font-weight: 500; color: var(--text-bright); }
.tariff:hover, .tariff.selected { border-color: var(--accent); background: rgba(69,162,158,0.1); transform: translateY(-2px); }

/* ANIMATIONS */
@keyframes slideUp { from { opacity: 0; transform: translateY(15px); } to { opacity: 1; transform: translateY(0); } }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }

/* RESPONSIVE */
@media (max-width: 768px) {
    .sidebar { position: absolute; left: -280px; height: 100%; box-shadow: 5px 0 15px rgba(0,0,0,0.5); }
    .sidebar.open { transform: translateX(280px); }
    .hamburger { display: block; }
    .bubble { max-width: 92%; }
}
</style>
</head>
<body>

<!-- FIRST VISIT MODAL -->
<div id="firstVisitModal" class="modal-overlay hidden">
    <div class="modal-content">
        <h2>Добро пожаловать в ASCEND AI</h2>
        <p>Перед началом использования, пожалуйста, ознакомьтесь с правилами платформы:</p>
        <ul>
            <li><a href="https://telegra.ph/Politika-konfidencialnosti-09-06-116" target="_blank">Политика конфиденциальности</a></li>
            <li><a href="https://telegra.ph/Polzovatelskoe-soglashenie-09-06-54" target="_blank">Пользовательское соглашение</a></li>
            <li><a href="https://t.me/lovnff" target="_blank">Контакты поддержки</a></li>
        </ul>
        <button class="modal-btn" onclick="acceptTerms()">Я согласен</button>
    </div>
</div>

<!-- PAY MODAL -->
<div id="payModal" class="modal-overlay hidden">
    <div class="modal-content">
        <h2>Лимит исчерпан</h2>
        <p>Ваш бесплатный запрос использован. Выберите тариф для продолжения:</p>
        <div class="tariffs">
            <div class="tariff" onclick="selectTariff(this)">50 запросов / 50 ₽</div>
            <div class="tariff" onclick="selectTariff(this)">150 запросов / 140 ₽</div>
            <div class="tariff" onclick="selectTariff(this)">500 запросов / 450 ₽</div>
            <div class="tariff" onclick="selectTariff(this)">Безлимит (1 мес) / 990 ₽</div>
        </div>
        <button class="modal-btn" onclick="processPayment()">Оплатить (СБП)</button>
        <button class="modal-btn close-modal" onclick="closePayModal()">Закрыть</button>
    </div>
</div>

<div class="app-container">
    <aside class="sidebar" id="sidebar">
        <div class="sidebar-header">
            <h2>Мои чаты</h2>
            <button class="new-chat-btn" onclick="createNewChat()">+</button>
        </div>
        <div class="chat-list" id="chatList"></div>
        <div class="sidebar-footer">
            <a href="https://telegra.ph/Politika-konfidencialnosti-09-06-116" target="_blank">Политика конфиденциальности</a>
            <a href="https://telegra.ph/Polzovatelskoe-soglashenie-09-06-54" target="_blank">Пользовательское соглашение</a>
            <a href="https://t.me/lovnff" target="_blank">Поддержка</a>
            <div class="mekbuda">mekbuda</div>
        </div>
    </aside>

    <main class="main-content">
        <header class="header">
            <div class="header-left">
                <button class="hamburger" onclick="toggleSidebar()">☰</button>
                <div class="logo">ASCEND <span>AI</span></div>
            </div>
            <button class="pay-btn" onclick="openPayModal()">Подписка</button>
        </header>

        <div class="chat-area" id="chatArea">
            <div class="chat-container" id="chatContainer">
                <!-- Messages render here -->
            </div>
        </div>

        <div class="input-area">
            <div class="input-wrapper">
                <textarea id="messageInput" placeholder="Введите ваш вопрос..." rows="1" oninput="autoResize(this)"></textarea>
                <button class="send-btn" id="sendBtn" onclick="sendMessage()">
                    <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
                </button>
            </div>
        </div>
    </main>
</div>

<script>
const DEVICE_ID_KEY = 'ascend_device_id';
const CHATS_KEY = 'ascend_chats';
const AGREED_KEY = 'ascend_agreed';

let deviceId = localStorage.getItem(DEVICE_ID_KEY);
if (!deviceId) {
    deviceId = crypto.randomUUID();
    localStorage.setItem(DEVICE_ID_KEY, deviceId);
}

let chats = JSON.parse(localStorage.getItem(CHATS_KEY)) || [];
let currentChatId = null;

// Initialization
window.onload = () => {
    if (!localStorage.getItem(AGREED_KEY)) {
        document.getElementById('firstVisitModal').classList.remove('hidden');
    }
    if (chats.length === 0) createNewChat();
    else selectChat(chats[0].id);
    renderSidebar();
};

function acceptTerms() {
    localStorage.setItem(AGREED_KEY, 'true');
    document.getElementById('firstVisitModal').classList.add('hidden');
}

function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('open');
}

function openPayModal() { document.getElementById('payModal').classList.remove('hidden'); }
function closePayModal() { document.getElementById('payModal').classList.add('hidden'); }

function selectTariff(el) {
    document.querySelectorAll('.tariff').forEach(t => t.classList.remove('selected'));
    el.classList.add('selected');
}

function processPayment() {
    alert('Оплата по СБП будет доступна в следующем обновлении. Спасибо за интерес!');
}

function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
}

function saveChats() {
    localStorage.setItem(CHATS_KEY, JSON.stringify(chats));
}

function createNewChat() {
    const id = crypto.randomUUID();
    chats.unshift({ id, title: 'Новый диалог', messages: [], date: Date.now() });
    saveChats();
    selectChat(id);
    renderSidebar();
    if(window.innerWidth <= 768) toggleSidebar();
}

function selectChat(id) {
    currentChatId = id;
    renderSidebar();
    renderMessages();
    if(window.innerWidth <= 768 && document.getElementById('sidebar').classList.contains('open')) {
        toggleSidebar();
    }
}

function deleteChat(id, e) {
    e.stopPropagation();
    chats = chats.filter(c => c.id !== id);
    saveChats();
    if (chats.length === 0) createNewChat();
    else if (currentChatId === id) selectChat(chats[0].id);
    renderSidebar();
}

function renderSidebar() {
    const list = document.getElementById('chatList');
    list.innerHTML = '';
    chats.forEach(c => {
        const div = document.createElement('div');
        div.className = `chat-item ${c.id === currentChatId ? 'active' : ''}`;
        div.onclick = () => selectChat(c.id);
        div.innerHTML = `
            <div class="chat-item-title">${escapeHtml(c.title)}</div>
            <button class="delete-chat" onclick="deleteChat('${c.id}', event)">×</button>
        `;
        list.appendChild(div);
    });
}

function renderMessages() {
    const container = document.getElementById('chatContainer');
    container.innerHTML = '';
    const chat = chats.find(c => c.id === currentChatId);
    if (!chat) return;
    
    if (chat.messages.length === 0) {
        container.innerHTML = `
            <div class="message ai" style="animation:none;opacity:1;transform:none;">
                <div class="bubble">Привет! Я ASCEND AI. 🧠<br><br>Я могу помочь с вопросами о красоте, уходе, тренировках и здоровье. Если у меня нет ответа, я найду его в интернете.<br><br>Чем могу помочь?</div>
            </div>`;
        return;
    }

    chat.messages.forEach((m, idx) => {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${m.role}`;
        msgDiv.style.animationDelay = `${idx * 0.05}s`;
        msgDiv.innerHTML = `<div class="bubble">${escapeHtml(m.content)}</div>`;
        container.appendChild(msgDiv);
    });
    scrollToBottom();
}

function scrollToBottom() {
    const area = document.getElementById('chatArea');
    area.scrollTop = area.scrollHeight;
}

function addMessageToUI(role, text) {
    const container = document.getElementById('chatContainer');
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;
    msgDiv.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
    container.appendChild(msgDiv);
    scrollToBottom();
}

function showTyping() {
    const container = document.getElementById('chatContainer');
    const div = document.createElement('div');
    div.id = 'typingIndicator';
    div.className = 'message ai';
    div.innerHTML = `
        <div class="typing-indicator">
            <div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>
        </div>`;
    container.appendChild(div);
    scrollToBottom();
}

function hideTyping() {
    const ind = document.getElementById('typingIndicator');
    if (ind) ind.remove();
}

async function sendMessage() {
    const input = document.getElementById('messageInput');
    const btn = document.getElementById('sendBtn');
    const text = input.value.trim();
    if (!text) return;

    const chat = chats.find(c => c.id === currentChatId);
    if (!chat) return;

    if (chat.messages.length === 0) {
        chat.title = text.length > 25 ? text.substring(0, 25) + '...' : text;
        renderSidebar();
    }

    chat.messages.push({ role: 'user', content: text });
    addMessageToUI('user', text);
    saveChats();
    
    input.value = '';
    input.style.height = 'auto';
    btn.disabled = true;
    showTyping();

    // Prepare history payload for context memory
    const historyPayload = chat.messages.slice(0, -1).map(m => ({ role: m.role, content: m.content }));

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: currentChatId,
                device_id: deviceId,
                message: text,
                history: historyPayload
            })
        });

        const data = await response.json();
        hideTyping();

        if (data.requires_payment) {
            chat.messages.pop(); // Revert user message locally if blocked
            renderMessages();
            openPayModal();
        } else {
            const answer = data.answer || "Ошибка получения ответа.";
            chat.messages.push({ role: 'ai', content: answer });
            addMessageToUI('ai', answer);
            saveChats();
        }
    } catch (e) {
        hideTyping();
        addMessageToUI('ai', 'Ошибка соединения с сервером.');
    }
    btn.disabled = false;
}

document.getElementById('messageInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

function escapeHtml(val) {
    return String(val || "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


@app.post("/api/chat")
async def chat(data: ChatRequest):
    message = data.message.strip()
    if not message: raise HTTPException(400, "Пустой запрос.")

    # Rate Limiting Logic (Anti-abuse, 1 free request per device)
    if data.device_id and data.device_id != "unknown":
        if DEVICE_USAGE.get(data.device_id, 0) >= 1:
            return {"requires_payment": True, "answer": ""}

    small_talk = detect_small_talk(message)
    if small_talk:
        answer = small_talk
        web_results = []
    else:
        local_results = search_local_knowledge(message)
        web_results = [] if is_llm_direct_mode() and llm_available() else collect_web_information(message)
        answer = generate_response(message, local_results, web_results)

    # Increment usage counter after successful response
    if data.device_id and data.device_id != "unknown":
        DEVICE_USAGE[data.device_id] = DEVICE_USAGE.get(data.device_id, 0) + 1

    return {
        "requires_payment": False,
        "answer": answer,
        "sources": [{"title": r.get("title", ""), "url": r.get("url", "")} for r in web_results],
    }


@app.on_event("startup")
async def startup():
    load_knowledge()
    load_settings()
    print("=" * 60, flush=True)
    print("                  ASCEND AI 2.2.0 (Modernized)", flush=True)
    print("=" * 60, flush=True)
