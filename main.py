import os
import re
import json
import math
import hashlib
import hmac
import secrets
import time
import random

from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlencode, urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup

import traceback

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "ASCEND AI"

# Пароль нужен ТОЛЬКО для одного служебного эндпоинта ручного пополнения
# баланса (/api/manual/credits) — веб-панели администратора больше нет,
# поэтому его нигде не нужно вводить в браузере. Вызывать эндпоинт нужно
# напрямую (curl/Postman) с заголовком X-Admin-Password, пока не
# подключена автоматическая оплата по СБП.
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "CHANGE_THIS_PASSWORD")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY", "")

# ------------------------------------------------------------
# Юридическая информация / поддержка / тарифы
# ------------------------------------------------------------

PRIVACY_URL = "https://telegra.ph/Politika-konfidencialnosti-09-06-116"
TERMS_URL = "https://telegra.ph/Polzovatelskoe-soglashenie-09-06-54"
SUPPORT_TELEGRAM_HANDLE = "@lovnff"
SUPPORT_TELEGRAM_URL = "https://t.me/lovnff"

# Пакеты запросов. Себестоимость одного запроса к платным/бесплатным LLM
# обычно ~2-5 копеек, поэтому цены ниже дают комфортную маржу.
PRICING_PLANS = [
    {
        "id": "start",
        "name": "Старт",
        "requests": 50,
        "price": 59,
        "per_request": "≈1.18₽ за запрос",
        "popular": False,
    },
    {
        "id": "standard",
        "name": "Стандарт",
        "requests": 150,
        "price": 149,
        "per_request": "≈0.99₽ за запрос",
        "popular": True,
    },
    {
        "id": "pro",
        "name": "Профи",
        "requests": 400,
        "price": 349,
        "per_request": "≈0.87₽ за запрос",
        "popular": False,
    },
    {
        "id": "unlimited",
        "name": "Безлимит на месяц",
        "requests": None,
        "price": 499,
        "per_request": "без ограничений по числу запросов",
        "popular": False,
    },
]


# ============================================================
# LLM SETTINGS (API-ключи задаются ТОЛЬКО через переменные окружения —
# Render → Environment. Никакой веб-формы для ввода ключей больше нет,
# поэтому они никогда не "летают" через браузер и не хранятся в БД
# в открытом виде рядом с публичным интерфейсом.)
# ============================================================
#
#   DeepSeek:  https://platform.deepseek.com  -> API Keys
#   Qwen:      https://dashscope.console.aliyun.com -> API-Key Management
#   OpenRouter (бесплатные модели): https://openrouter.ai -> Keys
#   provod.ai: личный кабинет провайдера
#
# Порядок попыток при генерации ответа: provod.ai -> OpenRouter (бесплатные
# модели) -> DeepSeek напрямую -> Qwen напрямую. Каждый уровень пропускается,
# если для него не задан ключ.
#

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

# Все ключи и режимы читаются один раз из переменных окружения при
# старте процесса. Никакого рантайм-переопределения через сайт нет —
# чтобы поменять ключ, достаточно поменять переменную окружения на
# Render и передеплоить/перезапустить сервис.
RUNTIME_SETTINGS = {
    "openrouter_api_key": os.getenv("OPENROUTER_API_KEY", ""),
    "deepseek_api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    "qwen_api_key": os.getenv("QWEN_API_KEY", ""),
    "provod_api_key": os.getenv("PROVOD_API_KEY", ""),
    "provod_model": PROVOD_DEFAULT_MODEL,
    "llm_direct_mode": os.getenv("LLM_DIRECT_MODE", "true"),
}


def is_llm_direct_mode():
    return get_setting("llm_direct_mode").strip().lower() in ("1", "true", "yes", "on")


def get_setting(key):
    return RUNTIME_SETTINGS.get(key, "") or ""


# ============================================================
# SEARCH ENGINES (multi-provider fallback chain)
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
# LIMITS
# ============================================================

MAX_MEMORY = 30
MAX_CHAT_HISTORY = 300
MAX_SEARCH_RESULTS = 6
MAX_SOURCE_TEXT = 3500
MAX_MESSAGE_LENGTH = 5000
PAGE_TIMEOUT = 12
MAX_LLM_HISTORY_MESSAGES = 12


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title=APP_NAME, version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    print("=" * 60, flush=True)
    print("UNHANDLED EXCEPTION on", request.method, request.url.path, flush=True)
    traceback.print_exc()
    print("=" * 60, flush=True)

    return JSONResponse(
        status_code=500,
        content={"detail": f"Внутренняя ошибка сервера: {type(exc).__name__}"},
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print("VALIDATION ERROR on", request.method, request.url.path, ":", exc.errors(), flush=True)
    return JSONResponse(
        status_code=422,
        content={"detail": "Некорректные данные запроса.", "errors": exc.errors()},
    )


# ============================================================
# SUPABASE REST CLIENT
# ============================================================

def supabase_request(method, table, data=None, params=None, prefer=None):
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        return []

    url = f"{SUPABASE_URL}/rest/v1/{table}"

    if params:
        try:
            url += "?" + urlencode(params, doseq=True)
        except Exception as e:
            print("SUPABASE PARAM ERROR:", repr(e))
            return []

    body = None
    if data is not None:
        try:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        except Exception as e:
            print("SUPABASE JSON ERROR:", repr(e))
            return []

    headers = {
        "apikey": SUPABASE_SECRET_KEY,
        "Authorization": "Bearer " + SUPABASE_SECRET_KEY,
        "Content-Type": "application/json",
        "Prefer": prefer or "return=representation",
    }

    try:
        response = requests.request(
            method=method, url=url, headers=headers, data=body, timeout=20
        )
    except Exception as e:
        print("SUPABASE REQUEST ERROR:", repr(e))
        return []

    if response.status_code >= 400:
        print("SUPABASE ERROR:", response.status_code, response.text[:1000])
        return []

    if not response.text:
        return []

    try:
        return response.json()
    except Exception:
        return []


# ============================================================
# TEXT
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
    text = str(text)
    text = text.lower()
    text = text.replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text):
    words = normalize(text).split()
    return [w for w in words if w not in RUSSIAN_STOPWORDS and len(w) >= 2]


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

    padded_text = f" {normalized_text} "

    for canonical, variants in SYNONYMS.items():
        found = False
        for variant in variants:
            normalized_variant = normalize(variant)
            if not normalized_variant:
                continue

            padded_variant = f" {normalized_variant} "
            if padded_variant in padded_text:
                found = True
                break

        if found:
            expanded.add(canonical)
            for variant in variants:
                for word in tokenize(variant):
                    expanded.add(word)

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
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.ready = False

    def build(self, knowledge):
        vocabulary = set()
        categories = set()

        for item in knowledge:
            text = (
                item.get("question", "")
                + " "
                + item.get("answer", "")
                + " "
                + " ".join(item.get("tags", []))
            )
            for word in expand_query(text):
                vocabulary.add(word)

            category = item.get("category")
            if category:
                categories.add(category)

        self.vocabulary = sorted(vocabulary)
        self.word_index = {w: i for i, w in enumerate(self.vocabulary)}
        self.categories = sorted(categories)
        self.category_index = {c: i for i, c in enumerate(self.categories)}

        if not self.vocabulary or not self.categories:
            self.ready = False
            return

        input_size = len(self.vocabulary)
        hidden_size = min(128, max(16, input_size // 2))
        output_size = len(self.categories)

        rng = np.random.default_rng(42)

        self.W1 = rng.normal(0, np.sqrt(2 / input_size), (input_size, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.W2 = rng.normal(0, np.sqrt(2 / hidden_size), (hidden_size, output_size))
        self.b2 = np.zeros(output_size)

        self.ready = True

    def vectorize(self, text):
        vector = np.zeros(len(self.vocabulary))
        for word in expand_query(text):
            index = self.word_index.get(word)
            if index is not None:
                vector[index] += 1

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm

        return vector

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        x = x - np.max(x)
        exp = np.exp(x)
        return exp / (np.sum(exp) + 1e-9)

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        h = self.relu(z1)
        z2 = h @ self.W2 + self.b2
        output = self.softmax(z2)
        return z1, h, output

    def train(self, knowledge, epochs=180, learning_rate=0.035):
        self.build(knowledge)

        if not self.ready:
            return {"success": False, "epochs": 0}

        dataset = []
        for item in knowledge:
            question = item.get("question", "")
            tags = item.get("tags", [])
            text = question + " " + " ".join(tags)
            vector = self.vectorize(text)

            category = item.get("category")
            label = self.category_index.get(category)
            if label is None:
                continue

            dataset.append((vector, label))

        if not dataset:
            return {"success": False, "epochs": 0}

        for _ in range(epochs):
            for x, label in dataset:
                z1, h, prediction = self.forward(x)

                target = np.zeros(len(self.categories))
                target[label] = 1

                error = prediction - target

                dW2 = np.outer(h, error)
                db2 = error

                dh = error @ self.W2.T
                dz1 = dh * (z1 > 0)

                dW1 = np.outer(x, dz1)
                db1 = dz1

                self.W2 -= learning_rate * dW2
                self.b2 -= learning_rate * db2
                self.W1 -= learning_rate * dW1
                self.b1 -= learning_rate * db1

        return {
            "success": True,
            "epochs": epochs,
            "samples": len(dataset),
            "vocabulary": len(self.vocabulary),
            "categories": len(self.categories),
        }

    def predict(self, text):
        if not self.ready:
            return None, 0.0

        x = self.vectorize(text)
        if not np.any(x):
            return None, 0.0

        _, _, output = self.forward(x)
        index = int(np.argmax(output))

        return self.categories[index], float(output[index])


# ============================================================
# DEFAULT KNOWLEDGE
# ============================================================

DEFAULT_KNOWLEDGE = [
    {
        "title": "Жирная кожа",
        "category": "skin",
        "question": "Что делать если у меня жирная кожа?",
        "answer": """
Если кожа быстро становится жирной, не стоит постоянно
и агрессивно обезжиривать её.

Базовый уход:

1. Умывай лицо мягким очищающим средством утром и вечером.
2. Не используй агрессивное мыло и спиртовые средства без необходимости.
3. Рассмотри средства с ниацинамидом или салициловой кислотой,
   если они подходят твоей коже.
4. Используй лёгкий увлажняющий крем.
5. Днём используй солнцезащитное средство.
6. Не выдавливай воспаления.

Если есть выраженное болезненное акне, лучше обратиться к дерматологу.
""",
        "tags": ["жирная кожа", "себум", "кожа", "лицо", "акне", "прыщи"],
    },
    {
        "title": "Прыщи",
        "category": "skin",
        "question": "Как избавиться от прыщей и акне?",
        "answer": """
При склонности к акне лучше выстроить простой регулярный уход.

Утром:
• мягкое очищение;
• увлажнение;
• солнцезащита.

Вечером:
• очищение;
• средство против акне, подходящее твоей коже;
• увлажнение.

Не начинай сразу несколько новых активных средств.

Если акне тяжёлое, болезненное или оставляет рубцы,
стоит обратиться к дерматологу.
""",
        "tags": ["прыщи", "акне", "угри", "кожа", "лицо"],
    },
    {
        "title": "Улучшение внешности",
        "category": "appearance",
        "question": "Как улучшить внешность?",
        "answer": """
На внешний вид влияет сразу несколько факторов.

Полезная база:

• нормальный режим сна;
• регулярная физическая активность;
• сбалансированное питание;
• уход за кожей;
• уход за волосами;
• личная гигиена;
• солнцезащита;
• подходящая одежда и причёска.

Лучше постепенно улучшать несколько направлений,
чем искать одно чудо-средство.
""",
        "tags": ["внешность", "лицо", "красота", "уход"],
    },
    {
        "title": "Питание",
        "category": "nutrition",
        "question": "Что есть чтобы лучше выглядеть?",
        "answer": """
Для внешнего вида обычно важнее сбалансированный рацион,
чем экстремальная диета.

Старайся регулярно получать:

• достаточное количество белка;
• овощи и фрукты;
• цельные продукты;
• полезные жиры;
• достаточное количество жидкости.

Не нужно исключать целые группы продуктов без конкретной причины.
""",
        "tags": ["питание", "еда", "рацион", "диета", "внешность"],
    },
    {
        "title": "Сон",
        "category": "lifestyle",
        "question": "Как сон влияет на внешность?",
        "answer": """
Стабильный режим сна важен для общего самочувствия.

Полезно:

• ложиться примерно в одно время;
• вставать примерно в одно время;
• уменьшить яркий экран перед сном;
• не употреблять много кофеина поздно вечером;
• обеспечить комфортные условия для сна.

Главное — стабильность режима.
""",
        "tags": ["сон", "режим", "внешность", "лицо", "недосып"],
    },
    {
        "title": "Тренировки",
        "category": "fitness",
        "question": "Как тренироваться чтобы улучшить тело?",
        "answer": """
Для улучшения физической формы можно сочетать силовые тренировки
и кардио.

Основные принципы:

• постепенно увеличивай нагрузку;
• соблюдай технику упражнений;
• тренируй основные мышечные группы;
• оставляй время на восстановление;
• следи за питанием и сном.

Не обязательно тренироваться каждый день.
""",
        "tags": ["тренировки", "спорт", "мышцы", "тело", "зал"],
    },
    {
        "title": "Тёмные круги и синяки под глазами",
        "category": "темные круги",
        "question": "Что делать с тёмными кругами / синяками под глазами?",
        "answer": """
Тёмные круги под глазами обычно связаны с несколькими факторами:
тонкая кожа в этой зоне, недосып, обезвоживание, наследственность,
пигментация или расширенные сосуды.

Что может помочь:

1. Наладь режим сна (7–9 часов, стабильное время отхода ко сну).
2. Пей достаточно воды в течение дня.
3. Используй крем для области вокруг глаз с кофеином,
   витамином К или ретинолом (если кожа не чувствительная).
4. Прикладывай холодный компресс на несколько минут утром.
5. Используй солнцезащитный крем — УФ усиливает пигментацию.
6. Высыпайся и ограничь соль и алкоголь вечером — это уменьшает отёки.

Если круги появились резко, сопровождаются отёком, болью
или другими симптомами — стоит показаться врачу, чтобы
исключить, например, аллергию или проблемы с носовыми пазухами.
""",
        "tags": ["темные круги", "синяки", "мешки", "под глазами", "глаза", "недосып"],
    },
]


# ============================================================
# GLOBAL STATE
# ============================================================

knowledge_cache = []
brain = NeuralBrain()

_local_credits_cache = {}


# ============================================================
# LOAD KNOWLEDGE
# ============================================================

def _with_stable_ids(items):
    result = []
    for item in items:
        copy = dict(item)
        copy.setdefault("id", stable_hash(copy.get("title", "") + copy.get("question", "")))
        result.append(copy)
    return result


def load_knowledge():
    global knowledge_cache

    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        knowledge_cache = _with_stable_ids(DEFAULT_KNOWLEDGE)
        brain.train(knowledge_cache)
        print("Supabase not configured.")
        print("Using default knowledge:", len(knowledge_cache))
        return

    rows = supabase_request(
        "GET",
        "knowledge",
        params={"select": "*", "approved": "eq.true", "order": "created_at.desc"},
    )

    if rows:
        knowledge_cache = rows
    else:
        knowledge_cache = _with_stable_ids(DEFAULT_KNOWLEDGE)

    brain.train(knowledge_cache)

    print("Knowledge:", len(knowledge_cache))
    print("Brain ready:", brain.ready)


# ============================================================
# LOCAL KNOWLEDGE SEARCH
# ============================================================

def similarity(a, b):
    a_words = set(expand_query(a))
    b_words = set(expand_query(b))

    if not a_words or not b_words:
        return 0.0

    intersection = len(a_words & b_words)
    union = len(a_words | b_words)

    return intersection / max(1, union)


def search_local_knowledge(query):
    predicted_category, confidence = brain.predict(query)

    results = []
    query_expanded = set(expand_query(query))

    for item in knowledge_cache:
        question = item.get("question", "")
        tags = " ".join(item.get("tags", []))
        title = item.get("title", "")

        score_question = similarity(query, question)
        score_tags = similarity(query, tags)
        score_title = similarity(query, title)

        direct_similarity = max(score_question, score_tags, score_title)

        category_bonus = 0.0
        if (
            predicted_category
            and item.get("category") == predicted_category
            and direct_similarity >= 0.08
        ):
            category_bonus = confidence * 0.15

        score = (
            score_question * 0.50
            + score_tags * 0.25
            + score_title * 0.15
            + category_bonus
        )

        combined_text = normalize(question + " " + title + " " + tags)

        exact_bonus = 0.0
        for word in query_expanded:
            if len(word) >= 4 and word in combined_text:
                exact_bonus += 0.03
        exact_bonus = min(exact_bonus, 0.15)

        score += exact_bonus

        results.append((score, item))

    results.sort(key=lambda x: x[0], reverse=True)

    filtered_results = [item for item in results if item[0] >= 0.10]

    print("LOCAL QUERY:", query)
    print("LOCAL PREDICTED CATEGORY:", predicted_category)
    print("LOCAL CATEGORY CONFIDENCE:", confidence)

    if filtered_results:
        print(
            "LOCAL TOP RESULT:",
            filtered_results[0][1].get("title", ""),
            "score=",
            filtered_results[0][0],
        )
    else:
        print("LOCAL RESULT: none")

    return filtered_results[:5]


# ============================================================
# WEB HELPERS
# ============================================================

def clean_text(text):
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def valid_http_url(url):
    try:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"}
    except Exception:
        return False


BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}


def _parse_searxng_payload(payload, limit):
    raw_results = payload.get("results", [])
    if not isinstance(raw_results, list):
        raw_results = []

    results = []
    seen_urls = set()

    for raw in raw_results:
        if not isinstance(raw, dict):
            continue

        title = clean_text(raw.get("title", ""))
        url_value = str(raw.get("url", "") or raw.get("link", ""))
        snippet = clean_text(raw.get("content", "") or raw.get("snippet", "") or "")

        if not title:
            continue
        if not valid_http_url(url_value):
            continue
        if url_value in seen_urls:
            continue

        seen_urls.add(url_value)

        results.append(
            {
                "title": title[:250],
                "url": url_value,
                "snippet": snippet[:1500],
                "source": "searxng",
            }
        )

        if len(results) >= limit:
            break

    return results


def searxng_search(query, limit=MAX_SEARCH_RESULTS):
    query = query.strip()
    if not query:
        return []

    headers = {
        **BROWSER_HEADERS,
        "Accept": "application/json",
    }

    for instance in SEARXNG_INSTANCES:
        base = instance.rstrip("/")
        url = base + "/search"

        for attempt in range(1, SEARXNG_MAX_RETRIES_PER_INSTANCE + 1):
            try:
                response = requests.get(
                    url,
                    params={
                        "q": query,
                        "format": "json",
                        "language": "ru-RU",
                        "safesearch": "1",
                        "categories": "general",
                    },
                    headers=headers,
                    timeout=SEARXNG_TIMEOUT,
                    allow_redirects=True,
                )
            except Exception as e:
                print(f"SEARXNG REQUEST ERROR (attempt {attempt}):", repr(e))
                break

            if response.status_code == 429:
                if attempt < SEARXNG_MAX_RETRIES_PER_INSTANCE:
                    delay = SEARXNG_RETRY_BACKOFF_BASE * attempt
                    time.sleep(delay)
                    continue
                else:
                    break

            if response.status_code >= 400:
                break

            content_type = response.headers.get("content-type", "").lower()
            if "json" not in content_type:
                break

            try:
                payload = response.json()
            except Exception:
                break

            results = _parse_searxng_payload(payload, limit)

            if results:
                return results

            break

    return []


def duckduckgo_html_search(query, limit=MAX_SEARCH_RESULTS):
    query = query.strip()
    if not query:
        return []

    url = "https://html.duckduckgo.com/html/"

    try:
        response = requests.post(
            url,
            data={"q": query, "kl": "ru-ru"},
            headers=BROWSER_HEADERS,
            timeout=SEARXNG_TIMEOUT,
        )
    except Exception as e:
        print("DUCKDUCKGO REQUEST ERROR:", repr(e))
        return []

    if response.status_code >= 400:
        return []

    try:
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        print("DUCKDUCKGO PARSE ERROR:", repr(e))
        return []

    results = []
    seen_urls = set()

    for result_div in soup.select(".result"):
        link = result_div.select_one("a.result__a")
        if not link:
            continue

        title = clean_text(link.get_text(" ", strip=True))
        href = link.get("href", "")

        snippet_tag = result_div.select_one(".result__snippet")
        snippet = clean_text(snippet_tag.get_text(" ", strip=True)) if snippet_tag else ""

        if not title or not valid_http_url(href):
            continue
        if href in seen_urls:
            continue

        seen_urls.add(href)

        results.append(
            {
                "title": title[:250],
                "url": href,
                "snippet": snippet[:1500],
                "source": "duckduckgo",
            }
        )

        if len(results) >= limit:
            break

    return results


SEARCH_ENGINES = [
    ("searxng", searxng_search),
    ("duckduckgo", duckduckgo_html_search),
]


REDIRECT_STUB_MARKERS = (
    "please click here if the page does not redirect",
    "click here if you are not redirected",
    "redirecting you to",
    "javascript is disabled",
)


def is_redirect_stub(text):
    if not text:
        return False
    normalized = text.strip().lower()
    if len(normalized) < 400 and any(marker in normalized for marker in REDIRECT_STUB_MARKERS):
        return True
    return False


def web_search_with_fallback(query, limit=MAX_SEARCH_RESULTS):
    for name, engine_fn in SEARCH_ENGINES:
        try:
            results = engine_fn(query, limit)
        except Exception as e:
            print(f"SEARCH ENGINE '{name}' CRASHED:", repr(e))
            results = []

        if results:
            print(f"WEB SEARCH ENGINE USED: {name}")
            return results, name

    print("WEB SEARCH: all engines failed.")
    return [], None


def fetch_page_text(url):
    if not valid_http_url(url):
        return ""

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ASCEND-AI/3.0)",
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml",
    }

    try:
        response = requests.get(
            url, headers=headers, timeout=PAGE_TIMEOUT, allow_redirects=True
        )

        if response.status_code >= 400:
            return ""

        content_type = response.headers.get("content-type", "").lower()
        if "text/html" not in content_type:
            return ""

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form"]):
            tag.decompose()

        text = clean_text(soup.get_text(" ", strip=True))
        text = text[:MAX_SOURCE_TEXT]

        return text

    except Exception as e:
        print("SOURCE FETCH ERROR:", repr(e))
        return ""


def collect_web_information(query):
    search_results, engine_used = web_search_with_fallback(query)

    if not search_results:
        return []

    enriched = []
    for result in search_results:
        page_text = fetch_page_text(result["url"])

        if is_redirect_stub(page_text):
            page_text = ""

        if not page_text and not clean_text(result.get("snippet", "")):
            continue

        enriched.append({**result, "page_text": page_text})

    return enriched


def save_web_sources(session_id, query, results):
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        return

    for result in results:
        payload = {
            "session_id": session_id,
            "query": query,
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "snippet": result.get("snippet", ""),
            "page_text": result.get("page_text", ""),
            "source": result.get("source", "web"),
        }

        supabase_request("POST", "web_sources", payload)


def build_web_context(results):
    pieces = []

    for index, item in enumerate(results, start=1):
        title = item.get("title", "")
        url = item.get("url", "")
        snippet = item.get("snippet", "")
        page_text = item.get("page_text", "")

        text = page_text or snippet
        if not text:
            continue

        pieces.append(
            f"\nИСТОЧНИК {index}\nНазвание: {title}\nURL: {url}\nИнформация:\n\n{text}\n"
        )

    return "\n".join(pieces)


def clean_web_text(results):
    pieces = []
    for item in results:
        text = item.get("page_text") or item.get("snippet") or ""
        text = clean_text(text)
        if text:
            pieces.append(text)

    return "\n\n".join(pieces)


def split_sentences(text):
    text = text.replace("\n", " ")
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [clean_text(x) for x in parts if len(clean_text(x)) > 20]


def rank_sentences(query, text, limit=8):
    sentences = split_sentences(text)
    qwords = set(expand_query(query))

    scored = []
    for sentence in sentences:
        swords = set(expand_query(sentence))
        overlap = len(qwords & swords)

        if overlap:
            score = overlap / math.sqrt(max(1, len(swords)))
            scored.append((score, sentence))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [sentence for _, sentence in scored[:limit]]


# ============================================================
# LLM ANSWER SYNTHESIS
# ============================================================

API_KEY_SETTINGS = ("openrouter_api_key", "deepseek_api_key", "qwen_api_key", "provod_api_key")


def llm_available():
    return any(get_setting(k) for k in API_KEY_SETTINGS)


def _chat_completion_request(url, api_key, model, messages, extra_headers=None):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    response = requests.post(
        url,
        headers=headers,
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 900,
        },
        timeout=LLM_TIMEOUT,
    )
    return response


def _try_openrouter(messages):
    api_key = get_setting("openrouter_api_key")
    if not api_key:
        return None

    for model in OPENROUTER_FREE_MODELS:
        try:
            response = _chat_completion_request(OPENROUTER_URL, api_key, model, messages)
        except Exception as e:
            print("LLM REQUEST ERROR (openrouter):", repr(e), model)
            continue

        if response.status_code in (404, 429) or response.status_code >= 400:
            continue

        try:
            content = response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print("LLM PARSE ERROR (openrouter):", repr(e))
            continue

        if content:
            return content

    return None


def _try_deepseek_direct(messages):
    api_key = get_setting("deepseek_api_key")
    if not api_key:
        return None

    try:
        response = _chat_completion_request(DEEPSEEK_URL, api_key, DEEPSEEK_MODEL, messages)
    except Exception as e:
        print("LLM REQUEST ERROR (deepseek):", repr(e))
        return None

    if response.status_code >= 400:
        return None

    try:
        content = response.json()["choices"][0]["message"]["content"].strip()
        return content or None
    except Exception as e:
        print("LLM PARSE ERROR (deepseek):", repr(e))
        return None


def _try_qwen_direct(messages):
    api_key = get_setting("qwen_api_key")
    if not api_key:
        return None

    try:
        response = _chat_completion_request(QWEN_URL, api_key, QWEN_MODEL, messages)
    except Exception as e:
        print("LLM REQUEST ERROR (qwen):", repr(e))
        return None

    if response.status_code >= 400:
        return None

    try:
        content = response.json()["choices"][0]["message"]["content"].strip()
        return content or None
    except Exception as e:
        print("LLM PARSE ERROR (qwen):", repr(e))
        return None


def _try_provod_direct(messages):
    api_key = get_setting("provod_api_key")
    if not api_key:
        return None

    model = get_setting("provod_model") or PROVOD_DEFAULT_MODEL

    try:
        response = _chat_completion_request(PROVOD_URL, api_key, model, messages)
    except Exception as e:
        print("LLM REQUEST ERROR (provod):", repr(e))
        return None

    if response.status_code >= 400:
        return None

    try:
        content = response.json()["choices"][0]["message"]["content"].strip()
        return content or None
    except Exception as e:
        print("LLM PARSE ERROR (provod):", repr(e))
        return None


def call_llm(system_prompt, user_prompt, history=None):
    messages = [{"role": "system", "content": system_prompt}]

    if history:
        for item in history[-MAX_LLM_HISTORY_MESSAGES:]:
            role = item.get("role")
            content = (item.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_prompt})

    providers = (
        _try_provod_direct,
        _try_openrouter,
        _try_deepseek_direct,
        _try_qwen_direct,
    )

    for provider_fn in providers:
        result = provider_fn(messages)
        if result:
            return result

    print("LLM: all providers failed")
    return None


def llm_answer_from_local(query, knowledge_answer, web_results, history=None):
    web_context = build_web_context(web_results) if web_results else ""

    system_prompt = (
        "Ты — дружелюбный ассистент по уходу за собой (кожа, внешность, "
        "питание, сон, тренировки). Отвечай на русском языке, простым "
        "разговорным текстом, без списков источников и без вставки URL "
        "в текст ответа. Можешь использовать нумерованные шаги или "
        "маркированные пункты для советов, если это уместно. Учитывай "
        "предыдущие сообщения пользователя в этом диалоге, если они есть. "
        "Не выдумывай медицинские факты — если сомневаешься, порекомендуй "
        "обратиться к врачу."
    )

    user_prompt = (
        f"Вопрос пользователя: {query}\n\n"
        f"Проверенный ответ из базы знаний (используй как основу):\n"
        f"{knowledge_answer}\n"
    )

    if web_context:
        user_prompt += (
            f"\nДополнительная информация из свежего веб-поиска "
            f"(используй только если она релевантна и не противоречит "
            f"базе знаний; не перечисляй источники и не вставляй ссылки):\n"
            f"{web_context}\n"
        )

    user_prompt += (
        "\nПерескажи это связным текстом на русском, дружелюбно и по делу."
    )

    return call_llm(system_prompt, user_prompt, history=history)


def llm_answer_from_web(query, web_results, history=None):
    web_context = build_web_context(web_results)
    if not web_context:
        return None

    system_prompt = (
        "Ты — дружелюбный ассистент по уходу за собой (кожа, внешность, "
        "питание, сон, тренировки). Отвечай на русском языке связным "
        "текстом на основе предоставленной информации из интернета. "
        "Учитывай предыдущие сообщения пользователя в этом диалоге, если "
        "они есть. НЕ перечисляй источники, НЕ вставляй URL и названия "
        "сайтов в текст ответа — просто дай полезный ответ по существу. "
        "Если информации недостаточно для уверенного ответа, честно "
        "скажи об этом и порекомендуй обратиться к специалисту."
    )

    user_prompt = (
        f"Вопрос пользователя: {query}\n\n"
        f"Информация, найденная в интернете:\n{web_context}\n\n"
        "Дай связный, дружелюбный ответ по существу вопроса."
    )

    return call_llm(system_prompt, user_prompt, history=history)


def llm_answer_general(query, history=None):
    system_prompt = (
        "Ты — дружелюбный ассистент по уходу за собой (кожа, внешность, "
        "питание, сон, тренировки). Веб-поиск сейчас недоступен, поэтому "
        "отвечай на основе своих собственных знаний по теме. Учитывай "
        "предыдущие сообщения пользователя в этом диалоге, если они есть. "
        "Если вопрос затрагивает несколько разных проблем сразу — ответь "
        "по каждой отдельным пунктом. Пиши на русском, дружелюбно и по "
        "делу. Не выдумывай медицинские факты — если сомневаешься, честно "
        "скажи об этом и порекомендуй обратиться к врачу/дерматологу. "
        "В конце ОБЯЗАТЕЛЬНО одной короткой строкой предупреди, что "
        "ответ дан без сверки со свежими источниками из интернета."
    )

    user_prompt = f"Вопрос пользователя: {query}\n\nДай полезный ответ по существу."

    return call_llm(system_prompt, user_prompt, history=history)


def fallback_web_answer(query, web_results):
    web_text = clean_web_text(web_results)
    if not web_text:
        return ""

    sentences = rank_sentences(query, web_text, limit=8)

    if not sentences:
        sentences = split_sentences(web_text)[:5]

    if not sentences:
        return ""

    answer = "🌐 Я нашёл информацию по твоему вопросу в интернете.\n\n"

    for sentence in sentences:
        answer += "• " + sentence + "\n"

    answer += (
        "\n⚠️ Информация собрана из найденных в интернете источников. "
        "Для важных вопросов проверяй первоисточники."
    )

    return answer


def generate_response(query, memory, local_results, web_results):
    best_score = 0.0
    best_item = None

    if local_results:
        best_score, best_item = local_results[0]

    if best_item and best_score >= 0.18:
        knowledge_answer = best_item.get("answer", "").strip()

        if llm_available():
            llm_answer = llm_answer_from_local(query, knowledge_answer, web_results, history=memory)
            if llm_answer:
                return llm_answer

        answer = knowledge_answer

        if web_results:
            web_text = clean_web_text(web_results)
            sentences = rank_sentences(query, web_text, limit=4)

            if sentences:
                answer += "\n\n🌐 Дополнение из актуального поиска:\n"
                for sentence in sentences:
                    answer += "\n• " + sentence

        return answer

    if web_results:
        if llm_available():
            llm_answer = llm_answer_from_web(query, web_results, history=memory)
            if llm_answer:
                return llm_answer

        web_answer = fallback_web_answer(query, web_results)
        if web_answer:
            return web_answer

    if llm_available():
        general_answer = llm_answer_general(query, history=memory)
        if general_answer:
            return general_answer

    return (
        "Я не смог получить результаты веб-поиска прямо сейчас.\n\n"
        "Попробуй повторить запрос немного позже или сформулировать его подробнее."
    )


# ============================================================
# MEMORY
# ============================================================

def save_message(session_id, role, content):
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        return None

    rows = supabase_request(
        "POST",
        "chat_messages",
        {"session_id": session_id, "role": role, "content": content},
    )

    if rows:
        return rows[0]

    return None


def get_memory(session_id):
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        return []

    rows = supabase_request(
        "GET",
        "chat_messages",
        params={
            "select": "role,content,created_at",
            "session_id": f"eq.{session_id}",
            "order": "created_at.desc",
            "limit": str(MAX_MEMORY),
        },
    )

    rows.reverse()
    return rows


def get_chat_messages(session_id, limit=MAX_CHAT_HISTORY):
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        return []

    rows = supabase_request(
        "GET",
        "chat_messages",
        params={
            "select": "role,content,created_at",
            "session_id": f"eq.{session_id}",
            "order": "created_at.asc",
            "limit": str(limit),
        },
    )

    return rows or []


# ============================================================
# CHAT SESSIONS
# ============================================================
#
#   create table if not exists chat_sessions (
#       session_id text primary key,
#       device_id text,
#       title text default 'Новый чат',
#       created_at timestamptz default now(),
#       updated_at timestamptz default now()
#   );
#   create index if not exists idx_chat_sessions_device
#       on chat_sessions(device_id);
#

CHAT_TITLE_MAX_LEN = 42


def make_chat_title(message):
    message = clean_text(message)
    if len(message) <= CHAT_TITLE_MAX_LEN:
        return message or "Новый чат"
    return message[:CHAT_TITLE_MAX_LEN].rstrip() + "…"


def create_chat_session(session_id, device_id, title="Новый чат"):
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        return
    supabase_request(
        "POST",
        "chat_sessions",
        {"session_id": session_id, "device_id": device_id or "", "title": title},
    )


def list_chat_sessions(device_id):
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY or not device_id:
        return []

    rows = supabase_request(
        "GET",
        "chat_sessions",
        params={
            "select": "session_id,title,created_at,updated_at",
            "device_id": f"eq.{device_id}",
            "order": "updated_at.desc",
            "limit": "100",
        },
    )

    return rows or []


def touch_chat_session(session_id, title=None):
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        return

    payload = {"updated_at": datetime.now(timezone.utc).isoformat()}
    if title:
        payload["title"] = title

    supabase_request(
        "PATCH", "chat_sessions", payload, params={"session_id": f"eq.{session_id}"}
    )


def delete_chat_session(session_id):
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        return

    supabase_request("DELETE", "chat_sessions", params={"session_id": f"eq.{session_id}"})
    supabase_request("DELETE", "chat_messages", params={"session_id": f"eq.{session_id}"})


# ============================================================
# CREDITS / БЕСПЛАТНЫЙ ПЕРВЫЙ ЗАПРОС
# ============================================================
#
# Каждому IP-адресу положен один бесплатный запрос. Ключ строится из
# IP (а НЕ из localStorage/device_id), поэтому очистка данных браузера,
# смена вкладки или повторное открытие сайта не даёт новый бесплатный
# запрос — только смена реального IP-адреса.
#
#   create table if not exists user_credits (
#       key text primary key,
#       credits integer default 0,
#       free_used boolean default false,
#       updated_at timestamptz default now()
#   );
#
# Без Supabase используется хранилище в памяти процесса (сбрасывается
# при рестарте) — для продакшена настоятельно рекомендуется Supabase.
#

CREDITS_TABLE = "user_credits"


def get_client_key(request: Request):
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() if forwarded else ""
    if not ip and request.client:
        ip = request.client.host
    ip = ip or "unknown"
    return hashlib.sha256(f"ascend-credits:{ip}".encode("utf-8")).hexdigest()[:40]


def get_credit_state(key):
    if SUPABASE_URL and SUPABASE_SECRET_KEY:
        rows = supabase_request(
            "GET", CREDITS_TABLE, params={"select": "*", "key": f"eq.{key}"}
        )
        if rows:
            return rows[0]
        return {"key": key, "credits": 0, "free_used": False}

    return dict(_local_credits_cache.get(key, {"key": key, "credits": 0, "free_used": False}))


def save_credit_state(key, credits_left, free_used):
    if SUPABASE_URL and SUPABASE_SECRET_KEY:
        supabase_request(
            "POST",
            CREDITS_TABLE,
            {"key": key, "credits": credits_left, "free_used": free_used},
            params={"on_conflict": "key"},
            prefer="resolution=merge-duplicates,return=representation",
        )
    else:
        _local_credits_cache[key] = {
            "key": key,
            "credits": credits_left,
            "free_used": free_used,
        }


def consume_access(key):
    state = get_credit_state(key)
    credits_left = int(state.get("credits") or 0)
    free_used = bool(state.get("free_used"))

    if not free_used:
        save_credit_state(key, credits_left, True)
        return {"allowed": True, "mode": "free", "credits": credits_left}

    if credits_left > 0:
        credits_left -= 1
        save_credit_state(key, credits_left, True)
        return {"allowed": True, "mode": "paid", "credits": credits_left}

    return {"allowed": False, "mode": "blocked", "credits": 0}


def save_training_log(question, answer, category, source):
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        return

    supabase_request(
        "POST",
        "training_log",
        {
            "question": question,
            "answer": answer,
            "category": category,
            "source": source,
            "approved": True,
        },
    )


# ============================================================
# MODELS
# ============================================================

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


# ============================================================
# HTML — ГЛАВНАЯ СТРАНИЦА ЧАТА
# ============================================================

HTML = r"""
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ASCEND AI</title>
<style>
:root{
  --bg:#06060a;
  --bg-soft:#0b0b10;
  --surface:#121218;
  --surface-2:#181820;
  --border:#242430;
  --border-soft:#1c1c24;
  --text:#f5f5f8;
  --text-dim:#93939f;
  --accent:#f7d45b;
  --accent-2:#ff9f5b;
  --accent-3:#8b7bff;
  --accent-grad:linear-gradient(135deg,#f7d45b 0%,#ff9f5b 55%,#ff6bd4 100%);
  --accent-grad-2:linear-gradient(120deg,#8b7bff,#f7d45b,#ff9f5b,#8b7bff);
  --danger:#ff6b6b;
  --radius-lg:24px;
  --radius-md:16px;
  --radius-sm:10px;
}
*{box-sizing:border-box;}
html,body{height:100%;}
body{
  margin:0;
  background:var(--bg);
  color:var(--text);
  font-family:'Inter',-apple-system,BlinkMacSystemFont,Arial,sans-serif;
  overflow:hidden;
  -webkit-font-smoothing:antialiased;
}
button,textarea,input{font:inherit;color:inherit;}
button{cursor:pointer;}
a{color:var(--accent);}

/* ---------------- animated background ---------------- */
.bg-blobs{position:fixed;inset:0;overflow:hidden;z-index:0;pointer-events:none;}
.grain{position:fixed;inset:0;z-index:2;pointer-events:none;opacity:.035;mix-blend-mode:overlay;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");}
.blob{position:absolute;border-radius:50%;filter:blur(100px);opacity:.32;}
.blob-1{width:560px;height:560px;background:radial-gradient(circle,#f7d45b,transparent 70%);top:-180px;left:-140px;animation:float1 24s ease-in-out infinite;}
.blob-2{width:500px;height:500px;background:radial-gradient(circle,#ff9f5b,transparent 70%);bottom:-160px;right:-120px;animation:float2 28s ease-in-out infinite;}
.blob-3{width:420px;height:420px;background:radial-gradient(circle,#8b7bff,transparent 70%);top:38%;left:58%;animation:float3 32s ease-in-out infinite;opacity:.24;}
.blob-4{width:320px;height:320px;background:radial-gradient(circle,#ff6bd4,transparent 70%);bottom:20%;left:12%;animation:float4 26s ease-in-out infinite;opacity:.16;}
@keyframes float1{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(70px,90px) scale(1.08)}}
@keyframes float2{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(-80px,-60px) scale(1.05)}}
@keyframes float3{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(-50px,70px) scale(1.18)}}
@keyframes float4{0%,100%{transform:translate(0,0)}50%{transform:translate(40px,-40px)}}

.app{position:relative;z-index:1;height:100vh;display:flex;}

/* ---------------- sidebar ---------------- */
.sidebar{
  position:fixed;top:0;left:0;bottom:0;width:300px;max-width:85vw;
  background:rgba(12,12,17,.94);backdrop-filter:blur(22px);
  border-right:1px solid var(--border);
  transform:translateX(-100%);
  transition:transform .35s cubic-bezier(.4,0,.2,1);
  z-index:40;display:flex;flex-direction:column;padding:18px 14px;
}
.sidebar.open{transform:translateX(0);}
.sidebar-overlay{
  position:fixed;inset:0;background:rgba(0,0,0,.55);backdrop-filter:blur(3px);
  opacity:0;pointer-events:none;transition:opacity .3s;z-index:35;
}
.sidebar-overlay.open{opacity:1;pointer-events:auto;}
.new-chat-btn{
  display:flex;align-items:center;gap:10px;justify-content:center;
  background:var(--accent-grad);background-size:220% 220%;color:#171410;font-weight:800;
  border:0;border-radius:var(--radius-md);padding:13px;margin-bottom:16px;
  transition:transform .18s, box-shadow .18s, background-position .4s;
}
.new-chat-btn:hover{transform:translateY(-1px);box-shadow:0 10px 28px rgba(247,212,91,.28);background-position:100% 50%;}
.new-chat-btn:active{transform:translateY(0) scale(.98);}
.chat-list{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:4px;}
.chat-item{
  display:flex;align-items:center;justify-content:space-between;gap:8px;
  padding:11px 12px;border-radius:12px;border:1px solid transparent;
  color:var(--text-dim);cursor:pointer;transition:background .18s,color .18s,border-color .18s,transform .15s;
}
.chat-item:hover{background:var(--surface);color:var(--text);transform:translateX(2px);}
.chat-item.active{background:var(--surface-2);color:var(--text);border-color:var(--border);}
.chat-item .title{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:14px;}
.chat-item .del{opacity:0;transition:opacity .15s;background:none;border:0;color:var(--text-dim);padding:4px;border-radius:8px;}
.chat-item:hover .del{opacity:1;}
.chat-item .del:hover{color:var(--danger);background:rgba(255,107,107,.1);}
.chat-empty{color:var(--text-dim);font-size:13px;padding:14px 8px;text-align:center;}
.sidebar-footer{border-top:1px solid var(--border);padding-top:12px;margin-top:10px;display:flex;flex-direction:column;gap:6px;}
.sidebar-link{background:none;border:0;text-align:left;color:var(--text-dim);font-size:13px;padding:8px 6px;border-radius:8px;transition:background .15s,color .15s;}
.sidebar-link:hover{background:var(--surface);color:var(--text);}
.build-tag{text-align:center;color:#4b4b56;font-size:10px;letter-spacing:.5px;margin-top:6px;opacity:.65;}

/* ---------------- main column ---------------- */
.main{flex:1;display:flex;flex-direction:column;height:100vh;min-width:0;}
.topbar{
  height:64px;flex:none;display:flex;align-items:center;justify-content:space-between;
  padding:0 18px;border-bottom:1px solid var(--border);
  background:rgba(8,8,12,.72);backdrop-filter:blur(16px);gap:10px;
}
.icon-btn{
  width:40px;height:40px;border-radius:12px;border:1px solid var(--border);
  background:var(--surface);display:flex;align-items:center;justify-content:center;
  transition:background .18s,transform .18s,border-color .18s;
}
.icon-btn:hover{background:var(--surface-2);transform:translateY(-1px);border-color:#34343f;}
.brand{display:flex;align-items:center;gap:10px;font-weight:900;font-size:18px;letter-spacing:.3px;}
.brand .dot{width:9px;height:9px;border-radius:50%;background:var(--accent-grad);box-shadow:0 0 14px var(--accent);animation:pulseDot 2.4s ease-in-out infinite;}
@keyframes pulseDot{0%,100%{transform:scale(1);opacity:1;}50%{transform:scale(1.35);opacity:.6;}}
.brand .grad-text{
  background:var(--accent-grad-2);background-size:300% 300%;
  -webkit-background-clip:text;background-clip:text;color:transparent;
  animation:gradShift 8s ease infinite;
}
@keyframes gradShift{0%{background-position:0% 50%;}50%{background-position:100% 50%;}100%{background-position:0% 50%;}}
.brand span.sub{opacity:.45;font-weight:700;-webkit-text-fill-color:var(--text-dim);}
.topbar-right{display:flex;align-items:center;gap:8px;}
.pill{
  display:flex;align-items:center;gap:6px;border:1px solid var(--border);background:var(--surface);
  border-radius:999px;padding:8px 13px;font-size:12.5px;color:var(--text-dim);white-space:nowrap;
  transition:background .18s, color .18s, border-color .18s, transform .15s;
}
.pill.clickable:hover{background:var(--surface-2);color:var(--text);transform:translateY(-1px);}
.pill.warn{color:var(--accent-2);border-color:rgba(255,159,91,.4);animation:warnPulse 2.2s ease-in-out infinite;}
@keyframes warnPulse{0%,100%{box-shadow:0 0 0 0 rgba(255,159,91,0);}50%{box-shadow:0 0 0 4px rgba(255,159,91,.12);}}

/* ---------------- messages ---------------- */
.messages{flex:1;overflow-y:auto;padding:28px 0 10px;scroll-behavior:smooth;}
.messages-inner{max-width:760px;margin:0 auto;padding:0 18px;}
.message{display:flex;margin-bottom:18px;animation:msgIn .4s cubic-bezier(.22,1,.36,1);}
@keyframes msgIn{from{opacity:0;transform:translateY(14px) scale(.98);}to{opacity:1;transform:translateY(0) scale(1);}}
.message.user{justify-content:flex-end;}
.message.ai{justify-content:flex-start;}
.avatar{
  width:32px;height:32px;border-radius:10px;flex:none;margin-right:10px;
  background:var(--accent-grad);background-size:220% 220%;display:flex;align-items:center;justify-content:center;
  font-size:15px;box-shadow:0 4px 16px rgba(247,212,91,.28);animation:avatarShift 6s ease infinite;
}
@keyframes avatarShift{0%,100%{background-position:0% 50%;}50%{background-position:100% 50%;}}
.bubble{
  max-width:min(640px,82%);padding:15px 18px;border-radius:18px;line-height:1.65;
  white-space:pre-wrap;font-size:15px;
}
.message.ai .bubble{background:var(--surface);border:1px solid var(--border);border-top-left-radius:6px;box-shadow:0 6px 22px rgba(0,0,0,.18);}
.message.user .bubble{background:var(--accent-grad);color:#171410;font-weight:600;border-top-right-radius:6px;box-shadow:0 6px 22px rgba(247,212,91,.16);}
.sources{max-width:640px;margin:-8px 0 22px 42px;}
.sources-title{color:var(--text-dim);font-size:12.5px;margin-bottom:7px;}
.source-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:10px 12px;margin-top:6px;transition:border-color .18s,transform .15s;}
.source-card:hover{border-color:#3a3a48;transform:translateX(2px);}
.source-card .s-title{font-size:13px;margin-bottom:3px;}
.source-card a{font-size:12px;text-decoration:none;word-break:break-word;opacity:.85;}

/* ---------------- typing indicator (upgraded) ---------------- */
.typing-row{display:flex;align-items:center;margin-bottom:18px;animation:msgIn .35s cubic-bezier(.22,1,.36,1);}
.typing-bubble{
  position:relative;display:flex;align-items:center;gap:7px;background:var(--surface);
  border:1px solid transparent;border-radius:18px;border-top-left-radius:6px;
  padding:15px 20px;overflow:hidden;
}
.typing-bubble::before{
  content:"";position:absolute;inset:0;padding:1px;border-radius:inherit;
  background:var(--accent-grad-2);background-size:300% 300%;
  -webkit-mask:linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite:xor;mask-composite:exclude;
  animation:gradShift 3.5s linear infinite;opacity:.85;
}
.typing-bubble span{width:7px;height:7px;border-radius:50%;background:var(--accent);animation:bounce 1.1s infinite;position:relative;z-index:1;}
.typing-bubble span:nth-child(2){animation-delay:.15s;background:var(--accent-2);}
.typing-bubble span:nth-child(3){animation-delay:.3s;background:var(--accent-3);}
@keyframes bounce{0%,60%,100%{transform:translateY(0);opacity:.5;}30%{transform:translateY(-6px);opacity:1;}}
.typing-label{font-size:12.5px;color:var(--text-dim);margin-left:8px;position:relative;z-index:1;}

/* ---------------- composer ---------------- */
.composer{padding:14px 18px 22px;flex:none;}
.composer-inner{max-width:760px;margin:0 auto;}
.composer-box{
  display:flex;align-items:flex-end;gap:10px;background:var(--surface);
  border:1px solid var(--border);padding:9px 9px 9px 18px;border-radius:20px;
  transition:border-color .2s, box-shadow .2s;
}
.composer-box:focus-within{border-color:#454558;box-shadow:0 0 0 4px rgba(247,212,91,.1);}
.composer textarea{
  flex:1;border:0;outline:0;background:transparent;resize:none;
  padding:11px 0;min-height:24px;max-height:150px;font-size:15px;
}
.composer textarea::placeholder{color:#6d6d78;}
.send{
  width:44px;height:44px;border-radius:14px;border:0;background:var(--accent-grad);background-size:220% 220%;
  display:flex;align-items:center;justify-content:center;flex:none;
  transition:transform .18s, opacity .18s, background-position .3s;
}
.send:hover{transform:scale(1.06);background-position:100% 50%;}
.send:active{transform:scale(.94);}
.send:disabled{opacity:.4;cursor:not-allowed;transform:none;}
.hint{text-align:center;color:#57575f;font-size:11.5px;margin-top:10px;}
.hint button{background:none;border:0;color:#57575f;text-decoration:underline;font-size:11.5px;padding:0;}

/* ---------------- modals ---------------- */
.modal-overlay{
  position:fixed;inset:0;background:rgba(4,4,7,.74);backdrop-filter:blur(7px);
  display:flex;align-items:center;justify-content:center;z-index:100;
  opacity:0;pointer-events:none;transition:opacity .28s;padding:20px;
}
.modal-overlay.open{opacity:1;pointer-events:auto;}
.modal-card{
  background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);
  max-width:460px;width:100%;padding:28px;max-height:86vh;overflow-y:auto;position:relative;
  transform:translateY(16px) scale(.97);transition:transform .3s cubic-bezier(.22,1,.36,1);
  box-shadow:0 30px 80px rgba(0,0,0,.45);
}
.modal-overlay.open .modal-card{transform:translateY(0) scale(1);}
.modal-card h2{margin:0 0 6px;font-size:21px;}
.modal-card p{color:var(--text-dim);font-size:14.5px;line-height:1.6;}
.modal-links{display:flex;flex-direction:column;gap:10px;margin:18px 0;}
.modal-link-btn{
  display:flex;align-items:center;justify-content:space-between;
  background:var(--surface-2);border:1px solid var(--border);border-radius:12px;
  padding:13px 16px;color:var(--text);text-decoration:none;font-size:14px;transition:border-color .18s,transform .15s;
}
.modal-link-btn:hover{border-color:var(--accent);transform:translateX(2px);}
.modal-checkbox-row{display:flex;align-items:flex-start;gap:10px;margin:16px 0;font-size:13px;color:var(--text-dim);}
.modal-checkbox-row input{margin-top:3px;accent-color:var(--accent);}
.primary-btn{
  width:100%;background:var(--accent-grad);background-size:220% 220%;color:#161410;border:0;border-radius:13px;
  padding:14px;font-weight:800;font-size:15px;transition:opacity .18s, transform .18s, background-position .3s;
}
.primary-btn:disabled{opacity:.4;cursor:not-allowed;}
.primary-btn:not(:disabled):hover{transform:translateY(-1px);background-position:100% 50%;}
.ghost-btn{
  width:100%;background:transparent;color:var(--text-dim);border:1px solid var(--border);
  border-radius:13px;padding:12px;font-size:14px;margin-top:8px;transition:color .15s,border-color .15s;
}
.ghost-btn:hover{color:var(--text);border-color:#3a3a48;}
.close-x{position:absolute;top:16px;right:16px;background:none;border:0;color:var(--text-dim);font-size:20px;transition:color .15s,transform .15s;}
.close-x:hover{color:var(--text);transform:rotate(90deg);}

/* pricing */
.plans{display:flex;flex-direction:column;gap:12px;margin:18px 0;}
.plan-card{
  border:1px solid var(--border);border-radius:16px;padding:16px 18px;position:relative;
  background:var(--surface-2);transition:border-color .2s, transform .15s;
}
.plan-card:hover{transform:translateY(-1px);border-color:#3a3a48;}
.plan-card.popular{border-color:var(--accent);box-shadow:0 0 0 1px rgba(247,212,91,.28), 0 10px 30px rgba(247,212,91,.08);}
.plan-badge{
  position:absolute;top:-10px;right:16px;background:var(--accent-grad);color:#171410;
  font-size:11px;font-weight:800;padding:3px 10px;border-radius:999px;
}
.plan-top{display:flex;justify-content:space-between;align-items:baseline;}
.plan-name{font-weight:800;font-size:15.5px;}
.plan-price{font-weight:900;font-size:19px;}
.plan-meta{color:var(--text-dim);font-size:12.5px;margin-top:4px;}
.free-banner{
  display:flex;align-items:center;gap:10px;background:linear-gradient(135deg,rgba(247,212,91,.14),rgba(255,159,91,.08));
  border:1px solid rgba(247,212,91,.3);border-radius:14px;padding:12px 14px;margin-bottom:16px;font-size:13px;color:var(--text);
}

::-webkit-scrollbar{width:8px;height:8px;}
::-webkit-scrollbar-thumb{background:#2a2a35;border-radius:8px;}
::-webkit-scrollbar-track{background:transparent;}

@media(max-width:720px){
  .bubble{max-width:88%;}
  .sources{margin-left:8px;}
  .sidebar{width:280px;}
}
</style>
</head>
<body>

<div class="grain"></div>
<div class="bg-blobs">
  <div class="blob blob-1"></div>
  <div class="blob blob-2"></div>
  <div class="blob blob-3"></div>
  <div class="blob blob-4"></div>
</div>

<div class="app">

  <div id="sidebarOverlay" class="sidebar-overlay" onclick="closeSidebar()"></div>

  <aside id="sidebar" class="sidebar">
    <button class="new-chat-btn" onclick="createNewChat()">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M12 5v14M5 12h14" stroke="#171410" stroke-width="2.4" stroke-linecap="round"/></svg>
      Новый чат
    </button>
    <div id="chatList" class="chat-list">
      <div class="chat-empty">Загрузка чатов…</div>
    </div>
    <div class="sidebar-footer">
      <button class="sidebar-link" onclick="openInfoModal()">ℹ️ Инфо, поддержка, тарифы</button>
      <div class="build-tag">ASCEND AI · ядро mekbuda</div>
    </div>
  </aside>

  <div class="main">
    <div class="topbar">
      <div style="display:flex;align-items:center;gap:12px;">
        <button class="icon-btn" onclick="openSidebar()" aria-label="Чаты">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M4 6h16M4 12h16M4 18h16" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
        </button>
        <div class="brand"><span class="dot"></span><span class="grad-text">ASCEND</span>&nbsp;<span class="sub">AI</span></div>
      </div>
      <div class="topbar-right">
        <div id="balancePill" class="pill clickable" onclick="openPricingModal()">···</div>
        <button class="icon-btn" onclick="openInfoModal()" aria-label="Инфо">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.8"/><path d="M12 11v5.5M12 8v.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
        </button>
      </div>
    </div>

    <div id="messages" class="messages">
      <div class="messages-inner" id="messagesInner">
        <div class="message ai">
          <div class="avatar">🧠</div>
          <div class="bubble">Привет! Я ASCEND AI.

Помогу разобраться с уходом за кожей, внешностью, питанием, сном и тренировками — а если в моей базе знаний ответа нет, поищу актуальную информацию в интернете.

Первый запрос — бесплатно 🎁 Что тебя интересует?</div>
        </div>
      </div>
    </div>

    <div class="composer">
      <div class="composer-inner">
        <div class="composer-box">
          <textarea id="messageInput" placeholder="Напиши свой вопрос..." rows="1"></textarea>
          <button id="sendButton" class="send" onclick="sendMessage()" aria-label="Отправить">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M4 12L20 4L14 20L11 13L4 12Z" fill="#171410"/></svg>
          </button>
        </div>
        <div class="hint">Ответы носят справочный характер и не заменяют консультацию специалиста · <button onclick="openInfoModal()">Тарифы и поддержка</button></div>
      </div>
    </div>
  </div>
</div>

<!-- Welcome / terms modal -->
<div id="welcomeModal" class="modal-overlay">
  <div class="modal-card">
    <h2>Добро пожаловать 👋</h2>
    <p>ASCEND AI — ассистент по уходу за собой: кожа, внешность, питание, сон и тренировки. Первый запрос — бесплатно, дальше — доступ по недорогим пакетам запросов.</p>
    <div class="modal-links">
      <a class="modal-link-btn" href="__PRIVACY_URL__" target="_blank" rel="noopener noreferrer">Политика конфиденциальности <span>↗</span></a>
      <a class="modal-link-btn" href="__TERMS_URL__" target="_blank" rel="noopener noreferrer">Пользовательское соглашение <span>↗</span></a>
      <a class="modal-link-btn" href="__SUPPORT_URL__" target="_blank" rel="noopener noreferrer">Поддержка в Telegram <span>↗</span></a>
    </div>
    <label class="modal-checkbox-row">
      <input type="checkbox" id="acceptCheckbox" onchange="document.getElementById('acceptBtn').disabled=!this.checked;">
      Я ознакомился(ась) и принимаю Политику конфиденциальности и Пользовательское соглашение.
    </label>
    <button id="acceptBtn" class="primary-btn" disabled onclick="acceptTerms()">Продолжить</button>
  </div>
</div>

<!-- Info modal -->
<div id="infoModal" class="modal-overlay">
  <div class="modal-card">
    <button class="close-x" onclick="closeModal('infoModal')">✕</button>
    <h2>Информация</h2>
    <p>Документы, поддержка и тарифы сервиса.</p>
    <div class="modal-links">
      <a class="modal-link-btn" href="__PRIVACY_URL__" target="_blank" rel="noopener noreferrer">Политика конфиденциальности <span>↗</span></a>
      <a class="modal-link-btn" href="__TERMS_URL__" target="_blank" rel="noopener noreferrer">Пользовательское соглашение <span>↗</span></a>
      <a class="modal-link-btn" href="__SUPPORT_URL__" target="_blank" rel="noopener noreferrer">Поддержка: __SUPPORT_HANDLE__ <span>↗</span></a>
    </div>
    <button class="primary-btn" onclick="closeModal('infoModal');openPricingModal();">💳 Посмотреть тарифы</button>
  </div>
</div>

<!-- Pricing modal -->
<div id="pricingModal" class="modal-overlay">
  <div class="modal-card">
    <button class="close-x" onclick="closeModal('pricingModal')">✕</button>
    <h2>Тарифы</h2>
    <div class="free-banner">🎁 Первый запрос всегда бесплатный. Дальше — недорогие пакеты, без автосписаний.</div>
    <div id="plansList" class="plans"><div class="chat-empty">Загрузка тарифов…</div></div>
    <a class="ghost-btn" style="display:block;text-align:center;text-decoration:none;" href="__SUPPORT_URL__" target="_blank" rel="noopener noreferrer">Оплата картой / СБП — скоро. Написать в поддержку</a>
  </div>
</div>

<!-- Paywall modal -->
<div id="paywallModal" class="modal-overlay">
  <div class="modal-card">
    <button class="close-x" onclick="closeModal('paywallModal')">✕</button>
    <h2>Баланс исчерпан</h2>
    <p>Бесплатный запрос уже использован. Чтобы продолжить общение с ASCEND AI, выбери подходящий пакет запросов.</p>
    <button class="primary-btn" onclick="closeModal('paywallModal');openPricingModal();">Посмотреть тарифы</button>
  </div>
</div>

<script>
const DEVICE_KEY = "ascend_device_id";
const CHAT_KEY = "ascend_current_chat";
const TERMS_KEY = "ascend_terms_accepted";

let deviceId = localStorage.getItem(DEVICE_KEY);
if (!deviceId) {
    deviceId = crypto.randomUUID();
    localStorage.setItem(DEVICE_KEY, deviceId);
}

let currentChatId = localStorage.getItem(CHAT_KEY);
let sending = false;

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

/* ---------------- modals ---------------- */
function openModal(id){ document.getElementById(id).classList.add("open"); }
function closeModal(id){ document.getElementById(id).classList.remove("open"); }
function openInfoModal(){ openModal("infoModal"); }
function openPricingModal(){ openModal("pricingModal"); loadPricing(); }

function maybeShowWelcome(){
    if (!localStorage.getItem(TERMS_KEY)) {
        openModal("welcomeModal");
    }
}
function acceptTerms(){
    localStorage.setItem(TERMS_KEY, "1");
    closeModal("welcomeModal");
}

/* ---------------- sidebar ---------------- */
function openSidebar(){
    document.getElementById("sidebar").classList.add("open");
    document.getElementById("sidebarOverlay").classList.add("open");
    loadChatList();
}
function closeSidebar(){
    document.getElementById("sidebar").classList.remove("open");
    document.getElementById("sidebarOverlay").classList.remove("open");
}

async function loadChatList(){
    const list = document.getElementById("chatList");
    try {
        const res = await fetch("/api/chats?device_id=" + encodeURIComponent(deviceId));
        const chats = await res.json();
        if (!Array.isArray(chats) || chats.length === 0) {
            list.innerHTML = '<div class="chat-empty">Пока нет сохранённых чатов</div>';
            return;
        }
        list.innerHTML = "";
        chats.forEach(chat => {
            const item = document.createElement("div");
            item.className = "chat-item" + (chat.session_id === currentChatId ? " active" : "");
            item.onclick = () => switchChat(chat.session_id);
            item.innerHTML = `
                <span class="title">${escapeHtml(chat.title || "Новый чат")}</span>
                <button class="del" title="Удалить" onclick="deleteChat(event,'${chat.session_id}')">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none"><path d="M6 7h12M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2m-8 0 1 13a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1l1-13" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>
                </button>`;
            list.appendChild(item);
        });
    } catch (e) {
        list.innerHTML = '<div class="chat-empty">Не удалось загрузить список</div>';
    }
}

async function createNewChat(){
    try {
        const res = await fetch("/api/chats/new", {
            method: "POST",
            headers: {"Content-Type":"application/json"},
            body: JSON.stringify({device_id: deviceId})
        });
        const data = await res.json();
        currentChatId = data.session_id;
        localStorage.setItem(CHAT_KEY, currentChatId);
        renderWelcomeOnly();
        closeSidebar();
        loadChatList();
    } catch (e) {
        console.error(e);
    }
}

async function switchChat(sessionId){
    currentChatId = sessionId;
    localStorage.setItem(CHAT_KEY, sessionId);
    closeSidebar();
    await loadCurrentChatMessages();
    loadChatList();
}

async function deleteChat(event, sessionId){
    event.stopPropagation();
    if (!confirm("Удалить этот чат безвозвратно?")) return;
    try {
        await fetch("/api/chats/" + sessionId, { method: "DELETE" });
    } catch (e) { console.error(e); }
    if (sessionId === currentChatId) {
        await createNewChat();
    } else {
        loadChatList();
    }
}

function renderWelcomeOnly(){
    const inner = document.getElementById("messagesInner");
    inner.innerHTML = `
        <div class="message ai">
            <div class="avatar">🧠</div>
            <div class="bubble">Новый чат начат. О чём хочешь спросить?</div>
        </div>`;
}

async function loadCurrentChatMessages(){
    const inner = document.getElementById("messagesInner");
    try {
        const res = await fetch("/api/chats/" + currentChatId + "/messages");
        const msgs = await res.json();
        if (!Array.isArray(msgs) || msgs.length === 0) {
            renderWelcomeOnly();
            return;
        }
        inner.innerHTML = "";
        msgs.forEach(m => addMessage(m.role === "user" ? "user" : "ai", m.content));
    } catch (e) {
        renderWelcomeOnly();
    }
}

/* ---------------- messages ---------------- */
function scrollToBottom(){
    const box = document.getElementById("messages");
    box.scrollTop = box.scrollHeight;
}

function addMessage(role, text){
    const inner = document.getElementById("messagesInner");
    const wrapper = document.createElement("div");
    wrapper.className = "message " + (role === "user" ? "user" : "ai");
    if (role === "user") {
        wrapper.innerHTML = `<div class="bubble"></div>`;
    } else {
        wrapper.innerHTML = `<div class="avatar">🧠</div><div class="bubble"></div>`;
    }
    wrapper.querySelector(".bubble").textContent = text;
    inner.appendChild(wrapper);
    scrollToBottom();
}

function addSources(sources){
    if (!sources || sources.length === 0) return;
    const inner = document.getElementById("messagesInner");
    const wrap = document.createElement("div");
    wrap.className = "sources";
    let html = '<div class="sources-title">🌐 Источники</div>';
    sources.forEach(s => {
        html += `<div class="source-card"><div class="s-title">${escapeHtml(s.title || s.url)}</div><a href="${escapeHtml(s.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(s.url)}</a></div>`;
    });
    wrap.innerHTML = html;
    inner.appendChild(wrap);
    scrollToBottom();
}

const TYPING_LABELS = ["Думаю…", "Ищу лучший ответ…", "Собираю мысли…", "Почти готово…"];

function showTyping(){
    const inner = document.getElementById("messagesInner");
    const row = document.createElement("div");
    row.className = "typing-row";
    row.id = "typingRow";
    row.innerHTML = `<div class="avatar">🧠</div><div class="typing-bubble"><span></span><span></span><span></span><span class="typing-label">${TYPING_LABELS[0]}</span></div>`;
    inner.appendChild(row);
    scrollToBottom();

    let i = 0;
    row.dataset.timer = setInterval(() => {
        i = (i + 1) % TYPING_LABELS.length;
        const label = row.querySelector(".typing-label");
        if (label) label.textContent = TYPING_LABELS[i];
    }, 1800);
}
function hideTyping(){
    const row = document.getElementById("typingRow");
    if (row) {
        clearInterval(row.dataset.timer);
        row.remove();
    }
}

function autosize(el){
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 150) + "px";
}
document.addEventListener("DOMContentLoaded", () => {
    const input = document.getElementById("messageInput");
    input.addEventListener("input", () => autosize(input));
    input.addEventListener("keydown", function(event){
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });
});

async function sendMessage(){
    if (sending) return;
    const input = document.getElementById("messageInput");
    const button = document.getElementById("sendButton");
    const message = input.value.trim();
    if (!message) return;
    if (message.length > 5000) { alert("Сообщение слишком длинное."); return; }

    if (!currentChatId) { await createNewChat(); }

    sending = true;
    addMessage("user", message);
    input.value = "";
    autosize(input);
    button.disabled = true;
    showTyping();

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: currentChatId, message: message })
        });

        let data;
        try { data = await response.json(); } catch { data = { detail: "Сервер вернул некорректный ответ." }; }

        hideTyping();

        if (response.status === 402) {
            addMessage("ai", data.detail || "Бесплатный запрос использован. Пополните баланс, чтобы продолжить.");
            openModal("paywallModal");
        } else if (!response.ok) {
            addMessage("ai", data.detail || "Ошибка сервера.");
        } else {
            addMessage("ai", data.answer || "Сервер не вернул ответ.");
            addSources(data.sources);
            updateBalancePill(data.credits);
            loadChatList();
        }
    } catch (error) {
        console.error(error);
        hideTyping();
        addMessage("ai", "Ошибка соединения с сервером.");
    }

    button.disabled = false;
    sending = false;
}

/* ---------------- balance / pricing ---------------- */
function updateBalancePill(credits){
    const pill = document.getElementById("balancePill");
    if (!credits) { loadBalance(); return; }
    if (credits.mode === "free") {
        pill.textContent = "🎁 Бесплатный запрос использован";
        pill.classList.remove("warn");
    } else if (credits.mode === "paid" && credits.remaining > 0) {
        pill.textContent = "💳 Баланс: " + credits.remaining;
        pill.classList.remove("warn");
    } else {
        pill.textContent = "⚠️ Пополнить баланс";
        pill.classList.add("warn");
    }
}

async function loadBalance(){
    const pill = document.getElementById("balancePill");
    try {
        const res = await fetch("/api/credits");
        const data = await res.json();
        if (!data.free_used) {
            pill.textContent = "🎁 Первый запрос бесплатно";
            pill.classList.remove("warn");
        } else if (data.credits > 0) {
            pill.textContent = "💳 Баланс: " + data.credits;
            pill.classList.remove("warn");
        } else {
            pill.textContent = "⚠️ Пополнить баланс";
            pill.classList.add("warn");
        }
    } catch (e) {
        pill.textContent = "💳 Тарифы";
    }
}

async function loadPricing(){
    const box = document.getElementById("plansList");
    try {
        const res = await fetch("/api/pricing");
        const plans = await res.json();
        box.innerHTML = "";
        plans.forEach(p => {
            const card = document.createElement("div");
            card.className = "plan-card" + (p.popular ? " popular" : "");
            card.innerHTML = `
                ${p.popular ? '<div class="plan-badge">Популярный</div>' : ""}
                <div class="plan-top">
                    <div class="plan-name">${escapeHtml(p.name)}</div>
                    <div class="plan-price">${p.price}₽</div>
                </div>
                <div class="plan-meta">${p.requests ? p.requests + " запросов" : "Без ограничений"} · ${escapeHtml(p.per_request)}</div>
            `;
            box.appendChild(card);
        });
    } catch (e) {
        box.innerHTML = '<div class="chat-empty">Не удалось загрузить тарифы</div>';
    }
}

/* ---------------- init ---------------- */
(async function init(){
    maybeShowWelcome();
    loadBalance();

    if (!currentChatId) {
        await createNewChat();
    } else {
        await loadCurrentChatMessages();
    }
})();
</script>
</body>
</html>
"""

HTML = (
    HTML.replace("__PRIVACY_URL__", PRIVACY_URL)
    .replace("__TERMS_URL__", TERMS_URL)
    .replace("__SUPPORT_URL__", SUPPORT_TELEGRAM_URL)
    .replace("__SUPPORT_HANDLE__", SUPPORT_TELEGRAM_HANDLE)
)


# ============================================================
# LEGAL / SUPPORT / PRICING PAGES
# ============================================================

PROJECT_NAME_LEGAL = "ASCEND AI"

LEGAL_PAGE_STYLE = """
<style>
* { box-sizing: border-box; }
body { margin: 0; background: #09090b; color: #f4f4f5; font-family: -apple-system, Inter, Arial, sans-serif; line-height: 1.7; }
.wrap { max-width: 720px; margin: 0 auto; padding: 40px 20px 80px; }
h1 { font-size: 28px; margin-bottom: 6px; }
.updated { color: #898991; font-size: 13px; margin-bottom: 30px; }
h2 { font-size: 18px; margin-top: 30px; color: #f7d45b; }
p, li { color: #c7c7cc; font-size: 14px; }
a { color: #f7d45b; }
.back { display: inline-block; margin-bottom: 20px; color: #898991; text-decoration: none; font-size: 13px; }
.price-card { background: #141416; border: 1px solid #28282c; border-radius: 16px; padding: 20px; margin: 16px 0; position:relative; }
.price-card .amount { font-size: 22px; font-weight: 900; color: #f7d45b; }
.price-card .meta { color:#898991; font-size:12.5px; margin-top:6px; }
.price-badge { position:absolute; top:-10px; right:16px; background:#f7d45b; color:#111; font-size:11px; font-weight:800; padding:3px 10px; border-radius:999px; }
.contact-btn { display: inline-block; margin-top: 10px; background: #f7d45b; color: #111; padding: 12px 20px; border-radius: 12px; text-decoration: none; font-weight: 800; }
</style>
"""


@app.get("/privacy")
async def privacy_page():
    return RedirectResponse(PRIVACY_URL)


@app.get("/terms")
async def terms_page():
    return RedirectResponse(TERMS_URL)


@app.get("/contacts", response_class=HTMLResponse)
async def contacts_page():
    return f"""
<!DOCTYPE html><html lang="ru"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Поддержка — {PROJECT_NAME_LEGAL}</title>
{LEGAL_PAGE_STYLE}</head><body><div class="wrap">
<a class="back" href="/">← Назад на сайт</a>
<h1>Поддержка</h1>
<p>Если у тебя вопрос по работе сервиса, оплате или ты хочешь запросить удаление своих данных — напиши нам.</p>
<a class="contact-btn" href="{SUPPORT_TELEGRAM_URL}" target="_blank" rel="noopener noreferrer">Написать в Telegram: {SUPPORT_TELEGRAM_HANDLE}</a>
<p style="margin-top:30px;">Среднее время ответа — до 24 часов.</p>
<p><a href="{PRIVACY_URL}" target="_blank" rel="noopener noreferrer">Политика конфиденциальности</a> · <a href="{TERMS_URL}" target="_blank" rel="noopener noreferrer">Пользовательское соглашение</a></p>
</div></body></html>
"""


@app.get("/pricing", response_class=HTMLResponse)
async def pricing_page():
    cards_html = ""
    for plan in PRICING_PLANS:
        badge = '<div class="price-badge">Популярный</div>' if plan.get("popular") else ""
        requests_label = f'{plan["requests"]} запросов' if plan.get("requests") else "Без ограничений"
        cards_html += f"""
<div class="price-card">
{badge}
<div class="amount">{plan['price']} ₽</div>
<div>{plan['name']} — {requests_label}</div>
<div class="meta">{plan['per_request']}</div>
</div>
"""

    return f"""
<!DOCTYPE html><html lang="ru"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Тарифы — {PROJECT_NAME_LEGAL}</title>
{LEGAL_PAGE_STYLE}</head><body><div class="wrap">
<a class="back" href="/">← Назад на сайт</a>
<h1>Тарифы</h1>
<p>Первый запрос — бесплатно. Дальше — недорогие пакеты запросов, без автосписаний и подписок "по умолчанию".</p>
{cards_html}
<p style="margin-top:30px;">Оплата картой / СБП скоро появится на сайте. Пока — обращайтесь в <a href="/contacts">поддержку</a>.</p>
</div></body></html>
"""


# ============================================================
# ROOT
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "brain_ready": brain.ready,
        "knowledge": len(knowledge_cache),
        "search_engines": [name for name, _ in SEARCH_ENGINES],
        "llm_enabled": llm_available(),
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_SECRET_KEY),
    }


# ============================================================
# SMALL TALK / GREETINGS
# ============================================================

GREETING_WORDS = {
    "привет", "здравствуй", "здравствуйте", "приветик", "хай",
    "хеллоу", "хелло", "йо", "ку", "здарова", "здорово",
}

FAREWELL_WORDS = {"пока", "прощай", "досвидания", "бывай", "увидимся"}

THANKS_WORDS = {"спасибо", "благодарю", "спс", "сенкс", "thanks"}

HOWAREYOU_PHRASES = {
    "как дела", "как ты", "как жизнь", "как оно", "че как", "что нового",
}

GREETING_REPLIES = [
    "Привет! Расскажи, что тебя интересует — кожа, внешность, питание, "
    "сон или тренировки?",
    "Привет 👋 С чем помочь сегодня?",
]

FAREWELL_REPLIES = ["Пока! Возвращайся, если появятся вопросы 🙂"]

THANKS_REPLIES = ["Пожалуйста! Обращайся, если будут ещё вопросы."]

HOWAREYOU_REPLIES = [
    "Спасибо, у меня всё в порядке! А у тебя как дела? "
    "И расскажи, чем могу помочь.",
]


def detect_small_talk(message):
    normalized = normalize(message)

    for phrase in HOWAREYOU_PHRASES:
        if phrase in normalized:
            return random.choice(HOWAREYOU_REPLIES)

    words = normalized.split()

    if not words or len(words) > 4:
        return None

    word_set = set(words)

    if word_set & GREETING_WORDS and word_set <= (GREETING_WORDS | {"как", "дела", "там"}):
        return random.choice(GREETING_REPLIES)

    if word_set & FAREWELL_WORDS and word_set <= FAREWELL_WORDS:
        return random.choice(FAREWELL_REPLIES)

    if word_set & THANKS_WORDS and word_set <= THANKS_WORDS:
        return random.choice(THANKS_REPLIES)

    return None


@app.post("/api/chat")
def chat(data: ChatRequest, request: Request):
    """
    Обычная (sync) def — FastAPI сам уводит в threadpool, поэтому
    блокирующие вызовы (requests к поисковикам/LLM/Supabase) не вешают
    event loop и не мешают другим конкурентным запросам.
    """
    message = data.message.strip()

    if not message:
        raise HTTPException(400, "Пустой запрос.")

    if len(message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(400, "Сообщение слишком длинное.")

    memory = get_memory(data.session_id)
    is_new_chat = len(memory) == 0

    save_message(data.session_id, "user", message)

    if is_new_chat:
        touch_chat_session(data.session_id, title=make_chat_title(message))
    else:
        touch_chat_session(data.session_id)

    small_talk_answer = detect_small_talk(message)

    if small_talk_answer:
        assistant_message = save_message(data.session_id, "assistant", small_talk_answer)

        return {
            "answer": small_talk_answer,
            "sources": [],
            "knowledge_found": False,
            "web_found": False,
            "memory_used": len(memory),
            "message_id": assistant_message.get("id") if assistant_message else None,
            "credits": {"mode": "free", "remaining": None},
        }

    # ------------------------------------------------------
    # Проверка баланса (первый запрос бесплатно, дальше — по балансу)
    # ------------------------------------------------------
    client_key = get_client_key(request)
    access = consume_access(client_key)

    if not access["allowed"]:
        raise HTTPException(
            status_code=402,
            detail=(
                "Бесплатный запрос уже использован, а баланс пуст. "
                "Пополните баланс, чтобы продолжить общение — тарифы доступны "
                "в разделе «Инфо»."
            ),
        )

    local_results = search_local_knowledge(message)

    direct_mode = is_llm_direct_mode() and llm_available()

    if direct_mode:
        web_results = []
    else:
        web_results = collect_web_information(message)
        save_web_sources(data.session_id, message, web_results)

    answer = generate_response(message, memory, local_results, web_results)

    assistant_message = save_message(data.session_id, "assistant", answer)

    category = None
    if local_results:
        category = local_results[0][1].get("category")

    save_training_log(message, answer, category or "web", "search")

    sources = []
    for result in web_results:
        sources.append({"title": result.get("title", ""), "url": result.get("url", "")})

    return {
        "answer": answer,
        "sources": sources,
        "knowledge_found": bool(local_results),
        "web_found": bool(web_results),
        "memory_used": len(memory),
        "message_id": assistant_message.get("id") if assistant_message else None,
        "credits": {"mode": access["mode"], "remaining": access["credits"]},
    }


# ============================================================
# CHATS (список / создание / переключение / удаление)
# ============================================================

@app.post("/api/chats/new")
def new_chat(data: NewChatBody):
    session_id = secrets.token_hex(16)
    create_chat_session(session_id, data.device_id, "Новый чат")
    return {"session_id": session_id, "title": "Новый чат"}


@app.get("/api/chats")
def get_chats(device_id: str = ""):
    if not device_id:
        return []
    return list_chat_sessions(device_id)


@app.get("/api/chats/{session_id}/messages")
def chat_messages_endpoint(session_id: str):
    return get_chat_messages(session_id)


@app.delete("/api/chats/{session_id}")
def delete_chat(session_id: str):
    delete_chat_session(session_id)
    return {"success": True}


# ============================================================
# CREDITS / PRICING (публичные эндпоинты)
# ============================================================

@app.get("/api/credits")
def credits_endpoint(request: Request):
    key = get_client_key(request)
    state = get_credit_state(key)
    return {
        "free_used": bool(state.get("free_used")),
        "credits": int(state.get("credits") or 0),
    }


@app.get("/api/pricing")
def pricing_endpoint():
    return PRICING_PLANS


# ============================================================
# MANUAL CREDIT TOP-UP (служебный эндпоинт, БЕЗ веб-панели)
# ============================================================
#
# Веб-панели администратора больше нет — API-ключи задаются только
# переменными окружения (Render → Environment), их нигде не нужно
# вводить в браузере. Но до подключения автооплаты по СБП всё ещё
# нужен способ вручную начислить купленные запросы конкретному IP.
# Для этого используется этот эндпоинт — вызывать его нужно НЕ из
# браузера, а напрямую (curl/Postman/httpie), передав заголовок
# X-Admin-Password со значением переменной окружения ADMIN_PASSWORD:
#
#   curl -X POST https://<ваш-домен>/api/manual/credits \
#        -H "X-Admin-Password: <ADMIN_PASSWORD>" \
#        -H "Content-Type: application/json" \
#        -d '{"ip": "1.2.3.4", "credits": 50}'
#

def check_manual_secret(request: Request):
    provided = request.headers.get("X-Admin-Password", "")
    if not provided or not secrets.compare_digest(
        provided.encode("utf-8"), ADMIN_PASSWORD.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Нет доступа.")


@app.post("/api/manual/credits")
def manual_topup_credits(request: Request, data: CreditTopUp):
    check_manual_secret(request)

    ip = data.ip.strip()
    if not ip:
        raise HTTPException(400, "Укажите IP-адрес.")

    key = hashlib.sha256(f"ascend-credits:{ip}".encode("utf-8")).hexdigest()[:40]
    state = get_credit_state(key)
    new_credits = int(state.get("credits") or 0) + int(data.credits)
    save_credit_state(key, new_credits, True)

    return {"success": True, "key": key, "credits": new_credits}


# ============================================================
# FEEDBACK
# ============================================================

@app.post("/api/feedback")
def feedback(data: FeedbackRequest):
    if data.rating < 1 or data.rating > 5:
        raise HTTPException(400, "Оценка должна быть от 1 до 5.")

    if SUPABASE_URL and SUPABASE_SECRET_KEY:
        supabase_request(
            "POST",
            "ai_feedback",
            {
                "session_id": data.session_id,
                "message_id": data.message_id,
                "rating": data.rating,
                "comment": data.comment or "",
            },
        )

    return {"success": True}


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup():
    try:
        load_knowledge()
    except Exception as e:
        print("STARTUP ERROR in load_knowledge:", repr(e), flush=True)
        traceback.print_exc()

    print("", flush=True)
    print("=" * 60, flush=True)
    print("                  ASCEND AI", flush=True)
    print("=" * 60, flush=True)
    print("Knowledge:", len(knowledge_cache), flush=True)
    print("Neural brain:", brain.ready, flush=True)
    print("Search engines:", [name for name, _ in SEARCH_ENGINES], flush=True)
    print("SearXNG instances:", len(SEARXNG_INSTANCES), flush=True)
    print("LLM enabled:", llm_available(), flush=True)
    print("Supabase configured:", bool(SUPABASE_URL and SUPABASE_SECRET_KEY), flush=True)

    if SUPABASE_URL and SUPABASE_SECRET_KEY:
        print(
            "Не забудьте создать таблицы chat_sessions и user_credits "
            "(SQL — см. комментарии в коде рядом с функциями работы с ними).",
            flush=True,
        )

    if ADMIN_PASSWORD == "CHANGE_THIS_PASSWORD":
        print("!" * 60, flush=True)
        print("ВНИМАНИЕ: ADMIN_PASSWORD не задан — используется значение", flush=True)
        print("по умолчанию. Задайте переменную окружения ADMIN_PASSWORD", flush=True)
        print("на Render перед тем, как открывать сервис пользователям —", flush=True)
        print("он защищает служебный эндпоинт /api/manual/credits.", flush=True)
        print("!" * 60, flush=True)

    print("=" * 60, flush=True)
    print("")
