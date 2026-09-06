import os
import re
import json
import math
import hashlib
import hmac
import secrets
import time
import random

from typing import Optional
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
# LLM SETTINGS (API-ключи, а НЕ логин/пароль от личных аккаунтов)
# ============================================================
#
# ВАЖНО: здесь используются официальные API-ключи провайдеров,
# а не email/пароль от личного кабинета. У DeepSeek и у Qwen
# (Alibaba DashScope) нет программного входа по паролю — только
# API-ключи, которые выдаются в личном кабинете разработчика:
#
#   DeepSeek:  https://platform.deepseek.com  -> API Keys
#   Qwen:      https://dashscope.console.aliyun.com -> API-Key Management
#   OpenRouter (бесплатные модели): https://openrouter.ai -> Keys
#
# Ключи можно задать двумя способами:
#   1. Через переменные окружения (ниже) — значения по умолчанию.
#   2. Через админ-панель (/api/admin/settings) — перекрывают env
#      и сохраняются в Supabase (таблица app_settings), если она
#      настроена, иначе живут только в памяти процесса до рестарта.
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

# Бесплатные модели OpenRouter. Список периодически "протухает" —
# если увидите в логах "LLM: all models failed", загляните на
# openrouter.ai/models?max_price=0 и обновите список.
OPENROUTER_FREE_MODELS = [
    "deepseek/deepseek-chat-v3.1:free",
    "qwen/qwen3-235b-a22b:free",
    "deepseek/deepseek-r1-distill-qwen-14b:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]

DEEPSEEK_MODEL = "deepseek-chat"
QWEN_MODEL = "qwen-plus"

# provod.ai — российский агрегатор моделей (OpenAI-совместимый API).
# Имя модели у разных агрегаторов оформлено по-разному (например,
# "xiaomi/mimo-v2.5" или просто "mimo-v2.5"), поэтому оно НЕ зашито
# жёстко: указывается в переменной окружения PROVOD_MODEL или через
# админ-панель — скопируйте точное имя модели из личного кабинета
# provod.ai (обычно отображается рядом с ценой модели).
PROVOD_DEFAULT_MODEL = os.getenv("PROVOD_MODEL", "xiaomi/mimo-v2.5")

# Значения по умолчанию из переменных окружения; runtime_settings
# ниже может их переопределить через админ-панель. Ключи можно задать
# либо в Environment Variables на Render, либо через админку — работает
# любой из способов, админка просто перекрывает env при сохранении.
_DEFAULT_SETTINGS = {
    "openrouter_api_key": os.getenv("OPENROUTER_API_KEY", ""),
    "deepseek_api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    "qwen_api_key": os.getenv("QWEN_API_KEY", ""),
    "provod_api_key": os.getenv("PROVOD_API_KEY", ""),
    "provod_model": PROVOD_DEFAULT_MODEL,
    # "Прямой LLM режим": каждый вопрос сразу идёт к LLM (провайдеры
    # пробуются в обычном порядке: provod -> openrouter -> deepseek ->
    # qwen), локальная база используется как подсказка если совпала,
    # а веб-поиск (нестабильные публичные SearXNG/DuckDuckGo) вообще
    # не задействуется. Включено по умолчанию, если задан хотя бы
    # один LLM-ключ — так надёжнее, чем зависеть от внешних поисковиков.
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
# SEARCH ENGINES (multi-provider fallback chain)
# ============================================================
#
# Порядок отказоустойчивости:
#
#   1. SearXNG (несколько публичных инстансов, с ретраями)
#   2. DuckDuckGo HTML (не требует ключа, мягче лимиты для серверных IP)
#
# Bing сознательно исключён: с 2025-2026 их разметка оборачивает
# ссылки в JS-редирект (bing.com/ck/a), который без headless-браузера
# не раскрывается — скрапинг просто возвращает межстраничные заглушки
# "please click here if the page does not redirect automatically".
#
# Если оба уровня не дали результатов — возвращаем пустой список,
# и generate_response() честно сообщает пользователю, что поиск не удался.
#

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

# Сколько раз повторить запрос к ОДНОМУ И ТОМУ ЖЕ инстансу при 429,
# прежде чем переходить к следующему.
SEARXNG_MAX_RETRIES_PER_INSTANCE = 2

# Базовая задержка (сек) для экспоненциального backoff при 429.
SEARXNG_RETRY_BACKOFF_BASE = 1.5


# ============================================================
# LIMITS
# ============================================================

MAX_MEMORY = 30
MAX_SEARCH_RESULTS = 6
MAX_SOURCE_TEXT = 3500
MAX_MESSAGE_LENGTH = 5000
PAGE_TIMEOUT = 12
MAX_LLM_HISTORY_MESSAGES = 12  # сколько последних сообщений диалога передавать в LLM как контекст


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title=APP_NAME, version="2.1.1")

# ------------------------------------------------------------
# CORS — разрешаем запросы отовсюду. Само по себе это не решает
# все возможные проблемы, но полностью исключает CORS как причину
# "Некорректный ответ сервера" / странных провалов fetch() без логов.
# ------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# ГЛОБАЛЬНЫЙ ОБРАБОТЧИК ОШИБОК
# ------------------------------------------------------------
#
# По умолчанию необработанное исключение в FastAPI/Starlette отдаёт
# клиенту НЕ JSON, а голый текст "Internal Server Error" — именно
# из-за этого фронтенд не мог распарсить ответ ("Некорректный ответ
# сервера"). Теперь ЛЮБАЯ ошибка сервера гарантированно:
#   1) возвращается клиенту как валидный JSON с понятным полем detail;
#   2) печатается в лог ПОЛНОСТЬЮ (с traceback), гарантированно flush'ится.
#

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
    """
    Минимальный REST-клиент Supabase.

    `prefer` позволяет переопределить заголовок Prefer — это нужно,
    например, для upsert (POST + ?on_conflict=... требует
    "resolution=merge-duplicates", иначе PostgREST просто вернёт
    ошибку конфликта первичного ключа при повторной записи).
    """

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
    """
    ИСПРАВЛЕНО: раньше вариант синонима искался как обычная подстрока
    (`normalized_variant in normalized_text`), из-за чего короткие
    варианты вроде "лица" (синоним категории "лицо") ложно совпадали
    внутри других слов — например, внутри "столица". Теперь сравнение
    идёт по целым словам/фразам: и текст, и вариант оборачиваются
    пробелами, поэтому совпадение засчитывается только на границе слова.
    """
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
    """
    Простая классификационная нейросеть.
    Она НЕ генерирует текст, используется только как
    дополнительный сигнал для классификации вопроса.
    """

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


# ============================================================
# LOAD KNOWLEDGE
# ============================================================

def _with_stable_ids(items):
    """
    ИСПРАВЛЕНО: записи из DEFAULT_KNOWLEDGE не имели поля "id" (оно
    появляется только у строк из Supabase), поэтому удаление знания
    через админку (admin_delete_knowledge, сравнение по id) молча
    ничего не находило, если Supabase не настроен или пуст. Теперь
    каждой дефолтной записи присваивается стабильный id на основе
    её текста.
    """
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

        # категория сама по себе не создаёт совпадение
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


# ============================================================
# SEARCH ENGINE 1: SEARXNG (with retry/backoff per instance)
# ============================================================

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

        print("")
        print("------------------------------------------")
        print("SEARXNG QUERY:", query)
        print("SEARXNG INSTANCE:", base)

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

                print(
                    f"SEARXNG HTTP (attempt {attempt}):",
                    response.status_code,
                )

            except Exception as e:
                print(f"SEARXNG REQUEST ERROR (attempt {attempt}):", repr(e))
                break  # network-level failure: no point retrying this instance

            if response.status_code == 429:
                # Rate limited — задержка и повтор на ТОМ ЖЕ инстансе,
                # прежде чем сдаться и перейти к следующему.
                if attempt < SEARXNG_MAX_RETRIES_PER_INSTANCE:
                    delay = SEARXNG_RETRY_BACKOFF_BASE * attempt
                    print(f"SEARXNG 429 — retry in {delay:.1f}s")
                    time.sleep(delay)
                    continue
                else:
                    print("SEARXNG BAD STATUS: 429 (giving up on instance)")
                    break

            if response.status_code >= 400:
                print("SEARXNG BAD STATUS:", response.status_code)
                break

            content_type = response.headers.get("content-type", "").lower()
            if "json" not in content_type:
                print("SEARXNG NOT JSON:", content_type)
                break

            try:
                payload = response.json()
            except Exception as e:
                print("SEARXNG JSON ERROR:", repr(e))
                break

            results = _parse_searxng_payload(payload, limit)

            print("SEARXNG RESULTS:", len(results))
            for i, r in enumerate(results, start=1):
                print(f"SEARXNG RESULT {i}:", r.get("title", "")[:120], r.get("url", ""))

            if results:
                print("SEARXNG SEARCH SUCCESS via", base)
                print("------------------------------------------")
                return results

            print("SEARXNG EMPTY RESULTS")
            break

    print("SEARXNG SEARCH FAILED: ALL INSTANCES")
    print("------------------------------------------")
    return []


# ============================================================
# SEARCH ENGINE 2: DUCKDUCKGO HTML (fallback, no API key needed)
# ============================================================

def duckduckgo_html_search(query, limit=MAX_SEARCH_RESULTS):
    query = query.strip()
    if not query:
        return []

    url = "https://html.duckduckgo.com/html/"

    print("")
    print("------------------------------------------")
    print("DUCKDUCKGO QUERY:", query)

    try:
        response = requests.post(
            url,
            data={"q": query, "kl": "ru-ru"},
            headers=BROWSER_HEADERS,
            timeout=SEARXNG_TIMEOUT,
        )

        print("DUCKDUCKGO HTTP:", response.status_code)

    except Exception as e:
        print("DUCKDUCKGO REQUEST ERROR:", repr(e))
        return []

    if response.status_code >= 400:
        print("DUCKDUCKGO BAD STATUS:", response.status_code)
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

    print("DUCKDUCKGO RESULTS:", len(results))
    for i, r in enumerate(results, start=1):
        print(f"DUCKDUCKGO RESULT {i}:", r.get("title", "")[:120], r.get("url", ""))
    print("------------------------------------------")

    return results


# ============================================================
# UNIFIED WEB SEARCH (tries all engines in order)
# ============================================================

# Каждый элемент: (имя_для_логов, функция)
SEARCH_ENGINES = [
    ("searxng", searxng_search),
    ("duckduckgo", duckduckgo_html_search),
]


# ============================================================
# GARBAGE / REDIRECT PAGE DETECTION
# ============================================================
#
# Некоторые поисковики (Bing и другие) оборачивают ссылки в JS-редирект.
# При скрапинге без браузера получаем не контент, а межстраничную
# заглушку. Отфильтровываем такие результаты, чтобы они не попадали
# ни в ответ, ни в источники.

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


# ============================================================
# FETCH WEB PAGE
# ============================================================

def fetch_page_text(url):
    if not valid_http_url(url):
        return ""

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ASCEND-AI/2.1)",
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml",
    }

    print("FETCH SOURCE:", url)

    try:
        response = requests.get(
            url, headers=headers, timeout=PAGE_TIMEOUT, allow_redirects=True
        )

        print("SOURCE HTTP:", response.status_code)

        if response.status_code >= 400:
            print("SOURCE ERROR STATUS")
            return ""

        content_type = response.headers.get("content-type", "").lower()
        if "text/html" not in content_type:
            print("SOURCE NOT HTML:", content_type)
            return ""

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form"]):
            tag.decompose()

        text = clean_text(soup.get_text(" ", strip=True))
        text = text[:MAX_SOURCE_TEXT]

        print("SOURCE TEXT LENGTH:", len(text))

        return text

    except Exception as e:
        print("SOURCE FETCH ERROR:", repr(e))
        return ""


# ============================================================
# COLLECT WEB INFORMATION
# ============================================================

def collect_web_information(query):
    print("")
    print("==========================================")
    print("WEB SEARCH START")
    print("QUERY:", query)

    search_results, engine_used = web_search_with_fallback(query)

    if not search_results:
        print("WEB SEARCH: no engine returned results.")
        print("==========================================")
        return []

    print("WEB SEARCH ENGINE:", engine_used)
    print("WEB SEARCH RESULTS:", len(search_results))

    enriched = []
    for index, result in enumerate(search_results, start=1):
        page_text = fetch_page_text(result["url"])

        if is_redirect_stub(page_text):
            print(f"SOURCE {index}: SKIPPED (redirect stub / no real content)")
            page_text = ""

        # Если и страница не открылась (или это заглушка), но есть
        # осмысленный сниппет от поисковика — используем хотя бы его.
        # Полностью бесполезные результаты (ни текста, ни сниппета)
        # отбрасываем совсем, чтобы не засорять контекст и источники.
        if not page_text and not clean_text(result.get("snippet", "")):
            print(f"SOURCE {index}: DROPPED (no content, no snippet)")
            continue

        enriched.append({**result, "page_text": page_text})

        print(
            f"SOURCE {index}: title={result.get('title', '')[:100]} "
            f"text={len(page_text)} snippet={len(result.get('snippet', ''))}"
        )

    print("WEB SEARCH COMPLETE")
    print("==========================================")

    return enriched


# ============================================================
# SAVE WEB SOURCES
# ============================================================

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


# ============================================================
# WEB CONTEXT
# ============================================================

def build_web_context(results):
    """
    Контекст С заголовками источников — используется только как вход
    для LLM (модель сама разберёт структуру). НЕ использовать для
    экстрактивного нарезания предложений — заголовки будут утекать
    в "предложения" (см. clean_web_text для этого случая).
    """

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
    """
    Чистый текст БЕЗ заголовков/URL источников — только сам контент.
    Используется для экстрактивного резюме (rank_sentences), чтобы
    в ответ не попадали куски вида "ИСТОЧНИК 1 Название: ... URL: ...".
    """

    pieces = []
    for item in results:
        text = item.get("page_text") or item.get("snippet") or ""
        text = clean_text(text)
        if text:
            pieces.append(text)

    return "\n\n".join(pieces)


# ============================================================
# SENTENCES
# ============================================================

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
# SETTINGS PERSISTENCE (Supabase key-value table `app_settings`)
# ============================================================

SETTINGS_KEYS = (
    "openrouter_api_key",
    "deepseek_api_key",
    "qwen_api_key",
    "provod_api_key",
    "provod_model",
    "llm_direct_mode",
)


def load_settings():
    """Подтягивает сохранённые ключи из Supabase поверх значений из env."""

    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        return

    rows = supabase_request(
        "GET",
        "app_settings",
        params={"select": "key,value"},
    )

    for row in rows or []:
        key = row.get("key")
        value = row.get("value")
        if key in SETTINGS_KEYS and value:
            runtime_settings[key] = value

    print("Settings loaded from Supabase:", [k for k in SETTINGS_KEYS if runtime_settings.get(k)])


def save_setting(key, value):
    """
    Сохраняет ключ в память процесса и, если настроен Supabase, в БД (upsert).

    ИСПРАВЛЕНО: запись раньше уходила как обычный POST с ?on_conflict=key,
    но БЕЗ заголовка Prefer: resolution=merge-duplicates. PostgREST в этом
    случае не делает upsert, а просто пытается вставить новую строку — при
    повторном сохранении того же ключа (например, обновление API-ключа)
    Supabase отвечал конфликтом первичного ключа, supabase_request тихо
    возвращал [], и новое значение реально НЕ сохранялось (хотя в памяти
    процесса выглядело так, будто всё ок — до следующего рестарта).
    """

    if key not in SETTINGS_KEYS:
        raise ValueError(f"Unknown setting key: {key}")

    runtime_settings[key] = value

    if SUPABASE_URL and SUPABASE_SECRET_KEY:
        supabase_request(
            "POST",
            "app_settings",
            {"key": key, "value": value},
            params={"on_conflict": "key"},
            prefer="resolution=merge-duplicates,return=representation",
        )


# ============================================================
# LLM ANSWER SYNTHESIS
# ============================================================
#
# Цепочка провайдеров (каждый пропускается, если для него нет ключа):
#
#   1. provod.ai    — напрямую, если задан provod_api_key
#   2. OpenRouter   — перебор бесплатных моделей (DeepSeek/Qwen/Llama)
#   3. DeepSeek API — напрямую, если задан deepseek_api_key
#   4. Qwen API     — напрямую (DashScope, OpenAI-совместимый режим),
#                     если задан qwen_api_key
#

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
        print("LLM TRY: openrouter /", model)
        try:
            response = _chat_completion_request(OPENROUTER_URL, api_key, model, messages)
            print("LLM HTTP:", response.status_code, "openrouter /", model)
        except Exception as e:
            print("LLM REQUEST ERROR (openrouter):", repr(e), model)
            continue

        if response.status_code in (404, 429) or response.status_code >= 400:
            print("LLM SKIP (openrouter):", response.status_code, model)
            continue

        try:
            content = response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print("LLM PARSE ERROR (openrouter):", repr(e))
            continue

        if content:
            print("LLM SUCCESS: openrouter /", model)
            return content

    return None


def _try_deepseek_direct(messages):
    api_key = get_setting("deepseek_api_key")
    if not api_key:
        return None

    print("LLM TRY: deepseek direct")
    try:
        response = _chat_completion_request(DEEPSEEK_URL, api_key, DEEPSEEK_MODEL, messages)
        print("LLM HTTP:", response.status_code, "deepseek")
    except Exception as e:
        print("LLM REQUEST ERROR (deepseek):", repr(e))
        return None

    if response.status_code >= 400:
        print("LLM BAD STATUS (deepseek):", response.status_code, response.text[:300])
        return None

    try:
        content = response.json()["choices"][0]["message"]["content"].strip()
        print("LLM SUCCESS: deepseek direct")
        return content or None
    except Exception as e:
        print("LLM PARSE ERROR (deepseek):", repr(e))
        return None


def _try_qwen_direct(messages):
    api_key = get_setting("qwen_api_key")
    if not api_key:
        return None

    print("LLM TRY: qwen direct")
    try:
        response = _chat_completion_request(QWEN_URL, api_key, QWEN_MODEL, messages)
        print("LLM HTTP:", response.status_code, "qwen")
    except Exception as e:
        print("LLM REQUEST ERROR (qwen):", repr(e))
        return None

    if response.status_code >= 400:
        print("LLM BAD STATUS (qwen):", response.status_code, response.text[:300])
        return None

    try:
        content = response.json()["choices"][0]["message"]["content"].strip()
        print("LLM SUCCESS: qwen direct")
        return content or None
    except Exception as e:
        print("LLM PARSE ERROR (qwen):", repr(e))
        return None


def _try_provod_direct(messages):
    api_key = get_setting("provod_api_key")
    if not api_key:
        return None

    model = get_setting("provod_model") or PROVOD_DEFAULT_MODEL

    print("LLM TRY: provod direct /", model)
    try:
        response = _chat_completion_request(PROVOD_URL, api_key, model, messages)
        print("LLM HTTP:", response.status_code, "provod /", model)
    except Exception as e:
        print("LLM REQUEST ERROR (provod):", repr(e))
        return None

    if response.status_code >= 400:
        print("LLM BAD STATUS (provod):", response.status_code, response.text[:500])
        # Частая причина 400/404 здесь — неверное имя модели. Если
        # видите такую ошибку в логах, проверьте точный слаг модели
        # в личном кабинете provod.ai и обновите его в админке/переменной
        # окружения PROVOD_MODEL.
        return None

    try:
        content = response.json()["choices"][0]["message"]["content"].strip()
        print("LLM SUCCESS: provod direct /", model)
        return content or None
    except Exception as e:
        print("LLM PARSE ERROR (provod):", repr(e))
        return None


def call_llm(system_prompt, user_prompt, history=None):
    """
    Пробует провайдеров по порядку: provod.ai -> OpenRouter (бесплатные
    модели) -> DeepSeek напрямую -> Qwen напрямую. Возвращает None, если
    ни один не сработал (нет ключей, все недоступны, сетевые ошибки) —
    тогда вызывающий код откатывается на экстрактивный режим без LLM.

    ИСПРАВЛЕНО: раньше в LLM всегда уходило только текущее сообщение —
    история диалога (`memory`, загружаемая из Supabase в get_memory())
    нигде дальше не использовалась, и ассистент фактически не помнил
    предыдущие реплики в рамках сессии. Теперь `history` (список вида
    {"role": "user"/"assistant", "content": "..."}) подмешивается в
    messages перед текущим вопросом — ограниченный последними
    MAX_LLM_HISTORY_MESSAGES сообщениями, чтобы не раздувать промпт.
    """

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
    """
    Пересказывает готовый ответ из локальной базы знаний естественным
    текстом, опционально дополняя его свежей информацией из веба —
    без списков "ИСТОЧНИК N / URL".
    """

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
    """
    Формирует ответ только на основе веб-результатов, когда в локальной
    базе знаний ничего подходящего не нашлось.
    """

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
    """
    Последний рубеж: ни локальная база, ни веб-поиск ничего не дали
    (например, все поисковики недоступны одновременно), но LLM настроена.
    Отвечаем на основе собственных знаний модели, честно предупредив,
    что это не проверено свежим веб-поиском — лучше, чем отказ.
    Если запрос затрагивает несколько тем сразу (например, "перхоть,
    синяки под глазами и чёрные точки"), просим модель ответить по
    каждой части отдельно.
    """

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


# ============================================================
# WEB ANSWER (extractive fallback — used only if LLM unavailable)
# ============================================================

def fallback_web_answer(query, web_results):
    """
    Резервный режим без LLM: вытаскивает наиболее релевантные предложения
    из ЧИСТОГО текста источников (без заголовков/URL — см. clean_web_text)
    и оформляет их списком. Используется, только если OPENROUTER_API_KEY
    не задан или LLM недоступна.
    """

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


# ============================================================
# RESPONSE GENERATOR
# ============================================================

def generate_response(query, memory, local_results, web_results):
    best_score = 0.0
    best_item = None

    if local_results:
        best_score, best_item = local_results[0]

    print("LOCAL BEST SCORE:", best_score)
    if best_item:
        print("LOCAL BEST:", best_item.get("title", ""))
    print("WEB RESULTS:", len(web_results))
    print("LLM AVAILABLE:", llm_available())

    # ------------------------------------------------------
    # LOCAL KNOWLEDGE FOUND
    # ------------------------------------------------------
    if best_item and best_score >= 0.18:
        knowledge_answer = best_item.get("answer", "").strip()

        if llm_available():
            llm_answer = llm_answer_from_local(query, knowledge_answer, web_results, history=memory)
            if llm_answer:
                return llm_answer

        # Фолбэк без LLM: старый ответ из базы + чистые (без заголовков
        # источников) дополняющие предложения, если они есть.
        answer = knowledge_answer

        if web_results:
            web_text = clean_web_text(web_results)
            sentences = rank_sentences(query, web_text, limit=4)

            if sentences:
                answer += "\n\n🌐 Дополнение из актуального поиска:\n"
                for sentence in sentences:
                    answer += "\n• " + sentence

        return answer

    # ------------------------------------------------------
    # NO LOCAL MATCH — RELY ON WEB
    # ------------------------------------------------------
    if web_results:
        if llm_available():
            llm_answer = llm_answer_from_web(query, web_results, history=memory)
            if llm_answer:
                return llm_answer

        web_answer = fallback_web_answer(query, web_results)
        if web_answer:
            return web_answer

    # ------------------------------------------------------
    # NI ЛОКАЛЬНОЙ БАЗЫ, НИ ВЕБА — последний рубеж: сама LLM
    # ------------------------------------------------------
    if llm_available():
        print("Falling back to LLM general knowledge (no local/web data)")
        general_answer = llm_answer_general(query, history=memory)
        if general_answer:
            return general_answer

    # ------------------------------------------------------
    # ВООБЩЕ НИЧЕГО НЕ ПОЛУЧИЛОСЬ
    # ------------------------------------------------------
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


# ============================================================
# TRAINING LOG
# ============================================================

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
# ADMIN AUTH
# ============================================================

def create_admin_token():
    """
    ИСПРАВЛЕНО: подпись раньше строилась как sha256(пароль + ":" + timestamp) —
    голый хэш конкатенации, который в теории уязвим к length-extension
    атакам на SHA-256. Теперь используется HMAC-SHA256 (hmac.new), для
    которого такая атака неприменима.
    """
    timestamp = str(int(time.time()))
    signature = hmac.new(
        ADMIN_PASSWORD.encode("utf-8"), timestamp.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return timestamp + "." + signature


def verify_admin_token(token):
    if not token:
        return False

    parts = token.split(".")
    if len(parts) != 2:
        return False

    timestamp, signature = parts

    try:
        timestamp_int = int(timestamp)
    except Exception:
        return False

    if abs(int(time.time()) - timestamp_int) > 43200:
        return False

    expected = hmac.new(
        ADMIN_PASSWORD.encode("utf-8"), timestamp.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    return secrets.compare_digest(signature, expected)


def check_admin(request: Request):
    token = request.headers.get("X-Admin-Token", "")
    if not verify_admin_token(token):
        raise HTTPException(status_code=401, detail="Нет доступа.")


# ============================================================
# MODELS
# ============================================================

class ChatRequest(BaseModel):
    session_id: str
    message: str


class AdminLogin(BaseModel):
    password: str


class KnowledgeCreate(BaseModel):
    title: str
    category: str
    question: str
    answer: str
    tags: list[str] = []


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: Optional[str] = None
    rating: int
    comment: Optional[str] = ""


class SettingsUpdate(BaseModel):
    openrouter_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    qwen_api_key: Optional[str] = None
    provod_api_key: Optional[str] = None
    provod_model: Optional[str] = None
    llm_direct_mode: Optional[bool] = None


# ============================================================
# HTML (chat + admin interface embedded in the backend)
# ============================================================

HTML = r"""
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ASCEND AI</title>
<style>
* { box-sizing: border-box; }
body { margin: 0; min-height: 100vh; background: #09090b; color: white; font-family: Inter, Arial, sans-serif; }
button, textarea, input { font: inherit; }
button { cursor: pointer; }
.container { width: min(1100px, 94%); margin: 0 auto; }
.header { height: 75px; border-bottom: 1px solid #252529; display: flex; align-items: center; }
.header-inner { display: flex; justify-content: space-between; align-items: center; }
.logo { font-size: 22px; font-weight: 900; letter-spacing: .5px; }
.logo span { opacity: .4; }
.header-actions { display: flex; gap: 8px; }
.header-button { background: #151518; color: white; border: 1px solid #2b2b30; border-radius: 12px; padding: 10px 15px; }
.chat { height: calc(100vh - 75px); display: flex; flex-direction: column; }
.messages { flex: 1; overflow-y: auto; padding: 35px 0; }
.message { display: flex; margin-bottom: 22px; }
.message.user { justify-content: flex-end; }
.message.ai { justify-content: flex-start; }
.bubble { max-width: min(760px, 85%); padding: 16px 19px; border-radius: 19px; line-height: 1.6; white-space: pre-wrap; }
.message.ai .bubble { background: #141416; border: 1px solid #28282c; }
.message.user .bubble { background: #f7d45b; color: #111; }
.sources { max-width: 760px; margin-top: -10px; margin-bottom: 25px; }
.source-card { background: #111113; border: 1px solid #252529; border-radius: 12px; padding: 12px; margin-top: 7px; }
.source-card a { color: #f7d45b; text-decoration: none; word-break: break-word; }
.composer { padding: 15px 0 25px; }
.composer-box { display: flex; gap: 10px; background: #111113; border: 1px solid #29292e; padding: 9px; border-radius: 17px; }
.composer textarea { flex: 1; border: 0; outline: 0; background: transparent; color: white; resize: none; padding: 13px; min-height: 50px; max-height: 150px; }
.send { min-width: 110px; border: 0; border-radius: 12px; background: #f7d45b; color: #111; font-weight: 800; }
.send:disabled { opacity: .55; cursor: not-allowed; }
.admin { display: none; padding: 35px 0 60px; }
.card { background: #111113; border: 1px solid #29292e; border-radius: 18px; padding: 22px; margin-bottom: 18px; }
.card h2 { margin-top: 0; }
.field { margin-bottom: 15px; }
.field label { display: block; opacity: .65; margin-bottom: 7px; font-size: 13px; }
.field input, .field textarea { width: 100%; background: #09090b; color: white; border: 1px solid #29292e; border-radius: 11px; padding: 13px; outline: none; }
.field textarea { min-height: 150px; resize: vertical; }
.primary { background: #f7d45b; color: #111; border: 0; border-radius: 11px; padding: 12px 17px; font-weight: 800; }
.knowledge-item { border-top: 1px solid #29292e; padding: 17px 0; }
.knowledge-item:first-child { border-top: 0; }
.badge { display: inline-block; background: #242428; padding: 5px 8px; border-radius: 7px; font-size: 12px; opacity: .8; }
.status { margin-top: 12px; opacity: .7; font-size: 13px; }
.hidden { display: none !important; }
@media(max-width:700px) {
    .bubble { max-width: 94%; }
    .send { min-width: 80px; }
    .header-button { padding: 9px 11px; }
}
</style>
</head>
<body>
<header class="header">
<div class="container header-inner">
<div class="logo">ASCEND <span>AI</span></div>
<div class="header-actions">
<button class="header-button" onclick="showChat()">💬 Чат</button>
<button class="header-button" onclick="showAdmin()">⚙ Админка</button>
</div>
</div>
</header>
<main class="container">
<section id="chatSection">
<div class="chat">
<div id="messages" class="messages">
<div class="message ai">
<div class="bubble">
Привет! Я ASCEND AI 🧠

Я могу помочь с вопросами
об уходе за кожей, лице,
внешности, питании,
волосах и тренировках.

Если вопроса нет в моей
базе знаний, я могу искать
актуальную информацию
в интернете.

Что тебя интересует?
</div>
</div>
</div>
<div class="composer">
<div class="composer-box">
<textarea id="messageInput" placeholder="Напиши свой вопрос..."></textarea>
<button id="sendButton" class="send" onclick="sendMessage()">Отправить</button>
</div>
</div>
</div>
</section>
<section id="adminSection" class="admin">
<div id="adminLoginCard" class="card">
<h2>⚙️ Админка</h2>
<p>Вход в панель управления нейросетью.</p>
<div class="field">
<label>Пароль</label>
<input id="adminPassword" type="password" placeholder="Пароль администратора">
</div>
<button class="primary" onclick="loginAdmin()">Войти</button>
<div id="loginStatus" class="status"></div>
</div>
<div id="adminPanel" class="hidden">
<div class="card">
<h2>🧠 Состояние нейросети</h2>
<div id="brainStats">Загрузка...</div>
</div>
<div class="card">
<h2>🔑 API-ключи для нейросетей</h2>
<p style="opacity:.7;font-size:13px;margin-top:-8px;">
Вставляй сюда официальные API-ключи (не пароль от личного кабинета).
DeepSeek: platform.deepseek.com → API Keys. Qwen: dashscope.console.aliyun.com.
OpenRouter (бесплатные модели): openrouter.ai → Keys.
</p>
<div id="llmStatus" class="status"></div>
<div class="field">
<label>OpenRouter API Key</label>
<input id="openrouterKey" type="password" placeholder="sk-or-v1-...">
</div>
<div class="field">
<label>DeepSeek API Key</label>
<input id="deepseekKey" type="password" placeholder="sk-...">
</div>
<div class="field">
<label>Qwen (DashScope) API Key</label>
<input id="qwenKey" type="password" placeholder="sk-...">
</div>
<div class="field">
<label>provod.ai API Key</label>
<input id="provodKey" type="password" placeholder="sk-...">
</div>
<div class="field">
<label>provod.ai — имя модели (точно как в личном кабинете)</label>
<input id="provodModel" placeholder="xiaomi/mimo-v2.5">
</div>
<div class="field" style="display:flex;align-items:center;gap:10px;">
<input id="llmDirectMode" type="checkbox" style="width:auto;">
<label style="margin:0;">Прямой LLM режим (не ходить в веб-поиск, отвечать сразу через LLM)</label>
</div>
<button class="primary" onclick="saveSettings()">💾 Сохранить ключи</button>
<div id="settingsStatus" class="status"></div>
</div>
<div class="card">
<h2>📚 Добавить знание</h2>
<div class="field">
<label>Название</label>
<input id="title" placeholder="Например: Перхоть">
</div>
<div class="field">
<label>Категория</label>
<input id="category" placeholder="hair">
</div>
<div class="field">
<label>Пример вопроса пользователя</label>
<input id="question" placeholder="Что делать с перхотью?">
</div>
<div class="field">
<label>Ответ нейросети</label>
<textarea id="answer" placeholder="Напиши правильный ответ..."></textarea>
</div>
<div class="field">
<label>Теги через запятую</label>
<input id="tags" placeholder="перхоть, волосы, кожа головы">
</div>
<button class="primary" onclick="addKnowledge()">🧠 Обучить нейросеть</button>
<div id="trainStatus" class="status"></div>
</div>
<div class="card">
<h2>📖 База знаний</h2>
<div id="knowledgeList">Загрузка...</div>
</div>
<div class="card">
<h2>🌐 Последние поиски</h2>
<div id="webSearchList">Загрузка...</div>
</div>
</div>
</section>
</main>
<script>
const SESSION_KEY = "ascend_session_id";
let sessionId = localStorage.getItem(SESSION_KEY);
if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem(SESSION_KEY, sessionId);
}
let adminToken = localStorage.getItem("ascend_admin_token");

function addMessage(role, text) {
    const messages = document.getElementById("messages");
    const wrapper = document.createElement("div");
    wrapper.className = "message " + (role === "user" ? "user" : "ai");
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    wrapper.appendChild(bubble);
    messages.appendChild(wrapper);
    messages.scrollTop = messages.scrollHeight;
}

function addSources(sources) {
    if (!sources || sources.length === 0) { return; }
    const messages = document.getElementById("messages");
    const wrapper = document.createElement("div");
    wrapper.className = "sources";
    const heading = document.createElement("div");
    heading.textContent = "🌐 Источники:";
    heading.style.opacity = "0.65";
    heading.style.marginBottom = "8px";
    wrapper.appendChild(heading);
    sources.forEach(source => {
        const card = document.createElement("div");
        card.className = "source-card";
        const title = document.createElement("div");
        title.textContent = source.title || source.url;
        const link = document.createElement("a");
        link.href = source.url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = source.url;
        card.appendChild(title);
        card.appendChild(link);
        wrapper.appendChild(card);
    });
    messages.appendChild(wrapper);
    messages.scrollTop = messages.scrollHeight;
}

async function sendMessage() {
    const input = document.getElementById("messageInput");
    const button = document.getElementById("sendButton");
    const message = input.value.trim();
    if (!message) { return; }
    if (message.length > 5000) { alert("Сообщение слишком длинное."); return; }
    addMessage("user", message);
    input.value = "";
    button.disabled = true;
    button.textContent = "Ищу...";
    addMessage("ai", "🌐 Проверяю базу и ищу информацию в интернете...");
    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId, message: message })
        });
        let data;
        try { data = await response.json(); } catch { data = { detail: "Сервер вернул некорректный ответ." }; }
        const messages = document.getElementById("messages");
        if (messages.lastElementChild) { messages.lastElementChild.remove(); }
        if (!response.ok) {
            addMessage("ai", data.detail || "Ошибка сервера.");
        } else {
            addMessage("ai", data.answer || "Сервер не вернул ответ.");
            addSources(data.sources);
        }
    } catch (error) {
        console.error(error);
        const messages = document.getElementById("messages");
        if (messages.lastElementChild) { messages.lastElementChild.remove(); }
        addMessage("ai", "Ошибка соединения с сервером.");
    }
    button.disabled = false;
    button.textContent = "Отправить";
}

document.getElementById("messageInput").addEventListener("keydown", function(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});

function showChat() {
    document.getElementById("chatSection").style.display = "block";
    document.getElementById("adminSection").style.display = "none";
}

function showAdmin() {
    document.getElementById("chatSection").style.display = "none";
    document.getElementById("adminSection").style.display = "block";
}

async function loginAdmin() {
    const password = document.getElementById("adminPassword").value;
    const status = document.getElementById("loginStatus");
    status.textContent = "Проверка...";
    try {
        const response = await fetch("/api/admin/login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ password })
        });
        let data;
        try { data = await response.json(); } catch { data = { detail: "Некорректный ответ сервера." }; }
        if (!response.ok) { status.textContent = data.detail || "Неверный пароль."; return; }
        adminToken = data.token;
        localStorage.setItem("ascend_admin_token", adminToken);
        document.getElementById("adminPanel").classList.remove("hidden");
        status.textContent = "Авторизация успешна.";
        loadAdminData();
    } catch {
        status.textContent = "Ошибка соединения.";
    }
}

function adminHeaders() {
    return { "Content-Type": "application/json", "X-Admin-Token": adminToken };
}

async function addKnowledge() {
    const title = document.getElementById("title").value.trim();
    const category = document.getElementById("category").value.trim();
    const question = document.getElementById("question").value.trim();
    const answer = document.getElementById("answer").value.trim();
    const tags = document.getElementById("tags").value.split(",").map(x => x.trim()).filter(Boolean);
    const status = document.getElementById("trainStatus");
    status.textContent = "Обучаю нейросеть...";
    try {
        const response = await fetch("/api/admin/knowledge", {
            method: "POST",
            headers: adminHeaders(),
            body: JSON.stringify({ title, category, question, answer, tags })
        });
        let data;
        try { data = await response.json(); } catch { data = { detail: "Некорректный ответ сервера." }; }
        if (!response.ok) { status.textContent = data.detail || "Ошибка."; return; }
        status.textContent = "✅ Знание добавлено. Нейросеть переобучена.";
        document.getElementById("title").value = "";
        document.getElementById("category").value = "";
        document.getElementById("question").value = "";
        document.getElementById("answer").value = "";
        document.getElementById("tags").value = "";
        loadAdminData();
    } catch (error) {
        console.error(error);
        status.textContent = "Ошибка соединения с сервером.";
    }
}

async function loadAdminData() {
    if (!adminToken) { return; }
    try {
        const statsResponse = await fetch("/api/admin/stats", { headers: adminHeaders() });
        if (statsResponse.ok) {
            const stats = await statsResponse.json();
            document.getElementById("brainStats").innerHTML = `
                <p>🧠 Модель: <strong>${stats.brain_ready ? "готова" : "не готова"}</strong></p>
                <p>📚 Знаний: <strong>${stats.knowledge}</strong></p>
                <p>🔤 Словарь: <strong>${stats.vocabulary}</strong></p>
                <p>🏷️ Категорий: <strong>${stats.categories}</strong></p>
            `;
        }
        await loadSettingsStatus();
        const knowledgeResponse = await fetch("/api/admin/knowledge", { headers: adminHeaders() });
        if (!knowledgeResponse.ok) { return; }
        const knowledge = await knowledgeResponse.json();
        const list = document.getElementById("knowledgeList");
        list.innerHTML = "";
        knowledge.forEach(item => {
            const element = document.createElement("div");
            element.className = "knowledge-item";
            element.innerHTML = `
                <span class="badge">${escapeHtml(item.category || "")}</span>
                <h3>${escapeHtml(item.title || "")}</h3>
                <p><strong>Вопрос:</strong><br>${escapeHtml(item.question || "")}</p>
                <p>${escapeHtml(item.answer || "")}</p>
            `;
            list.appendChild(element);
        });
        const webResponse = await fetch("/api/admin/web-sources", { headers: adminHeaders() });
        if (webResponse.ok) {
            const webData = await webResponse.json();
            const webList = document.getElementById("webSearchList");
            webList.innerHTML = "";
            webData.forEach(item => {
                const element = document.createElement("div");
                element.className = "knowledge-item";
                element.innerHTML = `
                    <span class="badge">🌐 ${escapeHtml(item.source || "web")}</span>
                    <h3>${escapeHtml(item.title || "")}</h3>
                    <p><strong>Запрос:</strong> ${escapeHtml(item.query || "")}</p>
                    <a href="${escapeHtml(item.url || "#")}" target="_blank" rel="noopener noreferrer">Открыть источник</a>
                `;
                webList.appendChild(element);
            });
        }
    } catch (error) {
        console.error(error);
    }
}

async function loadSettingsStatus() {
    try {
        const response = await fetch("/api/admin/settings", { headers: adminHeaders() });
        if (!response.ok) { return; }
        const data = await response.json();
        const status = document.getElementById("llmStatus");
        const line = (label, set, masked) =>
            `${label}: ${set ? "✅ задан (" + escapeHtml(masked) + ")" : "— не задан"}`;
        status.innerHTML = [
            line("provod.ai", data.provod_api_key_set, data.provod_api_key_masked),
            `provod.ai модель: ${escapeHtml(data.provod_model || "")}`,
            line("OpenRouter", data.openrouter_api_key_set, data.openrouter_api_key_masked),
            line("DeepSeek", data.deepseek_api_key_set, data.deepseek_api_key_masked),
            line("Qwen", data.qwen_api_key_set, data.qwen_api_key_masked),
            `Прямой LLM режим: ${data.llm_direct_mode ? "✅ включён" : "выключен"}`,
            `LLM-ответы: ${data.llm_enabled ? "✅ включены" : "выключены (экстрактивный режим)"}`
        ].join("<br>");
        if (data.provod_model) {
            document.getElementById("provodModel").placeholder = data.provod_model;
        }
        document.getElementById("llmDirectMode").checked = !!data.llm_direct_mode;
    } catch (error) {
        console.error(error);
    }
}

async function saveSettings() {
    const status = document.getElementById("settingsStatus");
    status.textContent = "Сохраняю...";

    const body = {};
    const openrouterKey = document.getElementById("openrouterKey").value.trim();
    const deepseekKey = document.getElementById("deepseekKey").value.trim();
    const qwenKey = document.getElementById("qwenKey").value.trim();
    const provodKey = document.getElementById("provodKey").value.trim();
    const provodModel = document.getElementById("provodModel").value.trim();

    if (openrouterKey) { body.openrouter_api_key = openrouterKey; }
    if (deepseekKey) { body.deepseek_api_key = deepseekKey; }
    if (qwenKey) { body.qwen_api_key = qwenKey; }
    if (provodKey) { body.provod_api_key = provodKey; }
    if (provodModel) { body.provod_model = provodModel; }
    body.llm_direct_mode = document.getElementById("llmDirectMode").checked;

    try {
        const response = await fetch("/api/admin/settings", {
            method: "POST",
            headers: adminHeaders(),
            body: JSON.stringify(body)
        });
        let data;
        try { data = await response.json(); } catch { data = { detail: "Некорректный ответ сервера." }; }
        if (!response.ok) { status.textContent = data.detail || "Ошибка."; return; }
        status.textContent = "✅ Сохранено.";
        document.getElementById("openrouterKey").value = "";
        document.getElementById("deepseekKey").value = "";
        document.getElementById("qwenKey").value = "";
        document.getElementById("provodKey").value = "";
        document.getElementById("provodModel").value = "";
        loadSettingsStatus();
    } catch (error) {
        console.error(error);
        status.textContent = "Ошибка соединения с сервером.";
    }
}


function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}
</script>
</body>
</html>
"""


# ============================================================
# LEGAL / SUPPORT / PRICING PAGES
# ============================================================

PROJECT_NAME_LEGAL = "ASCEND AI"
SUPPORT_TELEGRAM = "@your_username_here"  # замени на реальный юзернейм

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
.price-card { background: #141416; border: 1px solid #28282c; border-radius: 16px; padding: 20px; margin: 16px 0; }
.price-card .amount { font-size: 22px; font-weight: 900; color: #f7d45b; }
.contact-btn { display: inline-block; margin-top: 10px; background: #f7d45b; color: #111; padding: 12px 20px; border-radius: 12px; text-decoration: none; font-weight: 800; }
</style>
"""


@app.get("/privacy", response_class=HTMLResponse)
async def privacy_page():
    updated_date = time.strftime("%d.%m.%Y")
    return f"""
<!DOCTYPE html><html lang="ru"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Политика конфиденциальности — {PROJECT_NAME_LEGAL}</title>
{LEGAL_PAGE_STYLE}</head><body><div class="wrap">
<a class="back" href="/">← Назад на сайт</a>
<h1>Политика конфиденциальности</h1>
<div class="updated">Актуально на {updated_date}</div>

<h2>1. Общие положения</h2>
<p>Настоящая Политика конфиденциальности регулирует порядок обработки и защиты информации, которую пользователь передаёт при использовании сервиса {PROJECT_NAME_LEGAL} (далее — «Сервис»). Используя Сервис, пользователь подтверждает согласие с условиями настоящей Политики.</p>

<h2>2. Какие данные собираются</h2>
<ul>
<li>Идентификатор сессии/аккаунта, используемый для работы чата и сохранения истории сообщений.</li>
<li>Техническая информация: IP-адрес, тип устройства и браузера — для обеспечения безопасности и стабильности работы.</li>
<li>История сообщений с ассистентом — для формирования корректных ответов в рамках диалога.</li>
</ul>
<p>Сервис не запрашивает паспортные данные, документы или иную избыточную личную информацию.</p>

<h2>3. Цели обработки данных</h2>
<ul>
<li>Обеспечение работы функционала чата и истории сообщений;</li>
<li>Связь с пользователем по вопросам поддержки;</li>
<li>Улучшение качества и стабильности работы Сервиса.</li>
</ul>

<h2>4. Передача данных третьим лицам</h2>
<p>Администрация не передаёт данные третьим лицам, за исключением случаев, предусмотренных законодательством, необходимости исполнения обязательств перед пользователем (в т.ч. обработка платежей платёжным партнёром) или прямого согласия пользователя.</p>

<h2>5. Хранение и защита данных</h2>
<p>Данные хранятся в течение срока, необходимого для целей обработки. Администрация принимает разумные технические и организационные меры защиты, но не может гарантировать абсолютную безопасность передачи данных через интернет.</p>

<h2>6. Права пользователя</h2>
<p>Пользователь вправе запросить удаление своих данных, обратившись в поддержку: {SUPPORT_TELEGRAM}.</p>

<h2>7. Изменения политики</h2>
<p>Администрация вправе изменять условия настоящей Политики. Продолжение использования Сервиса после публикации изменений означает согласие с новой редакцией.</p>

<p style="margin-top:40px;">Контакты поддержки: <a href="https://t.me/{SUPPORT_TELEGRAM.lstrip('@')}">{SUPPORT_TELEGRAM}</a></p>
</div></body></html>
"""


@app.get("/terms", response_class=HTMLResponse)
async def terms_page():
    updated_date = time.strftime("%d.%m.%Y")
    return f"""
<!DOCTYPE html><html lang="ru"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Пользовательское соглашение — {PROJECT_NAME_LEGAL}</title>
{LEGAL_PAGE_STYLE}</head><body><div class="wrap">
<a class="back" href="/">← Назад на сайт</a>
<h1>Пользовательское соглашение</h1>
<div class="updated">Актуально на {updated_date}</div>

<h2>1. Предмет соглашения</h2>
<p>Настоящее соглашение регулирует условия использования сервиса {PROJECT_NAME_LEGAL} (далее — «Сервис»), включая доступ к функциям чата с ИИ-ассистентом и, при наличии, платным функциям.</p>

<h2>2. Условия использования</h2>
<ul>
<li>Пользователь обязуется не использовать Сервис для незаконной деятельности;</li>
<li>Ответы ассистента носят информационный характер и не заменяют консультацию врача, юриста или иного специалиста;</li>
<li>Администрация вправе ограничить доступ при нарушении условий соглашения.</li>
</ul>

<h2>3. Платные услуги</h2>
<p>При наличии платных функций их стоимость и условия оплаты указаны на странице <a href="/pricing">Тарифы</a>. Оплата обрабатывается через партнёра-эквайера; Сервис не хранит платёжные реквизиты пользователя.</p>

<h2>4. Возврат средств</h2>
<p>Условия возврата средств за неиспользованные платные функции обсуждаются индивидуально через поддержку: {SUPPORT_TELEGRAM}.</p>

<h2>5. Ограничение ответственности</h2>
<p>Сервис предоставляется «как есть». Администрация не гарантирует абсолютную точность ответов ИИ-ассистента и не несёт ответственности за решения, принятые пользователем на основе этих ответов.</p>

<h2>6. Изменение условий</h2>
<p>Администрация вправе изменять условия соглашения. Актуальная версия всегда доступна по адресу /terms.</p>

<p style="margin-top:40px;">Контакты поддержки: <a href="https://t.me/{SUPPORT_TELEGRAM.lstrip('@')}">{SUPPORT_TELEGRAM}</a></p>
</div></body></html>
"""


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
<a class="contact-btn" href="https://t.me/{SUPPORT_TELEGRAM.lstrip('@')}">Написать в Telegram: {SUPPORT_TELEGRAM}</a>
<p style="margin-top:30px;">Среднее время ответа — до 24 часов.</p>
</div></body></html>
"""


@app.get("/pricing", response_class=HTMLResponse)
async def pricing_page():
    return f"""
<!DOCTYPE html><html lang="ru"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Тарифы — {PROJECT_NAME_LEGAL}</title>
{LEGAL_PAGE_STYLE}</head><body><div class="wrap">
<a class="back" href="/">← Назад на сайт</a>
<h1>Тарифы</h1>
<p>Ниже — актуальные цены на платные функции {PROJECT_NAME_LEGAL}. Все цены указаны в рублях, включают комиссию платёжной системы.</p>

<div class="price-card">
<div class="amount">[ЦЕНА] ₽</div>
<div>[НАЗВАНИЕ ТАРИФА] — [что входит: например, N запросов к ИИ в день]</div>
</div>

<div class="price-card">
<div class="amount">[ЦЕНА] ₽</div>
<div>[НАЗВАНИЕ ТАРИФА] — [что входит]</div>
</div>

<p style="margin-top:30px;">Оплата — через СБП или банковской картой. Возврат при технических проблемах — по обращению в <a href="/contacts">поддержку</a>.</p>
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
    }


# ============================================================
# CHAT
# ============================================================

# ============================================================
# SMALL TALK / GREETINGS (обрабатываются БЕЗ похода в поиск и LLM)
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
def chat(data: ChatRequest):
    """
    ИСПРАВЛЕНО: раньше объявлялась как `async def`, но внутри выполняла
    только блокирующие синхронные вызовы (requests к поисковикам,
    LLM-провайдерам и Supabase — суммарно может занимать десятки секунд).
    Async-хендлер с блокирующим кодом внутри держит event loop занятым и
    "подвешивает" ВСЕ остальные запросы к серверу, пока не закончится.
    Обычная (sync) `def` FastAPI сама уводит в отдельный поток из
    threadpool — так конкурентные запросы перестают мешать друг другу.
    """
    message = data.message.strip()

    if not message:
        raise HTTPException(400, "Пустой запрос.")

    if len(message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(400, "Сообщение слишком длинное.")

    print("")
    print("=" * 60)
    print("NEW CHAT REQUEST:", message)
    print("=" * 60)

    memory = get_memory(data.session_id)

    save_message(data.session_id, "user", message)

    small_talk_answer = detect_small_talk(message)

    if small_talk_answer:
        print("SMALL TALK DETECTED — skipping search/LLM")

        assistant_message = save_message(data.session_id, "assistant", small_talk_answer)

        print("=" * 60)

        return {
            "answer": small_talk_answer,
            "sources": [],
            "knowledge_found": False,
            "web_found": False,
            "memory_used": len(memory),
            "message_id": assistant_message.get("id") if assistant_message else None,
        }

    local_results = search_local_knowledge(message)

    direct_mode = is_llm_direct_mode() and llm_available()

    if direct_mode:
        print("LLM DIRECT MODE: skipping web search")
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

    print("FINAL WEB SOURCES:", len(sources))
    print("=" * 60)

    return {
        "answer": answer,
        "sources": sources,
        "knowledge_found": bool(local_results),
        "web_found": bool(web_results),
        "memory_used": len(memory),
        "message_id": assistant_message.get("id") if assistant_message else None,
    }


# ============================================================
# ADMIN LOGIN
# ============================================================

@app.post("/api/admin/login")
async def admin_login(data: AdminLogin):
    print("ADMIN LOGIN ATTEMPT", flush=True)

    try:
        password_ok = secrets.compare_digest(
            data.password.encode("utf-8"),
            ADMIN_PASSWORD.encode("utf-8"),
        )
    except Exception as e:
        print("ADMIN LOGIN compare_digest ERROR:", repr(e), flush=True)
        raise HTTPException(500, f"Ошибка проверки пароля: {e}")

    if not password_ok:
        print("ADMIN LOGIN: wrong password", flush=True)
        raise HTTPException(401, "Неверный пароль.")

    try:
        token = create_admin_token()
    except Exception as e:
        print("ADMIN LOGIN create_admin_token ERROR:", repr(e), flush=True)
        raise HTTPException(500, f"Ошибка создания токена: {e}")

    print("ADMIN LOGIN: success", flush=True)

    return {"success": True, "token": token}


# ============================================================
# ADMIN STATS
# ============================================================

@app.get("/api/admin/stats")
async def admin_stats(request: Request):
    check_admin(request)

    return {
        "brain_ready": brain.ready,
        "knowledge": len(knowledge_cache),
        "vocabulary": len(brain.vocabulary),
        "categories": len(brain.categories),
    }


# ============================================================
# ADMIN KNOWLEDGE GET
# ============================================================

@app.get("/api/admin/knowledge")
async def admin_knowledge(request: Request):
    check_admin(request)
    return knowledge_cache


# ============================================================
# ADMIN KNOWLEDGE CREATE
# ============================================================

@app.post("/api/admin/knowledge")
def admin_add_knowledge(request: Request, data: KnowledgeCreate):
    check_admin(request)

    title = data.title.strip()
    category = normalize(data.category)
    question = data.question.strip()
    answer = data.answer.strip()
    tags = [x.strip() for x in data.tags if x.strip()]

    if not title:
        raise HTTPException(400, "Название обязательно.")
    if not category:
        raise HTTPException(400, "Категория обязательна.")
    if not question:
        raise HTTPException(400, "Вопрос обязателен.")
    if not answer:
        raise HTTPException(400, "Ответ обязателен.")

    item = {
        "title": title,
        "category": category,
        "question": question,
        "answer": answer,
        "tags": tags,
        "approved": True,
    }

    saved = []
    if SUPABASE_URL and SUPABASE_SECRET_KEY:
        saved = supabase_request("POST", "knowledge", item)

    if saved:
        knowledge_cache.append(saved[0])
    else:
        item["id"] = stable_hash(title + question + answer)
        knowledge_cache.append(item)

    result = brain.train(knowledge_cache)

    save_training_log(question, answer, category, "admin")

    return {"success": True, "training": result, "knowledge": len(knowledge_cache)}


# ============================================================
# ADMIN DELETE KNOWLEDGE
# ============================================================

@app.delete("/api/admin/knowledge/{knowledge_id}")
def admin_delete_knowledge(knowledge_id: str, request: Request):
    check_admin(request)

    global knowledge_cache

    knowledge_cache = [
        item for item in knowledge_cache if str(item.get("id")) != str(knowledge_id)
    ]

    if SUPABASE_URL and SUPABASE_SECRET_KEY:
        supabase_request("DELETE", "knowledge", params={"id": f"eq.{knowledge_id}"})

    brain.train(knowledge_cache)

    return {"success": True}


# ============================================================
# ADMIN WEB SOURCES
# ============================================================

@app.get("/api/admin/web-sources")
def admin_web_sources(request: Request):
    check_admin(request)

    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        return []

    rows = supabase_request(
        "GET",
        "web_sources",
        params={
            "select": "id,query,title,url,snippet,source,created_at",
            "order": "created_at.desc",
            "limit": "50",
        },
    )

    return rows


# ============================================================
# ADMIN SETTINGS
# ============================================================

@app.get("/api/admin/settings")
async def admin_get_settings(request: Request):
    check_admin(request)

    return {
        "openrouter_api_key_set": bool(get_setting("openrouter_api_key")),
        "openrouter_api_key_masked": mask_key(get_setting("openrouter_api_key")),
        "deepseek_api_key_set": bool(get_setting("deepseek_api_key")),
        "deepseek_api_key_masked": mask_key(get_setting("deepseek_api_key")),
        "qwen_api_key_set": bool(get_setting("qwen_api_key")),
        "qwen_api_key_masked": mask_key(get_setting("qwen_api_key")),
        "provod_api_key_set": bool(get_setting("provod_api_key")),
        "provod_api_key_masked": mask_key(get_setting("provod_api_key")),
        "provod_model": get_setting("provod_model") or PROVOD_DEFAULT_MODEL,
        "llm_direct_mode": is_llm_direct_mode(),
        "llm_enabled": llm_available(),
    }


@app.post("/api/admin/settings")
def admin_update_settings(request: Request, data: SettingsUpdate):
    check_admin(request)

    updated = []

    if data.openrouter_api_key is not None:
        save_setting("openrouter_api_key", data.openrouter_api_key.strip())
        updated.append("openrouter_api_key")

    if data.deepseek_api_key is not None:
        save_setting("deepseek_api_key", data.deepseek_api_key.strip())
        updated.append("deepseek_api_key")

    if data.qwen_api_key is not None:
        save_setting("qwen_api_key", data.qwen_api_key.strip())
        updated.append("qwen_api_key")

    if data.provod_api_key is not None:
        save_setting("provod_api_key", data.provod_api_key.strip())
        updated.append("provod_api_key")

    if data.provod_model is not None and data.provod_model.strip():
        save_setting("provod_model", data.provod_model.strip())
        updated.append("provod_model")

    if data.llm_direct_mode is not None:
        save_setting("llm_direct_mode", "true" if data.llm_direct_mode else "false")
        updated.append("llm_direct_mode")

    return {"success": True, "updated": updated, "llm_enabled": llm_available()}


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

    try:
        load_settings()
    except Exception as e:
        print("STARTUP ERROR in load_settings:", repr(e), flush=True)
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

    if ADMIN_PASSWORD == "CHANGE_THIS_PASSWORD":
        print("!" * 60, flush=True)
        print("ВНИМАНИЕ: ADMIN_PASSWORD не задан — используется пароль", flush=True)
        print("по умолчанию. Задайте переменную окружения ADMIN_PASSWORD", flush=True)
        print("перед тем как открывать сервис для реальных пользователей.", flush=True)
        print("!" * 60, flush=True)

    print("=" * 60, flush=True)
    print("")
