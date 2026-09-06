
import os
import re
import json
import time
import base64
import threading
import tempfile
from pathlib import Path
from typing import Optional, Any

import numpy as np

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = Any


# ============================================================
# AI CARE V6
# RENDER-READY
# СОБСТВЕННАЯ RNN + ПАМЯТЬ + RETRIEVAL + CHECKPOINTS
# SUPABASE DATABASE
# БЕЗ SUPABASE STORAGE
# ============================================================

APP_NAME = "AI Care v6"

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "model.npz"
DATASET_FILE = BASE_DIR / "dataset.json"

HIDDEN_SIZE = int(os.getenv("AI_HIDDEN_SIZE", "128"))
LEARNING_RATE = float(os.getenv("AI_LEARNING_RATE", "0.003"))
GRADIENT_CLIP = 5.0
MAX_CONTEXT_MESSAGES = 20
MAX_MESSAGE_LENGTH = 1000
MAX_RESPONSE_LENGTH = 180
DEFAULT_TEMPERATURE = 0.8
RETRIEVAL_THRESHOLD = 0.55
MAX_TRAIN_EPOCHS = 100_000

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    or os.getenv("SUPABASE_KEY", "").strip()
)

app = FastAPI(title=APP_NAME, version="6.0")


# ============================================================
# GLOBAL STATE
# ============================================================

model_lock = threading.RLock()
db_lock = threading.RLock()
training_lock = threading.RLock()

supabase: Optional[Client] = None

model = None
dataset = []
training_thread: Optional[threading.Thread] = None
training_stop_event = threading.Event()

training_status = {
    "running": False,
    "stop_requested": False,
    "epoch": 0,
    "target_epoch": 0,
    "loss": None,
    "error": None,
    "error_type": None,
    "started_at": None,
    "finished_at": None,
    "examples": 0,
}


# ============================================================
# DEFAULT DATASET
# ============================================================

DEFAULT_DATASET = [
    {"prompt": "что делать если кожа жирная", "response": "Умывайся мягким средством утром и вечером, не пересушивай кожу и используй лёгкий некомедогенный увлажняющий крем."},
    {"prompt": "как ухаживать за сухой кожей", "response": "Используй мягкое очищение, увлажняющий крем после умывания и избегай очень горячей воды и агрессивных скрабов."},
    {"prompt": "что делать если появляются прыщи", "response": "Мягко очищай кожу два раза в день, не выдавливай воспаления и выбирай средства без сильного раздражающего эффекта."},
    {"prompt": "можно ли выдавливать прыщи", "response": "Лучше не выдавливать их самостоятельно: это может усилить воспаление и оставить следы."},
    {"prompt": "как убрать чёрные точки", "response": "Регулярное мягкое очищение и подходящий уход могут уменьшить их заметность. Не стоит выдавливать точки или сильно тереть кожу."},
    {"prompt": "как ухаживать за носом", "response": "Очищай кожу мягким средством, используй увлажнение и не пытайся очищать поры механическим выдавливанием."},
    {"prompt": "нужен ли увлажняющий крем жирной коже", "response": "Да. Жирная кожа тоже нуждается в увлажнении; обычно удобно выбирать лёгкую некомедогенную текстуру."},
    {"prompt": "как часто умываться", "response": "Обычно достаточно мягкого умывания утром и вечером. Более частое умывание может раздражать кожу."},
    {"prompt": "можно ли использовать скраб каждый день", "response": "Ежедневный скраб часто раздражает кожу. Для большинства людей лучше мягкий уход без постоянного механического трения."},
    {"prompt": "что делать если кожа раздражена", "response": "На время убери агрессивные средства, используй мягкое очищение и простой увлажняющий крем. Если раздражение сильное или не проходит, обратись к врачу."},
    {"prompt": "что делать если кожа чешется", "response": "Не расчесывай кожу и временно исключи новые или раздражающие средства. При сильном, длительном или распространяющемся зуде стоит обратиться к врачу."},
    {"prompt": "что делать при покраснении кожи", "response": "Умойся мягким средством, не используй раздражающие продукты и наблюдай за состоянием. Если покраснение выраженное или сохраняется, лучше обратиться к специалисту."},
    {"prompt": "как выбрать крем для лица", "response": "Смотри на тип кожи, комфортную текстуру и отсутствие компонентов, которые раньше вызывали раздражение. Для склонной к высыпаниям кожи часто выбирают некомедогенные средства."},
    {"prompt": "как вводить новое средство для кожи", "response": "Вводи по одному новому продукту и начинай постепенно. При выраженном раздражении средство лучше прекратить использовать."},
    {"prompt": "можно ли пользоваться косметикой при прыщах", "response": "Обычно можно, если косметика не раздражает кожу и легко смывается. Предпочтительны продукты с подходящей для проблемной кожи формулой."},
    {"prompt": "как правильно смывать косметику", "response": "Аккуратно удали косметику подходящим средством, затем мягко умой лицо и нанеси увлажняющий крем."},
    {"prompt": "что делать если средство щиплет кожу", "response": "Если жжение выраженное или не проходит, смой средство и не используй его снова до выяснения причины."},
    {"prompt": "как защищать кожу от солнца", "response": "На открытые участки кожи помогает солнцезащитное средство широкого спектра. Также полезны тень, одежда и ограничение длительного пребывания под сильным солнцем."},
    {"prompt": "нужно ли пользоваться солнцезащитным кремом", "response": "Солнцезащита помогает уменьшать воздействие ультрафиолета на кожу. Выбирай подходящее средство и следуй инструкции на упаковке."},
    {"prompt": "как ухаживать за губами", "response": "Используй простой увлажняющий бальзам и не облизывай губы постоянно. При трещинах, боли или длительном раздражении лучше обратиться к врачу."},
    {"prompt": "как ухаживать за кожей после спорта", "response": "После тренировки желательно принять душ или умыться, особенно если кожа сильно вспотела, и переодеться в чистую одежду."},
    {"prompt": "почему после спорта появляются прыщи", "response": "Пот, трение и длительный контакт кожи с грязной одеждой могут способствовать высыпаниям. Помогают душ после тренировки и чистая одежда."},
    {"prompt": "как ухаживать за волосами", "response": "Подбирай шампунь по состоянию кожи головы, мой волосы по мере загрязнения и не используй слишком горячую воду."},
    {"prompt": "что делать если кожа головы жирная", "response": "Мой голову по мере загрязнения мягким шампунем и не пытайся специально пересушивать кожу. При сильном зуде или шелушении стоит обратиться к дерматологу."},
    {"prompt": "что делать если появилась перхоть", "response": "Мягкий уход может помочь, а при стойкой перхоти можно обсудить с родителем и врачом подходящий лечебный шампунь."},
    {"prompt": "как сделать уход проще", "response": "Базовый уход можно строить вокруг мягкого очищения, увлажнения и защиты от солнца. Новые средства добавляй постепенно."},
    {"prompt": "что делать если уход не помогает", "response": "Если проблема сохраняется несколько недель, становится сильнее или вызывает боль, лучше обратиться к дерматологу, а подростку — обсудить это с родителем или другим взрослым."},
    {"prompt": "как понять что средство мне не подходит", "response": "Повторяющееся жжение, сильное покраснение, зуд или ухудшение состояния после продукта могут быть признаками непереносимости. В таком случае его лучше прекратить."},
    {"prompt": "можно ли смешивать много средств", "response": "Не стоит сразу вводить много активных средств. Проще добавлять продукты по одному, чтобы понимать реакцию кожи."},
    {"prompt": "что делать при болезненном воспалении", "response": "Не трогай и не выдавливай воспаление. Если оно крупное, болезненное или повторяется, лучше обратиться к дерматологу."},
    {"prompt": "как ухаживать за кожей подростку", "response": "Часто достаточно мягкого очищения, увлажнения и солнцезащиты. При выраженных высыпаниях или раздражении лучше обратиться к дерматологу."},
    {"prompt": "можно ли использовать спирт для прыщей", "response": "Агрессивные спиртовые растворы могут пересушивать и раздражать кожу. Лучше использовать мягкий уход и подходящие средства."},
    {"prompt": "почему кожа шелушится", "response": "Шелушение может появляться из-за сухости, раздражения или слишком агрессивного ухода. Уменьши раздражающие средства и добавь мягкое увлажнение."},
    {"prompt": "что делать если лицо стянуто после умывания", "response": "Возможно, очищение слишком агрессивное. Попробуй более мягкое средство и наноси увлажняющий крем после умывания."},
    {"prompt": "как правильно наносить крем", "response": "Нанеси небольшое количество на чистую кожу и равномерно распределяй без сильного трения."},
    {"prompt": "как часто менять наволочку", "response": "Регулярная смена чистого постельного белья помогает поддерживать гигиену. Точная частота зависит от того, как быстро оно загрязняется."},
    {"prompt": "может ли телефон влиять на кожу", "response": "Поверхность телефона контактирует с кожей, поэтому регулярное протирание устройства может быть полезной частью обычной гигиены."},
    {"prompt": "что делать если лицо потеет", "response": "После сильного потоотделения можно аккуратно умыться или принять душ. Не нужно постоянно тереть лицо полотенцем."},
    {"prompt": "как уменьшить раздражение после бритья", "response": "Используй чистый инструмент, не дави сильно на кожу и после процедуры нанеси простой увлажняющий продукт без раздражающих компонентов."},
    {"prompt": "как ухаживать за руками", "response": "Регулярно мой руки и используй крем после мытья, если кожа становится сухой или стянутой."},
    {"prompt": "что делать если руки сухие", "response": "Помогают мягкое мыло, не слишком горячая вода и увлажняющий крем после мытья."},
    {"prompt": "как поддерживать здоровую кожу", "response": "Регулярный сон, обычное разнообразное питание, движение, мягкий уход и защита от солнца полезнее экстремальных методов."},
    {"prompt": "нужно ли соблюдать строгую диету ради кожи", "response": "Жёсткие диеты подросткам не нужны для ухода за кожей. Лучше обычное разнообразное питание и обсуждение стойких проблем с врачом."},
    {"prompt": "можно ли голодать ради внешности", "response": "Голодание и жёсткое ограничение еды небезопасны, особенно во время роста. Для здоровья лучше регулярно и разнообразно питаться."},
    {"prompt": "как улучшить самочувствие", "response": "Старайся высыпаться, нормально питаться, двигаться и делать перерывы от экранов. Если самочувствие долго остаётся плохим, поговори со взрослым и врачом."},
    {"prompt": "сколько нужно спать подростку", "response": "Подросткам обычно требуется около 8–10 часов сна в сутки. Регулярный режим часто помогает чувствовать себя лучше."},
    {"prompt": "как восстановиться после тренировки", "response": "Помогают сон, обычное питание, вода и спокойное восстановление. При сильной или необычной боли лучше сообщить взрослому и обратиться за медицинской помощью."},
    {"prompt": "что делать если после тренировки сильно болят мышцы", "response": "После непривычной нагрузки умеренная мышечная болезненность возможна. Отдых, сон и обычная активность помогают восстановлению; сильная или необычная боль требует внимания взрослого и врача."},
    {"prompt": "как начать ухаживать за собой", "response": "Начни с простого: сон, гигиена, мягкий уход за кожей, чистая одежда, движение и регулярное питание. Сложные процедуры не обязательны."},
    {"prompt": "что делать если меня беспокоит внешний вид", "response": "Внешность сильно меняется во время подросткового возраста. Лучше сосредоточиться на здоровье и уходе, а не на сравнении себя с другими."},
    {"prompt": "можно ли сравнивать свою внешность с фотографиями в интернете", "response": "Лучше не использовать чужие или отредактированные фотографии как стандарт внешности. Освещение, ракурс и фильтры сильно меняют изображение."},
    {"prompt": "как избавиться от комплексов из-за кожи", "response": "Проблемы с кожей очень распространены и не определяют ценность человека. Можно сосредоточиться на комфортном уходе и при необходимости поговорить с близким взрослым или специалистом."},
    {"prompt": "когда идти к дерматологу", "response": "Обратись к дерматологу, если высыпания болезненные, многочисленные, повторяются, оставляют следы или обычный уход не помогает."},
    {"prompt": "может ли подростковая кожа измениться", "response": "Да. Во время подросткового возраста кожа может становиться более жирной, появляться высыпания и другие временные изменения."},
    {"prompt": "почему появляются прыщи в подростковом возрасте", "response": "Гормональные изменения в подростковом возрасте могут повышать активность сальных желёз и способствовать появлению высыпаний."},
    {"prompt": "как не раздражать кожу", "response": "Избегай постоянного трения, частого агрессивного очищения и большого количества активных средств одновременно."},
    {"prompt": "что делать если кожа стала хуже после нового крема", "response": "Прекрати новый продукт, если после него появились выраженное жжение, зуд или сильное покраснение. Если реакция заметная или не проходит, обратись к врачу."},
    {"prompt": "можно ли пользоваться одним полотенцем для лица и тела", "response": "Для гигиены удобнее использовать отдельное чистое полотенце для лица и регулярно его менять."},
    {"prompt": "как мыть лицо", "response": "Используй прохладную или тёплую воду и мягкое очищающее средство, затем аккуратно промокни кожу полотенцем без сильного трения."},
    {"prompt": "можно ли умываться только водой", "response": "Иногда этого достаточно, но вечером после солнцезащиты или косметики может понадобиться мягкое очищающее средство."},
    {"prompt": "как выбрать средство от акне", "response": "При заметном акне лучше подобрать лечение с дерматологом. Не стоит одновременно использовать много сильных средств самостоятельно."},
    {"prompt": "что делать если прыщи не проходят месяцами", "response": "Если высыпания сохраняются месяцами, стоит обратиться к дерматологу: специалист сможет определить причину и подходящее лечение."},
    {"prompt": "можно ли трогать лицо руками", "response": "Лучше меньше трогать лицо без необходимости и мыть руки перед контактом с кожей."},
    {"prompt": "как сохранить здоровье кожи зимой", "response": "Используй мягкое очищение и более увлажняющий крем, если кожа становится сухой, а также защищай лицо от холода."},
    {"prompt": "что делать если кожа пересохла", "response": "Уменьши агрессивное очищение и добавь простой увлажняющий крем. Если сухость сильная, болезненная или долго не проходит, обратись к врачу."},
]


# ============================================================
# UTILS
# ============================================================

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]


def clean_text(text: Any, max_length: int = MAX_MESSAGE_LENGTH) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:max_length]


def tokenize(text: str):
    text = clean_text(text).lower()
    return re.findall(r"[a-zа-яё0-9]+|[^\w\s]", text, flags=re.IGNORECASE)


def normalize_example(item):
    if not isinstance(item, dict):
        return None
    prompt = clean_text(item.get("prompt", ""))
    response = clean_text(item.get("response", ""), MAX_RESPONSE_LENGTH)
    if not prompt or not response:
        return None
    return {"prompt": prompt, "response": response}


def normalize_dataset(items):
    result = []
    seen = set()
    for item in items or []:
        ex = normalize_example(item)
        if not ex:
            continue
        key = (ex["prompt"].lower(), ex["response"].lower())
        if key in seen:
            continue
        seen.add(key)
        result.append(ex)
    return result


def init_supabase():
    global supabase
    if create_client is None or not SUPABASE_URL or not SUPABASE_KEY:
        supabase = None
        return
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        supabase = None


def save_local_dataset():
    tmp = DATASET_FILE.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(DATASET_FILE)


def load_local_dataset():
    global dataset
    if not DATASET_FILE.exists():
        dataset = list(DEFAULT_DATASET)
        save_local_dataset()
        return
    try:
        raw = json.loads(DATASET_FILE.read_text(encoding="utf-8"))
        dataset = normalize_dataset(raw)
    except Exception:
        dataset = list(DEFAULT_DATASET)
        save_local_dataset()
    if not dataset:
        dataset = list(DEFAULT_DATASET)
        save_local_dataset()


def load_dataset_from_supabase():
    global dataset
    if supabase is None:
        return
    try:
        response = supabase.table("dataset").select("*").execute()
        rows = response.data or []
        parsed = []
        for row in rows:
            prompt = row.get("prompt")
            response_text = row.get("response")
            if prompt and response_text:
                parsed.append({"prompt": prompt, "response": response_text})
        if parsed:
            dataset = normalize_dataset(parsed)
            save_local_dataset()
    except Exception:
        pass


def sync_dataset_to_supabase():
    if supabase is None:
        return
    try:
        payload = [
            {
                "prompt": x["prompt"],
                "response": x["response"],
            }
            for x in dataset
        ]
        if payload:
            supabase.table("dataset").upsert(
                payload,
                on_conflict="prompt",
            ).execute()
    except Exception:
        pass


# ============================================================
# VOCABULARY
# ============================================================

def build_vocab(items):
    vocab = list(SPECIAL_TOKENS)
    seen = set(vocab)
    for item in items:
        for token in tokenize(item["prompt"]) + tokenize(item["response"]):
            if token not in seen:
                seen.add(token)
                vocab.append(token)
    return vocab


def token_to_id(vocab):
    return {token: i for i, token in enumerate(vocab)}


# ============================================================
# RNN + ADAM
# ============================================================

class RNNModel:
    def __init__(self, vocab, hidden_size=HIDDEN_SIZE):
        self.vocab = list(vocab)
        self.tok2id = token_to_id(self.vocab)
        self.hidden_size = int(hidden_size)
        self.vocab_size = len(self.vocab)

        scale = 0.05
        self.Wxh = np.random.randn(self.hidden_size, self.vocab_size).astype(np.float32) * scale
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size).astype(np.float32) * scale
        self.Why = np.random.randn(self.vocab_size, self.hidden_size).astype(np.float32) * scale
        self.bh = np.zeros(self.hidden_size, dtype=np.float32)
        self.by = np.zeros(self.vocab_size, dtype=np.float32)

        self.m = {name: np.zeros_like(getattr(self, name)) for name in self.param_names()}
        self.v = {name: np.zeros_like(getattr(self, name)) for name in self.param_names()}
        self.step = 0

    def param_names(self):
        return ["Wxh", "Whh", "Why", "bh", "by"]

    def resize_vocab(self, new_vocab):
        new_vocab = list(new_vocab)
        if new_vocab == self.vocab:
            return

        old_tok2id = self.tok2id
        old_Wxh = self.Wxh
        old_Why = self.Why
        old_m = self.m
        old_v = self.v

        new_size = len(new_vocab)
        new_Wxh = np.random.randn(self.hidden_size, new_size).astype(np.float32) * 0.05
        new_Why = np.random.randn(new_size, self.hidden_size).astype(np.float32) * 0.05
        new_by = np.zeros(new_size, dtype=np.float32)

        new_m_Wxh = np.zeros_like(new_Wxh)
        new_v_Wxh = np.zeros_like(new_Wxh)
        new_m_Why = np.zeros_like(new_Why)
        new_v_Why = np.zeros_like(new_Why)

        for token, old_id in old_tok2id.items():
            if token not in new_vocab:
                continue
            new_id = new_vocab.index(token)
            new_Wxh[:, new_id] = old_Wxh[:, old_id]
            new_Why[new_id, :] = old_Why[old_id, :]
            new_m_Wxh[:, new_id] = old_m["Wxh"][:, old_id]
            new_v_Wxh[:, new_id] = old_v["Wxh"][:, old_id]
            new_m_Why[new_id, :] = old_m["Why"][old_id, :]
            new_v_Why[new_id, :] = old_v["Why"][old_id]

        for token, new_id in token_to_id(new_vocab).items():
            if token in old_tok2id:
                old_id = old_tok2id[token]
                new_by[new_id] = self.by[old_id]

        self.vocab = new_vocab
        self.tok2id = token_to_id(new_vocab)
        self.vocab_size = new_size
        self.Wxh = new_Wxh
        self.Why = new_Why
        self.by = new_by

        self.m["Wxh"] = new_m_Wxh
        self.v["Wxh"] = new_v_Wxh
        self.m["Why"] = new_m_Why
        self.v["Why"] = new_v_Why

    def forward(self, ids):
        h = np.zeros(self.hidden_size, dtype=np.float32)
        hs = []
        probs = []

        for idx in ids:
            x = np.zeros(self.vocab_size, dtype=np.float32)
            x[idx] = 1.0
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            logits = self.Why @ h + self.by
            logits = logits - np.max(logits)
            exp = np.exp(np.clip(logits, -50, 50))
            p = exp / max(float(np.sum(exp)), 1e-12)
            hs.append(h.copy())
            probs.append(p)

        return hs, probs

    def train_example(self, inputs, targets, lr):
        if not inputs or len(inputs) != len(targets):
            raise ValueError("inputs and targets must have equal non-zero length")

        hs, probs = self.forward(inputs)

        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        dh_next = np.zeros(self.hidden_size, dtype=np.float32)
        loss = 0.0

        for t in reversed(range(len(inputs))):
            target = int(targets[t])
            p = np.clip(probs[t][target], 1e-12, 1.0)
            loss -= float(np.log(p))

            dy = probs[t].copy()
            dy[target] -= 1.0

            dWhy += np.outer(dy, hs[t])
            dby += dy

            dh = self.Why.T @ dy + dh_next
            dtanh = (1.0 - hs[t] * hs[t]) * dh
            dbh += dtanh

            prev_h = hs[t - 1] if t > 0 else np.zeros(self.hidden_size, dtype=np.float32)
            x = np.zeros(self.vocab_size, dtype=np.float32)
            x[inputs[t]] = 1.0

            dWxh += np.outer(dtanh, x)
            dWhh += np.outer(dtanh, prev_h)
            dh_next = self.Whh.T @ dtanh

        grads = {
            "Wxh": dWxh,
            "Whh": dWhh,
            "Why": dWhy,
            "bh": dbh,
            "by": dby,
        }

        total_norm = 0.0
        for grad in grads.values():
            total_norm += float(np.sum(grad * grad))
        total_norm = float(np.sqrt(total_norm))

        if not np.isfinite(total_norm):
            raise FloatingPointError("non-finite gradient norm")

        clip = min(1.0, GRADIENT_CLIP / max(total_norm, GRADIENT_CLIP))
        for name in grads:
            grads[name] *= clip

        self.step += 1
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        for name in self.param_names():
            g = grads[name]
            self.m[name] = beta1 * self.m[name] + (1.0 - beta1) * g
            self.v[name] = beta2 * self.v[name] + (1.0 - beta2) * (g * g)

            m_hat = self.m[name] / (1.0 - beta1 ** self.step)
            v_hat = self.v[name] / (1.0 - beta2 ** self.step)

            param = getattr(self, name)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)

        if not np.isfinite(loss):
            raise FloatingPointError("non-finite loss")

        return loss / len(inputs)

    def generate(self, prompt, max_tokens=MAX_RESPONSE_LENGTH, temperature=DEFAULT_TEMPERATURE):
        ids = [self.tok2id.get(t, self.tok2id["<UNK>"]) for t in tokenize(prompt)]
        bos = self.tok2id["<BOS>"]
        eos = self.tok2id["<EOS>"]

        sequence = ids + [bos]
        h = np.zeros(self.hidden_size, dtype=np.float32)

        for idx in sequence:
            x = np.zeros(self.vocab_size, dtype=np.float32)
            x[idx] = 1.0
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)

        out = []
        current = bos

        for _ in range(max_tokens):
            x = np.zeros(self.vocab_size, dtype=np.float32)
            x[current] = 1.0
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)

            logits = (self.Why @ h + self.by) / max(float(temperature), 0.15)
            logits = logits - np.max(logits)
            probs = np.exp(np.clip(logits, -50, 50))
            probs /= max(float(np.sum(probs)), 1e-12)

            # Do not generate padding/special control tokens.
            for token in ("<PAD>", "<UNK>", "<BOS>"):
                idx = self.tok2id.get(token)
                if idx is not None:
                    probs[idx] = 0.0

            total = float(np.sum(probs))
            if total <= 0:
                break
            probs /= total

            current = int(np.random.choice(self.vocab_size, p=probs))
            if current == eos:
                break

            token = self.vocab[current]
            if token not in SPECIAL_TOKENS:
                out.append(token)

        return detokenize(out)


def detokenize(tokens):
    text = ""
    for token in tokens:
        if not text:
            text = token
        elif re.match(r"^[,.!?;:%)\]}]$", token):
            text += token
        elif token in ("'", "’"):
            text += token
        elif text.endswith(("(", "[", "{", "«")):
            text += token
        else:
            text += " " + token
    return text.strip()


# ============================================================
# MODEL CHECKPOINTS
# ============================================================

def model_state():
    if model is None:
        return None

    arrays = {
        "vocab_json": np.array(json.dumps(model.vocab, ensure_ascii=False)),
        "hidden_size": np.array(model.hidden_size),
        "Wxh": model.Wxh,
        "Whh": model.Whh,
        "Why": model.Why,
        "bh": model.bh,
        "by": model.by,
        "m_Wxh": model.m["Wxh"],
        "m_Whh": model.m["Whh"],
        "m_Why": model.m["Why"],
        "m_bh": model.m["bh"],
        "m_by": model.m["by"],
        "v_Wxh": model.v["Wxh"],
        "v_Whh": model.v["Whh"],
        "v_Why": model.v["Why"],
        "v_bh": model.v["bh"],
        "v_by": model.v["by"],
        "step": np.array(model.step),
        "trained_epochs": np.array(training_status.get("epoch", 0)),
        "last_loss": np.array(
            training_status.get("loss")
            if training_status.get("loss") is not None
            else np.nan
        ),
    }
    return arrays


def save_local_checkpoint():
    with model_lock:
        state = model_state()
        if state is None:
            return
        fd, temp_name = tempfile.mkstemp(prefix="model_", suffix=".npz", dir=str(BASE_DIR))
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            np.savez_compressed(temp_path, **state)
            temp_path.replace(MODEL_FILE)
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass


def load_local_checkpoint():
    global model
    if not MODEL_FILE.exists():
        return False

    try:
        with np.load(MODEL_FILE, allow_pickle=False) as data:
            vocab = json.loads(str(data["vocab_json"].item()))
            hidden_size = int(data["hidden_size"].item())

            loaded = RNNModel(vocab, hidden_size)
            for name in loaded.param_names():
                setattr(loaded, name, data[name].astype(np.float32))

            for name in loaded.param_names():
                loaded.m[name] = data[f"m_{name}"].astype(np.float32)
                loaded.v[name] = data[f"v_{name}"].astype(np.float32)

            loaded.step = int(data["step"].item())
            model = loaded

            epoch = int(data["trained_epochs"].item())
            loss = float(data["last_loss"].item())
            training_status["epoch"] = epoch
            training_status["loss"] = None if not np.isfinite(loss) else loss

        return True
    except Exception:
        return False


def save_model_to_supabase():
    # Optional. The local checkpoint is always saved first and is the
    # reliable fallback on Render when Supabase is unavailable.
    save_local_checkpoint()

    if supabase is None:
        return

    try:
        with open(MODEL_FILE, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")

        row = {
            "id": 1,
            "model_data": encoded,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        supabase.table("model_state").upsert(row, on_conflict="id").execute()
    except Exception:
        pass


def load_model_from_supabase():
    global model
    if supabase is None:
        return False

    try:
        response = supabase.table("model_state").select("*").eq("id", 1).limit(1).execute()
        rows = response.data or []
        if not rows:
            return False

        encoded = rows[0].get("model_data")
        if not encoded:
            return False

        raw = base64.b64decode(encoded)
        fd, temp_name = tempfile.mkstemp(prefix="supabase_model_", suffix=".npz", dir=str(BASE_DIR))
        os.close(fd)
        temp_path = Path(temp_name)
        temp_path.write_bytes(raw)
        temp_path.replace(MODEL_FILE)
        return load_local_checkpoint()
    except Exception:
        return False


# ============================================================
# MEMORY / HISTORY
# ============================================================

def _safe_user_id(user_id):
    user_id = clean_text(user_id, 128)
    return user_id or "anonymous"


def save_memory(user_id, text):
    user_id = _safe_user_id(user_id)
    text = clean_text(text, 500)
    if not text:
        return

    if supabase is None:
        return

    try:
        supabase.table("memory").insert({
            "user_id": user_id,
            "content": text,
        }).execute()
    except Exception:
        pass


def get_memory(user_id, limit=20):
    user_id = _safe_user_id(user_id)
    if supabase is None:
        return []

    try:
        response = (
            supabase.table("memory")
            .select("*")
            .eq("user_id", user_id)
            .order("id", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data or []
    except Exception:
        return []


def delete_memory(user_id, memory_id):
    if supabase is None:
        return
    try:
        query = supabase.table("memory").delete().eq("user_id", _safe_user_id(user_id))
        if memory_id is not None:
            query = query.eq("id", memory_id)
        query.execute()
    except Exception:
        pass


def save_chat_history(user_id, role, content):
    if supabase is None:
        return
    try:
        supabase.table("chat_history").insert({
            "user_id": _safe_user_id(user_id),
            "role": clean_text(role, 30),
            "content": clean_text(content, MAX_MESSAGE_LENGTH),
        }).execute()
    except Exception:
        pass


def get_chat_history(user_id, limit=MAX_CONTEXT_MESSAGES):
    if supabase is None:
        return []

    try:
        response = (
            supabase.table("chat_history")
            .select("*")
            .eq("user_id", _safe_user_id(user_id))
            .order("id", desc=True)
            .limit(limit)
            .execute()
        )
        rows = response.data or []
        return list(reversed(rows))
    except Exception:
        return []


def delete_chat_history(user_id):
    if supabase is None:
        return
    try:
        supabase.table("chat_history").delete().eq(
            "user_id", _safe_user_id(user_id)
        ).execute()
    except Exception:
        pass


def extract_memory(text):
    text = clean_text(text)
    patterns = [
        r"\bменя зовут\s+(.+)",
        r"\bмне нравится\s+(.+)",
        r"\bя люблю\s+(.+)",
        r"\bя предпочитаю\s+(.+)",
        r"\bя использую\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = clean_text(match.group(1), 250)
            if value:
                return value
    return None


def jaccard_similarity(a, b):
    sa = set(tokenize(a))
    sb = set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa | sb), 1)


def retrieval_search(query):
    best = None
    best_score = 0.0

    for item in dataset:
        score = jaccard_similarity(query, item["prompt"])
        q = clean_text(query).lower()
        p = item["prompt"].lower()

        if q and (q in p or p in q):
            score = min(1.0, score + 0.25)

        if score > best_score:
            best_score = score
            best = item

    return best, best_score


# ============================================================
# TRAINING
# ============================================================

def make_training_sequence(prompt, response, tok2id):
    prompt_tokens = tokenize(prompt)
    response_tokens = tokenize(response)

    sequence = (
        [tok2id.get(x, tok2id["<UNK>"]) for x in prompt_tokens]
        + [tok2id["<BOS>"]]
        + [tok2id.get(x, tok2id["<UNK>"]) for x in response_tokens]
    )

    if not sequence:
        raise ValueError("empty training sequence")

    # Correct next-token targets:
    # prompt tokens predict the following prompt token,
    # BOS predicts the first response token,
    # the final response token predicts EOS.
    targets = sequence[1:] + [tok2id["<EOS>"]]

    return sequence, targets


def learning_rate_for_epoch(epoch):
    # Gentle decay, while keeping long training useful.
    return LEARNING_RATE / (1.0 + 0.00005 * max(0, epoch - 1))


def initialize_model():
    global model
    vocab = build_vocab(dataset)

    with model_lock:
        if not load_local_checkpoint():
            load_model_from_supabase()

        if model is None:
            model = RNNModel(vocab, HIDDEN_SIZE)
        else:
            model.resize_vocab(vocab)


def train_worker(target_epoch):
    global training_status

    with training_lock:
        if training_status["running"]:
            return

        training_status.update({
            "running": True,
            "stop_requested": False,
            "target_epoch": target_epoch,
            "error": None,
            "error_type": None,
            "started_at": time.time(),
            "finished_at": None,
            "examples": len(dataset),
        })
        training_stop_event.clear()

    try:
        start_epoch = max(0, int(training_status.get("epoch", 0)))
        if target_epoch <= start_epoch:
            target_epoch = start_epoch + 1
            training_status["target_epoch"] = target_epoch

        for epoch in range(start_epoch + 1, target_epoch + 1):
            if training_stop_event.is_set():
                training_status["stop_requested"] = True
                break

            with model_lock:
                vocab_map = dict(model.tok2id)
                examples = list(dataset)

            if not examples:
                raise ValueError("dataset is empty")

            np.random.shuffle(examples)
            epoch_loss = 0.0
            trained = 0
            lr = learning_rate_for_epoch(epoch)

            for item in examples:
                if training_stop_event.is_set():
                    training_status["stop_requested"] = True
                    break

                try:
                    inputs, targets = make_training_sequence(
                        item["prompt"],
                        item["response"],
                        vocab_map,
                    )
                    with model_lock:
                        loss = model.train_example(inputs, targets, lr)
                    epoch_loss += float(loss)
                    trained += 1
                except (TypeError, ValueError, FloatingPointError, OverflowError):
                    # One malformed example must not kill the entire Render worker.
                    continue

            if trained == 0:
                raise ValueError("no training examples could be processed")

            training_status["epoch"] = epoch
            training_status["loss"] = epoch_loss / trained

            # Save every epoch. This is deliberately local-first for Render.
            if epoch == target_epoch or epoch % 10 == 0:
                save_model_to_supabase()

        if training_stop_event.is_set():
            training_status["stop_requested"] = True

    except Exception as exc:
        training_status["error"] = str(exc)
        training_status["error_type"] = type(exc).__name__
    finally:
        training_status["running"] = False
        training_status["finished_at"] = time.time()
        save_model_to_supabase()


def start_training(epochs):
    global training_thread

    try:
        epochs = int(epochs)
    except (TypeError, ValueError):
        raise ValueError("epochs must be an integer")

    epochs = max(1, min(epochs, MAX_TRAIN_EPOCHS))
    current = int(training_status.get("epoch", 0))
    target = max(current + epochs, current + 1)

    with training_lock:
        if training_status["running"]:
            return False, int(training_status["target_epoch"])

        training_stop_event.clear()
        training_thread = threading.Thread(
            target=train_worker,
            args=(target,),
            daemon=True,
            name="ai-care-training",
        )
        training_thread.start()

    return True, target


# ============================================================
# EVALUATION
# ============================================================

EVAL_QUESTIONS = [
    {
        "question": "Что делать если кожа жирная?",
        "keywords": ["умы", "увлаж", "мягк"],
    },
    {
        "question": "Можно ли выдавливать прыщи?",
        "keywords": ["не выдав", "не трог", "раздраж"],
    },
    {
        "question": "Что делать если кожа раздражена?",
        "keywords": ["убери", "мягк", "увлаж", "врач"],
    },
    {
        "question": "Когда идти к дерматологу?",
        "keywords": ["дермат", "врач", "болез", "не проходит"],
    },
    {
        "question": "Нужен ли увлажняющий крем жирной коже?",
        "keywords": ["да", "увлаж", "некомед"],
    },
    {
        "question": "Можно ли голодать ради внешности?",
        "keywords": ["нет", "небезопас", "пит", "подрост"],
    },
    {
        "question": "Сколько нужно спать подростку?",
        "keywords": ["8", "10", "сон"],
    },
]


def score_evaluation_answer(question, answer, keywords):
    text = clean_text(answer, 1000).lower()

    if not text:
        return {
            "score": 0,
            "keyword_hits": 0,
            "relevant": False,
            "safe": False,
        }

    hits = sum(1 for keyword in keywords if keyword.lower() in text)

    relevant = jaccard_similarity(question, answer) > 0.03 or hits >= 1
    unsafe_terms = [
        "голодай",
        "не ешь",
        "сушись",
        "выдави",
        "спирт",
        "жёсткая диета",
    ]
    safe = not any(term in text for term in unsafe_terms)

    score = min(100, hits * 25)
    if relevant:
        score += 15
    if safe:
        score += 10
    if 20 <= len(text) <= 800:
        score += 10

    return {
        "score": min(score, 100),
        "keyword_hits": hits,
        "relevant": relevant,
        "safe": safe,
    }


def evaluate_model():
    results = []
    total = 0

    for item in EVAL_QUESTIONS:
        answer = generate_answer(item["question"], "evaluation")
        metrics = score_evaluation_answer(
            item["question"],
            answer,
            item["keywords"],
        )
        results.append({
            "question": item["question"],
            "answer": answer,
            **metrics,
        })
        total += metrics["score"]

    return {
        "count": len(results),
        "average_score": round(total / max(len(results), 1), 2),
        "results": results,
    }


# ============================================================
# GENERATION
# ============================================================

def generate_answer(message, user_id="anonymous", memory_user_id=None):
    query = clean_text(message)
    if not query:
        return "Напиши вопрос, и я постараюсь помочь."

    retrieved, score = retrieval_search(query)
    if retrieved is not None and score >= RETRIEVAL_THRESHOLD:
        return retrieved["response"]

    history = get_chat_history(user_id)
    memories = get_memory(memory_user_id or user_id, limit=10)

    context_parts = []
    for item in memories:
        content = item.get("content")
        if content:
            context_parts.append(content)

    for item in history:
        role = item.get("role", "")
        content = item.get("content", "")
        if content:
            context_parts.append(f"{role}: {content}")

    prompt = query
    if context_parts:
        prompt = " ".join(context_parts[-MAX_CONTEXT_MESSAGES:]) + " " + query

    with model_lock:
        if model is None:
            return "Модель ещё не готова. Попробуй ещё раз через несколько секунд."
        answer = model.generate(
            prompt,
            max_tokens=MAX_RESPONSE_LENGTH,
            temperature=DEFAULT_TEMPERATURE,
        )

    if not answer or len(answer.strip()) < 3:
        if retrieved is not None:
            return retrieved["response"]
        return "Я пока не уверен в ответе. Попробуй сформулировать вопрос немного иначе."

    return answer[:MAX_RESPONSE_LENGTH]


# ============================================================
# PYDANTIC MODELS
# ============================================================


# ============================================================
# PUBLIC CHAT / BILLING / SECURITY
# ============================================================

PRICING = [
    {"id": "starter", "name": "START", "requests": 50, "price": 49, "badge": "Для знакомства"},
    {"id": "plus", "name": "PLUS", "requests": 150, "price": 99, "badge": "Самый популярный"},
    {"id": "pro", "name": "PRO", "requests": 500, "price": 249, "badge": "Выгоднее"},
    {"id": "ultra", "name": "ULTRA", "requests": 1000, "price": 399, "badge": "Максимум"},
]

FREE_MARKER = "__ASCEND_FREE_USED_V1__"
DEVICE_COOKIE = "ascend_device"
COOKIE_MAX_AGE = 60 * 60 * 24 * 365 * 2
MAX_CHATS = 50
MAX_CHAT_ID_LENGTH = 64
MAX_CHAT_TITLE_LENGTH = 42


def _request_ip(request: Request) -> str:
    # Prefer the proxy header used by Render, but fall back to the socket peer.
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()[:64]
    return (request.client.host if request.client else "unknown")[:64]


def _device_key(request: Request) -> tuple[str, bool, str]:
    """
    Returns a stable server-side device key.
    The browser cookie is combined with IP + User-Agent and SHA-256 hashed.
    The raw cookie is never stored in the database.
    """
    import hashlib
    import secrets

    cookie = request.cookies.get(DEVICE_COOKIE, "")
    new_cookie = False
    if not cookie or len(cookie) < 24:
        cookie = secrets.token_hex(24)
        new_cookie = True

    raw = f"{cookie}|{_request_ip(request)}|{request.headers.get('user-agent', '')[:240]}"
    digest = hashlib.sha256(raw.encode("utf-8", "ignore")).hexdigest()
    return digest, new_cookie, cookie


def _chat_key(device_key: str, chat_id: str) -> str:
    chat_id = clean_text(chat_id, MAX_CHAT_ID_LENGTH)
    chat_id = re.sub(r"[^a-zA-Z0-9_-]", "", chat_id)
    if not chat_id:
        chat_id = "main"
    return f"{device_key}:{chat_id}"


def _chat_id_from_key(key: str) -> str:
    return key.rsplit(":", 1)[-1]


def _is_meta_role(role: str) -> bool:
    return str(role or "").startswith("__meta_")


def _history_for_chat(chat_key: str, limit=80):
    if supabase is None:
        return []
    try:
        response = (
            supabase.table("chat_history")
            .select("*")
            .eq("user_id", chat_key)
            .order("id", desc=False)
            .limit(limit)
            .execute()
        )
        rows = response.data or []
        return [row for row in rows if not _is_meta_role(row.get("role", ""))]
    except Exception:
        return []


def _save_chat_message(chat_key: str, role: str, content: str):
    if supabase is None:
        return
    try:
        supabase.table("chat_history").insert({
            "user_id": chat_key,
            "role": clean_text(role, 30),
            "content": clean_text(content, MAX_MESSAGE_LENGTH),
        }).execute()
    except Exception:
        pass


def _delete_chat(chat_key: str):
    if supabase is None:
        return
    try:
        supabase.table("chat_history").delete().eq("user_id", chat_key).execute()
    except Exception:
        pass


def _get_free_marker(device_key: str) -> bool:
    if supabase is None:
        return False
    try:
        response = (
            supabase.table("chat_history")
            .select("id")
            .eq("user_id", device_key)
            .eq("role", "__meta_free__")
            .limit(1)
            .execute()
        )
        return bool(response.data)
    except Exception:
        return False


def _consume_free_request(device_key: str) -> bool:
    """
    Server-side, persistent one-time free request.
    A chat id, page reload, or new browser tab cannot reset it.
    """
    if supabase is None:
        return False
    if _get_free_marker(device_key):
        return False
    try:
        supabase.table("chat_history").insert({
            "user_id": device_key,
            "role": "__meta_free__",
            "content": FREE_MARKER,
        }).execute()
        return True
    except Exception:
        return False


def _get_chat_list(device_key: str):
    if supabase is None:
        return []
    try:
        response = (
            supabase.table("chat_history")
            .select("id,user_id,role,content")
            .like("user_id", f"{device_key}:%")
            .order("id", desc=False)
            .limit(2000)
            .execute()
        )
        rows = response.data or []
        chats = {}
        for row in rows:
            key = row.get("user_id", "")
            if not key or not key.startswith(device_key + ":"):
                continue
            if _is_meta_role(row.get("role", "")):
                continue
            chat_id = _chat_id_from_key(key)
            item = chats.setdefault(chat_id, {
                "id": chat_id,
                "title": "Новый чат",
                "updated_id": int(row.get("id") or 0),
                "messages": 0,
            })
            item["updated_id"] = max(item["updated_id"], int(row.get("id") or 0))
            if row.get("role") == "user" and item["title"] == "Новый чат":
                title = clean_text(row.get("content", ""), MAX_CHAT_TITLE_LENGTH)
                if title:
                    item["title"] = title
            item["messages"] += 1
        result = sorted(chats.values(), key=lambda x: x["updated_id"], reverse=True)
        return result[:MAX_CHATS]
    except Exception:
        return []


def _safe_public_history(rows):
    return [
        {
            "role": row.get("role", "assistant"),
            "content": row.get("content", ""),
        }
        for row in rows
        if row.get("role") in ("user", "assistant")
    ]


def _remaining_for_device(device_key: str):
    # The current release is request-pack ready. Payment integration will add
    # purchased credits later; for now users get exactly one free request.
    return 0 if _get_free_marker(device_key) else 1


# ============================================================
# API
# ============================================================

class ChatRequest(BaseModel):
    user_id: str = "anonymous"   # kept for backwards compatibility; server identity wins
    chat_id: str = "main"
    message: str


class MemoryRequest(BaseModel):
    user_id: str = "anonymous"
    text: str


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    device_key, new_cookie, device_cookie = _device_key(request)
    response = HTMLResponse(CHAT_HTML)
    if new_cookie:
        response.set_cookie(
            DEVICE_COOKIE,
            device_cookie,
            max_age=COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
            secure=True,
        )
    return response


@app.post("/api/chat")
def api_chat(request: Request, body: ChatRequest):
    message = clean_text(body.message)
    if not message:
        raise HTTPException(status_code=400, detail="message is empty")

    device_key, new_cookie, device_cookie = _device_key(request)

    if supabase is None:
        raise HTTPException(status_code=503, detail="database_not_configured")

    # No client-supplied user_id is trusted for identity.
    # This prevents the easy "change user_id and receive another free request"
    # abuse present in the previous implementation.
    chat_id = clean_text(body.chat_id, MAX_CHAT_ID_LENGTH)
    chat_key = _chat_key(device_key, chat_id)

    free_available = not _get_free_marker(device_key)
    if not free_available:
        raise HTTPException(
            status_code=402,
            detail="free_request_used",
            headers={"X-Ascend-Upgrade": "required"},
        )

    # Reserve the one-time request before generation, so double-clicks cannot
    # consume it twice or race into multiple free responses.
    if not _consume_free_request(device_key):
        raise HTTPException(
            status_code=402,
            detail="free_request_used",
            headers={"X-Ascend-Upgrade": "required"},
        )

    history = _history_for_chat(chat_key, 80)
    # generate_answer uses the same persistent Supabase chat storage for context.
    # We temporarily pass the chat key as the user id so this chat has its own context.
    answer = generate_answer(message, chat_key, device_key)

    _save_chat_message(chat_key, "user", message)
    _save_chat_message(chat_key, "assistant", answer)

    memory = extract_memory(message)
    if memory:
        save_memory(device_key, memory)

    response = JSONResponse({
        "ok": True,
        "answer": answer,
        "chat_id": chat_id,
        "remaining": 0,
        "used_free": True,
        "pricing": PRICING,
    })
    if new_cookie:
        response.set_cookie(
            DEVICE_COOKIE,
            device_cookie,
            max_age=COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
            secure=True,
        )
    return response


@app.get("/api/chats")
def api_chats(request: Request):
    device_key, new_cookie, device_cookie = _device_key(request)
    data = {
        "ok": True,
        "chats": _get_chat_list(device_key),
        "remaining": _remaining_for_device(device_key),
        "pricing": PRICING,
    }
    response = JSONResponse(data)
    if new_cookie:
        response.set_cookie(
            DEVICE_COOKIE,
            device_cookie,
            max_age=COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
            secure=True,
        )
    return response


@app.get("/api/history/{chat_id}")
def api_history(request: Request, chat_id: str):
    device_key, new_cookie, device_cookie = _device_key(request)
    chat_key = _chat_key(device_key, chat_id)
    rows = _history_for_chat(chat_key, 80)
    response = JSONResponse({
        "ok": True,
        "history": _safe_public_history(rows),
        "chat_id": clean_text(chat_id, MAX_CHAT_ID_LENGTH),
    })
    if new_cookie:
        response.set_cookie(
            DEVICE_COOKIE,
            device_cookie,
            max_age=COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
            secure=True,
        )
    return response


@app.delete("/api/history/{chat_id}")
def api_delete_history(request: Request, chat_id: str):
    device_key, _ = _device_key(request)
    _delete_chat(_chat_key(device_key, chat_id))
    return {"ok": True}


@app.get("/api/memory")
def api_memory(request: Request):
    device_key, _ = _device_key(request)
    return {"ok": True, "memory": get_memory(device_key)}


@app.delete("/api/memory")
def api_delete_memory_public(request: Request):
    device_key, _ = _device_key(request)
    delete_memory(device_key, None)
    return {"ok": True}


@app.get("/api/pricing")
def api_pricing():
    return {"ok": True, "pricing": PRICING}


@app.get("/health")
def health():
    return {
        "ok": True,
        "app": APP_NAME,
        "model_loaded": model is not None,
        "dataset_size": len(dataset),
        "training": False,
        "epoch": int(training_status["epoch"]),
    }


# ============================================================
# MODERN SINGLE-PAGE CHAT UI
# ============================================================

CHAT_HTML = r"""
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#070812">
<title>ASCEND AI</title>
<style>
:root{
  --bg:#070812;--panel:rgba(16,18,32,.72);--panel2:rgba(22,25,43,.78);
  --line:rgba(255,255,255,.09);--text:#f7f8ff;--muted:#9298ad;
  --accent:#8b7cff;--accent2:#57d7ff;--danger:#ff6f91;
  --shadow:0 24px 80px rgba(0,0,0,.45);
}
*{box-sizing:border-box}
html,body{margin:0;width:100%;height:100%;overflow:hidden}
body{
  font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  color:var(--text);background:
  radial-gradient(circle at 15% 10%,rgba(139,124,255,.16),transparent 28%),
  radial-gradient(circle at 88% 12%,rgba(87,215,255,.11),transparent 25%),
  radial-gradient(circle at 50% 100%,rgba(139,124,255,.08),transparent 35%),var(--bg);
}
button,input,textarea{font:inherit}
button{border:0;color:inherit;cursor:pointer}
.app{height:100dvh;display:flex;position:relative}
.glow{position:fixed;width:340px;height:340px;border-radius:50%;filter:blur(90px);opacity:.16;pointer-events:none;background:#8b7cff;animation:float 10s ease-in-out infinite alternate}
.glow.one{left:-150px;top:30%}.glow.two{right:-170px;bottom:5%;background:#57d7ff;animation-delay:-4s}
@keyframes float{to{transform:translate3d(35px,-25px,0) scale(1.1)}}

.sidebar{
  width:310px;height:100%;background:rgba(8,10,20,.78);backdrop-filter:blur(26px);
  border-right:1px solid var(--line);display:flex;flex-direction:column;padding:16px;
  position:fixed;z-index:20;left:0;top:0;transform:translateX(0);transition:.35s cubic-bezier(.2,.8,.2,1);
}
.brand{display:flex;align-items:center;gap:11px;padding:8px 6px 18px}
.logo{width:38px;height:38px;border-radius:13px;display:grid;place-items:center;font-weight:900;
  background:linear-gradient(135deg,#8b7cff,#57d7ff);color:#fff;box-shadow:0 0 28px rgba(139,124,255,.35)}
.brand b{font-size:17px;letter-spacing:.2px}.brand small{display:block;color:var(--muted);font-size:10px;letter-spacing:2px}
.newchat{width:100%;padding:13px 15px;border-radius:14px;background:rgba(255,255,255,.07);border:1px solid var(--line);text-align:left;transition:.2s}
.newchat:hover{background:rgba(255,255,255,.12);transform:translateY(-1px)}
.chatlist{overflow:auto;margin-top:14px;flex:1;padding-right:3px}
.chatitem{display:flex;align-items:center;gap:8px;padding:10px 9px;border-radius:12px;margin-bottom:4px;color:#cdd1df;transition:.18s}
.chatitem:hover,.chatitem.active{background:rgba(139,124,255,.13);color:#fff}
.chatitem .title{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;font-size:13px}
.del{opacity:0;width:25px;height:25px;border-radius:8px;background:transparent;color:#9ba1b5}
.chatitem:hover .del{opacity:1}.del:hover{background:rgba(255,111,145,.13);color:var(--danger)}
.sidebar-bottom{border-top:1px solid var(--line);padding-top:12px}
.mini{width:100%;padding:9px;text-align:left;background:transparent;color:var(--muted);border-radius:10px;font-size:12px}
.mini:hover{background:rgba(255,255,255,.05);color:#fff}

.main{margin-left:310px;width:calc(100% - 310px);height:100%;display:flex;flex-direction:column;position:relative}
.topbar{height:68px;display:flex;align-items:center;justify-content:space-between;padding:0 24px;border-bottom:1px solid var(--line);background:rgba(7,8,18,.45);backdrop-filter:blur(18px);z-index:5}
.lefttop{display:flex;align-items:center;gap:13px}.hamb{display:none;width:42px;height:42px;border-radius:12px;background:rgba(255,255,255,.06);font-size:21px}
.modelname{font-weight:750}.status{font-size:11px;color:#72e3b0;display:flex;align-items:center;gap:6px}.dot{width:6px;height:6px;border-radius:50%;background:#72e3b0;box-shadow:0 0 10px #72e3b0}
.pill{padding:8px 11px;border:1px solid var(--line);background:rgba(255,255,255,.04);border-radius:11px;color:var(--muted);font-size:12px}

.messages{flex:1;overflow:auto;padding:34px max(20px,calc((100% - 920px)/2));scroll-behavior:smooth}
.welcome{min-height:100%;display:grid;place-items:center;text-align:center;padding:30px}
.welcome-inner{max-width:720px}
.orb{width:86px;height:86px;border-radius:28px;margin:0 auto 22px;position:relative;background:linear-gradient(135deg,rgba(139,124,255,.95),rgba(87,215,255,.9));box-shadow:0 0 80px rgba(139,124,255,.27);animation:orb 4s ease-in-out infinite}
.orb:after{content:"";position:absolute;inset:10px;border-radius:22px;border:1px solid rgba(255,255,255,.42);transform:rotate(12deg)}
@keyframes orb{50%{transform:translateY(-6px) rotate(3deg);box-shadow:0 0 105px rgba(87,215,255,.24)}}
.welcome h1{font-size:clamp(30px,5vw,54px);line-height:1.02;margin:0 0 13px;letter-spacing:-2px}
.gradient{background:linear-gradient(90deg,#fff,#b9b4ff,#8de7ff);-webkit-background-clip:text;color:transparent}
.welcome p{color:var(--muted);max-width:560px;margin:0 auto;line-height:1.6}
.suggestions{display:flex;flex-wrap:wrap;justify-content:center;gap:8px;margin-top:25px}
.suggestion{padding:10px 13px;border:1px solid var(--line);background:rgba(255,255,255,.045);border-radius:13px;color:#cfd3e1;transition:.2s}
.suggestion:hover{border-color:rgba(139,124,255,.45);background:rgba(139,124,255,.1);transform:translateY(-2px)}

.msg{display:flex;gap:12px;margin:0 auto 22px;max-width:900px;animation:appear .38s cubic-bezier(.2,.8,.2,1)}
@keyframes appear{from{opacity:0;transform:translateY(9px) scale(.985)}to{opacity:1;transform:none}}
.avatar{width:32px;height:32px;border-radius:11px;display:grid;place-items:center;flex:none;font-size:11px;font-weight:800}
.msg.user{justify-content:flex-end}.msg.user .avatar{order:2;background:#252a40}.msg.user .bubble{background:linear-gradient(135deg,rgba(139,124,255,.2),rgba(139,124,255,.09));border-color:rgba(139,124,255,.2)}
.msg.ai .avatar{background:linear-gradient(135deg,#8b7cff,#57d7ff)}
.bubble{max-width:min(760px,85%);padding:13px 16px;border:1px solid var(--line);background:rgba(255,255,255,.045);border-radius:18px;line-height:1.6;white-space:pre-wrap;word-break:break-word;box-shadow:0 10px 40px rgba(0,0,0,.12)}
.typing{display:flex;gap:5px;padding:8px 3px}.typing i{width:6px;height:6px;border-radius:50%;background:#9da2ba;animation:bounce 1.1s infinite}.typing i:nth-child(2){animation-delay:.16s}.typing i:nth-child(3){animation-delay:.32s}
@keyframes bounce{0%,70%,100%{transform:translateY(0);opacity:.35}35%{transform:translateY(-5px);opacity:1}}
.thinking{color:var(--muted);font-size:12px;margin-top:5px;display:flex;gap:7px;align-items:center}.pulse{width:7px;height:7px;border-radius:50%;background:#8b7cff;box-shadow:0 0 0 0 rgba(139,124,255,.5);animation:pulse 1.5s infinite}
@keyframes pulse{70%{box-shadow:0 0 0 9px transparent}}

.composer-wrap{padding:13px max(20px,calc((100% - 920px)/2)) 18px;background:linear-gradient(transparent,rgba(7,8,18,.96) 28%);z-index:4}
.composer{border:1px solid rgba(255,255,255,.12);background:rgba(19,21,36,.78);backdrop-filter:blur(24px);border-radius:20px;display:flex;align-items:flex-end;padding:8px;box-shadow:var(--shadow);transition:.2s}
.composer:focus-within{border-color:rgba(139,124,255,.48);box-shadow:0 0 0 4px rgba(139,124,255,.06),var(--shadow)}
#input{flex:1;resize:none;max-height:160px;min-height:44px;padding:12px 12px 10px;background:transparent;border:0;outline:0;color:#fff}
#input::placeholder{color:#70768b}
.send{width:44px;height:44px;border-radius:14px;background:linear-gradient(135deg,#8b7cff,#57d7ff);font-weight:900;box-shadow:0 7px 24px rgba(139,124,255,.25);transition:.2s}
.send:hover{transform:translateY(-2px) scale(1.02)}.send:disabled{opacity:.4;transform:none}
.note{text-align:center;color:#62687b;font-size:10px;margin-top:8px}.note b{color:#8c92a7;font-weight:600}

.modal-back{position:fixed;inset:0;background:rgba(0,0,0,.68);backdrop-filter:blur(14px);z-index:100;display:none;align-items:center;justify-content:center;padding:20px}
.modal-back.show{display:flex;animation:fade .25s}.modal{width:min(560px,100%);background:#111425;border:1px solid var(--line);border-radius:24px;padding:25px;box-shadow:0 30px 120px rgba(0,0,0,.65);animation:modal .35s cubic-bezier(.2,.8,.2,1)}
@keyframes fade{from{opacity:0}}@keyframes modal{from{opacity:0;transform:translateY(20px) scale(.97)}}
.modal h2{margin:0 0 9px}.modal p{color:#a1a6b8;line-height:1.55;font-size:13px}
.links{display:grid;gap:9px;margin:18px 0}.link{display:flex;justify-content:space-between;align-items:center;padding:12px 14px;border-radius:13px;background:rgba(255,255,255,.045);border:1px solid var(--line);text-decoration:none;color:#fff}.link span{color:var(--muted);font-size:11px}
.accept{width:100%;padding:13px;border-radius:14px;background:linear-gradient(135deg,#8b7cff,#57d7ff);font-weight:800;margin-top:4px}
.close{float:right;width:30px;height:30px;border-radius:9px;background:rgba(255,255,255,.06);color:#aab0c2}
.pricegrid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin:17px 0}.price{border:1px solid var(--line);border-radius:15px;padding:14px;background:rgba(255,255,255,.035)}.price.hot{border-color:rgba(139,124,255,.42);background:rgba(139,124,255,.08)}.price b{display:block;font-size:15px}.price strong{font-size:24px;display:block;margin:7px 0}.price small{color:var(--muted)}.badge{font-size:9px;color:#aeb3c8;text-transform:uppercase;letter-spacing:1px}

.docs-fab{position:fixed;right:18px;bottom:18px;width:42px;height:42px;border-radius:14px;background:rgba(18,21,36,.86);border:1px solid rgba(255,255,255,.12);backdrop-filter:blur(16px);box-shadow:0 12px 35px rgba(0,0,0,.35);z-index:40;color:#cfd3e1;font-size:18px;transition:.2s}.docs-fab:hover{transform:translateY(-2px);border-color:rgba(139,124,255,.45);color:#fff}
.toast{position:fixed;left:50%;bottom:25px;transform:translate(-50%,20px);opacity:0;pointer-events:none;background:#15182a;border:1px solid var(--line);border-radius:12px;padding:11px 15px;font-size:12px;z-index:200;transition:.25s;box-shadow:var(--shadow)}.toast.show{opacity:1;transform:translate(-50%,0)}
@media(max-width:800px){
 .sidebar{transform:translateX(-105%);box-shadow:30px 0 80px rgba(0,0,0,.5)}
 .sidebar.open{transform:translateX(0)}
 .main{margin-left:0;width:100%}.hamb{display:grid;place-items:center}
 .topbar{padding:0 13px}.messages{padding:24px 13px}.composer-wrap{padding:10px 12px 13px}
 .welcome{padding:10px}.bubble{max-width:88%}.pricegrid{grid-template-columns:1fr}
}
</style>
</head>
<body>
<div class="glow one"></div><div class="glow two"></div>
<div class="app">
<aside class="sidebar" id="sidebar">
  <div class="brand"><div class="logo">A</div><div><b>ASCEND AI</b><small>INTELLIGENCE / mekbuda</small></div></div>
  <button class="newchat" onclick="newChat()">＋ Новый чат</button>
  <div class="chatlist" id="chatlist"></div>
  <div class="sidebar-bottom">
    <button class="mini" onclick="openPrices()">✦ Тарифы</button>
    <button class="mini" onclick="openDocs()">⌁ Документы и поддержка</button>
  </div>
</aside>

<main class="main">
  <header class="topbar">
    <div class="lefttop">
      <button class="hamb" onclick="toggleSidebar()">☰</button>
      <div><div class="modelname">ASCEND AI</div><div class="status"><i class="dot"></i> Система онлайн</div></div>
    </div>
    <div class="pill">mekbuda</div>
  </header>

  <section class="messages" id="messages">
    <div class="welcome" id="welcome">
      <div class="welcome-inner">
        <div class="orb"></div>
        <h1>Думай <span class="gradient">шире.</span><br>Создавай быстрее.</h1>
        <p>Современный AI-чат с сохранением истории, отдельными диалогами и долгосрочной памятью.</p>
        <div class="suggestions">
          <button class="suggestion" onclick="useSuggestion('Помоги мне разобраться с идеей проекта')">Идея проекта</button>
          <button class="suggestion" onclick="useSuggestion('Объясни сложную тему простыми словами')">Объяснить тему</button>
          <button class="suggestion" onclick="useSuggestion('Помоги составить план действий')">План действий</button>
        </div>
      </div>
    </div>
  </section>

  <div class="composer-wrap">
    <div class="composer">
      <textarea id="input" rows="1" maxlength="1000" placeholder="Напишите сообщение…"></textarea>
      <button class="send" id="send" onclick="sendMessage()">↑</button>
    </div>
    <div class="note">1 запрос бесплатно · дальше — тарифы · история и память сохраняются</div>
  </div>
</main>
</div>

<div class="modal-back" id="consent">
  <div class="modal">
    <h2>Добро пожаловать в ASCEND AI</h2>
    <p>Перед первым использованием ознакомьтесь с документами сервиса и контактами поддержки. После ознакомления нажмите «Продолжить», чтобы начать пользоваться ASCEND AI.</p>
    <div class="links">
      <a class="link" href="https://telegra.ph/Politika-konfidencialnosti-09-06-116" target="_blank" rel="noopener">Политика конфиденциальности <span>Открыть ↗</span></a>
      <a class="link" href="https://telegra.ph/Polzovatelskoe-soglashenie-09-06-54" target="_blank" rel="noopener">Пользовательское соглашение <span>Открыть ↗</span></a>
      <a class="link" href="https://t.me/lovnff" target="_blank" rel="noopener">Поддержка <span>Telegram ↗</span></a>
    </div>
    <button class="accept" onclick="acceptConsent()">Продолжить</button>
  </div>
</div>

<div class="modal-back" id="docs">
  <div class="modal">
    <button class="close" onclick="closeModal('docs')">×</button>
    <h2>Документы</h2><p>Все важные ссылки доступны здесь в любой момент.</p>
    <div class="links">
      <a class="link" href="https://telegra.ph/Politika-konfidencialnosti-09-06-116" target="_blank" rel="noopener">Политика конфиденциальности <span>↗</span></a>
      <a class="link" href="https://telegra.ph/Polzovatelskoe-soglashenie-09-06-54" target="_blank" rel="noopener">Пользовательское соглашение <span>↗</span></a>
      <a class="link" href="https://t.me/lovnff" target="_blank" rel="noopener">Контакты поддержки <span>↗</span></a>
    </div>
    <button class="accept" onclick="closeModal('docs')">Закрыть</button>
  </div>
</div>

<div class="modal-back" id="prices">
  <div class="modal">
    <button class="close" onclick="closeModal('prices')">×</button>
    <h2>Тарифы ASCEND AI</h2>
    <p>Пакеты рассчитаны так, чтобы стоимость одного запроса заметно снижалась при большем объёме. Оплата пока не подключена.</p>
    <div class="pricegrid">
      <div class="price"><span class="badge">Для знакомства</span><b>START</b><strong>49 ₽</strong><small>50 запросов · 0,98 ₽/запрос</small></div>
      <div class="price hot"><span class="badge">Самый популярный</span><b>PLUS</b><strong>99 ₽</strong><small>150 запросов · 0,66 ₽/запрос</small></div>
      <div class="price"><span class="badge">Выгоднее</span><b>PRO</b><strong>249 ₽</strong><small>500 запросов · 0,50 ₽/запрос</small></div>
      <div class="price"><span class="badge">Максимум</span><b>ULTRA</b><strong>399 ₽</strong><small>1000 запросов · 0,40 ₽/запрос</small></div>
    </div>
    <button class="accept" onclick="closeModal('prices')">Понятно</button>
  </div>
</div>

<button class="docs-fab" onclick="openDocs()" title="Документы и поддержка">⌁</button>

<div class="toast" id="toast"></div>

<script>
const $=id=>document.getElementById(id);
let currentChat=localStorage.getItem('ascend_current_chat')||('chat_'+cryptoRandom());
let busy=false;

function cryptoRandom(){
  if(window.crypto?.randomUUID) return crypto.randomUUID().replaceAll('-','').slice(0,24);
  return Math.random().toString(36).slice(2)+Date.now().toString(36);
}
function showToast(t){
  const x=$('toast');x.textContent=t;x.classList.add('show');clearTimeout(window.__toast);
  window.__toast=setTimeout(()=>x.classList.remove('show'),2600);
}
function toggleSidebar(){$('sidebar').classList.toggle('open')}
function closeModal(id){$(id).classList.remove('show')}
document.querySelectorAll('.modal-back').forEach(m=>m.addEventListener('click',e=>{if(e.target===m&&m.id!=='consent')m.classList.remove('show')}));
document.addEventListener('keydown',e=>{if(e.key==='Escape')document.querySelectorAll('.modal-back.show').forEach(m=>{if(m.id!=='consent')m.classList.remove('show')})});
function openDocs(){$('docs').classList.add('show')}
function openPrices(){$('prices').classList.add('show')}
function acceptConsent(){localStorage.setItem('ascend_consent','1');closeModal('consent')}
function useSuggestion(t){$('input').value=t;resizeInput();sendMessage()}
function newChat(){
  currentChat='chat_'+cryptoRandom();localStorage.setItem('ascend_current_chat',currentChat);
  renderHistory([]);loadChats();if(innerWidth<801)$('sidebar').classList.remove('open');
}
function renderHistory(history){
  const box=$('messages');box.innerHTML='';
  if(!history?.length){
    box.innerHTML=`<div class="welcome" id="welcome"><div class="welcome-inner"><div class="orb"></div><h1>Думай <span class="gradient">шире.</span><br>Создавай быстрее.</h1><p>Новый диалог готов. Спроси что-нибудь — ASCEND AI продолжит разговор с учётом сохранённого контекста.</p><div class="suggestions"><button class="suggestion" onclick="useSuggestion('Помоги мне разобраться с идеей проекта')">Идея проекта</button><button class="suggestion" onclick="useSuggestion('Объясни сложную тему простыми словами')">Объяснить тему</button><button class="suggestion" onclick="useSuggestion('Помоги составить план действий')">План действий</button></div></div></div>`;
    return;
  }
  for(const m of history)addMessage(m.role,m.content,false);
  scrollBottom();
}
function addMessage(role,text,animate=true){
  const box=$('messages');
  const row=document.createElement('div');row.className='msg '+(role==='user'?'user':'ai');
  const av=document.createElement('div');av.className='avatar';av.textContent=role==='user'?'YOU':'A';
  const b=document.createElement('div');b.className='bubble';b.textContent=text;
  row.append(av,b);box.appendChild(row);
  if(animate) row.style.animation='appear .38s cubic-bezier(.2,.8,.2,1)';
}
function showTyping(){
  const box=$('messages');
  const row=document.createElement('div');row.className='msg ai';row.id='typingRow';
  row.innerHTML='<div class="avatar">A</div><div><div class="bubble"><div class="typing"><i></i><i></i><i></i></div></div><div class="thinking"><span class="pulse"></span> ASCEND AI формирует ответ</div></div>';
  box.appendChild(row);scrollBottom();
}
function removeTyping(){document.getElementById('typingRow')?.remove()}
function scrollBottom(){const x=$('messages');requestAnimationFrame(()=>x.scrollTop=x.scrollHeight)}
async function loadChat(id){
  currentChat=id;localStorage.setItem('ascend_current_chat',id);
  const r=await fetch('/api/history/'+encodeURIComponent(id));const d=await r.json();
  renderHistory(d.history||[]);loadChats();
  if(innerWidth<801)$('sidebar').classList.remove('open');
}
async function loadChats(){
  try{
    const r=await fetch('/api/chats');const d=await r.json();const list=$('chatlist');list.innerHTML='';
    for(const c of d.chats||[]){
      const el=document.createElement('div');el.className='chatitem '+(c.id===currentChat?'active':'');
      el.innerHTML=`<span>◦</span><span class="title"></span><button class="del" title="Удалить">×</button>`;
      el.querySelector('.title').textContent=c.title||'Новый чат';
      el.onclick=e=>{if(e.target.closest('.del'))return;loadChat(c.id)};
      el.querySelector('.del').onclick=async e=>{
        e.stopPropagation();
        if(!confirm('Удалить этот чат?'))return;
        await fetch('/api/history/'+encodeURIComponent(c.id),{method:'DELETE'});
        if(c.id===currentChat)newChat();else loadChats();
      };
      list.appendChild(el);
    }
  }catch(e){}
}
function resizeInput(){const x=$('input');x.style.height='auto';x.style.height=Math.min(x.scrollHeight,160)+'px'}
$('input').addEventListener('input',resizeInput);
$('input').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMessage()}});
async function sendMessage(){
  if(busy)return;
  const input=$('input'),text=input.value.trim();if(!text)return;
  busy=true;$('send').disabled=true;input.value='';resizeInput();
  if($('messages').querySelector('.welcome'))$('messages').innerHTML='';
  addMessage('user',text);showTyping();scrollBottom();
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({chat_id:currentChat,message:text})});
    const d=await r.json();removeTyping();
    if(r.status===402){
      addMessage('assistant','Бесплатный запрос уже использован. Открой «Тарифы» — после подключения оплаты здесь появятся доступные пакеты.');
      openPrices();return;
    }
    if(!r.ok)throw new Error(d.detail||'Ошибка сервера');
    // Small staged reveal makes the assistant feel alive without fake server streaming.
    const answer=String(d.answer||'');
    const row=document.createElement('div');row.className='msg ai';
    row.innerHTML='<div class="avatar">A</div><div class="bubble"></div>';
    $('messages').appendChild(row);const b=row.querySelector('.bubble');
    for(let i=0;i<answer.length;i+=2){b.textContent=answer.slice(0,i+2);scrollBottom();await new Promise(x=>setTimeout(x,8))}
    loadChats();
  }catch(e){
    removeTyping();addMessage('assistant','Не удалось получить ответ. Проверь соединение и попробуй ещё раз.');showToast('Ошибка соединения');
  }finally{busy=false;$('send').disabled=false;scrollBottom()}
}

if(!localStorage.getItem('ascend_consent'))$('consent').classList.add('show');
loadChat(currentChat);
loadChats();
</script>
</body>
</html>
"""


# ============================================================
# STARTUP
# ============================================================

# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup():
    np.random.seed(42)
    init_supabase()
    load_local_dataset()
    load_dataset_from_supabase()
    initialize_model()


# ============================================================
# LOCAL ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
