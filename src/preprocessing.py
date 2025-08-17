import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import emoji
import nltk

from nltk.stem import WordNetLemmatizer
from wordcloud import STOPWORDS as WC_STOPWORDS


# ==============================
# 1) Tải tài nguyên NLTK
# ==============================
nltk.download("wordnet")
nltk.download("omw-1.4")
lemmatizer = WordNetLemmatizer()

# Stopwords: KHÔNG bỏ từ phủ định
CUSTOM_STOPS = set(WC_STOPWORDS) - {
    "not", "no", "nor", "against", "ain", "aren", "couldn", "didn", "doesn",
    "hadn", "hasn", "haven", "isn", "mightn", "mustn", "needn", "shan",
    "shouldn", "wasn", "weren", "won", "wouldn"
}


# ==============================
# 2) Regex compile trước cho nhanh
# ==============================
RE_HTML = re.compile(r"<.*?>")
RE_URL = re.compile(r"(https?://\S+|www\.\S+)")
RE_MENTION = re.compile(r"@\w+")
RE_HASH = re.compile(r"#\w+")
# Giữ chữ, số, khoảng trắng và ký hiệu tài chính: $ % + - . , /
RE_KEEP = re.compile(r"[^A-Za-z0-9\$\%\+\-\,\./\s]")
RE_MULTIWS = re.compile(r"\s+")

# ==============================
# 3) Tiền xử lý văn bản
# ==============================
def expand_contractions(text: str) -> str:
    """Mở rộng các contractions phổ biến trong tiếng Anh."""
    t = re.sub(r"won\'t", "will not", text)
    t = re.sub(r"can\'t", "can not", t)
    t = re.sub(r"n\'t", " not", t)
    t = re.sub(r"\'re", " are", t)
    t = re.sub(r"\'s", " is", t)
    t = re.sub(r"\'d", " would", t)
    t = re.sub(r"\'ll", " will", t)
    t = re.sub(r"\'t", " not", t)
    t = re.sub(r"\'ve", " have", t)
    t = re.sub(r"\'m", " am", t)
    return t


def preprocess_text_fin(
    text: str,
    *,
    use_lemma: bool = True,
    remove_emoji: bool = False,
    normalize_currency: bool = True,
) -> str:
    """Tiền xử lý văn bản: bỏ HTML/URL/mention/hashtag, mở rộng contractions, chuẩn hóa ký hiệu tiền tệ, loại bỏ ký tự không cần thiết, lemmatize & bỏ stopwords (giữ từ phủ định)."""
    if text is None:
        return ""

    t = str(text).lower()

    # 1) Bỏ HTML, URL, mention/hashtag
    t = RE_HTML.sub(" ", t)
    t = RE_URL.sub(" ", t)
    t = RE_MENTION.sub(" ", t)
    t = RE_HASH.sub(" ", t)

    # 2) Mở rộng contractions
    t = expand_contractions(t)

    # 3) Chuẩn hóa tiền tệ thành token (tùy chọn)
    if normalize_currency:
        t = t.replace("$", " <currency> ")

    # 4) Emoji/Non-ascii (tùy chọn)
    if remove_emoji:
        t = emoji.replace_emoji(t, replace=" ")

    # 5) Lọc ký tự nhưng GIỮ số & % + - . , /
    t = RE_KEEP.sub(" ", t)
    t = RE_MULTIWS.sub(" ", t).strip()

    # 6) Lemmatization + bỏ stopwords (giữ từ phủ định)
    if use_lemma:
        tokens = []
        for w in t.split():
            if w in CUSTOM_STOPS:
                continue
            tokens.append(lemmatizer.lemmatize(w))
        t = " ".join(tokens)

    return t