import re
import emoji
from nltk.stem import WordNetLemmatizer
from wordcloud import STOPWORDS as WC_STOPWORDS

lemmatizer = WordNetLemmatizer()

# 1) Stopwords: giữ phủ định + giữ từ vựng vĩ mô
NEG_KEPT = {
    "not","no","nor","against","ain","aren","couldn","didn","doesn",
    "hadn","hasn","haven","isn","mightn","mustn","needn","shan",
    "shouldn","wasn","weren","won","wouldn"
}
MACRO_KEEP = {"us","uk","eu","ecb","fed","boj","boe","pmi","gdp","cpi","ppi"}
CUSTOM_STOPS = (set(WC_STOPWORDS) - NEG_KEPT) - MACRO_KEEP

# 2) Regex compile
RE_HTML    = re.compile(r"<.*?>")
RE_URL     = re.compile(r"(https?://\S+|www\.\S+)")
RE_MENTION = re.compile(r"@\w+")
RE_HASH    = re.compile(r"#\w+")
RE_MULTIWS = re.compile(r"\s+")
# Giữ chữ/số và kí hiệu $, %, +, -, ., ,, /  (phần còn lại thay = space)
RE_KEEP    = re.compile(r"[^A-Za-z0-9\$\%\+\-\,\./\s]")

# Chuẩn hoá viết tắt quốc gia/khối
RE_US = re.compile(r"\bU\.S\.A?\.?\b", re.I)
RE_UK = re.compile(r"\bU\.K\.?\b", re.I)
RE_EU = re.compile(r"\bE\.U\.?\b", re.I)

def expand_contractions(text: str) -> str:
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

def _lemma_smart(w: str) -> str:
    # Heuristic: nếu là động từ thường gặp (đuôi ed/ing/s), thử lemmatize theo verb
    if len(w) > 3 and (w.endswith("ed") or w.endswith("ing") or w.endswith("es") or w.endswith("s")):
        v = lemmatizer.lemmatize(w, pos="v")
        if v != w:
            return v
    return lemmatizer.lemmatize(w)

def preprocess_text_fin(
    text: str,
    *,
    use_lemma: bool = True,
    remove_emoji: bool = False,
    normalize_currency: bool = True,
) -> str:
    if text is None:
        return ""

    t = str(text)

    # 0) Chuẩn hoá US/UK/EU trước khi lower để giữ token đúng
    t = RE_US.sub("US", t)
    t = RE_UK.sub("UK", t)
    t = RE_EU.sub("EU", t)

    # 1) Lower
    t = t.lower()

    # 2) Bỏ HTML/URL/mention/hashtag
    t = RE_HTML.sub(" ", t)
    t = RE_URL.sub(" ", t)
    t = RE_MENTION.sub(" ", t)
    t = RE_HASH.sub(" ", t)

    # 3) Mở rộng contractions
    t = expand_contractions(t)

    # 4) Chuẩn hoá tiền tệ/thập phân/percent
    if normalize_currency:
        t = t.replace("$", " <currency> ")
    # bỏ dấu phẩy nghìn: 2,200 -> 2200
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    # percent từ chữ về kí hiệu
    t = re.sub(r"\bper\s*cent\b", "%", t)
    t = re.sub(r"\bpercent(age)?\b", "%", t)
    t = re.sub(r"\bpct\b", "%", t)

    # 5) Emoji (tuỳ chọn)
    if remove_emoji:
        t = emoji.replace_emoji(t, replace=" ")

    # 6) Lọc kí tự ngoài whitelist
    t = RE_KEEP.sub(" ", t)

    # 7) Co khoảng trắng
    t = RE_MULTIWS.sub(" ", t).strip()

    # 8) Lemma + bỏ stopwords (giữ phủ định & macro tokens)
    if use_lemma:
        toks = []
        for w in t.split():
            if w in CUSTOM_STOPS:
                continue
            toks.append(_lemma_smart(w))
        t = " ".join(toks)

    return t
