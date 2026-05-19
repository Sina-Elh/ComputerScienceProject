import re
import string

import nltk
from nltk.corpus import stopwords

from src.config import BIAS_WORDS

_STOP_WORDS = None


def ensure_nltk_resources() -> None:
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")


def get_stop_words() -> set[str]:
    global _STOP_WORDS

    if _STOP_WORDS is None:
        ensure_nltk_resources()
        _STOP_WORDS = set(stopwords.words("english"))

    return _STOP_WORDS


def preprocess_text(text: str) -> str:
    stop_words = get_stop_words()

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)

    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

    return " ".join(tokens)


def remove_bias_words(text: str) -> str:
    for word in BIAS_WORDS:
        text = text.replace(word, "")
    return text


def prepare_content(title: str, body: str) -> str:
    combined_text = f"{title} {body}"
    processed = preprocess_text(combined_text)
    return remove_bias_words(processed)
