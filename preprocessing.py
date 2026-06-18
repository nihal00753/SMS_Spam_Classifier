"""
Shared text-cleaning and feature-pipeline code for the SMS spam classifier.

Why this file exists
---------------------
Earlier versions of this project had `transform_text()` copy-pasted into the
notebook, train.py, and app.py separately. They drifted out of sync with each
other (different models, different features) with nothing to catch it. This
module is the single source of truth: train.py, app.py, and the notebook all
import from here, so the model that gets evaluated is exactly the model that
gets deployed.
"""

import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

nltk.download("stopwords", quiet=True)

_ps = PorterStemmer()
_STOP_WORDS = set(stopwords.words("english"))

# Order matters: URLs/phones/currency are tokenized BEFORE generic digits are
# stripped, so e.g. "$1000" becomes the token "currency" instead of just
# vanishing. Their *presence* is a strong spam signal even though the exact
# amount/number isn't useful as a feature.
_URL_RE = re.compile(r"http\S+|www\S+")
_PHONE_RE = re.compile(r"\b\d{7,}\b")
_CURRENCY_RE = re.compile(r"[£$€]\d+")
_DIGIT_RE = re.compile(r"\d+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def transform_text(text: str) -> str:
    """Clean and normalize a single raw SMS/email message into a stemmed,
    token-normalized string ready for TF-IDF vectorization."""
    text = text.lower()
    text = _URL_RE.sub("url", text)
    text = _PHONE_RE.sub("phonenumber", text)
    text = _CURRENCY_RE.sub("currency", text)
    text = _DIGIT_RE.sub("", text)
    text = text.translate(_PUNCT_TABLE)
    tokens = text.split()
    tokens = [_ps.stem(w) for w in tokens if w not in _STOP_WORDS]
    return " ".join(tokens)


def clean_series(texts):
    """sklearn-compatible wrapper (used inside FunctionTransformer) so the
    whole pipeline -- cleaning included -- can be pickled as one object."""
    return pd.Series(texts).apply(transform_text)


def build_pipeline(classifier, use_char_features: bool = True) -> Pipeline:
    """Build the full raw-text -> prediction sklearn Pipeline for a given
    classifier.

    Features are word-level TF-IDF (unigrams + bigrams, catches phrases like
    "free entry" or "claim now") concatenated with char-level TF-IDF (2-4
    character n-grams via 'char_wb', catches obfuscated/misspelled spam
    tokens like "WIN", "fr33", or "cl@im" that stemming alone won't
    normalize). Wrapping the cleaning step in the same Pipeline means a
    single joblib.dump() captures everything needed to go from raw text to
    a prediction -- nothing for app.py to reimplement or get out of sync.

    Set use_char_features=False to compare against word-only features
    (used in the notebook's model-selection step); production code
    (train.py / app.py) always uses the default hybrid feature set.
    """
    branches = [("word", TfidfVectorizer(max_features=3000, ngram_range=(1, 2)))]
    if use_char_features:
        branches.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=2000)))

    return Pipeline([
        ("clean", FunctionTransformer(clean_series)),
        ("features", FeatureUnion(branches)),
        ("clf", classifier),
    ])


def load_data(path: str) -> pd.DataFrame:
    """Load spam.csv into a clean two-column (target, text) DataFrame."""
    df = pd.read_csv(path, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["target", "text"]
    df.drop_duplicates(inplace=True)
    df["target"] = df["target"].map({"ham": 0, "spam": 1})
    return df
