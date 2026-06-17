import os
import re
import string
import pickle
import scipy.sparse as sp
import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

ps = PorterStemmer()
sw = set(stopwords.words('english'))


def transform(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', 'url', text)
    text = re.sub(r'\b\d{7,}\b', 'phonenumber', text)
    text = re.sub(r'[£$€]\d+', 'currency', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join([ps.stem(w) for w in text.split() if w not in sw])


@st.cache_resource(show_spinner="Training model, please wait...")
def train_model():
    df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
    df.columns = ['target', 'text']
    df.drop_duplicates(inplace=True)
    df['target'] = df['target'].map({'ham': 0, 'spam': 1})
    df['clean'] = df['text'].apply(transform)

    X_train, _, y_train, _ = train_test_split(
        df['clean'], df['target'],
        test_size=0.2, random_state=2, stratify=df['target']
    )

    wv = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    cv = TfidfVectorizer(analyzer='char_wb', max_features=2000, ngram_range=(2, 4))
    X_tr = sp.hstack([wv.fit_transform(X_train), cv.fit_transform(X_train)])

    model = LogisticRegression(C=3, class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_tr, y_train)

    return wv, cv, model


# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📱")
st.title("📱 SMS Spam Classifier")
st.markdown("Enter an SMS or email message to check if it's **spam** or **not spam**.")

wv, cv, model = train_model()

# ── Input ────────────────────────────────────────────────────
input_sms = st.text_area("Enter the message", height=150, placeholder="Type your message here…")

if st.button("Predict", type="primary"):
    if not input_sms.strip():
        st.warning("Please enter a message first.")
    else:
        cleaned = transform(input_sms)
        X = sp.hstack([wv.transform([cleaned]), cv.transform([cleaned])])

        result    = model.predict(X)[0]
        proba     = model.predict_proba(X)[0]
        spam_prob = proba[1]
        ham_prob  = proba[0]

        st.markdown("---")

        if result == 1:
            st.error("🚨 **SPAM**")
        else:
            st.success("✅ **Not Spam**")

        confidence = spam_prob if result == 1 else ham_prob
        st.markdown(f"**Confidence:** {confidence * 100:.1f}%")
        st.progress(float(confidence))

        col1, col2 = st.columns(2)
        col1.metric("Spam probability", f"{spam_prob * 100:.1f}%")
        col2.metric("Ham probability",  f"{ham_prob  * 100:.1f}%")