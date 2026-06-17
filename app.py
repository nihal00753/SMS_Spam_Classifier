
__import__('os').environ['STREAMLIT_SERVER_MAX_MESSAGE_SIZE'] = '500'
import streamlit as st
import pickle
import re
import string
import scipy.sparse as sp
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt',    quiet=True)
nltk.download('stopwords',quiet=True)
nltk.download('punkt_tab',quiet=True)

vecs  = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl',      'rb'))

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))


def transform_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', 'url', text)
    text = re.sub(r'\b\d{7,}\b', 'phonenumber', text)
    text = re.sub(r'[£$€]\d+', 'currency', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [ps.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)


st.set_page_config(page_title="SMS Spam Classifier", page_icon="📱")

st.title("📱 SMS Spam Classifier")
st.markdown("Enter an SMS or email message to check if it's **spam** or **not spam**.")

input_sms = st.text_area("Enter the message", height=150, placeholder="Type your message here…")

if st.button("Predict", type="primary"):
    if not input_sms.strip():
        st.warning("Please enter a message first.")
    else:
        transformed = transform_text(input_sms)
        Xw = vecs['word'].transform([transformed])
        Xc = vecs['char'].transform([transformed])
        X  = sp.hstack([Xw, Xc])

        result = model.predict(X)[0]
        proba  = model.predict_proba(X)[0]
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
