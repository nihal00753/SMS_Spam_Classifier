

import json
import os
import sys
import joblib
import streamlit as st

from preprocessing import transform_text

import subprocess

# Train on first boot if the artifact doesn't exist yet.
# This makes the app self-contained: no committed .joblib,
# no cross-version sklearn mismatch between local and Streamlit Cloud.
if not os.path.exists("pipeline.joblib"):
    with st.spinner("First boot: training model (takes ~15 seconds)..."):
        subprocess.run([sys.executable, "train.py"], check=True)

PIPELINE_PATH = "pipeline.joblib"
METRICS_PATH = "metrics.json"

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📱")


@st.cache_resource(show_spinner="Loading model...")
def load_pipeline():
    if not os.path.exists(PIPELINE_PATH):
        st.error(
            f"`{PIPELINE_PATH}` not found. Run `python train.py` first to "
            "train and save the model, then restart this app."
        )
        st.stop()
    return joblib.load(PIPELINE_PATH)


def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


pipeline = load_pipeline()
metrics = load_metrics()

with st.sidebar:
    st.header("📱 SMS Spam Classifier")
    st.markdown(
        "Hybrid word + character TF-IDF features, feeding a Logistic "
        "Regression classifier explicitly selected to minimize false "
        "positives (real messages wrongly marked as spam)."
    )
    st.markdown("**Tech stack:** scikit-learn · NLTK · pandas · Streamlit")
    if metrics:
        st.markdown(
            f"**Held-out test performance:** {metrics['precision']*100:.0f}% "
            f"precision, {metrics['recall']*100:.0f}% recall"
        )
    st.markdown("[View source on GitHub](https://github.com/nihal00753/SMS_Spam_Classifier)")

st.title("📱 SMS Spam Classifier")
st.markdown("Enter an SMS or email message to check if it's **spam** or **not spam**.")

EXAMPLES = {
    "Try a spam example": (
        "Congratulations! You've WON a $1000 gift card. Call 09061743810 "
        "now to claim your prize"
    ),
    "Try a ham example": "Hey, are we still meeting for lunch at 1pm tomorrow?",
}

if "sms_input" not in st.session_state:
    st.session_state.sms_input = ""

example_cols = st.columns(len(EXAMPLES))
for col, (label, text) in zip(example_cols, EXAMPLES.items()):
    if col.button(label, use_container_width=True):
        st.session_state.sms_input = text

input_sms = st.text_area(
    "Enter the message", height=150, placeholder="Type your message here…",
    key="sms_input",
)

if st.button("Predict", type="primary"):
    if not input_sms.strip():
        st.warning("Please enter a message first.")
    else:
        proba = pipeline.predict_proba([input_sms])[0]
        spam_prob, ham_prob = proba[1], proba[0]
        result = int(spam_prob >= 0.5)  # default cutoff -- confirmed near-optimal in the notebook

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
        col2.metric("Ham probability", f"{ham_prob * 100:.1f}%")

        with st.expander("See how your message was cleaned before scoring"):
            st.code(transform_text(input_sms) or "(nothing left after cleaning)")

if metrics:
    with st.expander("Model performance (held-out test set)"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        c2.metric("Precision", f"{metrics['precision']*100:.2f}%")
        c3.metric("Recall", f"{metrics['recall']*100:.2f}%")
        c4.metric("F1", f"{metrics['f1']*100:.2f}%")
        st.caption(
            f"Evaluated on {metrics['n_test']} held-out messages "
            f"(model trained on {metrics['n_train']} messages for this evaluation; "
            "the deployed model is refit on the full dataset)."
        )