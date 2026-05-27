import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer
import pickle

# Load model and tokenizer
model = tf.keras.models.load_model("sarcasm_detection.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 100  # same max_len used in training
class_names = ["Not Sarcastic", "Sarcastic"]

# LIME prediction wrapper
def predict_proba(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post')
    preds = model.predict(padded)
    return np.hstack((1 - preds, preds))

explainer = LimeTextExplainer(class_names=class_names)

# UI layout
st.set_page_config(page_title="Sarcasm Detector", layout="centered")
st.title("🧠 Sarcasm Detector")
st.markdown("Enter a sentence to see if it's sarcastic and understand the reasoning with LIME.")

user_input = st.text_area("✍️ Input your sentence:", placeholder="e.g. good job failing in all subjects")

if st.button("Detect Sarcasm"):
    if user_input.strip():
        # Prediction
        padded_seq = pad_sequences(tokenizer.texts_to_sequences([user_input]), maxlen=max_len)
        prob = model.predict(padded_seq)[0][0]
        label = "Sarcastic" if prob > 0.05 else "Not Sarcastic"
        confidence = round(prob if prob > 0.6 else 1 - prob, 4)

        st.subheader(f"🤖 Prediction: {label}")
        st.caption(f"Confidence: {confidence}")

        # LIME explanation
        with st.spinner("Explaining with LIME..."):
            exp = explainer.explain_instance(user_input, predict_proba, num_features=8)
            st.markdown("### 🧠 Words Influencing the Prediction")
            for word, score in exp.as_list():
                st.markdown(f"- **{word}**: {score:.4f}")
    else:
        st.warning("Please enter a sentence first.")
