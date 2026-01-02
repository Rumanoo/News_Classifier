import streamlit as st
import torch
from transformers import BartTokenizer, BartForSequenceClassification

MODEL_NAME = "Ruman56/news_classifier_model"

@st.cache_resource
def load_model():
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model = BartForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.set_page_config(page_title="News Topic Classifier (BART)")
st.title("📰 News Topic Classifier (BART)")

text = st.text_area("Enter news text", height=200)

if st.button("Classify"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        st.success(f"Predicted Class ID: {prediction}")
