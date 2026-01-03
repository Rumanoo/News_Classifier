import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "Ruman56/news_classifier_model"

LABEL_MAP = {
    "LABEL_0": "Politics",
    "LABEL_1": "Sports",
    "LABEL_2": "Business",
    "LABEL_3": "Sci/Tech"
}

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.set_page_config(page_title="📰 News Topic Classifier")
st.title("📰 News Topic Classifier")

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
            probs = torch.softmax(outputs.logits, dim=1)
            pred_id = torch.argmax(probs, dim=1).item()

        raw_label = model.config.id2label[pred_id]
        final_label = LABEL_MAP.get(raw_label, raw_label)
        confidence = probs[0][pred_id].item() * 100

        st.success(f"🧠 Prediction: **{final_label}**")
        st.info(f"Confidence: **{confidence:.2f}%**")
