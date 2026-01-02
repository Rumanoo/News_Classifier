
import streamlit as st
import torch
from transformers import BartTokenizer, BartForSequenceClassification

st.set_page_config(page_title="News Topic Classifier")

st.title("📰 News Topic Classifier (BART)")

labels = ["World", "Sports", "Business", "Sci/Tech"]

@st.cache_resource
def load_model():
    tokenizer = BartTokenizer.from_pretrained("news_classifier_model")
    model = BartForSequenceClassification.from_pretrained("news_classifier_model")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

text = st.text_input("Enter a news headline:")

if st.button("Classify"):
    if text.strip() == "":
        st.warning("Please enter a headline.")
    else:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)

        prediction = torch.argmax(outputs.logits, dim=1).item()
        st.success(f"Predicted Category: {labels[prediction]}")
