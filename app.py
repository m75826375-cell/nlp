import streamlit as st
import pandas as pd
import os
import gdown
import torch
import re
import numpy as np
import plotly.express as px

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    pipeline
)
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------
# 🔥 STREAMLIT CONFIG
# -----------------------------------------------------------
st.set_page_config(layout="wide", page_title="ADE Dashboard")
st.title("🧠 ADE Detection System")

# -----------------------------------------------------------
# 🔥 GOOGLE DRIVE MODEL DOWNLOAD
# -----------------------------------------------------------
MODEL_DIR = "models"
NER_MODEL_PATH = os.path.join(MODEL_DIR, "ner_model")
CLS_MODEL_PATH = os.path.join(MODEL_DIR, "cls_model")

NER_FOLDER_ID = "1AAu4rU5aYSzpnnT7wQ1iHRYdPefkpPaA"
CLS_FOLDER_ID = "1eRUtxtS_hNj6VVewr9AFilEE_heVyKAl"

@st.cache_resource
def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(NER_MODEL_PATH):
        st.info("⬇️ Downloading NER model...")
        gdown.download_folder(
            id=NER_FOLDER_ID,
            output=NER_MODEL_PATH,
            quiet=False,
            use_cookies=False
        )

    if not os.path.exists(CLS_MODEL_PATH):
        st.info("⬇️ Downloading Classifier model...")
        gdown.download_folder(
            id=CLS_FOLDER_ID,
            output=CLS_MODEL_PATH,
            quiet=False,
            use_cookies=False
        )

    return NER_MODEL_PATH, CLS_MODEL_PATH


NER_MODEL_PATH, CLS_MODEL_PATH = download_models()

# -----------------------------------------------------------
# 🔥 LOAD MODELS (CACHED)
# -----------------------------------------------------------
@st.cache_resource
def load_ner_model():
    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


@st.cache_resource
def load_classifier():
    tokenizer = AutoTokenizer.from_pretrained(CLS_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(CLS_MODEL_PATH)
    return tokenizer, model


ner_tokenizer, ner_model, device = load_ner_model()
clf_tokenizer, clf_model = load_classifier()

classifier = pipeline(
    "text-classification",
    model=clf_model,
    tokenizer=clf_tokenizer,
    return_all_scores=True,
    device=0 if torch.cuda.is_available() else -1
)

# -----------------------------------------------------------
# 📂 FILE UPLOAD
# -----------------------------------------------------------
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload CSV to continue")
    st.stop()

df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.lower()

if "symptom_text" not in df.columns or "age" not in df.columns:
    st.error("CSV must contain symptom_text and age")
    st.stop()

# -----------------------------------------------------------
# 🔥 AGE GROUPING
# -----------------------------------------------------------
def age_group(age):
    try:
        age = int(age)
    except:
        return "Unknown"
    if age < 18:
        return "Child"
    elif age < 40:
        return "Young Adult"
    elif age < 60:
        return "Middle Age"
    else:
        return "Senior"

df["age_group"] = df["age"].apply(age_group)

# -----------------------------------------------------------
# 🔥 NER PREDICTION
# -----------------------------------------------------------
@st.cache_data
def predict_entities(texts):
    results = []
    for text in texts:
        inputs = ner_tokenizer(text, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            outputs = ner_model(**inputs)

        preds = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
        tokens = ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        entities = [t for t, p in zip(tokens, preds) if p != 0]
        results.append(" ".join(entities))

    return results

st.subheader("🔍 NER Results")
df["entities"] = predict_entities(df["symptom_text"].tolist())
st.dataframe(df.head(10))

# -----------------------------------------------------------
# 🔥 CLASSIFICATION
# -----------------------------------------------------------
def predict_severity(text):
    preds = classifier(text)[0]
    scores = [p["score"] for p in preds]
    return ["Severe", "Moderate", "Mild"][np.argmax(scores)]

df["severity"] = df["symptom_text"].apply(predict_severity)

st.subheader("📊 Severity")
st.dataframe(df[["symptom_text", "severity"]])

# -----------------------------------------------------------
# 🔥 CLUSTERING
# -----------------------------------------------------------
model_embed = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model_embed.encode(df["symptom_text"].tolist())

kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(embeddings)

tsne = TSNE(n_components=2, random_state=42)
coords = tsne.fit_transform(embeddings)

df["x"], df["y"] = coords[:, 0], coords[:, 1]

fig = px.scatter(df, x="x", y="y", color="severity", title="Clusters")
st.plotly_chart(fig, use_container_width=True)
