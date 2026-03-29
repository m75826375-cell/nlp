import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from html import escape
import torch,re
import plotly.express as px

import numpy as np
import shap
import streamlit.components.v1 as components
from sklearn.preprocessing import StandardScaler
    
from config import model_path,C_MODEL_PATH

# -----------------------------------------------------------
# Streamlit Config
# -----------------------------------------------------------
st.set_page_config(layout="wide", page_title="ADEGuard Dashboard")

st.title("🧠 ADEGuard 🧠")
st.subheader("Hybrid ADE Detection & Severity Analysis")

# -----------------------------------------------------------
# Sidebar: Upload CSV
# -----------------------------------------------------------
st.sidebar.markdown("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload CSV with columns 'symptom_text' and 'AGE'", type=["csv"])
if uploaded_file is None:
    st.info("👈 Please upload a CSV file to start analysis.")
    st.stop()

df = pd.read_csv(uploaded_file)

# Validate columns
if "symptom_text" not in df.columns or "age" not in df.columns:
    st.error("CSV must contain 'symptom_text' and 'AGE' columns.")
    st.stop()

if not all(col in df.columns for col in ["symptom_text", "age"]):
    st.error("CSV must contain 'symptom_text', 'age' columns")
else:
    # -----------------------------
    # 2️⃣ Age grouping
    # -----------------------------
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
# Tabs
# -----------------------------------------------------------
tabs = st.tabs(["Named Entity Recognition", "Severity Classification & Explainability" ,"Clustering","📊 Clinical ADE Insights Dashboard"])

# -----------------------------------------------------------
# 1️⃣ NER Tab
# -----------------------------------------------------------
with tabs[0]:
    st.subheader("ADE/DRUG Detection - Named Entity Recognition")

    # -----------------------------
    # 1️⃣ Load BioBERT NER
    # -----------------------------
    # ✅ Define device first
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    import torch

    #model_path = "path/to/your/model"

    # ✅ Step 1: define device FIRST
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Step 2: load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # ✅ Step 3: load model WITHOUT moving it yet
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,  # ensures full weight load
    )

    # ✅ Step 4: move to device
    model = model.to(device)

    model.eval()
    print(f"✅ Model loaded successfully on {device}")



    label_list = ["B-ADE", "B-DRUG", "I-ADE", "I-DRUG", "O"]
    id2label = {i: label for i, label in enumerate(label_list)}

    # -----------------------------
    # 2️⃣ Post-processing dictionary
    # -----------------------------
    POSTPROCESS_DICT = {
        "DRUG": {"pfizer", "moderna", "astrazeneca", "covaxin",
                 "janssen", "johnson", "johnson and johnson", "biontech", "covishield"},
         "ADE": {
            # Core ADEs
            "fever", "headache", "dizziness", "nausea", "rash", "fatigue",
            "chills", "itching", "sweating", "chest pain", "pain", "body ache",
            "swelling", "redness", "muscle pain", "joint pain", "vomiting","discomfort",
            # Extended variants & synonyms
            "chest discomfort", "bodyache", "injection site pain",
            "injection site swelling", "injection site redness", 
            "arm pain", "arm soreness", "muscle soreness", "weakness",
            "tingling", "numbness", "fainting", "shortness of breath",
            "palpitations", "blurred vision", "abdominal pain",
            "stomach ache", "loss of appetite", "pain at injection site","passed away","died",
            "burning sensation", "injection site tenderness","dead","death","fatal","fatality","deceased"
        }
        #"ADE": {"fever", "headache", "dizziness", "nausea",
        #        "rash", "fatigue", "chills", "itching", "sweating", "chest pain","pain", "body ache",
        #        "swelling", "redness", "muscle pain", "joint pain", "vomiting"}
    }

    def normalize(text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def postprocess_entities(text, entities):
        """Add missed entities using dictionary match (case-consistent)"""
        new_entities = {"DRUG": list(entities["DRUG"]), "ADE": list(entities["ADE"])}
        text_norm = normalize(text)

        for ent_type, vocab in POSTPROCESS_DICT.items():
            for word in vocab:
                word_norm = normalize(word)
                if word_norm in text_norm and not any(word_norm in normalize(e) for e in new_entities[ent_type]):
                    new_entities[ent_type].append(word)
        return new_entities

    def clean_entities(entities):
        cleaned = {"DRUG": [], "ADE": []}
        for ade in entities.get("ADE", []):
            ade = ade.strip("., ")
            if ade and ade.lower() not in ["and", "reported", "later", "severe"]:
                cleaned["ADE"].append(ade)
        for drug in entities.get("DRUG", []):
            drug = re.sub(r"\band\b.*", "", drug)
            drug = drug.strip("., ")
            if re.search(r"[A-Z]", drug) or drug.lower() in POSTPROCESS_DICT["DRUG"]:
                cleaned["DRUG"].append(drug)
        return cleaned

    # -----------------------------
    # 3️⃣ NER + Postprocess Prediction
    # -----------------------------
    @st.cache_data(show_spinner=False)
    def predict_entities(texts):
        all_entities, all_highlights = [], []

        for text in texts:
            tokens = re.findall(r"\w+|[^\w\s]", text)
            encoded = tokenizer(tokens, is_split_into_words=True,
                                truncation=True, max_length=512, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**encoded)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                confs, preds = torch.max(probs, dim=-1)
                predictions = preds[0].cpu().numpy()
                confidences = confs[0].cpu().numpy()

            word_ids = encoded.word_ids(batch_index=0)
            pred_labels = []
            previous_word_idx = None

            for idx, word_idx in enumerate(word_ids):
                if word_idx is None or word_idx == previous_word_idx:
                    continue
                label = id2label[predictions[idx]]
                conf = float(confidences[idx])
                pred_labels.append((tokens[word_idx], label, conf))
                previous_word_idx = word_idx

            # Merge contiguous entities
            entities = {"DRUG": [], "ADE": []}
            current_entity, current_words = None, []

            for w, l, conf in pred_labels:
                if l.startswith("B-") or l.startswith("I-"):
                    entity_type = l.split("-")[1]
                    if current_entity == entity_type:
                        current_words.append(w)
                    else:
                        if current_entity and current_words:
                            entities[current_entity].append(" ".join(current_words))
                        current_entity = entity_type
                        current_words = [w]
                else:
                    if current_entity and current_words:
                        entities[current_entity].append(" ".join(current_words))
                    current_entity, current_words = None, []

            if current_entity and current_words:
                entities[current_entity].append(" ".join(current_words))

            # Clean and postprocess entities
            entities_clean = clean_entities(entities)
            entities_final = postprocess_entities(text, entities_clean)

            all_entities.append((entities_final.get("ADE", []), entities_final.get("DRUG", [])))
            all_highlights.append(pred_labels)

        return all_entities, all_highlights

    # -----------------------------
    # 4️⃣ Run NER + Dictionary Correction
    # -----------------------------
    st.info("Extracting ADE/Drug entities using BioBERT")
    entity_results, highlights = predict_entities(df["symptom_text"].tolist())

    df["ADE"], df["DRUG"] = zip(*entity_results)

    # -----------------------------
    # 4️⃣a Add dictionary highlights for token visualization
    # -----------------------------
    def add_dict_highlights(row):
        extra_highlights = list(row["highlights"])
        text_tokens = [t for t, _, _ in extra_highlights]
        text_lower = [t.lower() for t in text_tokens]

        # Helper to mark a span of tokens
        def mark_span(start_idx, end_idx, tag):
            for i in range(start_idx, end_idx + 1):
                tok, old_tag, conf = extra_highlights[i]
                if old_tag == "O":
                    extra_highlights[i] = (tok, tag, conf)

        # ADE highlights
        for ade in row["ADE"]:
            ade_tokens = ade.lower().split()
            n = len(ade_tokens)
            for i in range(len(text_lower) - n + 1):
                if text_lower[i:i+n] == ade_tokens:
                    mark_span(i, i+n-1, "B-ADE")

        # DRUG highlights
        for drug in row["DRUG"]:
            drug_tokens = drug.lower().split()
            n = len(drug_tokens)
            for i in range(len(text_lower) - n + 1):
                if text_lower[i:i+n] == drug_tokens:
                    mark_span(i, i+n-1, "B-DRUG")

        return extra_highlights

    df["highlights"] = highlights
    df["highlights"] = df.apply(add_dict_highlights, axis=1)

    # -----------------------------
    # 5️⃣ Display Results
    # -----------------------------
    st.success("✅ Entity Extraction Complete!")
    # Show "None" for no ADE/DRUG detected
    df["ADE"] = df["ADE"].apply(lambda x: ", ".join(x) if x else "None")
    df["DRUG"] = df["DRUG"].apply(lambda x: ", ".join(x) if x else "None")

    st.dataframe(df[["symptom_text", "age_group", "ADE", "DRUG"]].head(20))

    # Token-level highlights
    st.subheader("Token-Level ADE/Drug Highlights")
    row_idx = st.number_input("Select Row Index to Highlight Tokens", min_value=0, max_value=len(df)-1, value=0)

    highlight_row = df.iloc[row_idx]
    html_text = ""
    for token, tag, conf in highlight_row["highlights"]:
        color = "#ffcccc" if tag in ["B-ADE", "I-ADE"] else "#cce5ff" if tag in ["B-DRUG", "I-DRUG"] else "white"
        tooltip = f"{tag} ({conf*100:.1f}%)" if tag != "O" else ""
        html_text += f'<span title="{tooltip}" style="background-color:{color};padding:2px;margin:1px;border-radius:2px;">{escape(token)}</span> '

    st.markdown(html_text, unsafe_allow_html=True)




# ==================================================
# TAB 1 — SEVERITY CLASSIFIER + SHAP
# ==================================================
with tabs[1]:
    st.subheader("Severity Classification and Explainability")

    # --- Load model ---
    @st.cache_resource
    def load_classifier(path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(path)
            return tokenizer, model
        except Exception as e:
            st.error(f"❌ Error loading classifier: {e}")
            return None, None

    tokenizer, model = load_classifier(C_MODEL_PATH)

    if model and tokenizer:
        clf = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,         # ✅ truncate long inputs
            max_length=512,          # ✅ limit to 512 tokens
            padding=True 
        )

        id2label = {0: "Severe", 1:"Moderate", 2: "Mild"}

        # --- Predict severity ---
        def predict_severity(text):
            preds = clf(text)
            # Handle the structure returned by pipeline with return_all_scores=True
            if isinstance(preds, list) and len(preds) > 0:
                if isinstance(preds[0], list):
                    # preds is [[{label: "Severe", score: 0.1}, {label: "Moderate", score: 0.3}, {label: "Mild", score: 0.6}]]
                    scores = [d["score"] for d in preds[0]]
                else:
                    # preds is [{label: "Mild", score: 0.6}] - single prediction
                    scores = [preds[0]["score"]]
                    # For single prediction, we need to get all scores
                    full_preds = clf(text, return_all_scores=True)
                    if isinstance(full_preds, list) and len(full_preds) > 0 and isinstance(full_preds[0], list):
                        scores = [d["score"] for d in full_preds[0]]
            else:
                scores = [0.0, 0.0, 0.0]  # fallback
            
            pred_id = int(np.argmax(scores))
            label = id2label.get(pred_id, "Unknown")
            #label = id2label.get(int(np.argmax(scores)), "Unknown")
            return label, scores

        df["pred_label"], df["pred_scores"] = zip(*df["symptom_text"].apply(predict_severity))

        st.markdown("Classifier Predictions")
        st.dataframe(df[["symptom_text", "pred_label"]])

        # --- SHAP explainability section ---
        

        @st.cache_resource(show_spinner=False)
        def get_explainer(_pipeline_model):
            return shap.Explainer(_pipeline_model)

        #def st_shap(js_html, height=300):
        #    shap_html = f"<head>{shap.getjs()}</head><body>{js_html}</body>"
        #    components.html(shap_html, height=height)
        def st_shap(plot_html, height=300):
            """Render SHAP plots inside Streamlit properly."""
            shap_html = f"<head>{shap.getjs()}</head><body>{plot_html}</body>"
            components.html(shap_html, height=height)
        
        explainer = get_explainer(clf)

        row_idx = st.number_input(
            "Select Row Index for SHAP Explanation",
            min_value=0,
            max_value=len(df) - 1,
            value=0
        )

        example_text = df.iloc[row_idx]["symptom_text"]

        with st.spinner("Computing SHAP values..."):
            shap_values = explainer([example_text])

        st.success("✅ SHAP values computed")

        # Token-level explanation (raw HTML string)
        text_html = shap.plots.text(shap_values[0], display=False)  # returns str
        st_shap(text_html, height=200)

        # Tokenize and get offsets for word aggregation
        encoding = tokenizer(example_text, return_offsets_mapping=True)
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        offsets = encoding["offset_mapping"]

        # Aggregate subword tokens to words
        # Aggregate subword tokens to words using mean absolute SHAP values
        word_map = []
        word_scores = []
        current_word = ""
        current_vals = []

        for tok, val, (s, e) in zip(tokens, shap_values[0].values, offsets):
            if tok in tokenizer.all_special_tokens:
                continue
            if tok.startswith("##"):
                current_word += tok[2:]
                current_vals.append(val)
            else:
                if current_word:
                    # Use mean absolute value instead of mean raw value
                    word_map.append(current_word)
                    word_scores.append(np.mean(np.abs(current_vals)))
                current_word = tok
                current_vals = [val]
        if current_word:
            word_map.append(current_word)
            word_scores.append(np.mean(np.abs(current_vals)))


        # Normalize scores (0-1)
        norm_scores = (np.array(word_scores) - np.min(word_scores)) / (np.ptp(word_scores) + 1e-6)

        # Highlighted text for UI
        highlighted_text = ""
        for word, score in zip(word_map, norm_scores):
            color = f"rgba(255,0,0,{0.3 + 0.7 * score})"
            highlighted_text += f"<span style='background-color:{color};padding:2px;margin:1px;border-radius:3px'>{word}</span> "

        st.markdown("### Token-level Importance Highlight")
        st.markdown(highlighted_text, unsafe_allow_html=True)

        # Bar chart display
        importance_df = pd.DataFrame({"Word": word_map, "Importance": word_scores})
        importance_df = importance_df.sort_values("Importance", ascending=False)
        st.subheader("Word Importance Scores")
        st.bar_chart(importance_df.set_index("Word"))


# ==================================================
# TAB 2 — CLUSTERING WITH HYBRID SEVERITY
# ==================================================
with tabs[2]:
    st.subheader("ADE/DRUG Clustering")

    # --- Sentence embeddings ---
    model_embed = SentenceTransformer("all-MiniLM-L6-v2")
    df["entities_text"] = df.apply(lambda r: " ".join(r["ADE"] + r["DRUG"]), axis=1)
    embeddings = model_embed.encode(df["entities_text"].astype(str).tolist(), show_progress_bar=True)

    # --- Classifier setup (reuse if available) ---
    @st.cache_resource
    def load_pipeline(path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(path)
            return pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                return_all_scores=True,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            st.error(f"❌ Error loading classifier model: {e}")
            return None

    clf = load_pipeline(C_MODEL_PATH)

    # --- Hybrid severity with explainability ---
    id2label = {0: "Severe", 1:"Moderate", 2: "Mild"}
    label2level = {"severe": "high", "moderate": "medium","mild": "low"}

    def hybrid_severity_explain(text):
        """
        Hybrid severity classifier:
        - Combines optional ML classifier + rule-based detection
        - Handles:
            • Severe symptoms of mild ADE → downgraded to medium
            • Mild symptoms with persistent/significant context → upgraded to medium
            • True high severity (hospitalization, death, life-threatening) → high
            • Explicitly mild ADEs remain low
        """
        text_low = text.lower()
        classifier_label = "unknown"
        source = "rule-based"

        # -----------------------------
        # 1️⃣ Classifier prediction (if available)
        # -----------------------------
        if clf:
            try:
                preds = clf(text)
                pred_idx = int(np.argmax([d["score"] for d in preds[0]]))
                label = id2label.get(pred_idx, "Unknown").lower()
                if label in label2level:
                    classifier_label = label2level[label]
                    source = "classifier"
            except Exception:
                pass

        # -----------------------------
        # 2️⃣ Rule-based detection
        # -----------------------------
        high_keywords = [
            "severe", "very severe", "serious", "critical", "acute", "intense", "extreme",
            "life-threatening", "fatal", "deadly", "lethal", "grave", "profound",
            "excruciating", "unbearable", "debilitating", "disabling",
            "hospitalized", "admitted", "icu", "emergency", "urgent", "unresponsive",
            "worsening", "deteriorating", "progressive", "aggravated",
            "terminal", "end-stage", "passed away", "death", "deceased", "fatality"
        ]

        mild_terms = [
            "fever", "vomiting", "nausea", "rash", "headache", "soreness",
            "injection site pain", "chills", "tiredness", "body ache", "fatigue",
            "dizziness", "swelling", "redness", "itching", "arm pain", "joint pain",
            "muscle pain", "weakness", "malaise", "loss of appetite", "diarrhea",
            "abdominal pain", "pain", "injection site"
        ]

        mild_keywords = [
            "mild", "slight", "minimal", "light", "minor", "faint", "tolerable",
            "transient", "temporary", "local", "small", "limited", "low", "short-term",
            "negligible", "occasional", "manageable", "brief", "minor irritation",
            "mild discomfort"
        ]

        rule_label = "unknown"

        # --- High severity detection (downgrade mild symptoms if prefixed by 'severe') ---
        if any(k in text_low for k in high_keywords):
            if not any(f"severe {m}" in text_low for m in mild_terms):
                rule_label = "high"
            else:
                rule_label = "medium"

        # --- Medium severity keywords ---
        elif any(w in text_low for w in [
            "moderate", "average", "medium", "noticeable", "persistent", "significant",
            "sustained", "ongoing", "recurrent", "prolonged", "controlled",
            "stable", "symptomatic", "continuing", "non-critical"
        ]):
            rule_label = "medium"

        # --- Low severity detection (upgrade if context indicates persistence/significance) ---
        elif any(w in text_low for w in mild_keywords):
            if any(w in text_low for w in ["persistent", "prolonged", "recurrent", "noticeable", "significant"]):
                rule_label = "medium"
            else:
                rule_label = "low"

        # -----------------------------
        # 3️⃣ De-escalation for common mild symptoms
        # -----------------------------
        # Only downgrade to medium if not explicitly marked 'mild'
        if any(phrase in text_low for phrase in mild_terms) and not any(f"mild {m}" in text_low for m in mild_terms):
            return "medium", "rule-based downgrade"

        # -----------------------------
        # 4️⃣ Combine classifier + rule-based results
        # -----------------------------
        priority = {"unknown": 0, "low": 1, "medium": 2, "high": 3}

        if priority[rule_label] > priority[classifier_label]:
            return rule_label, "rule-based override"
        else:
            return classifier_label, source


                



    # --- Clustering ---
    n_samples = len(df)
    n_clusters = min(3, n_samples)
    perplexity = max(2, min(30, (n_samples - 1) / 3))

  
   

    def perform_weighted_clustering(df, n_clusters=None, severity_weight=2.5):
        """
        Perform KMeans clustering with weighted severity.
        """
        # --- Convert severity labels to numeric scores ---
        severity_map = {"low": 1, "medium": 2, "high": 3}
        df["severity_score"] = df["modifier"].map(severity_map).fillna(2)

        # --- Prepare features ---
        ade_encoded = pd.get_dummies(df["ADE"].astype(str), prefix="ADE")

        # Weighted severity
        df["weighted_severity"] = df["severity_score"] * severity_weight

        # Combine features
        X = pd.concat([df[["age", "weighted_severity"]], ade_encoded], axis=1)

        # --- Standardize ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # --- Define cluster count dynamically ---
        if n_clusters is None:
            n_clusters = min(5, len(df) - 1)

        # --- KMeans ---
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df["cluster"] = kmeans.fit_predict(X_scaled)

        return df, kmeans

    # --- Apply hybrid severity ---
    results = df["symptom_text"].apply(hybrid_severity_explain)
    df[["modifier", "severity_source"]] = pd.DataFrame(results.tolist(), index=df.index)

    

    df, kmeans_model = perform_weighted_clustering(df)


    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    X_tsne = tsne.fit_transform(embeddings)
    df["x"], df["y"] = X_tsne[:, 0], X_tsne[:, 1]

     # --- Create descriptive cluster labels for clinicians ---
    cluster_summary = (
        df.groupby("cluster")
          .agg({
              "modifier": lambda x: x.mode()[0] if not x.mode().empty else "unknown",
              "age_group": lambda x: x.mode()[0] if not x.mode().empty else "unknown"
          })
          .reset_index()
    )
    cluster_summary["Cluster_Label"] = cluster_summary.apply(
        lambda r: f"{r['modifier'].capitalize()} severity group ({r['age_group']})", axis=1
    )
    df = df.merge(cluster_summary[["cluster", "Cluster_Label"]], on="cluster", how="left")

    # --- Hover text and plots ---
    df["hover_text"] = [
        f"Entities: {ent}<br>Age: {age}<br>Severity: {mod}<br>Source: {src}<br>Cluster: {clab}"
        for ent, age, mod, src, clab in zip(df["entities_text"], df["age_group"], df["modifier"], df["severity_source"], df["Cluster_Label"])
    ]

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="modifier",
        hover_name="hover_text",
        title="Clusters by Severity",
        color_discrete_map={"low": "green", "medium": "orange", "high": "red"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- By Age Group ---
    fig3 = px.scatter(
        df,
        x="x",
        y="y",
        color="age_group",
        hover_name="hover_text",
        color_discrete_sequence=px.colors.qualitative.Set2,
        title="Clusters by Age Group"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # --- Cluster Summary ---
    st.markdown("### 🧩 Cluster Summary")
    st.dataframe(cluster_summary)

    st.session_state["clustered_df"] = df  # ✅ Store for next tab




# --- TAB 2: Clinical Insights Dashboard ---
with tabs[3]:
    st.subheader("📊 Clinical ADE Insights Dashboard")
    st.markdown("""
    This dashboard helps clinical teams explore AI-classified adverse drug events 
    by drug, ade, severity, and age group.
    """)

    # --- Load clustered data from Tab 2 ---
    df_exp = st.session_state.get("clustered_df", df).copy()

    # --- Explode list columns ---
    for col in ["DRUG", "ADE"]:
        df_exp[col] = df_exp[col].apply(lambda x: x if isinstance(x, list) else [x])
        df_exp = df_exp.explode(col)

    # --- Filters ---
    col1, col2, col3 = st.columns(3)
    unique_drugs = sorted(df_exp["DRUG"].dropna().unique())
    selected_drug = col1.selectbox("Select a Drug", ["All"] + unique_drugs)

    unique_ade = sorted(df_exp["ADE"].dropna().unique())
    selected_ade = col2.selectbox("Select an ADE", ["All"] + unique_ade)

    unique_clusters = sorted(df_exp["Cluster_Label"].dropna().unique())
    selected_cluster = col3.selectbox("Select Cluster", ["All"] + unique_clusters)

    # --- Apply Filters ---
    filtered_df = df_exp.copy()
    if selected_drug != "All":
        filtered_df = filtered_df[filtered_df["DRUG"] == selected_drug]
    if selected_ade != "All":
        filtered_df = filtered_df[filtered_df["ADE"] == selected_ade]
    if selected_cluster != "All":
        filtered_df = filtered_df[filtered_df["Cluster_Label"] == selected_cluster]

    # --- Charts ---
    col1, col2 = st.columns(2)  # Create two columns

    with col1:
        st.subheader("📈 Severity Distribution")
        st.bar_chart(filtered_df["modifier"].value_counts())

    with col2:
        st.subheader("👥 Age Group Distribution")
        st.bar_chart(filtered_df["age_group"].value_counts())

    # --- Data Preview ---
    st.subheader("📄 Filtered Case Details")
    st.dataframe(filtered_df[["symptom_text", "age", "age_group", "ADE", "DRUG", "modifier", "Cluster_Label"]].head(20))

    

    # --- Clinical Summary Table ---
    def generate_clinical_summary(df_input):
        df_sum = df_input.copy()
        for col in ["DRUG", "ADE"]:
            df_sum[col] = df_sum[col].apply(lambda x: x if isinstance(x, list) else [x])
            df_sum = df_sum.explode(col)
        summary = df_sum.groupby(["Cluster_Label", "DRUG", "ADE"]).size().reset_index(name="case_count")
        return summary

    clinical_summary = generate_clinical_summary(filtered_df)

    st.subheader("📋 Clinical Summary Table")
    st.dataframe(clinical_summary)

    # --- Downloads ---
    st.sidebar.markdown("### 📥 Download Clinical Data")

    raw_csv = (
        filtered_df.rename(columns={"modifier": "severity"})[
            ["symptom_text", "age", "age_group", "ADE", "DRUG", "severity", "Cluster_Label"]
        ]
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.sidebar.download_button("Download Filtered Cases CSV", raw_csv, file_name="filtered_cases.csv")

    summary_csv = clinical_summary.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button("Download Clinical Summary CSV", summary_csv, file_name="clinical_summary.csv")

    st.sidebar.success("✅ Clustering complete — Data ready for clinical analysis!")
