from transformers import pipeline
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from config import C_MODEL_PATH

tokenizer = AutoTokenizer.from_pretrained(C_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(C_MODEL_PATH)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------------
# 1️⃣ Label Mapping
# -----------------------------
id2label = {0: "Severe", 1: "Moderate", 2: "Mild"}

# -----------------------------
# 2️⃣ Load pipeline with your trained model
# -----------------------------
clf = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    return_all_scores=True
)

# -----------------------------
# 3️⃣ Severity Prediction
# -----------------------------
def predict_severity(text, severe_threshold=0.65, neutral_threshold=0.45):
    preds = clf(text)[0]  # list of dicts: [{'label': 'LABEL_0', 'score': 0.9}, ...]
    probs = np.array([p["score"] for p in preds])
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    # --- Default classifier label ---
    label = id2label.get(pred_idx, "Unknown")
    text_lower = text.lower()

    # -----------------------------
    # 1️⃣ Confidence-based overrides
    # -----------------------------
    severe_keywords = ["hospital", "critical", "death", "severe", "intensive care","extreme", "intense", "life threatening"]
    mild_keywords = ["mild", "slight", "minor", "improved","light", "no immediate side effects"]

    has_severe_kw = any(kw in text_lower for kw in severe_keywords)
    has_mild_kw = any(kw in text_lower for kw in mild_keywords)

    if label == "Severe" and not has_severe_kw and confidence < severe_threshold:
        label = "Moderate"
    elif label == "Mild" and not has_mild_kw and confidence < neutral_threshold:
        label = "Moderate"

    # -----------------------------
    # 2️⃣ Rule-based severity adjustment
    # -----------------------------
    # Mild vs medium adjustment
    mild_terms = [
        "fever", "vomiting", "nausea", "rash", "headache", "soreness",
        "injection site pain", "chills", "tiredness", "body ache", "fatigue",
        "dizziness", "swelling", "redness", "itching", "arm pain", "joint pain",
        "muscle pain", "weakness", "malaise", "loss of appetite", "diarrhea",
        "abdominal pain", "pain", "injection site"
    ]

    # 2a. Severe symptom downgrade if mild symptom is prefixed by 'severe'
    high_keywords = ["severe", "very severe", "critical", "acute", "intense", "extreme",
                     "life-threatening", "fatal", "hospitalized", "admitted", "icu"]
    if any(k in text_lower for k in high_keywords):
        if any(f"severe {m}" in text_lower for m in mild_terms):
            label = "Moderate"  # downgrade severe+mild combo

    # 2b. Mild symptom upgrade if context indicates persistence/significance
    mild_keywords_context = ["mild", "slight", "minimal", "light", "minor", "faint", "tolerable",
                             "transient", "temporary", "local", "small", "limited", "low", "short-term",
                             "negligible", "occasional", "manageable", "brief", "minor irritation",
                             "mild discomfort"]
    if any(w in text_lower for w in mild_keywords_context):
        if any(w in text_lower for w in ["persistent", "prolonged", "recurrent", "noticeable", "significant"]):
            label = "Moderate"

    # 2c. Common mild ADE de-escalation: downgrade unprefixed mild symptoms to Moderate
    if any(phrase in text_lower for phrase in mild_terms) and not any(f"mild {m}" in text_lower for m in mild_terms):
        label = "Moderate"

    # -----------------------------
    # 3️⃣ Prepare probability dict
    # -----------------------------
    prob_dict = {id2label[i]: float(probs[i]) for i in range(len(probs))}

    return label, confidence, prob_dict


# -----------------------------
# 4️⃣ Example texts
# -----------------------------
examples = [
    "Patient developed high fever and severe headache after vaccination.",
    "Mild pain in the arm for one day.",
    "Critical condition and death due to allergic reaction.",
    "Slight fatigue for two days, now recovered.",
    "had high fever and vomiting."
]

# -----------------------------
# 5️⃣ Run predictions
# -----------------------------
for text in examples:
    label, conf, probs = predict_severity(text)
    print(f"🩸 Text: {text}")
    print(f"→ Predicted: {label} (confidence: {conf:.3f})")
    print(f"   Probabilities: {probs}\n")
