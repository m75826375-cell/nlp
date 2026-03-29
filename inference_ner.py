from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import re
from config import model_path

# -----------------------------
# 1️⃣ Load model & tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

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
            "palpitations", "blurred vision", "abdominal pain","died",
            "dead","death","fatal","fatality","deceased","passed away",
            "stomach ache", "loss of appetite", "pain at injection site",
            "burning sensation", "injection site tenderness"
        }
}

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def postprocess_entities(text, entities):
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
        if re.search(r"[A-Z]", drug) and len(drug.split()) <= 5:
            cleaned["DRUG"].append(drug)

    return cleaned

# -----------------------------
# 3️⃣ Predict function
# -----------------------------
def predict_entities(sentences):
    id2label = {0:"B-ADE", 1:"B-DRUG", 2:"I-ADE", 3:"I-DRUG", 4:"O"}

    for sent in sentences:
        print("\n==============================")
        print("Sentence:", sent)

        # Tokenize
        tokens = re.findall(r"\w+|[^\w\s]", sent)
        encoded = tokenizer(tokens, is_split_into_words=True,
                            truncation=True, max_length=512, return_tensors=None)
        inputs = {k: torch.tensor([v]).to(device) for k, v in encoded.items()}

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # (1, seq_len, num_labels)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)[0].cpu().numpy()

        # Align tokens
        word_ids = encoded.word_ids(batch_index=0)
        pred_labels = []
        prev_word_idx = None
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx == prev_word_idx:
                continue
            word = tokens[word_idx]
            label_id = predictions[idx]
            label = id2label[label_id]
            confidence = probs[0, idx, label_id].item()
            pred_labels.append((word, label, confidence))
            prev_word_idx = word_idx

        print("🔹 Token-level predictions (with confidence):")
        for word, label, conf in pred_labels:
            print(f"{word:15} -> {label:10} (conf: {conf:.4f})")

        # Merge contiguous entities
        entities = {"DRUG": [], "ADE": []}
        current_entity = None
        current_words = []
        current_confs = []

        for word, label, conf in pred_labels:
            if label in ["B-DRUG", "I-DRUG"]:
                if current_entity == "DRUG":
                    current_words.append(word)
                    current_confs.append(conf)
                else:
                    if current_entity and current_words:
                        entities[current_entity].append((" ".join(current_words), sum(current_confs)/len(current_confs)))
                    current_entity = "DRUG"
                    current_words = [word]
                    current_confs = [conf]
            elif label in ["B-ADE", "I-ADE"]:
                if current_entity == "ADE":
                    current_words.append(word)
                    current_confs.append(conf)
                else:
                    if current_entity and current_words:
                        entities[current_entity].append((" ".join(current_words), sum(current_confs)/len(current_confs)))
                    current_entity = "ADE"
                    current_words = [word]
                    current_confs = [conf]
            else:
                if current_entity and current_words:
                    entities[current_entity].append((" ".join(current_words), sum(current_confs)/len(current_confs)))
                current_entity = None
                current_words = []
                current_confs = []

        if current_entity and current_words:
            entities[current_entity].append((" ".join(current_words), sum(current_confs)/len(current_confs)))

        print("\n🔹 Entity-level predictions (raw model + avg confidence):")
        for ent_type, ent_list in entities.items():
            if ent_list:
                for ent, conf in ent_list:
                    print(f"{ent_type}: {ent} (conf: {conf:.4f})")
            else:
                print(f"{ent_type}: None")

        # Post-process
        entities_clean = {"DRUG": [e for e, _ in entities["DRUG"]],
                          "ADE": [e for e, _ in entities["ADE"]]}
        entities_post = postprocess_entities(sent, entities_clean)

        print("\n🔹 Entity-level predictions (post-processed, fuzzy match):")
        for ent_type, ent_list in entities_post.items():
            print(f"{ent_type}: {', '.join(ent_list) if ent_list else 'None'}")

# -----------------------------
# 4️⃣ Example usage
# -----------------------------
if __name__ == "__main__":
    sentences = [
        "After taking AstraZeneca vaccine, the patient experienced nausea and chest pain.",
        "He was given Covaxin but developed rash and severe itching.",
        "The subject reported fatigue, dizziness, and fever following the Pfizer booster.",
        "Moderna shot was administered and he died.",
        "Patient got Pfizer-BioNTech vaccine and later reported severe dizziness, fatigue, and rash."
    ]

    predict_entities(sentences)

