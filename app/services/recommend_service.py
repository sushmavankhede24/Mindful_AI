import os
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load dataset + embeddings
recommendations = joblib.load(os.path.join(MODELS_DIR, "recommendations.pkl"))
embeddings = joblib.load(os.path.join(MODELS_DIR, "embeddings.pkl"))

# Normalize categories
recommendations["themeName"] = recommendations["themeName"].str.strip().str.title()

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")
model.encode("init", show_progress_bar=False)  # warmup

# Precompute category embeddings
categories = recommendations["themeName"].unique()
category_embeddings = {
    cat: embeddings[recommendations[recommendations["themeName"] == cat].index].mean(axis=0)
    for cat in categories
}

# Keyword map
keyword_map = {
    "Sleep": ["sleep", "insomnia", "restless", "tired"],
    "Stress": ["stress", "anxious", "tense", "overwhelmed"],
    "Focus": ["focus", "concentration", "work", "study"],
    "Relax": ["relax", "unwind", "calm", "relaxation"],
    "Confidence": ["interview", "presentation", "confidence", "nervous", "assessment", "job", "meeting", "exam"]
}

# Similarity threshold for relevance
SIM_THRESHOLD = 0.4

def get_recommendation(prompt: str) -> dict:
    prompt_lower = prompt.lower()

    # -----------------------------
    # Step 0: Row-level keyword check
    # -----------------------------
    for cat, keywords in keyword_map.items():
        if any(k in prompt_lower for k in keywords):
            # find all rows in this category
            cat_idxs = recommendations[recommendations["themeName"] == cat.title()].index
            cat_embs = embeddings[cat_idxs]
            if len(cat_embs) == 0:
                continue
            prompt_emb = model.encode([prompt], convert_to_numpy=True)
            sims = cosine_similarity([prompt_emb[0]], cat_embs)[0]
            best_idx = cat_idxs[np.argmax(sims)]
            rec = recommendations.iloc[best_idx]
            return {"category": rec["themeName"], "recommendation": rec["RecordingDescription"]}

    # -----------------------------
    # Step 1: Encode prompt
    # -----------------------------
    prompt_emb = model.encode([prompt], convert_to_numpy=True)

    # -----------------------------
    # Step 2: Determine category via embeddings
    # -----------------------------
    sims = {cat: cosine_similarity([prompt_emb[0]], [emb])[0][0] for cat, emb in category_embeddings.items()}
    chosen_category = max(sims, key=sims.get)

    # -----------------------------
    # Step 3: Find best meditation in that category
    # -----------------------------
    cat_idxs = recommendations[recommendations["themeName"] == chosen_category].index
    cat_embs = embeddings[cat_idxs]

    if len(cat_embs) == 0:
        return {
            "category": "unknown",
            "recommendation": "❌ Sorry, I don’t have a meditation suggestion for this input."
        }

    sims = cosine_similarity([prompt_emb[0]], cat_embs)[0]
    max_sim = sims.max()
    best_idx = cat_idxs[np.argmax(sims)]
    rec = recommendations.iloc[best_idx]

    # -----------------------------
    # Step 4: Threshold check for relevance
    # -----------------------------
    if max_sim < SIM_THRESHOLD:
        return {
            "category": "unknown",
            "recommendation": "Sorry, I don’t have a meditation suggestion for this input."
        }

    # -----------------------------
    # Step 5: Return recommendation
    # -----------------------------
    return {"category": rec["themeName"], "recommendation": rec["RecordingDescription"]}
