import os

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "..", "models")

RECOMMENDATIONS_PATH = os.path.join(MODELS_DIR, "recommendations.pkl")
EMBEDDINGS_PATH = os.path.join(MODELS_DIR, "embeddings.pkl")

# Load data
recommendations = joblib.load(RECOMMENDATIONS_PATH)
embeddings = joblib.load(EMBEDDINGS_PATH)

recommendations["themeName"] = recommendations["themeName"].str.strip().str.capitalize()

# Sentence Transformer
model = SentenceTransformer("all-MiniLM-L6-v2")
model.encode("init", show_progress_bar=False)

categories = recommendations["themeName"].unique()
category_embeddings = {
    cat: embeddings[recommendations[recommendations["themeName"] == cat].index].mean(
        axis=0
    )
    for cat in categories
}

keyword_map = {
    "Sleep": ["sleep", "insomnia", "restless", "tired"],
    "Stress": ["stress", "anxious", "tense", "overwhelmed"],
    "Focus": ["focus", "concentration", "work", "study"],
    "Relax": ["relax", "unwind", "calm", "relaxation"],
    "Confidence": [
        "interview",
        "presentation",
        "confidence",
        "nervous",
        "assessment",
        "job",
        "meeting",
        "exam",
    ],
}

SIM_THRESHOLD = 0.4


def get_recommendation(prompt: str) -> dict:
    prompt_lower = prompt.lower()

    # Step 0: Keyword-based mapping (priority)
    for cat, keywords in keyword_map.items():
        if any(k in prompt_lower for k in keywords):
            cat_name = cat
            cat_idxs = recommendations[recommendations["themeName"] == cat_name].index
            if len(cat_idxs) > 0:
                best_idx = cat_idxs[0]  # first in category
                rec = recommendations.iloc[best_idx]
                return {
                    "category": rec["themeName"],
                    "recommendation": rec["RecordingDescription"],
                }

    # Step 1: Fallback to similarity only if no keyword matched
    prompt_emb = model.encode([prompt], convert_to_numpy=True)
    sims = {
        cat: cosine_similarity([prompt_emb[0]], [emb])[0][0]
        for cat, emb in category_embeddings.items()
    }
    chosen_category = max(sims, key=sims.get)

    cat_idxs = recommendations[recommendations["themeName"] == chosen_category].index
    if len(cat_idxs) == 0:
        return {
            "category": "unknown",
            "recommendation": "No match found — "
            "but your next meditation might be just a word away!",
        }

    cat_embs = embeddings[cat_idxs]
    sims = cosine_similarity([prompt_emb[0]], cat_embs)[0]
    max_sim = sims.max()
    best_idx = cat_idxs[np.argmax(sims)]
    rec = recommendations.iloc[best_idx]

    if max_sim < SIM_THRESHOLD:
        return {
            "category": "unknown",
            "recommendation": "No match found — "
            "but your next meditation might be just a word away!",
        }

    return {"category": rec["themeName"], "recommendation": rec["RecordingDescription"]}
