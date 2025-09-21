import os

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODELS_DIR = os.getenv(
    "MODELS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
)
RECOMMENDATIONS_PATH = os.path.join(MODELS_DIR, "recommendations.pkl")
EMBEDDINGS_PATH = os.path.join(MODELS_DIR, "embeddings.pkl")

SIM_THRESHOLD = 0.4
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

_model = None
_recommendations = None
_embeddings = None
_category_embeddings = None


def _lazy_load():
    global _model, _recommendations, _embeddings, _category_embeddings
    if _model is None:
        _recommendations = joblib.load(RECOMMENDATIONS_PATH)
        _embeddings = joblib.load(EMBEDDINGS_PATH).astype(np.float32)

        _recommendations["themeName"] = (
            _recommendations["themeName"].str.strip().str.capitalize()
        )
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _model.encode("init", show_progress_bar=False)

        categories = _recommendations["themeName"].unique()
        _category_embeddings = {
            cat: _embeddings[
                _recommendations[_recommendations["themeName"] == cat].index
            ].mean(axis=0)
            for cat in categories
        }


def get_recommendation(prompt: str) -> dict:
    _lazy_load()

    prompt_lower = prompt.lower()

    for cat, keywords in keyword_map.items():
        if any(k in prompt_lower for k in keywords):
            cat_idxs = _recommendations[
                _recommendations["themeName"] == cat.capitalize()
            ].index
            if len(cat_idxs) == 0:
                continue
            cat_embs = _embeddings[cat_idxs]
            prompt_emb = _model.encode([prompt], convert_to_numpy=True)
            sims = cosine_similarity([prompt_emb[0]], cat_embs)[0]
            best_idx = cat_idxs[np.argmax(sims)]
            rec = _recommendations.iloc[best_idx]
            return {
                "category": rec["themeName"],
                "recommendation": rec["RecordingDescription"],
            }

    prompt_emb = _model.encode([prompt], convert_to_numpy=True)
    sims = {
        cat: cosine_similarity([prompt_emb[0]], [emb])[0][0]
        for cat, emb in _category_embeddings.items()
    }
    chosen_category = max(sims, key=sims.get)
    cat_idxs = _recommendations[_recommendations["themeName"] == chosen_category].index

    if len(cat_idxs) == 0:
        return {
            "category": "unknown",
            "recommendation": "No match found — but your"
            "next meditation might be just a word away!",
        }

    cat_embs = _embeddings[cat_idxs]
    sims = cosine_similarity([prompt_emb[0]], cat_embs)[0]
    max_sim = sims.max()
    best_idx = cat_idxs[np.argmax(sims)]
    rec = _recommendations.iloc[best_idx]

    if max_sim < SIM_THRESHOLD:
        return {
            "category": "unknown",
            "recommendation": "No match found — but "
            "your next meditation might be just a word away!",
        }

    return {"category": rec["themeName"], "recommendation": rec["RecordingDescription"]}
