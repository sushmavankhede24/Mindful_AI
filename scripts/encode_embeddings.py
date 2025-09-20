import os
import pandas as pd
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("data/meditation_data.csv")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings (as numpy arrays)
embeddings = model.encode(df["RecordingDescription"].tolist(), convert_to_numpy=True)

# Save artifacts
os.makedirs("app/models", exist_ok=True)
joblib.dump(embeddings, "app/models/embeddings.pkl")
joblib.dump(df, "app/models/recommendations.pkl")

print("✅ Embeddings and dataset saved.")
