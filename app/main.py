from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import recommend

app = FastAPI(title="Meditation Recommender API")

# -----------------------------
# Enable CORS for Bubble.io frontend
# -----------------------------
origins = ["*"]
# for more security
# or ["https://your-bubble-app.bubbleapps.io"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Include routes
# -----------------------------
app.include_router(recommend.router, prefix="", tags=["recommend"])


# Optional root endpoint
@app.get("/")
def root():
    return {"message": "Meditation Recommender API is running!"}
