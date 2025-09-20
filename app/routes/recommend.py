from fastapi import APIRouter
from pydantic import BaseModel
from app.services.recommend_service import get_recommendation

router = APIRouter()

class PromptRequest(BaseModel):
    prompt: str

@router.post("/recommend")
def recommend_endpoint(request: PromptRequest):
    return get_recommendation(request.prompt)

