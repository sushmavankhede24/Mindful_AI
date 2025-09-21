from fastapi import APIRouter, Body
from pydantic import BaseModel, Field

from app.services.recommend_service import get_recommendation

router = APIRouter()


# Request model with validation and default meditation preprompt
class PromptRequest(BaseModel):
    prompt: str = Field(
        "Guide me through a meditation to relax and focus my mind.",
        min_length=1,
        description="The meditation prompt or request from the user",
    )


@router.post("/recommend")
def recommend_endpoint(request: PromptRequest = Body(...)):
    """
    Receive a meditation prompt and return a recommendation.
    If no prompt is provided, a default calming meditation prompt is used.
    """
    return get_recommendation(request.prompt)
