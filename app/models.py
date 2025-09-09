# app/models.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class BuildRequest(BaseModel):
    force: Optional[bool] = False

class Questionnaire(BaseModel):
    avg_price_level: Optional[str] = None   # e.g., "low","mid","high"
    favorite_categories: Optional[List[str]] = []
    preferred_brands: Optional[List[str]] = []
    price_sensitivity: Optional[float] = 1.0   # 0 (not sensitive) .. 2 (very sensitive)
    prefer_newness: Optional[bool] = False
    explicit_favorites: Optional[List[str]] = []  # product_id / product_name

class RecommendRequest(BaseModel):
    questionnaire: Questionnaire
    top_k: Optional[int] = 10
    weights: Optional[List[float]] = [0.45, 0.35, 0.20]  # content, cf, demographic
    mmr_lambda: Optional[float] = 0.7

class RecommendItem(BaseModel):
    product_id: str
    product_name: Optional[str]
    score: float
    content: float
    cf: float
    compatibility: float

class RecommendResponse(BaseModel):
    recommendations: List[RecommendItem]
