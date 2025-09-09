from fastapi import FastAPI
from .api.routes import router as api_router

app = FastAPI(title="Product Genome Hybrid Recommender")
app.include_router(api_router, prefix="/api")

@app.get("/")
def root():
    return {"status":"ok", "message":"Product Genome Recommender API — POST /api/build then /api/recommend"}
