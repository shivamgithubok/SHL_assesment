from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn
from typing import List

from schema import HealthResponse, RecommendationRequest, RecommendationResponse, Assessment
from recommender import AssessmentRecommender

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="API for recommending SHL assessments based on job descriptions using Gemini embeddings",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency injection for the recommender
def get_recommender():
    return AssessmentRecommender()

@app.get("/", tags=["Root"])
async def root():
    return {"message": "Welcome to the SHL Assessment Recommendation API. Visit /docs to explore."}

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    return {"status": "ok"}

@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(
    request: RecommendationRequest,
    recommender: AssessmentRecommender = Depends(get_recommender)
):
    if not request.text or len(request.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Text input must be at least 10 characters long")
    
    recommendations = recommender.recommend(request.text, request.top_n)
    
    return RecommendationResponse(
        query=request.text,
        recommendations=[Assessment(**rec) for rec in recommendations],
        count=len(recommendations)
    )

@app.post("/refresh", tags=["Administration"])
async def refresh_data(recommender: AssessmentRecommender = Depends(get_recommender)):
    recommender.refresh_data()
    return {"status": "success", "message": "Assessment data refreshed"}

def start():
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start()
