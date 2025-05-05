from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Optional, Union, Any

class HealthResponse(BaseModel):
    """Response model for the health endpoint."""
    status: str = "ok"

class RecommendationRequest(BaseModel):
    """Request model for the recommendation endpoint."""
    text: str = Field(..., 
                    title="Input Text",
                    description="Job description or natural language query to get assessment recommendations",
                    min_length=10)
    top_n: Optional[int] = Field(10, 
                               title="Number of Recommendations",
                               description="Number of recommendations to return",
                               ge=1, le=50)

class Assessment(BaseModel):
    """Model for an assessment recommendation."""
    name: str
    url: Optional[str] = None
    description: Optional[str] = None
    remote_testing: Optional[str] = None
    irt_support: Optional[str] = None
    duration: Optional[str] = None
    test_type: Optional[str] = None
    similarity_score: float

class RecommendationResponse(BaseModel):
    """Response model for the recommendation endpoint."""
    query: str
    recommendations: List[Assessment]
    count: int