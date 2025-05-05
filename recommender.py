import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import logging
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

from database import AssessmentDatabase
from embeddings import EmbeddingGenerator

# Load environment variables from .env file
load_dotenv()

# Get the Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    logging.warning("No Google API key found. Please set GOOGLE_API_KEY in the .env file")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AssessmentRecommender:
    def __init__(self):
        self.database = AssessmentDatabase()
        self.embedding_generator = EmbeddingGenerator(api_key=GOOGLE_API_KEY)
        self.assessments_df = None

        # Load assessments data
        self._load_assessments()

    def _load_assessments(self):
        """Load assessment data from the database."""
        self.assessments_df = self.database.load_assessments()
        if self.assessments_df.empty:
            logging.warning("No assessment data loaded. Recommendations will not be available.")
        else:
            self.assessments_df = self.assessments_df[self.assessments_df['embedding'].notna()]
            logging.info(f"Loaded {len(self.assessments_df)} assessments with embeddings")

    def recommend(self, query: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """Generate recommendations based on a query."""
        if self.assessments_df is None or self.assessments_df.empty:
            logging.error("No assessment data available for recommendations")
            return []

        # Generate embedding for the query
        query_embedding = self.embedding_generator.generate_embedding_for_query(query)

        if query_embedding is None:
            logging.error("Failed to generate embedding for the query")
            return []

        # Calculate similarity scores
        similarities = []
        for idx, row in self.assessments_df.iterrows():
            if row['embedding'] is not None:
                similarity = self._calculate_similarity(query_embedding, row['embedding'])
                similarities.append((idx, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        top_indices = [idx for idx, _ in similarities[:top_n]]
        top_scores = [score for _, score in similarities[:top_n]]

        recommendations = []
        for i, idx in enumerate(top_indices):
            row = self.assessments_df.iloc[idx]
            recommendations.append({
                'name': row['name'],
                'url': row['url'],
                'description': row['description'],
                'remote_testing': row['remote_testing'],
                'irt_support': row['irt_support'],
                'duration': row['duration'],
                'test_type': row['test_type'],
                'similarity_score': float(top_scores[i])
            })

        return recommendations

    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        vec1 = np.array(embedding1).reshape(1, -1)
        vec2 = np.array(embedding2).reshape(1, -1)
        return cosine_similarity(vec1, vec2)[0][0]

    def refresh_data(self):
        """Reload assessment data from the database."""
        self._load_assessments()


if __name__ == "__main__":
    recommender = AssessmentRecommender()
    test_query = "Looking for a cognitive ability assessment for software developers"
    recommendations = recommender.recommend(test_query)

    print(f"Top recommendations for query: '{test_query}'")
    for i, rec in enumerate(recommendations):
        print(f"{i+1}. {rec['name']} (Score: {rec['similarity_score']:.4f})")
        print(f"   Type: {rec['test_type']}, Duration: {rec['duration']}")
        print(f"   URL: {rec['url']}")
        print()
