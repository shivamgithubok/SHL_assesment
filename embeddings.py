import os
import logging
import numpy as np
import pandas as pd
from typing import List, Optional
import google.generativeai as genai
from google.generativeai import types
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmbeddingGenerator:
    """
    Generates text embeddings using Google's Generative AI (text-embedding-004).
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            logging.warning("No Google API key found.")
            self.client = None
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.client = genai
                logging.info("Successfully initialized Gemini embedding client")
            except Exception as e:
                logging.error(f"Error initializing Gemini client: {e}")
                self.client = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        if not self.client:
            logging.error("Embedding client not available.")
            return None

        if not text or text.strip() == "":
            logging.warning("Empty text provided for embedding generation")
            return None

        try:
            result = self.client.embed_content(
                model="models/text-embedding-004",
                content=text
            )
            embedding = result["embedding"]
            return embedding
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            raise

    def generate_embeddings_for_assessments(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.client:
            logging.error("Embedding client not initialized.")
            return df

        result_df = df.copy()
        result_df['embedding'] = None

        for idx, row in result_df.iterrows():
            text_to_embed = f"{row['name']} - {row['description']}"
            if not text_to_embed.strip():
                logging.warning(f"Empty text at index {idx}")
                continue

            try:
                embedding = self.generate_embedding(text_to_embed)
                if embedding:
                    result_df.at[idx, 'embedding'] = embedding
                    logging.info(f"Generated embedding for: {row['name']}")
                else:
                    logging.warning(f"Failed to generate embedding for: {row['name']}")
            except Exception as e:
                logging.error(f"Exception for {row['name']}: {e}")
        
        return result_df

    def generate_embedding_for_query(self, query: str) -> Optional[List[float]]:
        return self.generate_embedding(query)

if __name__ == "__main__":
    generator = EmbeddingGenerator(api_key='your-google-api-key-here')
    if generator.client:
        test_text = "This is a test sentence to check if embeddings work correctly."
        embedding = generator.generate_embedding(test_text)
        if embedding:
            print(f"Successfully generated embedding of dimension: {len(embedding)}")
        else:
            print("Failed to generate test embedding")
    else:
        print("Client initialization failed.")
