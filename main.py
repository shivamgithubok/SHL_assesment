import os
import logging
import argparse
import pandas as pd
from scraper import scrape_and_save
from embeddings import EmbeddingGenerator
from database import AssessmentDatabase
import api
import uvicorn
import subprocess
from app import app  # Import Flask app for gunicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_data(force_scrape=False):
    """
    Initialize the system by scraping data and generating embeddings if needed.
    
    Args:
        force_scrape (bool): Force scraping even if data exists
    """
    db = AssessmentDatabase()
    csv_path = 'assessments.csv'
    
    # Check if we need to scrape data
    if force_scrape or not os.path.exists(csv_path):
        logging.info("Scraping SHL product catalog...")
        assessments_df = scrape_and_save(csv_path)
        if assessments_df is None or assessments_df.empty:
            logging.error("Failed to scrape assessment data")
            return False
    else:
        logging.info(f"Loading existing assessment data from {csv_path}")
        assessments_df = pd.read_csv(csv_path)
    
    # Generate embeddings if needed
    embeddings_path = 'assessments_with_embeddings.csv'
    if force_scrape or not os.path.exists(embeddings_path):
        logging.info("Generating embeddings for assessments...")
        
        embedding_generator = EmbeddingGenerator()
        
        if embedding_generator.model is None:
            logging.error("Failed to initialize embedding model. Check API key.")
            return False
        
        assessments_with_embeddings = embedding_generator.generate_embeddings_for_assessments(assessments_df)
        
        # Save to database
        db.save_assessments(assessments_with_embeddings)
        logging.info("Assessment data with embeddings saved to database")
    else:
        logging.info("Assessment data with embeddings already exists in database")
    
    return True

def start_fastapi():
    """Start the FastAPI server."""
    uvicorn.run("api:app", host="0.0.0.0", port=8000)

def start_streamlit():
    """Start the Streamlit UI server."""
    streamlit_cmd = ["streamlit", "run", "ui.py", "--server.port=5000", "--server.address=0.0.0.0"]
    subprocess.Popen(streamlit_cmd)

def main():
    parser = argparse.ArgumentParser(description='SHL Assessment Recommendation System')
    parser.add_argument('--force-scrape', action='store_true', help='Force scraping of SHL catalog')
    parser.add_argument('--api-only', action='store_true', help='Start only the API server')
    parser.add_argument('--ui-only', action='store_true', help='Start only the UI server')
    args = parser.parse_args()
    
    # Initialize data
    success = initialize_data(args.force_scrape)
    
    if not success:
        logging.error("Failed to initialize data. Exiting.")
        return
    
    # Start servers based on arguments
    if args.api_only:
        logging.info("Starting API server only")
        start_fastapi()
    elif args.ui_only:
        logging.info("Starting UI server only")
        start_streamlit()
    else:
        # Start both servers
        logging.info("Starting both API and UI servers")
        
        # Start Streamlit in a separate process
        start_streamlit()
        
        # Start FastAPI in the main process
        start_fastapi()

if __name__ == "__main__":
    main()
