import streamlit as st
import pandas as pd
import requests
import json
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define API endpoints
API_BASE_URL = "http://localhost:8000"
RECOMMEND_ENDPOINT = f"{API_BASE_URL}/recommend"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

def check_api_health() -> bool:
    """Check if the API is running."""
    try:
        response = requests.get(HEALTH_ENDPOINT)
        return response.status_code == 200 and response.json().get("status") == "ok"
    except Exception as e:
        logging.error(f"API health check failed: {e}")
        return False

def get_recommendations(text: str, top_n: int = 10) -> List[Dict[str, Any]]:
    """
    Get assessment recommendations from the API.
    
    Args:
        text (str): Job description or query
        top_n (int): Number of recommendations to return
        
    Returns:
        List[Dict]: List of assessment recommendations
    """
    try:
        payload = {
            "text": text,
            "top_n": top_n
        }
        
        response = requests.post(RECOMMEND_ENDPOINT, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("recommendations", [])
        else:
            logging.error(f"API request failed with status code {response.status_code}: {response.text}")
            return []
    except Exception as e:
        logging.error(f"Error getting recommendations: {e}")
        return []

def main():
    st.set_page_config(
        page_title="SHL Assessment Recommender",
        page_icon="📊",
        layout="wide"
    )
    
    # Custom CSS for styling
    st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .recommendation-card {
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .score-badge {
        background-color: #4CAF50;
        color: white;
        padding: 5px 10px;
        border-radius: 10px;
        font-size: 0.8em;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App title and description
    st.title("SHL Assessment Recommendation System")
    st.markdown("### Find the most relevant SHL assessments for job roles using AI")
    
    # Check API health
    api_healthy = check_api_health()
    
    if not api_healthy:
        st.error("⚠️ API service is not available. Please make sure the backend server is running.")
        st.stop()
    
    # Input form
    with st.form("recommendation_form"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            text_input = st.text_area(
                "Enter job description or query:",
                height=150,
                placeholder="Example: Looking for assessments to evaluate software engineering candidates with focus on problem-solving and coding skills"
            )
        
        with col2:
            top_n = st.number_input(
                "Number of recommendations:",
                min_value=1,
                max_value=50,
                value=10
            )
            
            submit_button = st.form_submit_button("Get Recommendations")
    
    # Display recommendations if form is submitted
    if submit_button and text_input:
        if len(text_input.strip()) < 10:
            st.warning("Please enter a more detailed description (at least 10 characters).")
        else:
            with st.spinner("Generating recommendations..."):
                recommendations = get_recommendations(text_input, top_n)
            
            if recommendations:
                st.success(f"Found {len(recommendations)} relevant assessments")
                
                # Convert to DataFrame for easier display
                df = pd.DataFrame(recommendations)
                
                # Display recommendations in a nice format
                for i, rec in enumerate(recommendations):
                    with st.container():
                        col1, col2 = st.columns([4, 1])
                        
                        with col1:
                            st.markdown(f"### {i+1}. {rec['name']}")
                            
                            if rec.get('description'):
                                st.markdown(f"**Description:** {rec['description'][:200]}..." if len(rec['description']) > 200 else f"**Description:** {rec['description']}")
                            
                            # Create a table for metadata
                            metadata = {
                                "Test Type": rec.get('test_type', 'Unknown'),
                                "Duration": rec.get('duration', 'Unknown'),
                                "Remote Testing": rec.get('remote_testing', 'Unknown'),
                                "IRT Support": rec.get('irt_support', 'Unknown')
                            }
                            
                            metadata_df = pd.DataFrame([metadata])
                            st.dataframe(metadata_df, hide_index=True)
                            
                            if rec.get('url'):
                                st.markdown(f"[View Assessment Details]({rec['url']})")
                        
                        with col2:
                            # Display similarity score as a percentage
                            score_pct = int(rec['similarity_score'] * 100)
                            st.markdown(f"""
                            <div style='text-align: center; padding: 10px;'>
                                <h1 style='font-size: 2.5em; margin-bottom: 5px;'>{score_pct}%</h1>
                                <p style='margin-top: 0;'>Relevance</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                
                # Add option to download results as CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Recommendations as CSV",
                    data=csv,
                    file_name="shl_recommendations.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No recommendations found. Try modifying your query.")
    
    # Add information about the system
    with st.expander("About this system"):
        st.markdown("""
        This system recommends SHL assessments based on job descriptions or natural language queries.
        
        **How it works:**
        1. The system has scraped the SHL product catalog to gather assessment information
        2. Google's Gemini AI is used to generate text embeddings for both assessments and queries
        3. Recommendations are made based on cosine similarity between embeddings
        
        **Features:**
        - Matches assessments based on semantic meaning, not just keywords
        - Provides relevance scores to help you evaluate matches
        - Includes detailed metadata about each assessment
        """)

if __name__ == "__main__":
    main()
