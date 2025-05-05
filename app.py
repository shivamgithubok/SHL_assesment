from flask import Flask, render_template, jsonify, redirect, url_for, request
import os
import logging
import json
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
import requests
from embeddings import EmbeddingGenerator
from recommender import AssessmentRecommender

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)
# create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-dev-key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)  # needed for url_for to generate with https

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///assessments.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
# initialize the app with the extension
db.init_app(app)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Simple health check endpoint"""
    return jsonify({"status": "ok"})

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """API endpoint to get recommendations from the recommender"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
            
        text = data['text']
        top_n = data.get('top_n', 10)
        
        if len(text.strip()) < 10:
            return jsonify({"error": "Text input must be at least 10 characters long"}), 400
            
        # Get recommendations using the recommender
        recommender = AssessmentRecommender()
        recommendations = recommender.recommend(text, top_n)
        
        # Return recommendations
        return jsonify({
            "query": text,
            "recommendations": recommendations,
            "count": len(recommendations)
        })
    except Exception as e:
        logging.error(f"Error getting recommendations: {e}")
        return jsonify({"error": "Failed to get recommendations"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)