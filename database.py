import os
import json
import sqlite3
import pandas as pd
import pickle
import logging
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AssessmentDatabase:
    """
    Handles storing and retrieving assessment data with embeddings.
    Supports both CSV and SQLite storage methods.
    """
    def __init__(self, db_path: str = 'assessments.db', csv_path: str = 'assessments_with_embeddings.csv'):
        self.db_path = db_path
        self.csv_path = csv_path
        self.use_sqlite = True  # Flag to control which storage method to use
        
        # Initialize database if using SQLite
        if self.use_sqlite:
            self._init_sqlite_db()
    
    def _init_sqlite_db(self):
        """Initialize SQLite database with required schema."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create assessments table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                url TEXT,
                description TEXT,
                remote_testing TEXT,
                irt_support TEXT,
                duration TEXT,
                test_type TEXT,
                embedding BLOB
            )
            ''')
            
            conn.commit()
            conn.close()
            logging.info(f"SQLite database initialized at {self.db_path}")
        except Exception as e:
            logging.error(f"Error initializing SQLite database: {e}")
    
    def save_assessments(self, df: pd.DataFrame) -> bool:
        """
        Save assessment data with embeddings to storage.
        
        Args:
            df (pd.DataFrame): DataFrame containing assessment data with embeddings
            
        Returns:
            bool: Success status
        """
        if self.use_sqlite:
            return self._save_to_sqlite(df)
        else:
            return self._save_to_csv(df)
    
    def _save_to_sqlite(self, df: pd.DataFrame) -> bool:
        """Save assessment data to SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Clear existing data
            cursor.execute("DELETE FROM assessments")
            
            # Insert data
            for _, row in df.iterrows():
                # Pickle the embedding to store as binary
                embedding_pickle = pickle.dumps(row['embedding']) if row['embedding'] is not None else None
                
                cursor.execute('''
                INSERT INTO assessments (name, url, description, remote_testing, irt_support, duration, test_type, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['name'],
                    row['url'],
                    row['description'],
                    row['remote_testing'],
                    row['irt_support'],
                    row['duration'],
                    row['test_type'],
                    embedding_pickle
                ))
            
            conn.commit()
            conn.close()
            logging.info(f"Saved {len(df)} assessments to SQLite database")
            return True
        except Exception as e:
            logging.error(f"Error saving to SQLite database: {e}")
            return False
    
    def _save_to_csv(self, df: pd.DataFrame) -> bool:
        """Save assessment data to CSV file with serialized embeddings."""
        try:
            # Make a copy of the dataframe
            save_df = df.copy()
            
            # Convert embeddings to JSON strings for storage
            save_df['embedding'] = save_df['embedding'].apply(
                lambda x: json.dumps(x) if x is not None else None
            )
            
            # Save to CSV
            save_df.to_csv(self.csv_path, index=False)
            logging.info(f"Saved {len(df)} assessments to CSV at {self.csv_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving to CSV: {e}")
            return False
    
    def load_assessments(self) -> pd.DataFrame:
        """
        Load assessment data with embeddings from storage.
        
        Returns:
            pd.DataFrame: DataFrame containing assessment data with embeddings
        """
        if self.use_sqlite:
            return self._load_from_sqlite()
        else:
            return self._load_from_csv()
    
    def _load_from_sqlite(self) -> pd.DataFrame:
        """Load assessment data from SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Query all assessment data
            query = "SELECT name, url, description, remote_testing, irt_support, duration, test_type, embedding FROM assessments"
            
            # Load data into DataFrame
            df = pd.read_sql_query(query, conn)
            
            # Convert binary embedding data back to lists
            df['embedding'] = df['embedding'].apply(
                lambda x: pickle.loads(x) if x is not None else None
            )
            
            conn.close()
            logging.info(f"Loaded {len(df)} assessments from SQLite database")
            return df
        except Exception as e:
            logging.error(f"Error loading from SQLite database: {e}")
            return pd.DataFrame()
    
    def _load_from_csv(self) -> pd.DataFrame:
        """Load assessment data from CSV file and parse embeddings."""
        try:
            if not os.path.exists(self.csv_path):
                logging.warning(f"CSV file not found at {self.csv_path}")
                return pd.DataFrame()
                
            df = pd.read_csv(self.csv_path)
            
            # Convert JSON embedding strings back to lists
            df['embedding'] = df['embedding'].apply(
                lambda x: json.loads(x) if isinstance(x, str) else None
            )
            
            logging.info(f"Loaded {len(df)} assessments from CSV")
            return df
        except Exception as e:
            logging.error(f"Error loading from CSV: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    # For testing
    db = AssessmentDatabase()
    # Create sample data
    data = {
        'name': ['Test Assessment'],
        'url': ['https://example.com'],
        'description': ['This is a test assessment'],
        'remote_testing': ['Yes'],
        'irt_support': ['No'],
        'duration': ['30 minutes'],
        'test_type': ['Cognitive Assessment'],
        'embedding': [[0.1, 0.2, 0.3, 0.4]]
    }
    df = pd.DataFrame(data)
    
    # Test saving and loading
    db.save_assessments(df)
    loaded_df = db.load_assessments()
    print("Loaded data:")
    print(loaded_df)
