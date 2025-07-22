import sqlite3
from datetime import datetime
from typing import Optional
import json

# Create a database class to store the analysis history
class AnalysisDatabase:
    def __init__(self, db_path="analysis_history.db"):
        """Initialize the database connection and create tables if they don't exist."""
        self.db_path = db_path
        self._create_tables()

    # Create the necessary tables if they don't exist
    def _create_tables(self):
        """Create the necessary tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create datasets table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            row_count INTEGER,
            column_count INTEGER,
            columns TEXT
        )
        ''')

        # Create questions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER,
            question TEXT NOT NULL,
            category TEXT NOT NULL,
            reasoning TEXT,
            is_custom BOOLEAN DEFAULT 0,
            creation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (dataset_id) REFERENCES datasets (id)
        )
        ''')

        # Create code_analyses table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS code_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id INTEGER,
            code TEXT NOT NULL,
            explanation TEXT,
            result TEXT,
            execution_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (question_id) REFERENCES questions (id)
        )
        ''')

        conn.commit()
        conn.close()

    # Save dataset information and return the dataset ID
    def save_dataset(self, filename: str, row_count: int, column_count: int, columns: list) -> int:
        """Save dataset information and return the dataset ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO datasets (filename, row_count, column_count, columns) VALUES (?, ?, ?, ?)",
            (filename, row_count, column_count, json.dumps(columns))
        )
        dataset_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return dataset_id 

    # Save a question and return the question ID
    def save_question(self, dataset_id: int, question: str, category: str, 
                     reasoning: Optional[str] = None, is_custom: bool = False) -> int:
        """Save a question and return the question ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO questions (dataset_id, question, category, reasoning, is_custom) VALUES (?, ?, ?, ?, ?)",
            (dataset_id, question, category, reasoning or "", is_custom)
        )
        question_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return question_id

    # Save code analysis and return the analysis ID
    def save_code_analysis(self, question_id: int, code: str, explanation: Optional[str] = None, 
                          result: Optional[str] = None) -> int:
        """Save code analysis and return the analysis ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO code_analyses (question_id, code, explanation, result) VALUES (?, ?, ?, ?)",
            (question_id, code, explanation or "", result or "")
        )
        analysis_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return analysis_id

    # Retrieve all questions for a given dataset
    def get_dataset_questions(self, dataset_id: int) -> list:
        """Retrieve all questions for a given dataset."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM questions WHERE dataset_id = ? ORDER BY creation_time",
            (dataset_id,)
        )
        questions = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return questions

    # Retrieve all code analyses for a given question
    def get_question_analyses(self, question_id: int) -> list:
        """Retrieve all code analyses for a given question."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM code_analyses WHERE question_id = ? ORDER BY execution_time",
            (question_id,)
        )
        analyses = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return analyses

    # Retrieve the most recent datasets
    def get_recent_datasets(self, limit: int = 5) -> list:
        """Retrieve the most recent datasets."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM datasets ORDER BY upload_time DESC LIMIT ?",
            (limit,)
        )
        datasets = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return datasets

    # Retrieve the most recent questions
    def get_analysis_history(self) -> list:
        """Get a complete history of analyses with dataset and question information."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
        SELECT 
            d.filename, 
            q.question, 
            q.category, 
            ca.code, 
            ca.explanation, 
            ca.result, 
            ca.execution_time
        FROM code_analyses ca
        JOIN questions q ON ca.question_id = q.id
        JOIN datasets d ON q.dataset_id = d.id
        ORDER BY ca.execution_time DESC
        ''')
        
        history = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return history

    def get_dataset_with_questions(self, dataset_id):
        """Get a dataset and all its questions."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get dataset info
        cursor.execute(
            "SELECT * FROM datasets WHERE id = ?",
            (dataset_id,)
        )
        dataset = cursor.fetchone()
        
        if not dataset:
            conn.close()
            return None
        
        # Get all questions for this dataset
        cursor.execute(
            "SELECT * FROM questions WHERE dataset_id = ? ORDER BY category, creation_time",
            (dataset_id,)
        )
        questions = [dict(row) for row in cursor.fetchall()]
        
        result = dict(dataset)
        result['questions'] = questions
        
        conn.close()
        return result