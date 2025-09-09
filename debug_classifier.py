#!/usr/bin/env python3
"""
Debug script for query classifier
"""
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.append(str(backend_path))

from core.query_classifier import QueryClassifier
from core.sql_generator import QuestionType

def test_classifier():
    """Test the query classifier"""
    print("Testing Query Classifier...")
    
    classifier = QueryClassifier()
    
    # Test query
    query = "How does the daily fraud rate fluctuate over time?"
    print(f"Query: {query}")
    
    try:
        result = classifier.classify_query(query)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_classifier()
