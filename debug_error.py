#!/usr/bin/env python3
"""
Debug script to find the exact location of the QuestionType comparison error
"""
import sys
from pathlib import Path
import traceback

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.append(str(backend_path))

def test_classifier_with_traceback():
    """Test the query classifier with detailed error tracing"""
    print("Testing Query Classifier with error tracing...")
    
    try:
        from core.query_classifier import QueryClassifier
        from core.sql_generator import QuestionType
        
        classifier = QueryClassifier()
        
        # Test query
        query = "How does the daily fraud rate fluctuate over time?"
        print(f"Query: {query}")
        
        # Enable detailed error reporting
        import sys
        sys.settrace(trace_calls)
        
        result = classifier.classify_query(query)
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

def trace_calls(frame, event, arg):
    """Trace function calls to find where the error occurs"""
    if event == 'call':
        filename = frame.f_code.co_filename
        func_name = frame.f_code.co_name
        line_no = frame.f_lineno
        
        # Only trace files in our backend directory
        if 'backend' in filename:
            print(f"Calling {func_name} in {filename}:{line_no}")
            
            # Check if this is where the error might occur
            if 'max' in func_name or 'min' in func_name:
                print(f"  -> Found max/min function call")
                # Print local variables
                for key, value in frame.f_locals.items():
                    print(f"    {key}: {type(value)} = {value}")
    
    return trace_calls

if __name__ == "__main__":
    test_classifier_with_traceback()
