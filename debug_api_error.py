#!/usr/bin/env python3
"""
Debug script to reproduce the API error
"""
import sys
from pathlib import Path
import traceback

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.append(str(backend_path))

def test_api_error():
    """Test the exact same flow as the API"""
    print("Testing API error reproduction...")
    
    try:
        from services.fraud_analysis_service import FraudAnalysisService
        
        # Initialize service exactly like the API does
        service = FraudAnalysisService()
        if not service.initialize():
            print("❌ Service initialization failed")
            return
        
        print("✅ Service initialized")
        
        # Test the exact question that fails in API
        question = "How does the daily fraud rate fluctuate over time?"
        print(f"Testing question: {question}")
        
        # This should reproduce the API error
        response = service.process_question(question)
        
        if 'error' in response:
            print(f"❌ Error reproduced: {response['error']}")
        else:
            print("✅ No error - this is unexpected")
            print(f"Response: {response}")
        
        service.cleanup()
        
    except Exception as e:
        print(f"❌ Exception: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_api_error()
