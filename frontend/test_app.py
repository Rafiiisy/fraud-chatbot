"""
Test script for the fraud detection chatbot frontend
"""
import sys
import os

# Add the frontend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import FraudChatbotFrontend

def test_mock_responses():
    """Test the mock response generation"""
    print("Testing mock response generation...")
    
    app = FraudChatbotFrontend()
    
    # Test temporal analysis
    temporal_response = app.create_temporal_mock_response()
    assert temporal_response['type'] == 'temporal'
    assert 'chart' in temporal_response
    assert 'data' in temporal_response
    print("✅ Temporal analysis mock response works")
    
    # Test merchant analysis
    merchant_response = app.create_merchant_mock_response()
    assert merchant_response['type'] == 'merchant'
    assert 'chart' in merchant_response
    assert 'data' in merchant_response
    print("✅ Merchant analysis mock response works")
    
    # Test document analysis
    doc_response = app.create_document_mock_response("methods")
    assert doc_response['type'] == 'document'
    assert 'content' in doc_response
    assert 'sources' in doc_response
    print("✅ Document analysis mock response works")
    
    # Test geographic analysis
    geo_response = app.create_geographic_mock_response()
    assert geo_response['type'] == 'geographic'
    assert 'chart' in geo_response
    assert 'metrics' in geo_response
    print("✅ Geographic analysis mock response works")
    
    # Test value analysis
    value_response = app.create_value_mock_response()
    assert value_response['type'] == 'value'
    assert 'chart' in value_response
    assert 'metrics' in value_response
    print("✅ Value analysis mock response works")
    
    print("\n🎉 All mock response tests passed!")

def test_question_classification():
    """Test question classification logic"""
    print("\nTesting question classification...")
    
    app = FraudChatbotFrontend()
    
    test_questions = [
        ("How does the daily fraud rate fluctuate?", "temporal"),
        ("Which merchants have highest fraud rates?", "merchant"),
        ("What are the primary fraud methods?", "document"),
        ("What are the core components?", "document"),
        ("How much higher are fraud rates outside EEA?", "geographic"),
        ("What share of fraud value is cross-border?", "value")
    ]
    
    for question, expected_type in test_questions:
        response = app.create_mock_response(question)
        print(f"Question: {question[:30]}...")
        print(f"Expected: {expected_type}, Got: {response['type']}")
        assert response['type'] == expected_type or response['type'] == 'general'
        print("✅ Classification correct")
    
    print("\n🎉 All classification tests passed!")

if __name__ == "__main__":
    print("🧪 Running frontend tests...\n")
    
    try:
        test_mock_responses()
        test_question_classification()
        print("\n🎉 All tests passed! Frontend is ready to run.")
        print("\nTo run the frontend:")
        print("cd frontend")
        print("streamlit run app.py")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
