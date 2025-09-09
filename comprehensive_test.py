#!/usr/bin/env python3
"""
Comprehensive Fraud Chatbot Performance and Quality Evaluation
"""
import sys
from pathlib import Path
import time
import json
from typing import Dict, List, Any

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.append(str(backend_path))

def test_direct_components():
    """Test individual components directly"""
    print("🔧 Testing Individual Components")
    print("=" * 50)
    
    try:
        from core.query_classifier import QueryClassifier
        from core.sql_generator import SQLGenerator
        from core.response_generator import ResponseGenerator
        from data.database import DatabaseManager
        
        # Test Query Classifier
        print("1. Testing Query Classifier...")
        classifier = QueryClassifier()
        test_queries = [
            "How does the daily fraud rate fluctuate over time?",
            "Which merchants exhibit the highest incidence of fraudulent transactions?",
            "What are the primary methods by which credit card fraud is committed?",
            "What are the core components of an effective fraud detection system?",
            "How much higher are fraud rates when the transaction counterpart is located outside the EEA?",
            "What share of total card fraud value in H1 2023 was due to cross-border transactions?"
        ]
        
        for query in test_queries:
            try:
                result = classifier.classify_query(query)
                print(f"   ✅ '{query[:50]}...' -> {result[0].value} (confidence: {result[1]:.2f})")
            except Exception as e:
                print(f"   ❌ '{query[:50]}...' -> Error: {e}")
        
        # Test SQL Generator
        print("\n2. Testing SQL Generator...")
        sql_gen = SQLGenerator()
        for query in test_queries[:4]:  # Test first 4 queries (database-related)
            try:
                question_type, confidence, _ = classifier.classify_query(query)
                sql_info = sql_gen.generate_sql(query, question_type)
                print(f"   ✅ '{query[:30]}...' -> SQL generated")
                print(f"      SQL: {sql_info['sql'][:100]}...")
            except Exception as e:
                print(f"   ❌ '{query[:30]}...' -> Error: {e}")
        
        # Test Database Connection
        print("\n3. Testing Database Connection...")
        db_manager = DatabaseManager()
        if db_manager.connect():
            print("   ✅ Database connected successfully")
            
            # Test a simple query
            success, data, error = db_manager.execute_query("SELECT COUNT(*) as total FROM transactions")
            if success:
                print(f"   ✅ Query executed successfully: {data.iloc[0]['total']} total records")
            else:
                print(f"   ❌ Query failed: {error}")
            
            db_manager.disconnect()
        else:
            print("   ❌ Database connection failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Component testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_service_directly():
    """Test the fraud analysis service directly"""
    print("\n🔧 Testing Fraud Analysis Service Directly")
    print("=" * 50)
    
    try:
        from services.fraud_analysis_service import FraudAnalysisService
        
        service = FraudAnalysisService()
        if service.initialize():
            print("✅ Service initialized successfully")
            
            # Test questions
            test_queries = [
                "How does the daily fraud rate fluctuate over time?",
                "Which merchants exhibit the highest incidence of fraudulent transactions?",
                "What are the primary methods by which credit card fraud is committed?",
                "What are the core components of an effective fraud detection system?",
                "How much higher are fraud rates when the transaction counterpart is located outside the EEA?",
                "What share of total card fraud value in H1 2023 was due to cross-border transactions?"
            ]
            
            for i, query in enumerate(test_queries, 1):
                print(f"\n{i}. Testing: '{query}'")
                try:
                    start_time = time.time()
                    response = service.process_question(query)
                    end_time = time.time()
                    
                    if 'error' in response:
                        print(f"   ❌ Error: {response['error']}")
                    else:
                        print(f"   ✅ Success (took {end_time - start_time:.2f}s)")
                        print(f"      Question Type: {response.get('question_type', 'unknown')}")
                        print(f"      Summary: {response.get('summary', 'No summary')[:100]}...")
                        if 'chart' in response and response['chart']:
                            print(f"      Chart: {response['chart']['type']}")
                        if 'sql_query' in response:
                            print(f"      SQL: {response['sql_query'][:100]}...")
                except Exception as e:
                    print(f"   ❌ Exception: {e}")
            
            service.cleanup()
            return True
        else:
            print("❌ Service initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Service testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoints():
    """Test API endpoints"""
    print("\n🌐 Testing API Endpoints")
    print("=" * 50)
    
    import requests
    
    base_url = "http://localhost:5000"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health endpoint working")
            data = response.json()
            print(f"   Service Status: {data.get('service_status', {})}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False
    
    # Test question endpoint
    test_queries = [
        "How does the daily fraud rate fluctuate over time?",
        "Which merchants exhibit the highest incidence of fraudulent transactions?",
        "What are the primary methods by which credit card fraud is committed?",
        "What are the core components of an effective fraud detection system?",
        "How much higher are fraud rates when the transaction counterpart is located outside the EEA?",
        "What share of total card fraud value in H1 2023 was due to cross-border transactions?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing API with: '{query[:50]}...'")
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/question",
                json={"question": query},
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Success (took {end_time - start_time:.2f}s)")
                print(f"      Question Type: {data.get('question_type', 'unknown')}")
                print(f"      Summary: {data.get('summary', 'No summary')[:100]}...")
                if 'chart' in data and data['chart']:
                    print(f"      Chart: {data['chart']['type']}")
                if 'sql_query' in data:
                    print(f"      SQL: {data['sql_query'][:100]}...")
            else:
                print(f"   ❌ Failed: {response.status_code}")
                print(f"      Response: {response.text[:200]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return True

def evaluate_performance():
    """Evaluate overall performance and quality"""
    print("\n📊 Performance and Quality Evaluation")
    print("=" * 50)
    
    # Test response times
    print("1. Response Time Analysis:")
    test_queries = [
        "How does the daily fraud rate fluctuate over time?",
        "Which merchants exhibit the highest incidence of fraudulent transactions?",
        "What are the primary methods by which credit card fraud is committed?"
    ]
    
    response_times = []
    success_count = 0
    
    for query in test_queries:
        try:
            from services.fraud_analysis_service import FraudAnalysisService
            service = FraudAnalysisService()
            if service.initialize():
                start_time = time.time()
                response = service.process_question(query)
                end_time = time.time()
                
                response_time = end_time - start_time
                response_times.append(response_time)
                
                if 'error' not in response:
                    success_count += 1
                    print(f"   ✅ '{query[:40]}...' - {response_time:.2f}s")
                else:
                    print(f"   ❌ '{query[:40]}...' - Error: {response['error']}")
                
                service.cleanup()
        except Exception as e:
            print(f"   ❌ '{query[:40]}...' - Exception: {e}")
    
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        print(f"\n   Average Response Time: {avg_time:.2f}s")
        print(f"   Success Rate: {success_count}/{len(test_queries)} ({success_count/len(test_queries)*100:.1f}%)")
    
    # Test data quality
    print("\n2. Data Quality Analysis:")
    try:
        from data.database import DatabaseManager
        db_manager = DatabaseManager()
        if db_manager.connect():
            # Check data summary
            summary = db_manager.get_data_summary()
            print(f"   Total Records: {summary.get('total_records', 'Unknown')}")
            print(f"   Fraud Records: {summary.get('fraud_records', 'Unknown')}")
            print(f"   Fraud Rate: {summary.get('fraud_rate', 'Unknown')}")
            print(f"   Date Range: {summary.get('date_range', 'Unknown')}")
            db_manager.disconnect()
    except Exception as e:
        print(f"   ❌ Data quality analysis failed: {e}")

def main():
    """Main test function"""
    print("🧪 Comprehensive Fraud Chatbot Evaluation")
    print("=" * 60)
    
    # Test individual components
    component_success = test_direct_components()
    
    # Test service directly
    service_success = test_service_directly()
    
    # Test API endpoints (if service is running)
    api_success = test_api_endpoints()
    
    # Evaluate performance
    evaluate_performance()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 30)
    print(f"Component Testing: {'✅ PASSED' if component_success else '❌ FAILED'}")
    print(f"Service Testing: {'✅ PASSED' if service_success else '❌ FAILED'}")
    print(f"API Testing: {'✅ PASSED' if api_success else '❌ FAILED'}")
    
    if component_success and service_success:
        print("\n🎉 Overall: System is working correctly!")
    else:
        print("\n⚠️  Overall: Some issues detected that need attention.")

if __name__ == "__main__":
    main()
