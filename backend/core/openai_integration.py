"""
OpenAI Integration for Enhanced Query Classification and Response Generation
"""
import os
import requests
import json
from typing import Dict, List, Optional, Tuple, Any
import logging
from enum import Enum

# Load environment variables
try:
    from dotenv import load_dotenv
    # Try multiple paths for the env file
    env_paths = ['env', '../env', './env', 'backend/env']
    for path in env_paths:
        if os.path.exists(path):
            load_dotenv(path)
            break
    else:
        # If no env file found, try loading from current directory
        load_dotenv()
except ImportError:
    pass

class QuestionType(Enum):
    """Question types for classification"""
    TEMPORAL_ANALYSIS = "temporal_analysis"
    MERCHANT_ANALYSIS = "merchant_analysis"
    FRAUD_METHODS = "fraud_methods"
    SYSTEM_COMPONENTS = "system_components"
    GEOGRAPHIC_ANALYSIS = "geographic_analysis"
    VALUE_ANALYSIS = "value_analysis"
    GENERAL_QUESTION = "general_question"

class OpenAIIntegration:
    """
    Integration with OpenAI API for enhanced query processing
    """
    
    def __init__(self, api_key: str = None):
        """Initialize OpenAI integration"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-4o-mini"
        self.logger = logging.getLogger(__name__)
    
    def test_connection(self) -> bool:
        """Test connection to OpenAI API"""
        try:
            test_prompt = "Hello, please respond with 'Connection successful'"
            response = self._make_api_call(test_prompt, max_tokens=10)
            return response is not None and "successful" in response.lower()
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def classify_query_with_llm(self, question: str) -> Tuple[QuestionType, float, Dict[str, Any]]:
        """Classify query using OpenAI"""
        try:
            prompt = f"""
You are an expert fraud detection analyst. Classify the following question into one of these categories:

AVAILABLE DATA SOURCES:
- CSV Database: Contains fraud transaction data with columns: amt (amount), is_fraud (0/1), merchant, category, date, etc.
- Documents: EBA_ECB 2024 Report on Payment Fraud.pdf, Understanding Credit Card Frauds.pdf

CATEGORIES:
1. TEMPORAL_ANALYSIS - Questions about fraud rates over time, trends, fluctuations (use CSV data)
2. MERCHANT_ANALYSIS - Questions about which merchants/categories have highest fraud (use CSV data)
3. FRAUD_METHODS - Questions about how fraud is committed, fraud techniques (use documents)
4. SYSTEM_COMPONENTS - Questions about fraud detection system components (use documents)
5. GEOGRAPHIC_ANALYSIS - Questions about fraud rates by region, EEA vs non-EEA (use CSV data)
6. VALUE_ANALYSIS - Questions about fraud value, financial impact, H1 2023 data, shares, percentages (use CSV data)
7. GENERAL_QUESTION - General questions not fitting above categories

Question: "{question}"

IMPORTANT: 
- If the question asks for specific data, numbers, percentages, or analysis from the database, classify as VALUE_ANALYSIS, TEMPORAL_ANALYSIS, MERCHANT_ANALYSIS, or GEOGRAPHIC_ANALYSIS
- If the question asks about fraud methods, techniques, or system components, classify as FRAUD_METHODS or SYSTEM_COMPONENTS
- Questions about "share of total", "percentage", "value", "H1 2023" should be VALUE_ANALYSIS

Respond with ONLY the category name (e.g., "TEMPORAL_ANALYSIS") and a confidence score (0.0-1.0) in this format:
CATEGORY: [category_name]
CONFIDENCE: [score]
REASONING: [brief explanation]
"""
            
            response = self._make_api_call(prompt)
            
            if not response:
                return QuestionType.GENERAL_QUESTION, 0.5, {"method": "fallback", "reason": "API call failed"}
            
            # Parse response
            lines = response.strip().split('\n')
            category = QuestionType.GENERAL_QUESTION
            confidence = 0.5
            reasoning = "No reasoning provided"
            
            for line in lines:
                if line.startswith("CATEGORY:"):
                    category_name = line.split(":", 1)[1].strip()
                    try:
                        category = QuestionType(category_name.lower())
                    except ValueError:
                        category = QuestionType.GENERAL_QUESTION
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        confidence = 0.5
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()
            
            metadata = {
                "method": "openai_llm",
                "reasoning": reasoning,
                "api_response": response
            }
            
            return category, confidence, metadata
            
        except Exception as e:
            self.logger.error(f"Error in LLM classification: {e}")
            return QuestionType.GENERAL_QUESTION, 0.3, {"method": "error_fallback", "error": str(e)}
    
    def enhance_response_with_llm(self, question: str, question_type: str, 
                                current_response: str, data_available: bool = True,
                                document_context: str = "") -> str:
        """Enhance response using OpenAI"""
        try:
            prompt = f"""
You are an expert fraud detection analyst. Enhance the following response to make it more professional, accurate, and helpful.

Original Question: "{question}"
Question Type: {question_type}
Data Available: {data_available}
Document Context: {document_context}

Current Response: "{current_response}"

Please provide an enhanced response that:
1. Is more professional and clear
2. Includes relevant insights from the data
3. Provides actionable recommendations when appropriate
4. Maintains accuracy and avoids speculation
5. Uses proper financial terminology

Enhanced Response:
"""
            
            response = self._make_api_call(prompt)
            
            if not response:
                return current_response
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Error in response enhancement: {e}")
            return current_response
    
    def _make_api_call(self, prompt: str, max_tokens: int = 1000) -> Optional[str]:
        """Make API call to OpenAI"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3
            }
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                self.logger.error(f"API call failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error making API call: {e}")
            return None

# Test if this works
if __name__ == "__main__":
    print("Testing OpenAI integration...")
    try:
        openai = OpenAIIntegration()
        print("✅ OpenAIIntegration created")
        result = openai.test_connection()
        print(f"Connection test result: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")
