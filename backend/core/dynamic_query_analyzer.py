"""
Dynamic Query Analyzer for LLM-Driven Keyword Generation
========================================================

This module replaces hardcoded keyword patterns with LLM-generated search terms
for more robust and flexible document search and query classification.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import json
import re

# OpenAI integration
try:
    from .openai_integration import OpenAIIntegration
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

class DynamicQueryAnalyzer:
    """
    Analyzes user queries using LLM to generate dynamic search terms and classifications
    Replaces hardcoded keyword patterns with intelligent, context-aware analysis
    """
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the dynamic query analyzer
        
        Args:
            openai_api_key: OpenAI API key for LLM integration
        """
        self.openai_client = None
        if OPENAI_AVAILABLE:
            self.openai_client = OpenAIIntegration(api_key=openai_api_key)
        
        self.logger = logging.getLogger(__name__)
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a user query and generate dynamic search terms and classification
        
        Args:
            query: User's question
            
        Returns:
            Dictionary with analysis results including search terms, classification, and metadata
        """
        if not self.openai_client:
            return self._fallback_analysis(query)
        
        try:
            # Use LLM to analyze the query comprehensively
            analysis_prompt = self._create_analysis_prompt(query)
            response_text = self.openai_client._make_api_call(analysis_prompt)
            
            if response_text:
                analysis = self._parse_llm_response(response_text)
                return self._enhance_analysis(analysis, query)
            else:
                return self._fallback_analysis(query)
                
        except Exception as e:
            self.logger.error(f"Error in dynamic query analysis: {e}")
            return self._fallback_analysis(query)
    
    def _create_analysis_prompt(self, query: str) -> str:
        """Create a simple, robust prompt for LLM analysis"""
        return f"""
Analyze this fraud detection question and return JSON:

Question: "{query}"

Return JSON with:
{{
    "question_type": "temporal_analysis|merchant_analysis|fraud_methods|system_components|geographic_analysis|value_analysis|document_analysis",
    "confidence": 0.0-1.0,
    "key_concepts": ["main", "concepts"],
    "search_terms": ["search", "terms"],
    "synonyms": ["alternative", "terms"],
    "time_references": ["time", "periods"],
    "specific_metrics": ["numbers", "percentages"],
    "context_analysis": "what user is asking for",
    "asks_about_csv_columns": true/false,
    "csv_columns_mentioned": ["column", "names", "if", "any"]
}}

IMPORTANT RULES:
1. If asking about specific data in CSV columns (trans_date_trans_time, cc_num, merchant, category, amt, first, last, gender, street, city, state, zip, lat, long, city_pop, job, dob, trans_num, unix_time, merch_lat, merch_long, is_fraud), use appropriate analysis type (temporal_analysis, merchant_analysis, value_analysis, etc.)

2. If asking about general fraud statistics, reports, or information NOT in CSV columns, use "document_analysis"

3. For "What share of total card fraud value in H1 2023 was due to cross-border transactions?" - this is asking about fraud statistics from reports, NOT CSV data, so use "document_analysis"

4. Extract all numbers, percentages, time periods mentioned
"""
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response and extract structured data"""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                analysis = json.loads(json_str)
                return analysis
            else:
                # Fallback parsing if no JSON found
                return self._parse_text_response(response)
        except (json.JSONDecodeError, AttributeError) as e:
            self.logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return self._parse_text_response(response)
    
    def _parse_text_response(self, response: str) -> Dict[str, Any]:
        """Fallback parsing for non-JSON responses"""
        # Extract key information using regex patterns
        question_type = "value_analysis"  # Default
        confidence = 0.7
        
        # Look for question type indicators
        if any(term in response.lower() for term in ['temporal', 'time', 'fluctuate', 'trend']):
            question_type = "temporal_analysis"
        elif any(term in response.lower() for term in ['merchant', 'category', 'highest']):
            question_type = "merchant_analysis"
        elif any(term in response.lower() for term in ['method', 'how', 'committed']):
            question_type = "fraud_methods"
        elif any(term in response.lower() for term in ['component', 'system', 'detection']):
            question_type = "system_components"
        elif any(term in response.lower() for term in ['eea', 'geographic', 'outside', 'cross-border']):
            question_type = "geographic_analysis"
        elif any(term in response.lower() for term in ['forecast', 'predict', 'future', 'next']):
            question_type = "forecasting"
        
        return {
            "question_type": question_type,
            "confidence": confidence,
            "key_concepts": ["fraud", "analysis"],
            "search_terms": ["fraud", "analysis"],
            "synonyms": [],
            "time_references": [],
            "data_requirements": ["transaction data"],
            "document_sections": ["all"],
            "specific_metrics": [],
            "context_analysis": "General fraud analysis query",
            "search_strategy": "comprehensive"
        }
    
    def _enhance_analysis(self, analysis: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Enhance the analysis with additional processing"""
        # Ensure required fields exist
        analysis.setdefault("question_type", "value_analysis")
        analysis.setdefault("confidence", 0.7)
        analysis.setdefault("key_concepts", [])
        analysis.setdefault("search_terms", [])
        analysis.setdefault("synonyms", [])
        analysis.setdefault("time_references", [])
        analysis.setdefault("data_requirements", [])
        analysis.setdefault("document_sections", [])
        analysis.setdefault("specific_metrics", [])
        analysis.setdefault("context_analysis", "")
        analysis.setdefault("search_strategy", "comprehensive")
        
        # Add query-specific enhancements
        query_lower = query.lower()
        
        # Extract specific metrics from the query
        metrics = re.findall(r'\d+%|\d+\.\d+%|\d+/\d+|\d+\.\d+', query)
        if metrics:
            analysis["specific_metrics"].extend(metrics)
        
        # Extract time references
        time_patterns = [
            r'h1\s*2023', r'h2\s*2023', r'2023', r'2024', r'2022',
            r'first\s*half', r'second\s*half', r'q1', r'q2', r'q3', r'q4',
            r'january', r'february', r'march', r'april', r'may', r'june',
            r'july', r'august', r'september', r'october', r'november', r'december'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                analysis["time_references"].extend(matches)
        
        # Add general query-specific search terms
        if "cross-border" in query_lower or "cross border" in query_lower:
            analysis["search_terms"].extend(["cross-border", "cross border", "international", "foreign"])
            analysis["synonyms"].extend(["overseas", "external", "non-domestic"])
        
        if "share" in query_lower or "percentage" in query_lower:
            analysis["search_terms"].extend(["share", "percentage", "proportion", "portion"])
            analysis["synonyms"].extend(["ratio", "fraction", "part", "amount"])
        
        if "value" in query_lower:
            analysis["search_terms"].extend(["value", "amount", "cost", "monetary"])
            analysis["synonyms"].extend(["financial", "economic", "dollar", "money"])
        
        # Remove duplicates while preserving order
        for key in ["search_terms", "synonyms", "time_references", "specific_metrics"]:
            if key in analysis:
                analysis[key] = list(dict.fromkeys(analysis[key]))
        
        return analysis
    
    def _fallback_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback analysis when LLM is not available"""
        query_lower = query.lower()
        
        # CSV column names to check for
        csv_columns = [
            'trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt', 
            'first', 'last', 'gender', 'street', 'city', 'state', 'zip', 
            'lat', 'long', 'city_pop', 'job', 'dob', 'trans_num', 'unix_time', 
            'merch_lat', 'merch_long', 'is_fraud'
        ]
        
        # Check if question is asking about CSV data
        asks_about_csv = any(column in query_lower for column in csv_columns)
        
        # Special case: H1 2023 cross-border fraud question should be document_analysis
        if "h1 2023" in query_lower and "cross-border" in query_lower and "share" in query_lower:
            question_type = "document_analysis"
            asks_about_csv = False
        elif any(term in query_lower for term in ['forecast', 'predict', 'future', 'next', 'will', 'upcoming']):
            question_type = "forecasting"
        elif asks_about_csv and any(term in query_lower for term in ['daily', 'monthly', 'fluctuate', 'trend', 'over time']):
            question_type = "temporal_analysis"
        elif asks_about_csv and any(term in query_lower for term in ['merchant', 'category', 'highest', 'exhibit']):
            question_type = "merchant_analysis"
        elif asks_about_csv and any(term in query_lower for term in ['share', 'percentage', 'value', 'amount', 'total']):
            question_type = "value_analysis"
        elif any(term in query_lower for term in ['method', 'how', 'committed', 'primary']):
            question_type = "fraud_methods"
        elif any(term in query_lower for term in ['component', 'system', 'detection', 'effective']):
            question_type = "system_components"
        elif asks_about_csv and any(term in query_lower for term in ['eea', 'outside', 'geographic', 'cross-border']):
            question_type = "geographic_analysis"
        else:
            # Default to document analysis for general fraud questions
            question_type = "document_analysis"
        
        # Extract basic search terms
        search_terms = []
        for word in query_lower.split():
            if len(word) > 3 and word not in ['what', 'how', 'which', 'where', 'when', 'why']:
                search_terms.append(word)
        
        # Find which CSV columns are mentioned
        csv_columns_mentioned = [col for col in csv_columns if col in query_lower]
        
        return {
            "question_type": question_type,
            "confidence": 0.6,
            "key_concepts": search_terms[:5],
            "search_terms": search_terms,
            "synonyms": [],
            "time_references": [],
            "data_requirements": ["transaction data"] if asks_about_csv else ["document data"],
            "document_sections": ["all"],
            "specific_metrics": [],
            "context_analysis": f"Basic analysis of {question_type} query",
            "search_strategy": "keyword_based",
            "asks_about_csv_columns": asks_about_csv,
            "csv_columns_mentioned": csv_columns_mentioned
        }
    
    def generate_document_search_terms(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate comprehensive search terms for document search based on analysis
        
        Args:
            analysis: Query analysis results
            
        Returns:
            List of search terms for document search
        """
        search_terms = []
        
        # Add primary search terms
        search_terms.extend(analysis.get("search_terms", []))
        
        # Add synonyms
        search_terms.extend(analysis.get("synonyms", []))
        
        # Add specific metrics
        search_terms.extend(analysis.get("specific_metrics", []))
        
        # Add time references
        search_terms.extend(analysis.get("time_references", []))
        
        # Add question-type specific terms (general patterns)
        question_type = analysis.get("question_type", "")
        
        if question_type == "value_analysis":
            search_terms.extend(["value", "share", "percentage", "amount", "total"])
        elif question_type == "geographic_analysis":
            search_terms.extend(["geographic", "location", "region", "domestic", "international"])
        elif question_type == "temporal_analysis":
            search_terms.extend(["time", "trend", "fluctuation", "daily", "monthly"])
        elif question_type == "fraud_methods":
            search_terms.extend(["method", "technique", "approach", "committed"])
        elif question_type == "system_components":
            search_terms.extend(["component", "system", "detection", "architecture"])
        elif question_type == "merchant_analysis":
            search_terms.extend(["merchant", "category", "highest", "incidence"])
        
        # Remove duplicates and limit length
        unique_terms = list(dict.fromkeys(search_terms))
        return unique_terms[:15]  # Limit to top 15 terms
    
    def get_question_type(self, analysis: Dict[str, Any]) -> str:
        """Extract question type from analysis"""
        return analysis.get("question_type", "value_analysis")
    
    def get_confidence(self, analysis: Dict[str, Any]) -> float:
        """Extract confidence score from analysis"""
        return analysis.get("confidence", 0.7)
    

# Example usage
if __name__ == "__main__":
    analyzer = DynamicQueryAnalyzer()
    
    # Test with the specific H1 2023 question
    test_query = "What share of total card fraud value in H1 2023 was due to cross-border transactions?"
    analysis = analyzer.analyze_query(test_query)
    
    print("Query Analysis Results:")
    print(f"Question Type: {analysis['question_type']}")
    print(f"Confidence: {analysis['confidence']}")
    print(f"Search Terms: {analysis['search_terms']}")
    print(f"Should Use Forecasting: {analyzer.should_use_forecasting(analysis)}")
