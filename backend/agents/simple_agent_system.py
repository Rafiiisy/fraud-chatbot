"""
Simplified Agent System for Python 3.9 Compatibility
Provides structured AI responses without heavy dependencies
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from dataclasses import dataclass

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.append(str(backend_path))

from core.ai.openai_integration import OpenAIIntegration


@dataclass
class FraudAnalysisResult:
    """Structured result for fraud analysis"""
    question: str
    answer: str
    confidence: float
    data_sources: List[str]
    key_insights: List[str]
    recommendations: List[str]
    risk_level: str
    supporting_evidence: Dict[str, Any]


class SimpleAgentSystem:
    """
    Simplified agent system that provides structured responses using OpenAI
    Compatible with Python 3.9
    """
    
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize OpenAI integration
        self.openai_client = OpenAIIntegration(api_key=self.openai_api_key)
    
    def analyze_fraud_question(self, question: str, context_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze a fraud question using structured AI approach
        
        Args:
            question: The fraud analysis question
            context_data: Additional context data
            
        Returns:
            Structured analysis result
        """
        try:
            # Prepare context
            context = self._prepare_context(question, context_data)
            
            # Create structured prompt
            prompt = self._create_structured_prompt(question, context)
            
            # Get AI response
            response = self.openai_client._make_api_call(prompt, max_tokens=1000)
            
            if not response:
                return {
                    "success": False,
                    "error": "No response generated",
                    "method": "Simple Agent System",
                    "confidence": 0.0
                }
            
            # Parse response into structured format
            return self._parse_response(question, response, context_data)
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}",
                "method": "Simple Agent System Error",
                "confidence": 0.0
            }
    
    def _prepare_context(self, question: str, context_data: Dict[str, Any] = None) -> str:
        """Prepare context for the AI agent"""
        context_parts = [f"Question: {question}"]
        
        if context_data:
            if 'sql_data' in context_data:
                df = context_data['sql_data']
                if isinstance(df, pd.DataFrame) and not df.empty:
                    context_parts.append(f"Data Analysis:\n{df.to_string()}")
            
            if 'document_chunks' in context_data:
                chunks = context_data['document_chunks']
                if chunks:
                    doc_text = "\n".join([chunk.get('text', '') for chunk in chunks])
                    context_parts.append(f"Document Research:\n{doc_text}")
            
            if 'sql_query' in context_data:
                context_parts.append(f"SQL Query Used:\n{context_data['sql_query']}")
        
        return "\n\n".join(context_parts)
    
    def _create_structured_prompt(self, question: str, context: str) -> str:
        """Create a structured prompt for the AI agent"""
        return f"""
        You are an expert fraud analysis AI agent. Analyze the following question and provide a structured response.
        
        {context}
        
        Please provide a comprehensive analysis that includes:
        
        1. **Answer**: Direct, clear answer to the question
        2. **Confidence**: Rate your confidence from 0.0 to 1.0
        3. **Data Sources**: List the specific data sources used
        4. **Key Insights**: 3-5 key findings from the analysis
        5. **Recommendations**: 2-4 actionable recommendations
        6. **Risk Level**: Assess risk as low/medium/high/critical
        7. **Supporting Evidence**: Key statistics, trends, or data points
        
        Format your response as:
        ANSWER: [Your comprehensive answer]
        CONFIDENCE: [0.0-1.0]
        DATA_SOURCES: [List of sources]
        KEY_INSIGHTS: [Insight 1; Insight 2; Insight 3]
        RECOMMENDATIONS: [Recommendation 1; Recommendation 2]
        RISK_LEVEL: [low/medium/high/critical]
        SUPPORTING_EVIDENCE: [Key evidence and statistics]
        """
    
    def _parse_response(self, question: str, response: str, context_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Parse AI response into structured format"""
        try:
            # Extract structured information from response
            lines = response.split('\n')
            
            answer = ""
            confidence = 0.7
            data_sources = ["AI Analysis"]
            key_insights = []
            recommendations = []
            risk_level = "medium"
            supporting_evidence = {}
            
            current_section = None
            
            for line in lines:
                line = line.strip()
                if line.startswith("ANSWER:"):
                    answer = line.replace("ANSWER:", "").strip()
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.replace("CONFIDENCE:", "").strip())
                    except:
                        confidence = 0.7
                elif line.startswith("DATA_SOURCES:"):
                    sources = line.replace("DATA_SOURCES:", "").strip()
                    data_sources = [s.strip() for s in sources.split(',') if s.strip()]
                elif line.startswith("KEY_INSIGHTS:"):
                    insights = line.replace("KEY_INSIGHTS:", "").strip()
                    key_insights = [i.strip() for i in insights.split(';') if i.strip()]
                elif line.startswith("RECOMMENDATIONS:"):
                    recs = line.replace("RECOMMENDATIONS:", "").strip()
                    recommendations = [r.strip() for r in recs.split(';') if r.strip()]
                elif line.startswith("RISK_LEVEL:"):
                    risk_level = line.replace("RISK_LEVEL:", "").strip().lower()
                elif line.startswith("SUPPORTING_EVIDENCE:"):
                    evidence = line.replace("SUPPORTING_EVIDENCE:", "").strip()
                    supporting_evidence = {"evidence": evidence}
            
            # If parsing failed, use the full response as answer
            if not answer:
                answer = response
            
            # Add context-based insights
            if context_data and 'sql_data' in context_data:
                data_sources.append("Transaction Data")
                supporting_evidence["data_points"] = len(context_data['sql_data'])
            
            if context_data and 'document_chunks' in context_data:
                data_sources.append("Fraud Reports")
                supporting_evidence["document_sources"] = len(context_data['document_chunks'])
            
            return {
                "success": True,
                "answer": answer,
                "confidence": confidence,
                "data_sources": data_sources,
                "key_insights": key_insights,
                "recommendations": recommendations,
                "risk_level": risk_level,
                "supporting_evidence": supporting_evidence,
                "method": "Simple Agent System",
                "agent_used": "OpenAI with Structured Prompting"
            }
            
        except Exception as e:
            # Fallback to basic response
            return {
                "success": True,
                "answer": response,
                "confidence": 0.7,
                "data_sources": ["AI Analysis"],
                "key_insights": ["Analysis generated using AI"],
                "recommendations": ["Review findings and validate with domain experts"],
                "risk_level": "medium",
                "supporting_evidence": {"method": "AI-generated analysis"},
                "method": "Simple Agent System (Fallback)",
                "agent_used": "OpenAI Basic"
            }
    
    def analyze_merchant_fraud(self, merchant_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze merchant fraud patterns"""
        question = "Which merchants have the highest fraud rates and what are the risk factors?"
        context_data = {"sql_data": merchant_data}
        return self.analyze_fraud_question(question, context_data)
    
    def analyze_temporal_patterns(self, temporal_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal fraud patterns"""
        question = "How do fraud rates change over time and what are the seasonal patterns?"
        context_data = {"sql_data": temporal_data}
        return self.analyze_fraud_question(question, context_data)
    
    def test_connection(self) -> bool:
        """Test if the agent system is working"""
        try:
            result = self.analyze_fraud_question("What is fraud?")
            return result.get("success", False)
        except Exception as e:
            print(f"Simple agent system test failed: {e}")
            return False
