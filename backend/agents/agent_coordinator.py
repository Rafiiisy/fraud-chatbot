"""
Agent Coordinator for PydanticAI and CrewAI Integration
Coordinates between different AI agent systems for optimal fraud analysis
"""
import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.append(str(backend_path))

from .simple_agent_system import SimpleAgentSystem


class AgentCoordinator:
    """
    Coordinates AI agents for comprehensive fraud analysis
    Uses simplified agent system compatible with Python 3.9
    """
    
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize simplified agent system
        self.agent_system = SimpleAgentSystem(self.openai_api_key)
        
        # Agent selection criteria
        self.agent_selection_rules = {
            "structured_analysis": ["merchant_analysis", "temporal_analysis", "risk_assessment"],
            "comprehensive_research": ["fraud_methods", "system_components", "best_practices"],
            "hybrid_analysis": ["complex_questions", "multi_faceted_analysis"]
        }
    
    def analyze_question(self, question: str, question_type: str, 
                        context_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze a question using the simplified agent system
        
        Args:
            question: The fraud analysis question
            question_type: Type of question (from query classifier)
            context_data: Additional context data
            
        Returns:
            Comprehensive analysis result
        """
        try:
            # Use the simplified agent system
            return self.agent_system.analyze_fraud_question(question, context_data)
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Agent coordination failed: {str(e)}",
                "method": "Error",
                "confidence": 0.0
            }
    
    def analyze_merchant_fraud(self, merchant_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze merchant fraud using simplified agent system"""
        return self.analyze_question(
            "Which merchants have the highest fraud rates and what are the risk factors?",
            "merchant_analysis",
            {"sql_data": merchant_data}
        )
    
    def analyze_temporal_patterns(self, temporal_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns using simplified agent system"""
        return self.analyze_question(
            "How do fraud rates change over time and what are the seasonal patterns?",
            "temporal_analysis",
            {"sql_data": temporal_data}
        )
    
    def test_connection(self) -> bool:
        """Test if the agent coordinator is working"""
        try:
            # Test the simplified agent system
            return self.agent_system.test_connection()
            
        except Exception as e:
            print(f"Agent coordinator test failed: {e}")
            return False
