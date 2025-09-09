"""
AI Agent Systems for Fraud Analysis
Implements simplified agent system for comprehensive fraud analysis
"""

from .simple_agent_system import SimpleAgentSystem, FraudAnalysisResult
from .agent_coordinator import AgentCoordinator

__all__ = [
    'SimpleAgentSystem',
    'FraudAnalysisResult',
    'AgentCoordinator'
]
