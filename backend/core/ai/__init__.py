"""
AI Integration Module
====================

This module contains AI-related components:
- OpenAI integration
- Fraud prediction models
"""

from .openai_integration import OpenAIIntegration
from .fraud_predictor import FraudPredictor

__all__ = [
    'OpenAIIntegration',
    'FraudPredictor'
]
