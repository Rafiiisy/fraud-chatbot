"""
Core Module
===========

This is the main core module that exposes all the key components
organized by functionality:

- Query Processing: DynamicQueryAnalyzer, QueryClassifier, SQLGenerator, AISQLGenerator
- Document Processing: HybridDocumentProcessor, DocumentProcessor, etc.
- AI Integration: OpenAIIntegration, FraudPredictor
- Data: DatabaseAPIClient
- Response: ResponseGenerator, ChartGenerator
"""

# Import main classes from submodules
from .query import (
    DynamicQueryAnalyzer,
    QueryClassifier,
    SQLGenerator,
    AISQLGenerator
)

from .document import (
    HybridDocumentProcessor,
    DocumentProcessor,
    SimpleDocumentProcessor,
    OptimizedDocumentProcessor
)

from .ai import (
    OpenAIIntegration,
    FraudPredictor
)

from .data import (
    DatabaseAPIClient
)

from .response import (
    ResponseGenerator,
    ChartGenerator
)

__all__ = [
    # Query Processing
    'DynamicQueryAnalyzer',
    'QueryClassifier',
    'SQLGenerator', 
    'AISQLGenerator',
    
    # Document Processing
    'HybridDocumentProcessor',
    'DocumentProcessor',
    'SimpleDocumentProcessor',
    'OptimizedDocumentProcessor',
    
    # AI Integration
    'OpenAIIntegration',
    'FraudPredictor',
    
    # Data
    'DatabaseAPIClient',
    
    # Response
    'ResponseGenerator',
    'ChartGenerator'
]