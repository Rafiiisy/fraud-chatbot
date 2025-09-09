"""
Query Classifier for Fraud Detection Chatbot
Routes questions to appropriate handlers based on content analysis
Enhanced with Deepseek R1 LLM integration
"""
import re
import os
import logging
from typing import Dict, List, Tuple, Optional
from enum import Enum
from .sql_generator import QuestionType

# Import database manager with proper path handling
import sys
import os
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.append(str(backend_path))

from data.api_database_manager import APIDatabaseManager

# Import OpenAI integration
try:
    from ..ai.openai_integration import OpenAIIntegration, QuestionType as LLMQuestionType
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Import dynamic query analyzer
try:
    from .dynamic_query_analyzer import DynamicQueryAnalyzer
    DYNAMIC_ANALYZER_AVAILABLE = True
except ImportError:
    DYNAMIC_ANALYZER_AVAILABLE = False


class QueryClassifier:
    """
    Classifies user queries and routes them to appropriate handlers
    """
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize query classifier
        
        Args:
            use_llm: Whether to use LLM for classification (default: True)
        """
        self.use_llm = use_llm and OPENAI_AVAILABLE
        
        # Initialize OpenAI integration if available
        if self.use_llm:
            try:
                self.openai = OpenAIIntegration()
                self.logger = logging.getLogger(__name__)
            except Exception as e:
                self.logger = logging.getLogger(__name__)
                self.logger.warning(f"Failed to initialize OpenAI integration: {e}")
                self.use_llm = False
        
        # Initialize dynamic query analyzer
        self.query_analyzer = None
        if DYNAMIC_ANALYZER_AVAILABLE:
            self.query_analyzer = DynamicQueryAnalyzer()
        
        # Initialize logger if not already done
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(__name__)
        
        # Initialize database API manager for data availability checks
        try:
            self.db_manager = APIDatabaseManager()
            self.db_available = True
        except Exception as e:
            self.logger.warning(f"Database API not available for data checks: {e}")
            self.db_manager = None
            self.db_available = False
        
        # Define keyword patterns for each question type
        self.patterns = {
            QuestionType.TEMPORAL_ANALYSIS: [
                r'\b(daily|monthly|fluctuate|over time|trend|period|fluctuation)\b',
                r'\b(how does.*rate.*change|rate.*over.*time|temporal|time series)\b',
                r'\b(fluctuate|variation|change.*over.*time)\b',
                r'\b(how does.*fraud.*rate.*fluctuate|fraud.*rate.*over.*time|fraud.*rate.*trend)\b',
                r'\b(fluctuate.*over.*period|rate.*fluctuation|fraud.*rate.*variation)\b'
            ],
            QuestionType.MERCHANT_ANALYSIS: [
                r'\b(merchant|category|categories)\b',
                r'\b(highest|highest incidence|exhibit.*fraud|fraud.*merchant)\b',
                r'\b(which.*merchant|which.*category|top.*merchant|worst.*merchant)\b',
                r'\b(incidence.*fraud|fraud.*incidence)\b',
                r'\b(which.*merchants.*exhibit|merchants.*exhibit.*highest|merchant.*categories.*exhibit)\b',
                r'\b(highest.*incidence.*fraudulent|fraudulent.*transactions.*merchant)\b'
            ],
            QuestionType.FRAUD_METHODS: [
                r'\b(method|methods|committed|technique)\b',
                r'\b(primary.*method|fraud.*committed|how.*fraud.*happen)\b',
                r'\b(credit card fraud.*method|fraud.*technique|fraud.*approach)\b',
                r'\b(primary methods by which|methods by which.*fraud|fraud.*committed)\b',
                r'\b(what.*primary.*method|how.*fraud.*committed|fraud.*technique)\b',
                r'\b(how.*fraud.*committed|fraud.*method.*committed|fraud.*technique.*committed)\b'
            ],
            QuestionType.SYSTEM_COMPONENTS: [
                r'\b(component|components|system|detection.*system|fraud.*detection)\b',
                r'\b(effective.*fraud|fraud.*detection.*system|core.*component)\b',
                r'\b(architecture|framework|structure|elements)\b'
            ],
            QuestionType.GEOGRAPHIC_ANALYSIS: [
                r'\b(eea|outside.*eea|counterpart|location|geographic|region)\b',
                r'\b(how much.*higher.*eea|fraud.*rate.*eea|outside.*eea)\b',
                r'\b(cross.*border|domestic.*vs|international|foreign)\b',
                r'\b(transaction.*counterpart.*located|counterpart.*located.*outside)\b',
                r'\b(fraud.*rate.*when.*counterpart|higher.*fraud.*rate.*outside)\b',
                r'\b(geographic.*analysis|location.*analysis|region.*comparison)\b',
                r'\b(how much.*higher|fraud.*rate.*when|located.*outside)\b',
                r'\b(counterpart.*located|outside.*eea|eea.*outside)\b'
            ],
            QuestionType.VALUE_ANALYSIS: [
                r'\b(value|h1.*2023|share.*total|fraud.*value|card.*fraud.*value)\b',
                r'\b(cross.*border.*transaction|total.*fraud.*value)\b',
                r'\b(percentage.*share|share.*due.*cross.*border)\b'
            ],
            QuestionType.FORECASTING: [
                r'\b(forecast|predict|future|next|upcoming|trend|projection)\b',
                r'\b(estimate|will|going to|expected|forecast|prediction)\b',
                r'\b(outlook|prognosis|next week|next month|next quarter|next year)\b',
                r'\b(in 30 days|in 3 months|in 6 months|in 1 year|tomorrow)\b',
                r'\b(next day|upcoming|future|predict.*fraud|forecast.*fraud)\b',
                r'\b(what.*will.*be.*fraud|fraud.*rate.*next|fraud.*trend.*next)\b',
                r'\b(what.*will.*be|will.*be|can.*expect|expect.*next)\b',
                r'\b(predict.*trends|forecast.*rates|fraud.*rate.*next)\b'
            ]
        }
        
        # Confidence thresholds
        self.confidence_thresholds = {
            QuestionType.TEMPORAL_ANALYSIS: 0.15,  # Lowered threshold for better temporal analysis detection
            QuestionType.MERCHANT_ANALYSIS: 0.15,  # Lowered threshold for better merchant analysis detection
            QuestionType.FRAUD_METHODS: 0.2,  # Lowered threshold for better fraud methods detection
            QuestionType.SYSTEM_COMPONENTS: 0.3,
            QuestionType.GEOGRAPHIC_ANALYSIS: 0.2,  # Lowered threshold for better EEA detection
            QuestionType.VALUE_ANALYSIS: 0.4,
            QuestionType.FORECASTING: 0.3
        }
    
    def classify_query(self, query: str) -> Tuple[QuestionType, float, Dict[str, any]]:
        """
        Classify a user query and determine the appropriate handler
        
        Args:
            query: User's question/query
            
        Returns:
            Tuple of (question_type, confidence_score, metadata)
        """
        # Note: Data availability check will be done after initial classification
        
        # Try dynamic query analyzer first if available
        if self.query_analyzer:
            try:
                analysis = self.query_analyzer.analyze_query(query)
                question_type_str = self.query_analyzer.get_question_type(analysis)
                confidence = self.query_analyzer.get_confidence(analysis)
                
                # Convert string to enum
                question_type = self._convert_string_to_question_type(question_type_str)
                
                
                # Check data availability for value analysis questions with time references
                if (question_type == QuestionType.VALUE_ANALYSIS and 
                    self.db_available and self.db_manager):
                    
                    time_reference = self._extract_time_reference(query.lower())
                    if time_reference:
                        data_available = self._check_data_availability(time_reference)
                        if not data_available:
                            # Route value analysis questions about missing time periods to forecasting
                            self.logger.info(f"Dynamic analyzer classified value analysis with missing data for {time_reference}, routing to forecasting")
                            return QuestionType.FORECASTING, 0.8, {
                                'method': 'dynamic_value_analysis_to_forecasting',
                                'original_type': 'value_analysis',
                                'analysis': analysis,
                                'time_reference': time_reference,
                                'data_available': False,
                                'reason': 'Dynamic analyzer classified value analysis question about missing time period, using forecasting to predict'
                            }
                
                return question_type, confidence, {
                    'method': 'dynamic_analyzer',
                    'analysis': analysis,
                    'confidence_threshold': 0.7,
                    'is_high_confidence': confidence >= 0.7
                }
                
            except Exception as e:
                self.logger.warning(f"Dynamic query analyzer failed: {e}, falling back to LLM")
        
        # Try LLM classification as fallback if available
        if self.use_llm:
            try:
                llm_type, llm_confidence, llm_metadata = self.openai.classify_query_with_llm(query)
                
                # Convert LLM question type to our enum
                question_type = self._convert_llm_question_type(llm_type)
                
                # If LLM confidence is high enough, use it
                if llm_confidence >= 0.7:
                    metadata = {
                        'method': 'llm_primary',
                        'llm_metadata': llm_metadata,
                        'confidence_threshold': 0.7,
                        'is_high_confidence': True
                    }
                    
                    # Check data availability for value analysis questions with time references
                    if (question_type == QuestionType.VALUE_ANALYSIS and 
                        self.db_available and self.db_manager):
                        
                        time_reference = self._extract_time_reference(query.lower())
                        if time_reference:
                            data_available = self._check_data_availability(time_reference)
                            if not data_available:
                                # Route value analysis questions about missing time periods to forecasting
                                self.logger.info(f"LLM classified value analysis with missing data for {time_reference}, routing to forecasting")
                                return QuestionType.FORECASTING, 0.8, {
                                    'method': 'llm_value_analysis_to_forecasting',
                                    'original_type': 'value_analysis',
                                    'llm_metadata': llm_metadata,
                                    'time_reference': time_reference,
                                    'data_available': False,
                                    'reason': 'LLM classified value analysis question about missing time period, using forecasting to predict'
                                }
                    
                    return question_type, llm_confidence, metadata
                
                # If LLM confidence is medium, use it as a hint for regex classification
                elif llm_confidence >= 0.4:
                    final_type, final_confidence, final_metadata = self._hybrid_classification(query, question_type, llm_confidence, llm_metadata)
                    
                    # Check data availability for value analysis questions with time references
                    if (final_type == QuestionType.VALUE_ANALYSIS and 
                        self.db_available and self.db_manager):
                        
                        time_reference = self._extract_time_reference(query.lower())
                        if time_reference:
                            data_available = self._check_data_availability(time_reference)
                            if not data_available:
                                # Route value analysis questions about missing time periods to forecasting
                                self.logger.info(f"Hybrid classified value analysis with missing data for {time_reference}, routing to forecasting")
                                return QuestionType.FORECASTING, 0.8, {
                                    'method': 'hybrid_value_analysis_to_forecasting',
                                    'original_type': 'value_analysis',
                                    'hybrid_metadata': final_metadata,
                                    'time_reference': time_reference,
                                    'data_available': False,
                                    'reason': 'Hybrid classified value analysis question about missing time period, using forecasting to predict'
                                }
                    
                    return final_type, final_confidence, final_metadata
                
            except Exception as e:
                self.logger.warning(f"LLM classification failed, falling back to regex: {e}")
        
        # Fallback to regex-based classification
        question_type, confidence, metadata = self._regex_classification(query)
        
        # Check data availability for value analysis questions with time references
        if (question_type == QuestionType.VALUE_ANALYSIS and 
            self.db_available and self.db_manager):
            
            time_reference = self._extract_time_reference(query.lower())
            if time_reference:
                data_available = self._check_data_availability(time_reference)
                if not data_available:
                    # Route value analysis questions about missing time periods to forecasting
                    self.logger.info(f"Value analysis question with missing data for {time_reference}, routing to forecasting")
                    return QuestionType.FORECASTING, 0.8, {
                        'method': 'value_analysis_to_forecasting',
                        'original_type': 'value_analysis',
                        'time_reference': time_reference,
                        'data_available': False,
                        'reason': 'Value analysis question about missing time period, using forecasting to predict'
                    }
        
        return question_type, confidence, metadata
    
    def _convert_string_to_question_type(self, question_type_str: str) -> QuestionType:
        """
        Convert string question type from dynamic analyzer to enum
        
        Args:
            question_type_str: String question type from analyzer
            
        Returns:
            QuestionType enum value
        """
        type_mapping = {
            'temporal_analysis': QuestionType.TEMPORAL_ANALYSIS,
            'merchant_analysis': QuestionType.MERCHANT_ANALYSIS,
            'fraud_methods': QuestionType.FRAUD_METHODS,
            'system_components': QuestionType.SYSTEM_COMPONENTS,
            'geographic_analysis': QuestionType.GEOGRAPHIC_ANALYSIS,
            'value_analysis': QuestionType.VALUE_ANALYSIS,
            'forecasting': QuestionType.FORECASTING,
            'document_analysis': QuestionType.FRAUD_METHODS  # Route to document processor
        }
        
        return type_mapping.get(question_type_str.lower(), QuestionType.VALUE_ANALYSIS)
    
    def _convert_llm_question_type(self, llm_type) -> QuestionType:
        """Convert LLM question type to our enum"""
        type_mapping = {
            LLMQuestionType.TEMPORAL_ANALYSIS: QuestionType.TEMPORAL_ANALYSIS,
            LLMQuestionType.MERCHANT_ANALYSIS: QuestionType.MERCHANT_ANALYSIS,
            LLMQuestionType.FRAUD_METHODS: QuestionType.FRAUD_METHODS,
            LLMQuestionType.SYSTEM_COMPONENTS: QuestionType.SYSTEM_COMPONENTS,
            LLMQuestionType.GEOGRAPHIC_ANALYSIS: QuestionType.GEOGRAPHIC_ANALYSIS,
            LLMQuestionType.VALUE_ANALYSIS: QuestionType.VALUE_ANALYSIS,
            LLMQuestionType.GENERAL_QUESTION: QuestionType.GENERAL_QUESTION
        }
        return type_mapping.get(llm_type, QuestionType.GENERAL_QUESTION)
    
    def _hybrid_classification(self, query: str, llm_type: QuestionType, 
                             llm_confidence: float, llm_metadata: Dict) -> Tuple[QuestionType, float, Dict]:
        """Use LLM result as hint for regex classification"""
        # Get regex classification
        regex_type, regex_confidence, regex_metadata = self._regex_classification(query)
        
        # If both agree, boost confidence
        if llm_type == regex_type:
            final_confidence = min(1.0, (llm_confidence + regex_confidence) / 2 + 0.1)
            method = 'hybrid_agreement'
        else:
            # If they disagree, use LLM result but with lower confidence
            final_confidence = llm_confidence * 0.8
            method = 'hybrid_llm_preference'
        
        metadata = {
            'method': method,
            'llm_type': llm_type.value,
            'llm_confidence': llm_confidence,
            'regex_type': regex_type.value,
            'regex_confidence': regex_confidence,
            'llm_metadata': llm_metadata,
            'regex_metadata': regex_metadata
        }
        
        return llm_type, final_confidence, metadata
    
    def _regex_classification(self, query: str) -> Tuple[QuestionType, float, Dict]:
        """Original regex-based classification"""
        query_lower = query.lower()
        
        # Check data availability first for time-based questions
        if self.db_available and self.db_manager:
            time_reference = self._extract_time_reference(query_lower)
            if time_reference:
                data_available = self._check_data_availability(time_reference)
                if not data_available:
                    # If data is missing for the requested time period, route to forecasting
                    self.logger.info(f"Data not available for {time_reference}, routing to forecasting")
                    return QuestionType.FORECASTING, 0.9, {
                        'method': 'data_availability_check',
                        'time_reference': time_reference,
                        'data_available': False,
                        'reason': 'Missing historical data, using forecasting to predict'
                    }
        
        # Calculate confidence scores for each question type
        scores = {}
        matched_patterns = {}
        
        for question_type, patterns in self.patterns.items():
            score = 0
            matched = []
            
            for pattern in patterns:
                matches = re.findall(pattern, query_lower, re.IGNORECASE)
                if matches:
                    # Give more weight to geographic patterns
                    weight = 0.3 if question_type == QuestionType.GEOGRAPHIC_ANALYSIS else 0.2
                    score += len(matches) * weight
                    matched.extend(matches)
            
            # Special boost for EEA-related questions
            if question_type == QuestionType.GEOGRAPHIC_ANALYSIS:
                if 'eea' in query_lower and ('outside' in query_lower or 'counterpart' in query_lower):
                    score += 0.5  # Strong boost for EEA + outside/counterpart combination
            
            # Special boost for temporal analysis questions
            if question_type == QuestionType.TEMPORAL_ANALYSIS:
                if 'fluctuate' in query_lower and ('rate' in query_lower or 'fraud' in query_lower):
                    score += 0.4  # Strong boost for fluctuation + rate/fraud combination
                if 'daily' in query_lower and 'monthly' in query_lower:
                    score += 0.3  # Boost for questions asking about both daily and monthly
            
            # Special boost for merchant analysis questions
            if question_type == QuestionType.MERCHANT_ANALYSIS:
                if 'which' in query_lower and ('merchant' in query_lower or 'category' in query_lower):
                    score += 0.4  # Strong boost for "which merchant/category" questions
                if 'highest' in query_lower and 'incidence' in query_lower:
                    score += 0.3  # Boost for "highest incidence" questions
            
            # Normalize score by query length
            if not isinstance(score, (int, float)):
                self.logger.warning(f"Score is not a number: {type(score)} = {score}")
                score = 0
            normalized_score = min(score / max(len(query.split()), 1), 1.0)
            scores[question_type] = normalized_score
            matched_patterns[question_type] = matched
        
        # Find the best match
        try:
            best_type = None
            best_score = -1
            for question_type, score in scores.items():
                if score > best_score:
                    best_type = question_type
                    best_score = score
        except Exception as e:
            self.logger.error(f"Error finding best question type: {e}")
            raise
        
        # Check if confidence meets threshold
        threshold = self.confidence_thresholds.get(best_type, 0.3)
        
        if best_score < threshold:
            # If confidence is too low, try fallback classification
            best_type, best_score = self._fallback_classification(query_lower)
        elif best_score >= threshold and best_score <= threshold + 0.1:  # Allow some tolerance around threshold
            # If confidence is at or near threshold, check for document keywords override
            if self._should_override_to_document_analysis(query_lower, best_type):
                best_type, best_score = self._get_document_analysis_type(query_lower)
        
        # Prepare metadata
        metadata = {
            'method': 'regex',
            'all_scores': {k.value: v for k, v in scores.items()},
            'matched_patterns': {k.value: v for k, v in matched_patterns.items()},
            'confidence_threshold': threshold,
            'is_high_confidence': best_score >= threshold
        }
        
        return best_type, best_score, metadata
    
    def _should_override_to_document_analysis(self, query_lower: str, current_type: QuestionType) -> bool:
        """
        Check if a database analysis classification should be overridden to document analysis
        
        Args:
            query_lower: Lowercase query
            current_type: Currently classified question type
            
        Returns:
            True if should override to document analysis
        """
        # Only override database analysis types
        if current_type not in [QuestionType.TEMPORAL_ANALYSIS, QuestionType.MERCHANT_ANALYSIS, 
                               QuestionType.GEOGRAPHIC_ANALYSIS, QuestionType.VALUE_ANALYSIS]:
            return False
        
        # Check for strong document analysis keywords
        document_keywords = [
            'methods by which', 'ways fraud is', 'fraud techniques',
            'primary methods', 'how fraud is committed', 'fraud approaches',
            'fraud strategies', 'fraud tactics', 'fraud schemes',
            'fraud methods', 'committed fraud', 'fraud committed',
            'core components', 'system components', 'detection system',
            'fraud detection', 'effective fraud', 'fraud prevention'
        ]
        
        return any(keyword in query_lower for keyword in document_keywords)
    
    def _get_document_analysis_type(self, query_lower: str) -> Tuple[QuestionType, float]:
        """
        Determine the appropriate document analysis type based on keywords
        
        Args:
            query_lower: Lowercase query
            
        Returns:
            Tuple of (question_type, confidence_score)
        """
        # Check for fraud methods keywords
        fraud_methods_keywords = [
            'methods by which', 'ways fraud is', 'fraud techniques',
            'primary methods', 'how fraud is committed', 'fraud approaches',
            'fraud strategies', 'fraud tactics', 'fraud schemes',
            'fraud methods', 'committed fraud', 'fraud committed'
        ]
        
        # Check for system components keywords
        system_components_keywords = [
            'core components', 'system components', 'detection system',
            'fraud detection', 'effective fraud', 'fraud prevention',
            'fraud detection system', 'components of', 'system architecture'
        ]
        
        if any(keyword in query_lower for keyword in fraud_methods_keywords):
            return QuestionType.FRAUD_METHODS, 0.7
        elif any(keyword in query_lower for keyword in system_components_keywords):
            return QuestionType.SYSTEM_COMPONENTS, 0.7
        else:
            # Default to fraud methods if document keywords found but unclear
            return QuestionType.FRAUD_METHODS, 0.6
    
    def _fallback_classification(self, query_lower: str) -> Tuple[QuestionType, float]:
        """
        Fallback classification when primary patterns don't match well
        
        Args:
            query_lower: Lowercase query
            
        Returns:
            Tuple of (question_type, confidence_score)
        """
        # Check data availability first for time-based questions
        if self.db_available and self.db_manager:
            time_reference = self._extract_time_reference(query_lower)
            if time_reference:
                data_available = self._check_data_availability(time_reference)
                if not data_available:
                    # If data is missing for the requested time period, route to forecasting
                    self.logger.info(f"Data not available for {time_reference}, routing to forecasting (fallback)")
                    return QuestionType.FORECASTING, 0.9
        
        # Simple keyword-based fallback - prioritize specific patterns over generic ones
        if any(word in query_lower for word in ['forecast', 'predict', 'future', 'next', 'will', 'expect']):
            return QuestionType.FORECASTING, 0.7
        elif any(word in query_lower for word in ['value', 'share', 'total', 'percentage', 'amount', 'h1', 'h2']):
            return QuestionType.VALUE_ANALYSIS, 0.6
        elif any(word in query_lower for word in ['method', 'methods', 'how', 'fraud', 'committed', 'primary']):
            return QuestionType.FRAUD_METHODS, 0.6
        elif any(word in query_lower for word in ['eea', 'geographic', 'location', 'counterpart', 'outside', 'cross-border']):
            return QuestionType.GEOGRAPHIC_ANALYSIS, 0.6
        elif any(word in query_lower for word in ['time', 'trend', 'daily', 'monthly']):
            return QuestionType.TEMPORAL_ANALYSIS, 0.5
        elif any(word in query_lower for word in ['merchant', 'category', 'which']):
            return QuestionType.MERCHANT_ANALYSIS, 0.5
        elif any(word in query_lower for word in ['system', 'component', 'detection']):
            return QuestionType.SYSTEM_COMPONENTS, 0.5
        else:
            # Default to temporal analysis
            return QuestionType.TEMPORAL_ANALYSIS, 0.3
    
    def get_handler_info(self, question_type: QuestionType, metadata: Dict = None) -> Dict[str, any]:
        """
        Get information about the handler for a question type
        
        Args:
            question_type: Type of question
            metadata: Optional metadata (for compatibility)
            
        Returns:
            Dictionary with handler information
        """
        handler_info = {
            QuestionType.TEMPORAL_ANALYSIS: {
                'handler': 'sql_generator',
                'requires_data': True,
                'chart_type': 'line_chart',
                'description': 'Time series analysis of fraud rates'
            },
            QuestionType.MERCHANT_ANALYSIS: {
                'handler': 'sql_generator',
                'requires_data': True,
                'chart_type': 'bar_chart',
                'description': 'Analysis of fraud rates by merchant/category'
            },
            QuestionType.FRAUD_METHODS: {
                'handler': 'document_processor',
                'requires_data': False,
                'chart_type': None,
                'description': 'Document-based analysis of fraud methods'
            },
            QuestionType.SYSTEM_COMPONENTS: {
                'handler': 'document_processor',
                'requires_data': False,
                'chart_type': None,
                'description': 'Document-based analysis of system components'
            },
            QuestionType.GEOGRAPHIC_ANALYSIS: {
                'handler': 'sql_generator',
                'requires_data': True,
                'chart_type': 'comparison_chart',
                'description': 'Geographic comparison of fraud rates'
            },
            QuestionType.VALUE_ANALYSIS: {
                'handler': 'sql_generator',
                'requires_data': True,
                'chart_type': 'pie_chart',
                'description': 'Analysis of fraud value distribution'
            },
            QuestionType.FORECASTING: {
                'handler': 'forecasting_agent',
                'requires_data': True,
                'chart_type': 'forecast_chart',
                'description': 'ARIMA-based fraud trend forecasting'
            }
        }
        
        return handler_info.get(question_type, {
            'handler': 'unknown',
            'requires_data': False,
            'chart_type': None,
            'description': 'Unknown question type'
        })
    
    def extract_parameters(self, query: str, question_type: QuestionType) -> Dict[str, any]:
        """
        Extract specific parameters from the query
        
        Args:
            query: User's query
            question_type: Type of question
            
        Returns:
            Dictionary of extracted parameters
        """
        parameters = {}
        query_lower = query.lower()
        
        if question_type == QuestionType.TEMPORAL_ANALYSIS:
            # Extract time period preference
            if 'daily' in query_lower:
                parameters['period'] = 'daily'
            elif 'monthly' in query_lower:
                parameters['period'] = 'monthly'
            else:
                parameters['period'] = 'monthly'  # default
        
        elif question_type == QuestionType.MERCHANT_ANALYSIS:
            # Extract analysis type preference
            if 'category' in query_lower or 'categories' in query_lower:
                parameters['analysis_type'] = 'category'
            else:
                parameters['analysis_type'] = 'merchant'  # default
        
        elif question_type == QuestionType.GEOGRAPHIC_ANALYSIS:
            # Extract geographic focus
            if 'eea' in query_lower:
                parameters['focus'] = 'eea'
            elif 'cross-border' in query_lower or 'cross border' in query_lower:
                parameters['focus'] = 'cross_border'
            else:
                parameters['focus'] = 'comparison'  # default
        
        elif question_type == QuestionType.VALUE_ANALYSIS:
            # Extract time period and value focus
            if 'h1 2023' in query_lower or 'first half 2023' in query_lower:
                parameters['time_period'] = 'h1_2023'
            else:
                parameters['time_period'] = 'h1_2023'  # default
            
            if 'cross-border' in query_lower or 'cross border' in query_lower:
                parameters['focus'] = 'cross_border'
            else:
                parameters['focus'] = 'total'  # default
        
        return parameters
    
    def validate_query(self, query: str) -> Tuple[bool, List[str]]:
        """
        Validate if the query is suitable for processing
        
        Args:
            query: User's query
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check if query is empty or too short
        if not query or len(query.strip()) < 3:
            errors.append("Query is too short. Please provide a more detailed question.")
        
        # Check if query contains fraud-related keywords
        fraud_keywords = ['fraud', 'fraudulent', 'fraud rate', 'fraud detection', 'credit card']
        if not any(keyword in query.lower() for keyword in fraud_keywords):
            errors.append("Query should be related to fraud analysis.")
        
        # Check for potentially problematic content
        problematic_patterns = [
            r'\b(drop|delete|insert|update|alter|create)\b',
            r'--',
            r'/\*',
            r'\*/'
        ]
        
        for pattern in problematic_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                errors.append("Query contains potentially harmful SQL keywords.")
                break
        
        return len(errors) == 0, errors
    
    def get_suggested_questions(self) -> List[Dict[str, str]]:
        """
        Get a list of suggested questions for each question type
        
        Returns:
            List of suggested questions with metadata
        """
        suggestions = [
            {
                'question': 'How does the daily fraud rate fluctuate over the two-year period?',
                'type': 'temporal_analysis',
                'description': 'Analyze fraud rate trends over time'
            },
            {
                'question': 'Which merchants exhibit the highest incidence of fraudulent transactions?',
                'type': 'merchant_analysis',
                'description': 'Find merchants with highest fraud rates'
            },
            {
                'question': 'What are the primary methods by which credit card fraud is committed?',
                'type': 'fraud_methods',
                'description': 'Learn about fraud techniques from documents'
            },
            {
                'question': 'What are the core components of an effective fraud detection system?',
                'type': 'system_components',
                'description': 'Understand fraud detection system architecture'
            },
            {
                'question': 'How much higher are fraud rates when the transaction counterpart is located outside the EEA?',
                'type': 'geographic_analysis',
                'description': 'Compare fraud rates by geographic region'
            },
            {
                'question': 'What share of total card fraud value in H1 2023 was due to cross-border transactions?',
                'type': 'value_analysis',
                'description': 'Analyze fraud value distribution by transaction type'
            }
        ]
        
        return suggestions
    
    
    def _extract_time_reference(self, query_lower: str) -> Optional[Dict[str, str]]:
        """
        Extract time references from the query
        
        Args:
            query_lower: Lowercase query string
            
        Returns:
            Dictionary with time reference info or None
        """
        import re
        from datetime import datetime
        
        # Patterns for different time references
        patterns = {
            'h1_2023': r'\bh1\s*2023\b',
            'h2_2023': r'\bh2\s*2023\b',
            'q1_2023': r'\bq1\s*2023\b',
            'q2_2023': r'\bq2\s*2023\b',
            'q3_2023': r'\bq3\s*2023\b',
            'q4_2023': r'\bq4\s*2023\b',
            'year_2023': r'\b2023\b',
            'year_2024': r'\b2024\b',
            'year_2025': r'\b2025\b',
            'next_month': r'\bnext\s+month\b',
            'next_quarter': r'\bnext\s+quarter\b',
            'next_year': r'\bnext\s+year\b',
            'last_month': r'\blast\s+month\b',
            'last_quarter': r'\blast\s+quarter\b',
            'last_year': r'\blast\s+year\b'
        }
        
        for pattern_name, pattern in patterns.items():
            if re.search(pattern, query_lower):
                return {
                    'type': pattern_name,
                    'pattern': pattern,
                    'query': query_lower
                }
        
        return None
    
    def _check_data_availability(self, time_reference: Dict[str, str]) -> bool:
        """
        Check if data is available for the given time reference
        
        Args:
            time_reference: Time reference dictionary from _extract_time_reference
            
        Returns:
            True if data is available, False otherwise
        """
        if not self.db_manager:
            return True  # Assume data is available if we can't check
        
        try:
            # Connect to database API if not already connected
            if not self.db_manager.connected:
                if not self.db_manager.connect():
                    self.logger.warning("Could not connect to database API for data availability check")
                    return True  # Assume available if we can't check
            
            # Define date ranges for different time references
            date_ranges = {
                'h1_2023': ("2023-01-01", "2023-07-01"),
                'h2_2023': ("2023-07-01", "2024-01-01"),
                'q1_2023': ("2023-01-01", "2023-04-01"),
                'q2_2023': ("2023-04-01", "2023-07-01"),
                'q3_2023': ("2023-07-01", "2023-10-01"),
                'q4_2023': ("2023-10-01", "2024-01-01"),
                'year_2023': ("2023-01-01", "2024-01-01"),
                'year_2024': ("2024-01-01", "2025-01-01"),
                'year_2025': ("2025-01-01", "2026-01-01"),
            }
            
            ref_type = time_reference['type']
            
            # For future references, always return False (no data available)
            if ref_type in ['next_month', 'next_quarter', 'next_year']:
                return False
            
            # For historical references, check if data exists
            if ref_type in date_ranges:
                start_date, end_date = date_ranges[ref_type]
                
                # Use API-based data availability check
                success, available, count, error = self.db_manager.check_data_availability(start_date, end_date)
                
                if success:
                    self.logger.info(f"Data availability check for {ref_type}: {count} records found")
                    return available
                else:
                    self.logger.warning(f"Data availability check failed for {ref_type}: {error}")
                    return False
            
            # For other references, assume data is available
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking data availability: {e}")
            return True  # Assume available if we can't check
