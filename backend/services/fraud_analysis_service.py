"""
Main Fraud Analysis Service
Orchestrates all components to answer fraud-related questions
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import logging

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.append(str(backend_path))

from core.query.sql_generator import QuestionType
from core.query.ai_sql_generator import AISQLGenerator
from core.document.hybrid_document_processor import HybridDocumentProcessor
from core.response.chart_generator import ChartGenerator
from core.response.response_generator import ResponseGenerator
from core.query.query_classifier import QueryClassifier
from data.api_database_manager import APIDatabaseManager
from agents.agent_coordinator import AgentCoordinator
from agents.forecasting_agent import ForecastingAgent
from core.evaluation.answer_evaluator import AnswerEvaluator


class FraudAnalysisService:
    """
    Main service that orchestrates fraud analysis components
    """
    
    def __init__(self, data_dir: str = "dataset"):
        self.data_dir = data_dir
        
        # Initialize components with LLM integration
        self.ai_sql_generator = AISQLGenerator()
        self.document_processor = HybridDocumentProcessor(data_dir, use_faiss=True)
        self.chart_generator = ChartGenerator()
        self.response_generator = ResponseGenerator()
        self.query_classifier = QueryClassifier(use_llm=True)
        self.database_manager = APIDatabaseManager()
        
        # Initialize AI agent coordinator
        try:
            self.agent_coordinator = AgentCoordinator(openai_api_key=os.getenv('OPENAI_API_KEY'))
            self.agents_available = True
        except Exception as e:
            # Logger not available yet, use print
            print(f"AI agents not available: {e}")
            self.agent_coordinator = None
            self.agents_available = False
        
        # Setup logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize forecasting agent
        try:
            self.forecasting_agent = ForecastingAgent(openai_api_key=os.getenv('OPENAI_API_KEY'))
            self.forecasting_available = True
            self.logger.info("Forecasting agent initialized successfully")
        except Exception as e:
            self.logger.warning(f"Forecasting agent not available: {e}")
            self.forecasting_agent = None
            self.forecasting_available = False
        
        # Initialize answer evaluator
        try:
            self.answer_evaluator = AnswerEvaluator(data_dir)
            self.evaluation_available = True
            self.logger.info("Answer evaluator initialized successfully")
        except Exception as e:
            self.logger.warning(f"Answer evaluator not available: {e}")
            self.answer_evaluator = None
            self.evaluation_available = False
        
        # Service state
        self.initialized = False
        self.document_index_built = False
    
    def initialize(self) -> bool:
        """
        Initialize the service and load all required data
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Initializing Fraud Analysis Service...")
            
            # Connect to database
            if not self.database_manager.connect():
                self.logger.error("Failed to connect to database")
                return False
            
            # Load CSV data
            if not self.database_manager.load_csv_data():
                self.logger.error("Failed to load CSV data")
                return False
            
            # Process documents
            if not self._process_documents():
                self.logger.warning("Document processing failed, continuing without document search")
            
            self.initialized = True
            self.logger.info("Fraud Analysis Service initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize service: {e}")
            return False
    
    def _process_documents(self) -> bool:
        """
        Process PDF documents using simplified OpenAI-based processor
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if document processor is available
            if not self.document_processor.is_available():
                self.logger.warning("Document processing not available. Continuing without document search.")
                return False
            
            # Process documents
            self.logger.info("Processing documents with OpenAI-based processor...")
            if self.document_processor.process_documents():
                self.document_index_built = True
                self.logger.info("Document processing completed successfully")
                return True
            else:
                self.logger.error("Failed to process documents")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing documents: {e}")
            return False
    
    async def process_question(self, question: str) -> Dict[str, Any]:
        """
        Process a user question and return a complete response
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing complete response
        """
        if not self.initialized:
            return self._format_response(
                answer="Service not initialized. Please try again later.",
                status='error',
                error='Service not initialized. Call initialize() first.'
            )
        
        try:
            # Validate query
            is_valid, errors = self.query_classifier.validate_query(question)
            if not is_valid:
                return self._format_response(
                    answer="I couldn't understand your question. Please try rephrasing it.",
                    status='error',
                    error='Invalid query: ' + '; '.join(errors)
                )
            
            # Classify question
            question_type, confidence, metadata = self.query_classifier.classify_query(question)
            self.logger.info(f"Question classified as: {question_type.value} (confidence: {confidence:.2f})")
            
            # Route to appropriate handler
            handler_info = self.query_classifier.get_handler_info(question_type, metadata)
            
            # Use AI agents if available, otherwise fall back to traditional handlers
            if question_type == QuestionType.FORECASTING and self.forecasting_available and self.forecasting_agent:
                self.logger.info(f"Routing forecasting question to forecasting agent")
                return self._handle_forecasting_question(question, question_type, confidence, metadata)
            elif self.agents_available and self.agent_coordinator:
                self.logger.info(f"Routing question to AI agents")
                return await self._handle_question_with_agents(question, question_type, confidence, metadata)
            elif handler_info['handler'] == 'sql_generator':
                self.logger.info(f"Routing question to AI SQL generator")
                return self._handle_ai_sql_question(question, question_type, confidence, metadata)
            elif handler_info['handler'] == 'document_processor':
                self.logger.info(f"Routing question to document processor")
                return self._handle_document_question(question, question_type, confidence, metadata)
            else:
                self.logger.error(f"Unknown handler: {handler_info['handler']} for question type: {question_type}")
                self.logger.error(f"Forecasting available: {self.forecasting_available}, Agent: {self.forecasting_agent is not None}")
                return self._format_response(
                    answer="I encountered an error processing your question.",
                    status='error',
                    error=f'Unknown handler: {handler_info["handler"]}'
                )
                
        except Exception as e:
            self.logger.error(f"Error processing question: {e}")
            return self._format_response(
                answer="I encountered an error while processing your question.",
                status='error',
                error=f'Error processing question: {str(e)}'
            )
    
    async def _handle_question_with_agents(self, question: str, question_type: QuestionType,
                                         confidence: float, metadata: Dict) -> Dict[str, Any]:
        """
        Handle questions using AI agent systems (PydanticAI + CrewAI)
        
        Args:
            question: User's question
            question_type: Type of question
            confidence: Classification confidence
            metadata: Classification metadata
            
        Returns:
            Response dictionary
        """
        try:
            # Prepare context data
            context_data = {}
            
            # Get SQL data if it's a database question
            if question_type in [QuestionType.MERCHANT_ANALYSIS, QuestionType.TEMPORAL_ANALYSIS, 
                               QuestionType.GEOGRAPHIC_ANALYSIS, QuestionType.VALUE_ANALYSIS]:
                try:
                    # Generate and execute SQL query using AI SQL generator
                    sql_result = self.ai_sql_generator.generate_sql(question)
                    if sql_result.get('success', False):
                        success, data, error = self.database_manager.execute_query(sql_result['sql'])
                        if success and data is not None:
                            context_data['sql_data'] = data
                            context_data['sql_query'] = sql_result['sql']
                except Exception as e:
                    self.logger.warning(f"Failed to get SQL data: {e}")
            
            # Get document data if it's a document question
            if question_type in [QuestionType.FRAUD_METHODS, QuestionType.SYSTEM_COMPONENTS]:
                try:
                    if self.document_index_built:
                        search_result = self.document_processor.search_documents(question, max_results=5)
                        if search_result.get('success', False):
                            context_data['document_chunks'] = search_result.get('sources', [])
                except Exception as e:
                    self.logger.warning(f"Failed to get document data: {e}")
            
            # Use agent coordinator to analyze the question
            agent_result = self.agent_coordinator.analyze_question(
                question, question_type.value, context_data
            )
            
            if not agent_result.get('success', False):
                # Fallback to AI SQL generator for database questions
                self.logger.warning("Agent analysis failed, falling back to AI SQL generator")
                if question_type in [QuestionType.MERCHANT_ANALYSIS, QuestionType.TEMPORAL_ANALYSIS, 
                                   QuestionType.GEOGRAPHIC_ANALYSIS, QuestionType.VALUE_ANALYSIS]:
                    return self._handle_ai_sql_question(question, question_type, confidence, metadata)
                else:
                    return self._handle_document_question(question, question_type, confidence, metadata)
            
            # Generate chart if we have data
            chart = None
            if 'sql_data' in context_data and context_data['sql_data'] is not None:
                try:
                    chart = self._generate_chart(question_type, context_data['sql_data'], 
                                               context_data.get('sql_query', {}))
                except Exception as e:
                    self.logger.warning(f"Failed to generate chart: {e}")
            
            # Format response
            response = {
                'answer': agent_result.get('answer', 'No answer generated'),
                'confidence': agent_result.get('confidence', confidence),
                'question_type': question_type.value,
                'metadata': metadata,
                'agent_method': agent_result.get('method', 'Unknown'),
                'agent_used': agent_result.get('agent_used', 'Unknown'),
                'status': 'success',
                'handler': 'sql_generator' if question_type in [QuestionType.MERCHANT_ANALYSIS, QuestionType.TEMPORAL_ANALYSIS, 
                                                               QuestionType.GEOGRAPHIC_ANALYSIS, QuestionType.VALUE_ANALYSIS] else 'document_processor',
                'chart_type': self.query_classifier.get_handler_info(question_type, metadata).get('chart_type')
            }
            
            # Add additional agent insights
            if 'key_insights' in agent_result:
                response['key_insights'] = agent_result['key_insights']
            if 'recommendations' in agent_result:
                response['recommendations'] = agent_result['recommendations']
            if 'risk_level' in agent_result:
                response['risk_level'] = agent_result['risk_level']
            if 'supporting_evidence' in agent_result:
                response['supporting_evidence'] = agent_result['supporting_evidence']
            
            # Add chart if available
            if chart:
                response['chart'] = chart
            
            # Add SQL query if available
            if 'sql_query' in context_data:
                response['sql_query'] = context_data['sql_query']
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in agent-based question handling: {e}")
            # Fallback to AI SQL generator for database questions
            if question_type in [QuestionType.MERCHANT_ANALYSIS, QuestionType.TEMPORAL_ANALYSIS, 
                               QuestionType.GEOGRAPHIC_ANALYSIS, QuestionType.VALUE_ANALYSIS]:
                return self._handle_ai_sql_question(question, question_type, confidence, metadata)
            else:
                return self._handle_document_question(question, question_type, confidence, metadata)
    
    def process_question_sync(self, question: str) -> Dict[str, Any]:
        """
        Synchronous version of process_question for fallback
        """
        if not self.initialized:
            return self._format_response(
                answer="Service not initialized. Please try again later.",
                status='error',
                error='Service not initialized. Call initialize() first.'
            )
        
        try:
            # Validate query
            is_valid, errors = self.query_classifier.validate_query(question)
            if not is_valid:
                return self._format_response(
                    answer="I couldn't understand your question. Please try rephrasing it.",
                    status='error',
                    error='Invalid query: ' + '; '.join(errors)
                )
            
            # Classify question
            question_type, confidence, metadata = self.query_classifier.classify_query(question)
            self.logger.info(f"Question classified as: {question_type.value} (confidence: {confidence:.2f})")
            
            # Route to appropriate handler (sync version)
            handler_info = self.query_classifier.get_handler_info(question_type, metadata)
            
            # Use forecasting agent if available for forecasting questions
            if question_type == QuestionType.FORECASTING and self.forecasting_available and self.forecasting_agent:
                self.logger.info(f"Routing forecasting question to forecasting agent")
                return self._handle_forecasting_question(question, question_type, confidence, metadata)
            elif handler_info['handler'] == 'sql_generator':
                # Use AI SQL generator for all database questions
                self.logger.info(f"Routing question to AI SQL generator")
                return self._handle_ai_sql_question(question, question_type, confidence, metadata)
            elif handler_info['handler'] == 'document_processor':
                self.logger.info(f"Routing question to document processor")
                return self._handle_document_question(question, question_type, confidence, metadata)
            else:
                return self._format_response(
                    answer="I encountered an error processing your question.",
                    status='error',
                    error=f'Unknown handler: {handler_info["handler"]}'
                )
                
        except Exception as e:
            self.logger.error(f"Error processing question: {e}")
            return self._format_response(
                answer="I encountered an error while processing your question.",
                status='error',
                error=f'Error processing question: {str(e)}'
            )
    
    
    def _should_fallback_to_documents(self, sql_result: Dict, question: str, question_type: QuestionType) -> bool:
        """
        Determine if a failed SQL query should fallback to document processing
        
        Args:
            sql_result: Result from AI SQL generator
            question: User's question
            question_type: Type of question
            
        Returns:
            True if should fallback to documents, False otherwise
        """
        # Check for missing data scenarios that might be in documents
        if "error" in sql_result:
            error = sql_result["error"].lower()
            
            # EEA/European data questions
            if any(term in error for term in ["eea", "europe", "european", "international", "cross-border"]):
                return True
            
            # 2023/2024 data questions (recent reports)
            if any(term in error for term in ["2023", "2024", "recent"]):
                return True
        
        # Regulatory/industry questions that are better answered by documents
        if question_type in [QuestionType.SYSTEM_COMPONENTS, QuestionType.FRAUD_METHODS]:
            return True
        
        # Check question content for document-relevant terms
        question_lower = question.lower()
        document_terms = [
            "eea", "european", "europe", "regulatory", "report", "study", "analysis",
            "industry", "standards", "guidelines", "recommendations", "best practices",
            "2023", "2024", "recent", "latest", "current"
        ]
        
        if any(term in question_lower for term in document_terms):
            return True
        
        return False

    def _handle_ai_sql_question(self, question: str, question_type: QuestionType, 
                               confidence: float, metadata: Dict) -> Dict[str, Any]:
        """
        Handle questions using the AI SQL generator with smart document fallback
        
        Args:
            question: User's question
            question_type: Type of question
            confidence: Classification confidence
            metadata: Classification metadata
            
        Returns:
            Response dictionary
        """
        try:
            print(f"\n=== AI SQL GENERATOR DEBUG ===")
            print(f"Question: {question}")
            print(f"Question Type: {question_type}")
            print(f"Confidence: {confidence}")
            print(f"Metadata: {metadata}")
            
            # Generate SQL using AI SQL generator
            sql_result = self.ai_sql_generator.generate_sql(question)
            print(f"AI SQL Result: {sql_result}")
            
            if not sql_result.get("success", False):
                # Check if this should fallback to document processing
                if self._should_fallback_to_documents(sql_result, question, question_type):
                    self.logger.info("AI SQL found no data, falling back to document processing")
                    print("=== FALLING BACK TO HYBRID DOCUMENT PROCESSING ===")
                    print(f"Reason: {sql_result.get('error', 'No data available')}")
                    print(f"Question type: {question_type.value}")
                    print(f"Question: {question}")
                    return self._handle_document_question(question, question_type, confidence, metadata)
                else:
                    # Handle other SQL generation failures
                    return {
                        'answer': f"I understand you're asking about {question}, but the current dataset only contains data from 2019-2020. {sql_result.get('suggestion', '')}",
                        'status': 'partial_success',
                        'handler': 'ai_sql_generator',
                        'error': sql_result.get('error'),
                        'suggestion': sql_result.get('suggestion'),
                        'fallback_question': sql_result.get('fallback_question'),
                        'available_data': sql_result.get('available_data')
                    }
            
            # Execute the generated SQL
            success, data, error = self.database_manager.execute_query(sql_result['sql'])
            if not success:
                print(f"Database query failed: {error}")
                return {
                    'error': f'Database query failed: {error}',
                    'status': 'error',
                    'handler': 'ai_sql_generator'
                }
            
            print(f"Database query successful, data shape: {data.shape if data is not None else 'None'}")
            print("=== END AI SQL GENERATOR DEBUG ===\n")
            
            # Generate chart
            chart = None
            if data is not None and not data.empty:
                chart = self._generate_ai_chart(question_type, data, sql_result)
            
            # Generate response using AI SQL generator insights
            response = self._generate_ai_response(question, question_type, data, sql_result, confidence, metadata)
            
            # Add chart and metadata
            response['chart'] = chart
            response['sql_query'] = sql_result['sql']
            response['confidence'] = confidence
            response['metadata'] = metadata
            response['handler'] = 'ai_sql_generator'
            response['ai_generated'] = True
            response['description'] = sql_result.get('description', 'AI-generated analysis')
            
            if 'adaptation_note' in sql_result:
                response['adaptation_note'] = sql_result['adaptation_note']
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error handling AI SQL question: {e}")
            return {
                'error': f'Error handling AI SQL question: {str(e)}',
                'status': 'error',
                'handler': 'ai_sql_generator'
            }
    
    def _generate_ai_chart(self, question_type: QuestionType, data: pd.DataFrame, 
                          sql_result: Dict) -> Optional[Dict[str, Any]]:
        """
        Generate appropriate chart for AI SQL generator results
        
        Args:
            question_type: Type of question
            data: DataFrame with data
            sql_result: AI SQL generation result
            
        Returns:
            Chart configuration dictionary
        """
        try:
            if data is None or data.empty:
                return None
            
            # Determine chart type based on data structure
            if 'amount_range' in data.columns:
                return {
                    'type': 'bar_chart',
                    'data': data.to_dict('records'),
                    'x_col': 'amount_range',
                    'y_col': 'fraud_percentage',
                    'title': 'Fraud Rate by Transaction Amount Range',
                    'orientation': 'vertical'
                }
            elif 'state' in data.columns:
                return {
                    'type': 'bar_chart',
                    'data': data.to_dict('records'),
                    'x_col': 'state',
                    'y_col': 'fraud_percentage',
                    'title': 'Fraud Rate by US State',
                    'orientation': 'horizontal'
                }
            elif 'time_period' in data.columns:
                if 'period_type' in data.columns:
                    # Combined daily/monthly analysis
                    return {
                        'type': 'multi_line_chart',
                        'data': data.to_dict('records'),
                        'x_col': 'time_period',
                        'y_col': 'fraud_percentage',
                        'group_col': 'period_type',
                        'title': 'Fraud Rate Over Time (Daily vs Monthly)',
                        'description': 'Shows fraud rate trends at both daily and monthly granularity'
                    }
                else:
                    # Single period type
                    return {
                        'type': 'line_chart',
                        'data': data.to_dict('records'),
                        'x_col': 'time_period',
                        'y_col': 'fraud_percentage',
                        'title': 'Fraud Rate Over Time'
                    }
            elif 'merchant' in data.columns:
                return {
                    'type': 'bar_chart',
                    'data': data.to_dict('records'),
                    'x_col': 'merchant',
                    'y_col': 'fraud_percentage',
                    'title': 'Fraud Rate by Merchant',
                    'orientation': 'horizontal'
                }
            elif 'region_type' in data.columns:
                return {
                    'type': 'comparison_chart',
                    'data': data.to_dict('records'),
                    'x_col': 'region_type',
                    'y_col': 'fraud_percentage',
                    'title': 'Fraud Rate by Region Type'
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating AI chart: {e}")
            return None
    
    def _generate_ai_response(self, question: str, question_type: QuestionType, 
                             data: pd.DataFrame, sql_result: Dict, 
                             confidence: float = 0.8, metadata: Dict = None) -> Dict[str, Any]:
        """
        Generate response for AI SQL generator results
        
        Args:
            question: User's question
            question_type: Type of question
            data: DataFrame with data
            sql_result: AI SQL generation result
            
        Returns:
            Response dictionary
        """
        try:
            if metadata is None:
                metadata = {}
                
            if data is None or data.empty:
                return {
                    'answer': 'No data available for this analysis.',
                    'status': 'partial_success',
                    'insights': []
                }
            
            # Generate insights based on data
            insights = []
            
            if 'amount_range' in data.columns:
                # Value analysis
                high_fraud_range = data.loc[data['fraud_percentage'].idxmax()]
                insights.append(f"Highest fraud rate: {high_fraud_range['amount_range']} transactions at {high_fraud_range['fraud_percentage']:.2f}%")
                
                if len(data) > 1:
                    fraud_rate_diff = data['fraud_percentage'].max() - data['fraud_percentage'].min()
                    insights.append(f"Fraud rate varies by {fraud_rate_diff:.2f} percentage points across amount ranges")
            
            elif 'state' in data.columns:
                # Geographic analysis
                top_state = data.iloc[0]
                insights.append(f"Highest fraud state: {top_state['state']} with {top_state['fraud_percentage']:.2f}% fraud rate")
                
                if len(data) > 1:
                    state_variance = data['fraud_percentage'].std()
                    insights.append(f"Fraud rates vary significantly across states (std dev: {state_variance:.2f}%)")
            
            elif 'time_period' in data.columns:
                # Enhanced temporal analysis
                if 'period_type' in data.columns:
                    # Combined daily/monthly analysis
                    daily_data = data[data['period_type'] == 'daily']
                    monthly_data = data[data['period_type'] == 'monthly']
                    
                    if not daily_data.empty:
                        daily_volatility = daily_data['fraud_percentage'].std()
                        daily_mean = daily_data['fraud_percentage'].mean()
                        daily_median = daily_data['fraud_percentage'].median()
                        
                        insights.append(f"Daily fraud rate volatility: {daily_volatility:.2f}% (standard deviation)")
                        insights.append(f"Average daily fraud rate: {daily_mean:.2f}%")
                        insights.append(f"Median daily fraud rate: {daily_median:.2f}%")
                        
                        max_daily = daily_data.loc[daily_data['fraud_percentage'].idxmax()]
                        min_daily = daily_data.loc[daily_data['fraud_percentage'].idxmin()]
                        insights.append(f"Highest daily fraud rate: {max_daily['time_period']} with {max_daily['fraud_percentage']:.2f}%")
                        insights.append(f"Lowest daily fraud rate: {min_daily['time_period']} with {min_daily['fraud_percentage']:.2f}%")
                        
                        # Calculate trend over time
                        if len(daily_data) > 1:
                            first_half = daily_data.iloc[:len(daily_data)//2]['fraud_percentage'].mean()
                            second_half = daily_data.iloc[len(daily_data)//2:]['fraud_percentage'].mean()
                            trend_direction = "increasing" if second_half > first_half else "decreasing"
                            trend_magnitude = abs(second_half - first_half)
                            insights.append(f"Daily fraud rate shows {trend_direction} trend ({trend_magnitude:.2f}% change from first to second half)")
                    
                    if not monthly_data.empty:
                        monthly_trend = "increasing" if monthly_data['fraud_percentage'].iloc[-1] > monthly_data['fraud_percentage'].iloc[0] else "decreasing"
                        monthly_volatility = monthly_data['fraud_percentage'].std()
                        monthly_mean = monthly_data['fraud_percentage'].mean()
                        
                        insights.append(f"Monthly fraud rate shows {monthly_trend} trend over time")
                        insights.append(f"Monthly fraud rate volatility: {monthly_volatility:.2f}%")
                        insights.append(f"Average monthly fraud rate: {monthly_mean:.2f}%")
                        
                        max_monthly = monthly_data.loc[monthly_data['fraud_percentage'].idxmax()]
                        min_monthly = monthly_data.loc[monthly_data['fraud_percentage'].idxmin()]
                        insights.append(f"Peak monthly fraud period: {max_monthly['time_period']} with {max_monthly['fraud_percentage']:.2f}% fraud rate")
                        insights.append(f"Lowest monthly fraud period: {min_monthly['time_period']} with {min_monthly['fraud_percentage']:.2f}% fraud rate")
                        
                        # Compare daily vs monthly patterns
                        if not daily_data.empty:
                            daily_vol = daily_data['fraud_percentage'].std()
                            monthly_vol = monthly_data['fraud_percentage'].std()
                            if daily_vol > monthly_vol:
                                insights.append(f"Daily patterns show higher volatility ({daily_vol:.2f}%) than monthly patterns ({monthly_vol:.2f}%)")
                            else:
                                insights.append(f"Monthly patterns show higher volatility ({monthly_vol:.2f}%) than daily patterns ({daily_vol:.2f}%)")
                else:
                    # Single period type
                    if len(data) > 1:
                        trend = "increasing" if data['fraud_percentage'].iloc[-1] > data['fraud_percentage'].iloc[0] else "decreasing"
                        volatility = data['fraud_percentage'].std()
                        mean_rate = data['fraud_percentage'].mean()
                        median_rate = data['fraud_percentage'].median()
                        
                        insights.append(f"Fraud rate shows {trend} trend over time")
                        insights.append(f"Fraud rate volatility: {volatility:.2f}% (standard deviation)")
                        insights.append(f"Average fraud rate: {mean_rate:.2f}%")
                        insights.append(f"Median fraud rate: {median_rate:.2f}%")
                        
                        max_period = data.loc[data['fraud_percentage'].idxmax()]
                        min_period = data.loc[data['fraud_percentage'].idxmin()]
                        insights.append(f"Peak fraud period: {max_period['time_period']} with {max_period['fraud_percentage']:.2f}% fraud rate")
                        insights.append(f"Lowest fraud period: {min_period['time_period']} with {min_period['fraud_percentage']:.2f}% fraud rate")
            
            elif 'merchant' in data.columns:
                # Enhanced merchant analysis
                top_merchant = data.iloc[0]
                bottom_merchant = data.iloc[-1]
                total_merchants = len(data)
                avg_fraud_rate = data['fraud_percentage'].mean()
                median_fraud_rate = data['fraud_percentage'].median()
                fraud_rate_std = data['fraud_percentage'].std()
                
                insights.append(f"Highest risk merchant: {top_merchant['merchant']} with {top_merchant['fraud_percentage']:.2f}% fraud rate")
                insights.append(f"Lowest risk merchant: {bottom_merchant['merchant']} with {bottom_merchant['fraud_percentage']:.2f}% fraud rate")
                insights.append(f"Average fraud rate across {total_merchants} merchants: {avg_fraud_rate:.2f}%")
                insights.append(f"Median fraud rate: {median_fraud_rate:.2f}%")
                insights.append(f"Fraud rate standard deviation: {fraud_rate_std:.2f}%")
                
                # Risk categorization
                high_risk_threshold = avg_fraud_rate + fraud_rate_std
                low_risk_threshold = avg_fraud_rate - fraud_rate_std
                high_risk_merchants = data[data['fraud_percentage'] > high_risk_threshold]
                low_risk_merchants = data[data['fraud_percentage'] < low_risk_threshold]
                
                insights.append(f"High-risk merchants (>1 std dev above average): {len(high_risk_merchants)} merchants")
                insights.append(f"Low-risk merchants (<1 std dev below average): {len(low_risk_merchants)} merchants")
                
                # Top 5 analysis
                top_5 = data.head(5)
                insights.append(f"Top 5 highest risk merchants:")
                for i, (_, merchant) in enumerate(top_5.iterrows(), 1):
                    insights.append(f"  {i}. {merchant['merchant']}: {merchant['fraud_percentage']:.2f}%")
                
                # Risk distribution analysis
                if len(data) >= 10:
                    quartiles = data['fraud_percentage'].quantile([0.25, 0.5, 0.75])
                    insights.append(f"Fraud rate quartiles: Q1={quartiles[0.25]:.2f}%, Q2={quartiles[0.5]:.2f}%, Q3={quartiles[0.75]:.2f}%")
                    
                    # Calculate interquartile range
                    iqr = quartiles[0.75] - quartiles[0.25]
                    insights.append(f"Interquartile range: {iqr:.2f}% (shows spread of middle 50% of merchants)")
            
            # Generate comprehensive summary answer
            if 'region_type' in data.columns and len(data) == 2:
                # Cross-border analysis
                high_value = data[data['region_type'].str.contains('High-Value', case=False, na=False)]
                low_value = data[data['region_type'].str.contains('Low-Value', case=False, na=False)]
                
                if not high_value.empty and not low_value.empty:
                    high_rate = high_value.iloc[0]['fraud_percentage']
                    low_rate = low_value.iloc[0]['fraud_percentage']
                    difference = ((high_rate - low_rate) / low_rate) * 100 if low_rate > 0 else 0
                    
                    answer = f"Geographic analysis shows that high-value transactions (cross-border proxy) have a {difference:.1f}% higher fraud rate than low-value transactions (domestic proxy). High-value transactions: {high_rate:.2f}% fraud rate, Low-value transactions: {low_rate:.2f}% fraud rate."
                else:
                    answer = f"Geographic analysis completed. {sql_result.get('description', 'Analysis results available.')}"
            elif 'time_period' in data.columns:
                # Enhanced temporal analysis summary
                if 'period_type' in data.columns:
                    daily_data = data[data['period_type'] == 'daily']
                    monthly_data = data[data['period_type'] == 'monthly']
                    
                    if not daily_data.empty and not monthly_data.empty:
                        daily_avg = daily_data['fraud_percentage'].mean()
                        monthly_avg = monthly_data['fraud_percentage'].mean()
                        daily_vol = daily_data['fraud_percentage'].std()
                        monthly_vol = monthly_data['fraud_percentage'].std()
                        
                        answer = f"Temporal analysis reveals fraud rate patterns across both daily and monthly periods. Daily fraud rates average {daily_avg:.2f}% with {daily_vol:.2f}% volatility, while monthly rates average {monthly_avg:.2f}% with {monthly_vol:.2f}% volatility. "
                        
                        if daily_vol > monthly_vol:
                            answer += f"Daily patterns show higher volatility ({daily_vol:.2f}%) than monthly patterns ({monthly_vol:.2f}%), indicating more day-to-day variation in fraud rates."
                        else:
                            answer += f"Monthly patterns show higher volatility ({monthly_vol:.2f}%) than daily patterns ({daily_vol:.2f}%), suggesting more month-to-month variation in fraud rates."
                    else:
                        period_type = "daily" if not daily_data.empty else "monthly"
                        period_data = daily_data if not daily_data.empty else monthly_data
                        avg_rate = period_data['fraud_percentage'].mean()
                        volatility = period_data['fraud_percentage'].std()
                        trend = "increasing" if period_data['fraud_percentage'].iloc[-1] > period_data['fraud_percentage'].iloc[0] else "decreasing"
                        
                        answer = f"Temporal analysis shows {period_type} fraud rates averaging {avg_rate:.2f}% with {volatility:.2f}% volatility. The overall trend is {trend} over the analyzed period."
                else:
                    avg_rate = data['fraud_percentage'].mean()
                    volatility = data['fraud_percentage'].std()
                    trend = "increasing" if data['fraud_percentage'].iloc[-1] > data['fraud_percentage'].iloc[0] else "decreasing"
                    
                    answer = f"Temporal analysis shows fraud rates averaging {avg_rate:.2f}% with {volatility:.2f}% volatility. The overall trend is {trend} over the analyzed period."
            elif 'merchant' in data.columns:
                # Enhanced merchant analysis summary
                top_merchant = data.iloc[0]
                avg_rate = data['fraud_percentage'].mean()
                total_merchants = len(data)
                high_risk_count = len(data[data['fraud_percentage'] > avg_rate + data['fraud_percentage'].std()])
                
                answer = f"Merchant analysis across {total_merchants} merchants reveals significant variation in fraud rates. The highest risk merchant is {top_merchant['merchant']} with {top_merchant['fraud_percentage']:.2f}% fraud rate, while the average across all merchants is {avg_rate:.2f}%. "
                
                if high_risk_count > 0:
                    answer += f"{high_risk_count} merchants show high-risk patterns (above 1 standard deviation from average), indicating concentrated fraud risk among specific merchants."
                else:
                    answer += "Fraud rates are relatively evenly distributed across merchants, with no extreme outliers identified."
            else:
                answer = f"Analysis completed successfully. {sql_result.get('description', 'Results show fraud patterns across different categories.')}"
            
            return self._format_response(
                answer=answer,
                status='success',
                data=data,
                handler='ai_sql_generator',
                question_type=question_type.value,
                confidence=confidence,
                metadata=metadata,
                insights=insights
            )
            
        except Exception as e:
            self.logger.error(f"Error generating AI response: {e}")
            return self._format_response(
                answer=f"Analysis completed but encountered an error: {str(e)}",
                status='partial_success',
                error=str(e)
            )

    
    def _handle_document_question(self, question: str, question_type: QuestionType,
                                 confidence: float, metadata: Dict) -> Dict[str, Any]:
        """
        Handle document-related questions using hybrid FAISS + OpenAI approach
        
        Args:
            question: User's question
            question_type: Type of question
            confidence: Classification confidence
            metadata: Classification metadata
            
        Returns:
            Response dictionary
        """
        try:
            print(f"\n=== HYBRID DOCUMENT PROCESSING DEBUG ===")
            print(f"Question: {question}")
            print(f"Question Type: {question_type.value}")
            print(f"Confidence: {confidence}")
            
            if not self.document_index_built:
                print("❌ Document index not built, processing documents...")
                if not self._process_documents():
                    return {
                        'error': 'Document search not available. Documents not processed.',
                        'status': 'error'
                    }
                print("✅ Document index built successfully")
            
            # Check hybrid processor status
            doc_summary = self.document_processor.get_document_summary()
            print(f"📊 Document Summary:")
            print(f"  - Documents loaded: {doc_summary['documents_loaded']}")
            print(f"  - Document names: {doc_summary['document_names']}")
            print(f"  - FAISS available: {doc_summary['faiss_available']}")
            print(f"  - Chunks created: {doc_summary['chunks_created']}")
            print(f"  - Processing available: {doc_summary['processing_available']}")
            
            # Search documents using hybrid approach
            max_results = 10 if any(term in question.lower() for term in ["eea", "european", "europe"]) else 5
            print(f"🔍 Searching with max_results: {max_results}")
            print(f"🔄 Using hybrid approach: FAISS + OpenAI")
            
            search_result = self.document_processor.search_documents(question, max_results=max_results, use_hybrid=True)
            
            # Log search result details
            print(f"📋 Search Result:")
            print(f"  - Success: {search_result.get('success', False)}")
            print(f"  - Method used: {search_result.get('method', 'unknown')}")
            print(f"  - Confidence: {search_result.get('confidence', 'N/A')}")
            print(f"  - Cost saved: {search_result.get('cost_saved', 'N/A')}")
            print(f"  - Sources found: {len(search_result.get('sources', []))}")
            
            if search_result.get('method') == 'hybrid':
                print(f"  - FAISS sources: {search_result.get('faiss_sources', 'N/A')}")
                print(f"  - Enhanced with OpenAI: {search_result.get('enhanced_with_openai', 'N/A')}")
            
            # Show cost tracking
            cost_stats = self.document_processor.get_cost_stats()
            print(f"💰 Cost Statistics:")
            print(f"  - Total queries: {cost_stats['total_queries']}")
            print(f"  - FAISS queries: {cost_stats['faiss_queries']}")
            print(f"  - OpenAI queries: {cost_stats['openai_queries']}")
            print(f"  - Hybrid queries: {cost_stats['hybrid_queries']}")
            print(f"  - Cost savings: {cost_stats['cost_savings_percentage']:.1f}%")
            print(f"  - Tokens saved: {cost_stats['estimated_tokens_saved']}")
            
            if not search_result.get('success', False):
                print("❌ Document search failed")
                return {
                    'error': f"Document search failed: {search_result.get('error', 'Unknown error')}",
                    'status': 'error'
                }
            
            # Extract answer and sources from search result
            answer = search_result.get('answer', 'No answer found')
            sources = search_result.get('sources', [])
            
            print(f"📝 Generated Answer:")
            print(f"  - Length: {len(answer)} characters")
            print(f"  - Sources: {len(sources)}")
            if sources:
                print(f"  - Source documents: {[s.get('document', 'Unknown') for s in sources[:3]]}")
            
            print("=== END HYBRID DOCUMENT PROCESSING DEBUG ===\n")
            search_confidence = search_result.get('confidence', 0.8)
            
            # Enhance answer for EEA questions
            if any(term in question.lower() for term in ["eea", "european", "europe"]):
                answer = self._enhance_eea_response(question, answer, sources)
            
            # Generate chart if relevant
            chart = None
            if any(term in question.lower() for term in ["fraud rate", "percentage", "higher", "lower"]):
                chart = self._generate_document_chart(question, sources)
            
            # Return the enhanced document processor response
            return self._format_response(
                answer=answer,
                status='success',
                handler='document_processor',
                question_type=question_type.value,
                confidence=min(confidence, search_confidence),
                metadata=metadata,
                sources=sources,
                chart=chart
            )
            
        except Exception as e:
            self.logger.error(f"Error handling document question: {e}")
            return self._format_response(
                answer="I encountered an error while processing your question.",
                status='error',
                error=f'Error handling document question: {str(e)}'
            )
    
    def _format_response(self, answer: str, status: str, data: Any = None, 
                        chart: Dict = None, handler: str = None, question_type: str = None,
                        confidence: float = None, metadata: Dict = None, 
                        insights: List = None, sources: List = None, 
                        error: str = None) -> Dict[str, Any]:
        """
        Format API response in a classic JSON structure
        
        Args:
            answer: The main answer text
            status: Response status (success, error, partial_success)
            data: Any data returned
            chart: Chart configuration if applicable
            handler: Which handler processed the request
            question_type: Type of question asked
            confidence: Confidence score
            metadata: Additional metadata
            insights: List of insights
            sources: Document sources
            error: Error message if any
            
        Returns:
            Formatted response dictionary
        """
        response = {
            "success": status == "success",
            "status": status,
            "data": {
                "answer": answer,
                "question_type": question_type,
                "confidence": confidence
            }
        }
        
        # Add optional fields if present
        if data is not None:
            # Convert DataFrame to serializable format
            if hasattr(data, 'to_dict'):
                # It's a pandas DataFrame
                response["data"]["analysis_data"] = data.to_dict('records')
            else:
                # It's already serializable
                response["data"]["analysis_data"] = data
            
        if chart is not None:
            response["data"]["chart"] = chart
            
        if insights:
            response["data"]["insights"] = insights
            
        if sources:
            response["data"]["sources"] = sources
            
        if handler:
            response["data"]["handler"] = handler
            
        if metadata:
            response["data"]["metadata"] = metadata
            
        if error:
            response["data"]["error"] = error
            
        return response

    def _enhance_eea_response(self, question: str, answer: str, sources: List[Dict]) -> str:
        """
        Enhance response for EEA-specific questions
        
        Args:
            question: User's question
            answer: Base answer from document search
            sources: Document sources
            
        Returns:
            Enhanced answer with EEA-specific insights
        """
        try:
            # Look for EEA-specific data in sources
            eea_data = []
            for source in sources:
                content = source.get('content', '').lower()
                if any(term in content for term in ['eea', 'european economic area', 'european', 'cross-border']):
                    eea_data.append(source)
            
            if eea_data:
                # Extract specific EEA fraud statistics
                enhanced_answer = answer
                
                # Look for specific fraud rate comparisons
                for source in eea_data:
                    content = source.get('content', '')
                    
                    # Look for percentage comparisons
                    import re
                    percentage_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', content)
                    if percentage_matches:
                        enhanced_answer += f"\n\n📊 EEA Fraud Statistics Found:"
                        enhanced_answer += f"\n- Document contains fraud rate data: {', '.join(percentage_matches[:3])}%"
                    
                    # Look for specific EEA vs non-EEA comparisons
                    if 'eea' in content.lower() and ('non-eea' in content.lower() or 'outside' in content.lower()):
                        enhanced_answer += f"\n- Contains EEA vs non-EEA fraud rate comparisons"
                
                # Add source information
                enhanced_answer += f"\n\n📄 Source: {eea_data[0].get('source', 'EBA/ECB Report')}"
                
                return enhanced_answer
            else:
                return answer
                
        except Exception as e:
            self.logger.warning(f"Error enhancing EEA response: {e}")
            return answer
    
    def _generate_document_chart(self, question: str, sources: List[Dict]) -> Optional[Dict[str, Any]]:
        """
        Generate chart from document data for fraud rate questions
        
        Args:
            question: User's question
            sources: Document sources
            
        Returns:
            Chart configuration dictionary
        """
        try:
            # Look for numerical data in sources
            chart_data = []
            
            for source in sources:
                content = source.get('content', '')
                
                # Extract percentage data
                import re
                percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', content)
                
                if percentages:
                    # Try to identify what each percentage represents
                    if 'eea' in content.lower():
                        for i, pct in enumerate(percentages[:2]):  # Take first 2 percentages
                            chart_data.append({
                                'category': 'EEA' if i == 0 else 'Non-EEA',
                                'fraud_rate': float(pct),
                                'source': source.get('source', 'Document')
                            })
                    else:
                        for i, pct in enumerate(percentages[:3]):  # Take first 3 percentages
                            chart_data.append({
                                'category': f'Category {i+1}',
                                'fraud_rate': float(pct),
                                'source': source.get('source', 'Document')
                            })
            
            if chart_data:
                return {
                    'type': 'bar_chart',
                    'data': chart_data,
                    'x_col': 'category',
                    'y_col': 'fraud_rate',
                    'title': 'Fraud Rates from Document Analysis',
                    'description': 'Fraud rate data extracted from regulatory documents'
                }
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error generating document chart: {e}")
            return None
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get current status of the service
        
        Returns:
            Dictionary with service status information
        """
        status = {
            'initialized': self.initialized,
            'database_connected': self.database_manager.connection is not None,
            'data_loaded': self.database_manager.tables_loaded,
            'document_index_built': self.document_index_built,
            'data_dir': self.data_dir
        }
        
        # Add document processor status and cost optimization stats
        if hasattr(self.document_processor, 'get_document_summary'):
            status['document_summary'] = self.document_processor.get_document_summary()
        
        if hasattr(self.document_processor, 'get_cost_stats'):
            status['cost_optimization'] = self.document_processor.get_cost_stats()
        
        return status
    
    def get_suggested_questions(self) -> List[Dict[str, str]]:
        """
        Get list of suggested questions
        
        Returns:
            List of suggested questions
        """
        return self.query_classifier.get_suggested_questions()
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of loaded data
        
        Returns:
            Dictionary with data summary
        """
        if not self.initialized:
            return {'error': 'Service not initialized'}
        
        return self.database_manager.get_data_summary()
    
    def evaluate_answer(self, question_id: str, chatbot_response: str) -> Dict[str, Any]:
        """
        Evaluate a chatbot response for Q5 and Q6 questions
        
        Args:
            question_id: "5" or "6"
            chatbot_response: The chatbot's response text
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            if not self.evaluation_available or not self.answer_evaluator:
                return {
                    'success': False,
                    'error': 'Answer evaluation not available'
                }
            
            # Perform evaluation
            evaluation_result = self.answer_evaluator.evaluate_response(question_id, chatbot_response)
            
            return evaluation_result
            
        except Exception as e:
            self.logger.error(f"Error evaluating answer: {e}")
            return {
                'success': False,
                'error': f'Evaluation failed: {str(e)}'
            }
    
    def _handle_forecasting_question(self, question: str, question_type: QuestionType,
                                   confidence: float, metadata: Dict) -> Dict[str, Any]:
        """
        Handle forecasting questions using ARIMA models
        
        Args:
            question: User's question
            question_type: Type of question
            confidence: Classification confidence
            metadata: Classification metadata
            
        Returns:
            Response dictionary
        """
        try:
            self.logger.info(f"Handling forecasting question: {question}")
            
            # Use forecasting agent
            forecast_response = self.forecasting_agent.generate_forecast_response(question)
            
            if forecast_response.get('success', False):
                # Add handler information
                forecast_response['handler'] = 'forecasting_agent'
                forecast_response['chart_type'] = 'forecast'
                return forecast_response
            else:
                # If forecasting fails, provide a helpful response
                error_msg = forecast_response.get('error', 'Forecasting failed')
                if 'statsmodels not available' in error_msg or 'Failed to test models' in error_msg:
                    return {
                        'answer': 'I apologize, but the forecasting system is currently unavailable due to missing dependencies. However, I can help you with other fraud analysis questions using the available data.',
                        'status': 'partial_success',
                        'handler': 'forecasting_agent',
                        'error': 'Forecasting dependencies not available',
                        'suggestion': 'Try asking about fraud patterns in the available data (2019-2020) or ask about fraud methods and system components.'
                    }
                else:
                    return {
                        'error': error_msg,
                        'status': 'error',
                        'handler': 'forecasting_agent'
                    }
                
        except Exception as e:
            self.logger.error(f"Error handling forecasting question: {e}")
            return {
                'error': f'Error processing forecasting question: {str(e)}',
                'status': 'error',
                'handler': 'forecasting_agent'
            }
    
    def cleanup(self):
        """Clean up resources"""
        if self.database_manager:
            self.database_manager.disconnect()
        self.logger.info("Service cleanup completed")
