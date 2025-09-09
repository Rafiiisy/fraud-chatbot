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

from core.sql_generator import SQLGenerator, QuestionType
from core.simple_document_processor import SimpleDocumentProcessor
from core.chart_generator import ChartGenerator
from core.response_generator import ResponseGenerator
from core.query_classifier import QueryClassifier
from data.api_database_manager import APIDatabaseManager
from agents.agent_coordinator import AgentCoordinator
from agents.forecasting_agent import ForecastingAgent


class FraudAnalysisService:
    """
    Main service that orchestrates fraud analysis components
    """
    
    def __init__(self, data_dir: str = "dataset"):
        self.data_dir = data_dir
        
        # Initialize components with LLM integration
        self.sql_generator = SQLGenerator()
        self.document_processor = SimpleDocumentProcessor(data_dir)
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
            return {
                'error': 'Service not initialized. Call initialize() first.',
                'status': 'error'
            }
        
        try:
            # Validate query
            is_valid, errors = self.query_classifier.validate_query(question)
            if not is_valid:
                return {
                    'error': 'Invalid query: ' + '; '.join(errors),
                    'status': 'error'
                }
            
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
                self.logger.info(f"Routing question to SQL generator")
                return self._handle_database_question(question, question_type, confidence, metadata)
            elif handler_info['handler'] == 'document_processor':
                self.logger.info(f"Routing question to document processor")
                return self._handle_document_question(question, question_type, confidence, metadata)
            else:
                self.logger.error(f"Unknown handler: {handler_info['handler']} for question type: {question_type}")
                self.logger.error(f"Forecasting available: {self.forecasting_available}, Agent: {self.forecasting_agent is not None}")
                return {
                    'error': f'Unknown handler: {handler_info["handler"]}',
                    'status': 'error'
                }
                
        except Exception as e:
            self.logger.error(f"Error processing question: {e}")
            return {
                'error': f'Error processing question: {str(e)}',
                'status': 'error'
            }
    
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
                    # Generate and execute SQL query
                    sql_info = self.sql_generator.generate_sql(question, question_type)
                    if self.sql_generator.validate_sql(sql_info['sql']):
                        success, data, error = self.database_manager.execute_query(sql_info['sql'])
                        if success and data is not None:
                            context_data['sql_data'] = data
                            context_data['sql_query'] = sql_info['sql']
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
                # Fallback to traditional handlers
                self.logger.warning("Agent analysis failed, falling back to traditional handlers")
                if question_type in [QuestionType.MERCHANT_ANALYSIS, QuestionType.TEMPORAL_ANALYSIS, 
                                   QuestionType.GEOGRAPHIC_ANALYSIS, QuestionType.VALUE_ANALYSIS]:
                    return self._handle_database_question(question, question_type, confidence, metadata)
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
            # Fallback to traditional handlers
            if question_type in [QuestionType.MERCHANT_ANALYSIS, QuestionType.TEMPORAL_ANALYSIS, 
                               QuestionType.GEOGRAPHIC_ANALYSIS, QuestionType.VALUE_ANALYSIS]:
                return self._handle_database_question(question, question_type, confidence, metadata)
            else:
                return self._handle_document_question(question, question_type, confidence, metadata)
    
    def process_question_sync(self, question: str) -> Dict[str, Any]:
        """
        Synchronous version of process_question for fallback
        """
        if not self.initialized:
            return {
                'error': 'Service not initialized. Call initialize() first.',
                'status': 'error'
            }
        
        try:
            # Validate query
            is_valid, errors = self.query_classifier.validate_query(question)
            if not is_valid:
                return {
                    'error': 'Invalid query: ' + '; '.join(errors),
                    'status': 'error'
                }
            
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
                self.logger.info(f"Routing question to SQL generator")
                return self._handle_database_question(question, question_type, confidence, metadata)
            elif handler_info['handler'] == 'document_processor':
                self.logger.info(f"Routing question to document processor")
                return self._handle_document_question(question, question_type, confidence, metadata)
            else:
                return {
                    'error': f'Unknown handler: {handler_info["handler"]}',
                    'status': 'error'
                }
                
        except Exception as e:
            self.logger.error(f"Error processing question: {e}")
            return {
                'error': f'Error processing question: {str(e)}',
                'status': 'error'
            }
    
    def _handle_database_question(self, question: str, question_type: QuestionType, 
                                 confidence: float, metadata: Dict) -> Dict[str, Any]:
        """
        Handle database-related questions
        
        Args:
            question: User's question
            question_type: Type of question
            confidence: Classification confidence
            metadata: Classification metadata
            
        Returns:
            Response dictionary
        """
        try:
            print(f"\n=== FRAUD ANALYSIS SERVICE DEBUG ===")
            print(f"Question: {question}")
            print(f"Question Type: {question_type}")
            print(f"Confidence: {confidence}")
            print(f"Metadata: {metadata}")
            
            # Generate SQL query
            sql_info = self.sql_generator.generate_sql(question, question_type)
            print(f"SQL Info: {sql_info}")
            
            # Validate SQL
            if not self.sql_generator.validate_sql(sql_info['sql']):
                print("SQL validation failed!")
                return {
                    'error': 'Generated SQL query failed validation',
                    'status': 'error'
                }
            
            print("SQL validation passed, executing query...")
            
            # Execute query
            success, data, error = self.database_manager.execute_query(sql_info['sql'])
            if not success:
                print(f"Database query failed: {error}")
                return {
                    'error': f'Database query failed: {error}',
                    'status': 'error'
                }
            
            print(f"Database query successful, data shape: {data.shape if data is not None else 'None'}")
            print("=== END FRAUD ANALYSIS SERVICE DEBUG ===\n")
            
            # Generate chart
            chart = None
            if data is not None and not data.empty:
                chart = self._generate_chart(question_type, data, sql_info)
            
            # Generate response
            response = self.response_generator.generate_response(
                question, question_type.value, data, None, sql_info
            )
            
            # Add chart and metadata
            response['chart'] = chart
            response['sql_query'] = sql_info['sql']
            response['confidence'] = confidence
            response['metadata'] = metadata
            response['handler'] = 'sql_generator'
            response['chart_type'] = self.query_classifier.get_handler_info(question_type, metadata).get('chart_type')
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error handling database question: {e}")
            return {
                'error': f'Error handling database question: {str(e)}',
                'status': 'error'
            }
    
    def _handle_document_question(self, question: str, question_type: QuestionType,
                                 confidence: float, metadata: Dict) -> Dict[str, Any]:
        """
        Handle document-related questions using OpenAI-based search
        
        Args:
            question: User's question
            question_type: Type of question
            confidence: Classification confidence
            metadata: Classification metadata
            
        Returns:
            Response dictionary
        """
        try:
            if not self.document_index_built:
                return {
                    'error': 'Document search not available. Documents not processed.',
                    'status': 'error'
                }
            
            # Search documents using OpenAI-based processor
            search_result = self.document_processor.search_documents(question, max_results=5)
            
            if not search_result.get('success', False):
                return {
                    'error': f"Document search failed: {search_result.get('error', 'Unknown error')}",
                    'status': 'error'
                }
            
            # Extract answer and sources from OpenAI response
            answer = search_result.get('answer', 'No answer found')
            sources = search_result.get('sources', [])
            search_confidence = search_result.get('confidence', 0.8)
            
            # Return the document processor response directly
            response = {
                'question_type': question_type.value,
                'answer': answer,
                'sources': sources,
                'confidence': min(confidence, search_confidence),
                'metadata': metadata,
                'document_search_performed': True,
                'handler': 'document_processor',
                'chart_type': None
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error handling document question: {e}")
            return {
                'error': f'Error handling document question: {str(e)}',
                'status': 'error'
            }
    
    def _generate_chart(self, question_type: QuestionType, data: pd.DataFrame, 
                       sql_info: Dict) -> Optional[Dict[str, Any]]:
        """
        Generate appropriate chart for the question type
        
        Args:
            question_type: Type of question
            data: DataFrame with data
            sql_info: SQL generation information
            
        Returns:
            Chart configuration dictionary
        """
        try:
            if data is None or data.empty:
                return None
            
            if question_type == QuestionType.TEMPORAL_ANALYSIS:
                period = sql_info.get('period', 'monthly')
                return {
                    'type': 'line_chart',
                    'data': data.to_dict('records'),
                    'x_col': 'date' if 'date' in data.columns else data.columns[0],
                    'y_col': 'fraud_rate',
                    'title': f'Fraud Rate Over {period.title()}'
                }
            
            elif question_type == QuestionType.MERCHANT_ANALYSIS:
                analysis_type = sql_info.get('analysis_type', 'merchant')
                return {
                    'type': 'bar_chart',
                    'data': data.to_dict('records'),
                    'x_col': analysis_type,
                    'y_col': 'fraud_rate',
                    'title': f'Top {analysis_type.title()}s by Fraud Rate',
                    'orientation': 'horizontal'
                }
            
            elif question_type == QuestionType.GEOGRAPHIC_ANALYSIS:
                return {
                    'type': 'comparison_chart',
                    'data': data.to_dict('records'),
                    'x_col': 'region',
                    'y_col': 'fraud_rate',
                    'title': 'Fraud Rate by Region (EEA vs Non-EEA)'
                }
            
            elif question_type == QuestionType.VALUE_ANALYSIS:
                return {
                    'type': 'pie_chart',
                    'data': data.to_dict('records'),
                    'labels_col': 'transaction_type',
                    'values_col': 'fraud_value',
                    'title': 'Fraud Value Distribution by Transaction Type (H1 2023)'
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating chart: {e}")
            return None
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get current status of the service
        
        Returns:
            Dictionary with service status information
        """
        return {
            'initialized': self.initialized,
            'database_connected': self.database_manager.connection is not None,
            'data_loaded': self.database_manager.tables_loaded,
            'document_index_built': self.document_index_built,
            'data_dir': self.data_dir
        }
    
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
