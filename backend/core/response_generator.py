"""
Response Generator for Fraud Detection Chatbot
Assembles responses for different types of questions with appropriate formatting
Enhanced with Deepseek R1 LLM integration
"""
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

# Import OpenAI integration
try:
    from .openai_integration import OpenAIIntegration
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class ResponseGenerator:
    """
    Generates structured responses for different types of fraud analysis questions
    """
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize response generator
        
        Args:
            use_llm: Whether to use LLM for response enhancement (default: True)
        """
        self.use_llm = use_llm and OPENAI_AVAILABLE
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI integration if available
        if self.use_llm:
            try:
                self.openai = OpenAIIntegration()
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI integration: {e}")
                self.use_llm = False
        
        self.response_templates = {
            'temporal_analysis': self._generate_temporal_response,
            'merchant_analysis': self._generate_merchant_response,
            'geographic_analysis': self._generate_geographic_response,
            'value_analysis': self._generate_value_response,
            'fraud_methods': self._generate_document_response,
            'system_components': self._generate_document_response
        }
    
    def generate_response(self, question: str, question_type: str, 
                         data: Optional[pd.DataFrame] = None,
                         document_chunks: Optional[List[Dict]] = None,
                         sql_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate a complete response for a given question
        
        Args:
            question: User's original question
            question_type: Type of analysis (temporal_analysis, etc.)
            data: DataFrame with analysis results
            document_chunks: List of relevant document chunks
            sql_info: SQL query information
            
        Returns:
            Dictionary containing formatted response
        """
        response = {
            'question': question,
            'question_type': question_type,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        # Generate base response based on question type
        if question_type in self.response_templates:
            response.update(self.response_templates[question_type](
                question, data, document_chunks, sql_info
            ))
        else:
            response.update(self._generate_generic_response(
                question, data, document_chunks, sql_info
            ))
        
        # Enhance response with LLM if available
        if self.use_llm and 'answer' in response:
            try:
                # Prepare context for LLM
                data_summary = self._prepare_data_summary(data)
                document_context = self._prepare_document_context(document_chunks)
                
                # Enhance the response
                enhanced_answer = self.openai.enhance_response_with_llm(
                    question=question,
                    question_type=question_type,
                    current_response=response['answer'],
                    data_available=data is not None and not data.empty,
                    document_context=document_context
                )
                
                # Add enhanced response
                response['answer'] = enhanced_answer
                response['llm_enhanced'] = True
                
                # Add additional insights if data is available
                if data is not None and not data.empty:
                    insights = self.openai.generate_insights_with_llm(
                        question=question,
                        data_summary=data_summary,
                        document_context=document_context
                    )
                    response['insights'] = insights
                
            except Exception as e:
                self.logger.warning(f"LLM enhancement failed: {e}")
                response['llm_enhanced'] = False
        
        return response
    
    def _generate_temporal_response(self, question: str, data: pd.DataFrame,
                                   document_chunks: List[Dict], sql_info: Dict) -> Dict[str, Any]:
        """Generate response for temporal analysis questions"""
        if data is None or data.empty:
            return {
                'summary': "No temporal data available for analysis.",
                'insights': [],
                'chart_type': 'line_chart',
                'data': None
            }
        
        # Calculate key statistics
        avg_fraud_rate = data['fraud_rate'].mean()
        max_fraud_rate = data['fraud_rate'].max()
        min_fraud_rate = data['fraud_rate'].min()
        trend = self._calculate_trend(data['fraud_rate'])
        
        # Generate insights
        insights = [
            f"The average fraud rate over the period is {avg_fraud_rate:.2%}",
            f"Fraud rates range from {min_fraud_rate:.2%} to {max_fraud_rate:.2%}",
            f"The overall trend is {trend['direction']} with a {trend['strength']} change"
        ]
        
        # Add specific insights based on data
        if len(data) > 1:
            recent_rate = data['fraud_rate'].iloc[-1]
            early_rate = data['fraud_rate'].iloc[0]
            
            # Handle division by zero case
            if early_rate > 0:
                change = ((recent_rate - early_rate) / early_rate) * 100
                insights.append(f"Fraud rate changed by {change:+.1f}% from start to end of period")
            else:
                # If early rate is 0, just show the absolute change
                absolute_change = recent_rate - early_rate
                if absolute_change > 0:
                    insights.append(f"Fraud rate increased by {absolute_change:.3f} percentage points from start to end of period")
                elif absolute_change < 0:
                    insights.append(f"Fraud rate decreased by {abs(absolute_change):.3f} percentage points from start to end of period")
                else:
                    insights.append("Fraud rate remained at 0% throughout the period")
        
        # Generate summary
        period = sql_info.get('period', 'time') if sql_info else 'time'
        summary = f"Analysis of fraud rate fluctuations over {period} shows significant variation. "
        summary += f"The average fraud rate is {avg_fraud_rate:.2%}, with a {trend['direction']} trend."
        
        return {
            'summary': summary,
            'insights': insights,
            'chart_type': 'line_chart',
            'chart_data': {
                'x_col': 'date' if 'date' in data.columns else data.columns[0],
                'y_col': 'fraud_rate',
                'title': f'Fraud Rate Over {period.title()}'
            },
            'data': data.to_dict('records'),
            'statistics': {
                'average_fraud_rate': avg_fraud_rate,
                'max_fraud_rate': max_fraud_rate,
                'min_fraud_rate': min_fraud_rate,
                'trend': trend
            }
        }
    
    def _generate_merchant_response(self, question: str, data: pd.DataFrame,
                                   document_chunks: List[Dict], sql_info: Dict) -> Dict[str, Any]:
        """Generate response for merchant analysis questions"""
        if data is None or data.empty:
            return {
                'summary': "No merchant data available for analysis.",
                'insights': [],
                'chart_type': 'bar_chart',
                'data': None
            }
        
        # Get top merchants
        top_merchants = data.head(10)
        worst_merchant = top_merchants.iloc[0]
        analysis_type = sql_info.get('analysis_type', 'merchant') if sql_info else 'merchant'
        
        # Generate insights
        insights = [
            f"The {analysis_type} with the highest fraud rate is '{worst_merchant[analysis_type]}' at {worst_merchant['fraud_rate']:.2%}",
            f"Top 10 {analysis_type}s account for {len(top_merchants)} entries in the analysis"
        ]
        
        # Add fraud count insights
        if 'fraud_count' in data.columns:
            total_fraud = data['fraud_count'].sum()
            worst_fraud_count = worst_merchant['fraud_count']
            insights.append(f"The worst {analysis_type} has {worst_fraud_count} fraudulent transactions out of {worst_merchant['total_transactions']} total")
        
        # Generate summary
        summary = f"Analysis of {analysis_type} fraud rates reveals significant variation. "
        summary += f"The highest fraud rate is {worst_merchant['fraud_rate']:.2%} for {worst_merchant[analysis_type]}."
        
        return {
            'summary': summary,
            'insights': insights,
            'chart_type': 'bar_chart',
            'chart_data': {
                'x_col': analysis_type,
                'y_col': 'fraud_rate',
                'title': f'Top {analysis_type.title()}s by Fraud Rate',
                'orientation': 'horizontal'
            },
            'data': data.to_dict('records'),
            'top_merchants': top_merchants.to_dict('records')
        }
    
    def _generate_geographic_response(self, question: str, data: pd.DataFrame,
                                     document_chunks: List[Dict], sql_info: Dict) -> Dict[str, Any]:
        """Generate response for geographic analysis questions"""
        if data is None or data.empty:
            return {
                'summary': "No geographic data available for analysis.",
                'insights': [],
                'chart_type': 'comparison_chart',
                'data': None
            }
        
        # Calculate comparison
        high_value_data = data[data['region'].str.contains('High-Value', case=False, na=False)]
        low_value_data = data[data['region'].str.contains('Low-Value', case=False, na=False)]
        
        if not high_value_data.empty and not low_value_data.empty:
            high_value_rate = high_value_data['fraud_rate'].iloc[0]
            low_value_rate = low_value_data['fraud_rate'].iloc[0]
            difference = high_value_rate - low_value_rate
            percentage_higher = (difference / low_value_rate) * 100 if low_value_rate > 0 else 0
            
            insights = [
                f"High-Value (Cross-border proxy) fraud rate: {high_value_rate:.2%}",
                f"Low-Value (Domestic proxy) fraud rate: {low_value_rate:.2%}",
                f"High-Value transactions have a {percentage_higher:.1f}% {'higher' if percentage_higher > 0 else 'lower'} fraud rate than Low-Value transactions"
            ]
            
            summary = f"Geographic analysis shows that fraud rates differ between transaction types. "
            if percentage_higher > 0:
                summary += f"High-Value transactions (cross-border proxy) have a {percentage_higher:.1f}% higher fraud rate than Low-Value transactions (domestic proxy)."
            else:
                summary += f"Low-Value transactions (domestic proxy) have a {abs(percentage_higher):.1f}% higher fraud rate than High-Value transactions (cross-border proxy)."
        elif not high_value_data.empty:
            high_value_rate = high_value_data['fraud_rate'].iloc[0]
            insights = [
                f"High-Value (Cross-border proxy) fraud rate: {high_value_rate:.2%}",
                "No Low-Value (Domestic proxy) data available for comparison"
            ]
            summary = f"Geographic analysis shows High-Value transactions fraud rate of {high_value_rate:.2%}, but no Low-Value transactions data is available for comparison."
        elif not low_value_data.empty:
            low_value_rate = low_value_data['fraud_rate'].iloc[0]
            insights = [
                f"Low-Value (Domestic proxy) fraud rate: {low_value_rate:.2%}",
                "No High-Value (Cross-border proxy) data available for comparison"
            ]
            summary = f"Geographic analysis shows Low-Value transactions fraud rate of {low_value_rate:.2%}, but no High-Value transactions data is available for comparison."
        else:
            insights = ["Insufficient data for geographic comparison"]
            summary = "Geographic analysis could not be completed due to insufficient data."
        
        return {
            'summary': summary,
            'insights': insights,
            'chart_type': 'comparison_chart',
            'chart_data': {
                'x_col': 'region',
                'y_col': 'fraud_rate',
                'title': 'Fraud Rate by Region (EEA vs Non-EEA)'
            },
            'data': data.to_dict('records')
        }
    
    def _generate_value_response(self, question: str, data: pd.DataFrame,
                                document_chunks: List[Dict], sql_info: Dict) -> Dict[str, Any]:
        """Generate response for value analysis questions"""
        print(f"\n=== RESPONSE GENERATOR DEBUG ===")
        print(f"Question: {question}")
        print(f"Data shape: {data.shape if data is not None else 'None'}")
        if data is not None and not data.empty:
            print(f"Data columns: {data.columns.tolist()}")
            print(f"Data preview: {data.head().to_dict()}")
        print("=== END RESPONSE GENERATOR DEBUG ===\n")
        
        if data is None or data.empty:
            return {
                'summary': "No value data available for analysis.",
                'insights': [],
                'chart_type': 'pie_chart',
                'data': None
            }
        
        # Calculate total values
        total_fraud_value = data['fraud_value'].sum()
        
        # Look for cross-border data with flexible matching
        cross_border_data = data[data['transaction_type'].str.contains('Cross-border', case=False, na=False)]
        domestic_data = data[data['transaction_type'].str.contains('Domestic', case=False, na=False)]
        
        if not cross_border_data.empty:
            cross_border_share = cross_border_data['percentage_share'].iloc[0]
            cross_border_value = cross_border_data['fraud_value'].iloc[0]
            
            insights = [
                f"Total fraud value in H1 2023: ${total_fraud_value:,.2f}",
                f"Cross-border fraud value: ${cross_border_value:,.2f}",
                f"Cross-border transactions account for {cross_border_share:.1f}% of total fraud value"
            ]
            
            summary = f"In H1 2023, cross-border transactions accounted for {cross_border_share:.1f}% of total fraud value. "
            summary += f"This represents ${cross_border_value:,.2f} out of ${total_fraud_value:,.2f} total fraud value."
        else:
            insights = ["Insufficient data for value analysis"]
            summary = "Value analysis could not be completed due to insufficient data."
        
        return {
            'summary': summary,
            'insights': insights,
            'chart_type': 'pie_chart',
            'chart_data': {
                'labels_col': 'transaction_type',
                'values_col': 'fraud_value',
                'title': 'Fraud Value Distribution by Transaction Type (H1 2023)'
            },
            'data': data.to_dict('records'),
            'total_fraud_value': total_fraud_value
        }
    
    def _generate_document_response(self, question: str, data: pd.DataFrame,
                                   document_chunks: List[Dict], sql_info: Dict) -> Dict[str, Any]:
        """Generate response for document-based questions"""
        if not document_chunks:
            return {
                'summary': "No relevant information found in the documents.",
                'insights': [],
                'chart_type': None,
                'data': None
            }
        
        # Extract key information from chunks
        sources = list(set([chunk.get('source', 'Unknown') for chunk in document_chunks]))
        top_chunks = sorted(document_chunks, key=lambda x: x.get('similarity_score', 0), reverse=True)[:5]
        
        # Generate summary from top chunks
        summary_parts = []
        for chunk in top_chunks[:3]:
            text = chunk.get('text', '')[:200] + "..." if len(chunk.get('text', '')) > 200 else chunk.get('text', '')
            summary_parts.append(text)
        
        summary = " ".join(summary_parts)
        
        # Generate insights
        insights = [
            f"Found {len(document_chunks)} relevant document sections",
            f"Sources: {', '.join(sources)}",
            f"Top relevance score: {top_chunks[0].get('similarity_score', 0):.3f}"
        ]
        
        return {
            'summary': summary,
            'insights': insights,
            'chart_type': None,
            'data': None,
            'document_chunks': top_chunks,
            'sources': sources
        }
    
    def _generate_generic_response(self, question: str, data: pd.DataFrame,
                                  document_chunks: List[Dict], sql_info: Dict) -> Dict[str, Any]:
        """Generate a generic response when question type is unknown"""
        return {
            'summary': f"Analysis completed for: {question}",
            'insights': ["Generic analysis performed"],
            'chart_type': 'bar_chart' if data is not None else None,
            'data': data.to_dict('records') if data is not None else None
        }
    
    def _calculate_trend(self, values: pd.Series) -> Dict[str, str]:
        """Calculate trend direction and strength from a series of values"""
        if len(values) < 2:
            return {'direction': 'stable', 'strength': 'no change'}
        
        # Simple linear trend calculation
        x = range(len(values))
        y = values.values
        
        # Calculate slope
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        if n * sum_x2 - sum_x ** 2 == 0:
            slope = 0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        # Determine direction and strength
        if abs(slope) < 0.001:
            direction = 'stable'
            strength = 'minimal'
        elif slope > 0:
            direction = 'increasing'
            strength = 'strong' if slope > 0.01 else 'moderate'
        else:
            direction = 'decreasing'
            strength = 'strong' if abs(slope) > 0.01 else 'moderate'
        
        return {'direction': direction, 'strength': strength}
    
    def format_response_for_display(self, response: Dict[str, Any]) -> str:
        """
        Format response for display in the chat interface
        
        Args:
            response: Response dictionary
            
        Returns:
            Formatted string for display
        """
        formatted = f"## {response['question']}\n\n"
        
        if 'summary' in response:
            formatted += f"**Summary:** {response['summary']}\n\n"
        
        if 'insights' in response and response['insights']:
            formatted += "**Key Insights:**\n"
            for insight in response['insights']:
                formatted += f"- {insight}\n"
            formatted += "\n"
        
        if 'statistics' in response:
            formatted += "**Statistics:**\n"
            for key, value in response['statistics'].items():
                if isinstance(value, float):
                    formatted += f"- {key.replace('_', ' ').title()}: {value:.2%}\n"
                else:
                    formatted += f"- {key.replace('_', ' ').title()}: {value}\n"
            formatted += "\n"
        
        if 'sources' in response and response['sources']:
            formatted += "**Sources:**\n"
            for source in response['sources']:
                formatted += f"- {source}\n"
            formatted += "\n"
        
        return formatted
    
    def _prepare_data_summary(self, data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """
        Prepare data summary for LLM context
        
        Args:
            data: DataFrame with analysis results
            
        Returns:
            Dictionary with data summary
        """
        if data is None or data.empty:
            return {"status": "no_data"}
        
        summary = {
            "status": "data_available",
            "shape": data.shape,
            "columns": list(data.columns),
            "data_types": data.dtypes.to_dict()
        }
        
        # Add basic statistics for numeric columns
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary["statistics"] = data[numeric_cols].describe().to_dict()
        
        # Add sample data (first 3 rows)
        summary["sample_data"] = data.head(3).to_dict('records')
        
        return summary
    
    def _prepare_document_context(self, document_chunks: Optional[List[Dict]]) -> str:
        """
        Prepare document context for LLM
        
        Args:
            document_chunks: List of relevant document chunks
            
        Returns:
            Formatted document context string
        """
        if not document_chunks:
            return ""
        
        context_parts = []
        for i, chunk in enumerate(document_chunks[:3]):  # Limit to top 3 chunks
            context_parts.append(f"Source {i+1}: {chunk.get('text', '')[:500]}...")
        
        return "\n\n".join(context_parts)
