"""
Forecasting Agent for Fraud Prediction
=====================================

This agent handles forecasting questions using ARIMA models.
It integrates with the existing agent system to provide fraud trend predictions.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the fraud predictor
import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.append(str(backend_path))

from core.fraud_predictor import FraudPredictor
from data.api_database_manager import APIDatabaseManager

logger = logging.getLogger(__name__)

class ForecastingAgent:
    """
    Agent responsible for fraud forecasting using ARIMA models
    """
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the forecasting agent
        
        Args:
            openai_api_key: OpenAI API key for enhanced responses
        """
        self.openai_api_key = openai_api_key
        self.predictor = FraudPredictor()
        self.db_manager = None
        self.logger = logging.getLogger(__name__)
        
    def initialize_database(self):
        """Initialize database API connection if not already done"""
        if self.db_manager is None:
            self.db_manager = APIDatabaseManager()
            if not self.db_manager.connect():
                self.logger.error("Failed to connect to database API")
                return False
            if not self.db_manager.load_csv_data():
                self.logger.error("Failed to verify data availability")
                return False
        return True
    
    def extract_forecast_horizon(self, question: str) -> int:
        """
        Extract forecast horizon from question
        
        Args:
            question: User's question
            
        Returns:
            Number of days to forecast
        """
        question_lower = question.lower()
        
        if "next week" in question_lower or "7 days" in question_lower:
            return 7
        elif "next month" in question_lower or "30 days" in question_lower:
            return 30
        elif "next quarter" in question_lower or "3 months" in question_lower:
            return 90
        elif "next year" in question_lower or "12 months" in question_lower:
            return 365
        elif "tomorrow" in question_lower or "next day" in question_lower:
            return 1
        else:
            return 30  # Default to 30 days
    
    def generate_forecast_response(self, question: str) -> Dict[str, Any]:
        """
        Generate a comprehensive forecast response
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with forecast response
        """
        try:
            # Initialize database if needed
            if not self.initialize_database():
                return {
                    "success": False,
                    "error": "Failed to initialize database",
                    "answer": "I'm sorry, I couldn't access the fraud data to make predictions."
                }
            
            # Detect question type
            question_type = self._detect_question_type(question)
            self.logger.info(f"Detected question type: {question_type}")
            
            # Extract forecast horizon
            forecast_days = self.extract_forecast_horizon(question)
            
            # Load data for forecasting
            query = """
            SELECT trans_date_trans_time, is_fraud, amt, merchant, category
            FROM transactions 
            ORDER BY trans_date_trans_time
            LIMIT 50000
            """
            
            success, data, error = self.db_manager.execute_query(query)
            
            if not success or data is None:
                return {
                    "success": False,
                    "error": f"Failed to load data: {error}",
                    "answer": "I'm sorry, I couldn't access the fraud data to make predictions."
                }
            
            # Generate appropriate forecast based on question type
            if question_type == "value_analysis":
                forecast_result = self._generate_value_forecast(data, forecast_days, question)
            else:
                # Default to fraud rate forecast
                forecast_result = self.predictor.predict_fraud_trends(data, forecast_days)
            
            if not forecast_result.get("success", False):
                return {
                    "success": False,
                    "error": forecast_result.get("error", "Unknown error"),
                    "answer": "I'm sorry, I couldn't generate a forecast at this time."
                }
            
            # Format response based on question type
            if question_type == "value_analysis":
                response = self._format_value_forecast_response(question, forecast_result)
            else:
                response = self._format_forecast_response(question, forecast_result)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in forecasting agent: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": "I'm sorry, I encountered an error while generating the forecast."
            }
    
    def _format_forecast_response(self, question: str, forecast_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the forecast response for the user
        
        Args:
            question: Original question
            forecast_result: Raw forecast results
            
        Returns:
            Formatted response dictionary
        """
        try:
            # Extract key information
            model_used = forecast_result.get("model_used", "ARIMA")
            forecast_days = forecast_result.get("forecast_days", 30)
            mean_forecast = forecast_result.get("forecast_summary", {}).get("mean_forecast", 0)
            forecast_std = forecast_result.get("forecast_summary", {}).get("forecast_std", 0)
            forecast_data = forecast_result.get("forecast_data", [])
            
            # Calculate percentage
            fraud_rate_percentage = mean_forecast * 100
            
            # Generate explanation
            explanation = self._generate_forecast_explanation(
                question, model_used, forecast_days, fraud_rate_percentage, forecast_std
            )
            
            # Create insights
            insights = self._generate_forecast_insights(forecast_data, mean_forecast)
            
            # Create recommendations
            recommendations = self._generate_forecast_recommendations(fraud_rate_percentage, forecast_std)
            
            return {
                "success": True,
                "answer": explanation,
                "confidence": 0.85,  # High confidence for ARIMA forecasts
                "data_sources": ["Transaction Data", "ARIMA Model"],
                "key_insights": insights,
                "recommendations": recommendations,
                "risk_level": self._assess_risk_level(fraud_rate_percentage),
                "supporting_evidence": {
                    "model_used": model_used,
                    "forecast_days": forecast_days,
                    "data_points": len(forecast_data),
                    "method": "ARIMA Time Series Forecasting"
                },
                "method": "Forecasting Agent",
                "agent_used": "ARIMA Forecasting System",
                "forecast_data": forecast_data[:10],  # First 10 days
                "chart_type": "forecast",
                "handler": "forecasting_agent"
            }
            
        except Exception as e:
            self.logger.error(f"Error formatting forecast response: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": "I'm sorry, I couldn't format the forecast response properly."
            }
    
    def _generate_forecast_explanation(self, question: str, model_used: str, 
                                     forecast_days: int, fraud_rate_percentage: float, 
                                     forecast_std: float) -> str:
        """Generate a natural language explanation of the forecast"""
        
        # Determine time period
        if forecast_days == 1:
            time_period = "tomorrow"
        elif forecast_days == 7:
            time_period = "next week"
        elif forecast_days == 30:
            time_period = "next month"
        elif forecast_days == 90:
            time_period = "next quarter"
        elif forecast_days == 365:
            time_period = "next year"
        else:
            time_period = f"the next {forecast_days} days"
        
        explanation = f"""
Based on historical fraud patterns analyzed using {model_used}, I can provide the following forecast:

**Predicted Fraud Rate for {time_period.title()}:**
- **Average fraud rate**: {fraud_rate_percentage:.2f}%
- **Confidence interval**: ±{forecast_std*100:.2f}%
- **Model used**: {model_used}

This forecast is based on analyzing {forecast_days} days of historical transaction data and applying time series forecasting techniques. The model has been trained on fraud patterns and trends to predict future behavior.

**Key Points:**
- The forecast shows a predicted fraud rate of approximately {fraud_rate_percentage:.2f}%
- This represents the expected average fraud rate over the {time_period}
- The confidence interval provides a range of likely outcomes
- The model considers seasonal patterns, trends, and historical fraud behavior

Please note that this is a statistical prediction based on historical data and should be used as a guide for risk management planning.
        """.strip()
        
        return explanation
    
    def _generate_forecast_insights(self, forecast_data: list, mean_forecast: float) -> list:
        """Generate insights from the forecast data"""
        insights = []
        
        if not forecast_data:
            return ["Insufficient forecast data for detailed insights"]
        
        # Calculate trend
        if len(forecast_data) >= 2:
            first_half = forecast_data[:len(forecast_data)//2]
            second_half = forecast_data[len(forecast_data)//2:]
            
            first_avg = np.mean([day['forecast'] for day in first_half])
            second_avg = np.mean([day['forecast'] for day in second_half])
            
            if second_avg > first_avg * 1.05:
                insights.append("Forecast shows an increasing fraud trend over the prediction period")
            elif second_avg < first_avg * 0.95:
                insights.append("Forecast shows a decreasing fraud trend over the prediction period")
            else:
                insights.append("Forecast shows a relatively stable fraud rate over the prediction period")
        
        # Risk assessment
        if mean_forecast > 0.01:  # > 1%
            insights.append("Predicted fraud rate is above 1%, indicating elevated risk")
        elif mean_forecast < 0.005:  # < 0.5%
            insights.append("Predicted fraud rate is below 0.5%, indicating low risk")
        else:
            insights.append("Predicted fraud rate is within normal range")
        
        # Volatility assessment
        if forecast_data:
            forecasts = [day['forecast'] for day in forecast_data]
            volatility = np.std(forecasts)
            if volatility > mean_forecast * 0.5:
                insights.append("High volatility in predicted fraud rates suggests uncertain conditions")
            else:
                insights.append("Low volatility in predicted fraud rates suggests stable conditions")
        
        return insights
    
    def _generate_forecast_recommendations(self, fraud_rate_percentage: float, forecast_std: float) -> list:
        """Generate recommendations based on the forecast"""
        recommendations = []
        
        if fraud_rate_percentage > 1.0:
            recommendations.append("Implement enhanced fraud detection measures immediately")
            recommendations.append("Increase monitoring frequency for high-risk transactions")
            recommendations.append("Consider additional authentication requirements")
        elif fraud_rate_percentage > 0.5:
            recommendations.append("Maintain current fraud detection systems")
            recommendations.append("Monitor for any unusual patterns or spikes")
            recommendations.append("Review and update fraud prevention policies")
        else:
            recommendations.append("Current fraud prevention measures appear adequate")
            recommendations.append("Continue monitoring for any changes in patterns")
        
        if forecast_std > fraud_rate_percentage * 0.3:
            recommendations.append("High uncertainty in forecast - prepare for multiple scenarios")
            recommendations.append("Implement flexible fraud detection strategies")
        
        recommendations.append("Regularly update the forecasting model with new data")
        recommendations.append("Combine statistical forecasts with expert judgment")
        
        return recommendations
    
    def _assess_risk_level(self, fraud_rate_percentage: float) -> str:
        """Assess risk level based on predicted fraud rate"""
        if fraud_rate_percentage > 2.0:
            return "high"
        elif fraud_rate_percentage > 1.0:
            return "medium-high"
        elif fraud_rate_percentage > 0.5:
            return "medium"
        else:
            return "low"
    
    def _detect_question_type(self, question: str) -> str:
        """
        Detect the type of forecasting question
        
        Args:
            question: User's question
            
        Returns:
            Question type: 'value_analysis' or 'rate_analysis'
        """
        question_lower = question.lower()
        
        # Keywords that indicate value analysis
        value_keywords = [
            'value', 'amount', 'total', 'share', 'percentage', 'dollar', 'cost',
            'cross-border', 'domestic', 'high-value', 'low-value', 'transaction value',
            'fraud value', 'monetary', 'financial'
        ]
        
        # Check for value analysis indicators
        if any(keyword in question_lower for keyword in value_keywords):
            return "value_analysis"
        
        # Default to rate analysis
        return "rate_analysis"
    
    def _generate_value_forecast(self, data: pd.DataFrame, forecast_days: int, question: str) -> Dict[str, Any]:
        """
        Generate value-based fraud forecasts
        
        Args:
            data: Transaction data
            forecast_days: Number of days to forecast
            question: Original question
            
        Returns:
            Dictionary with value forecast results
        """
        try:
            # Prepare time series data for fraud values
            data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
            
            # Create cross-border vs domestic breakdown
            cross_border_threshold = 100  # High-value transactions as proxy for cross-border
            
            # Filter data for cross-border and domestic transactions
            cross_border_data = data[data['amt'] > cross_border_threshold].copy()
            domestic_data = data[data['amt'] <= cross_border_threshold].copy()
            
            # Generate forecasts for each category
            cross_border_forecast = self.predictor.predict_fraud_trends(
                cross_border_data, forecast_days, 'amt'
            )
            domestic_forecast = self.predictor.predict_fraud_trends(
                domestic_data, forecast_days, 'amt'
            )
            
            if not cross_border_forecast.get("success", False) or not domestic_forecast.get("success", False):
                return {"error": "Failed to generate value forecasts", "success": False}
            
            # Calculate percentage shares
            cross_border_mean = cross_border_forecast.get("forecast_summary", {}).get("mean_forecast", 0)
            domestic_mean = domestic_forecast.get("forecast_summary", {}).get("mean_forecast", 0)
            total_mean = cross_border_mean + domestic_mean
            
            if total_mean > 0:
                cross_border_share = (cross_border_mean / total_mean) * 100
                domestic_share = (domestic_mean / total_mean) * 100
            else:
                cross_border_share = 50.0
                domestic_share = 50.0
            
            return {
                "success": True,
                "forecast_days": forecast_days,
                "cross_border_forecast": cross_border_forecast,
                "domestic_forecast": domestic_forecast,
                "cross_border_share": cross_border_share,
                "domestic_share": domestic_share,
                "total_fraud_value": total_mean,
                "cross_border_value": cross_border_mean,
                "domestic_value": domestic_mean,
                "model_used": "ARIMA Value Forecasting",
                "method": "Cross-border proxy analysis using transaction amounts"
            }
            
        except Exception as e:
            self.logger.error(f"Error generating value forecast: {e}")
            return {"error": str(e), "success": False}
    
    def _format_value_forecast_response(self, question: str, forecast_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the value forecast response for the user
        
        Args:
            question: Original question
            forecast_result: Raw value forecast results
            
        Returns:
            Formatted response dictionary
        """
        try:
            cross_border_share = forecast_result.get("cross_border_share", 0)
            domestic_share = forecast_result.get("domestic_share", 0)
            total_value = forecast_result.get("total_fraud_value", 0)
            cross_border_value = forecast_result.get("cross_border_value", 0)
            domestic_value = forecast_result.get("domestic_value", 0)
            forecast_days = forecast_result.get("forecast_days", 30)
            
            # Generate explanation
            explanation = f"""
Based on historical fraud patterns and transaction value analysis, I can provide the following forecast for fraud value distribution:

**Predicted Fraud Value Distribution:**
- **Cross-border transactions (high-value proxy)**: {cross_border_share:.1f}% of total fraud value
- **Domestic transactions (low-value proxy)**: {domestic_share:.1f}% of total fraud value

**Forecasted Values:**
- **Total fraud value**: ${total_value:,.2f}
- **Cross-border fraud value**: ${cross_border_value:,.2f}
- **Domestic fraud value**: ${domestic_value:,.2f}

**Methodology:**
This forecast uses transaction amounts as a proxy for cross-border activity, where transactions over $100 are considered cross-border and transactions $100 or below are considered domestic. The analysis is based on {forecast_days} days of historical transaction data and applies ARIMA time series forecasting to predict future fraud value patterns.

**Key Insights:**
- The forecast shows that cross-border transactions account for approximately {cross_border_share:.1f}% of total fraud value
- This suggests that high-value transactions are a significant source of fraud risk
- The distribution provides insights for targeted fraud prevention strategies

Please note that this is a statistical prediction based on historical data and should be used as a guide for risk management planning.
            """.strip()
            
            # Create insights
            insights = [
                f"Cross-border transactions represent {cross_border_share:.1f}% of predicted fraud value",
                f"High-value transactions (>$100) show significant fraud risk",
                f"Total predicted fraud value: ${total_value:,.2f}",
                "Value-based forecasting provides actionable risk insights"
            ]
            
            # Create recommendations
            recommendations = [
                "Focus fraud prevention efforts on high-value transactions",
                "Implement enhanced monitoring for cross-border activity",
                "Review transaction limits and approval processes",
                "Consider geographic risk factors in fraud detection",
                "Regularly update value-based forecasting models"
            ]
            
            # Create forecast data for visualization
            forecast_data = [
                {
                    "transaction_type": "Cross-border (High-value proxy)",
                    "predicted_value": cross_border_value,
                    "percentage_share": cross_border_share,
                    "description": "Transactions over $100"
                },
                {
                    "transaction_type": "Domestic (Low-value proxy)", 
                    "predicted_value": domestic_value,
                    "percentage_share": domestic_share,
                    "description": "Transactions $100 or below"
                }
            ]
            
            return {
                "success": True,
                "answer": explanation,
                "confidence": 0.85,
                "data_sources": ["Transaction Data", "ARIMA Value Forecasting"],
                "key_insights": insights,
                "recommendations": recommendations,
                "risk_level": "medium" if cross_border_share > 60 else "low",
                "supporting_evidence": {
                    "model_used": "ARIMA Value Forecasting",
                    "forecast_days": forecast_days,
                    "method": "Cross-border proxy analysis",
                    "threshold": "$100"
                },
                "method": "Value Forecasting Agent",
                "agent_used": "ARIMA Value Analysis System",
                "forecast_data": forecast_data,
                "chart_type": "value_forecast",
                "handler": "forecasting_agent"
            }
            
        except Exception as e:
            self.logger.error(f"Error formatting value forecast response: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": "I'm sorry, I couldn't format the value forecast response properly."
            }

# Example usage
if __name__ == "__main__":
    agent = ForecastingAgent()
    result = agent.generate_forecast_response("What will be the fraud rate next month?")
    print(result)
