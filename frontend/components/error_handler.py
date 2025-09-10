"""
Error Handler Component
Handles error formatting and user-friendly error messages
"""
import streamlit as st
from typing import Dict, Any


class ErrorHandler:
    """
    Handles error formatting and provides user-friendly error messages
    """
    
    def __init__(self):
        pass
    
    def format_error_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format error response with user-friendly messages
        
        Args:
            response_data: Backend error response
            
        Returns:
            Formatted error response for frontend display
        """
        # Extract error from the correct location in the response
        data = response_data.get('data', {})
        error_message = data.get('error', response_data.get('error', 'Unknown error'))
        error_details = data.get('message', response_data.get('message', 'No additional details'))
        
        # Format error message based on type
        if "not related to fraud analysis" in error_message:
            formatted_message = "🚫 **Question Not Related to Fraud Analysis**"
            formatted_details = f"I'm a fraud detection chatbot. {error_message}\n\n**Try asking about:**\n- Fraud rate patterns over time\n- Merchant fraud analysis\n- Geographic fraud comparisons\n- Payment security methods\n- Fraud detection system components"
        elif "too short" in error_message:
            formatted_message = "📝 **Question Too Short**"
            formatted_details = f"{error_message}\n\n**Please provide more details** about what you'd like to know about fraud analysis."
        elif "harmful SQL keywords" in error_message:
            formatted_message = "⚠️ **Invalid Query**"
            formatted_details = f"{error_message}\n\n**Please ask questions in natural language** about fraud analysis instead of using technical commands."
        elif "Service not initialized" in error_message:
            formatted_message = "🔧 **Service Not Ready**"
            formatted_details = f"{error_message}\n\n**Please wait a moment** and try again. The system is still starting up."
        elif "Backend API not available" in error_message:
            formatted_message = "🔌 **Backend Connection Error**"
            formatted_details = f"{error_message}\n\n**Please ensure:**\n- Backend service is running on http://localhost:5000\n- Docker containers are started\n- No firewall blocking the connection"
        else:
            formatted_message = f"❌ **Error: {error_message}**"
            formatted_details = error_details
        
        return {
            "error": True,
            "message": formatted_message,
            "details": formatted_details
        }
    
    def format_processing_error(self, error: Exception) -> Dict[str, Any]:
        """
        Format processing errors with user-friendly messages
        
        Args:
            error: Exception that occurred
            
        Returns:
            Formatted error response for frontend display
        """
        error_str = str(error)
        
        if "Connection refused" in error_str or "ConnectionError" in error_str:
            formatted_message = "🔌 **Connection Error**"
            formatted_details = "Could not connect to the backend service.\n\n**Please check:**\n- Backend is running on http://localhost:5000\n- Docker containers are started\n- Network connection is working"
        elif "Timeout" in error_str or "timeout" in error_str:
            formatted_message = "⏱️ **Request Timeout**"
            formatted_details = "The request took too long to process.\n\n**Please try:**\n- Asking a simpler question\n- Waiting a moment and trying again\n- Checking if the backend is responding"
        elif "JSON" in error_str or "json" in error_str:
            formatted_message = "📄 **Response Format Error**"
            formatted_details = "Received an invalid response from the backend.\n\n**Please try:**\n- Refreshing the page\n- Asking the question again\n- Checking backend logs for errors"
        else:
            formatted_message = f"❌ **Processing Error: {error_str}**"
            formatted_details = "An unexpected error occurred while processing your question.\n\n**Please try:**\n- Asking the question again\n- Refreshing the page\n- Contacting support if the issue persists"
        
        return {
            "error": True,
            "message": formatted_message,
            "details": formatted_details
        }
    
    def get_suggested_questions(self) -> list:
        """
        Get suggested questions for users when they ask irrelevant questions
        
        Returns:
            List of suggested questions
        """
        return [
            "How does the daily fraud rate fluctuate over the two-year period?",
            "Which merchants exhibit the highest incidence of fraudulent transactions?",
            "What are the primary methods by which credit card fraud is committed?",
            "What are the core components of an effective fraud detection system?",
            "How much higher are fraud rates when the transaction counterpart is located outside the EEA?",
            "What share of total card fraud value in H1 2023 was due to cross-border transactions?"
        ]
