"""
API Client for communicating with the fraud detection backend
"""
import requests
import json
from typing import Dict, Any, Optional
import streamlit as st


class FraudAPIClient:
    """Client for communicating with the fraud detection backend API"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def health_check(self) -> bool:
        """Check if the backend API is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Send a question to the backend API and return the response
        
        Args:
            question: The user's question about fraud data
            
        Returns:
            Dictionary containing the API response or error information
        """
        try:
            payload = {"question": question}
            response = self.session.post(
                f"{self.base_url}/question",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"API Error: {response.status_code}",
                    "message": response.text,
                    "status": "error"
                }
                
        except requests.exceptions.Timeout:
            return {
                "error": "Request Timeout",
                "message": "The request took too long to process. Please try again.",
                "status": "error"
            }
        except requests.exceptions.ConnectionError:
            return {
                "error": "Connection Error",
                "message": "Could not connect to the backend API. Please ensure the backend is running.",
                "status": "error"
            }
        except requests.exceptions.RequestException as e:
            return {
                "error": "Request Error",
                "message": f"An error occurred while making the request: {str(e)}",
                "status": "error"
            }
        except json.JSONDecodeError:
            return {
                "error": "Invalid Response",
                "message": "The backend returned an invalid response format.",
                "status": "error"
            }
        except Exception as e:
            return {
                "error": "Unexpected Error",
                "message": f"An unexpected error occurred: {str(e)}",
                "status": "error"
            }
    
    def get_suggestions(self) -> Dict[str, Any]:
        """Get suggested questions from the backend"""
        try:
            response = self.session.get(f"{self.base_url}/suggestions", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"suggestions": []}
        except requests.exceptions.RequestException:
            return {"suggestions": []}
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get the current status of the backend service"""
        try:
            response = self.session.get(f"{self.base_url}/status", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"initialized": False}
        except requests.exceptions.RequestException:
            return {"initialized": False}


def get_api_client() -> FraudAPIClient:
    """Get or create the API client instance"""
    if 'api_client' not in st.session_state:
        st.session_state.api_client = FraudAPIClient()
    return st.session_state.api_client
