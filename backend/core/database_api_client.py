"""
Database API Client for Fraud Detection Chatbot
Communicates with the database container via REST API
"""
import requests
import logging
from typing import Dict, List, Optional, Tuple, Any
import json

logger = logging.getLogger(__name__)

class DatabaseAPIClient:
    """
    Client for communicating with the database API
    """
    
    def __init__(self, database_api_url: str = "http://fraud-database:5432"):
        """
        Initialize database API client
        
        Args:
            database_api_url: URL of the database API service
        """
        self.database_api_url = database_api_url.rstrip('/')
        self.logger = logging.getLogger(__name__)
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Tuple[bool, Any, str]:
        """
        Make HTTP request to database API
        
        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint
            data: Request data for POST requests
            
        Returns:
            Tuple of (success, response_data, error_message)
        """
        try:
            url = f"{self.database_api_url}{endpoint}"
            
            if method.upper() == 'GET':
                response = requests.get(url, timeout=30)
            elif method.upper() == 'POST':
                response = requests.post(url, json=data, timeout=30)
            else:
                return False, None, f"Unsupported HTTP method: {method}"
            
            if response.status_code == 200:
                return True, response.json(), None
            else:
                error_msg = f"API request failed with status {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f": {error_data.get('error', 'Unknown error')}"
                except:
                    error_msg += f": {response.text}"
                return False, None, error_msg
                
        except requests.exceptions.ConnectionError:
            return False, None, f"Cannot connect to database API at {self.database_api_url}"
        except requests.exceptions.Timeout:
            return False, None, "Database API request timed out"
        except Exception as e:
            return False, None, f"Database API request failed: {str(e)}"
    
    def health_check(self) -> Tuple[bool, str]:
        """
        Check database API health
        
        Returns:
            Tuple of (is_healthy, message)
        """
        success, data, error = self._make_request('GET', '/health')
        
        if success and data:
            is_healthy = data.get('status') == 'healthy'
            message = data.get('message', 'Unknown status')
            return is_healthy, message
        else:
            return False, error or "Health check failed"
    
    def execute_query(self, sql: str, params: Optional[Tuple] = None) -> Tuple[bool, Optional[List[Dict]], str]:
        """
        Execute SQL query via database API
        
        Args:
            sql: SQL query to execute
            params: Optional parameters for prepared statement
            
        Returns:
            Tuple of (success, results, error_message)
        """
        request_data = {
            'sql': sql
        }
        
        if params:
            request_data['params'] = list(params)
        
        success, data, error = self._make_request('POST', '/query', request_data)
        
        if success and data:
            if data.get('success', False):
                return True, data.get('data', []), None
            else:
                return False, None, data.get('error', 'Query execution failed')
        else:
            return False, None, error or "Query execution failed"
    
    def check_data_availability(self, start_date: str, end_date: str) -> Tuple[bool, bool, int, str]:
        """
        Check if data is available for a specific time period
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Tuple of (success, is_available, count, error_message)
        """
        request_data = {
            'start_date': start_date,
            'end_date': end_date
        }
        
        success, data, error = self._make_request('POST', '/data/availability', request_data)
        
        if success and data:
            if data.get('success', False):
                is_available = data.get('available', False)
                count = data.get('count', 0)
                return True, is_available, count, None
            else:
                return False, False, 0, data.get('error', 'Data availability check failed')
        else:
            return False, False, 0, error or "Data availability check failed"
    
    def get_database_stats(self) -> Tuple[bool, Optional[Dict], str]:
        """
        Get database statistics
        
        Returns:
            Tuple of (success, stats, error_message)
        """
        success, data, error = self._make_request('GET', '/stats')
        
        if success and data:
            if data.get('success', False):
                return True, data.get('stats', {}), None
            else:
                return False, None, data.get('error', 'Stats retrieval failed')
        else:
            return False, None, error or "Stats retrieval failed"
    
    def test_connection(self) -> bool:
        """
        Test connection to database API
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            is_healthy, message = self.health_check()
            if is_healthy:
                self.logger.info(f"Database API connection successful: {message}")
                return True
            else:
                self.logger.warning(f"Database API connection failed: {message}")
                return False
        except Exception as e:
            self.logger.error(f"Database API connection test failed: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Test the database API client
    client = DatabaseAPIClient("http://localhost:5432")
    
    print("Testing Database API Client...")
    
    # Test health check
    is_healthy, message = client.health_check()
    print(f"Health check: {is_healthy} - {message}")
    
    # Test query execution
    success, results, error = client.execute_query("SELECT COUNT(*) as count FROM transactions LIMIT 1")
    if success:
        print(f"Query successful: {results}")
    else:
        print(f"Query failed: {error}")
    
    # Test data availability
    success, available, count, error = client.check_data_availability("2023-01-01", "2023-07-01")
    if success:
        print(f"Data availability: {available} (count: {count})")
    else:
        print(f"Data availability check failed: {error}")
