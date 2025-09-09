"""
API-Based Database Manager for Fraud Detection Chatbot
Handles database operations via REST API calls to database container
"""
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging
import os
from core.database_api_client import DatabaseAPIClient

class APIDatabaseManager:
    """
    Manages database operations via API calls
    """
    
    def __init__(self, database_api_url: str = None):
        """
        Initialize API-based database manager
        
        Args:
            database_api_url: URL of the database API service
        """
        # Get database API URL from environment or use default
        self.database_api_url = database_api_url or os.getenv('DATABASE_API_URL', 'http://fraud-database:5432')
        
        # Initialize API client
        self.api_client = DatabaseAPIClient(self.database_api_url)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Connection status
        self.connected = False
        self.tables_loaded = True  # API-based, so data is always "loaded"
    
    @property
    def connection(self):
        """Compatibility property for connection status"""
        return self.api_client if self.connected else None
    
    def connect(self) -> bool:
        """
        Connect to database API
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Connecting to database API at {self.database_api_url}")
            
            # Test connection
            if self.api_client.test_connection():
                self.connected = True
                self.logger.info("Database API connection established")
                return True
            else:
                self.logger.error("Failed to connect to database API")
                return False
                
        except Exception as e:
            self.logger.error(f"Database API connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from database API"""
        self.connected = False
        self.logger.info("Disconnected from database API")
    
    def load_csv_data(self) -> bool:
        """
        Check if data is available (API-based, so data is always loaded)
        
        Returns:
            True if data is available, False otherwise
        """
        try:
            # Check if we can query the database
            success, results, error = self.api_client.execute_query("SELECT COUNT(*) as count FROM transactions LIMIT 1")
            
            if success and results:
                count = results[0]['count']
                self.logger.info(f"Database API data check: {count:,} records available")
                return True
            else:
                self.logger.error(f"Database API data check failed: {error}")
                return False
                
        except Exception as e:
            self.logger.error(f"Database API data check error: {e}")
            return False
    
    def execute_query(self, sql: str, params: Optional[Tuple] = None) -> Tuple[bool, Optional[pd.DataFrame], str]:
        """
        Execute SQL query via database API
        
        Args:
            sql: SQL query to execute
            params: Optional parameters for prepared statement
            
        Returns:
            Tuple of (success, dataframe, error_message)
        """
        if not self.connected:
            return False, None, "No database API connection"
        
        try:
            # Debug logging
            print(f"\n=== DATABASE API EXECUTION DEBUG ===")
            print(f"SQL Query: {sql}")
            print(f"Parameters: {params}")
            
            # Execute query via API
            success, results, error = self.api_client.execute_query(sql, params)
            
            if success and results is not None:
                # Convert results to DataFrame
                df = pd.DataFrame(results)
                
                print(f"Query executed successfully, returned {len(df)} rows")
                if len(df) > 0:
                    print(f"Data preview: {df.head().to_dict()}")
                print(f"=== END DATABASE API EXECUTION DEBUG ===")
                
                return True, df, None
            else:
                print(f"Query failed: {error}")
                print(f"=== END DATABASE API EXECUTION DEBUG ===")
                return False, None, error or "Query execution failed"
                
        except Exception as e:
            error_msg = f"Database API query execution failed: {str(e)}"
            self.logger.error(error_msg)
            return False, None, error_msg
    
    def check_data_availability(self, start_date: str, end_date: str) -> Tuple[bool, bool, int, str]:
        """
        Check if data is available for a specific time period
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Tuple of (success, is_available, count, error_message)
        """
        if not self.connected:
            return False, False, 0, "No database API connection"
        
        try:
            return self.api_client.check_data_availability(start_date, end_date)
        except Exception as e:
            error_msg = f"Data availability check failed: {str(e)}"
            self.logger.error(error_msg)
            return False, False, 0, error_msg
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of loaded data
        
        Returns:
            Dictionary with data summary
        """
        if not self.connected:
            return {'error': 'No database API connection'}
        
        try:
            success, stats, error = self.api_client.get_database_stats()
            
            if success and stats:
                return {
                    'total_records': stats.get('total_records', 0),
                    'fraud_records': stats.get('fraud_records', 0),
                    'fraud_rate': stats.get('fraud_rate', 0),
                    'date_range': stats.get('date_range', {}),
                    'connection_type': 'API',
                    'api_url': self.database_api_url
                }
            else:
                return {'error': error or 'Failed to get database stats'}
                
        except Exception as e:
            return {'error': f'Database stats retrieval failed: {str(e)}'}
    
    def health_check(self) -> Tuple[bool, str]:
        """
        Check database API health
        
        Returns:
            Tuple of (is_healthy, message)
        """
        try:
            return self.api_client.health_check()
        except Exception as e:
            return False, f"Health check failed: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Test the API database manager
    db_manager = APIDatabaseManager("http://localhost:5432")
    
    print("Testing API Database Manager...")
    
    # Connect
    if db_manager.connect():
        print("✅ Connected to database API")
        
        # Test query
        success, df, error = db_manager.execute_query("SELECT COUNT(*) as count FROM transactions LIMIT 1")
        if success:
            print(f"✅ Query successful: {df.iloc[0]['count']} records")
        else:
            print(f"❌ Query failed: {error}")
        
        # Test data availability
        success, available, count, error = db_manager.check_data_availability("2023-01-01", "2023-07-01")
        if success:
            print(f"✅ Data availability check: {available} (count: {count})")
        else:
            print(f"❌ Data availability check failed: {error}")
        
        # Get stats
        stats = db_manager.get_data_summary()
        print(f"✅ Database stats: {stats}")
        
    else:
        print("❌ Failed to connect to database API")
