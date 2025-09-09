"""
Database Connection and Query Execution
Handles CSV data loading and SQL query execution
"""
import pandas as pd
import sqlite3
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging


class DatabaseManager:
    """
    Manages database connections and query execution
    """
    
    def __init__(self, data_dir: str = "dataset", db_path: str = "fraud_data.db"):
        # If running from backend directory, adjust path to parent
        if Path.cwd().name == "backend":
            self.data_dir = Path("..") / data_dir
        else:
            self.data_dir = Path(data_dir)
        self.db_path = db_path
        self.connection = None
        self.tables_loaded = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> bool:
        """
        Establish database connection
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Use check_same_thread=False to allow multi-threading
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row  # Enable column access by name
            self.logger.info(f"Connected to database: {self.db_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.logger.info("Database connection closed")
    
    def load_csv_data(self) -> bool:
        """
        Load CSV data into SQLite database
        
        Returns:
            True if successful, False otherwise
        """
        if not self.connection:
            self.logger.error("No database connection")
            return False
        
        try:
            # Load training data
            train_file = self.data_dir / "archive" / "fraudTrain.csv"
            if train_file.exists():
                df_train = pd.read_csv(train_file)
                df_train.to_sql('transactions', self.connection, if_exists='replace', index=False)
                self.logger.info(f"Loaded {len(df_train)} training records")
            else:
                self.logger.warning(f"Training file not found: {train_file}")
                return False
            
            # Load test data and append to transactions table
            test_file = self.data_dir / "archive" / "fraudTest.csv"
            if test_file.exists():
                df_test = pd.read_csv(test_file)
                df_test.to_sql('transactions', self.connection, if_exists='append', index=False)
                self.logger.info(f"Loaded {len(df_test)} test records")
            else:
                self.logger.warning(f"Test file not found: {test_file}")
            
            # Create indexes for better performance
            self._create_indexes()
            
            self.tables_loaded = True
            self.logger.info("CSV data loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load CSV data: {e}")
            return False
    
    def _create_indexes(self):
        """Create database indexes for better query performance"""
        try:
            cursor = self.connection.cursor()
            
            # First, check what columns actually exist in the table
            cursor.execute("PRAGMA table_info(transactions)")
            columns = [row[1] for row in cursor.fetchall()]
            self.logger.debug(f"Available columns: {columns}")
            
            # Create indexes only on columns that actually exist
            potential_indexes = [
                ("trans_date_trans_time", "idx_trans_date_trans_time"),
                ("is_fraud", "idx_is_fraud"),
                ("merchant", "idx_merchant"),
                ("category", "idx_category"),
                ("amt", "idx_amt")
            ]
            
            indexes_to_create = []
            for column, index_name in potential_indexes:
                if column in columns:
                    indexes_to_create.append(f"CREATE INDEX IF NOT EXISTS {index_name} ON transactions({column})")
                else:
                    self.logger.warning(f"Column '{column}' not found in table, skipping index creation")
            
            # Create the indexes
            for index_sql in indexes_to_create:
                try:
                    cursor.execute(index_sql)
                    self.logger.debug(f"Created index: {index_sql}")
                except Exception as index_error:
                    self.logger.warning(f"Failed to create index '{index_sql}': {index_error}")
                    # Continue with other indexes even if one fails
            
            self.connection.commit()
            self.logger.info(f"Database indexes created successfully ({len(indexes_to_create)} indexes)")
            
        except Exception as e:
            self.logger.error(f"Failed to create indexes: {e}")
    
    def execute_query(self, sql: str, params: Optional[Tuple] = None) -> Tuple[bool, Optional[pd.DataFrame], str]:
        """
        Execute SQL query and return results
        
        Args:
            sql: SQL query to execute
            params: Optional parameters for prepared statement
            
        Returns:
            Tuple of (success, dataframe, error_message)
        """
        if not self.connection:
            return False, None, "No database connection"
        
        if not self.tables_loaded:
            return False, None, "Data not loaded. Call load_csv_data() first."
        
        try:
            # Debug logging
            print(f"\n=== DATABASE EXECUTION DEBUG ===")
            print(f"SQL Query: {sql}")
            print(f"Parameters: {params}")
            
            # Execute query
            if params:
                df = pd.read_sql_query(sql, self.connection, params=params)
            else:
                df = pd.read_sql_query(sql, self.connection)
            
            print(f"Query executed successfully, returned {len(df)} rows")
            if len(df) > 0:
                print(f"Data preview: {df.head().to_dict()}")
            else:
                print("No data returned from query")
            print("=== END DATABASE EXECUTION DEBUG ===\n")
            
            self.logger.info(f"Query executed successfully, returned {len(df)} rows")
            return True, df, None
            
        except Exception as e:
            error_msg = f"Query execution failed: {str(e)}"
            self.logger.error(error_msg)
            return False, None, error_msg
    
    def get_table_info(self, table_name: str = "transactions") -> Dict[str, Any]:
        """
        Get information about a table
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with table information
        """
        if not self.connection:
            return {"error": "No database connection"}
        
        try:
            # Get table schema
            cursor = self.connection.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            
            # Get sample data
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
            sample_data = cursor.fetchall()
            
            return {
                "table_name": table_name,
                "columns": [{"name": col[1], "type": col[2], "nullable": not col[3]} for col in columns],
                "row_count": row_count,
                "sample_data": sample_data
            }
            
        except Exception as e:
            return {"error": f"Failed to get table info: {str(e)}"}
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the loaded data
        
        Returns:
            Dictionary with data summary
        """
        if not self.connection or not self.tables_loaded:
            return {"error": "Data not loaded"}
        
        try:
            summary = {}
            
            # Basic counts
            cursor = self.connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM transactions")
            summary["total_transactions"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud = 1")
            summary["fraudulent_transactions"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud = 0")
            summary["legitimate_transactions"] = cursor.fetchone()[0]
            
            # Fraud rate
            summary["fraud_rate"] = summary["fraudulent_transactions"] / summary["total_transactions"]
            
            # Date range
            cursor.execute("SELECT MIN(trans_date), MAX(trans_date) FROM transactions")
            date_range = cursor.fetchone()
            summary["date_range"] = {
                "start": date_range[0],
                "end": date_range[1]
            }
            
            # Amount statistics
            cursor.execute("SELECT MIN(amount), MAX(amount), AVG(amount) FROM transactions")
            amount_stats = cursor.fetchone()
            summary["amount_stats"] = {
                "min": amount_stats[0],
                "max": amount_stats[1],
                "avg": amount_stats[2]
            }
            
            # Unique merchants and categories
            cursor.execute("SELECT COUNT(DISTINCT merchant) FROM transactions")
            summary["unique_merchants"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT category) FROM transactions")
            summary["unique_categories"] = cursor.fetchone()[0]
            
            return summary
            
        except Exception as e:
            return {"error": f"Failed to get data summary: {str(e)}"}
    
    def validate_data_quality(self) -> Dict[str, Any]:
        """
        Validate data quality and identify potential issues
        
        Returns:
            Dictionary with data quality report
        """
        if not self.connection or not self.tables_loaded:
            return {"error": "Data not loaded"}
        
        try:
            quality_report = {
                "issues": [],
                "warnings": [],
                "recommendations": []
            }
            
            cursor = self.connection.cursor()
            
            # Check for null values in critical columns
            critical_columns = ["trans_date", "amount", "is_fraud", "merchant"]
            for col in critical_columns:
                cursor.execute(f"SELECT COUNT(*) FROM transactions WHERE {col} IS NULL")
                null_count = cursor.fetchone()[0]
                if null_count > 0:
                    quality_report["issues"].append(f"Column '{col}' has {null_count} null values")
            
            # Check for duplicate transactions
            cursor.execute("""
                SELECT COUNT(*) FROM (
                    SELECT * FROM transactions 
                    GROUP BY trans_date, amount, merchant, category 
                    HAVING COUNT(*) > 1
                )
            """)
            duplicate_count = cursor.fetchone()[0]
            if duplicate_count > 0:
                quality_report["warnings"].append(f"Found {duplicate_count} potential duplicate transactions")
            
            # Check for unrealistic amounts
            cursor.execute("SELECT COUNT(*) FROM transactions WHERE amount <= 0")
            invalid_amounts = cursor.fetchone()[0]
            if invalid_amounts > 0:
                quality_report["issues"].append(f"Found {invalid_amounts} transactions with invalid amounts")
            
            # Check for future dates
            cursor.execute("SELECT COUNT(*) FROM transactions WHERE trans_date > date('now')")
            future_dates = cursor.fetchone()[0]
            if future_dates > 0:
                quality_report["warnings"].append(f"Found {future_dates} transactions with future dates")
            
            # Check fraud rate
            cursor.execute("SELECT AVG(is_fraud) FROM transactions")
            fraud_rate = cursor.fetchone()[0]
            if fraud_rate < 0.01:
                quality_report["warnings"].append(f"Very low fraud rate ({fraud_rate:.2%}) - check data quality")
            elif fraud_rate > 0.5:
                quality_report["warnings"].append(f"Very high fraud rate ({fraud_rate:.2%}) - check data quality")
            
            return quality_report
            
        except Exception as e:
            return {"error": f"Failed to validate data quality: {str(e)}"}
    
    def backup_database(self, backup_path: str) -> bool:
        """
        Create a backup of the database
        
        Args:
            backup_path: Path to save the backup
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            self.logger.info(f"Database backed up to: {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to backup database: {e}")
            return False
