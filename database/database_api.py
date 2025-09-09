"""
Database API for Fraud Detection Chatbot
Exposes SQLite database through REST API endpoints
"""
import os
import sys
import sqlite3
import pandas as pd
from pathlib import Path
import logging
from flask import Flask, jsonify, request
from flask_cors import CORS
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Database configuration
DB_PATH = "/app/data/fraud_data.db"

class DatabaseAPI:
    """Database API handler"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
    
    def connect(self):
        """Connect to SQLite database"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row  # Enable column access by name
            logger.info(f"Connected to database: {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def execute_query(self, sql: str, params: tuple = None):
        """Execute SQL query and return results"""
        try:
            # Create a new connection for this thread
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            
            # Fetch results
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            results = []
            for row in rows:
                results.append(dict(row))
            
            conn.close()
            return results, None
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return None, str(e)
    
    def get_health(self):
        """Check database health"""
        try:
            # Create a new connection for this thread
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM transactions LIMIT 1")
            result = cursor.fetchone()
            conn.close()
            
            if result and result['count'] is not None:
                return True, f"Database healthy with {result['count']} records"
            else:
                return False, "Database empty or corrupted"
                
        except Exception as e:
            return False, f"Health check failed: {e}"

# Initialize database API
db_api = DatabaseAPI(DB_PATH)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    is_healthy, message = db_api.get_health()
    
    if is_healthy:
        return jsonify({
            'status': 'healthy',
            'message': message,
            'service': 'database-api'
        }), 200
    else:
        return jsonify({
            'status': 'unhealthy',
            'message': message,
            'service': 'database-api'
        }), 503

@app.route('/query', methods=['POST'])
def execute_query():
    """Execute SQL query endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'sql' not in data:
            return jsonify({
                'error': 'Missing SQL query in request body',
                'example': {'sql': 'SELECT COUNT(*) as count FROM transactions'}
            }), 400
        
        sql = data['sql']
        params = data.get('params', None)
        
        # Basic SQL injection protection
        if not sql.strip().upper().startswith(('SELECT', 'WITH')):
            return jsonify({
                'error': 'Only SELECT queries are allowed'
            }), 400
        
        logger.info(f"Executing query: {sql}")
        if params:
            logger.info(f"With parameters: {params}")
        
        results, error = db_api.execute_query(sql, tuple(params) if params else None)
        
        if error:
            return jsonify({
                'error': error,
                'sql': sql,
                'params': params
            }), 500
        
        return jsonify({
            'success': True,
            'data': results,
            'count': len(results),
            'sql': sql,
            'params': params
        }), 200
        
    except Exception as e:
        logger.error(f"Query endpoint error: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/data/availability', methods=['POST'])
def check_data_availability():
    """Check data availability for specific time periods"""
    try:
        data = request.get_json()
        
        if not data or 'start_date' not in data or 'end_date' not in data:
            return jsonify({
                'error': 'Missing start_date and end_date in request body',
                'example': {'start_date': '2023-01-01', 'end_date': '2023-07-01'}
            }), 400
        
        start_date = data['start_date']
        end_date = data['end_date']
        
        sql = """
        SELECT COUNT(*) as count
        FROM transactions
        WHERE trans_date_trans_time >= ? AND trans_date_trans_time < ?
        """
        
        results, error = db_api.execute_query(sql, (start_date, end_date))
        
        if error:
            return jsonify({
                'error': error
            }), 500
        
        count = results[0]['count'] if results else 0
        available = count > 0
        
        return jsonify({
            'success': True,
            'available': available,
            'count': count,
            'start_date': start_date,
            'end_date': end_date
        }), 200
        
    except Exception as e:
        logger.error(f"Data availability check error: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/stats', methods=['GET'])
def get_database_stats():
    """Get database statistics"""
    try:
        stats = {}
        
        # Total records
        results, error = db_api.execute_query("SELECT COUNT(*) as total FROM transactions")
        if not error and results:
            stats['total_records'] = results[0]['total']
        
        # Fraud records
        results, error = db_api.execute_query("SELECT COUNT(*) as fraud_count FROM transactions WHERE is_fraud = 1")
        if not error and results:
            stats['fraud_records'] = results[0]['fraud_count']
        
        # Date range
        results, error = db_api.execute_query("""
            SELECT 
                MIN(trans_date_trans_time) as min_date,
                MAX(trans_date_trans_time) as max_date
            FROM transactions
        """)
        if not error and results:
            stats['date_range'] = {
                'min_date': results[0]['min_date'],
                'max_date': results[0]['max_date']
            }
        
        # Fraud rate
        if 'total_records' in stats and 'fraud_records' in stats and stats['total_records'] > 0:
            stats['fraud_rate'] = (stats['fraud_records'] / stats['total_records']) * 100
        
        return jsonify({
            'success': True,
            'stats': stats
        }), 200
        
    except Exception as e:
        logger.error(f"Stats endpoint error: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        'service': 'Fraud Detection Database API',
        'version': '1.0.0',
        'endpoints': {
            'health': 'GET /health',
            'query': 'POST /query',
            'data_availability': 'POST /data/availability',
            'stats': 'GET /stats'
        },
        'example_query': {
            'sql': 'SELECT COUNT(*) as count FROM transactions WHERE is_fraud = 1',
            'params': None
        }
    })

if __name__ == '__main__':
    logger.info("Starting Fraud Detection Database API...")
    
    # Connect to database
    if db_api.connect():
        logger.info("✅ Database API ready!")
        app.run(host='0.0.0.0', port=5432, debug=False)
    else:
        logger.error("❌ Failed to connect to database. Exiting.")
        sys.exit(1)
