"""
Database Initialization Script for Docker
Prepares the database with CSV data for the fraud detection chatbot
"""
import os
import sys
import sqlite3
import pandas as pd
from pathlib import Path
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_database():
    """Initialize the database with CSV data"""
    
    # Database path
    db_path = "/app/data/fraud_data.db"
    data_dir = Path("/app/data")
    
    # Ensure data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting database initialization...")
    
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(db_path)
        logger.info(f"Connected to database: {db_path}")
        
        # Check if data already exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='transactions'")
        if cursor.fetchone():
            logger.info("Database already initialized. Skipping data load.")
            conn.close()
            return True
        
        # Load training data - check multiple possible locations
        train_locations = [
            data_dir / "archive" / "fraudTrain.csv",  # Original location
            data_dir / "fraudTrain.csv",              # Direct location
            Path("/app/dataset/archive/fraudTrain.csv"),  # Mounted dataset location
            Path("/app/dataset/fraudTrain.csv")       # Alternative mounted location
        ]
        
        train_file = None
        for location in train_locations:
            if location.exists():
                train_file = location
                break
        
        if train_file and train_file.exists():
            logger.info(f"Loading training data from {train_file}")
            df_train = pd.read_csv(train_file)
            df_train.to_sql('transactions', conn, if_exists='replace', index=False)
            logger.info(f"Loaded {len(df_train)} training records")
        else:
            logger.warning(f"Training file not found in any of these locations: {train_locations}")
            # Create empty table structure
            create_empty_table(conn)
            return True
        
        # Load test data and append to transactions table
        test_locations = [
            data_dir / "archive" / "fraudTest.csv",   # Original location
            data_dir / "fraudTest.csv",               # Direct location
            Path("/app/dataset/archive/fraudTest.csv"),  # Mounted dataset location
            Path("/app/dataset/fraudTest.csv")        # Alternative mounted location
        ]
        
        test_file = None
        for location in test_locations:
            if location.exists():
                test_file = location
                break
        
        if test_file and test_file.exists():
            logger.info(f"Loading test data from {test_file}")
            df_test = pd.read_csv(test_file)
            df_test.to_sql('transactions', conn, if_exists='append', index=False)
            logger.info(f"Loaded {len(df_test)} test records")
        else:
            logger.warning(f"Test file not found in any of these locations: {test_locations}")
        
        # Create indexes for better performance
        create_indexes(conn)
        
        # Get final statistics
        cursor.execute("SELECT COUNT(*) FROM transactions")
        total_records = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud = 1")
        fraud_records = cursor.fetchone()[0]
        
        fraud_rate = (fraud_records / total_records * 100) if total_records > 0 else 0
        
        logger.info(f"Database initialization completed successfully!")
        logger.info(f"Total records: {total_records:,}")
        logger.info(f"Fraud records: {fraud_records:,}")
        logger.info(f"Fraud rate: {fraud_rate:.2f}%")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False

def create_empty_table(conn):
    """Create empty table structure if CSV files are not available"""
    logger.info("Creating empty table structure...")
    
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY,
            trans_date_trans_time TEXT,
            cc_num TEXT,
            merchant TEXT,
            category TEXT,
            amt REAL,
            first TEXT,
            last TEXT,
            gender TEXT,
            street TEXT,
            city TEXT,
            state TEXT,
            zip TEXT,
            lat REAL,
            long REAL,
            city_pop INTEGER,
            job TEXT,
            dob TEXT,
            trans_num TEXT,
            unix_time INTEGER,
            merch_lat REAL,
            merch_long REAL,
            is_fraud INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    logger.info("Empty table structure created")

def create_indexes(conn):
    """Create database indexes for better performance"""
    logger.info("Creating database indexes...")
    
    cursor = conn.cursor()
    
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_trans_date_trans_time ON transactions(trans_date_trans_time)",
        "CREATE INDEX IF NOT EXISTS idx_is_fraud ON transactions(is_fraud)",
        "CREATE INDEX IF NOT EXISTS idx_merchant ON transactions(merchant)",
        "CREATE INDEX IF NOT EXISTS idx_category ON transactions(category)",
        "CREATE INDEX IF NOT EXISTS idx_amt ON transactions(amt)",
        "CREATE INDEX IF NOT EXISTS idx_cc_num ON transactions(cc_num)",
        "CREATE INDEX IF NOT EXISTS idx_city ON transactions(city)",
        "CREATE INDEX IF NOT EXISTS idx_state ON transactions(state)"
    ]
    
    for index_sql in indexes:
        cursor.execute(index_sql)
    
    conn.commit()
    logger.info("Database indexes created")

def wait_for_data_files():
    """Wait for data files to be available (for volume mounting)"""
    logger.info("Waiting for data files to be available...")
    
    max_wait_time = 300  # 5 minutes
    wait_interval = 10   # 10 seconds
    waited = 0
    
    while waited < max_wait_time:
        # Check multiple possible locations for data files
        data_locations = [
            Path("/app/data/archive/fraudTrain.csv"),     # Original location
            Path("/app/data/fraudTrain.csv"),             # Direct location
            Path("/app/dataset/archive/fraudTrain.csv"),  # Mounted dataset location
            Path("/app/dataset/fraudTrain.csv")           # Alternative mounted location
        ]
        
        for location in data_locations:
            if location.exists():
                logger.info(f"Data files found at {location}!")
                return True
        
        logger.info(f"Waiting for data files... ({waited}s/{max_wait_time}s)")
        time.sleep(wait_interval)
        waited += wait_interval
    
    logger.warning("Data files not found within timeout period")
    return False

def main():
    """Main initialization function"""
    logger.info("Fraud Detection Database Initialization")
    logger.info("=" * 50)
    
    # Wait for data files if they're not immediately available
    if not wait_for_data_files():
        logger.warning("Proceeding with empty database structure")
    
    # Initialize database
    success = init_database()
    
    if success:
        logger.info("✅ Database initialization completed successfully!")
        return True
    else:
        logger.error("❌ Database initialization failed!")
        return False

if __name__ == "__main__":
    main()
