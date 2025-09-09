"""
Debug script for value analysis query
"""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.data.database import DatabaseManager

def debug_value_analysis_query():
    """Debug the value analysis query"""
    print("🔍 Debugging Value Analysis Query")
    print("=" * 50)
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager(db_path="fraud_data.db")
        
        if not db_manager.connect():
            print("❌ Failed to connect to database")
            return False
        
        if not db_manager.load_csv_data():
            print("❌ Failed to load CSV data")
            return False
        
        print("✅ Database connected and data loaded")
        
        # Test the problematic query
        query = """
        WITH h1_2023_fraud AS (
            SELECT 
                CASE 
                    WHEN amt > 100 THEN 'Cross-border (High-value proxy)'
                    ELSE 'Domestic (Low-value proxy)'
                END as transaction_type,
                SUM(amt * is_fraud) as fraud_value,
                COUNT(*) as total_transactions,
                SUM(is_fraud) as fraud_count
            FROM transactions
            WHERE trans_date_trans_time >= '2023-01-01' 
                AND trans_date_trans_time < '2023-07-01'
                AND is_fraud = 1
            GROUP BY 
                CASE 
                    WHEN amt > 100 THEN 'Cross-border (High-value proxy)'
                    ELSE 'Domestic (Low-value proxy)'
                END
        ),
        total_fraud AS (
            SELECT SUM(fraud_value) as total_fraud_value
            FROM h1_2023_fraud
        )
        SELECT 
            h.transaction_type,
            h.fraud_value,
            h.fraud_count,
            ROUND((h.fraud_value * 100.0 / t.total_fraud_value), 2) as percentage_share,
            t.total_fraud_value
        FROM h1_2023_fraud h
        CROSS JOIN total_fraud t
        ORDER BY h.fraud_value DESC
        """
        
        print("🔍 Executing value analysis query...")
        success, data, error = db_manager.execute_query(query)
        
        if success:
            print(f"✅ Query executed successfully!")
            print(f"📊 Rows returned: {len(data)}")
            if len(data) > 0:
                print(f"📈 Data:")
                print(data)
            else:
                print("⚠️ No data returned - this might be the issue!")
                
                # Let's check if there's any H1 2023 data at all
                print("\n🔍 Checking for H1 2023 data...")
                check_query = """
                SELECT 
                    COUNT(*) as total_records,
                    MIN(trans_date_trans_time) as min_date,
                    MAX(trans_date_trans_time) as max_date,
                    SUM(is_fraud) as total_fraud
                FROM transactions
                WHERE trans_date_trans_time >= '2023-01-01' 
                    AND trans_date_trans_time < '2023-07-01'
                """
                
                success2, check_data, error2 = db_manager.execute_query(check_query)
                if success2:
                    print(f"📊 H1 2023 data check:")
                    print(check_data)
                else:
                    print(f"❌ Check query failed: {error2}")
        else:
            print(f"❌ Query failed: {error}")
        
        return success
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    debug_value_analysis_query()
