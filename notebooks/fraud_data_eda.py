"""
Fraud Data Exploratory Data Analysis (EDA)
==========================================

This notebook performs comprehensive EDA on the fraud dataset to understand:
1. Data structure and schema
2. Data quality and missing values
3. Fraud patterns and distributions
4. Temporal analysis
5. Geographic analysis (if available)
6. Value analysis for H1 2023

Run this notebook to understand the data before fixing backend issues.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sqlite3
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load fraud data from CSV files"""
    print("=== LOADING FRAUD DATA ===")
    
    # Define paths
    data_dir = Path("../dataset")
    train_file = data_dir / "archive" / "fraudTrain.csv"
    test_file = data_dir / "archive" / "fraudTest.csv"
    
    print(f"Data directory: {data_dir}")
    print(f"Train file exists: {train_file.exists()}")
    print(f"Test file exists: {test_file.exists()}")
    
    # Load data
    if train_file.exists():
        train_df = pd.read_csv(train_file)
        print(f"Train data shape: {train_df.shape}")
    else:
        print("Train file not found!")
        return None, None
    
    if test_file.exists():
        test_df = pd.read_csv(test_file)
        print(f"Test data shape: {test_df.shape}")
    else:
        print("Test file not found!")
        return train_df, None
    
    return train_df, test_df

def basic_data_info(df, name="Dataset"):
    """Display basic information about the dataset"""
    print(f"\n=== {name.upper()} BASIC INFO ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nFirst few rows:")
    print(df.head())

def analyze_fraud_distribution(df, name="Dataset"):
    """Analyze fraud distribution"""
    print(f"\n=== {name.upper()} FRAUD DISTRIBUTION ===")
    
    # Basic fraud stats
    fraud_count = df['is_fraud'].sum()
    total_count = len(df)
    fraud_rate = fraud_count / total_count
    
    print(f"Total transactions: {total_count:,}")
    print(f"Fraud transactions: {fraud_count:,}")
    print(f"Fraud rate: {fraud_rate:.4f} ({fraud_rate*100:.2f}%)")
    
    # Plot fraud distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    df['is_fraud'].value_counts().plot(kind='bar')
    plt.title(f'{name} - Fraud Distribution')
    plt.xlabel('Is Fraud (0=No, 1=Yes)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    plt.subplot(1, 2, 2)
    df['is_fraud'].value_counts(normalize=True).plot(kind='bar')
    plt.title(f'{name} - Fraud Rate')
    plt.xlabel('Is Fraud (0=No, 1=Yes)')
    plt.ylabel('Proportion')
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.show()

def analyze_temporal_patterns(df, name="Dataset"):
    """Analyze temporal patterns in the data"""
    print(f"\n=== {name.upper()} TEMPORAL ANALYSIS ===")
    
    # Convert date columns
    if 'trans_date_trans_time' in df.columns:
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['date'] = df['trans_date_trans_time'].dt.date
        df['hour'] = df['trans_date_trans_time'].dt.hour
        df['month'] = df['trans_date_trans_time'].dt.month
        df['year'] = df['trans_date_trans_time'].dt.year
        
        print(f"Date range: {df['trans_date_trans_time'].min()} to {df['trans_date_trans_time'].max()}")
        
        # Daily fraud rate
        daily_fraud = df.groupby('date')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
        daily_fraud.columns = ['date', 'total_transactions', 'fraud_count', 'fraud_rate']
        
        print(f"\nDaily fraud rate statistics:")
        print(f"Mean daily fraud rate: {daily_fraud['fraud_rate'].mean():.4f}")
        print(f"Std daily fraud rate: {daily_fraud['fraud_rate'].std():.4f}")
        print(f"Min daily fraud rate: {daily_fraud['fraud_rate'].min():.4f}")
        print(f"Max daily fraud rate: {daily_fraud['fraud_rate'].max():.4f}")
        
        # Plot temporal patterns
        plt.figure(figsize=(15, 10))
        
        # Daily fraud rate over time
        plt.subplot(2, 2, 1)
        plt.plot(daily_fraud['date'], daily_fraud['fraud_rate'])
        plt.title('Daily Fraud Rate Over Time')
        plt.xlabel('Date')
        plt.ylabel('Fraud Rate')
        plt.xticks(rotation=45)
        
        # Hourly fraud rate
        plt.subplot(2, 2, 2)
        hourly_fraud = df.groupby('hour')['is_fraud'].mean()
        hourly_fraud.plot(kind='bar')
        plt.title('Fraud Rate by Hour of Day')
        plt.xlabel('Hour')
        plt.ylabel('Fraud Rate')
        plt.xticks(rotation=0)
        
        # Monthly fraud rate
        plt.subplot(2, 2, 3)
        monthly_fraud = df.groupby('month')['is_fraud'].mean()
        monthly_fraud.plot(kind='bar')
        plt.title('Fraud Rate by Month')
        plt.xlabel('Month')
        plt.ylabel('Fraud Rate')
        plt.xticks(rotation=0)
        
        # Transaction volume over time
        plt.subplot(2, 2, 4)
        plt.plot(daily_fraud['date'], daily_fraud['total_transactions'])
        plt.title('Daily Transaction Volume')
        plt.xlabel('Date')
        plt.ylabel('Transaction Count')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return daily_fraud
    else:
        print("No temporal columns found!")
        return None

def analyze_amount_patterns(df, name="Dataset"):
    """Analyze transaction amount patterns"""
    print(f"\n=== {name.upper()} AMOUNT ANALYSIS ===")
    
    if 'amt' in df.columns:
        # Basic amount statistics
        print(f"Amount statistics:")
        print(df['amt'].describe())
        
        # Amount by fraud status
        fraud_amounts = df[df['is_fraud'] == 1]['amt']
        legit_amounts = df[df['is_fraud'] == 0]['amt']
        
        print(f"\nFraud transaction amounts:")
        print(fraud_amounts.describe())
        print(f"\nLegitimate transaction amounts:")
        print(legit_amounts.describe())
        
        # Plot amount distributions
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(df['amt'], bins=50, alpha=0.7, edgecolor='black')
        plt.title('All Transaction Amounts')
        plt.xlabel('Amount')
        plt.ylabel('Frequency')
        plt.yscale('log')
        
        plt.subplot(1, 3, 2)
        plt.hist(fraud_amounts, bins=50, alpha=0.7, color='red', edgecolor='black')
        plt.title('Fraud Transaction Amounts')
        plt.xlabel('Amount')
        plt.ylabel('Frequency')
        plt.yscale('log')
        
        plt.subplot(1, 3, 3)
        plt.hist(legit_amounts, bins=50, alpha=0.7, color='green', edgecolor='black')
        plt.title('Legitimate Transaction Amounts')
        plt.xlabel('Amount')
        plt.ylabel('Frequency')
        plt.yscale('log')
        
        plt.tight_layout()
        plt.show()
        
        # High-value transaction analysis
        high_value_threshold = 100
        high_value_fraud = df[(df['amt'] > high_value_threshold) & (df['is_fraud'] == 1)]
        high_value_legit = df[(df['amt'] > high_value_threshold) & (df['is_fraud'] == 0)]
        
        print(f"\nHigh-value transactions (>${high_value_threshold}):")
        print(f"Total high-value: {len(df[df['amt'] > high_value_threshold])}")
        print(f"High-value fraud: {len(high_value_fraud)}")
        print(f"High-value legitimate: {len(high_value_legit)}")
        print(f"High-value fraud rate: {len(high_value_fraud) / len(df[df['amt'] > high_value_threshold]):.4f}")
        
        return high_value_threshold
    else:
        print("No amount column found!")
        return None

def analyze_merchant_patterns(df, name="Dataset"):
    """Analyze merchant and category patterns"""
    print(f"\n=== {name.upper()} MERCHANT ANALYSIS ===")
    
    if 'merchant' in df.columns:
        # Top merchants by fraud rate
        merchant_fraud = df.groupby('merchant')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
        merchant_fraud.columns = ['merchant', 'total_transactions', 'fraud_count', 'fraud_rate']
        merchant_fraud = merchant_fraud[merchant_fraud['total_transactions'] >= 10]  # Min 10 transactions
        merchant_fraud = merchant_fraud.sort_values('fraud_rate', ascending=False)
        
        print(f"Top 10 merchants by fraud rate (min 10 transactions):")
        print(merchant_fraud.head(10))
        
        # Plot merchant fraud rates
        plt.figure(figsize=(12, 6))
        plt.barh(range(len(merchant_fraud.head(20))), merchant_fraud.head(20)['fraud_rate'])
        plt.yticks(range(len(merchant_fraud.head(20))), merchant_fraud.head(20)['merchant'])
        plt.title('Top 20 Merchants by Fraud Rate')
        plt.xlabel('Fraud Rate')
        plt.tight_layout()
        plt.show()
    
    if 'category' in df.columns:
        # Category analysis
        category_fraud = df.groupby('category')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
        category_fraud.columns = ['category', 'total_transactions', 'fraud_count', 'fraud_rate']
        category_fraud = category_fraud.sort_values('fraud_rate', ascending=False)
        
        print(f"\nFraud rate by category:")
        print(category_fraud)
        
        # Plot category fraud rates
        plt.figure(figsize=(12, 6))
        plt.barh(range(len(category_fraud)), category_fraud['fraud_rate'])
        plt.yticks(range(len(category_fraud)), category_fraud['category'])
        plt.title('Fraud Rate by Category')
        plt.xlabel('Fraud Rate')
        plt.tight_layout()
        plt.show()

def analyze_h1_2023_data(df, name="Dataset"):
    """Analyze H1 2023 data specifically for value analysis"""
    print(f"\n=== {name.upper()} H1 2023 VALUE ANALYSIS ===")
    
    if 'trans_date_trans_time' in df.columns and 'amt' in df.columns:
        # Filter for H1 2023
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        h1_2023 = df[(df['trans_date_trans_time'] >= '2023-01-01') & 
                     (df['trans_date_trans_time'] < '2023-07-01')]
        
        print(f"H1 2023 data shape: {h1_2023.shape}")
        
        if len(h1_2023) > 0:
            # Fraud value analysis
            fraud_data = h1_2023[h1_2023['is_fraud'] == 1]
            total_fraud_value = fraud_data['amt'].sum()
            total_fraud_count = len(fraud_data)
            
            print(f"Total fraud transactions in H1 2023: {total_fraud_count:,}")
            print(f"Total fraud value in H1 2023: ${total_fraud_value:,.2f}")
            
            # High-value vs low-value analysis (proxy for cross-border)
            high_value_threshold = 100
            high_value_fraud = fraud_data[fraud_data['amt'] > high_value_threshold]
            low_value_fraud = fraud_data[fraud_data['amt'] <= high_value_threshold]
            
            high_value_amount = high_value_fraud['amt'].sum()
            low_value_amount = low_value_fraud['amt'].sum()
            
            print(f"\nHigh-value fraud (>${high_value_threshold}):")
            print(f"  Count: {len(high_value_fraud):,}")
            print(f"  Value: ${high_value_amount:,.2f}")
            print(f"  Share: {(high_value_amount/total_fraud_value)*100:.2f}%")
            
            print(f"\nLow-value fraud (≤${high_value_threshold}):")
            print(f"  Count: {len(low_value_fraud):,}")
            print(f"  Value: ${low_value_amount:,.2f}")
            print(f"  Share: {(low_value_amount/total_fraud_value)*100:.2f}%")
            
            # Create the data structure expected by the backend
            value_analysis_data = pd.DataFrame({
                'transaction_type': ['Cross-border (High-value proxy)', 'Domestic (Low-value proxy)'],
                'fraud_value': [high_value_amount, low_value_amount],
                'fraud_count': [len(high_value_fraud), len(low_value_fraud)],
                'percentage_share': [(high_value_amount/total_fraud_value)*100, (low_value_amount/total_fraud_value)*100],
                'total_fraud_value': [total_fraud_value, total_fraud_value]
            })
            
            print(f"\nValue analysis data structure:")
            print(value_analysis_data)
            
            return value_analysis_data
        else:
            print("No H1 2023 data found!")
            return None
    else:
        print("Required columns not found for H1 2023 analysis!")
        return None

def check_database_schema():
    """Check the SQLite database schema"""
    print(f"\n=== DATABASE SCHEMA CHECK ===")
    
    db_path = "../fraud_data.db"
    if Path(db_path).exists():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table info
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables in database: {[table[0] for table in tables]}")
        
        if 'transactions' in [table[0] for table in tables]:
            # Get column info
            cursor.execute("PRAGMA table_info(transactions);")
            columns = cursor.fetchall()
            print(f"\nTransactions table columns:")
            for col in columns:
                print(f"  {col[1]} ({col[2]})")
            
            # Get sample data
            cursor.execute("SELECT * FROM transactions LIMIT 5;")
            sample_data = cursor.fetchall()
            print(f"\nSample data (first 5 rows):")
            for row in sample_data:
                print(f"  {row}")
            
            # Check data counts
            cursor.execute("SELECT COUNT(*) FROM transactions;")
            total_count = cursor.fetchone()[0]
            print(f"\nTotal transactions in database: {total_count:,}")
            
            cursor.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud = 1;")
            fraud_count = cursor.fetchone()[0]
            print(f"Fraud transactions in database: {fraud_count:,}")
            
            # Check H1 2023 data
            cursor.execute("""
                SELECT COUNT(*) FROM transactions 
                WHERE trans_date_trans_time >= '2023-01-01' 
                AND trans_date_trans_time < '2023-07-01'
            """)
            h1_2023_count = cursor.fetchone()[0]
            print(f"H1 2023 transactions: {h1_2023_count:,}")
            
            cursor.execute("""
                SELECT COUNT(*) FROM transactions 
                WHERE trans_date_trans_time >= '2023-01-01' 
                AND trans_date_trans_time < '2023-07-01'
                AND is_fraud = 1
            """)
            h1_2023_fraud_count = cursor.fetchone()[0]
            print(f"H1 2023 fraud transactions: {h1_2023_fraud_count:,}")
        
        conn.close()
    else:
        print(f"Database file not found: {db_path}")

def main():
    """Main EDA function"""
    print("FRAUD DATA EXPLORATORY DATA ANALYSIS")
    print("=" * 50)
    
    # Load data
    train_df, test_df = load_data()
    
    if train_df is not None:
        # Basic info
        basic_data_info(train_df, "Train")
        
        # Fraud distribution
        analyze_fraud_distribution(train_df, "Train")
        
        # Temporal patterns
        daily_fraud = analyze_temporal_patterns(train_df, "Train")
        
        # Amount patterns
        high_value_threshold = analyze_amount_patterns(train_df, "Train")
        
        # Merchant patterns
        analyze_merchant_patterns(train_df, "Train")
        
        # H1 2023 analysis
        h1_2023_data = analyze_h1_2023_data(train_df, "Train")
        
        # Database schema check
        check_database_schema()
        
        print("\n=== EDA COMPLETE ===")
        print("Key findings:")
        print(f"- Total transactions: {len(train_df):,}")
        print(f"- Fraud rate: {train_df['is_fraud'].mean():.4f}")
        print(f"- Date range: {train_df['trans_date_trans_time'].min()} to {train_df['trans_date_trans_time'].max()}")
        if h1_2023_data is not None:
            print(f"- H1 2023 fraud value: ${h1_2023_data['total_fraud_value'].iloc[0]:,.2f}")
            print(f"- High-value fraud share: {h1_2023_data[h1_2023_data['transaction_type'].str.contains('Cross-border')]['percentage_share'].iloc[0]:.2f}%")

if __name__ == "__main__":
    main()
