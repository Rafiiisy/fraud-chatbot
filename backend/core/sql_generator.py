"""
SQL Generation Engine for Fraud Detection Analysis
Generates SQL queries for all 6 core questions from priority1.md
"""
import re
from typing import Dict, List, Optional, Tuple
from enum import Enum


class QuestionType(Enum):
    """Enumeration of supported question types"""
    TEMPORAL_ANALYSIS = "temporal_analysis"
    MERCHANT_ANALYSIS = "merchant_analysis"
    FRAUD_METHODS = "fraud_methods"
    SYSTEM_COMPONENTS = "system_components"
    GEOGRAPHIC_ANALYSIS = "geographic_analysis"
    VALUE_ANALYSIS = "value_analysis"
    FORECASTING = "forecasting"


class SQLGenerator:
    """
    Generates SQL queries for different types of fraud analysis questions
    """
    
    def __init__(self):
        self.table_name = "transactions"
        self.eea_countries = self._get_eea_countries()
    
    def _get_eea_countries(self) -> List[str]:
        """Get list of EEA countries for geographic analysis"""
        return [
            'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic',
            'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece',
            'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg',
            'Malta', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia',
            'Slovenia', 'Spain', 'Sweden', 'Iceland', 'Liechtenstein', 'Norway'
        ]
    
    def generate_sql(self, question: str, question_type: QuestionType) -> Dict[str, str]:
        """
        Generate SQL query based on question type
        
        Args:
            question: The user's question
            question_type: Type of analysis needed
            
        Returns:
            Dictionary containing SQL query and metadata
        """
        if question_type == QuestionType.TEMPORAL_ANALYSIS:
            return self.generate_temporal_analysis(question)
        elif question_type == QuestionType.MERCHANT_ANALYSIS:
            return self.generate_merchant_analysis(question)
        elif question_type == QuestionType.GEOGRAPHIC_ANALYSIS:
            return self.generate_geographic_analysis(question)
        elif question_type == QuestionType.VALUE_ANALYSIS:
            return self.generate_value_analysis(question)
        else:
            raise ValueError(f"Unsupported question type: {question_type}")
    
    def generate_temporal_analysis(self, question: str) -> Dict[str, str]:
        """
        Generate SQL for temporal analysis (Question 1)
        How does the daily or monthly fraud rate fluctuate over the two-year period?
        """
        # Check if user wants daily or monthly analysis
        period = "daily" if "daily" in question.lower() else "monthly"
        
        if period == "daily":
            sql = f"""
            SELECT 
                DATE(trans_date_trans_time) as date,
                AVG(is_fraud) as fraud_rate,
                COUNT(*) as total_transactions,
                SUM(is_fraud) as fraud_count
            FROM {self.table_name}
            WHERE trans_date_trans_time IS NOT NULL
            GROUP BY DATE(trans_date_trans_time)
            ORDER BY date
            """
        else:  # monthly
            sql = f"""
            SELECT 
                DATE_TRUNC('month', trans_date_trans_time) as month,
                AVG(is_fraud) as fraud_rate,
                COUNT(*) as total_transactions,
                SUM(is_fraud) as fraud_count
            FROM {self.table_name}
            WHERE trans_date_trans_time IS NOT NULL
            GROUP BY DATE_TRUNC('month', trans_date_trans_time)
            ORDER BY month
            """
        
        return {
            "sql": sql,
            "question_type": "temporal_analysis",
            "period": period,
            "description": f"Analyzes fraud rate trends over time ({period} aggregation)"
        }
    
    def generate_merchant_analysis(self, question: str) -> Dict[str, str]:
        """
        Generate SQL for merchant analysis (Question 2)
        Which merchants or merchant categories exhibit the highest incidence of fraudulent transactions?
        """
        # Check if user wants merchant names or categories
        analysis_type = "category" if "category" in question.lower() else "merchant"
        
        if analysis_type == "merchant":
            sql = f"""
            SELECT 
                merchant,
                AVG(is_fraud) as fraud_rate,
                COUNT(*) as total_transactions,
                SUM(is_fraud) as fraud_count,
                ROUND(AVG(is_fraud) * 100, 2) as fraud_percentage
            FROM {self.table_name}
            WHERE merchant IS NOT NULL
            GROUP BY merchant
            HAVING COUNT(*) >= 10
            ORDER BY fraud_rate DESC
            LIMIT 20
            """
        else:  # category
            sql = f"""
            SELECT 
                category,
                AVG(is_fraud) as fraud_rate,
                COUNT(*) as total_transactions,
                SUM(is_fraud) as fraud_count,
                ROUND(AVG(is_fraud) * 100, 2) as fraud_percentage
            FROM {self.table_name}
            WHERE category IS NOT NULL
            GROUP BY category
            ORDER BY fraud_rate DESC
            """
        
        return {
            "sql": sql,
            "question_type": "merchant_analysis",
            "analysis_type": analysis_type,
            "description": f"Analyzes fraud rates by {analysis_type}"
        }
    
    def generate_geographic_analysis(self, question: str) -> Dict[str, str]:
        """
        Generate SQL for geographic analysis (Question 5)
        How much higher are fraud rates when the transaction counterpart is located outside the EEA?
        
        Since we don't have merchant_country, we'll use a simplified approach based on
        transaction amounts and merchant categories to simulate geographic differences.
        """
        
        # Use a simplified approach: compare high-value vs low-value transactions
        # as a proxy for domestic vs international (cross-border) transactions
        sql = f"""
        WITH fraud_rates AS (
            SELECT 
                CASE 
                    WHEN amt > 100 THEN 'High-Value (Cross-border proxy)'
                    ELSE 'Low-Value (Domestic proxy)'
                END as transaction_type,
                AVG(is_fraud) as fraud_rate,
                COUNT(*) as total_transactions,
                SUM(is_fraud) as fraud_count,
                AVG(amt) as avg_amount
            FROM {self.table_name}
            WHERE amt IS NOT NULL
            GROUP BY 
                CASE 
                    WHEN amt > 100 THEN 'High-Value (Cross-border proxy)'
                    ELSE 'Low-Value (Domestic proxy)'
                END
        )
        SELECT 
            transaction_type,
            fraud_rate,
            total_transactions,
            fraud_count,
            ROUND(fraud_rate * 100, 2) as fraud_percentage,
            ROUND(avg_amount, 2) as avg_amount
        FROM fraud_rates
        ORDER BY fraud_rate DESC
        """
        
        return {
            "sql": sql,
            "question_type": "geographic_analysis",
            "description": "Compares fraud rates between high-value (cross-border proxy) and low-value (domestic proxy) transactions"
        }
    
    def generate_value_analysis(self, question: str) -> Dict[str, str]:
        """
        Generate SQL for value analysis (Question 6)
        What share of total card fraud value in H1 2023 was due to cross-border transactions?
        """
        print(f"\n=== SQL GENERATOR DEBUG ===")
        print(f"Question: {question}")
        print(f"Table name: {self.table_name}")
        
        # Create EEA country list for SQL IN clause
        eea_list = "', '".join(self.eea_countries)
        print(f"EEA countries: {self.eea_countries[:5]}...")  # Show first 5 countries
        
        # Since we don't have merchant_country, use transaction amount as proxy
        # High-value transactions are more likely to be cross-border
        sql = f"""
        WITH h1_2023_fraud AS (
            SELECT 
                CASE 
                    WHEN amt > 100 THEN 'Cross-border (High-value proxy)'
                    ELSE 'Domestic (Low-value proxy)'
                END as transaction_type,
                SUM(amt * is_fraud) as fraud_value,
                COUNT(*) as total_transactions,
                SUM(is_fraud) as fraud_count
            FROM {self.table_name}
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
        
        print(f"Generated SQL: {sql}")
        print("=== END SQL GENERATOR DEBUG ===\n")
        
        return {
            "sql": sql,
            "question_type": "value_analysis",
            "description": "Analyzes fraud value distribution for H1 2023 by transaction type (using amount as proxy for cross-border)"
        }
    
    def get_question_type(self, question: str) -> QuestionType:
        """
        Classify question type based on keywords in the question
        
        Args:
            question: User's question
            
        Returns:
            QuestionType enum value
        """
        question_lower = question.lower()
        
        # Temporal analysis keywords
        if any(keyword in question_lower for keyword in ['daily', 'monthly', 'fluctuate', 'over time', 'trend', 'period']):
            return QuestionType.TEMPORAL_ANALYSIS
        
        # Merchant analysis keywords
        elif any(keyword in question_lower for keyword in ['merchant', 'category', 'highest', 'incidence', 'exhibit']):
            return QuestionType.MERCHANT_ANALYSIS
        
        # Geographic analysis keywords
        elif any(keyword in question_lower for keyword in ['eea', 'outside', 'counterpart', 'location', 'geographic']):
            return QuestionType.GEOGRAPHIC_ANALYSIS
        
        # Value analysis keywords
        elif any(keyword in question_lower for keyword in ['value', 'h1 2023', 'share', 'cross-border', 'total']):
            return QuestionType.VALUE_ANALYSIS
        
        # Document analysis keywords
        elif any(keyword in question_lower for keyword in ['method', 'committed', 'fraud method', 'how']):
            return QuestionType.FRAUD_METHODS
        
        # System components keywords
        elif any(keyword in question_lower for keyword in ['component', 'system', 'detection', 'effective']):
            return QuestionType.SYSTEM_COMPONENTS
        
        else:
            # Default to temporal analysis if unclear
            return QuestionType.TEMPORAL_ANALYSIS
    
    def validate_sql(self, sql: str) -> bool:
        """
        Basic SQL validation to prevent injection attacks
        
        Args:
            sql: SQL query to validate
            
        Returns:
            True if SQL appears safe, False otherwise
        """
        # List of dangerous SQL keywords/patterns
        dangerous_patterns = [
            r'\bDROP\b',
            r'\bDELETE\b',
            r'\bINSERT\b',
            r'\bUPDATE\b',
            r'\bALTER\b',
            r'\bCREATE\b',
            r'\bTRUNCATE\b',
            r'\bEXEC\b',
            r'\bEXECUTE\b',
            r'--',
            r'/\*',
            r'\*/',
            r';\s*$',
            r'UNION\s+SELECT',
            r'OR\s+1\s*=\s*1',
            r'AND\s+1\s*=\s*1'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                return False
        
        return True
