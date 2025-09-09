"""
Response Generator Component
Handles mock response generation for different question types
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


class ResponseGenerator:
    def __init__(self):
        pass
    
    def create_mock_response(self, question):
        """Create mock response data for testing"""
        question_lower = question.lower()
        
        if "daily" in question_lower or "monthly" in question_lower or "fluctuate" in question_lower:
            return self.create_temporal_mock_response()
        elif "merchant" in question_lower or "category" in question_lower:
            return self.create_merchant_mock_response()
        elif "method" in question_lower or "committed" in question_lower:
            return self.create_document_mock_response("methods")
        elif "component" in question_lower or "system" in question_lower:
            return self.create_document_mock_response("components")
        elif "eea" in question_lower or "cross-border" in question_lower:
            return self.create_geographic_mock_response()
        elif "value" in question_lower or "share" in question_lower or "h1 2023" in question_lower:
            return self.create_value_mock_response()
        else:
            return self.create_general_mock_response()
    
    def create_temporal_mock_response(self):
        """Create mock response for temporal analysis"""
        # Generate sample time series data
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        fraud_rates = [0.02 + 0.01 * (i % 30) / 30 + 0.005 * (i % 7) / 7 for i in range(len(dates))]
        
        # Create line chart
        fig = px.line(
            x=dates, 
            y=fraud_rates,
            title="Daily Fraud Rate Over Time",
            labels={'x': 'Date', 'y': 'Fraud Rate'}
        )
        fig.update_layout(height=400)
        
        return {
            "type": "temporal",
            "title": "📈 Fraud Rate Analysis",
            "explanation": "The fraud rate shows significant fluctuations over the two-year period, with seasonal patterns and occasional spikes. The average fraud rate is approximately 2.5%, with higher rates during certain months.",
            "chart": fig,
            "data": pd.DataFrame({
                'date': dates,
                'fraud_rate': fraud_rates
            }),
            "metrics": {
                "average_rate": 2.5,
                "max_rate": 4.2,
                "min_rate": 1.8,
                "trend": "increasing"
            }
        }
    
    def create_merchant_mock_response(self):
        """Create mock response for merchant analysis"""
        merchants = ['fraud_Kirlin and Sons', 'fraud_Sporer-Keebler', 'fraud_Haley Group', 
                    'fraud_Johnston-Casper', 'fraud_Daugherty LLC', 'fraud_Romaguera Ltd',
                    'fraud_Reichel LLC', 'fraud_Goyette Group', 'fraud_Kilback Group']
        fraud_rates = [8.5, 7.2, 6.8, 6.1, 5.9, 5.4, 4.8, 4.2, 3.9]
        
        # Create bar chart
        fig = px.bar(
            x=fraud_rates,
            y=merchants,
            orientation='h',
            title="Top Merchants by Fraud Rate",
            labels={'x': 'Fraud Rate (%)', 'y': 'Merchant'}
        )
        fig.update_layout(height=500)
        
        return {
            "type": "merchant",
            "title": "🏪 Merchant Fraud Analysis",
            "explanation": "The analysis shows that merchants with 'fraud_' prefix in their names have significantly higher fraud rates, with fraud_Kirlin and Sons having the highest rate at 8.5%.",
            "chart": fig,
            "data": pd.DataFrame({
                'merchant': merchants,
                'fraud_rate': fraud_rates
            }),
            "metrics": {
                "highest_rate": 8.5,
                "average_rate": 5.9,
                "total_merchants": len(merchants)
            }
        }
    
    def create_document_mock_response(self, doc_type):
        """Create mock response for document analysis"""
        if doc_type == "methods":
            return {
                "type": "document",
                "title": "📄 Fraud Methods Analysis",
                "explanation": "Based on the document analysis, the primary methods of credit card fraud include:",
                "content": [
                    "1. **Card Not Present (CNP) Fraud**: Online transactions without physical card",
                    "2. **Card Cloning**: Copying card data to create counterfeit cards",
                    "3. **Identity Theft**: Using stolen personal information",
                    "4. **Account Takeover**: Gaining unauthorized access to accounts",
                    "5. **Friendly Fraud**: Disputing legitimate transactions"
                ],
                "sources": ["Understanding Credit Card Frauds.pdf", "EBA_ECB 2024 Report on Payment Fraud.pdf"]
            }
        else:  # components
            return {
                "type": "document",
                "title": "🔧 Fraud Detection System Components",
                "explanation": "The core components of an effective fraud detection system include:",
                "content": [
                    "1. **Real-time Monitoring**: Continuous transaction analysis",
                    "2. **Machine Learning Models**: Pattern recognition and anomaly detection",
                    "3. **Rule-based Systems**: Predefined fraud detection rules",
                    "4. **Risk Scoring**: Numerical assessment of transaction risk",
                    "5. **User Authentication**: Multi-factor authentication systems",
                    "6. **Behavioral Analysis**: User behavior pattern recognition"
                ],
                "sources": ["Understanding Credit Card Frauds.pdf"]
            }
    
    def create_geographic_mock_response(self):
        """Create mock response for geographic analysis"""
        regions = ['EEA', 'Non-EEA']
        fraud_rates = [2.1, 4.8]
        
        # Create comparison chart
        fig = px.bar(
            x=regions,
            y=fraud_rates,
            title="Fraud Rates: EEA vs Non-EEA",
            labels={'x': 'Region', 'y': 'Fraud Rate (%)'},
            color=regions,
            color_discrete_map={'EEA': '#1f77b4', 'Non-EEA': '#ff7f0e'}
        )
        fig.update_layout(height=400)
        
        percentage_higher = ((fraud_rates[1] - fraud_rates[0]) / fraud_rates[0]) * 100
        
        return {
            "type": "geographic",
            "title": "🌍 Geographic Fraud Analysis",
            "explanation": f"Fraud rates are significantly higher for transactions outside the EEA. Non-EEA transactions have a fraud rate of {fraud_rates[1]}%, which is {percentage_higher:.1f}% higher than EEA transactions ({fraud_rates[0]}%).",
            "chart": fig,
            "data": pd.DataFrame({
                'region': regions,
                'fraud_rate': fraud_rates
            }),
            "metrics": {
                "eea_rate": fraud_rates[0],
                "non_eea_rate": fraud_rates[1],
                "percentage_higher": percentage_higher
            }
        }
    
    def create_value_mock_response(self):
        """Create mock response for value analysis"""
        total_fraud_value = 1250000000  # 1.25B
        cross_border_value = 450000000  # 450M
        cross_border_share = (cross_border_value / total_fraud_value) * 100
        
        # Create pie chart
        fig = px.pie(
            values=[cross_border_value, total_fraud_value - cross_border_value],
            names=['Cross-border Fraud', 'Domestic Fraud'],
            title="Fraud Value Distribution H1 2023",
            color_discrete_map={'Cross-border Fraud': '#ff7f0e', 'Domestic Fraud': '#1f77b4'}
        )
        fig.update_layout(height=400)
        
        return {
            "type": "value",
            "title": "💰 Fraud Value Analysis",
            "explanation": f"In H1 2023, cross-border transactions accounted for {cross_border_share:.1f}% of total card fraud value. This represents €{cross_border_value:,.0f} out of €{total_fraud_value:,.0f} in total fraud value.",
            "chart": fig,
            "data": pd.DataFrame({
                'category': ['Cross-border Fraud', 'Domestic Fraud'],
                'value': [cross_border_value, total_fraud_value - cross_border_value],
                'percentage': [cross_border_share, 100 - cross_border_share]
            }),
            "metrics": {
                "total_fraud_value": total_fraud_value,
                "cross_border_value": cross_border_value,
                "cross_border_share": cross_border_share
            }
        }
    
    def create_general_mock_response(self):
        """Create general mock response"""
        return {
            "type": "general",
            "title": "🤖 General Response",
            "explanation": "I can help you analyze fraud data. Please try one of the sample questions or ask about fraud trends, merchant analysis, fraud methods, system components, geographic analysis, or value calculations.",
            "content": ["Try asking about specific fraud patterns or use the sample questions in the sidebar."]
        }
