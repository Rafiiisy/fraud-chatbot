"""
Chart Generation Engine for Fraud Detection Visualizations
Creates various chart types for different analysis questions
"""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class ChartGenerator:
    """
    Generates various types of charts for fraud analysis visualization
    """
    
    def __init__(self):
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        # Chart templates
        self.template = "plotly_white"
    
    def create_line_chart(self, data: pd.DataFrame, title: str, 
                         x_col: str, y_col: str, 
                         color_col: Optional[str] = None,
                         period: str = "daily") -> go.Figure:
        """
        Create time series line chart for temporal analysis
        
        Args:
            data: DataFrame with time series data
            title: Chart title
            x_col: Column name for x-axis (dates)
            y_col: Column name for y-axis (fraud rate)
            color_col: Optional column for color grouping
            period: Time period (daily/monthly)
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        if color_col and color_col in data.columns:
            # Multiple lines for different categories
            for category in data[color_col].unique():
                category_data = data[data[color_col] == category]
                fig.add_trace(go.Scatter(
                    x=category_data[x_col],
                    y=category_data[y_col],
                    mode='lines+markers',
                    name=category,
                    line=dict(width=2),
                    marker=dict(size=4)
                ))
        else:
            # Single line
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode='lines+markers',
                name='Fraud Rate',
                line=dict(color=self.color_scheme['primary'], width=3),
                marker=dict(size=6, color=self.color_scheme['primary'])
            ))
        
        # Format x-axis for dates
        if period == "daily":
            fig.update_xaxes(
                title="Date",
                tickformat="%Y-%m-%d",
                tickangle=45
            )
        else:  # monthly
            fig.update_xaxes(
                title="Month",
                tickformat="%Y-%m",
                tickangle=45
            )
        
        # Format y-axis for percentages
        fig.update_yaxes(
            title="Fraud Rate (%)",
            tickformat=".2%"
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16, color=self.color_scheme['dark'])
            ),
            template=self.template,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=80, b=60, l=60, r=60)
        )
        
        return fig
    
    def create_bar_chart(self, data: pd.DataFrame, title: str,
                        x_col: str, y_col: str,
                        color_col: Optional[str] = None,
                        orientation: str = "vertical",
                        top_n: int = 20) -> go.Figure:
        """
        Create bar chart for ranking analysis (merchants, categories)
        
        Args:
            data: DataFrame with ranking data
            title: Chart title
            x_col: Column name for x-axis (categories)
            y_col: Column name for y-axis (values)
            color_col: Optional column for color grouping
            orientation: Chart orientation (vertical/horizontal)
            top_n: Number of top items to show
            
        Returns:
            Plotly figure object
        """
        # Sort by y_col and take top_n
        sorted_data = data.nlargest(top_n, y_col)
        
        if orientation == "horizontal":
            fig = go.Figure(data=go.Bar(
                y=sorted_data[x_col],
                x=sorted_data[y_col],
                orientation='h',
                marker=dict(
                    color=sorted_data[y_col],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Fraud Rate")
                ),
                text=sorted_data[y_col].apply(lambda x: f"{x:.2%}"),
                textposition='auto'
            ))
            
            fig.update_layout(
                yaxis=dict(title=x_col),
                xaxis=dict(title=y_col)
            )
        else:
            fig = go.Figure(data=go.Bar(
                x=sorted_data[x_col],
                y=sorted_data[y_col],
                marker=dict(
                    color=sorted_data[y_col],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Fraud Rate")
                ),
                text=sorted_data[y_col].apply(lambda x: f"{x:.2%}"),
                textposition='auto'
            ))
            
            fig.update_layout(
                xaxis=dict(title=x_col),
                yaxis=dict(title=y_col)
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16, color=self.color_scheme['dark'])
            ),
            template=self.template,
            margin=dict(t=80, b=60, l=60, r=60)
        )
        
        # Rotate x-axis labels for better readability
        if orientation == "vertical":
            fig.update_xaxes(tickangle=45)
        
        return fig
    
    def create_comparison_chart(self, data: pd.DataFrame, title: str,
                              x_col: str, y_col: str,
                              group_col: str) -> go.Figure:
        """
        Create comparison chart for geographic or value analysis
        
        Args:
            data: DataFrame with comparison data
            title: Chart title
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            group_col: Column name for grouping
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Create grouped bar chart
        for group in data[group_col].unique():
            group_data = data[data[group_col] == group]
            fig.add_trace(go.Bar(
                name=group,
                x=group_data[x_col],
                y=group_data[y_col],
                text=group_data[y_col].apply(lambda x: f"{x:.2%}"),
                textposition='auto'
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16, color=self.color_scheme['dark'])
            ),
            template=self.template,
            barmode='group',
            xaxis=dict(title=x_col),
            yaxis=dict(title=y_col, tickformat=".2%"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=80, b=60, l=60, r=60)
        )
        
        return fig
    
    def create_pie_chart(self, data: pd.DataFrame, title: str,
                        labels_col: str, values_col: str) -> go.Figure:
        """
        Create pie chart for percentage distribution analysis
        
        Args:
            data: DataFrame with distribution data
            title: Chart title
            labels_col: Column name for labels
            values_col: Column name for values
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure(data=go.Pie(
            labels=data[labels_col],
            values=data[values_col],
            textinfo='label+percent',
            textposition='auto',
            hovertemplate='<b>%{label}</b><br>' +
                         'Value: %{value:,.0f}<br>' +
                         'Percentage: %{percent}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16, color=self.color_scheme['dark'])
            ),
            template=self.template,
            margin=dict(t=80, b=60, l=60, r=60)
        )
        
        return fig
    
    def create_dashboard(self, temporal_data: pd.DataFrame, 
                        merchant_data: pd.DataFrame,
                        geographic_data: pd.DataFrame) -> go.Figure:
        """
        Create a comprehensive dashboard with multiple charts
        
        Args:
            temporal_data: Time series data
            merchant_data: Merchant ranking data
            geographic_data: Geographic comparison data
            
        Returns:
            Plotly figure with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Fraud Rate Over Time",
                "Top Merchants by Fraud Rate",
                "Geographic Comparison",
                "Fraud Value Distribution"
            ),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Temporal analysis (top-left)
        fig.add_trace(
            go.Scatter(
                x=temporal_data['date'],
                y=temporal_data['fraud_rate'],
                mode='lines+markers',
                name='Fraud Rate',
                line=dict(color=self.color_scheme['primary'])
            ),
            row=1, col=1
        )
        
        # Merchant analysis (top-right)
        top_merchants = merchant_data.nlargest(10, 'fraud_rate')
        fig.add_trace(
            go.Bar(
                x=top_merchants['merchant'],
                y=top_merchants['fraud_rate'],
                name='Merchant Fraud Rate',
                marker=dict(color=self.color_scheme['secondary'])
            ),
            row=1, col=2
        )
        
        # Geographic analysis (bottom-left)
        fig.add_trace(
            go.Bar(
                x=geographic_data['region'],
                y=geographic_data['fraud_rate'],
                name='Regional Fraud Rate',
                marker=dict(color=self.color_scheme['danger'])
            ),
            row=2, col=1
        )
        
        # Value distribution (bottom-right)
        fig.add_trace(
            go.Pie(
                labels=geographic_data['region'],
                values=geographic_data['fraud_value'],
                name='Fraud Value Distribution'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Fraud Detection Analysis Dashboard",
                x=0.5,
                font=dict(size=20, color=self.color_scheme['dark'])
            ),
            template=self.template,
            height=800,
            showlegend=False
        )
        
        return fig
    
    def format_currency(self, value: float) -> str:
        """
        Format currency values for display
        
        Args:
            value: Numeric value to format
            
        Returns:
            Formatted currency string
        """
        if value >= 1e9:
            return f"${value/1e9:.1f}B"
        elif value >= 1e6:
            return f"${value/1e6:.1f}M"
        elif value >= 1e3:
            return f"${value/1e3:.1f}K"
        else:
            return f"${value:.2f}"
    
    def create_summary_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Create summary statistics for the data
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary with summary statistics
        """
        stats = {}
        
        if 'fraud_rate' in data.columns:
            stats['avg_fraud_rate'] = data['fraud_rate'].mean()
            stats['max_fraud_rate'] = data['fraud_rate'].max()
            stats['min_fraud_rate'] = data['fraud_rate'].min()
            stats['std_fraud_rate'] = data['fraud_rate'].std()
        
        if 'total_transactions' in data.columns:
            stats['total_transactions'] = data['total_transactions'].sum()
        
        if 'fraud_count' in data.columns:
            stats['total_fraud_count'] = data['fraud_count'].sum()
        
        if 'amount' in data.columns:
            stats['total_amount'] = data['amount'].sum()
            stats['avg_amount'] = data['amount'].mean()
        
        return stats
