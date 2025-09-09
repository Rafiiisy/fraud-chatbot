"""
Utility functions for the fraud detection chatbot frontend
"""
import streamlit as st
from datetime import datetime
import pandas as pd


def format_metric_value(key, value):
    """Format metric values for display"""
    if isinstance(value, (int, float)):
        if 'rate' in key.lower() or 'share' in key.lower():
            return f"{value:.1f}%"
        elif 'value' in key.lower() and value > 1000000:
            return f"€{value:,.0f}"
        else:
            return value
    return value


def format_timestamp(timestamp):
    """Format timestamp for display"""
    return timestamp.strftime('%Y-%m-%d %H:%M:%S')


def truncate_text(text, max_length=50):
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def create_empty_state_message():
    """Create empty state message for chat container"""
    return """
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>No conversations yet. Ask a question to get started!</p>
    </div>
    """


def create_loading_message():
    """Create loading message"""
    return "🤔 Analyzing your question..."


def validate_question(question):
    """Validate user question"""
    if not question or not question.strip():
        return False, "Please enter a question first."
    
    if len(question.strip()) < 5:
        return False, "Please enter a more detailed question."
    
    return True, ""


def get_question_type(question):
    """Determine question type based on keywords"""
    question_lower = question.lower()
    
    if any(keyword in question_lower for keyword in ["daily", "monthly", "fluctuate", "trend", "time"]):
        return "temporal"
    elif any(keyword in question_lower for keyword in ["merchant", "category", "highest", "incidence"]):
        return "merchant"
    elif any(keyword in question_lower for keyword in ["method", "committed", "fraud"]):
        return "document"
    elif any(keyword in question_lower for keyword in ["component", "system", "detection"]):
        return "document"
    elif any(keyword in question_lower for keyword in ["eea", "cross-border", "geographic"]):
        return "geographic"
    elif any(keyword in question_lower for keyword in ["value", "share", "h1 2023"]):
        return "value"
    else:
        return "general"


def create_chat_message_html(user_message, is_user=True):
    """Create HTML for chat message"""
    message_class = "user-message" if is_user else "assistant-message"
    icon = "👤" if is_user else "🤖"
    label = "You" if is_user else "Assistant"
    
    return f"""
    <div class="chat-message {message_class}">
        <strong>{icon} {label}:</strong><br>
        {user_message}
    </div>
    """


def create_metric_columns(metrics):
    """Create columns for metrics display"""
    if not metrics:
        return []
    
    num_metrics = len(metrics)
    if num_metrics <= 2:
        return st.columns(num_metrics)
    elif num_metrics <= 4:
        return st.columns(4)
    else:
        return st.columns(min(num_metrics, 6))


def format_dataframe_for_display(df):
    """Format dataframe for better display"""
    if df is None or df.empty:
        return df
    
    # Round numeric columns to 2 decimal places
    numeric_columns = df.select_dtypes(include=['float64']).columns
    for col in numeric_columns:
        df[col] = df[col].round(2)
    
    return df
