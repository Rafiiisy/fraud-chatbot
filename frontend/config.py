"""
Configuration settings for the fraud detection chatbot frontend
"""

# Page Configuration
PAGE_CONFIG = {
    "page_title": "Fraud Detection Chatbot",
    "page_icon": "🤖",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Chart Themes
CHART_THEMES = [
    "plotly",
    "plotly_white", 
    "plotly_dark",
    "ggplot2",
    "seaborn",
    "simple_white"
]

# Response Detail Levels
DETAIL_LEVELS = [
    "Basic",
    "Detailed", 
    "Comprehensive"
]

# Sample Questions
SAMPLE_QUESTIONS = [
    {
        "question": "How does the daily or monthly fraud rate fluctuate over the two-year period?",
        "type": "temporal",
        "icon": "📈"
    },
    {
        "question": "Which merchants or merchant categories exhibit the highest incidence of fraudulent transactions?",
        "type": "merchant",
        "icon": "🏪"
    },
    {
        "question": "What are the primary methods by which credit card fraud is committed?",
        "type": "document",
        "icon": "📄"
    },
    {
        "question": "What are the core components of an effective fraud detection system?",
        "type": "document",
        "icon": "🔧"
    },
    {
        "question": "How much higher are fraud rates when the transaction counterpart is located outside the EEA?",
        "type": "geographic",
        "icon": "🌍"
    },
    {
        "question": "What share of total card fraud value in H1 2023 was due to cross-border transactions?",
        "type": "value",
        "icon": "💰"
    }
]

# UI Constants
UI_CONSTANTS = {
    "chat_container_height": "60vh",
    "text_area_height": 100,
    "max_chat_history": 5,
    "processing_delay": 2  # seconds
}

# CSS Classes
CSS_CLASSES = {
    "main_header": "main-header",
    "question_card": "question-card", 
    "response_card": "response-card",
    "metric_card": "metric-card",
    "success_message": "success-message",
    "info_message": "info-message",
    "error_message": "error-message",
    "centered_content": "centered-content",
    "chat_container": "chat-container",
    "chat_message": "chat-message",
    "user_message": "user-message",
    "assistant_message": "assistant-message"
}
