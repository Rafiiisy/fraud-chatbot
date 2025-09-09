# Fraud Detection Chatbot Frontend

A modular Streamlit-based frontend for the fraud detection chatbot that can answer 6 core questions about fraud data.

## Features

### Core Question Support
- **Temporal Analysis**: Daily/monthly fraud rate fluctuations
- **Merchant Analysis**: Highest fraud incidence by merchant/category
- **Document Search**: Fraud methods and system components
- **Geographic Analysis**: Cross-border fraud rate comparisons
- **Value Analysis**: Fraud value share calculations

### UI Components
- Interactive chat interface
- Sample question buttons
- Dynamic chart generation
- Scrollable chat history
- Settings panel

## Project Structure

```
frontend/
├── app.py                     # Main Streamlit application
├── config.py                  # Configuration settings
├── utils.py                   # Utility functions
├── requirements.txt           # Dependencies
├── README.md                  # This file
└── components/                # Modular components
    ├── __init__.py
    ├── header.py              # Header and CSS
    ├── sidebar.py             # Sidebar with sample questions
    ├── chat_interface.py      # Chat input and container
    ├── response_generator.py  # Mock response generation
    └── response_display.py    # Response rendering
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
# or
python -m streamlit run app.py
```

3. Open your browser to `http://localhost:8501`

## Sample Questions

The chatbot can answer these 6 core questions:

1. "How does the daily or monthly fraud rate fluctuate over the two-year period?"
2. "Which merchants or merchant categories exhibit the highest incidence of fraudulent transactions?"
3. "What are the primary methods by which credit card fraud is committed?"
4. "What are the core components of an effective fraud detection system?"
5. "How much higher are fraud rates when the transaction counterpart is located outside the EEA?"
6. "What share of total card fraud value in H1 2023 was due to cross-border transactions?"

## Modular Architecture

### Components

- **Header** (`components/header.py`): Page configuration, CSS, and main header
- **Sidebar** (`components/sidebar.py`): Sample questions and settings
- **ChatInterface** (`components/chat_interface.py`): Chat input and scrollable container
- **ResponseGenerator** (`components/response_generator.py`): Mock response creation
- **ResponseDisplay** (`components/response_display.py`): Response rendering and formatting

### Configuration

- **config.py**: Centralized configuration for themes, questions, and UI constants
- **utils.py**: Common utility functions for formatting and validation

### Benefits of Modular Design

- **Maintainability**: Easy to update individual components
- **Reusability**: Components can be reused in different contexts
- **Testing**: Each component can be tested independently
- **Scalability**: Easy to add new features or modify existing ones
- **Code Organization**: Clear separation of concerns

## Development

### Adding New Components

1. Create a new file in `components/`
2. Define a class with the component logic
3. Import and use in `app.py`

### Modifying Existing Components

1. Update the specific component file
2. The changes will be reflected in the main app
3. No need to modify other components

## Next Steps

- Connect to backend SQL generation service
- Integrate with document processing pipeline
- Add real data visualization
- Implement error handling and validation
- Add unit tests for components
