# Fraud Detection Chatbot Backend

This backend implements the Priority 1 features from the fraud detection chatbot project, providing comprehensive analysis capabilities for answering the 6 core questions outlined in `priority1.md`.

## 🎯 Core Features

### 6 Core Questions Supported

1. **Temporal Analysis**: "How does the daily or monthly fraud rate fluctuate over the two-year period?"
2. **Merchant Analysis**: "Which merchants or merchant categories exhibit the highest incidence of fraudulent transactions?"
3. **Fraud Methods**: "What are the primary methods by which credit card fraud is committed?"
4. **System Components**: "What are the core components of an effective fraud detection system?"
5. **Geographic Analysis**: "How much higher are fraud rates when the transaction counterpart is located outside the EEA?"
6. **Value Analysis**: "What share of total card fraud value in H1 2023 was due to cross-border transactions?"

## 🏗️ Architecture

### Core Components

```
backend/
├── core/                    # Core analysis engines
│   ├── sql_generator.py     # SQL query generation for database questions
│   ├── document_processor.py # PDF processing and RAG pipeline
│   ├── chart_generator.py   # Visualization creation
│   ├── response_generator.py # Response assembly and formatting
│   └── query_classifier.py  # Question classification and routing
├── data/                    # Data management
│   ├── database.py          # Database connection and query execution
│   └── eea_countries.py     # EEA country classification
├── services/                # Main service orchestration
│   └── fraud_analysis_service.py # Main service that ties everything together
├── api/                     # REST API endpoints
│   └── fraud_api.py         # Flask API for external access
└── test_service.py          # Comprehensive test suite
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Ensure your dataset directory contains:
```
dataset/
├── archive/
│   ├── fraudTrain.csv
│   └── fraudTest.csv
├── EBA_ECB 2024 Report on Payment Fraud.pdf
└── Understanding Credit Card Frauds.pdf
```

### 3. Run Tests

```bash
cd backend
python test_service.py
```

### 4. Start API Server

```bash
cd backend
python api/fraud_api.py
```

The API will be available at `http://localhost:5000`

## 📊 Usage Examples

### Using the Service Directly

```python
from services.fraud_analysis_service import FraudAnalysisService

# Initialize service
service = FraudAnalysisService()
service.initialize()

# Process a question
response = service.process_question("How does the daily fraud rate fluctuate over time?")

print(response['summary'])
print(response['insights'])
```

### Using the API

```python
import requests

# Process a question via API
response = requests.post('http://localhost:5000/question', 
                        json={'question': 'Which merchants have the highest fraud rates?'})

data = response.json()
print(data['summary'])
```

## 🔧 Component Details

### SQL Generator (`core/sql_generator.py`)

- Generates SQL queries for all 6 question types
- Includes EEA country classification for geographic analysis
- Validates SQL queries for security
- Supports temporal, merchant, geographic, and value analysis

### Document Processor (`core/document_processor.py`)

- Extracts text from PDF documents using PyPDF2 and pdfplumber
- Creates document chunks with overlap for better search
- Generates embeddings using sentence-transformers
- Builds FAISS index for fast semantic search
- Supports RAG (Retrieval Augmented Generation) for document questions

### Chart Generator (`core/chart_generator.py`)

- Creates various chart types: line, bar, pie, comparison charts
- Supports temporal analysis with time series charts
- Generates merchant ranking visualizations
- Creates geographic comparison charts
- Formats currency and percentage values

### Response Generator (`core/response_generator.py`)

- Assembles complete responses for different question types
- Generates insights and summaries
- Formats responses for display
- Handles both database and document-based responses

### Query Classifier (`core/query_classifier.py`)

- Classifies questions using pattern matching
- Routes questions to appropriate handlers
- Extracts parameters from queries
- Validates query safety
- Provides suggested questions

## 📈 API Endpoints

### `POST /question`
Process a fraud analysis question

**Request:**
```json
{
  "question": "How does the daily fraud rate fluctuate over time?"
}
```

**Response:**
```json
{
  "question": "How does the daily fraud rate fluctuate over time?",
  "question_type": "temporal_analysis",
  "summary": "Analysis shows...",
  "insights": ["Key insight 1", "Key insight 2"],
  "chart": {
    "type": "line_chart",
    "data": {...}
  },
  "sql_query": "SELECT ...",
  "status": "success"
}
```

### `GET /suggestions`
Get suggested questions

### `GET /data/summary`
Get summary of loaded data

### `GET /status`
Get service status

### `GET /health`
Health check endpoint

## 🧪 Testing

The `test_service.py` script tests all 6 core questions:

```bash
python test_service.py
```

Expected output:
- ✅ All 6 questions processed successfully
- ✅ Correct question classification
- ✅ Charts generated for database questions
- ✅ Document search working for PDF questions
- ✅ SQL queries generated and executed

## 📋 Requirements

### Core Dependencies
- `pandas>=2.0.3` - Data manipulation
- `numpy>=1.24.3` - Numerical operations
- `plotly>=5.15.0` - Visualization
- `sqlalchemy>=2.0.0` - Database operations

### PDF Processing
- `PyPDF2>=3.0.1` - PDF text extraction
- `pdfplumber>=0.9.0` - Advanced PDF processing

### Machine Learning
- `sentence-transformers>=2.2.2` - Text embeddings
- `faiss-cpu>=1.7.4` - Vector search
- `scikit-learn>=1.3.0` - ML utilities

### API
- `flask` - REST API framework

## 🔒 Security Features

- SQL injection prevention through query validation
- Input sanitization for user queries
- Safe parameter binding for database queries
- Error handling to prevent information leakage

## 📊 Performance

- **Database queries**: < 10 seconds for typical analysis
- **Document search**: < 15 seconds for PDF processing
- **Memory usage**: Optimized for large datasets
- **Caching**: Document index cached for faster subsequent searches

## 🐛 Troubleshooting

### Common Issues

1. **"Service not initialized"**
   - Ensure CSV files are in the correct location
   - Check database connection

2. **"Document search not available"**
   - Verify PDF files are present
   - Check if document processing completed successfully

3. **"SQL query failed"**
   - Verify data is loaded correctly
   - Check query syntax

4. **Memory issues with large datasets**
   - Consider using data sampling for testing
   - Increase system memory allocation

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🚀 Next Steps

This Priority 1 implementation provides a solid foundation. Future enhancements could include:

- Real-time data streaming
- Advanced ML models for fraud prediction
- More sophisticated visualization options
- Enhanced document processing capabilities
- Performance optimizations for larger datasets

## 📝 License

This project is part of the Mekari fraud detection chatbot initiative.
