"""
Fraud Analysis API
REST API endpoints for the fraud detection chatbot backend
"""
from flask import Flask, request, jsonify
import sys
from pathlib import Path
import logging
import asyncio

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.append(str(backend_path))

from services.fraud_analysis_service import FraudAnalysisService

# Initialize Flask app
app = Flask(__name__)

# Initialize service
service = None

def initialize_service():
    """Initialize the fraud analysis service"""
    global service
    if service is None:
        service = FraudAnalysisService()
        if not service.initialize():
            raise RuntimeError("Failed to initialize fraud analysis service")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        if service is None:
            return jsonify({
                'success': False,
                'status': 'error',
                'data': {
                    'answer': 'Service not initialized',
                    'error': 'Service not initialized'
                }
            }), 503
        
        status = service.get_service_status()
        return jsonify({
            'success': True,
            'status': 'healthy',
            'data': {
                'answer': 'Service is healthy',
                'service_status': status
            }
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'status': 'error',
            'data': {
                'answer': 'Health check failed',
                'error': str(e)
            }
        }), 500

@app.route('/question', methods=['POST'])
def process_question():
    """Process a fraud analysis question"""
    try:
        if service is None:
            initialize_service()
        
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({
                'success': False,
                'status': 'error',
                'data': {
                    'answer': 'Please provide a question.',
                    'error': 'Question is required'
                }
            }), 400
        
        question = data['question']
        if not question.strip():
            return jsonify({
                'success': False,
                'status': 'error',
                'data': {
                    'answer': 'Please provide a question.',
                    'error': 'Question cannot be empty'
                }
            }), 400
        
        # Process the question
        response = service.process_question_sync(question)
        
        return jsonify(response), 200
        
    except Exception as e:
        logging.error(f"Error processing question: {e}")
        return jsonify({
            'success': False,
            'status': 'error',
            'data': {
                'answer': 'I encountered an error while processing your question.',
                'error': str(e)
            }
        }), 500

@app.route('/suggestions', methods=['GET'])
def get_suggestions():
    """Get suggested questions"""
    try:
        if service is None:
            initialize_service()
        
        suggestions = service.get_suggested_questions()
        return jsonify({'suggestions': suggestions}), 200
        
    except Exception as e:
        logging.error(f"Error getting suggestions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/data/summary', methods=['GET'])
def get_data_summary():
    """Get data summary"""
    try:
        if service is None:
            initialize_service()
        
        summary = service.get_data_summary()
        return jsonify(summary), 200
        
    except Exception as e:
        logging.error(f"Error getting data summary: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/chapters', methods=['GET'])
def get_chapters():
    """Get available document chapters"""
    try:
        if service is None:
            initialize_service()
        
        if not service.document_index_built:
            return jsonify({'error': 'Document search not available'}), 400
        
        chapters = service.optimized_document_processor.get_available_chapters()
        
        return jsonify({'chapters': chapters}), 200
        
    except Exception as e:
        logging.error(f"Error getting chapters: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/chapter/<chapter_id>', methods=['GET'])
def get_chapter_summary(chapter_id):
    """Get summary of a specific chapter"""
    try:
        if service is None:
            initialize_service()
        
        if not service.document_index_built:
            return jsonify({'error': 'Document search not available'}), 400
        
        chapter_summary = service.optimized_document_processor.get_chapter_summary(chapter_id)
        
        if chapter_summary is None:
            return jsonify({'error': 'Chapter not found'}), 404
        
        return jsonify(chapter_summary), 200
        
    except Exception as e:
        logging.error(f"Error getting chapter summary: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get service status"""
    try:
        if service is None:
            return jsonify({'initialized': False}), 200
        
        status = service.get_service_status()
        return jsonify(status), 200
        
    except Exception as e:
        logging.error(f"Error getting status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/answer-evaluation', methods=['POST'])
def evaluate_answer():
    """Evaluate chatbot response for Q5 and Q6 questions"""
    try:
        if service is None:
            initialize_service()
        
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Validate required fields
        if 'question_id' not in data:
            return jsonify({
                'success': False,
                'error': 'question_id is required'
            }), 400
        
        if 'chatbot_response' not in data:
            return jsonify({
                'success': False,
                'error': 'chatbot_response is required'
            }), 400
        
        question_id = str(data['question_id'])
        chatbot_response = str(data['chatbot_response'])
        
        # Validate question_id
        if question_id not in ['5', '6']:
            return jsonify({
                'success': False,
                'error': 'question_id must be "5" or "6"'
            }), 400
        
        # Validate response is not empty
        if not chatbot_response.strip():
            return jsonify({
                'success': False,
                'error': 'chatbot_response cannot be empty'
            }), 400
        
        # Perform evaluation
        evaluation_result = service.evaluate_answer(question_id, chatbot_response)
        
        return jsonify(evaluation_result), 200
        
    except Exception as e:
        logging.error(f"Error evaluating answer: {e}")
        return jsonify({
            'success': False,
            'error': f'Evaluation failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize service
    try:
        initialize_service()
        print("✅ Fraud Analysis Service initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize service: {e}")
        sys.exit(1)
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
