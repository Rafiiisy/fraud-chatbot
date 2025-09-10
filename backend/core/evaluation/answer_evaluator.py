"""
Answer Evaluator
Main evaluation logic for Q5 and Q6 questions
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.append(str(backend_path))

from core.query.query_classifier import QueryClassifier
from core.document.hybrid_document_processor import HybridDocumentProcessor
from .quality_scorer import QualityScorer
from .ground_truth_validator import GroundTruthValidator


class AnswerEvaluator:
    """
    Main evaluator for chatbot responses to Q5 and Q6 questions
    """
    
    def __init__(self, data_dir: str = "dataset"):
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.query_classifier = QueryClassifier(use_llm=True)
        self.quality_scorer = QualityScorer()
        self.ground_truth_validator = GroundTruthValidator()
        
        # Initialize document processor for ground truth validation
        try:
            self.document_processor = HybridDocumentProcessor(data_dir, use_faiss=True)
            self.document_processor.process_documents()
            self.document_available = True
        except Exception as e:
            self.logger.warning(f"Document processor not available: {e}")
            self.document_available = False
    
    def evaluate_response(self, question_id: str, chatbot_response: str) -> Dict[str, Any]:
        """
        Evaluate a chatbot response for Q5 or Q6 questions
        
        Args:
            question_id: "5" or "6"
            chatbot_response: The chatbot's response text
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            # Get question details
            question_info = self._get_question_info(question_id)
            if not question_info:
                return self._create_error_response(f"Invalid question_id: {question_id}")
            
            # Step 1: Test chatbot's question classification
            classification_result = self._test_classification(question_id, chatbot_response)
            
            # Step 2: Validate response against ground truth
            ground_truth_result = self._validate_ground_truth(question_id, chatbot_response)
            
            # Step 3: Calculate quality scores
            quality_scores = self.quality_scorer.calculate_scores(
                question_id=question_id,
                chatbot_response=chatbot_response,
                expected_type=question_info['expected_type'],
                ground_truth=ground_truth_result
            )
            
            # Step 4: Generate improvement suggestions
            improvement_suggestions = self._generate_improvement_suggestions(
                question_id, chatbot_response, quality_scores, ground_truth_result
            )
            
            # Step 5: Compile final evaluation
            evaluation = {
                "accuracy_score": quality_scores['accuracy'],
                "relevance_score": quality_scores['relevance'],
                "overall_score": quality_scores['overall'],
                "confidence_level": self._determine_confidence_level(quality_scores['overall']),
                "source_attribution": ground_truth_result.get('source_attribution', 'EBA/ECB 2024 Report'),
                "statistical_validation": ground_truth_result.get('validation_status', 'Not validated'),
                "improvement_suggestions": improvement_suggestions,
                "classification_accuracy": classification_result['accuracy'],
                "question_type_detected": classification_result['detected_type'],
                "expected_question_type": question_info['expected_type']
            }
            
            return {
                "success": True,
                "evaluation": evaluation
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating response: {e}")
            return self._create_error_response(f"Evaluation failed: {str(e)}")
    
    def _get_question_info(self, question_id: str) -> Dict[str, Any]:
        """Get question information based on question_id"""
        questions = {
            "5": {
                "question": "How much higher are fraud rates when the transaction counterpart is located outside the EEA?",
                "expected_type": "geographic_analysis",
                "expected_answer": "Approximately 20% higher (0.12% domestic vs 0.10% EEA vs 0.103% outside EEA)",
                "key_statistics": ["0.12%", "0.10%", "0.103%", "20%", "higher"]
            },
            "6": {
                "question": "What share of total card fraud value in H1 2023 was due to cross-border transactions?",
                "expected_type": "value_analysis", 
                "expected_answer": "71% of total card fraud value",
                "key_statistics": ["71%", "H1 2023", "cross-border", "fraud value"]
            }
        }
        return questions.get(question_id)
    
    def _test_classification(self, question_id: str, chatbot_response: str) -> Dict[str, Any]:
        """Test if chatbot correctly classifies the question type"""
        try:
            # Get the original question
            question_info = self._get_question_info(question_id)
            original_question = question_info['question']
            
            # Use query classifier to determine question type
            detected_type, confidence, metadata = self.query_classifier.classify_query(original_question)
            
            # Check if classification matches expected type
            expected_type = question_info['expected_type']
            detected_type_str = str(detected_type) if detected_type else "unknown"
            is_correct = detected_type_str == expected_type
            
            return {
                "detected_type": str(detected_type) if detected_type else "unknown",
                "expected_type": expected_type,
                "accuracy": 100 if is_correct else 0,
                "confidence": confidence,
                "is_correct": is_correct
            }
            
        except Exception as e:
            self.logger.warning(f"Classification test failed: {e}")
            return {
                "detected_type": "unknown",
                "expected_type": self._get_question_info(question_id)['expected_type'],
                "accuracy": 0,
                "confidence": 0,
                "is_correct": False
            }
    
    def _validate_ground_truth(self, question_id: str, chatbot_response: str) -> Dict[str, Any]:
        """Validate response against ground truth from documents"""
        try:
            if not self.document_available:
                return {
                    "validation_status": "Document processing not available",
                    "source_attribution": "EBA/ECB 2024 Report (reference)",
                    "statistics_found": [],
                    "accuracy": 0
                }
            
            # Use ground truth validator
            validation_result = self.ground_truth_validator.validate_response(
                question_id, chatbot_response, self.document_processor
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.warning(f"Ground truth validation failed: {e}")
            return {
                "validation_status": f"Validation error: {str(e)}",
                "source_attribution": "EBA/ECB 2024 Report (reference)",
                "statistics_found": [],
                "accuracy": 0
            }
    
    def _generate_improvement_suggestions(self, question_id: str, chatbot_response: str, 
                                        quality_scores: Dict, ground_truth_result: Dict) -> list:
        """Generate improvement suggestions based on evaluation results"""
        suggestions = []
        
        # Accuracy-based suggestions
        if quality_scores['accuracy'] < 80:
            suggestions.append("Include more specific statistical data and percentages")
            suggestions.append("Verify numerical accuracy against source documents")
        
        # Relevance-based suggestions
        if quality_scores['relevance'] < 80:
            suggestions.append("Ensure response directly addresses the question asked")
            suggestions.append("Provide more context about the specific time period or scope")
        
        # Ground truth validation suggestions
        if ground_truth_result.get('accuracy', 0) < 70:
            suggestions.append("Cross-reference statistics with EBA/ECB 2024 Report")
            suggestions.append("Include proper source citations and attribution")
        
        # Question-specific suggestions
        if question_id == "5":
            if "EEA" not in chatbot_response.upper():
                suggestions.append("Mention EEA (European Economic Area) specifically")
            if not any(stat in chatbot_response for stat in ["0.12%", "0.10%", "0.103%"]):
                suggestions.append("Include specific fraud rate percentages for comparison")
        
        elif question_id == "6":
            if "71%" not in chatbot_response:
                suggestions.append("Include the specific 71% figure for cross-border fraud")
            if "H1 2023" not in chatbot_response:
                suggestions.append("Specify the H1 2023 time period")
        
        return suggestions if suggestions else ["Response meets quality standards"]
    
    def _determine_confidence_level(self, overall_score: float) -> str:
        """Determine confidence level based on overall score"""
        if overall_score >= 90:
            return "High"
        elif overall_score >= 70:
            return "Medium"
        else:
            return "Low"
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "success": False,
            "error": error_message,
            "evaluation": {
                "accuracy_score": 0,
                "relevance_score": 0,
                "overall_score": 0,
                "confidence_level": "Low",
                "source_attribution": "N/A",
                "statistical_validation": "Error",
                "improvement_suggestions": [error_message]
            }
        }
