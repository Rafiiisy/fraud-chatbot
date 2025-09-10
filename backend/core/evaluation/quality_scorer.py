"""
Quality Scorer
Calculates accuracy, relevance, and overall quality scores
"""
import re
from typing import Dict, Any, List
import logging


class QualityScorer:
    """
    Calculates quality scores for chatbot responses
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_scores(self, question_id: str, chatbot_response: str, 
                        expected_type: str, ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate quality scores for a response
        
        Args:
            question_id: "5" or "6"
            chatbot_response: The response text
            expected_type: Expected question type
            ground_truth: Ground truth validation results
            
        Returns:
            Dictionary with accuracy, relevance, and overall scores
        """
        try:
            # Calculate accuracy score
            accuracy_score = self._calculate_accuracy_score(question_id, chatbot_response, ground_truth)
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(question_id, chatbot_response, expected_type)
            
            # Calculate overall score (weighted average)
            overall_score = self._calculate_overall_score(accuracy_score, relevance_score)
            
            return {
                "accuracy": round(accuracy_score, 1),
                "relevance": round(relevance_score, 1),
                "overall": round(overall_score, 1)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating scores: {e}")
            return {"accuracy": 0.0, "relevance": 0.0, "overall": 0.0}
    
    def _calculate_accuracy_score(self, question_id: str, response: str, ground_truth: Dict) -> float:
        """Calculate accuracy score based on factual correctness with weighted approach"""
        try:
            max_score = 100.0
            
            # Get expected statistics with primary/secondary classification
            expected_stats = self._get_expected_statistics(question_id)
            primary_stats = expected_stats.get("primary", [])
            secondary_stats = expected_stats.get("secondary", [])
            
            # Check for primary statistics (core answer) - 80% weight
            primary_found = 0
            for stat in primary_stats:
                if stat.lower() in response.lower():
                    primary_found += 1
            
            primary_score = (primary_found / len(primary_stats)) * 80 if primary_stats else 0
            
            # Check for secondary statistics (supporting details) - 10% weight
            secondary_found = 0
            for stat in secondary_stats:
                if stat.lower() in response.lower():
                    secondary_found += 1
            
            secondary_score = (secondary_found / len(secondary_stats)) * 10 if secondary_stats else 0
            
            # Ground truth validation score - 5% weight
            ground_truth_score = ground_truth.get('accuracy', 0) * 0.05
            
            # Core answer bonus - 5% weight (bonus for directly answering the main question)
            core_answer_bonus = self._check_core_answer(question_id, response) * 5
            
            # Combine scores
            accuracy_score = primary_score + secondary_score + ground_truth_score + core_answer_bonus
            
            return min(accuracy_score, max_score)
            
        except Exception as e:
            self.logger.warning(f"Accuracy calculation error: {e}")
            return 0.0
    
    def _calculate_relevance_score(self, question_id: str, response: str, expected_type: str) -> float:
        """Calculate relevance score based on how well response addresses the question"""
        try:
            base_score = 50.0  # Start with base relevance
            
            # Check if response directly addresses the question
            if self._addresses_question_directly(question_id, response):
                base_score += 30
            
            # Check for appropriate context and scope
            if self._has_appropriate_context(question_id, response):
                base_score += 20
            
            # Penalize if response is too generic or off-topic
            if self._is_too_generic(response):
                base_score -= 20
            
            return min(max(base_score, 0), 100)
            
        except Exception as e:
            self.logger.warning(f"Relevance calculation error: {e}")
            return 50.0
    
    def _calculate_overall_score(self, accuracy: float, relevance: float) -> float:
        """Calculate overall score with weighted average"""
        # Weighted average: 60% accuracy, 40% relevance
        return (accuracy * 0.6) + (relevance * 0.4)
    
    def _get_expected_statistics(self, question_id: str) -> Dict[str, List[str]]:
        """Get expected statistics for each question with primary/secondary classification"""
        if question_id == "5":
            return {
                "primary": ["ten times", "outside the EEA", "fraud rates", "card payments"],
                "secondary": ["77%", "SCA", "counterpart"]
            }
        elif question_id == "6":
            return {
                "primary": ["71%", "H1 2023", "cross-border", "card payment"],
                "secondary": ["43%", "47%", "28%", "value terms", "card fraud"]
            }
        else:
            return {"primary": [], "secondary": []}
    
    def _check_context_accuracy(self, question_id: str, response: str) -> float:
        """Check context-specific accuracy for each question type"""
        if question_id == "5":
            # Geographic analysis - check for EEA context
            if "EEA" in response.upper() and "outside" in response.lower():
                return 100
            elif "EEA" in response.upper() or "outside" in response.lower():
                return 70
            else:
                return 30
                
        elif question_id == "6":
            # Value analysis - check for H1 2023 context
            if "H1 2023" in response and "cross-border" in response.lower():
                return 100
            elif "H1 2023" in response or "cross-border" in response.lower():
                return 70
            else:
                return 30
        
        return 50
    
    def _check_core_answer(self, question_id: str, response: str) -> float:
        """Check if response directly answers the core question with bonus scoring"""
        response_lower = response.lower()
        
        if question_id == "5":
            # Q5: "How much higher are fraud rates when counterpart is outside EEA?"
            # Look for: "ten times" + "outside" + "EEA" + "fraud rates"
            core_indicators = [
                "ten times" in response_lower,
                "outside" in response_lower and "eea" in response_lower,
                "fraud rate" in response_lower or "fraud rates" in response_lower,
                "higher" in response_lower
            ]
            return sum(core_indicators) / len(core_indicators)
            
        elif question_id == "6":
            # Q6: "What share of total card fraud value in H1 2023 was due to cross-border transactions?"
            # Look for: "71%" + "cross-border" + "H1 2023" + "card fraud"
            core_indicators = [
                "71%" in response,
                "cross-border" in response_lower,
                "h1 2023" in response_lower or "2023" in response_lower,
                "card fraud" in response_lower
            ]
            return sum(core_indicators) / len(core_indicators)
        
        return 0.0
    
    def _addresses_question_directly(self, question_id: str, response: str) -> bool:
        """Check if response directly addresses the specific question"""
        response_lower = response.lower()
        
        if question_id == "5":
            # Should mention fraud rates and EEA
            return ("fraud rate" in response_lower or "fraud rates" in response_lower) and "eea" in response_lower
        
        elif question_id == "6":
            # Should mention fraud value and cross-border
            return ("fraud value" in response_lower or "fraud" in response_lower) and "cross-border" in response_lower
        
        return True
    
    def _has_appropriate_context(self, question_id: str, response: str) -> bool:
        """Check if response has appropriate context and scope"""
        response_lower = response.lower()
        
        if question_id == "5":
            # Should mention comparison between EEA and non-EEA
            return "eea" in response_lower and ("outside" in response_lower or "non-eea" in response_lower)
        
        elif question_id == "6":
            # Should mention time period and cross-border transactions
            return ("2023" in response_lower or "h1" in response_lower) and "cross-border" in response_lower
        
        return True
    
    def _is_too_generic(self, response: str) -> bool:
        """Check if response is too generic or off-topic"""
        generic_phrases = [
            "i don't know",
            "i cannot answer",
            "not available",
            "no data",
            "unable to",
            "sorry, but"
        ]
        
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in generic_phrases)
