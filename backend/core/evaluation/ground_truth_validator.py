"""
Ground Truth Validator
Validates responses against document-based ground truth
"""
import re
from typing import Dict, Any, List, Optional
import logging


class GroundTruthValidator:
    """
    Validates chatbot responses against ground truth from documents
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Ground truth data for Q5 and Q6
        self.ground_truth_data = {
            "5": {
                "expected_answer": "SCA was applied for the majority of electronic payments in value terms, especially for credit transfers (around 77%). In general, SCA-authenticated transactions showed lower fraud rates than non-SCA transactions, especially for card payments. Furthermore, fraud rates for card payments turned out to be significantly (about ten times) higher when the counterpart is located outside the EEA, where the application of SCA may not be requested.",
                "key_concepts": [
                    "ten times higher",
                    "outside the EEA", 
                    "fraud rates",
                    "card payments",
                    "SCA",
                    "counterpart"
                ],
                "key_statistics": ["ten times", "77%"],
                "source": "EBA/ECB 2024 Report on Payment Fraud",
                "section": "Geographic Analysis"
            },
            "6": {
                "expected_answer": "Regarding the geographical dimension of fraud, the presented results show that, while most payment transactions were domestic, most card payment fraud (71% in value terms in H1 2023) and a large share of credit transfer and direct debit fraud (43% and 47%, respectively, in H1 2023) were cross-border. A notable share of fraudulent card payments (28% in H1 2023) was thereby related to cross-border transactions outside the EEA.",
                "key_concepts": [
                    "71%",
                    "cross-border",
                    "H1 2023",
                    "card payment fraud",
                    "value terms",
                    "domestic",
                    "outside the EEA"
                ],
                "key_statistics": ["71%", "43%", "47%", "28%", "H1 2023"],
                "source": "EBA/ECB 2024 Report on Payment Fraud",
                "section": "Value Analysis"
            }
        }
    
    def validate_response(self, question_id: str, response: str, document_processor=None) -> Dict[str, Any]:
        """
        Validate response against ground truth
        
        Args:
            question_id: "5" or "6"
            response: Chatbot response text
            document_processor: Document processor instance (optional)
            
        Returns:
            Validation results dictionary
        """
        try:
            if question_id not in self.ground_truth_data:
                return self._create_error_result("Invalid question_id")
            
            ground_truth = self.ground_truth_data[question_id]
            
            # Extract statistics from response
            extracted_stats = self._extract_statistics(response)
            
            # Validate against ground truth
            validation_results = self._validate_statistics(question_id, extracted_stats, ground_truth, response)
            
            # Calculate accuracy percentage
            accuracy = self._calculate_validation_accuracy(validation_results)
            
            # Check source attribution
            source_attribution = self._check_source_attribution(response)
            
            return {
                "validation_status": "Validated against ground truth",
                "source_attribution": source_attribution,
                "statistics_found": extracted_stats,
                "accuracy": accuracy,
                "validation_details": validation_results
            }
            
        except Exception as e:
            self.logger.error(f"Ground truth validation error: {e}")
            return {
                "validation_status": f"Validation error: {str(e)}",
                "source_attribution": "EBA/ECB 2024 Report (reference)",
                "statistics_found": [],
                "accuracy": 0
            }
    
    def _extract_statistics(self, response: str) -> List[str]:
        """Extract numerical statistics and key phrases from response text"""
        # Pattern to match percentages and decimal numbers
        percentage_pattern = r'\d+\.?\d*%'
        decimal_pattern = r'\d+\.\d+%'
        
        # Find all matches
        percentages = re.findall(percentage_pattern, response)
        decimals = re.findall(decimal_pattern, response)
        
        # Also extract key phrases
        key_phrases = []
        if "ten times" in response.lower():
            key_phrases.append("ten times")
        if "71%" in response:
            key_phrases.append("71%")
        if "77%" in response:
            key_phrases.append("77%")
        if "43%" in response:
            key_phrases.append("43%")
        if "47%" in response:
            key_phrases.append("47%")
        if "28%" in response:
            key_phrases.append("28%")
        
        # Combine and deduplicate
        all_stats = list(set(percentages + decimals + key_phrases))
        
        return all_stats
    
    def _validate_statistics(self, question_id: str, extracted_stats: List[str], ground_truth: Dict, response: str = "") -> Dict[str, Any]:
        """Validate extracted statistics against ground truth"""
        validation_results = {}
        key_concepts = ground_truth["key_concepts"]
        key_stats = ground_truth["key_statistics"]
        
        if question_id == "5":
            # Validate Q5 - look for key concepts and statistics
            validation_results["ten_times_higher"] = any("ten times" in stat.lower() for stat in extracted_stats)
            validation_results["outside_eea_mentioned"] = "outside the eea" in response.lower()
            validation_results["fraud_rates_mentioned"] = "fraud rates" in response.lower()
            validation_results["card_payments_mentioned"] = "card payments" in response.lower()
            validation_results["sca_mentioned"] = "sca" in response.lower()
            validation_results["counterpart_mentioned"] = "counterpart" in response.lower()
            
        elif question_id == "6":
            # Validate Q6 - look for key concepts and statistics
            validation_results["71_percent_found"] = any("71%" in stat for stat in extracted_stats)
            validation_results["cross_border_mentioned"] = "cross-border" in response.lower()
            validation_results["h1_2023_mentioned"] = "h1 2023" in response.lower()
            validation_results["card_payment_fraud_mentioned"] = "card payment fraud" in response.lower()
            validation_results["value_terms_mentioned"] = "value terms" in response.lower()
            validation_results["domestic_mentioned"] = "domestic" in response.lower()
        
        return validation_results
    
    def _check_stat_match(self, extracted_stats: List[str], expected_stat: str) -> bool:
        """Check if expected statistic is found in extracted statistics"""
        # Normalize for comparison
        expected_normalized = expected_stat.replace("%", "").strip()
        
        for stat in extracted_stats:
            stat_normalized = stat.replace("%", "").strip()
            if stat_normalized == expected_normalized:
                return True
        
        return False
    
    def _calculate_validation_accuracy(self, validation_results: Dict[str, Any]) -> float:
        """Calculate accuracy percentage based on validation results"""
        if not validation_results:
            return 0.0
        
        total_checks = len(validation_results)
        passed_checks = sum(1 for result in validation_results.values() if result)
        
        return (passed_checks / total_checks) * 100
    
    def _check_source_attribution(self, response: str) -> str:
        """Check if response includes proper source attribution"""
        response_upper = response.upper()
        
        # Check for EBA/ECB reference
        if "EBA" in response_upper or "ECB" in response_upper:
            if "2024" in response or "report" in response.lower():
                return "EBA/ECB 2024 Report on Payment Fraud"
            else:
                return "EBA/ECB Report"
        
        # Check for other source references
        if "report" in response.lower():
            return "Document-based source"
        
        return "EBA/ECB 2024 Report (reference)"
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            "validation_status": f"Error: {error_message}",
            "source_attribution": "EBA/ECB 2024 Report (reference)",
            "statistics_found": [],
            "accuracy": 0
        }
