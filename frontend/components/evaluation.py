"""
Evaluation Component
Handles evaluation interface for Q5 and Q6 questions with quality scoring
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json


class EvaluationComponent:
    def __init__(self):
        self.evaluation_questions = [
            {
                "id": "5",
                "question": "How much higher are fraud rates when the transaction counterpart is located outside the EEA?",
                "expected_answer": "SCA was applied for the majority of electronic payments in value terms, especially for credit transfers (around 77%). In general, SCA-authenticated transactions showed lower fraud rates than non-SCA transactions, especially for card payments. Furthermore, fraud rates for card payments turned out to be significantly (about ten times) higher when the counterpart is located outside the EEA, where the application of SCA may not be requested.",
                "type": "geographic_analysis",
                "icon": "🌍"
            },
            {
                "id": "6", 
                "question": "What share of total card fraud value in H1 2023 was due to cross-border transactions?",
                "expected_answer": "Regarding the geographical dimension of fraud, the presented results show that, while most payment transactions were domestic, most card payment fraud (71% in value terms in H1 2023) and a large share of credit transfer and direct debit fraud (43% and 47%, respectively, in H1 2023) were cross-border. A notable share of fraudulent card payments (28% in H1 2023) was thereby related to cross-border transactions outside the EEA.",
                "type": "value_analysis",
                "icon": "💰"
            }
        ]
    
    def render_evaluation_page(self):
        """Render the main evaluation page"""
        # Create centered layout
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("<h1 style='text-align: center;'>🎯 Answer Evaluation System</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Evaluate chatbot responses against ground truth for Q5 and Q6 questions</p>", unsafe_allow_html=True)
            
            # Render live evaluation directly (no tabs needed)
            self._render_live_evaluation_tab()
    
    def _render_live_evaluation_tab(self):
        """Render the live evaluation interface"""
        # st.subheader("🔍 Live Answer Evaluation")
        # st.markdown("Test the chatbot's responses against known answers")
        
        # Question selection
        selected_question = st.selectbox(
            "Select Question to Evaluate:",
            options=self.evaluation_questions,
            format_func=lambda x: f"{x['icon']} {x['question'][:60]}..."
        )
        
        if selected_question:
            # Get Chatbot Response button - centered below dropdown
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                # Get chatbot response from backend API
                if st.button("Get Chatbot Response", key="get_response", type="primary", use_container_width=True):
                    # Call the backend API to get real response and evaluation
                    response = self._get_chatbot_response(selected_question['id'])
                    st.session_state[f"response_{selected_question['id']}"] = response
                    st.rerun()
            
            st.markdown("---")
            
            # Display question details
            st.markdown(f"**Question Type:** {selected_question['type'].replace('_', ' ').title()}")
            st.markdown(f"**Question:** {selected_question['question']}")
            st.markdown(f"**Expected Answer:** {selected_question['expected_answer']}")
            
            # Display response if available
            if f"response_{selected_question['id']}" in st.session_state:
                response = st.session_state[f"response_{selected_question['id']}"]
                st.markdown("**Chatbot Answer:**")
                st.info(response['answer'])
                
                # Evaluation metrics
                st.markdown("### 📊 Quality Evaluation")
                self._display_evaluation_metrics(response['evaluation'])
    
    
    
    def _get_chatbot_response(self, question_id):
        """Get chatbot response by calling the backend API"""
        try:
            import requests
            import json
            
            # Get the question text
            question_info = next((q for q in self.evaluation_questions if q['id'] == question_id), None)
            
            if not question_info:
                return self._create_error_response(f"Question ID {question_id} not found")
            
            question_text = question_info['question']
            
            # Call the main chatbot API to get response
            chatbot_url = "http://localhost:5000/question"
            chatbot_payload = {"question": question_text}
            
            response = requests.post(chatbot_url, json=chatbot_payload, timeout=30)
            
            if response.status_code == 200:
                chatbot_data = response.json()
                chatbot_answer = chatbot_data.get('data', {}).get('answer', 'No answer received')
                
                # Now call the evaluation API
                evaluation_url = "http://localhost:5000/answer-evaluation"
                evaluation_payload = {
                    "question_id": question_id,
                    "chatbot_response": chatbot_answer
                }
                
                eval_response = requests.post(evaluation_url, json=evaluation_payload, timeout=30)
                
                if eval_response.status_code == 200:
                    eval_data = eval_response.json()
                    if eval_data.get('success'):
                        return {
                            "answer": chatbot_answer,
                            "evaluation": eval_data['evaluation']
                        }
                    else:
                        return self._create_error_response(eval_data.get('error', 'Evaluation failed'))
                else:
                    return self._create_error_response(f"Evaluation API error: {eval_response.status_code}")
            else:
                return self._create_error_response(f"Chatbot API error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            return self._create_error_response("Cannot connect to backend. Please ensure the backend is running on http://localhost:5000")
        except requests.exceptions.Timeout:
            return self._create_error_response("Request timeout. Please try again.")
        except Exception as e:
            return self._create_error_response(f"Error calling backend: {str(e)}")
    
    def _create_error_response(self, error_message):
        """Create error response"""
        return {
            "answer": f"Error: {error_message}",
            "evaluation": {
                "accuracy_score": 0,
                "relevance_score": 0,
                "overall_score": 0,
                "confidence_level": "Low",
                "improvement_suggestions": [error_message],
                "source_attribution": "N/A",
                "statistical_validation": "N/A"
            }
        }
    
    def _display_evaluation_metrics(self, evaluation):
        """Display evaluation metrics in a nice format"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{evaluation['accuracy_score']}%", delta=None)
        
        with col2:
            st.metric("Relevance", f"{evaluation['relevance_score']}%", delta=None)
        
        with col3:
            st.metric("Overall Quality", f"{evaluation['overall_score']}%", delta=None)
        
        # Confidence indicator
        confidence_color = "green" if evaluation['confidence_level'] == "High" else "orange" if evaluation['confidence_level'] == "Medium" else "red"
        st.markdown(f"**Confidence Level:** :{confidence_color}[{evaluation['confidence_level']}]")
    
    
    def _save_evaluation_result(self, question_id, response):
        """Save evaluation result to session state"""
        if 'evaluation_history' not in st.session_state:
            st.session_state['evaluation_history'] = []
        
        evaluation_record = {
            'timestamp': datetime.now(),
            'question_id': question_id,
            'answer': response['answer'],
            'accuracy_score': response['evaluation']['accuracy_score'],
            'relevance_score': response['evaluation']['relevance_score'],
            'completeness_score': response['evaluation']['completeness_score'],
            'overall_score': response['evaluation']['overall_score'],
            'confidence_level': response['evaluation']['confidence_level'],
            'source_attribution': response['evaluation']['source_attribution']
        }
        
        st.session_state['evaluation_history'].append(evaluation_record)
    
    def _get_evaluation_history(self):
        """Get evaluation history from session state"""
        return st.session_state.get('evaluation_history', [])
