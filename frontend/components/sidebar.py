"""
Sidebar Component
Handles the sidebar with sample questions and settings
"""
import streamlit as st
from .storage_manager import get_storage_manager


class Sidebar:
    def __init__(self):
        self.storage_manager = get_storage_manager()
        self.sample_questions = [
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
    
    def _get_backend_status(self):
        """Get backend status from API"""
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from api_client import get_api_client
            
            api_client = get_api_client()
            return api_client.get_service_status()
        except Exception:
            return {"initialized": False}
    
    def render_sidebar(self):
        """Render the sidebar with navigation and sample questions"""
        with st.sidebar:
            # Backend Status - moved to top
            st.header("🔧 Server Status")
            status = self._get_backend_status()
            
            if status.get("initialized", False):
                st.markdown("🟢 **Connected**")
            else:
                st.markdown("🔴 **Disconnected**")
                st.caption("Ensure backend is running on http://localhost:5000")
            
            st.markdown("---")
            
            # Navigation
            st.header("Pages")
            
            # Page selection
            if 'current_page' not in st.session_state:
                st.session_state.current_page = 'chat'
            
            if st.button("🤖 Chat Interface", key="nav_chat"):
                st.session_state.current_page = 'chat'
                st.rerun()
            
            if st.button("🎯 Evaluation", key="nav_evaluation"):
                st.session_state.current_page = 'evaluation'
                st.rerun()
            
            st.markdown("---")
            
            # Sample Questions (only show on chat page)
            if st.session_state.current_page == 'chat':
                st.header("📋 Sample Questions")
                st.markdown("Click any question below to try it out:")
                
                for i, q in enumerate(self.sample_questions):
                    if st.button(f"{q['icon']} {q['question'][:50]}...", key=f"sample_{i}", help=q['question']):
                        st.session_state.current_question = q['question']
                        st.session_state.show_sample_questions = False
                        st.rerun()
                
                st.markdown("---")
                
                # About Section
                st.header("ℹ️ About")
                st.markdown("""
                This chatbot can answer questions about:
                - **Fraud Trends**: Time series analysis
                - **Merchant Analysis**: Fraud by merchants/categories
                - **Document Search**: Fraud methods and systems
                - **Geographic Analysis**: Cross-border fraud rates
                - **Value Analysis**: Fraud value calculations
                """)
            
            # Evaluation info (only show on evaluation page)
            elif st.session_state.current_page == 'evaluation':
                st.header("🎯 Evaluation Info")
                st.markdown("""
                **Evaluation System Features:**
                - **Live Evaluation**: Test Q5 and Q6 responses
                - **Quality Metrics**: Real-time scoring and analysis
                """)
                
                st.markdown("---")
                
                # st.header("📊 Quick Stats")
                # if 'evaluation_history' in st.session_state:
                #     history = st.session_state['evaluation_history']
                #     if history:
                #         avg_score = sum(eval['overall_score'] for eval in history) / len(history)
                #         st.metric("Average Score", f"{avg_score:.1f}%")
                #         st.metric("Total Evaluations", len(history))
                #     else:
                #         st.info("No evaluations yet")
                # else:
                #     st.info("No evaluations yet")
            
            # # Quick Stats
            # if status.get("initialized", False):
            #     st.markdown("---")
            #     st.header("📊 Quick Stats")
            #     if 'data_summary' in status:
            #         summary = status['data_summary']
            #         if 'total_records' in summary:
            #             st.metric("Total Records", f"{summary['total_records']:,}")
            #         if 'date_range' in summary:
            #             st.metric("Date Range", summary['date_range'])
    
