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
        """Render the sidebar with sample questions"""
        with st.sidebar:
            # Backend Status
            st.header("🔧 Server Status")
            status = self._get_backend_status()
            
            if status.get("initialized", False):
                st.markdown("🟢 **Connected**")
            else:
                st.markdown("🔴 **Disconnected**")
                st.caption("Ensure backend is running on http://localhost:5000")
            
            st.markdown("---")
            
            # Chat History Status
            self._render_chat_history_status()
            
            st.markdown("---")
            
            # Sample Questions
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
            
            # Quick Stats
            if status.get("initialized", False):
                st.markdown("---")
                st.header("📊 Quick Stats")
                if 'data_summary' in status:
                    summary = status['data_summary']
                    if 'total_records' in summary:
                        st.metric("Total Records", f"{summary['total_records']:,}")
                    if 'date_range' in summary:
                        st.metric("Date Range", summary['date_range'])
    
    def _render_chat_history_status(self):
        """Render chat history status with TTL information"""
        st.header("💬 Chat History")
        
        # Get current chat history count
        chat_count = len(st.session_state.get('chat_history', []))
        
        if chat_count > 0:
            st.info(f"📚 {chat_count} messages in current session")
            
            # Show TTL information
            if not self.storage_manager.is_expired():
                ttl_display = self.storage_manager.get_ttl_display()
                st.success(f"⏰ History expires in: {ttl_display}")
            else:
                st.warning("⏰ Chat history has expired")
            
            # Show storage info
            if 'persistent_chat_history' in st.session_state:
                stored_count = len(st.session_state.persistent_chat_history.get('data', []))
                st.caption(f"💾 {stored_count} messages saved to storage")
            
            # Clear history button
            if st.button("🗑️ Clear History", help="Clear all chat history"):
                st.session_state.chat_history = []
                self.storage_manager.clear_expired_history()
                st.rerun()
        else:
            st.caption("No messages yet. Start a conversation!")
