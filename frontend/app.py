"""
Main Streamlit Application
Modular fraud detection chatbot frontend
"""
import streamlit as st
import os
import sys

# Add components directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'components'))

from components.header import Header
from components.sidebar import Sidebar
from components.chat_interface import ChatInterface
from components.response_display import ResponseDisplay


class FraudChatbotApp:
    def __init__(self):
        self.header = Header()
        self.sidebar = Sidebar()
        self.chat_interface = ChatInterface()
        self.response_display = ResponseDisplay()
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'current_question' not in st.session_state:
            st.session_state.current_question = ""
        
        if 'response_data' not in st.session_state:
            st.session_state.response_data = None
        
        if 'show_sample_questions' not in st.session_state:
            st.session_state.show_sample_questions = True
        
        if 'latest_chart' not in st.session_state:
            st.session_state.latest_chart = None
        
        if 'displayed_charts' not in st.session_state:
            st.session_state.displayed_charts = set()
        
        if 'available_charts' not in st.session_state:
            st.session_state.available_charts = {}
    
    def run(self):
        """Main application runner"""
        # Setup page configuration and CSS
        self.header.setup_page_config()
        self.header.render_custom_css()
        
        # Render header
        self.header.render_header()
        
        # Create centered layout with sidebar on the side
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            self.sidebar.render_sidebar()
        
        with col2:
            # Main content area with chat container centered
            # Check if we have a response with a chart
            has_chart = False
            latest_chart = None
            if (st.session_state.chat_history and 
                len(st.session_state.chat_history) >= 2 and 
                st.session_state.chat_history[-1].get('response') and
                not st.session_state.chat_history[-1].get('is_processing', False)):
                
                latest_response = st.session_state.chat_history[-1]['response']
                if 'chart' in latest_response and latest_response['chart'] is not None:
                    has_chart = True
                    latest_chart = latest_response['chart']
            
            
            # Get chat content as HTML string
            chat_html = self.chat_interface.get_chat_html()
            
            # Create the complete HTML with proper structure and auto-scroll
            container_html = f"""
            <div style="display: flex; flex-direction: column; height: 40vh; justify-content: center; align-items: center;">
                <div id="chat-container" style="flex: 1; overflow-y: auto; padding: 1rem; border: 1px solid #333333; border-radius: 0.5rem; background-color: transparent; margin-bottom: 1rem; width: 100%; max-width: 1200px;">
                    {chat_html}
                </div>
            </div>
            <script>
                // Auto-scroll to bottom of chat container
                function scrollToBottom() {{
                    const container = document.getElementById('chat-container');
                    if (container) {{
                        container.scrollTop = container.scrollHeight;
                    }}
                }}
                
                // Local Storage Management with TTL
                const CHAT_STORAGE_KEY = 'fraud_chatbot_history';
                const TTL_MINUTES = 30;
                
                // Save chat history to localStorage
                function saveChatHistory(chatHistory) {{
                    try {{
                        const dataToSave = {{
                            chatHistory: chatHistory,
                            timestamp: Date.now(),
                            ttl: TTL_MINUTES * 60 * 1000 // Convert to milliseconds
                        }};
                        localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(dataToSave));
                        console.log('Chat history saved to localStorage');
                    }} catch (error) {{
                        console.error('Error saving chat history:', error);
                    }}
                }}
                
                // Load chat history from localStorage
                function loadChatHistory() {{
                    try {{
                        const stored = localStorage.getItem(CHAT_STORAGE_KEY);
                        if (!stored) return [];
                        
                        const data = JSON.parse(stored);
                        const now = Date.now();
                        
                        // Check if data has expired
                        if (now - data.timestamp > data.ttl) {{
                            localStorage.removeItem(CHAT_STORAGE_KEY);
                            console.log('Chat history expired, cleared from localStorage');
                            return [];
                        }}
                        
                        console.log('Chat history loaded from localStorage');
                        return data.chatHistory || [];
                    }} catch (error) {{
                        console.error('Error loading chat history:', error);
                        return [];
                    }}
                }}
                
                // Clean up expired chat history
                function cleanupExpiredHistory() {{
                    try {{
                        const stored = localStorage.getItem(CHAT_STORAGE_KEY);
                        if (!stored) return;
                        
                        const data = JSON.parse(stored);
                        const now = Date.now();
                        
                        if (now - data.timestamp > data.ttl) {{
                            localStorage.removeItem(CHAT_STORAGE_KEY);
                            console.log('Expired chat history cleaned up');
                        }}
                    }} catch (error) {{
                        console.error('Error cleaning up chat history:', error);
                    }}
                }}
                
                // Initialize chat history from localStorage
                function initializeChatHistory() {{
                    const savedHistory = loadChatHistory();
                    if (savedHistory.length > 0) {{
                        // Notify Streamlit about the loaded history
                        // This will be handled by the Python side
                        console.log('Found saved chat history:', savedHistory.length, 'messages');
                    }}
                }}
                
                // Scroll to bottom when page loads
                window.addEventListener('load', function() {{
                    scrollToBottom();
                    initializeChatHistory();
                    cleanupExpiredHistory();
                }});
                
                // Scroll to bottom after a short delay to ensure content is rendered
                setTimeout(scrollToBottom, 100);
                
                // Clean up expired history every 5 minutes
                setInterval(cleanupExpiredHistory, 5 * 60 * 1000);
                
            </script>
            """
            
            st.markdown(container_html, unsafe_allow_html=True)
            
            
            # Check if the latest response is still processing
            latest_processing = False
            if st.session_state.chat_history:
                latest_chat = st.session_state.chat_history[-1]
                if latest_chat.get('is_processing', False):
                    latest_processing = True
                    # Clear all chart state when processing to make it look fresh
                    st.session_state.displayed_charts.clear()
                    st.session_state.available_charts.clear()
            
            # Chart generation and management section (only when not processing)
            if not latest_processing and st.session_state.chat_history:
                # Check if the LATEST response has chart data (not any response)
                latest_has_chart = False
                if st.session_state.chat_history:
                    latest_chat = st.session_state.chat_history[-1]
                    if (latest_chat.get('response') and 
                        'chart' in latest_chat['response'] and 
                        latest_chat['response']['chart'] is not None):
                        latest_has_chart = True
                
                # Also check if there are any displayed charts
                has_displayed_charts = len(st.session_state.displayed_charts) > 0
                
                if latest_has_chart or has_displayed_charts:
                    st.markdown("---")
                    
                    # Always show generate button if there's chart data
                    col1, col2 = st.columns([1, 1])
                
                    with col1:
                        # Only show generate button if the latest response has chart data
                        if latest_has_chart:
                            # Get the latest response with chart data
                            latest_chat = st.session_state.chat_history[-1]
                            response_id = f"response_{len(st.session_state.chat_history) - 1}"
                            chart = latest_chat['response']['chart']
                            
                            # Check if this specific chart is already displayed
                            if response_id not in st.session_state.displayed_charts:
                                # Generate visualization button for the latest chart
                                if st.button("📈 Generate Visualization", type="primary"):
                                    st.session_state.displayed_charts.add(response_id)
                                    st.session_state.available_charts[response_id] = chart
                                    st.rerun()
                            else:
                                # This chart is already displayed
                                st.button("📈 Chart Already Generated", disabled=True, help="This chart is already displayed")
                        else:
                            # No chart data available in latest response
                            st.button("📈 No Chart Data", disabled=True, help="Latest response has no chart data")
                    
                    with col2:
                        # Only show clear button if there are displayed charts
                        if st.session_state.displayed_charts:
                            if st.button("🗑️ Clear Visualizations"):
                                st.session_state.displayed_charts.clear()
                                st.session_state.available_charts.clear()
                                st.rerun()
            
            # Display charts that user has requested (only when not processing)
            if not latest_processing and st.session_state.displayed_charts:
                st.markdown("---")
                st.subheader("📈 Visualizations")
                
                for response_id in st.session_state.displayed_charts:
                    if response_id in st.session_state.available_charts:
                        chart = st.session_state.available_charts[response_id]
                        try:
                            st.plotly_chart(chart, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error displaying chart: {e}")
                            st.write("Chart data is available but could not be displayed.")
        
        with col3:
            # Empty column for balance
            pass
        
        # Chat interface at the bottom - centered
        # Add some spacing
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Center the chat input interface
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Show restore message if applicable
            if st.session_state.get('show_restore_message', False):
                ttl_display = self.chat_interface.storage_manager.get_ttl_display()
                st.info(f"📚 Restored previous chat history (expires in {ttl_display})")
                st.session_state.show_restore_message = False
            
            self.chat_interface.render_chat_interface()


def main():
    """Main function"""
    app = FraudChatbotApp()
    app.run()


if __name__ == "__main__":
    main()