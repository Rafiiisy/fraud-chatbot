"""
Header Component
Handles the main header and page configuration
"""
import streamlit as st


class Header:
    def __init__(self):
        pass
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">🤖 Fraud Detection Chatbot</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <p style="font-size: 1rem; color: #cccccc;">
                Ask questions about fraud data and get intelligent analysis with visualizations
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def setup_page_config(self):
        """Setup page configuration"""
        st.set_page_config(
            page_title="Fraud Detection Chatbot",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def render_custom_css(self):
        """Render custom CSS styles"""
        st.markdown("""
        <style>
            /* Global dark theme */
            .stApp {
                background-color: #000000;
            }
            
            .main .block-container {
                background-color: #000000;
                padding-top: 1rem;
            }
            
            .stMarkdown {
                background-color: #000000;
            }
            
            .main-header {
                font-size: 2rem;
                font-weight: bold;
                color: #ffffff;
                text-align: center;
                margin-bottom: 0.5rem;
                position: sticky;
                top: 0;
                background-color: #000000;
                z-index: 1000;
                padding: 0.5rem 0;
            }
            
            .question-card {
                background-color: #1a1a1a;
                color: #ffffff;
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 0.5rem 0;
                border-left: 4px solid #4a9eff;
            }
            
            .response-card {
                background-color: #1a1a1a;
                color: #ffffff;
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 1rem 0;
                border: 1px solid #333333;
                box-shadow: 0 2px 4px rgba(255,255,255,0.1);
            }
            
            .metric-card {
                background-color: #1a1a1a;
                color: #ffffff;
                padding: 1rem;
                border-radius: 0.5rem;
                text-align: center;
                border: 1px solid #333333;
            }
            
            .success-message {
                background-color: #1a4d1a;
                color: #90ee90;
                padding: 0.75rem;
                border-radius: 0.25rem;
                border: 1px solid #2d5a2d;
            }
            
            .info-message {
                background-color: #1a2d4d;
                color: #87ceeb;
                padding: 0.75rem;
                border-radius: 0.25rem;
                border: 1px solid #2d4a6b;
            }
            
            .error-message {
                background-color: #4d1a1a;
                color: #ff6b6b;
                padding: 0.75rem;
                border-radius: 0.25rem;
                border: 1px solid #6b2d2d;
            }
            
            .centered-content {
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 200px;
                background-color: #000000;
            }
            
            .chat-container {
                height: 60vh;
                overflow-y: auto;
                padding: 1rem;
                border: 1px solid #333333;
                border-radius: 0.5rem;
                background-color: transparent;
                display: flex !important;
                flex-direction: column !important;
                align-items: center !important;
                justify-content: center !important;
            }
            
            .chat-container > div {
                width: 100% !important;
                display: flex !important;
                flex-direction: column !important;
                align-items: center !important;
            }
            
            .chat-message {
                margin-bottom: 1rem;
                padding: 0.75rem;
                border-radius: 0.5rem;
                width: 90%;
                max-width: 800px;
            }
            
            .user-message {
                background-color: #2a4a6b;
                color: #ffffff;
                border-left: 4px solid #4a9eff;
                margin-left: auto;
                margin-right: 0;
                position: relative;
            }
            
            .user-message .user-label {
                position: absolute;
                top: 0.5rem;
                right: 0.5rem;
                font-size: 0.8rem;
                opacity: 0.8;
            }
            
            .user-message .user-content {
                margin-top: 1.5rem;
            }
            
            .assistant-message {
                background-color: #4a2a6b;
                color: #ffffff;
                border-left: 4px solid #9c27b0;
                margin-left: 0;
                margin-right: auto;
                position: relative;
            }
            
            .assistant-message.loading {
                background-color: #3a1a5b;
                border-left: 4px solid #7b1fa2;
                animation: pulse-loading 2s infinite;
            }
            
            .assistant-message.error {
                background-color: #5d2a2a;
                border-left: 4px solid #f44336;
            }
            
            .assistant-label {
                position: absolute;
                top: 0.5rem;
                left: 0.5rem;
                font-size: 0.8rem;
                opacity: 0.8;
            }
            
            .assistant-content {
                margin-top: 1.5rem;
            }
            
            .loading-animation {
                display: flex;
                align-items: center;
                font-style: italic;
                color: #cccccc;
            }
            
            .loading-dots {
                animation: loading-dots 1.5s infinite;
                margin-left: 0.2rem;
            }
            
            .error-message {
                color: #ff6b6b;
                font-weight: bold;
                margin-bottom: 0.5rem;
            }
            
            .error-details {
                color: #ffa8a8;
                font-size: 0.9rem;
                font-style: italic;
            }
            
            /* Animations */
            @keyframes loading-dots {
                0%, 20% { content: '.'; }
                40% { content: '..'; }
                60%, 100% { content: '...'; }
            }
            
            @keyframes pulse-loading {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.7; }
            }
            
            .chat-message {
                animation: slide-in-up 0.3s ease-out;
            }
            
            @keyframes slide-in-up {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            /* Streamlit specific dark theme overrides */
            .stTextInput > div > div > input {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 1px solid #333333;
            }
            
            .stTextArea > div > div > textarea {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 1px solid #333333;
            }
            
            .stButton > button {
                background-color: #4a9eff;
                color: #ffffff;
                border: none;
            }
            
            .stButton > button:hover {
                background-color: #357abd;
            }
            
            .stSelectbox > div > div > select {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 1px solid #333333;
            }
            
            .stSidebar {
                background-color: #1a1a1a;
            }
            
            .stSidebar .stMarkdown {
                color: #ffffff;
            }
            
            /* Sidebar text styling */
            .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6 {
                color: #ffffff !important;
            }
            
            .stSidebar p, .stSidebar div, .stSidebar span {
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown p {
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown strong {
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown em {
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown code {
                color: #ffffff !important;
                background-color: transparent !important;
            }
            
            .stSidebar .stMarkdown pre {
                color: #ffffff !important;
                background-color: transparent !important;
            }
            
            .stSidebar .stMarkdown ul, .stSidebar .stMarkdown ol {
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown li {
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown blockquote {
                color: #ffffff !important;
                background-color: transparent !important;
            }
            
            .stSidebar .stMarkdown a {
                color: #4a9eff !important;
            }
            
            .stSidebar .stMarkdown a:hover {
                color: #357abd !important;
            }
            
            /* Specific sidebar text elements */
            .stSidebar .stMarkdown h1, .stSidebar .stMarkdown h2, .stSidebar .stMarkdown h3 {
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown p {
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown strong {
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown em {
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown ul {
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown li {
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown div {
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown span {
                color: #ffffff !important;
            }
            
            /* Override Streamlit's default text colors in sidebar */
            .stSidebar .stMarkdown {
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown * {
                color: #ffffff !important;
            }
            
            /* Force white text for all sidebar content */
            .stSidebar .stMarkdown h1, .stSidebar .stMarkdown h2, .stSidebar .stMarkdown h3, 
            .stSidebar .stMarkdown h4, .stSidebar .stMarkdown h5, .stSidebar .stMarkdown h6 {
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown p, .stSidebar .stMarkdown div, .stSidebar .stMarkdown span {
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown strong, .stSidebar .stMarkdown b {
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown em, .stSidebar .stMarkdown i {
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown ul, .stSidebar .stMarkdown ol {
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown li {
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown blockquote {
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown code {
                color: #ffffff !important;
                background-color: transparent !important;
            }
            
            .stSidebar .stMarkdown pre {
                color: #ffffff !important;
                background-color: transparent !important;
            }
            
            .stSidebar .stMarkdown a {
                color: #4a9eff !important;
            }
            
            .stSidebar .stMarkdown a:hover {
                color: #357abd !important;
            }
            
            /* Override any remaining black text */
            .stSidebar * {
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown * {
                color: #ffffff !important;
            }
            
            /* Remove black backgrounds from sidebar text */
            .stSidebar .stMarkdown {
                background-color: transparent !important;
            }
            
            .stSidebar .stMarkdown p {
                background-color: transparent !important;
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown div {
                background-color: transparent !important;
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown span {
                background-color: transparent !important;
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown h1, .stSidebar .stMarkdown h2, .stSidebar .stMarkdown h3, 
            .stSidebar .stMarkdown h4, .stSidebar .stMarkdown h5, .stSidebar .stMarkdown h6 {
                background-color: transparent !important;
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown strong, .stSidebar .stMarkdown b {
                background-color: transparent !important;
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown em, .stSidebar .stMarkdown i {
                background-color: transparent !important;
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown ul, .stSidebar .stMarkdown ol {
                background-color: transparent !important;
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown li {
                background-color: transparent !important;
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown blockquote {
                background-color: transparent !important;
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown code {
                background-color: transparent !important;
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown pre {
                background-color: transparent !important;
                color: #ffffff !important;
            }
            
            /* Force transparent backgrounds for all sidebar elements */
            .stSidebar * {
                background-color: transparent !important;
                color: #ffffff !important;
            }
            
            .stSidebar .stMarkdown * {
                background-color: transparent !important;
                color: #ffffff !important;
            }
            
            /* Force center alignment for empty chat state */
            .chat-container .stMarkdown {
                text-align: center !important;
                width: 100% !important;
            }
            
            .chat-container .stMarkdown div {
                text-align: center !important;
                width: 100% !important;
            }
            
            .empty-chat-state {
                text-align: center !important;
                color: #cccccc !important;
                padding: 2rem !important;
                width: 100% !important;
                display: flex !important;
                justify-content: center !important;
                align-items: center !important;
                min-height: 200px !important;
                margin: 0 auto !important;
            }
            
            .empty-chat-state p {
                margin: 0 auto !important;
                text-align: center !important;
                display: block !important;
                width: fit-content !important;
            }
            
            /* Override all Streamlit markdown containers in chat */
            .chat-container .stMarkdown > div {
                display: flex !important;
                justify-content: center !important;
                align-items: center !important;
                width: 100% !important;
                text-align: center !important;
            }
            
            /* Force center for any content in chat container */
            .chat-container * {
                text-align: center !important;
            }
            
            /* Prevent hyperlink conversion - make all links look like normal text */
            .chat-container a,
            .assistant-message a,
            .user-message a,
            .assistant-content a,
            .user-content a {
                color: #ffffff !important;
                text-decoration: none !important;
                pointer-events: none !important;
                background: none !important;
                border: none !important;
            }
            
            .chat-container a:hover,
            .assistant-message a:hover,
            .user-message a:hover,
            .assistant-content a:hover,
            .user-content a:hover {
                color: #ffffff !important;
                text-decoration: none !important;
                background: none !important;
                border: none !important;
            }
            
            /* Override Streamlit's default link styling */
            .stMarkdown a,
            .stMarkdown p a,
            .stMarkdown div a,
            .stMarkdown span a {
                color: #ffffff !important;
                text-decoration: none !important;
                background: none !important;
                border: none !important;
            }
            
            .stMarkdown a:hover,
            .stMarkdown p a:hover,
            .stMarkdown div a:hover,
            .stMarkdown span a:hover {
                color: #ffffff !important;
                text-decoration: none !important;
                background: none !important;
                border: none !important;
            }
            
            /* Force all text to be white and prevent any link styling */
            .chat-container *,
            .assistant-message *,
            .user-message *,
            .assistant-content *,
            .user-content * {
                color: #ffffff !important;
            }
            
            /* Specifically target any remaining blue text */
            .chat-container a:not([style*="color"]),
            .assistant-message a:not([style*="color"]),
            .user-message a:not([style*="color"]) {
                color: #ffffff !important;
            }
            
            /* Ensure spans don't have special styling */
            .chat-container span,
            .assistant-message span,
            .user-message span {
                color: inherit !important;
                background: none !important;
                font-weight: inherit !important;
                font-style: inherit !important;
            }
        </style>
        """, unsafe_allow_html=True)
